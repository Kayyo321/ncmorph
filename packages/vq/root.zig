//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Inference-only VQ tokenizer loader and forward pass.

const std = @import("std");
const kernels = @import("kernels");
const safetensors = @import("safetensors");

const Allocator = std.mem.Allocator;
extern "c" fn erff(value: f32) f32;

pub const config_t = struct {
    vocab_size: usize,
    context_length: usize,
    width: usize,
    codebook_size: usize,
    levels: []const usize,
    attention_heads: usize,
    learned_level0: bool = false,
    sampler_stages: usize = 0,
    sampler_intermediate_norm: bool = false,
    sampler_final_norm: bool = false,
};

pub const tensor_view_t = struct {
    name: []const u8,
    dtype: safetensors.Dtype,
    shape: []const u64,
    data: []const u8,
};

pub const quantized_t = struct {
    code_ids: [][]u32,
    reconstruction: []f32,
    allocator: Allocator,

    pub fn deinit(self: *quantized_t) void {
        for (self.code_ids) |ids| self.allocator.free(ids);
        self.allocator.free(self.code_ids);
        self.allocator.free(self.reconstruction);
    }
};

pub const tokenizer_t = struct {
    allocator: Allocator,
    config: config_t,
    tensors: std.StringHashMap(tensor_view_t),

    pub fn load(allocator: Allocator, config: config_t, tensors: []const tensor_view_t) !tokenizer_t {
        try validate_config(config);
        var map = std.StringHashMap(tensor_view_t).init(allocator);
        errdefer deinit_tensor_map(allocator, &map);
        for (tensors) |tensor| {
            if (map.contains(tensor.name)) return error.DuplicateTensor;
            const owned_name = try allocator.dupe(u8, tensor.name);
            errdefer allocator.free(owned_name);
            const owned_shape = try allocator.dupe(u64, tensor.shape);
            errdefer allocator.free(owned_shape);
            try map.put(owned_name, .{ .name = owned_name, .dtype = tensor.dtype, .shape = owned_shape, .data = tensor.data });
        }
        try validate_manifest(allocator, config, &map);
        return .{ .allocator = allocator, .config = config, .tensors = map };
    }

    /// Loads a complete tokenizer from a single safetensors file already read into memory.
    pub fn load_safetensors(allocator: Allocator, config: config_t, bytes: []const u8) !tokenizer_t {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const header = try safetensors.parse_header(arena.allocator(), bytes, bytes.len);
        const views = try arena.allocator().alloc(tensor_view_t, header.tensors.len);
        const data_start = safetensors.PrefixLen + header.header_len;
        for (header.tensors, 0..) |tensor, i| {
            const begin = std.math.cast(usize, tensor.data_begin) orelse return error.TensorShapeMismatch;
            const end = std.math.cast(usize, tensor.data_end) orelse return error.TensorShapeMismatch;
            const start_offset = std.math.add(usize, data_start, begin) catch return error.TensorShapeMismatch;
            const end_offset = std.math.add(usize, data_start, end) catch return error.TensorShapeMismatch;
            if (end_offset > bytes.len or start_offset > end_offset) return error.TensorShapeMismatch;
            views[i] = .{ .name = tensor.name, .dtype = tensor.dtype, .shape = tensor.shape, .data = bytes[start_offset..end_offset] };
        }
        return load(allocator, config, views);
    }

    pub fn deinit(self: *tokenizer_t) void {
        deinit_tensor_map(self.allocator, &self.tensors);
    }

    /// Encodes ids, quantizes each residual level and returns code ids concatenated by level.
    pub fn encode(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize) ![][]u32 {
        const quantized = try self.quantize(allocator, ids, batch, length);
        allocator.free(quantized.reconstruction);
        return quantized.code_ids;
    }

    /// Returns level-major code ids and the full cumulative reconstruction.
    pub fn quantize(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize) !quantized_t {
        const count = try input_count(self.config, ids, batch, length);
        const width = self.config.width;
        const feature_count = std.math.mul(usize, count, width) catch return error.InvalidInputShape;
        const hidden = try allocator.alloc(f32, feature_count);
        defer allocator.free(hidden);
        try self.encode_hidden(allocator, ids, batch, length, hidden);
        var residual = try allocator.dupe(f32, hidden);
        defer allocator.free(residual);
        const reconstruction = try allocator.alloc(f32, hidden.len);
        errdefer allocator.free(reconstruction);
        @memset(reconstruction, 0);
        const code_ids = try allocator.alloc([]u32, self.config.levels.len);
        errdefer allocator.free(code_ids);
        var initialized: usize = 0;
        errdefer for (code_ids[0..initialized]) |level_ids| allocator.free(level_ids);
        for (self.config.levels, 0..) |level_size, level_idx| {
            const level_ids = try allocator.alloc(u32, batch * level_size);
            code_ids[level_idx] = level_ids;
            initialized += 1;
            const down = try self.down_for_quant(allocator, residual, batch, length, level_size, level_idx);
            defer allocator.free(down);
            const codebook_name = try fmt_name(allocator, "vq.embeddings.{d}.weight", .{level_idx});
            defer allocator.free(codebook_name);
            const codebook = try self.tensor_f32(allocator, codebook_name);
            defer allocator.free(codebook);
            const quantized_vectors = try allocator.alloc(f32, down.len);
            defer allocator.free(quantized_vectors);
            for (0..batch * level_size) |pos| {
                const code = nearest_code(down[pos * width ..][0..width], codebook, self.config.codebook_size, width);
                level_ids[pos] = code;
                @memcpy(quantized_vectors[pos * width ..][0..width], codebook[@as(usize, code) * width ..][0..width]);
            }
            const up = try self.up_to_full(allocator, quantized_vectors, batch, level_size, length, level_idx);
            defer allocator.free(up);
            const contribution = try self.apply_quant_resi(allocator, up, batch, length, level_idx);
            defer allocator.free(contribution);
            for (0..residual.len) |i| {
                reconstruction[i] += contribution[i];
                residual[i] -= contribution[i];
            }
        }
        return .{ .code_ids = code_ids, .reconstruction = reconstruction, .allocator = allocator };
    }

    /// Returns encoder embeddings after pre-quantization GroupNorm.
    pub fn encode_features(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize) ![]f32 {
        const count = try input_count(self.config, ids, batch, length);
        const feature_count = std.math.mul(usize, count, self.config.width) catch return error.InvalidInputShape;
        const output = try allocator.alloc(f32, feature_count);
        errdefer allocator.free(output);
        try self.encode_hidden(allocator, ids, batch, length, output);
        return output;
    }

    /// Runs the complete evaluation forward pass and returns logits [batch, length, vocab].
    pub fn forward(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize) ![]f32 {
        _ = try input_count(self.config, ids, batch, length);
        const norm = try self.decoder_hidden(allocator, ids, batch, length);
        defer allocator.free(norm);
        const vocabulary_ids = try allocator.alloc(u32, self.config.vocab_size);
        defer allocator.free(vocabulary_ids);
        for (vocabulary_ids, 0..) |*vocab_id, index| vocab_id.* = @intCast(index);
        return self.selected_logits_from_hidden(allocator, norm, vocabulary_ids);
    }

    /// Returns logits for selected vocabulary rows [batch, length, selected_vocab].
    /// This supports reference checks on large vocabularies without materializing the full head.
    pub fn forward_selected(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize, vocabulary_ids: []const u32) ![]f32 {
        _ = try input_count(self.config, ids, batch, length);
        if (vocabulary_ids.len == 0) return error.InvalidInputShape;
        for (vocabulary_ids) |vocab_id| if (vocab_id >= self.config.vocab_size) return error.TokenIdOutOfRange;
        const norm = try self.decoder_hidden(allocator, ids, batch, length);
        defer allocator.free(norm);
        return self.selected_logits_from_hidden(allocator, norm, vocabulary_ids);
    }

    fn selected_logits_from_hidden(self: *const tokenizer_t, allocator: Allocator, norm: []const f32, vocabulary_ids: []const u32) ![]f32 {
        if (vocabulary_ids.len == 0 or norm.len % self.config.width != 0) return error.InvalidInputShape;
        for (vocabulary_ids) |vocab_id| if (vocab_id >= self.config.vocab_size) return error.TokenIdOutOfRange;
        const count = norm.len / self.config.width;
        const width = self.config.width;
        const output_count = std.math.mul(usize, count, vocabulary_ids.len) catch return error.InvalidInputShape;
        const output = try allocator.alloc(f32, output_count);
        errdefer allocator.free(output);
        const output_weight = self.tensors.get("decoder.output_layer.weight") orelse return error.MissingTensor;
        for (vocabulary_ids, 0..) |vocab_id, selected_idx| {
            const row_offset = std.math.mul(usize, vocab_id, width * 4) catch return error.TensorShapeMismatch;
            for (0..count) |position| {
                var sum_vector: @Vector(8, f32) = @splat(0);
                var channel: usize = 0;
                while (channel + 8 <= width) : (channel += 8) {
                    var input_vector: @Vector(8, f32) = undefined;
                    var weight_vector: @Vector(8, f32) = undefined;
                    inline for (0..8) |lane| {
                        input_vector[lane] = norm[position * width + channel + lane];
                        const offset = row_offset + (channel + lane) * 4;
                        weight_vector[lane] = @bitCast(std.mem.readInt(u32, output_weight.data[offset..][0..4], .little));
                    }
                    sum_vector += input_vector * weight_vector;
                }
                var sum = @reduce(.Add, sum_vector);
                while (channel < width) : (channel += 1) {
                    const offset = row_offset + channel * 4;
                    const weight = @as(f32, @bitCast(std.mem.readInt(u32, output_weight.data[offset..][0..4], .little)));
                    sum += norm[position * width + channel] * weight;
                }
                output[position * vocabulary_ids.len + selected_idx] = sum;
            }
        }
        return output;
    }

    fn decoder_hidden(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize) ![]f32 {
        var quantized = try self.quantize(allocator, ids, batch, length);
        defer quantized.deinit();
        try self.decode(allocator, quantized.reconstruction, batch, length);
        const norm = try allocator.alloc(f32, quantized.reconstruction.len);
        errdefer allocator.free(norm);
        const norm_weight = try self.tensor_f32(allocator, "decoder.ln_f.weight");
        defer allocator.free(norm_weight);
        layer_norm(quantized.reconstruction, norm_weight, null, norm, batch * length, self.config.width, 1e-6);
        return norm;
    }

    fn encode_hidden(self: *const tokenizer_t, allocator: Allocator, ids: []const u32, batch: usize, length: usize, output: []f32) !void {
        const width = self.config.width;
        const token_table = try self.tensor_f32(allocator, "encoder.wte.weight");
        defer allocator.free(token_table);
        const position_table = try self.tensor_f32(allocator, "encoder.wpe.weight");
        defer allocator.free(position_table);
        for (0..batch) |b| for (0..length) |t| {
            const id = ids[b * length + t];
            if (id >= self.config.vocab_size) return error.TokenIdOutOfRange;
            for (0..width) |c| output[(b * length + t) * width + c] = token_table[id * width + c] + position_table[t * width + c];
        };
        group_norm(output, batch, length, width, 8);
    }

    fn decode(self: *const tokenizer_t, allocator: Allocator, hidden: []f32, batch: usize, length: usize) !void {
        const width = self.config.width;
        const heads = self.config.attention_heads;
        const head_width = width / heads;
        const prefix = "decoder.middle_block.";
        const ln1w = try self.tensor_f32(allocator, "decoder.middle_block.ln_1.weight");
        defer allocator.free(ln1w);
        const ln1b = try self.tensor_f32(allocator, "decoder.middle_block.ln_1.bias");
        defer allocator.free(ln1b);
        const ln2w = try self.tensor_f32(allocator, "decoder.middle_block.ln_2.weight");
        defer allocator.free(ln2w);
        const ln2b = try self.tensor_f32(allocator, "decoder.middle_block.ln_2.bias");
        defer allocator.free(ln2b);
        const qkvw = try self.tensor_f32(allocator, "decoder.middle_block.attn.c_attn.weight");
        defer allocator.free(qkvw);
        const qkvb = try self.tensor_f32(allocator, "decoder.middle_block.attn.c_attn.bias");
        defer allocator.free(qkvb);
        const projw = try self.tensor_f32(allocator, "decoder.middle_block.attn.c_proj.weight");
        defer allocator.free(projw);
        const projb = try self.tensor_f32(allocator, "decoder.middle_block.attn.c_proj.bias");
        defer allocator.free(projb);
        const qw = try self.tensor_f32(allocator, "decoder.middle_block.attn.q_norm.weight");
        defer allocator.free(qw);
        const kw = try self.tensor_f32(allocator, "decoder.middle_block.attn.k_norm.weight");
        defer allocator.free(kw);
        const fcw = try self.tensor_f32(allocator, "decoder.middle_block.mlp.c_fc.weight");
        defer allocator.free(fcw);
        const fcb = try self.tensor_f32(allocator, "decoder.middle_block.mlp.c_fc.bias");
        defer allocator.free(fcb);
        const mlpw = try self.tensor_f32(allocator, "decoder.middle_block.mlp.c_proj.weight");
        defer allocator.free(mlpw);
        const mlpb = try self.tensor_f32(allocator, "decoder.middle_block.mlp.c_proj.bias");
        defer allocator.free(mlpb);
        _ = prefix;
        const normalized = try allocator.alloc(f32, hidden.len);
        defer allocator.free(normalized);
        layer_norm(hidden, ln1w, ln1b, normalized, batch * length, width, 1e-6);
        const qkv_count = std.math.mul(usize, hidden.len, 3) catch return error.InvalidInputShape;
        const qkv = try allocator.alloc(f32, qkv_count);
        defer allocator.free(qkv);
        matmul_bias(normalized, qkvw, qkvb, qkv, batch * length, width, width * 3);
        const attended = try allocator.alloc(f32, hidden.len);
        defer allocator.free(attended);
        @memset(attended, 0);
        for (0..batch) |b| for (0..heads) |h| {
            const q_norm = try allocator.alloc(f32, length * head_width);
            defer allocator.free(q_norm);
            const k_norm = try allocator.alloc(f32, length * head_width);
            defer allocator.free(k_norm);
            const vv = try allocator.alloc(f32, length * head_width);
            defer allocator.free(vv);
            for (0..length) |t| for (0..head_width) |c| {
                const base = (b * length + t) * width * 3;
                const idx = h * head_width + c;
                q_norm[t * head_width + c] = qkv[base + idx];
                k_norm[t * head_width + c] = qkv[base + width + idx];
                vv[t * head_width + c] = qkv[base + width * 2 + idx];
            };
            rms_norm_heads(q_norm, qw, length, head_width, 1e-6);
            rms_norm_heads(k_norm, kw, length, head_width, 1e-6);
            var scores = try allocator.alloc(f32, length * length);
            defer allocator.free(scores);
            for (0..length) |i| for (0..length) |j| {
                var sum: f32 = 0;
                for (0..head_width) |c| sum += q_norm[i * head_width + c] * k_norm[j * head_width + c];
                scores[i * length + j] = sum / @sqrt(@as(f32, @floatFromInt(head_width)));
            };
            softmax_rows(scores, length, length);
            for (0..length) |i| for (0..head_width) |c| {
                var sum: f32 = 0;
                for (0..length) |j| sum += scores[i * length + j] * vv[j * head_width + c];
                attended[(b * length + i) * width + h * head_width + c] = sum;
            };
        };
        const projected = try allocator.alloc(f32, hidden.len);
        defer allocator.free(projected);
        matmul_bias(attended, projw, projb, projected, batch * length, width, width);
        for (hidden, 0..) |*value, i| value.* += projected[i];
        layer_norm(hidden, ln2w, ln2b, normalized, batch * length, width, 1e-6);
        const mlp_count = std.math.mul(usize, hidden.len, 4) catch return error.InvalidInputShape;
        const mlp_hidden = try allocator.alloc(f32, mlp_count);
        defer allocator.free(mlp_hidden);
        matmul_bias(normalized, fcw, fcb, mlp_hidden, batch * length, width, width * 4);
        for (mlp_hidden) |*value| value.* = gelu_exact(value.*);
        const mlp_out = try allocator.alloc(f32, hidden.len);
        defer allocator.free(mlp_out);
        matmul_bias(mlp_hidden, mlpw, mlpb, mlp_out, batch * length, width * 4, width);
        for (hidden, 0..) |*value, i| value.* += mlp_out[i];
    }

    fn apply_quant_resi(self: *const tokenizer_t, allocator: Allocator, input: []const f32, batch: usize, length: usize, level: usize) ![]f32 {
        const width = self.config.width;
        const weight_name = try fmt_name(allocator, "vq.quant_resi.{d}.weight", .{level});
        defer allocator.free(weight_name);
        const weight = try self.tensor_f32(allocator, weight_name);
        defer allocator.free(weight);
        const bias_name = try fmt_name(allocator, "vq.quant_resi.{d}.bias", .{level});
        defer allocator.free(bias_name);
        const bias = try self.tensor_f32(allocator, bias_name);
        defer allocator.free(bias);
        const output = try allocator.alloc(f32, input.len);
        for (0..batch) |b| for (0..length) |t| for (0..width) |o| {
            var conv = bias[o];
            for (0..3) |k| {
                const pos: isize = @as(isize, @intCast(t)) + @as(isize, @intCast(k)) - 1;
                if (pos < 0 or pos >= length) continue;
                for (0..width) |c| conv += input[(b * length + @as(usize, @intCast(pos))) * width + c] * weight[(o * width + c) * 3 + k];
            }
            const i = (b * length + t) * width + o;
            output[i] = 0.5 * input[i] + 0.5 * conv;
        };
        return output;
    }

    fn down_for_quant(self: *const tokenizer_t, allocator: Allocator, input: []const f32, batch: usize, length: usize, level_length: usize, level: usize) ![]f32 {
        if (level == 0 and self.config.learned_level0) return self.apply_sampler(allocator, input, batch, length, true);
        if (level_length == length) return allocator.dupe(f32, input);
        return downsample(allocator, input, batch, length, self.config.width, level_length);
    }

    fn up_to_full(self: *const tokenizer_t, allocator: Allocator, input: []const f32, batch: usize, level_length: usize, length: usize, level: usize) ![]f32 {
        if (level == 0 and self.config.learned_level0) return self.apply_sampler(allocator, input, batch, level_length, false);
        if (level_length == length) return allocator.dupe(f32, input);
        return upsample(allocator, input, batch, level_length, length, self.config.width);
    }

    fn apply_sampler(self: *const tokenizer_t, allocator: Allocator, source: []const f32, batch: usize, input_length: usize, is_down: bool) ![]f32 {
        var value = try allocator.dupe(f32, source);
        errdefer allocator.free(value);
        var current_length = input_length;
        for (0..self.config.sampler_stages) |stage| {
            const module_index = sampler_module_index(self.config, stage);
            const suffix = if (is_down) "learned_down_l1" else "learned_up_l1";
            const weight_name = try std.fmt.allocPrint(allocator, "vq.{s}.{d}.weight", .{ suffix, module_index });
            defer allocator.free(weight_name);
            const bias_name = try std.fmt.allocPrint(allocator, "vq.{s}.{d}.bias", .{ suffix, module_index });
            defer allocator.free(bias_name);
            const weight = try self.tensor_f32(allocator, weight_name);
            defer allocator.free(weight);
            const bias = try self.tensor_f32(allocator, bias_name);
            defer allocator.free(bias);
            const next_length = if (is_down) current_length / 4 else std.math.mul(usize, current_length, 4) catch return error.InvalidInputShape;
            if (is_down and current_length % 4 != 0) return error.InvalidInputShape;
            const next = try allocator.alloc(f32, batch * next_length * self.config.width);
            if (is_down) {
                conv1d_stride4(value, weight, bias, next, batch, current_length, self.config.width);
            } else {
                conv_transpose1d_stride4(value, weight, bias, next, batch, current_length, self.config.width);
            }
            allocator.free(value);
            value = next;
            current_length = next_length;
            const should_norm = if (stage + 1 < self.config.sampler_stages)
                self.config.sampler_intermediate_norm
            else
                is_down and self.config.sampler_final_norm;
            if (should_norm) group_norm(value, batch, current_length, self.config.width, 8);
        }
        return value;
    }

    fn tensor_f32(self: *const tokenizer_t, allocator: Allocator, name: []const u8) ![]f32 {
        const tensor = self.tensors.get(name) orelse return error.MissingTensor;
        if (tensor.dtype != .F32 or tensor.data.len % 4 != 0) return error.UnsupportedTensorDtype;
        const values = try allocator.alloc(f32, tensor.data.len / 4);
        for (values, 0..) |*value, i| value.* = @bitCast(std.mem.readInt(u32, tensor.data[i * 4 ..][0..4], .little));
        return values;
    }
};

fn validate_config(config: config_t) !void {
    if (config.vocab_size == 0 or config.context_length == 0 or config.width == 0 or config.codebook_size == 0 or config.levels.len == 0) return error.InvalidConfig;
    if (config.attention_heads == 0 or config.width % config.attention_heads != 0 or config.width % 8 != 0) return error.InvalidConfig;
    if (config.levels[0] != 1 or config.levels[config.levels.len - 1] != config.context_length) return error.InvalidConfig;
    for (config.levels, 0..) |level_size, i| {
        if (level_size == 0 or level_size > config.context_length or (i > 0 and level_size <= config.levels[i - 1])) return error.InvalidConfig;
    }
    if (config.learned_level0 and config.sampler_stages == 0) return error.InvalidConfig;
    if (!config.learned_level0 and (config.sampler_intermediate_norm or config.sampler_final_norm or config.sampler_stages != 0)) return error.InvalidConfig;
    if (config.learned_level0) {
        var stride: usize = 1;
        for (0..config.sampler_stages) |_| stride = std.math.mul(usize, stride, 4) catch return error.InvalidConfig;
        if (stride != config.context_length) return error.InvalidConfig;
    }
    _ = std.math.mul(usize, config.vocab_size, config.width) catch return error.InvalidConfig;
    _ = std.math.mul(usize, config.codebook_size, config.width) catch return error.InvalidConfig;
    _ = std.math.mul(usize, config.context_length, config.width) catch return error.InvalidConfig;
    _ = std.math.mul(usize, config.context_length, config.context_length) catch return error.InvalidConfig;
    _ = std.math.mul(usize, config.width, 4) catch return error.InvalidConfig;
}

fn input_count(config: config_t, ids: []const u32, batch: usize, length: usize) !usize {
    if (batch == 0 or length != config.context_length) return error.InvalidInputShape;
    const count = std.math.mul(usize, batch, length) catch return error.InvalidInputShape;
    if (ids.len != count) return error.InvalidInputShape;
    return count;
}

fn validate_manifest(allocator: Allocator, config: config_t, tensors: *const std.StringHashMap(tensor_view_t)) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const temporary_allocator = arena.allocator();
    var expected = try expected_manifest(temporary_allocator, config);
    defer expected.deinit();
    if (tensors.count() != expected.count()) return error.TensorManifestMismatch;
    var it = expected.iterator();
    while (it.next()) |entry| {
        const actual = tensors.get(entry.key_ptr.*) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, actual.shape, entry.value_ptr.*)) return error.TensorShapeMismatch;
        const is_hit = std.mem.startsWith(u8, actual.name, "vq.codebook_hit_");
        const expected_dtype: safetensors.Dtype = if (is_hit) .I64 else .F32;
        if (actual.dtype != expected_dtype) return error.UnsupportedTensorDtype;
        const elems = try shape_product(actual.shape);
        const bytes_per_element: usize = if (is_hit) 8 else 4;
        const expected_bytes = std.math.mul(usize, elems, bytes_per_element) catch return error.TensorShapeMismatch;
        if (actual.data.len != expected_bytes) return error.TensorShapeMismatch;
    }
    var actual_it = tensors.iterator();
    while (actual_it.next()) |entry| if (!expected.contains(entry.key_ptr.*)) return error.UnexpectedTensor;
}

fn expected_manifest(allocator: Allocator, config: config_t) !std.StringHashMap([]const u64) {
    var expected = std.StringHashMap([]const u64).init(allocator);
    errdefer expected.deinit();
    try add_expected(allocator, &expected, "encoder.wte.weight", &.{ config.vocab_size, config.width });
    try add_expected(allocator, &expected, "encoder.wpe.weight", &.{ config.context_length, config.width });
    for (config.levels, 0..) |_, i| {
        try add_expected(allocator, &expected, try fmt_name(allocator, "vq.embeddings.{d}.weight", .{i}), &.{ config.codebook_size, config.width });
        try add_expected(allocator, &expected, try fmt_name(allocator, "vq.quant_resi.{d}.weight", .{i}), &.{ config.width, config.width, 3 });
        try add_expected(allocator, &expected, try fmt_name(allocator, "vq.quant_resi.{d}.bias", .{i}), &.{config.width});
        try add_expected(allocator, &expected, try fmt_name(allocator, "vq.ema_cluster_size_{d}", .{i}), &.{config.codebook_size});
        try add_expected(allocator, &expected, try fmt_name(allocator, "vq.ema_w_{d}", .{i}), &.{ config.codebook_size, config.width });
        try add_expected(allocator, &expected, try fmt_name(allocator, "vq.codebook_hit_{d}", .{i}), &.{config.codebook_size});
    }
    if (config.learned_level0) {
        for (0..config.sampler_stages) |stage| {
            const seq_idx = sampler_module_index(config, stage);
            try add_expected(allocator, &expected, try fmt_name(allocator, "vq.learned_down_l1.{d}.weight", .{seq_idx}), &.{ config.width, config.width, 4 });
            try add_expected(allocator, &expected, try fmt_name(allocator, "vq.learned_down_l1.{d}.bias", .{seq_idx}), &.{config.width});
            try add_expected(allocator, &expected, try fmt_name(allocator, "vq.learned_up_l1.{d}.weight", .{seq_idx}), &.{ config.width, config.width, 4 });
            try add_expected(allocator, &expected, try fmt_name(allocator, "vq.learned_up_l1.{d}.bias", .{seq_idx}), &.{config.width});
        }
    }
    const w = config.width;
    try add_expected(allocator, &expected, "decoder.ln_f.weight", &.{w});
    try add_expected(allocator, &expected, "decoder.middle_block.ln_1.weight", &.{w});
    try add_expected(allocator, &expected, "decoder.middle_block.ln_1.bias", &.{w});
    try add_expected(allocator, &expected, "decoder.middle_block.ln_2.weight", &.{w});
    try add_expected(allocator, &expected, "decoder.middle_block.ln_2.bias", &.{w});
    try add_expected(allocator, &expected, "decoder.middle_block.attn.c_attn.weight", &.{ 3 * w, w });
    try add_expected(allocator, &expected, "decoder.middle_block.attn.c_attn.bias", &.{3 * w});
    try add_expected(allocator, &expected, "decoder.middle_block.attn.c_proj.weight", &.{ w, w });
    try add_expected(allocator, &expected, "decoder.middle_block.attn.c_proj.bias", &.{w});
    try add_expected(allocator, &expected, "decoder.middle_block.attn.q_norm.weight", &.{w / config.attention_heads});
    try add_expected(allocator, &expected, "decoder.middle_block.attn.k_norm.weight", &.{w / config.attention_heads});
    try add_expected(allocator, &expected, "decoder.middle_block.mlp.c_fc.weight", &.{ 4 * w, w });
    try add_expected(allocator, &expected, "decoder.middle_block.mlp.c_fc.bias", &.{4 * w});
    try add_expected(allocator, &expected, "decoder.middle_block.mlp.c_proj.weight", &.{ w, 4 * w });
    try add_expected(allocator, &expected, "decoder.middle_block.mlp.c_proj.bias", &.{w});
    try add_expected(allocator, &expected, "decoder.output_layer.weight", &.{ config.vocab_size, w });
    return expected;
}

fn deinit_tensor_map(allocator: Allocator, map: *std.StringHashMap(tensor_view_t)) void {
    var iterator = map.iterator();
    while (iterator.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        allocator.free(entry.value_ptr.shape);
    }
    map.deinit();
}

fn sampler_module_index(config: config_t, stage: usize) usize {
    return stage * (if (config.sampler_intermediate_norm) @as(usize, 2) else 1);
}

fn add_expected(allocator: Allocator, map: *std.StringHashMap([]const u64), name: []const u8, shape: []const u64) !void {
    try map.put(name, try allocator.dupe(u64, shape));
}

fn shape_product(shape: []const u64) !usize {
    var product: usize = 1;
    for (shape) |dim| product = std.math.mul(usize, product, std.math.cast(usize, dim) orelse return error.TensorShapeMismatch) catch return error.TensorShapeMismatch;
    return product;
}

fn fmt_name(allocator: Allocator, comptime format: []const u8, args: anytype) ![]const u8 {
    return std.fmt.allocPrint(allocator, format, args);
}

fn nearest_code(vector: []const f32, codebook: []const f32, codes: usize, width: usize) u32 {
    var best: usize = 0;
    var best_distance: f32 = std.math.inf(f32);
    var vector_norm: f32 = 0;
    for (vector) |value| vector_norm += value * value;
    for (0..codes) |code| {
        var code_norm: f32 = 0;
        var dot: f32 = 0;
        for (0..width) |c| {
            const code_value = codebook[code * width + c];
            code_norm += code_value * code_value;
            dot += vector[c] * code_value;
        }
        const distance = vector_norm + code_norm - 2 * dot;
        if (distance < best_distance) {
            best_distance = distance;
            best = code;
        }
    }
    return @intCast(best);
}

fn conv1d_stride4(input: []const f32, weight: []const f32, bias: []const f32, output: []f32, batch: usize, input_length: usize, width: usize) void {
    const output_length = input_length / 4;
    for (0..batch) |b| for (0..output_length) |t| for (0..width) |out_channel| {
        var sum = bias[out_channel];
        for (0..width) |in_channel| for (0..4) |k| {
            const x = input[(b * input_length + t * 4 + k) * width + in_channel];
            sum += x * weight[(out_channel * width + in_channel) * 4 + k];
        };
        output[(b * output_length + t) * width + out_channel] = sum;
    };
}

fn conv_transpose1d_stride4(input: []const f32, weight: []const f32, bias: []const f32, output: []f32, batch: usize, input_length: usize, width: usize) void {
    const output_length = input_length * 4;
    for (0..batch) |b| for (0..output_length) |t| for (0..width) |out_channel| {
        const input_position = t / 4;
        const kernel_position = t % 4;
        var sum = bias[out_channel];
        for (0..width) |in_channel| {
            sum += input[(b * input_length + input_position) * width + in_channel] *
                weight[(in_channel * width + out_channel) * 4 + kernel_position];
        }
        output[(b * output_length + t) * width + out_channel] = sum;
    };
}

fn downsample(allocator: Allocator, input: []const f32, batch: usize, input_length: usize, width: usize, output_length: usize) ![]f32 {
    const output = try allocator.alloc(f32, batch * output_length * width);
    for (0..batch) |b| for (0..output_length) |j| for (0..width) |c| {
        const start = j * input_length / output_length;
        const end = ((j + 1) * input_length + output_length - 1) / output_length;
        var sum: f32 = 0;
        for (start..end) |i| sum += input[(b * input_length + i) * width + c];
        output[(b * output_length + j) * width + c] = sum / @as(f32, @floatFromInt(end - start));
    };
    return output;
}

fn upsample(allocator: Allocator, input: []const f32, batch: usize, input_length: usize, output_length: usize, width: usize) ![]f32 {
    const output = try allocator.alloc(f32, batch * output_length * width);
    for (0..batch) |b| for (0..output_length) |j| for (0..width) |c| {
        const source = (@as(f32, @floatFromInt(j)) + 0.5) * @as(f32, @floatFromInt(input_length)) / @as(f32, @floatFromInt(output_length)) - 0.5;
        const lower_f = @max(source, 0);
        const lower: usize = @intFromFloat(@floor(lower_f));
        const upper = @min(lower + 1, input_length - 1);
        const fraction = if (source <= 0) 0 else source - @as(f32, @floatFromInt(lower));
        const a = input[(b * input_length + lower) * width + c];
        const z = input[(b * input_length + upper) * width + c];
        output[(b * output_length + j) * width + c] = a + fraction * (z - a);
    };
    return output;
}

fn group_norm(input: []f32, batch: usize, length: usize, width: usize, groups: usize) void {
    const group_width = width / groups;
    for (0..batch) |b| for (0..groups) |g| {
        var mean: f64 = 0;
        const n = length * group_width;
        for (0..length) |t| {
            for (0..group_width) |c| mean += input[(b * length + t) * width + g * group_width + c];
        }
        mean /= @floatFromInt(n);
        var variance: f64 = 0;
        for (0..length) |t| for (0..group_width) |c| {
            const d = @as(f64, input[(b * length + t) * width + g * group_width + c]) - mean;
            variance += d * d;
        };
        variance /= @floatFromInt(n);
        const scale: f32 = @floatCast(1.0 / @sqrt(variance + 1e-5));
        for (0..length) |t| for (0..group_width) |c| {
            const i = (b * length + t) * width + g * group_width + c;
            input[i] = (input[i] - @as(f32, @floatCast(mean))) * scale;
        };
    };
}

fn layer_norm(input: []const f32, weight: []const f32, bias: ?[]const f32, output: []f32, rows: usize, width: usize, epsilon: f32) void {
    for (0..rows) |r| {
        var mean: f64 = 0;
        for (0..width) |c| mean += input[r * width + c];
        mean /= @floatFromInt(width);
        var variance: f64 = 0;
        for (0..width) |c| {
            const d = @as(f64, input[r * width + c]) - mean;
            variance += d * d;
        }
        variance /= @floatFromInt(width);
        const scale: f32 = @floatCast(1.0 / @sqrt(variance + epsilon));
        for (0..width) |c| output[r * width + c] = (input[r * width + c] - @as(f32, @floatCast(mean))) * scale * weight[c] + if (bias) |values| values[c] else 0;
    }
}

fn rms_norm_heads(input: []f32, weight: []const f32, rows: usize, width: usize, epsilon: f32) void {
    for (0..rows) |r| {
        var squares: f64 = 0;
        for (0..width) |c| squares += @as(f64, input[r * width + c]) * input[r * width + c];
        const scale: f32 = @floatCast(1.0 / @sqrt(squares / @as(f64, @floatFromInt(width)) + epsilon));
        for (0..width) |c| input[r * width + c] = input[r * width + c] * scale * weight[c];
    }
}

fn matmul(input: []const f32, weights: []const f32, output: []f32, rows: usize, inner: usize, cols: usize) void {
    kernels.matmul(input, weights, output, rows, inner, cols);
}

fn matmul_bias(input: []const f32, weights: []const f32, bias: []const f32, output: []f32, rows: usize, inner: usize, cols: usize) void {
    matmul(input, weights, output, rows, inner, cols);
    for (0..rows) |r| {
        for (0..cols) |c| output[r * cols + c] += bias[c];
    }
}

fn gelu_exact(value: f32) f32 {
    return 0.5 * value * (1.0 + erff(value * 0.7071067811865476));
}

const reference_config_t = struct {
    vocab_size: usize,
    length: usize,
    width: usize,
    levels: []usize,
    codebook_size: usize,
    heads: usize,
    cascade_norm: bool = false,
    cascade_norm_final: bool = false,
};

const reference_fixture_t = struct {
    reference_commit: []const u8 = "",
    config: reference_config_t,
    token_ids: [][]u32,
    level_ids: [][][]u32,
    encoded: []f32,
    reconstruction: []f32,
    logits: ?[]f32 = null,
    selected_vocab_ids: ?[]u32 = null,
    selected_logits: ?[]f32 = null,
    logits_file: ?[]const u8 = null,
    weights_file: []const u8 = "",
};

fn run_reference_fixture(allocator: Allocator, json_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const json_bytes = cwd.readFileAlloc(std.testing.io, json_path, allocator, .limited(16 * 1024 * 1024)) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer allocator.free(json_bytes);
    const parsed = try std.json.parseFromSlice(reference_fixture_t, allocator, json_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const fixture = parsed.value;
    if (!std.mem.eql(u8, fixture.reference_commit, "c70274273f489abc73832b2b98584a8483b6b203")) return error.ReferenceCommitMismatch;
    if (fixture.logits == null and fixture.logits_file == null) return error.MissingDecoderReferenceOutput;
    if (fixture.weights_file.len == 0) return error.MissingReferenceWeights;
    const weights_bytes = cwd.readFileAlloc(std.testing.io, fixture.weights_file, allocator, .limited(1_000_000_000)) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer allocator.free(weights_bytes);

    const config = config_t{
        .vocab_size = fixture.config.vocab_size,
        .context_length = fixture.config.length,
        .width = fixture.config.width,
        .codebook_size = fixture.config.codebook_size,
        .levels = fixture.config.levels,
        .attention_heads = fixture.config.heads,
        .learned_level0 = true,
        .sampler_stages = cascade_stages(fixture.config.length) orelse return error.InvalidConfig,
        .sampler_intermediate_norm = fixture.config.cascade_norm,
        .sampler_final_norm = fixture.config.cascade_norm_final,
    };
    var tokenizer = try tokenizer_t.load_safetensors(allocator, config, weights_bytes);
    defer tokenizer.deinit();
    const batch = fixture.token_ids.len;
    if (batch == 0) return error.InvalidInputShape;
    const length = fixture.config.length;
    const ids = try allocator.alloc(u32, batch * length);
    defer allocator.free(ids);
    for (fixture.token_ids, 0..) |row, batch_idx| {
        if (row.len != length) return error.InvalidInputShape;
        @memcpy(ids[batch_idx * length ..][0..length], row);
    }

    const encoded = try tokenizer.encode_features(allocator, ids, batch, length);
    defer allocator.free(encoded);
    try expect_close_slices(fixture.encoded, encoded, fixture.config.width);
    var quantized = try tokenizer.quantize(allocator, ids, batch, length);
    defer quantized.deinit();
    if (fixture.level_ids.len != quantized.code_ids.len) return error.InvalidInputShape;
    for (fixture.level_ids, 0..) |expected_level, level_idx| {
        if (expected_level.len != batch or expected_level[0].len != quantized.code_ids[level_idx].len / batch) return error.InvalidInputShape;
        for (expected_level, 0..) |expected_row, batch_idx| {
            const start = batch_idx * expected_row.len;
            try std.testing.expectEqualSlices(u32, expected_row, quantized.code_ids[level_idx][start .. start + expected_row.len]);
        }
    }
    try expect_close_slices(fixture.reconstruction, quantized.reconstruction, fixture.config.width);
    if (fixture.logits) |expected_logits| {
        const logits = try tokenizer.forward(allocator, ids, batch, length);
        defer allocator.free(logits);
        try expect_close_slices(expected_logits, logits, fixture.config.vocab_size);
    }
    if (fixture.selected_logits) |expected_logits| {
        const selected_ids = fixture.selected_vocab_ids orelse return error.MissingReferenceVocabularyIds;
        const logits = try tokenizer.forward_selected(allocator, ids, batch, length, selected_ids);
        defer allocator.free(logits);
        try expect_close_slices(expected_logits, logits, selected_ids.len);
    }
    if (fixture.logits_file) |logits_file| {
        try compare_full_logits_file(allocator, &tokenizer, ids, batch, length, fixture.config.vocab_size, logits_file);
    }
}

fn compare_full_logits_file(allocator: Allocator, tokenizer: *const tokenizer_t, ids: []const u32, batch: usize, length: usize, vocab_size: usize, path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const bytes = try cwd.readFileAlloc(std.testing.io, path, allocator, .limited(256 * 1024 * 1024));
    defer allocator.free(bytes);
    const position_count = std.math.mul(usize, batch, length) catch return error.InvalidInputShape;
    const value_count = std.math.mul(usize, position_count, vocab_size) catch return error.InvalidInputShape;
    if (bytes.len != value_count * 4) return error.GoldenShapeMismatch;
    const row_max = try allocator.alloc(f32, position_count);
    defer allocator.free(row_max);
    @memset(row_max, 1);
    for (0..position_count) |position| for (0..vocab_size) |vocab_id| {
        const index = position * vocab_size + vocab_id;
        const expected = @as(f32, @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little)));
        row_max[position] = @max(row_max[position], @abs(expected));
    };
    const chunk_size: usize = 512;
    const vocabulary_ids = try allocator.alloc(u32, @min(chunk_size, vocab_size));
    defer allocator.free(vocabulary_ids);
    const norm = try tokenizer.decoder_hidden(allocator, ids, batch, length);
    defer allocator.free(norm);
    var chunk_start: usize = 0;
    while (chunk_start < vocab_size) : (chunk_start += chunk_size) {
        const chunk_length = @min(chunk_size, vocab_size - chunk_start);
        for (0..chunk_length) |i| vocabulary_ids[i] = @intCast(chunk_start + i);
        {
            const actual = try tokenizer.selected_logits_from_hidden(allocator, norm, vocabulary_ids[0..chunk_length]);
            defer allocator.free(actual);
            for (0..position_count) |position| for (0..chunk_length) |i| {
                const expected_index = position * vocab_size + chunk_start + i;
                const expected = @as(f32, @bitCast(std.mem.readInt(u32, bytes[expected_index * 4 ..][0..4], .little)));
                const tolerance = 1e-4 * row_max[position];
                try std.testing.expect(@abs(expected - actual[position * chunk_length + i]) <= tolerance);
            };
        }
    }
}

fn cascade_stages(length: usize) ?usize {
    var stride: usize = 1;
    var stages: usize = 0;
    while (stride < length) : (stages += 1) {
        const next = std.math.mul(usize, stride, 4) catch return null;
        if (next > length) return null;
        stride = next;
    }
    return if (stride == length and stages != 0) stages else null;
}

fn expect_close_slices(expected: []const f32, actual: []const f32, row_width: usize) !void {
    if (expected.len != actual.len or row_width == 0 or expected.len % row_width != 0) return error.GoldenShapeMismatch;
    for (0..expected.len / row_width) |row| {
        var row_max: f32 = 1;
        for (0..row_width) |column| row_max = @max(row_max, @abs(expected[row * row_width + column]));
        const tolerance = 1e-4 * row_max;
        for (0..row_width) |column| {
            const want = expected[row * row_width + column];
            const got = actual[row * row_width + column];
            try std.testing.expect(@abs(want - got) <= tolerance);
        }
    }
}

fn softmax_rows(values: []f32, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        var max_value = -std.math.inf(f32);
        for (0..cols) |c| max_value = @max(max_value, values[r * cols + c]);
        var total: f32 = 0;
        for (0..cols) |c| {
            const v = @exp(values[r * cols + c] - max_value);
            values[r * cols + c] = v;
            total += v;
        }
        for (0..cols) |c| values[r * cols + c] /= total;
    }
}

test "nearest code returns the first minimum and rejects no valid index" {
    const codebook = [_]f32{ 1, 0, -1, 0, 0, 1 };
    try std.testing.expectEqual(@as(u32, 0), nearest_code(&.{ 0, 0 }, &codebook, 3, 2));
    try std.testing.expectEqual(@as(u32, 2), nearest_code(&.{ 0, 1 }, &codebook, 3, 2));
}

test "area downsampling uses adaptive overlapping bins" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 0, 1, 2, 3, 4, 5 };
    const output = try downsample(allocator, &input, 1, 6, 1, 4);
    defer allocator.free(output);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 1.5, 3.5, 4.5 }, output);
}

test "linear upsampling follows the half-pixel PyTorch coordinate rule" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1, 3 };
    const output = try upsample(allocator, &input, 1, 2, 4, 1);
    defer allocator.free(output);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1.5, 2.5, 3 }, output);
}

test "stride-four sampler convolutions preserve channel-major weights" {
    const input = [_]f32{ 2, 5 };
    const weight = [_]f32{ 1, 1, 1, 1 };
    const bias = [_]f32{0};
    var up: [8]f32 = undefined;
    conv_transpose1d_stride4(&input, &weight, &bias, &up, 1, 2, 1);
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2, 5, 5, 5, 5 }, &up);
    const down_weight = [_]f32{ 0.25, 0.25, 0.25, 0.25 };
    var down: [2]f32 = undefined;
    conv1d_stride4(&up, &down_weight, &bias, &down, 1, 8, 1);
    try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, &down);
}

test "VQ config requires a full hierarchy and a complete level-zero cascade" {
    try std.testing.expectError(error.InvalidConfig, validate_config(.{
        .vocab_size = 8,
        .context_length = 4,
        .width = 8,
        .codebook_size = 2,
        .levels = &.{ 1, 2 },
        .attention_heads = 2,
    }));
    try std.testing.expectError(error.InvalidConfig, validate_config(.{
        .vocab_size = 8,
        .context_length = 8,
        .width = 8,
        .codebook_size = 2,
        .levels = &.{ 1, 8 },
        .attention_heads = 2,
        .learned_level0 = true,
        .sampler_stages = 1,
    }));
}

test "weight loader rejects duplicate names and an incomplete manifest" {
    const config = config_t{
        .vocab_size = 8,
        .context_length = 4,
        .width = 8,
        .codebook_size = 2,
        .levels = &.{ 1, 4 },
        .attention_heads = 2,
    };
    const duplicate = [_]tensor_view_t{
        .{ .name = "x", .dtype = .F32, .shape = &.{1}, .data = &.{} },
        .{ .name = "x", .dtype = .F32, .shape = &.{1}, .data = &.{} },
    };
    try std.testing.expectError(error.DuplicateTensor, tokenizer_t.load(std.testing.allocator, config, &duplicate));
    try std.testing.expectError(error.TensorManifestMismatch, tokenizer_t.load(std.testing.allocator, config, &.{}));
}

test "weight loader rejects malformed shapes, dtypes, byte lengths, and names" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();
    const levels = [_]usize{ 1, 4, 16 };
    const config = config_t{
        .vocab_size = 17,
        .context_length = 16,
        .width = 8,
        .codebook_size = 4,
        .levels = &levels,
        .attention_heads = 2,
        .learned_level0 = true,
        .sampler_stages = 2,
    };
    var expected = try expected_manifest(arena_allocator, config);
    defer expected.deinit();
    const views = try arena_allocator.alloc(tensor_view_t, expected.count());
    var manifest_it = expected.iterator();
    var view_index: usize = 0;
    while (manifest_it.next()) |entry| : (view_index += 1) {
        const name = entry.key_ptr.*;
        const shape = entry.value_ptr.*;
        const is_hit = std.mem.startsWith(u8, name, "vq.codebook_hit_");
        const dtype: safetensors.Dtype = if (is_hit) .I64 else .F32;
        const bytes_per_element: usize = if (is_hit) 8 else 4;
        const data_size = try std.math.mul(usize, try shape_product(shape), bytes_per_element);
        const data = try arena_allocator.alloc(u8, data_size);
        @memset(data, 0);
        views[view_index] = .{ .name = name, .dtype = dtype, .shape = shape, .data = data };
    }
    var tokenizer = try tokenizer_t.load(allocator, config, views);
    tokenizer.deinit();
    try expect_mutated_loader_error(allocator, config, views, .shape, error.TensorShapeMismatch);
    try expect_mutated_loader_error(allocator, config, views, .parameter_dtype, error.UnsupportedTensorDtype);
    try expect_mutated_loader_error(allocator, config, views, .hit_dtype, error.UnsupportedTensorDtype);
    try expect_mutated_loader_error(allocator, config, views, .byte_length, error.TensorShapeMismatch);
    try expect_mutated_loader_error(allocator, config, views, .extra_name, error.TensorManifestMismatch);
    try expect_mutated_loader_error(allocator, config, views, .replacement_name, error.MissingTensor);
}

const loader_mutation_t = enum { shape, parameter_dtype, hit_dtype, byte_length, extra_name, replacement_name };

fn expect_mutated_loader_error(allocator: Allocator, config: config_t, valid_views: []const tensor_view_t, mutation: loader_mutation_t, expected_error: anyerror) !void {
    const add_extra = mutation == .extra_name;
    const views = try allocator.alloc(tensor_view_t, valid_views.len + @intFromBool(add_extra));
    defer allocator.free(views);
    @memcpy(views[0..valid_views.len], valid_views);
    if (add_extra) views[valid_views.len] = .{ .name = "unexpected.tensor", .dtype = .F32, .shape = &.{1}, .data = &.{ 0, 0, 0, 0 } };
    switch (mutation) {
        .shape => views[0].shape = &.{1},
        .parameter_dtype => {
            for (views) |*view| if (!std.mem.startsWith(u8, view.name, "vq.codebook_hit_")) {
                view.dtype = .I64;
                break;
            };
        },
        .hit_dtype => {
            for (views) |*view| if (std.mem.startsWith(u8, view.name, "vq.codebook_hit_")) {
                view.dtype = .F32;
                break;
            };
        },
        .byte_length => views[0].data = views[0].data[0 .. views[0].data.len - 4],
        .extra_name => {},
        .replacement_name => views[0].name = "renamed.tensor",
    }
    var tokenizer = tokenizer_t.load(allocator, config, views) catch |err| {
        try std.testing.expectEqual(expected_error, err);
        return;
    };
    tokenizer.deinit();
    return error.TestExpectedError;
}

test "tiny trained VQ matches the pinned reference fixture" {
    try run_reference_fixture(
        std.testing.allocator,
        "zig-out/golden/vq_forward/tiny_trained.json",
    );
}

test "released WMT14 VQ matches the pinned reference fixture" {
    try run_reference_fixture(
        std.testing.allocator,
        "zig-out/golden/vq_forward/wmt14_released.json",
    );
}
