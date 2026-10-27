//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Deterministic source-to-NCM tensor transplant and complete checksum report.

const std = @import("std");
const loader = @import("loader");
const ncm_manifest = @import("ncm_manifest");
const Allocator = std.mem.Allocator;

pub const TensorStatus = enum { Exact, Initialized, New, NotUsed };
pub const report_row_t = struct {
    name: []const u8,
    status: TensorStatus,
    source_names: []const []const u8,
    shape: []const u64,
    dtype: ?[]const u8,
    sha256: ?[]const u8,
};
pub const tensor_t = struct { name: []const u8, shape: []const u64, values: []f32 };
pub const transplant_t = struct {
    arena: std.heap.ArenaAllocator,
    seed: u64,
    tensors: []tensor_t,
    report: []report_row_t,

    pub fn deinit(self: *transplant_t) void {
        self.arena.deinit();
        self.* = undefined;
    }
};
pub const options_t = struct {
    vq_width: usize,
    prefix_vocab_size: usize,
    level_lengths: []const usize,
    codebook_size: usize,
    codebook_width: usize,
    post_upsample_conv_kernel: usize = 3,
    seed: u64 = 42,
};

/// Inventory order is stable: per-layer manifest order, global norm/input/query/level
/// tensors, post-upsampling convolutions by level, then codebook heads by level.
/// Random draws use Xoshiro256.floatNorm for normal values and Random.float for
/// uniform Linear values; truncation uses rejection sampling in that same stream.
pub fn transplant(parent_allocator: Allocator, source: loader.loaded_model_t, options: options_t) !transplant_t {
    var owned_arena = std.heap.ArenaAllocator.init(parent_allocator);
    errdefer owned_arena.deinit();
    const allocator = owned_arena.allocator();
    if (options.level_lengths.len == 0 or options.vq_width == 0 or options.prefix_vocab_size == 0 or options.codebook_size == 0 or options.codebook_width != options.vq_width or options.post_upsample_conv_kernel == 0 or options.post_upsample_conv_kernel % 2 == 0) return error.InvalidOptions;
    for (options.level_lengths) |length| if (length == 0) return error.InvalidOptions;
    const spec = source.spec;
    const hidden: usize = @intCast(spec.hidden_size);
    const layers: usize = @intCast(spec.num_hidden_layers);
    const query_width = try std.math.mul(usize, @intCast(spec.num_attention_heads), @intCast(spec.head_dim));
    const kv_width = try std.math.mul(usize, @intCast(spec.num_key_value_heads), @intCast(spec.head_dim));
    const bias_flags = ncm_manifest.bias_flags_t{ .qkv_bias = spec.qkv_bias, .o_bias = spec.o_bias, .mlp_bias = spec.mlp_bias };
    var targets: std.ArrayList(tensor_t) = .empty;
    const layer_manifest = try ncm_manifest.layer_weight_tensors(allocator, spec.hidden_size, spec.num_attention_heads, spec.head_dim, spec.num_key_value_heads, spec.intermediate_size, spec.qk_norm);
    defer ncm_manifest.deinit_tensors(allocator, layer_manifest);
    const bias_manifest = try ncm_manifest.bias_tensors(allocator, bias_flags, layers, spec.hidden_size, query_width, kv_width, spec.intermediate_size);
    defer ncm_manifest.deinit_tensors(allocator, bias_manifest);
    for (0..layers) |layer| {
        for (layer_manifest) |item| {
            const name = try std.fmt.allocPrint(allocator, "transformer.blocks.{d}.{s}", .{ layer, item.name["transformer.blocks.0.".len..] });
            try append_target(allocator, &targets, name, item.shape);
        }
        for (bias_manifest) |item| if (extract_layer(item.name) == layer) try append_target(allocator, &targets, item.name, item.shape);
    }
    try append_target(allocator, &targets, "transformer.ln_f.weight", &.{hidden});
    try append_target(allocator, &targets, "transformer.wte.weight", &.{ hidden, options.vq_width });
    try append_target(allocator, &targets, "transformer.wte.bias", &.{hidden});
    try append_target(allocator, &targets, "l0_queries", &.{ 1, options.level_lengths[0], options.vq_width });
    try append_target(allocator, &targets, "transformer.wle.weight", &.{ options.level_lengths.len, hidden });
    for (options.level_lengths, 0..) |_, level| {
        const conv_name = try std.fmt.allocPrint(allocator, "post_upsample_convs.{d}.weight", .{level});
        try append_target(allocator, &targets, conv_name, &.{ options.vq_width, options.vq_width, options.post_upsample_conv_kernel });
        const conv_bias = try std.fmt.allocPrint(allocator, "post_upsample_convs.{d}.bias", .{level});
        try append_target(allocator, &targets, conv_bias, &.{options.vq_width});
    }
    for (options.level_lengths, 0..) |_, level| {
        const head_weight = try std.fmt.allocPrint(allocator, "lm_heads.{d}.proj.weight", .{level});
        try append_target(allocator, &targets, head_weight, &.{ options.codebook_width, hidden });
        const head_scale = try std.fmt.allocPrint(allocator, "lm_heads.{d}.log_inv_T", .{level});
        try append_target(allocator, &targets, head_scale, &.{});
        const head_bias = try std.fmt.allocPrint(allocator, "lm_heads.{d}.bias", .{level});
        try append_target(allocator, &targets, head_bias, &.{options.codebook_size});
    }
    var rng_impl = std.Random.Xoshiro256.init(options.seed);
    const random = rng_impl.random();
    for (targets.items) |*target| {
        const source_names = try source_names_for(allocator, target.name);
        if (source_names.len > 0) {
            try copy_exact(allocator, source, target, source_names);
        } else {
            initialize_tensor(target, random, hidden);
        }
    }
    const report_len = targets.items.len + source.tensors.len + source.unused_tensors.len;
    const report = try allocator.alloc(report_row_t, report_len);
    var cursor: usize = 0;
    for (targets.items) |target| {
        const source_names = try source_names_for(allocator, target.name);
        const status: TensorStatus = if (source_names.len > 0) .Exact else if (std.mem.startsWith(u8, target.name, "lm_heads.")) .New else .Initialized;
        report[cursor] = .{ .name = target.name, .status = status, .source_names = source_names, .shape = target.shape, .dtype = "F32", .sha256 = try hash_values(allocator, target.values) };
        cursor += 1;
    }
    for (source.tensors) |item| {
        if (source_has_target(item.name, spec.num_hidden_layers)) continue;
        const owned_name = try allocator.dupe(u8, item.name);
        report[cursor] = .{ .name = owned_name, .status = .NotUsed, .source_names = try allocator.dupe([]const u8, &.{owned_name}), .shape = try source_shape(allocator, spec, item.name), .dtype = "F32", .sha256 = try hash_values(allocator, item.values) };
        cursor += 1;
    }
    for (source.unused_tensors) |info| {
        const owned_name = try allocator.dupe(u8, info.name);
        const dtype_name = try allocator.dupe(u8, @import("safetensors").dtype_name(info.dtype));
        report[cursor] = .{ .name = owned_name, .status = .NotUsed, .source_names = try allocator.dupe([]const u8, &.{owned_name}), .shape = try allocator.dupe(u64, info.shape), .dtype = dtype_name, .sha256 = null };
        cursor += 1;
    }
    const final_tensors = try targets.toOwnedSlice(allocator);
    return .{ .arena = owned_arena, .seed = options.seed, .tensors = final_tensors, .report = report[0..cursor] };
}

fn source_shape(a: Allocator, spec: @TypeOf(@as(loader.loaded_model_t, undefined).spec), name: []const u8) ![]const u64 {
    if (std.mem.eql(u8, name, "model.embed_tokens.weight") or std.mem.eql(u8, name, "lm_head.weight")) return a.dupe(u64, &.{ spec.vocab_size, spec.hidden_size });
    if (std.mem.eql(u8, name, "model.norm.weight")) return a.dupe(u64, &.{spec.hidden_size});
    return error.UnknownSourceShape;
}

fn extract_layer(name: []const u8) ?usize {
    const start = "transformer.blocks.".len;
    if (!std.mem.startsWith(u8, name, "transformer.blocks.")) return null;
    const end = std.mem.indexOfScalarPos(u8, name, start, '.') orelse return null;
    return std.fmt.parseInt(usize, name[start..end], 10) catch null;
}
fn append_target(a: Allocator, list: *std.ArrayList(tensor_t), name: []const u8, shape: []const u64) !void {
    var count: usize = 1;
    for (shape) |dim| count = try std.math.mul(usize, count, @intCast(dim));
    const owned_name = try a.dupe(u8, name);
    errdefer a.free(owned_name);
    const owned_shape = try a.dupe(u64, shape);
    errdefer a.free(owned_shape);
    const values = try a.alloc(f32, count);
    try list.append(a, .{ .name = owned_name, .shape = owned_shape, .values = values });
}
fn source_names_for(a: Allocator, target: []const u8) ![]const []const u8 {
    if (std.mem.eql(u8, target, "transformer.ln_f.weight")) return a.dupe([]const u8, &.{"model.norm.weight"});
    if (!std.mem.startsWith(u8, target, "transformer.blocks.")) return &.{};
    const layer = extract_layer(target) orelse return error.InvalidTargetName;
    const layer_end = std.mem.indexOfScalarPos(u8, target, "transformer.blocks.".len, '.') orelse return error.InvalidTargetName;
    const suffix = target[layer_end + 1 ..];
    const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{layer});
    const pairs = .{
        .{ "ln_1.weight", "input_layernorm.weight" },         .{ "ln_2.weight", "post_attention_layernorm.weight" },
        .{ "attn.c_proj.weight", "self_attn.o_proj.weight" }, .{ "attn.c_proj.bias", "self_attn.o_proj.bias" },
        .{ "attn.q_norm.weight", "self_attn.q_norm.weight" }, .{ "attn.k_norm.weight", "self_attn.k_norm.weight" },
        .{ "mlp.w_gate.weight", "mlp.gate_proj.weight" },     .{ "mlp.w_up.weight", "mlp.up_proj.weight" },
        .{ "mlp.c_proj.weight", "mlp.down_proj.weight" },     .{ "mlp.w_gate.bias", "mlp.gate_proj.bias" },
        .{ "mlp.w_up.bias", "mlp.up_proj.bias" },             .{ "mlp.c_proj.bias", "mlp.down_proj.bias" },
    };
    inline for (pairs) |pair| if (std.mem.eql(u8, suffix, pair[0])) return a.dupe([]const u8, &.{try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, pair[1] })});
    if (std.mem.eql(u8, suffix, "attn.c_attn.weight") or std.mem.eql(u8, suffix, "attn.c_attn.bias")) {
        const suffix_source = if (std.mem.endsWith(u8, suffix, ".weight")) "weight" else "bias";
        const attn_prefix = try std.fmt.allocPrint(a, "{s}self_attn.", .{prefix});
        const q = try std.fmt.allocPrint(a, "{s}q_proj.{s}", .{ attn_prefix, suffix_source });
        const k = try std.fmt.allocPrint(a, "{s}k_proj.{s}", .{ attn_prefix, suffix_source });
        const v = try std.fmt.allocPrint(a, "{s}v_proj.{s}", .{ attn_prefix, suffix_source });
        return a.dupe([]const u8, &.{ q, k, v });
    }
    return &.{};
}
fn copy_exact(a: Allocator, model: loader.loaded_model_t, target: *tensor_t, names: []const []const u8) !void {
    if (names.len == 1) {
        const item = find_source(model, names[0]) orelse return error.MissingSourceTensor;
        if (item.values.len != target.values.len) return error.SourceShapeMismatch;
        @memcpy(target.values, item.values);
        return;
    }
    var offset: usize = 0;
    for (names) |name| {
        const item = find_source(model, name) orelse return error.MissingSourceTensor;
        if (item.values.len > target.values.len - offset) return error.SourceShapeMismatch;
        @memcpy(target.values[offset..][0..item.values.len], item.values);
        offset += item.values.len;
    }
    _ = a;
    if (offset != target.values.len) return error.SourceShapeMismatch;
}
fn find_source(model: loader.loaded_model_t, name: []const u8) ?loader.loaded_tensor_t {
    for (model.tensors) |item| if (std.mem.eql(u8, item.name, name)) return item;
    return null;
}
fn source_has_target(name: []const u8, layer_count: u64) bool {
    if (std.mem.eql(u8, name, "model.norm.weight")) return true;
    if (!std.mem.startsWith(u8, name, "model.layers.")) return false;
    const start = "model.layers.".len;
    const end = std.mem.indexOfScalarPos(u8, name, start, '.') orelse return false;
    const index = std.fmt.parseInt(u64, name[start..end], 10) catch return false;
    return index < layer_count and !std.mem.endsWith(u8, name, "rotary_emb.inv_freq");
}
fn initialize_tensor(target: *tensor_t, random: std.Random, hidden: usize) void {
    const name = target.name;
    if (std.mem.endsWith(u8, name, ".bias") or std.mem.endsWith(u8, name, ".log_inv_T")) {
        @memset(target.values, 0);
        return;
    }
    if (std.mem.endsWith(u8, name, "ln_f.weight")) {
        @memset(target.values, 1);
        return;
    }
    if (std.mem.startsWith(u8, name, "post_upsample_convs.")) {
        @memset(target.values, 0);
        return;
    }
    if (std.mem.eql(u8, name, "l0_queries")) {
        const stddev = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden)));
        for (target.values) |*value| {
            var draw: f32 = undefined;
            while (true) {
                draw = random.floatNorm(f32) * stddev;
                if (@abs(draw) <= 2) break;
            }
            value.* = draw;
        }
        return;
    }
    if (std.mem.eql(u8, name, "transformer.wle.weight")) {
        for (target.values) |*v| v.* = random.floatNorm(f32);
        return;
    }
    const fan_in: f32 = if (target.shape.len > 1) @floatFromInt(target.shape[target.shape.len - 1]) else 1;
    const bound = 1.0 / @sqrt(fan_in);
    for (target.values) |*v| v.* = (random.float(f32) * 2 - 1) * bound;
}
fn hash_values(a: Allocator, values: []const f32) ![]const u8 {
    return hash_bytes(a, std.mem.sliceAsBytes(values));
}
fn hash_bytes(a: Allocator, bytes: []const u8) ![]const u8 {
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(digest, .lower));
}

fn find_target(result: transplant_t, name: []const u8) ?tensor_t {
    for (result.tensors) |tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor;
    return null;
}

fn expected_source_names_for_test(a: Allocator, target: []const u8) ![]const []const u8 {
    if (std.mem.eql(u8, target, "transformer.ln_f.weight")) return a.dupe([]const u8, &.{"model.norm.weight"});
    if (!std.mem.startsWith(u8, target, "transformer.blocks.")) return &.{};
    const start = "transformer.blocks.".len;
    const end = std.mem.indexOfScalarPos(u8, target, start, '.') orelse return error.InvalidTargetName;
    const layer = try std.fmt.parseInt(usize, target[start..end], 10);
    const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{layer});
    const suffix = target[end + 1 ..];
    const direct = .{
        .{ "ln_1.weight", "input_layernorm.weight" },
        .{ "ln_2.weight", "post_attention_layernorm.weight" },
        .{ "attn.c_proj.weight", "self_attn.o_proj.weight" },
        .{ "attn.c_proj.bias", "self_attn.o_proj.bias" },
        .{ "attn.q_norm.weight", "self_attn.q_norm.weight" },
        .{ "attn.k_norm.weight", "self_attn.k_norm.weight" },
        .{ "mlp.w_gate.weight", "mlp.gate_proj.weight" },
        .{ "mlp.w_up.weight", "mlp.up_proj.weight" },
        .{ "mlp.c_proj.weight", "mlp.down_proj.weight" },
        .{ "mlp.w_gate.bias", "mlp.gate_proj.bias" },
        .{ "mlp.w_up.bias", "mlp.up_proj.bias" },
        .{ "mlp.c_proj.bias", "mlp.down_proj.bias" },
    };
    inline for (direct) |pair| {
        if (std.mem.eql(u8, suffix, pair[0])) {
            return a.dupe([]const u8, &.{try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, pair[1] })});
        }
    }
    if (std.mem.eql(u8, suffix, "attn.c_attn.weight") or std.mem.eql(u8, suffix, "attn.c_attn.bias")) {
        const parameter = if (std.mem.endsWith(u8, suffix, ".weight")) "weight" else "bias";
        return a.dupe([]const u8, &.{
            try std.fmt.allocPrint(a, "{s}self_attn.q_proj.{s}", .{ prefix, parameter }),
            try std.fmt.allocPrint(a, "{s}self_attn.k_proj.{s}", .{ prefix, parameter }),
            try std.fmt.allocPrint(a, "{s}self_attn.v_proj.{s}", .{ prefix, parameter }),
        });
    }
    return &.{};
}

fn contains_name(names: []const []const u8, name: []const u8) bool {
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}

fn report_row_count(result: transplant_t, name: []const u8) usize {
    var count: usize = 0;
    for (result.report) |row| if (std.mem.eql(u8, row.name, name)) {
        count += 1;
    };
    return count;
}

fn expect_complete_report(a: Allocator, source: loader.loaded_model_t, result: transplant_t) !void {
    var expected_row_count = result.tensors.len + source.unused_tensors.len;
    for (source.tensors) |item| {
        var mapped = false;
        for (result.tensors) |target| {
            const names = try expected_source_names_for_test(a, target.name);
            if (contains_name(names, item.name)) {
                mapped = true;
                break;
            }
        }
        if (!mapped) expected_row_count += 1;
    }
    try std.testing.expectEqual(expected_row_count, result.report.len);

    var all_report_names: std.ArrayList([]const u8) = .empty;
    for (result.report) |row| {
        try std.testing.expect(!contains_name(all_report_names.items, row.name));
        try all_report_names.append(a, row.name);
    }

    for (result.tensors, 0..) |target, index| {
        const row = result.report[index];
        const expected_sources = try expected_source_names_for_test(a, target.name);
        try std.testing.expectEqual(@as(usize, 1), report_row_count(result, target.name));
        try std.testing.expectEqualStrings(target.name, row.name);
        try std.testing.expectEqualSlices(u64, target.shape, row.shape);
        try std.testing.expectEqualStrings("F32", row.dtype.?);
        try std.testing.expectEqualSlices(u8, try hash_values(a, target.values), row.sha256.?);
        try std.testing.expectEqual(expected_sources.len, row.source_names.len);
        for (expected_sources, row.source_names) |expected, actual| try std.testing.expectEqualStrings(expected, actual);
        const expected_status: TensorStatus = if (expected_sources.len != 0) .Exact else if (std.mem.startsWith(u8, target.name, "lm_heads.")) .New else .Initialized;
        try std.testing.expectEqual(expected_status, row.status);
    }

    for (source.tensors) |item| {
        var mapped = false;
        for (result.tensors) |target| {
            const names = try expected_source_names_for_test(a, target.name);
            if (contains_name(names, item.name)) {
                mapped = true;
                break;
            }
        }
        if (mapped) continue;
        try std.testing.expectEqual(@as(usize, 1), report_row_count(result, item.name));
        const row = find_report_row(result, item.name).?;
        try std.testing.expectEqual(TensorStatus.NotUsed, row.status);
        try std.testing.expectEqual(@as(usize, 1), row.source_names.len);
        try std.testing.expectEqualStrings(item.name, row.source_names[0]);
        try std.testing.expectEqualStrings("F32", row.dtype.?);
        try std.testing.expectEqualSlices(u8, try hash_values(a, item.values), row.sha256.?);
        const expected_shape = if (std.mem.eql(u8, item.name, "model.embed_tokens.weight") or std.mem.eql(u8, item.name, "lm_head.weight"))
            &[_]u64{ source.spec.vocab_size, source.spec.hidden_size }
        else if (std.mem.eql(u8, item.name, "model.norm.weight"))
            &[_]u64{source.spec.hidden_size}
        else
            return error.MissingExpectedSourceShape;
        try std.testing.expectEqualSlices(u64, expected_shape, row.shape);
    }

    for (source.unused_tensors) |item| {
        try std.testing.expectEqual(@as(usize, 1), report_row_count(result, item.name));
        const row = find_report_row(result, item.name).?;
        try std.testing.expectEqual(TensorStatus.NotUsed, row.status);
        try std.testing.expectEqual(@as(usize, 1), row.source_names.len);
        try std.testing.expectEqualStrings(item.name, row.source_names[0]);
        try std.testing.expectEqualSlices(u64, item.shape, row.shape);
        try std.testing.expectEqualStrings(@import("safetensors").dtype_name(item.dtype), row.dtype.?);
        try std.testing.expect(row.sha256 == null);
    }
}

fn find_report_row(result: transplant_t, name: []const u8) ?report_row_t {
    for (result.report) |row| if (std.mem.eql(u8, row.name, name)) return row;
    return null;
}

fn expect_replayed_initialization(first: transplant_t, second: transplant_t) !void {
    try std.testing.expectEqual(first.tensors.len, second.tensors.len);
    for (first.tensors, 0..) |left, index| {
        const right = second.tensors[index];
        try std.testing.expectEqualStrings(left.name, right.name);
        if (first.report[index].status == .Exact) continue;
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(left.values), std.mem.sliceAsBytes(right.values));
        try std.testing.expectEqualSlices(u8, first.report[index].sha256.?, second.report[index].sha256.?);
    }
}

test "M2 transplant covers twelve layers, fused QKV, initialization replay and hashes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var inputs: std.ArrayList(loader.loaded_tensor_t) = .empty;
    const spec = @import("hf_config").config.spec_t{
        .family = .Llama,
        .vocab_size = 5,
        .hidden_size = 4,
        .intermediate_size = 3,
        .num_hidden_layers = 12,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .head_dim = 2,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000,
        .rope_scaling = .Default,
        .qkv_bias = true,
        .o_bias = false,
        .mlp_bias = false,
        .qk_norm = false,
        .tie_word_embeddings = true,
        .sliding_window = null,
    };
    var layer_index: usize = 0;
    while (layer_index < 12) : (layer_index += 1) {
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.", .{layer_index});
        const names = [_][]const u8{
            "input_layernorm.weight",          "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.q_proj.bias",           "self_attn.k_proj.bias",   "self_attn.v_proj.bias",   "self_attn.o_proj.weight",
            "post_attention_layernorm.weight", "mlp.gate_proj.weight",    "mlp.up_proj.weight",      "mlp.down_proj.weight",
        };
        for (names, 0..) |suffix, index| {
            const name = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix });
            const count: usize = if (index == 1 or index == 2 or index == 3 or index == 7) 8 else if (index == 0 or index == 8) 4 else if (index == 4 or index == 5 or index == 6) 2 else 12;
            const vals = try a.alloc(f32, count);
            for (vals, 0..) |*v, i| v.* = @floatFromInt(layer_index * 1000 + index * 100 + i);
            try inputs.append(a, .{ .name = name, .values = vals });
        }
    }
    try inputs.append(a, .{ .name = "model.norm.weight", .values = try a.dupe(f32, &.{ 1, 2, 3, 4 }) });
    const embed = try a.alloc(f32, 20);
    const head = try a.alloc(f32, 20);
    @memset(embed, 1);
    @memset(head, 2);
    try inputs.append(a, .{ .name = "model.embed_tokens.weight", .values = embed });
    try inputs.append(a, .{ .name = "lm_head.weight", .values = head });
    const unused_info = @import("safetensors").tensor_info_t{ .name = "model.layers.0.self_attn.rotary_emb.inv_freq", .dtype = .Bf16, .shape = &.{1}, .data_begin = 0, .data_end = 2 };
    const model = loader.loaded_model_t{ .spec = spec, .weights = undefined, .tensors = inputs.items, .unused_tensors = &.{unused_info} };
    const opts = options_t{ .vq_width = 2, .prefix_vocab_size = 5, .level_lengths = &.{ 1, 2 }, .codebook_size = 7, .codebook_width = 2, .seed = 42 };
    var first = try transplant(a, model, opts);
    defer first.deinit();
    var second = try transplant(a, model, opts);
    defer second.deinit();
    try expect_complete_report(a, model, first);
    try expect_replayed_initialization(first, second);
    const layer10_qkv = find_target(first, "transformer.blocks.10.attn.c_attn.weight").?;
    try std.testing.expectEqualSlices(f32, &.{ 10100, 10101, 10102, 10103, 10104, 10105, 10106, 10107, 10200, 10201, 10202, 10203, 10204, 10205, 10206, 10207, 10300, 10301, 10302, 10303, 10304, 10305, 10306, 10307 }, layer10_qkv.values);
    const layer10_sources = try source_names_for(a, layer10_qkv.name);
    try std.testing.expectEqual(@as(usize, 3), layer10_sources.len);
    try std.testing.expectEqualStrings("model.layers.10.self_attn.q_proj.weight", layer10_sources[0]);
    var mapped_count: usize = 0;
    for (first.report[0..first.tensors.len], 0..) |row, row_index| {
        for (row.source_names) |source_name| {
            mapped_count += 1;
            for (first.report[0..row_index]) |previous| for (previous.source_names) |previous_name| {
                try std.testing.expect(!std.mem.eql(u8, source_name, previous_name));
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 145), mapped_count);
    for (first.tensors) |target| {
        const names = try source_names_for(a, target.name);
        if (names.len == 0) continue;
        const expected = try a.alloc(f32, target.values.len);
        var offset: usize = 0;
        for (names) |name| {
            const input = find_source(model, name).?;
            @memcpy(expected[offset..][0..input.values.len], input.values);
            offset += input.values.len;
        }
        try std.testing.expectEqual(target.values.len, offset);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(target.values));
    }
    var unique_target_count: usize = 0;
    for (first.tensors, 0..) |tensor, index| {
        unique_target_count += 1;
        for (first.tensors[0..index]) |previous| try std.testing.expect(!std.mem.eql(u8, tensor.name, previous.name));
        try std.testing.expectEqualSlices(u8, first.report[index].sha256.?, try hash_values(a, tensor.values));
    }
    try std.testing.expectEqual(first.tensors.len, unique_target_count);
    try std.testing.expectEqualSlices(f32, find_target(first, "transformer.wte.weight").?.values, find_target(second, "transformer.wte.weight").?.values);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", first.report[first.report.len - 3].name);
    try std.testing.expectEqualStrings("lm_head.weight", first.report[first.report.len - 2].name);
    try std.testing.expectEqualStrings("model.layers.0.self_attn.rotary_emb.inv_freq", first.report[first.report.len - 1].name);
    try std.testing.expectEqualStrings("BF16", first.report[first.report.len - 1].dtype.?);
    try std.testing.expect(first.report[first.report.len - 1].sha256 == null);
    try std.testing.expectEqualSlices(u64, &.{1}, first.report[first.report.len - 1].shape);
    try std.testing.expectEqualSlices(u8, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", try hash_bytes(a, "abc"));
    try std.testing.expectEqualSlices(u64, &.{ 1, 1, 2 }, find_target(first, "l0_queries").?.shape);
    try std.testing.expectEqual(@as(usize, 1), find_target(first, "lm_heads.0.log_inv_T").?.values.len);
    try std.testing.expectEqual(@as(f32, 0), find_target(first, "lm_heads.0.log_inv_T").?.values[0]);
    for (find_target(first, "transformer.wte.bias").?.values) |v| try std.testing.expectEqual(@as(f32, 0), v);
    for (find_target(first, "post_upsample_convs.0.weight").?.values) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

fn allocation_probe(a: Allocator) !void {
    const source_values = [_]f32{ 1, 1 };
    const source_tensors = [_]loader.loaded_tensor_t{.{ .name = "model.norm.weight", .values = &source_values }};
    const spec = @import("hf_config").config.spec_t{
        .family = .Llama,
        .vocab_size = 2,
        .hidden_size = 2,
        .intermediate_size = 2,
        .num_hidden_layers = 0,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .head_dim = 2,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10000,
        .rope_scaling = .Default,
        .qkv_bias = false,
        .o_bias = false,
        .mlp_bias = false,
        .qk_norm = false,
        .tie_word_embeddings = true,
        .sliding_window = null,
    };
    const model = loader.loaded_model_t{ .spec = spec, .weights = undefined, .tensors = &source_tensors, .unused_tensors = &.{} };
    var result = try transplant(a, model, .{ .vq_width = 2, .prefix_vocab_size = 2, .level_lengths = &.{1}, .codebook_size = 2, .codebook_width = 2 });
    result.deinit();
}

test "M2 transplant releases allocations on every failure path" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocation_probe, .{});
}

test "M2 transplant consumes generated family checkpoints and reports all exact bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_][]const u8{ "llama_default", "llama_default_bf16", "llama_llama3", "qwen2", "qwen3", "mistral", "qwen3_attention_bias" };
    for (cases) |case_name| {
        const path = try std.fmt.allocPrint(a, "zig-out/golden/{s}", .{case_name});
        _ = std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fmt.allocPrint(a, "{s}/config.json", .{path}), a, .limited(1 << 20)) catch |err| {
            if (err == error.FileNotFound) {
                std.debug.print("SKIP M2 generated-checkpoint gate: missing {s}; generate all fixtures with `python tools/golden_helper.py --output-dir zig-out/golden`\n", .{path});
                return error.SkipZigTest;
            }
            return err;
        };
        const source = try loader.load_checkpoint(a, std.testing.io, path);
        var output = try transplant(a, source, .{ .vq_width = 4, .prefix_vocab_size = 8, .level_lengths = &.{ 1, 2 }, .codebook_size = 7, .codebook_width = 4, .seed = 42 });
        defer output.deinit();
        var replay = try transplant(a, source, .{ .vq_width = 4, .prefix_vocab_size = 8, .level_lengths = &.{ 1, 2 }, .codebook_size = 7, .codebook_width = 4, .seed = 42 });
        defer replay.deinit();
        try expect_complete_report(a, source, output);
        try expect_replayed_initialization(output, replay);
        var exact_source_count: usize = 0;
        for (output.tensors, 0..) |target, index| {
            const row = output.report[index];
            try std.testing.expectEqualStrings(target.name, row.name);
            try std.testing.expectEqualSlices(u8, try hash_values(a, target.values), row.sha256.?);
            const names = row.source_names;
            if (names.len == 0) continue;
            exact_source_count += names.len;
            const expected = try a.alloc(f32, target.values.len);
            var offset: usize = 0;
            for (names) |name| {
                const item = find_source(source, name) orelse return error.MissingSourceTensor;
                @memcpy(expected[offset..][0..item.values.len], item.values);
                offset += item.values.len;
            }
            try std.testing.expectEqual(target.values.len, offset);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(expected), std.mem.sliceAsBytes(target.values));
        }
        var expected_exact_source_count: usize = 1;
        for (source.tensors) |item| if (source_has_target(item.name, source.spec.num_hidden_layers)) {
            if (!std.mem.eql(u8, item.name, "model.norm.weight")) expected_exact_source_count += 1;
        };
        try std.testing.expectEqual(expected_exact_source_count, exact_source_count);
        for (source.unused_tensors, 0..) |unused, unused_index| {
            const row = output.report[output.tensors.len + (output.report.len - output.tensors.len - source.unused_tensors.len) + unused_index];
            try std.testing.expectEqualStrings(unused.name, row.name);
            try std.testing.expect(row.sha256 == null);
            try std.testing.expectEqualSlices(u64, unused.shape, row.shape);
            try std.testing.expectEqualStrings(@import("safetensors").dtype_name(unused.dtype), row.dtype.?);
        }
    }
}
