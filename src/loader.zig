//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Manifest-first safetensors loading for the source decoder.

const std = @import("std");
const bf16 = @import("bf16");
const decoder = @import("decoder");
const ncm_body = @import("ncm_body");
const ncm_mask = @import("ncm_mask");
const ncm_position = @import("ncm_position");
const ncm_norm = @import("ncm_norm");
const ncm_model = @import("ncm_model");
const hf_config = @import("hf_config");
const safetensors = @import("safetensors");

const Allocator = std.mem.Allocator;
const tensor_map_t = std.StringHashMapUnmanaged([]f32);
const ChunkBytes = 64 * 1024;

const shard_t = struct {
    name: []const u8,
    header: safetensors.header_t,
};

const tensor_location_t = struct {
    info: safetensors.tensor_info_t,
    shard_name: []const u8,
    header_len: u64,
};

pub const loaded_model_t = struct {
    spec: hf_config.config.spec_t,
    weights: decoder.model_weights_t,
    tensors: []const loaded_tensor_t,
    unused_tensors: []const safetensors.tensor_info_t,
};

pub const loaded_tensor_t = struct {
    name: []const u8,
    values: []const f32,
};

pub const rejection_kind_t = enum { InvalidConfig, InvalidManifest, UnsupportedPickleWeights };

pub const rejection_t = struct {
    kind: rejection_kind_t,
    problems: []const hf_config.config.problem_t,
};

pub const load_result_t = union(enum) {
    loaded: loaded_model_t,
    rejected: rejection_t,
};

/// Reads and validates all checkpoint headers before reading tensor data. The returned f32
/// slices are allocated from `allocator`, which must outlive the returned model.
pub fn load_checkpoint(
    allocator: Allocator,
    io: std.Io,
    checkpoint_path: []const u8,
) !loaded_model_t {
    return load_checkpoint_impl(allocator, io, checkpoint_path, null);
}

/// Returns named configuration and manifest problems for checkpoints that are not loadable.
/// The diagnostic slices are allocated from `allocator`.
pub fn load_checkpoint_report(
    allocator: Allocator,
    io: std.Io,
    checkpoint_path: []const u8,
) !load_result_t {
    return load_checkpoint_report_impl(allocator, io, checkpoint_path, null);
}

fn load_checkpoint_impl(
    allocator: Allocator,
    io: std.Io,
    checkpoint_path: []const u8,
    weight_read_count: ?*usize,
) !loaded_model_t {
    return unwrap_load_result(try load_checkpoint_report_impl(allocator, io, checkpoint_path, weight_read_count));
}

fn unwrap_load_result(result: load_result_t) error{ InvalidConfig, InvalidManifest, UnsupportedPickleWeights }!loaded_model_t {
    return switch (result) {
        .loaded => |model| model,
        .rejected => |rejection| switch (rejection.kind) {
            .InvalidConfig => error.InvalidConfig,
            .InvalidManifest => error.InvalidManifest,
            .UnsupportedPickleWeights => error.UnsupportedPickleWeights,
        },
    };
}

fn load_checkpoint_report_impl(
    allocator: Allocator,
    io: std.Io,
    checkpoint_path: []const u8,
    weight_read_count: ?*usize,
) !load_result_t {
    var directory = try std.Io.Dir.cwd().openDir(io, checkpoint_path, .{});
    defer directory.close(io);

    const config_bytes = try directory.readFileAlloc(io, "config.json", allocator, .limited(1 << 20));
    const config_report = hf_config.config.parse_config(allocator, config_bytes) catch |err| {
        if (err == error.OutOfMemory) return err;
        const problems = try allocator.alloc(hf_config.config.problem_t, 1);
        problems[0] = .{ .feature = "config.json", .message = "invalid JSON; provide a valid supported decoder configuration" };
        return .{ .rejected = .{ .kind = .InvalidConfig, .problems = problems } };
    };
    const spec = config_report.spec orelse return .{ .rejected = .{ .kind = .InvalidConfig, .problems = config_report.problems } };

    const index_bytes = directory.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    var parsed_index: ?safetensors.index_t = null;
    var shard_names: std.ArrayList([]const u8) = .empty;
    if (index_bytes) |bytes| {
        parsed_index = safetensors.parse_index(allocator, bytes) catch |err| {
            const problems = try allocator.alloc(hf_config.config.problem_t, 1);
            problems[0] = .{ .feature = "model.safetensors.index.json", .message = try std.fmt.allocPrint(allocator, "invalid safetensors index: {s}", .{@errorName(err)}) };
            return .{ .rejected = .{ .kind = .InvalidManifest, .problems = problems } };
        };
        for (parsed_index.?.weight_map) |entry| {
            if (!contains_name(shard_names.items, entry.shard)) try shard_names.append(allocator, entry.shard);
        }
    } else {
        try shard_names.append(allocator, "model.safetensors");
    }

    var shards: std.ArrayList(shard_t) = .empty;
    var locations: std.ArrayList(tensor_location_t) = .empty;
    var present: std.ArrayList(safetensors.tensor_info_t) = .empty;
    for (shard_names.items) |shard_name| {
        const header = read_header(allocator, io, directory, shard_name) catch |err| {
            if (err == error.FileNotFound and index_bytes == null) {
                if (try find_pickle_weights(io, directory)) |pickle_name| {
                    const problems = try allocator.alloc(hf_config.config.problem_t, 1);
                    problems[0] = .{
                        .feature = pickle_name,
                        .message = "pickle-based weights are not supported; convert the checkpoint to safetensors outside ncmorph, then retry",
                    };
                    return .{ .rejected = .{ .kind = .UnsupportedPickleWeights, .problems = problems } };
                }
            }
            return err;
        };
        try shards.append(allocator, .{ .name = shard_name, .header = header });
        try present.appendSlice(allocator, header.tensors);
        for (header.tensors) |info| {
            try locations.append(allocator, .{
                .info = info,
                .shard_name = shard_name,
                .header_len = header.header_len,
            });
        }
    }

    const manifest_result = try hf_config.manifest.check(allocator, spec, present.items);
    var problems: std.ArrayList(hf_config.config.problem_t) = .empty;
    try problems.appendSlice(allocator, manifest_result.problems);
    if (parsed_index) |index| {
        const shard_views = try allocator.alloc(hf_config.manifest.shard_t, shards.items.len);
        for (shards.items, shard_views) |shard, *view| {
            view.* = .{ .file = shard.name, .tensors = shard.header.tensors };
        }
        const index_problems = try hf_config.manifest.check_index(allocator, index, shard_views);
        try problems.appendSlice(allocator, index_problems);
    }
    if (problems.items.len != 0) {
        return .{ .rejected = .{ .kind = .InvalidManifest, .problems = problems.items } };
    }

    var tensors: tensor_map_t = .empty;
    defer tensors.deinit(allocator);
    var loaded_tensors: std.ArrayList(loaded_tensor_t) = .empty;
    var unused_tensors: std.ArrayList(safetensors.tensor_info_t) = .empty;
    for (locations.items) |location| {
        if (is_unused_tensor(location.info.name)) {
            try unused_tensors.append(allocator, location.info);
            continue;
        }
        const values = try read_tensor(
            allocator,
            io,
            directory,
            location,
            weight_read_count,
        );
        try tensors.put(allocator, location.info.name, values);
        try loaded_tensors.append(allocator, .{ .name = location.info.name, .values = values });
    }

    var all_bf16 = true;
    for (locations.items) |location| {
        if (is_unused_tensor(location.info.name)) continue;
        if (location.info.dtype != .Bf16) all_bf16 = false;
    }
    const model = loaded_model_t{
        .spec = spec,
        .weights = try build_weights(allocator, spec, &tensors, all_bf16),
        .tensors = loaded_tensors.items,
        .unused_tensors = unused_tensors.items,
    };
    return .{ .loaded = model };
}

fn is_unused_tensor(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".self_attn.rotary_emb.inv_freq");
}

test "optional rotary inverse frequencies are reported as unused tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const source_path = "zig-out/golden/llama_default";
    const source_config = try std.fmt.allocPrint(allocator, "{s}/config.json", .{source_path});
    const config = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        source_config,
        allocator,
        .limited(1 << 20),
    ) catch |err| return skip_without_golden(err, source_config);
    const source_weights = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{source_path});
    const weights = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        source_weights,
        allocator,
        .limited(16 << 20),
    ) catch |err| return skip_without_golden(err, source_weights);

    const old_header_len = try safetensors.header_length(weights[0..safetensors.PrefixLen]);
    const old_header_end = safetensors.PrefixLen + @as(usize, @intCast(old_header_len));
    var closing_brace = old_header_end;
    while (closing_brace > safetensors.PrefixLen and weights[closing_brace - 1] != '}') closing_brace -= 1;
    if (closing_brace == safetensors.PrefixLen) return error.InvalidFixture;
    closing_brace -= 1;
    const original_header = weights[safetensors.PrefixLen..old_header_end];
    const data_len = weights.len - old_header_end;
    const added_tensor_name = "model.layers.0.self_attn.rotary_emb.inv_freq";
    const new_header = try std.fmt.allocPrint(
        allocator,
        "{s},\"{s}\":{{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[{d},{d}]}}}}{s}",
        .{ original_header[0 .. closing_brace - safetensors.PrefixLen], added_tensor_name, data_len, data_len + 8, original_header[closing_brace - safetensors.PrefixLen + 1 ..] },
    );
    const new_header_len = new_header.len;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    try write_text_file(temporary.dir, "config.json", config);
    var file = try temporary.dir.createFile(std.testing.io, "model.safetensors", .{});
    defer file.close(std.testing.io);
    var prefix: [safetensors.PrefixLen]u8 = undefined;
    std.mem.writeInt(u64, &prefix, new_header_len, .little);
    try file.writeStreamingAll(std.testing.io, &prefix);
    try file.writeStreamingAll(std.testing.io, new_header);
    try file.writeStreamingAll(std.testing.io, weights[old_header_end..]);
    try file.writeStreamingAll(std.testing.io, &.{ 0, 0, 0, 0, 0, 0, 0, 0 });

    var reads: usize = 0;
    const checkpoint_path = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}", .{temporary.sub_path});
    const result = try load_checkpoint_report_impl(allocator, std.testing.io, checkpoint_path, &reads);
    const loaded = switch (result) {
        .loaded => |model| model,
        .rejected => return error.InvalidFixture,
    };
    try std.testing.expectEqual(@as(usize, 1), loaded.unused_tensors.len);
    try std.testing.expectEqualStrings(added_tensor_name, loaded.unused_tensors[0].name);
    try std.testing.expectEqual(safetensors.Dtype.F32, loaded.unused_tensors[0].dtype);
    try std.testing.expectEqualSlices(u64, &.{2}, loaded.unused_tensors[0].shape);
    try std.testing.expect(find_loaded_tensor(loaded.tensors, added_tensor_name) == null);
    try std.testing.expectEqual(loaded.tensors.len, reads);
}

fn find_pickle_weights(io: std.Io, directory: std.Io.Dir) !?[]const u8 {
    const candidates = [_][]const u8{
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.bin",
        "model.bin.index.json",
        "pytorch_model.pt",
        "model.pt",
        "pytorch_model.pth",
        "model.pth",
    };
    for (candidates) |candidate| {
        directory.access(io, candidate, .{}) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        return candidate;
    }
    return null;
}

fn build_weights(allocator: Allocator, spec: hf_config.config.spec_t, tensors: *const tensor_map_t, bf16_compute: bool) !decoder.model_weights_t {
    const layer_count: usize = @intCast(spec.num_hidden_layers);
    const layers = try allocator.alloc(decoder.layer_weights_t, layer_count);
    for (layers, 0..) |*layer, index| {
        const prefix = try std.fmt.allocPrint(allocator, "model.layers.{d}.", .{index});
        const attention_prefix = try std.fmt.allocPrint(allocator, "{s}self_attn.", .{prefix});
        const mlp_prefix = try std.fmt.allocPrint(allocator, "{s}mlp.", .{prefix});
        layer.* = .{
            .input_norm = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}input_layernorm.weight", .{prefix})),
            .attention = .{
                .query_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}q_proj.weight", .{attention_prefix})),
                .query_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}q_proj.bias", .{attention_prefix})),
                .key_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}k_proj.weight", .{attention_prefix})),
                .key_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}k_proj.bias", .{attention_prefix})),
                .value_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}v_proj.weight", .{attention_prefix})),
                .value_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}v_proj.bias", .{attention_prefix})),
                .query_norm_weight = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}q_norm.weight", .{attention_prefix})),
                .key_norm_weight = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}k_norm.weight", .{attention_prefix})),
                .output_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}o_proj.weight", .{attention_prefix})),
                .output_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}o_proj.bias", .{attention_prefix})),
            },
            .post_attention_norm = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}post_attention_layernorm.weight", .{prefix})),
            .gate_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}gate_proj.weight", .{mlp_prefix})),
            .gate_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}gate_proj.bias", .{mlp_prefix})),
            .up_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}up_proj.weight", .{mlp_prefix})),
            .up_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}up_proj.bias", .{mlp_prefix})),
            .down_weight = try required_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}down_proj.weight", .{mlp_prefix})),
            .down_bias = optional_tensor(tensors, try std.fmt.allocPrint(allocator, "{s}down_proj.bias", .{mlp_prefix})),
        };
    }

    const embed_tokens = try required_tensor(tensors, "model.embed_tokens.weight");
    const final_norm = try required_tensor(tensors, "model.norm.weight");
    const lm_head = optional_tensor(tensors, "lm_head.weight");
    if (spec.tie_word_embeddings) {
        if (lm_head) |head| {
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(embed_tokens), std.mem.sliceAsBytes(head))) {
                return error.TiedHeadMismatch;
            }
        }
    } else if (lm_head == null) {
        return error.MissingLmHead;
    }

    return .{
        .embed_tokens = embed_tokens,
        .final_norm = final_norm,
        .lm_head = lm_head,
        .layers = layers,
        .bf16_compute = bf16_compute,
    };
}

fn required_tensor(tensors: *const tensor_map_t, name: []const u8) error{MissingTensor}![]f32 {
    return tensors.get(name) orelse error.MissingTensor;
}

fn optional_tensor(tensors: *const tensor_map_t, name: []const u8) ?[]const f32 {
    return tensors.get(name);
}

fn read_tensor(
    allocator: Allocator,
    io: std.Io,
    directory: std.Io.Dir,
    location: tensor_location_t,
    weight_read_count: ?*usize,
) ![]f32 {
    const tensor = location.info;
    const element_bits = tensor.dtype.bit_size();
    if (element_bits != 16 and element_bits != 32) return error.UnsupportedDtype;
    const element_bytes: usize = @intCast(element_bits / 8);
    const byte_length: usize = @intCast(tensor.data_end - tensor.data_begin);
    if (byte_length % element_bytes != 0) return error.InvalidTensorSize;
    const element_count = byte_length / element_bytes;
    const output = try allocator.alloc(f32, element_count);
    var file = try directory.openFile(io, location.shard_name, .{});
    defer file.close(io);
    const data_start = try std.math.add(u64, @intCast(safetensors.PrefixLen), location.header_len);
    const tensor_start = try std.math.add(u64, data_start, tensor.data_begin);
    var buffer: [ChunkBytes]u8 = undefined;
    var bytes_read: usize = 0;
    while (bytes_read < byte_length) {
        const count = @min(buffer.len, byte_length - bytes_read);
        if (weight_read_count) |counter| counter.* += 1;
        const read_offset = try std.math.add(u64, tensor_start, @intCast(bytes_read));
        const read_count = try file.readPositionalAll(io, buffer[0..count], read_offset);
        if (read_count != count) return error.UnexpectedEndOfFile;
        const chunk_start = bytes_read / element_bytes;
        const chunk_values = count / element_bytes;
        const source = buffer[0..count];
        const destination = output[chunk_start..][0..chunk_values];
        try decode_chunk(tensor.dtype, destination, source);
        bytes_read += count;
    }
    return output;
}

fn decode_chunk(dtype: safetensors.Dtype, destination: []f32, source: []const u8) !void {
    switch (dtype) {
        .F32 => {
            if (source.len != destination.len * 4) return error.InvalidTensorSize;
            for (destination, 0..) |*value, index| {
                const bits = std.mem.readInt(u32, source[index * 4 ..][0..4], .little);
                value.* = @bitCast(bits);
            }
        },
        .F16 => {
            if (source.len != destination.len * 2) return error.InvalidTensorSize;
            for (destination, 0..) |*value, index| {
                const bits = std.mem.readInt(u16, source[index * 2 ..][0..2], .little);
                value.* = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .Bf16 => {
            if (source.len != destination.len * 2) return error.InvalidTensorSize;
            bf16.widen_little_endian(destination, source);
        },
        else => return error.UnsupportedDtype,
    }
}

fn read_header(allocator: Allocator, io: std.Io, directory: std.Io.Dir, name: []const u8) !safetensors.header_t {
    var file = try directory.openFile(io, name, .{});
    defer file.close(io);
    const file_size = try file.length(io);
    var prefix: [safetensors.PrefixLen]u8 = undefined;
    try read_exact(io, file, &prefix, 0);
    const header_length = try safetensors.header_length(&prefix);
    const total_header_length = safetensors.PrefixLen + @as(usize, @intCast(header_length));
    const bytes = try allocator.alloc(u8, total_header_length);
    try read_exact(io, file, bytes, 0);
    return safetensors.parse_header(allocator, bytes, file_size);
}

fn read_exact(io: std.Io, file: std.Io.File, buffer: []u8, offset: u64) !void {
    const count = try file.readPositionalAll(io, buffer, offset);
    if (count != buffer.len) return error.UnexpectedEndOfFile;
}

fn contains_name(names: []const []const u8, name: []const u8) bool {
    for (names) |existing| if (std.mem.eql(u8, existing, name)) return true;
    return false;
}

fn within_tolerance(actual: f64, expected: f64, tolerance: f64) bool {
    if (!std.math.isFinite(actual) or !std.math.isFinite(expected) or !std.math.isFinite(tolerance)) return false;
    const difference = @abs(actual - expected);
    return std.math.isFinite(difference) and difference <= tolerance;
}

fn reference_provenance_problem(fields: std.json.ObjectMap) ?[]const u8 {
    const required_strings = [_]struct { key: []const u8, value: []const u8 }{
        .{ .key = "reference_method", .value = "Transformers 5.20.2 AutoModelForCausalLM / PyTorch 2.13.0+cpu, CPU BF16, eager, eval, inference_mode, use_cache=False" },
        .{ .key = "transformers_version", .value = "5.20.2" },
        .{ .key = "transformers_commit", .value = "18329f1065382e54db698e242cf98bd3f3e5a766" },
        .{ .key = "torch_version", .value = "2.13.0+cpu" },
        .{ .key = "torch_git_version", .value = "cf30153c4c131c8164ee7798e5022d810682e2cb" },
        .{ .key = "device", .value = "cpu" },
        .{ .key = "attention_backend", .value = "eager" },
        .{ .key = "onednn_version", .value = "3.12.0" },
        .{ .key = "onednn_git_version", .value = "80afa71049cd69a3df32adcccb623b12cd7baa22" },
        .{ .key = "cpu_architecture", .value = "x86_64" },
        .{ .key = "cpu_capability", .value = "AVX512" },
        .{ .key = "torch_build_cpu_capability", .value = "AVX512" },
    };
    for (required_strings) |field| {
        const value = fields.get(field.key) orelse return field.key;
        const actual = switch (value) {
            .string => |string| string,
            else => return field.key,
        };
        if (!std.mem.eql(u8, actual, field.value)) return field.key;
    }

    const required_integers = [_]struct { key: []const u8, value: i64 }{
        .{ .key = "reference_schema_version", .value = 2 },
        .{ .key = "torch_num_threads", .value = 1 },
    };
    for (required_integers) |field| {
        const value = fields.get(field.key) orelse return field.key;
        const actual = switch (value) {
            .integer => |integer| integer,
            else => return field.key,
        };
        if (actual != field.value) return field.key;
    }

    const onednn_enabled = fields.get("onednn_enabled") orelse return "onednn_enabled";
    if (onednn_enabled != .bool or !onednn_enabled.bool) return "onednn_enabled";
    return null;
}

test "golden comparison rejects nonfinite values" {
    const nan = std.math.nan(f64);
    const infinity = std.math.inf(f64);
    try std.testing.expect(within_tolerance(1.0, 1.0, 0.0));
    try std.testing.expect(within_tolerance(1.0, 1.00001, 0.00002));
    try std.testing.expect(!within_tolerance(nan, 1.0, 1.0));
    try std.testing.expect(!within_tolerance(1.0, nan, 1.0));
    try std.testing.expect(!within_tolerance(infinity, 1.0, infinity));
    try std.testing.expect(!within_tolerance(1.0, 1.0, nan));
    try std.testing.expect(!within_tolerance(std.math.floatMax(f64), -std.math.floatMax(f64), 1.0));
}

test "real reference provenance rejects missing or incorrect backend metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const valid_json =
        "{\"reference_schema_version\":2,\"reference_method\":\"Transformers 5.20.2 AutoModelForCausalLM / PyTorch 2.13.0+cpu, CPU BF16, eager, eval, inference_mode, use_cache=False\"," ++
        "\"transformers_version\":\"5.20.2\",\"transformers_commit\":\"18329f1065382e54db698e242cf98bd3f3e5a766\",\"torch_version\":\"2.13.0+cpu\",\"torch_git_version\":\"cf30153c4c131c8164ee7798e5022d810682e2cb\",\"device\":\"cpu\",\"attention_backend\":\"eager\",\"torch_num_threads\":1,\"onednn_enabled\":true,\"onednn_version\":\"3.12.0\",\"onednn_git_version\":\"80afa71049cd69a3df32adcccb623b12cd7baa22\",\"cpu_architecture\":\"x86_64\",\"cpu_capability\":\"AVX512\",\"torch_build_cpu_capability\":\"AVX512\"}";
    const valid = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), valid_json, .{});
    const valid_fields = switch (valid) {
        .object => |object| object,
        else => return error.InvalidFixture,
    };
    try std.testing.expect(reference_provenance_problem(valid_fields) == null);

    const missing_json =
        "{\"reference_schema_version\":2,\"reference_method\":\"Transformers 5.20.2 AutoModelForCausalLM / PyTorch 2.13.0+cpu, CPU BF16, eager, eval, inference_mode, use_cache=False\"," ++
        "\"transformers_version\":\"5.20.2\",\"transformers_commit\":\"18329f1065382e54db698e242cf98bd3f3e5a766\",\"torch_version\":\"2.13.0+cpu\",\"torch_git_version\":\"cf30153c4c131c8164ee7798e5022d810682e2cb\",\"device\":\"cpu\",\"torch_num_threads\":1,\"onednn_enabled\":true,\"onednn_version\":\"3.12.0\",\"onednn_git_version\":\"80afa71049cd69a3df32adcccb623b12cd7baa22\",\"cpu_architecture\":\"x86_64\",\"cpu_capability\":\"AVX512\",\"torch_build_cpu_capability\":\"AVX512\"}";
    const missing = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), missing_json, .{});
    const missing_fields = switch (missing) {
        .object => |object| object,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqualStrings("attention_backend", reference_provenance_problem(missing_fields).?);

    const wrong_json = try std.mem.replaceOwned(u8, arena.allocator(), valid_json, "\"attention_backend\":\"eager\"", "\"attention_backend\":\"sdpa\"");
    const wrong = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), wrong_json, .{});
    const wrong_fields = switch (wrong) {
        .object => |object| object,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqualStrings("attention_backend", reference_provenance_problem(wrong_fields).?);
}

test "manifest failure occurs before a tensor data read" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try write_text_file(temporary.dir, "config.json",
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":4,"hidden_size":2,"intermediate_size":3,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"rms_norm_eps":0.00001,"rope_theta":10000.0}
    );
    const header = "{\"unexpected.weight\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var file = try temporary.dir.createFile(std.testing.io, "model.safetensors", .{});
    defer file.close(std.testing.io);
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, header.len, .little);
    try file.writeStreamingAll(std.testing.io, &bytes);
    try file.writeStreamingAll(std.testing.io, header);
    try file.writeStreamingAll(std.testing.io, &[_]u8{ 0, 0, 0, 0 });

    var reads: usize = 0;
    try std.testing.expectError(
        error.InvalidManifest,
        load_checkpoint_impl(
            arena.allocator(),
            std.testing.io,
            try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{temporary.sub_path}),
            &reads,
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), reads);
}

test "load report retains actionable config diagnostics" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try write_text_file(temporary.dir, "config.json", "{\"architectures\":[\"UnsupportedForCausalLM\"],\"model_type\":\"unsupported\"}");

    const result = try load_checkpoint_report(
        arena.allocator(),
        std.testing.io,
        try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{temporary.sub_path}),
    );
    switch (result) {
        .loaded => return error.ExpectedRejection,
        .rejected => |rejection| {
            try std.testing.expectEqual(rejection_kind_t.InvalidConfig, rejection.kind);
            try std.testing.expect(rejection.problems.len > 0);
            try std.testing.expect(std.mem.eql(u8, rejection.problems[0].feature, "model_type"));
            try std.testing.expect(std.mem.indexOf(u8, rejection.problems[0].message, "unsupported") != null);
        },
    }
}

test "noncanonical padded layer index is rejected before any tensor data read" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try write_text_file(temporary.dir, "config.json",
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":4,"hidden_size":2,"intermediate_size":3,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"rms_norm_eps":0.00001,"rope_theta":10000.0}
    );
    try write_single_tensor_file(temporary.dir, "model.layers.00.input_layernorm.weight", 2);

    var reads: usize = 0;
    const result = try load_checkpoint_report_impl(
        arena.allocator(),
        std.testing.io,
        try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{temporary.sub_path}),
        &reads,
    );
    try std.testing.expectEqual(@as(usize, 0), reads);
    switch (result) {
        .loaded => return error.ExpectedRejection,
        .rejected => |rejection| {
            try std.testing.expectEqual(rejection_kind_t.InvalidManifest, rejection.kind);
            var found_noncanonical = false;
            for (rejection.problems) |problem| {
                if (std.mem.indexOf(u8, problem.message, "noncanonical layer index") != null and
                    std.mem.indexOf(u8, problem.message, "model.layers.00.input_layernorm.weight") != null)
                {
                    found_noncanonical = true;
                }
            }
            try std.testing.expect(found_noncanonical);
        },
    }
}

test "pickle-only checkpoint report gives safetensors conversion guidance" {
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try write_text_file(temporary.dir, "config.json",
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":4,"hidden_size":2,"intermediate_size":3,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"rms_norm_eps":0.00001,"rope_theta":10000.0}
    );
    try write_text_file(temporary.dir, "pytorch_model.bin", "not read by this loader test");

    const result = try load_checkpoint_report(
        arena.allocator(),
        std.testing.io,
        try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{temporary.sub_path}),
    );
    switch (result) {
        .loaded => return error.ExpectedRejection,
        .rejected => |rejection| {
            try std.testing.expectEqual(rejection_kind_t.UnsupportedPickleWeights, rejection.kind);
            try std.testing.expectEqual(@as(usize, 1), rejection.problems.len);
            try std.testing.expect(std.mem.indexOf(u8, rejection.problems[0].message, "convert the checkpoint to safetensors outside ncmorph") != null);
        },
    }
}

fn write_single_tensor_file(directory: std.Io.Dir, filename: []const u8, value_count: usize) !void {
    const header = try std.fmt.allocPrint(
        std.testing.allocator,
        "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}",
        .{ filename, value_count, value_count * 4 },
    );
    defer std.testing.allocator.free(header);
    var file = try directory.createFile(std.testing.io, "model.safetensors", .{});
    defer file.close(std.testing.io);
    var prefix: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix, header.len, .little);
    try file.writeStreamingAll(std.testing.io, &prefix);
    try file.writeStreamingAll(std.testing.io, header);
    const data = try std.testing.allocator.alloc(u8, value_count * 4);
    defer std.testing.allocator.free(data);
    @memset(data, 0);
    try file.writeStreamingAll(std.testing.io, data);
}

fn write_text_file(directory: std.Io.Dir, name: []const u8, contents: []const u8) !void {
    var file = try directory.createFile(std.testing.io, name, .{});
    defer file.close(std.testing.io);
    try file.writeStreamingAll(std.testing.io, contents);
}

test "decodes F32 F16 and BF16 little-endian values to F32" {
    const f32_bytes = [_]u8{ 0x00, 0x00, 0xC0, 0x3F };
    const f16_bytes = [_]u8{ 0x00, 0x3C, 0x00, 0xC0 };
    const bf16_bytes = [_]u8{ 0x80, 0x3F, 0x00, 0xC0 };
    var values: [1]f32 = undefined;
    try decode_chunk(.F32, &values, &f32_bytes);
    try std.testing.expectEqual(@as(f32, 1.5), values[0]);
    var half_values: [2]f32 = undefined;
    try decode_chunk(.F16, &half_values, &f16_bytes);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, &half_values);
    try decode_chunk(.Bf16, &half_values, &bf16_bytes);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, -2.0 }, &half_values);
}

test "tied output head is accepted only when its widened values match embeddings" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var tensors: tensor_map_t = .empty;
    defer tensors.deinit(arena.allocator());
    var embedding = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    var head = embedding;
    var norm = [_]f32{ 1, 1 };
    var filler = [_]f32{0};
    try tensors.put(arena.allocator(), "model.embed_tokens.weight", &embedding);
    try tensors.put(arena.allocator(), "model.norm.weight", &norm);
    try tensors.put(arena.allocator(), "lm_head.weight", &head);
    try tensors.put(arena.allocator(), "model.layers.0.input_layernorm.weight", &norm);
    try tensors.put(arena.allocator(), "model.layers.0.self_attn.q_proj.weight", &filler);
    try tensors.put(arena.allocator(), "model.layers.0.self_attn.k_proj.weight", &filler);
    try tensors.put(arena.allocator(), "model.layers.0.self_attn.v_proj.weight", &filler);
    try tensors.put(arena.allocator(), "model.layers.0.self_attn.o_proj.weight", &filler);
    try tensors.put(arena.allocator(), "model.layers.0.post_attention_layernorm.weight", &norm);
    try tensors.put(arena.allocator(), "model.layers.0.mlp.gate_proj.weight", &filler);
    try tensors.put(arena.allocator(), "model.layers.0.mlp.up_proj.weight", &filler);
    try tensors.put(arena.allocator(), "model.layers.0.mlp.down_proj.weight", &filler);

    const tied_spec = test_spec(true);
    _ = try build_weights(arena.allocator(), tied_spec, &tensors, false);
    head[0] = 0.9;
    try std.testing.expectError(error.TiedHeadMismatch, build_weights(arena.allocator(), tied_spec, &tensors, false));
}

test "tiny helper checkpoints load every safetensors weight bit exactly" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const case_names = [_][]const u8{
        "llama_default",
        "llama_llama3",
        "qwen2",
        "qwen3",
        "mistral",
        "qwen3_attention_bias",
    };
    for (case_names) |case_name| {
        const case_path = try std.fmt.allocPrint(arena.allocator(), "zig-out/golden/{s}", .{case_name});
        const reference_path = try std.fmt.allocPrint(arena.allocator(), "{s}/reference.json", .{case_path});
        const reference_bytes = std.Io.Dir.cwd().readFileAlloc(
            std.testing.io,
            reference_path,
            arena.allocator(),
            .limited(10 * 1024 * 1024),
        ) catch |err| return skip_without_golden(err, reference_path);
        const reference = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), reference_bytes, .{});
        const fields = switch (reference) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const expected_tensors = switch (fields.get("tensor_bits") orelse return error.InvalidFixture) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const loaded = try load_checkpoint(arena.allocator(), std.testing.io, case_path);
        try std.testing.expectEqual(expected_tensors.count(), loaded.tensors.len);
        var expected_iterator = expected_tensors.iterator();
        while (expected_iterator.next()) |entry| {
            const actual = find_loaded_tensor(loaded.tensors, entry.key_ptr.*) orelse return error.MissingTensor;
            const expected_bits = switch (entry.value_ptr.*) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(expected_bits.len, actual.len);
            for (expected_bits, actual) |expected, value| {
                const bits: u32 = switch (expected) {
                    .integer => |integer| @intCast(integer),
                    else => return error.InvalidFixture,
                };
                try std.testing.expectEqual(bits, @as(u32, @bitCast(value)));
            }
        }
    }
}

test "source decoder logits match all B9 Transformers golden cases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const case_names = [_][]const u8{ "llama_default", "llama_llama3", "qwen2", "qwen3", "mistral" };
    var maximum_error: f64 = 0;
    for (case_names) |case_name| {
        var case_maximum_error: f64 = 0;
        const case_path = try std.fmt.allocPrint(arena.allocator(), "zig-out/golden/{s}", .{case_name});
        const reference_path = try std.fmt.allocPrint(arena.allocator(), "{s}/reference.json", .{case_path});
        const reference_bytes = std.Io.Dir.cwd().readFileAlloc(
            std.testing.io,
            reference_path,
            arena.allocator(),
            .limited(10 * 1024 * 1024),
        ) catch |err| return skip_without_golden(err, reference_path);
        const reference = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), reference_bytes, .{});
        const fields = switch (reference) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const token_values = switch (fields.get("token_ids") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        const token_ids = try arena.allocator().alloc(usize, token_values.len);
        for (token_values, token_ids) |value, *token_id| {
            token_id.* = switch (value) {
                .integer => |integer| @intCast(integer),
                else => return error.InvalidFixture,
            };
        }
        const loaded = try load_checkpoint(arena.allocator(), std.testing.io, case_path);
        const vocab_size: usize = @intCast(loaded.spec.vocab_size);
        const actual = try arena.allocator().alloc(f32, token_ids.len * vocab_size);
        try decoder.forward(arena.allocator(), loaded.spec, loaded.weights, token_ids, actual);

        const expected_logits = switch (fields.get("logits") orelse return error.InvalidFixture) {
            .array => |batch| batch.items,
            else => return error.InvalidFixture,
        };
        if (expected_logits.len != 1) return error.InvalidFixture;
        const rows = switch (expected_logits[0]) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(token_ids.len, rows.len);
        for (rows, 0..) |row_value, token_index| {
            const row = switch (row_value) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(vocab_size, row.len);
            var row_max: f64 = 0;
            for (row) |value| row_max = @max(row_max, @abs(try json_number(value)));
            const tolerance = 1e-4 * @max(1.0, row_max);
            for (row, 0..) |value, vocab_index| {
                const expected = try json_number(value);
                const actual_value = actual[token_index * vocab_size + vocab_index];
                const difference = @abs(@as(f64, actual_value) - expected);
                case_maximum_error = @max(case_maximum_error, difference);
                maximum_error = @max(maximum_error, difference);
                try std.testing.expect(within_tolerance(actual_value, expected, tolerance));
            }
        }
        var bf16_weights = loaded.weights;
        bf16_weights.bf16_compute = true;
        const bf16_actual = try arena.allocator().alloc(f32, token_ids.len * vocab_size);
        try decoder.forward(arena.allocator(), loaded.spec, bf16_weights, token_ids, bf16_actual);
        const bf16_expected_logits = switch (fields.get("bf16_logits") orelse return error.InvalidFixture) {
            .array => |batch| batch.items,
            else => return error.InvalidFixture,
        };
        if (bf16_expected_logits.len != 1) return error.InvalidFixture;
        const bf16_rows = switch (bf16_expected_logits[0]) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(token_ids.len, bf16_rows.len);
        var bf16_max_error: f64 = 0;
        for (bf16_rows, 0..) |row_value, token_index| {
            const row = switch (row_value) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            var row_max: f64 = 0;
            for (row) |value| row_max = @max(row_max, @abs(try json_number(value)));
            const tolerance = 1e-4 * @max(1.0, row_max);
            for (row, 0..) |value, vocab_index| {
                const expected = try json_number(value);
                const difference = @abs(@as(f64, bf16_actual[token_index * vocab_size + vocab_index]) - expected);
                bf16_max_error = @max(bf16_max_error, difference);
                try std.testing.expect(within_tolerance(bf16_actual[token_index * vocab_size + vocab_index], expected, tolerance));
            }
        }
        std.debug.print("B9 {s} BF16 maximum absolute logit error: {d:.17}\n", .{ case_name, bf16_max_error });
        std.debug.print("B9 {s} maximum absolute logit error: {d:.17}\n", .{ case_name, case_maximum_error });
    }
    std.debug.print("B9 overall maximum absolute logit error: {d:.17}\n", .{maximum_error});
}

test "NCM source-mode hidden states match Transformers across tiny families" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const case_names = [_][]const u8{ "llama_default", "llama_llama3", "qwen2", "qwen3", "mistral" };
    var overall_maximum_error: f64 = 0;

    for (case_names) |case_name| {
        const allocator = arena.allocator();
        const case_path = try std.fmt.allocPrint(allocator, "zig-out/golden/{s}", .{case_name});
        const reference_path = try std.fmt.allocPrint(allocator, "{s}/reference.json", .{case_path});
        const reference_bytes = std.Io.Dir.cwd().readFileAlloc(
            std.testing.io,
            reference_path,
            allocator,
            .limited(10 * 1024 * 1024),
        ) catch |err| return skip_without_golden(err, reference_path);
        const reference = try std.json.parseFromSliceLeaky(std.json.Value, allocator, reference_bytes, .{});
        const fields = switch (reference) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const token_values = switch (fields.get("token_ids") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        const token_ids = try allocator.alloc(usize, token_values.len);
        for (token_values, token_ids) |value, *token_id| {
            token_id.* = switch (value) {
                .integer => |integer| @intCast(integer),
                else => return error.InvalidFixture,
            };
        }

        const loaded = try load_checkpoint(allocator, std.testing.io, case_path);
        const hidden_size: usize = @intCast(loaded.spec.hidden_size);
        const layer_count: usize = @intCast(loaded.spec.num_hidden_layers);
        const row_size = token_ids.len * hidden_size;
        const actual = try allocator.alloc(f32, (layer_count + 1) * row_size);
        try ncm_body.forward_source_mode(allocator, loaded.spec, loaded.weights, token_ids, actual);

        const states = switch (fields.get("hidden_states") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(layer_count + 1, states.len);
        var case_maximum_error: f64 = 0;
        for (states, 0..) |state_value, state_index| {
            const batches = switch (state_value) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(@as(usize, 1), batches.len);
            const token_rows = switch (batches[0]) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(token_ids.len, token_rows.len);
            for (token_rows, 0..) |row_value, token_index| {
                const row = switch (row_value) {
                    .array => |array| array.items,
                    else => return error.InvalidFixture,
                };
                try std.testing.expectEqual(hidden_size, row.len);
                var row_maximum: f64 = 0;
                for (row) |value| row_maximum = @max(row_maximum, @abs(try json_number(value)));
                const tolerance = 1e-4 * @max(1.0, row_maximum);
                for (row, 0..) |value, hidden_index| {
                    const expected = try json_number(value);
                    const actual_value = actual[state_index * row_size + token_index * hidden_size + hidden_index];
                    const difference = @abs(@as(f64, actual_value) - expected);
                    case_maximum_error = @max(case_maximum_error, difference);
                    overall_maximum_error = @max(overall_maximum_error, difference);
                    try std.testing.expect(within_tolerance(actual_value, expected, tolerance));
                }
            }
        }
        std.debug.print("N6 {s} maximum absolute hidden-state difference: {d:.17}\n", .{ case_name, case_maximum_error });
    }
    std.debug.print("N6 overall maximum absolute hidden-state difference: {d:.17}\n", .{overall_maximum_error});
}

test "N8 NCM source mode matches M0 decoder hidden states" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const case_names = [_][]const u8{ "llama_default", "llama_llama3", "qwen2", "qwen3", "mistral" };
    var overall_maximum_error: f64 = 0;

    for (case_names) |case_name| {
        const allocator = arena.allocator();
        const case_path = try std.fmt.allocPrint(allocator, "zig-out/golden/{s}", .{case_name});
        const reference_path = try std.fmt.allocPrint(allocator, "{s}/reference.json", .{case_path});
        const reference_bytes = std.Io.Dir.cwd().readFileAlloc(
            std.testing.io,
            reference_path,
            allocator,
            .limited(10 * 1024 * 1024),
        ) catch |err| return skip_without_golden(err, reference_path);
        const reference = try std.json.parseFromSliceLeaky(std.json.Value, allocator, reference_bytes, .{});
        const fields = switch (reference) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const token_values = switch (fields.get("token_ids") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        const token_ids = try allocator.alloc(usize, token_values.len);
        for (token_values, token_ids) |value, *token_id| token_id.* = @intCast(try json_integer(value));

        const loaded = try load_checkpoint(allocator, std.testing.io, case_path);
        const hidden_size: usize = @intCast(loaded.spec.hidden_size);
        const intermediate_size: usize = @intCast(loaded.spec.intermediate_size);
        const layer_count: usize = @intCast(loaded.spec.num_hidden_layers);
        const query_width: usize = @intCast(loaded.spec.num_attention_heads * loaded.spec.head_dim);
        const row_size = token_ids.len * hidden_size;
        const logits = try allocator.alloc(f32, token_ids.len * @as(usize, @intCast(loaded.spec.vocab_size)));
        const trace = try allocate_decoder_trace(allocator, token_ids.len, hidden_size, intermediate_size, query_width, layer_count);
        try decoder.forward_with_trace(allocator, loaded.spec, loaded.weights, token_ids, logits, trace);

        const final_block_offset = (layer_count - 1) * row_size;
        const m0_final = try allocator.alloc(f32, row_size);
        ncm_norm.rms_norm_rows(trace.block_output[final_block_offset..][0..row_size], loaded.weights.final_norm, m0_final, token_ids.len, hidden_size, @floatCast(loaded.spec.rms_norm_eps));
        const m0_states = try allocator.alloc(f32, (layer_count + 1) * row_size);
        @memcpy(m0_states[0..row_size], trace.block_input);
        for (1..layer_count) |state_index| {
            const source_offset = (state_index - 1) * row_size;
            @memcpy(m0_states[state_index * row_size ..][0..row_size], trace.block_output[source_offset..][0..row_size]);
        }
        @memcpy(m0_states[layer_count * row_size ..][0..row_size], m0_final);

        const q_width = @as(usize, @intCast(loaded.spec.num_attention_heads)) * @as(usize, @intCast(loaded.spec.head_dim));
        const kv_width = @as(usize, @intCast(loaded.spec.num_key_value_heads)) * @as(usize, @intCast(loaded.spec.head_dim));
        const fused_width = q_width + 2 * kv_width;
        const ncm_layers = try allocator.alloc(ncm_model.layer_weights_t, layer_count);
        for (loaded.weights.layers, ncm_layers) |layer, *ncm_layer| {
            const fused_weight = try allocator.alloc(f32, fused_width * hidden_size);
            var weight_offset: usize = 0;
            const attention_parts = [_][]const f32{ layer.attention.query_weight, layer.attention.key_weight, layer.attention.value_weight };
            for (attention_parts) |part| {
                @memcpy(fused_weight[weight_offset..][0..part.len], part);
                weight_offset += part.len;
            }
            const has_qkv_bias = layer.attention.query_bias != null or layer.attention.key_bias != null or layer.attention.value_bias != null;
            const fused_bias: ?[]f32 = if (has_qkv_bias) try allocator.alloc(f32, fused_width) else null;
            if (fused_bias) |bias| {
                set_bias_part(bias, 0, q_width, layer.attention.query_bias);
                set_bias_part(bias, q_width, kv_width, layer.attention.key_bias);
                set_bias_part(bias, q_width + kv_width, kv_width, layer.attention.value_bias);
            }
            ncm_layer.* = .{
                .fused_qkv_weight = fused_weight,
                .fused_qkv_bias = fused_bias,
                .output_weight = layer.attention.output_weight,
                .output_bias = layer.attention.output_bias,
                .query_norm_weight = layer.attention.query_norm_weight,
                .key_norm_weight = layer.attention.key_norm_weight,
                .ln_1_weight = layer.input_norm,
                .gate_weight = layer.gate_weight,
                .gate_bias = layer.gate_bias,
                .up_weight = layer.up_weight,
                .up_bias = layer.up_bias,
                .down_weight = layer.down_weight,
                .down_bias = layer.down_bias,
                .ln_2_weight = layer.post_attention_norm,
            };
        }
        const identity = try allocator.alloc(f32, hidden_size * hidden_size);
        @memset(identity, 0);
        for (0..hidden_size) |index| identity[index * hidden_size + index] = 1;
        const zero_bias = try allocator.alloc(f32, hidden_size);
        @memset(zero_bias, 0);
        const zeros = try allocator.alloc(f32, layer_count * hidden_size);
        @memset(zeros, 0);
        const token_inputs = try allocator.alloc(f32, row_size);
        for (token_ids, 0..) |token_id, index| @memcpy(token_inputs[index * hidden_size ..][0..hidden_size], loaded.weights.embed_tokens[token_id * hidden_size ..][0..hidden_size]);
        const rope_type: ncm_model.rope_type_t = switch (loaded.spec.rope_scaling) {
            .Default => .Default,
            .Llama3 => .Llama3,
        };
        const ncm_output = try allocator.alloc(f32, row_size);
        const ncm_states = try allocator.alloc(f32, (layer_count + 1) * row_size);
        const source_config: ncm_model.config_t = .{
            .hidden_size = hidden_size,
            .vq_width = hidden_size,
            .num_attention_heads = @intCast(loaded.spec.num_attention_heads),
            .num_key_value_heads = @intCast(loaded.spec.num_key_value_heads),
            .head_dim = @intCast(loaded.spec.head_dim),
            .intermediate_size = intermediate_size,
            .rms_norm_epsilon = @floatCast(loaded.spec.rms_norm_eps),
            .rope_type = rope_type,
            .rope_theta = @floatCast(loaded.spec.rope_theta),
            .rope_factor = if (loaded.spec.rope_scaling == .Llama3) @floatCast(loaded.spec.rope_scaling.Llama3.factor) else 1,
            .rope_low_freq_factor = if (loaded.spec.rope_scaling == .Llama3) @floatCast(loaded.spec.rope_scaling.Llama3.low_freq_factor) else 1,
            .rope_high_freq_factor = if (loaded.spec.rope_scaling == .Llama3) @floatCast(loaded.spec.rope_scaling.Llama3.high_freq_factor) else 4,
            .rope_original_max_position_embeddings = if (loaded.spec.rope_scaling == .Llama3) @intCast(loaded.spec.rope_scaling.Llama3.original_max_position_embeddings) else 8192,
            .level_lengths = &.{token_ids.len},
            .prefix_length = 0,
            .query_level0 = false,
            .use_level_encoding = false,
            .source_causal = true,
            .sliding_window = if (loaded.spec.sliding_window) |window| @intCast(window) else null,
        };
        try ncm_model.forward_with_trace(allocator, source_config, .{
            .prefix_token_embedding = &.{},
            .prefix_vocab_size = 0,
            .input_projection = identity,
            .input_projection_bias = zero_bias,
            .level_embedding = zeros,
            .l0_queries = &.{},
            .final_norm_weight = loaded.weights.final_norm,
            .layers = ncm_layers,
        }, &.{}, &.{}, token_inputs, ncm_output, ncm_states);

        var case_maximum_error: f64 = 0;
        for (ncm_states, m0_states) |actual, expected| {
            const difference = @abs(@as(f64, actual) - expected);
            case_maximum_error = @max(case_maximum_error, difference);
            overall_maximum_error = @max(overall_maximum_error, difference);
            try std.testing.expect(within_tolerance(actual, expected, 1e-4));
        }
        std.debug.print("N8 {s} maximum absolute hidden-state difference: {d:.17}\n", .{ case_name, case_maximum_error });
    }
    std.debug.print("N8 overall maximum absolute hidden-state difference: {d:.17}\n", .{overall_maximum_error});
}

fn set_bias_part(destination: []f32, offset: usize, width: usize, source: ?[]const f32) void {
    if (source) |values| @memcpy(destination[offset..][0..width], values) else @memset(destination[offset..][0..width], 0);
}

test "NCM masks and level positions match pinned NCM tiny models" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const reference_path = "zig-out/golden/ncm_masks_positions.json";
    const reference_bytes = std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        reference_path,
        allocator,
        .limited(2 * 1024 * 1024),
    ) catch |err| return skip_without_golden(err, reference_path);
    const reference = try std.json.parseFromSliceLeaky(std.json.Value, allocator, reference_bytes, .{});
    const fields = switch (reference) {
        .object => |object| object,
        else => return error.InvalidFixture,
    };
    const commit = switch (fields.get("ncm_commit") orelse return error.InvalidFixture) {
        .string => |value| value,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqualStrings("c70274273f489abc73832b2b98584a8483b6b203", commit);
    const cases = switch (fields.get("cases") orelse return error.InvalidFixture) {
        .array => |array| array.items,
        else => return error.InvalidFixture,
    };

    for (cases) |case_value| {
        const case = switch (case_value) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const prefix_length: usize = @intCast(try json_integer(case.get("prefix_length") orelse return error.InvalidFixture));
        const level_values = switch (case.get("level_lengths") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        const level_lengths = try allocator.alloc(usize, level_values.len);
        var target_length: usize = 0;
        for (level_values, level_lengths) |value, *level_length| {
            level_length.* = @intCast(try json_integer(value));
            target_length += level_length.*;
        }
        const sequence_length = prefix_length + target_length;
        const positions = try allocator.alloc(usize, sequence_length);
        try ncm_position.fill_level_positions(prefix_length, level_lengths, positions);
        const head_dim: usize = 4;
        const cosine = try allocator.alloc(f32, sequence_length * head_dim);
        const sine = try allocator.alloc(f32, sequence_length * head_dim);
        try ncm_position.default_tables_at_positions(allocator, head_dim, 10_000, positions, cosine, sine);
        try compare_rope_matrix(case.get("rope_cos") orelse return error.InvalidFixture, sequence_length, head_dim, cosine);
        try compare_rope_matrix(case.get("rope_sin") orelse return error.InvalidFixture, sequence_length, head_dim, sine);
        const expected_positions = switch (case.get("positions") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(sequence_length, expected_positions.len);
        for (expected_positions, positions) |expected, actual| try std.testing.expectEqual(
            try json_integer(expected),
            @as(i64, @intCast(actual)),
        );

        const per_level = switch (case.get("per_level_positions") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(level_lengths.len, per_level.len);
        for (level_lengths, per_level) |level_length, level_positions_value| {
            const level_positions = switch (level_positions_value) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(prefix_length + level_length, level_positions.len);
            for (0..prefix_length) |position| try std.testing.expectEqual(
                @as(i64, @intCast(position)),
                try json_integer(level_positions[position]),
            );
            for (0..level_length) |position| try std.testing.expectEqual(
                @as(i64, @intCast(position)),
                try json_integer(level_positions[prefix_length + position]),
            );
        }

        const expected_cos_generators = switch (case.get("rope_cos_gen") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        const expected_sin_generators = switch (case.get("rope_sin_gen") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(level_lengths.len, expected_cos_generators.len);
        try std.testing.expectEqual(level_lengths.len, expected_sin_generators.len);
        for (level_lengths, 0..) |level_length, level_index| {
            const generator_positions = try allocator.alloc(usize, prefix_length + level_length);
            for (0..prefix_length) |position| generator_positions[position] = position;
            for (0..level_length) |position| generator_positions[prefix_length + position] = position;
            const generator_cosine = try allocator.alloc(f32, generator_positions.len * head_dim);
            const generator_sine = try allocator.alloc(f32, generator_positions.len * head_dim);
            try ncm_position.default_tables_at_positions(allocator, head_dim, 10_000, generator_positions, generator_cosine, generator_sine);
            try compare_rope_matrix(expected_cos_generators[level_index], generator_positions.len, head_dim, generator_cosine);
            try compare_rope_matrix(expected_sin_generators[level_index], generator_positions.len, head_dim, generator_sine);
        }

        const convolution = switch (case.get("post_upsample_conv") orelse return error.InvalidFixture) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        const convolution_rows: usize = @intCast(try json_integer(convolution.get("row_count") orelse return error.InvalidFixture));
        const convolution_width: usize = @intCast(try json_integer(convolution.get("width") orelse return error.InvalidFixture));
        const convolution_weight = try json_f32_array(allocator, convolution.get("weight") orelse return error.InvalidFixture);
        const convolution_bias = try json_f32_array(allocator, convolution.get("bias") orelse return error.InvalidFixture);
        const convolution_input = try json_f32_array(allocator, convolution.get("input") orelse return error.InvalidFixture);
        const convolution_expected = try json_f32_array(allocator, convolution.get("output") orelse return error.InvalidFixture);
        try std.testing.expectEqual(level_lengths[0], convolution_rows);
        const convolution_actual = try allocator.dupe(f32, convolution_input);
        try ncm_model.apply_post_upsample_conv(convolution_actual, .{ .weight = convolution_weight, .bias = convolution_bias }, convolution_rows, convolution_width, allocator);
        try std.testing.expectEqualSlices(f32, convolution_expected, convolution_actual);

        const mask = try allocator.alloc(bool, sequence_length * sequence_length);
        try ncm_mask.block_diagonal_mask(prefix_length, level_lengths, &.{}, mask);
        try compare_json_mask(case.get("mask") orelse return error.InvalidFixture, sequence_length, mask);
        const prefix_pad_mask = try allocator.alloc(bool, prefix_length);
        @memset(prefix_pad_mask, false);
        if (prefix_length != 0) prefix_pad_mask[prefix_length - 1] = true;
        try ncm_mask.block_diagonal_mask(prefix_length, level_lengths, prefix_pad_mask, mask);
        try compare_json_mask(case.get("padded_mask") orelse return error.InvalidFixture, sequence_length, mask);
    }
}

fn json_f32_array(allocator: Allocator, value: std.json.Value) ![]f32 {
    const items = switch (value) {
        .array => |array| array.items,
        else => return error.InvalidFixture,
    };
    const output = try allocator.alloc(f32, items.len);
    for (items, output) |item, *number| number.* = @floatCast(try json_number(item));
    return output;
}

fn compare_rope_matrix(value: std.json.Value, row_count: usize, width: usize, actual: []const f32) !void {
    const rows = switch (value) {
        .array => |array| array.items,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqual(row_count, rows.len);
    try std.testing.expectEqual(row_count * width, actual.len);
    for (rows, 0..) |row_value, row_index| {
        const row = switch (row_value) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(width, row.len);
        for (row, 0..) |expected_value, column| {
            const expected = try json_number(expected_value);
            try std.testing.expect(within_tolerance(actual[row_index * width + column], expected, 1e-6));
        }
    }
}

test "C2-C5 real checkpoint logits match pinned Transformers" {
    const env_ptr = std.c.getenv("NCMORPH_CHECKPOINT_DIR") orelse {
        std.debug.print("SKIP real-checkpoint acceptance: set NCMORPH_CHECKPOINT_DIR to a directory with the C2-C5 fixtures\n", .{});
        return error.SkipZigTest;
    };
    const checkpoint_root = std.mem.sliceTo(env_ptr, 0);
    if (checkpoint_root.len == 0) {
        std.debug.print("SKIP real-checkpoint acceptance: NCMORPH_CHECKPOINT_DIR is empty\n", .{});
        return error.SkipZigTest;
    }

    const case_names = [_][]const u8{ "smollm2-135m", "qwen2.5-0.5b", "qwen3-0.6b", "llama-3.2-1b" };
    const expected_model_types = [_][]const u8{ "llama", "qwen2", "qwen3", "llama" };
    const expected_token_ids = [_]usize{ 1, 4, 7, 11 };
    const selected_case_ptr = std.c.getenv("NCMORPH_CHECKPOINT_CASE");
    const selected_case = if (selected_case_ptr) |value| std.mem.sliceTo(value, 0) else null;
    var overall_maximum_error: f64 = 0;
    var selected_case_found = selected_case == null;
    for (case_names, expected_model_types) |case_name, expected_model_type| {
        if (selected_case) |name| {
            if (!std.mem.eql(u8, name, case_name)) continue;
            selected_case_found = true;
        }
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const allocator = arena.allocator();
        const checkpoint_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ checkpoint_root, case_name });
        const reference_path = try std.fmt.allocPrint(allocator, "{s}/ncmorph_reference.json", .{checkpoint_path});
        const reference_bytes = try std.Io.Dir.cwd().readFileAlloc(
            std.testing.io,
            reference_path,
            allocator,
            .limited(64 << 20),
        );
        const reference = try std.json.parseFromSliceLeaky(std.json.Value, allocator, reference_bytes, .{});
        const fields = switch (reference) {
            .object => |object| object,
            else => return error.InvalidFixture,
        };
        if (reference_provenance_problem(fields)) |problem_field| {
            std.debug.print(
                "C gate {s} reference is missing or has mismatched provenance at {s}; regenerate with `/tmp/ncmorph-b0-venv/bin/python tools/golden_helper.py --real-checkpoint-dir {s}`\n",
                .{ case_name, problem_field, checkpoint_path },
            );
            return error.InvalidFixture;
        }
        const transformers_version = switch (fields.get("transformers_version") orelse return error.InvalidFixture) {
            .string => |value| value,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqualStrings("5.20.2", transformers_version);
        const model_type = switch (fields.get("model_type") orelse return error.InvalidFixture) {
            .string => |value| value,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqualStrings(expected_model_type, model_type);
        const reference_dtype = switch (fields.get("weight_dtype") orelse return error.InvalidFixture) {
            .string => |value| value,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqualStrings("torch.bfloat16", reference_dtype);

        const token_values = switch (fields.get("token_ids") orelse return error.InvalidFixture) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(expected_token_ids.len, token_values.len);
        const token_ids = try allocator.alloc(usize, expected_token_ids.len);
        for (token_values, expected_token_ids, token_ids) |value, expected, *token_id| {
            token_id.* = switch (value) {
                .integer => |integer| @intCast(integer),
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(expected, token_id.*);
        }

        const report = try load_checkpoint_report(allocator, std.testing.io, checkpoint_path);
        const loaded = switch (report) {
            .loaded => |model| model,
            .rejected => |rejection| {
                for (rejection.problems) |problem| {
                    std.debug.print("C gate {s} rejected {s}: {s}\n", .{ case_name, problem.feature, problem.message });
                }
                return error.CheckpointRejected;
            },
        };
        const vocab_size: usize = @intCast(loaded.spec.vocab_size);
        const actual = try allocator.alloc(f32, token_ids.len * vocab_size);
        const hidden_size: usize = @intCast(loaded.spec.hidden_size);
        const layer_count: usize = @intCast(loaded.spec.num_hidden_layers);
        const intermediate_size: usize = @intCast(loaded.spec.intermediate_size);
        const query_width: usize = @intCast(loaded.spec.num_attention_heads * loaded.spec.head_dim);
        const hidden_trace = try allocator.alloc(f32, layer_count * token_ids.len * hidden_size);
        const intermediate_trace = try allocator.alloc(f32, layer_count * token_ids.len * intermediate_size);
        const trace: decoder.decoder_trace_t = .{
            .token_count = token_ids.len,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .query_width = query_width,
            .layer_count = layer_count,
            .block_input = try allocator.alloc(f32, token_ids.len * hidden_size),
            .attention_context = try allocator.alloc(f32, layer_count * token_ids.len * query_width),
            .input_norm = try allocator.alloc(f32, hidden_trace.len),
            .attention_output = try allocator.alloc(f32, hidden_trace.len),
            .attention_residual = try allocator.alloc(f32, hidden_trace.len),
            .post_attention_norm = try allocator.alloc(f32, hidden_trace.len),
            .gate_projection = try allocator.alloc(f32, intermediate_trace.len),
            .up_projection = try allocator.alloc(f32, intermediate_trace.len),
            .gate_up_product = try allocator.alloc(f32, intermediate_trace.len),
            .mlp_output = try allocator.alloc(f32, hidden_trace.len),
            .block_output = hidden_trace,
        };
        try decoder.forward_with_trace(allocator, loaded.spec, loaded.weights, token_ids, actual, trace);

        if (fields.get("diagnostics")) |diagnostic_value| {
            const diagnostics = switch (diagnostic_value) {
                .object => |object| object,
                else => return error.InvalidFixture,
            };
            const stage_names = [_][]const u8{
                "block_input",
                "input_norm",
                "attention_context",
                "attention_output",
                "attention_residual",
                "post_attention_norm",
                "gate_projection",
                "up_projection",
                "gate_up_product",
                "mlp_output",
                "block_output",
            };
            for (0..@min(layer_count, 12)) |layer_index| {
                for (stage_names) |stage_name| {
                    if (std.mem.eql(u8, stage_name, "attention_context") and layer_index > 0) {
                        const context_key = try std.fmt.allocPrint(allocator, "layer_{d}_{s}", .{ layer_index, stage_name });
                        if (diagnostics.get(context_key) == null) continue;
                    }
                    const reference_key = try std.fmt.allocPrint(allocator, "layer_{d}_{s}", .{ layer_index, stage_name });
                    const expected = diagnostics.get(reference_key) orelse return error.InvalidFixture;
                    const actual_stage = trace_stage(trace, stage_name, layer_index) orelse return error.InvalidFixture;
                    const width = if (std.mem.endsWith(u8, stage_name, "projection") or std.mem.eql(u8, stage_name, "gate_up_product")) intermediate_size else if (std.mem.eql(u8, stage_name, "attention_context")) query_width else hidden_size;
                    const error_value = try trace_maximum_error(case_name, stage_name, actual_stage, expected, token_ids.len, width);
                    std.debug.print("C gate {s} layer {d} {s} maximum absolute difference: {d:.12}\n", .{
                        case_name,
                        layer_index,
                        stage_name,
                        error_value,
                    });
                }
            }
        }

        if (fields.get("hidden_states")) |hidden_values| {
            const states = switch (hidden_values) {
                .array => |array| array.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(layer_count + 1, states.len);
            // Transformers ties hidden_states[-1] to last_hidden_state after the final RMSNorm.
            // The raw last block output therefore has no matching pre-norm reference state.
            for (0..(layer_count -| 1)) |layer_index| {
                const state = switch (states[layer_index + 1]) {
                    .array => |batch| batch.items,
                    else => return error.InvalidFixture,
                };
                try std.testing.expectEqual(@as(usize, 1), state.len);
                const token_rows = switch (state[0]) {
                    .array => |rows| rows.items,
                    else => return error.InvalidFixture,
                };
                try std.testing.expectEqual(token_ids.len, token_rows.len);
                var layer_maximum_error: f64 = 0;
                for (token_rows, 0..) |row_value, token_index| {
                    const row = switch (row_value) {
                        .array => |values| values.items,
                        else => return error.InvalidFixture,
                    };
                    try std.testing.expectEqual(hidden_size, row.len);
                    const offset = layer_index * token_ids.len * hidden_size + token_index * hidden_size;
                    for (row, 0..) |value, feature_index| {
                        const expected = try json_number(value);
                        const difference = @abs(@as(f64, trace.block_output[offset + feature_index]) - expected);
                        layer_maximum_error = @max(layer_maximum_error, difference);
                    }
                }
                std.debug.print("C gate {s} hidden state after layer {d} maximum absolute difference: {d:.12}\n", .{
                    case_name,
                    layer_index,
                    layer_maximum_error,
                });
            }
        }

        const expected_logits = switch (fields.get("logits") orelse return error.InvalidFixture) {
            .array => |batch| batch.items,
            else => return error.InvalidFixture,
        };
        if (expected_logits.len != 1) return error.InvalidFixture;
        const expected_rows = switch (expected_logits[0]) {
            .array => |rows| rows.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(token_ids.len, expected_rows.len);
        var case_maximum_error: f64 = 0;
        var case_maximum_tolerance: f64 = 0;
        var worst_token_index: usize = 0;
        var worst_vocab_index: usize = 0;
        var exceeded_bound = false;
        var case_has_nonfinite = false;
        for (expected_rows, 0..) |row_value, token_index| {
            const expected_row = switch (row_value) {
                .array => |row| row.items,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(vocab_size, expected_row.len);
            var row_maximum: f64 = 0;
            for (expected_row) |value| row_maximum = @max(row_maximum, @abs(try json_number(value)));
            const tolerance = 1e-4 * @max(1.0, row_maximum);
            case_maximum_tolerance = @max(case_maximum_tolerance, tolerance);
            for (expected_row, 0..) |value, vocab_index| {
                const expected = try json_number(value);
                const actual_value = @as(f64, actual[token_index * vocab_size + vocab_index]);
                const difference = @abs(actual_value - expected);
                if (!within_tolerance(actual_value, expected, tolerance)) exceeded_bound = true;
                if (!std.math.isFinite(difference)) {
                    case_has_nonfinite = true;
                    continue;
                }
                if (difference > case_maximum_error) {
                    worst_token_index = token_index;
                    worst_vocab_index = vocab_index;
                }
                case_maximum_error = @max(case_maximum_error, difference);
                overall_maximum_error = @max(overall_maximum_error, difference);
            }
        }
        std.debug.print("C gate {s} maximum absolute logit difference: {d:.12} (tolerance max {d:.12}, token {d}, vocab {d}, nonfinite {any})\n", .{
            case_name,
            case_maximum_error,
            case_maximum_tolerance,
            worst_token_index,
            worst_vocab_index,
            case_has_nonfinite,
        });
        if (exceeded_bound) return error.D53Exceeded;
    }
    if (!selected_case_found) return error.UnknownCheckpointCase;
    std.debug.print("C2-C5 overall maximum absolute logit difference: {d:.12}\n", .{overall_maximum_error});
}

fn find_loaded_tensor(tensors: []const loaded_tensor_t, name: []const u8) ?[]const f32 {
    for (tensors) |tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor.values;
    return null;
}

fn json_number(value: std.json.Value) !f64 {
    return switch (value) {
        .float => |number| number,
        .integer => |number| @floatFromInt(number),
        else => error.InvalidFixture,
    };
}

fn json_integer(value: std.json.Value) !i64 {
    return switch (value) {
        .integer => |number| number,
        else => error.InvalidFixture,
    };
}

fn allocate_decoder_trace(
    allocator: Allocator,
    token_count: usize,
    hidden_size: usize,
    intermediate_size: usize,
    query_width: usize,
    layer_count: usize,
) !decoder.decoder_trace_t {
    const hidden_length = layer_count * token_count * hidden_size;
    const intermediate_length = layer_count * token_count * intermediate_size;
    const context_length = layer_count * token_count * query_width;
    return .{
        .token_count = token_count,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .query_width = query_width,
        .layer_count = layer_count,
        .block_input = try allocator.alloc(f32, token_count * hidden_size),
        .attention_context = try allocator.alloc(f32, context_length),
        .input_norm = try allocator.alloc(f32, hidden_length),
        .attention_output = try allocator.alloc(f32, hidden_length),
        .attention_residual = try allocator.alloc(f32, hidden_length),
        .post_attention_norm = try allocator.alloc(f32, hidden_length),
        .gate_projection = try allocator.alloc(f32, intermediate_length),
        .up_projection = try allocator.alloc(f32, intermediate_length),
        .gate_up_product = try allocator.alloc(f32, intermediate_length),
        .mlp_output = try allocator.alloc(f32, hidden_length),
        .block_output = try allocator.alloc(f32, hidden_length),
    };
}

fn compare_json_mask(expected_value: std.json.Value, sequence_length: usize, actual: []const bool) !void {
    const rows = switch (expected_value) {
        .array => |array| array.items,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqual(sequence_length, rows.len);
    for (rows, 0..) |row_value, query_index| {
        const row = switch (row_value) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(sequence_length, row.len);
        for (row, 0..) |value, key_index| {
            const expected = switch (value) {
                .bool => |boolean| boolean,
                else => return error.InvalidFixture,
            };
            try std.testing.expectEqual(expected, actual[query_index * sequence_length + key_index]);
        }
    }
}

fn trace_stage(trace: decoder.decoder_trace_t, name: []const u8, layer_index: usize) ?[]const f32 {
    if (std.mem.eql(u8, name, "block_input")) {
        if (layer_index == 0) return trace.block_input;
        const previous_offset = (layer_index - 1) * trace.token_count * trace.hidden_size;
        return trace.block_output[previous_offset..][0 .. trace.token_count * trace.hidden_size];
    }
    if (std.mem.eql(u8, name, "attention_context")) {
        const offset = layer_index * trace.token_count * trace.query_width;
        return trace.attention_context[offset..][0 .. trace.token_count * trace.query_width];
    }
    const hidden_offset = layer_index * trace.token_count * trace.hidden_size;
    const intermediate_offset = layer_index * trace.token_count * trace.intermediate_size;
    if (std.mem.eql(u8, name, "input_norm")) return trace.input_norm[hidden_offset..][0 .. trace.token_count * trace.hidden_size];
    if (std.mem.eql(u8, name, "attention_output")) return trace.attention_output[hidden_offset..][0 .. trace.token_count * trace.hidden_size];
    if (std.mem.eql(u8, name, "attention_residual")) return trace.attention_residual[hidden_offset..][0 .. trace.token_count * trace.hidden_size];
    if (std.mem.eql(u8, name, "post_attention_norm")) return trace.post_attention_norm[hidden_offset..][0 .. trace.token_count * trace.hidden_size];
    if (std.mem.eql(u8, name, "gate_projection")) return trace.gate_projection[intermediate_offset..][0 .. trace.token_count * trace.intermediate_size];
    if (std.mem.eql(u8, name, "up_projection")) return trace.up_projection[intermediate_offset..][0 .. trace.token_count * trace.intermediate_size];
    if (std.mem.eql(u8, name, "gate_up_product")) return trace.gate_up_product[intermediate_offset..][0 .. trace.token_count * trace.intermediate_size];
    if (std.mem.eql(u8, name, "mlp_output")) return trace.mlp_output[hidden_offset..][0 .. trace.token_count * trace.hidden_size];
    if (std.mem.eql(u8, name, "block_output")) return trace.block_output[hidden_offset..][0 .. trace.token_count * trace.hidden_size];
    return null;
}

fn trace_maximum_error(case_name: []const u8, stage_name: []const u8, actual: []const f32, expected: std.json.Value, token_count: usize, width: usize) !f64 {
    const batch = switch (expected) {
        .array => |array| array.items,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqual(@as(usize, 1), batch.len);
    const rows = switch (batch[0]) {
        .array => |array| array.items,
        else => return error.InvalidFixture,
    };
    try std.testing.expectEqual(token_count, rows.len);
    try std.testing.expectEqual(token_count * width, actual.len);
    var maximum: f64 = 0;
    for (rows, 0..) |row_value, row_index| {
        const row = switch (row_value) {
            .array => |array| array.items,
            else => return error.InvalidFixture,
        };
        try std.testing.expectEqual(width, row.len);
        var row_maximum: f64 = 0;
        for (row, 0..) |value, column_index| {
            const difference = @abs(@as(f64, actual[row_index * width + column_index]) - try json_number(value));
            row_maximum = @max(row_maximum, difference);
            maximum = @max(maximum, difference);
        }
        std.debug.print("C gate {s} layer 0 {s} token {d} max: {d:.12}\n", .{ case_name, stage_name, row_index, row_maximum });
    }
    return maximum;
}

fn skip_without_golden(err: anyerror, path: []const u8) anyerror {
    if (err == error.FileNotFound) {
        std.debug.print("SKIP golden acceptance: missing {s}; generate it with tools/golden_helper.py\n", .{path});
        return error.SkipZigTest;
    }
    return err;
}

fn test_spec(tied: bool) hf_config.config.spec_t {
    return .{
        .family = .Llama,
        .vocab_size = 4,
        .hidden_size = 2,
        .intermediate_size = 3,
        .num_hidden_layers = 1,
        .num_attention_heads = 1,
        .num_key_value_heads = 1,
        .head_dim = 2,
        .rms_norm_eps = 1e-5,
        .rope_theta = 10_000,
        .rope_scaling = .Default,
        .qkv_bias = false,
        .o_bias = false,
        .mlp_bias = false,
        .qk_norm = false,
        .tie_word_embeddings = tied,
        .sliding_window = null,
    };
}
