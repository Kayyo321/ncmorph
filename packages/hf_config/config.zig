//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Parses config.json for decoder-only LLM checkpoints. Family detection, the feature flags
//! and the rejected features follow the transformers v5.20.2 modeling and configuration code.
//! Every problem is collected, so one run reports each feature once.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ObjectMap = std.json.ObjectMap;

pub const Family = enum { Llama, Qwen2, Qwen3, Mistral };

/// One problem with the configuration: the key or feature involved, and what is wrong.
pub const problem_t = struct {
    feature: []const u8,
    message: []const u8,
};

pub const llama3_t = struct {
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    original_max_position_embeddings: u64,
};

pub const rope_scaling_t = union(enum) {
    Default,
    Llama3: llama3_t,
};

pub const spec_t = struct {
    family: Family,
    vocab_size: u64,
    hidden_size: u64,
    intermediate_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    num_key_value_heads: u64,
    head_dim: u64,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: rope_scaling_t,
    qkv_bias: bool,
    o_bias: bool,
    mlp_bias: bool,
    qk_norm: bool,
    tie_word_embeddings: bool,
    /// Sliding window in positions, or null for full causal attention.
    sliding_window: ?u64,
};

pub const report_t = struct {
    /// Null whenever a problem was found, because the values would then be incomplete.
    spec: ?spec_t,
    problems: []const problem_t,
};

const family_row_t = struct {
    model_type: []const u8,
    architecture: []const u8,
    family: Family,
};

const FamilyRows = [_]family_row_t{
    .{ .model_type = "llama", .architecture = "LlamaForCausalLM", .family = .Llama },
    .{ .model_type = "qwen2", .architecture = "Qwen2ForCausalLM", .family = .Qwen2 },
    .{ .model_type = "qwen3", .architecture = "Qwen3ForCausalLM", .family = .Qwen3 },
    .{ .model_type = "mistral", .architecture = "MistralForCausalLM", .family = .Mistral },
};

/// Head width of Qwen3 when `head_dim` is absent: the config class default, not hidden / heads.
const Qwen3DefaultHeadDim: u64 = 128;

/// RoPE base when config.json has no rope_theta. transformers v5.20.2 applies
/// RotaryEmbeddingConfigMixin.default_theta (modeling_rope_utils.py) to every family here.
const DefaultRopeTheta: f64 = 10000.0;

fn find_family(model_type: []const u8) ?family_row_t {
    for (FamilyRows) |row| {
        if (std.mem.eql(u8, row.model_type, model_type)) return row;
    }
    return null;
}

/// Collects problems for one parse.
const collector_t = struct {
    arena: Allocator,
    problems: std.ArrayList(problem_t) = .empty,

    fn add(self: *collector_t, feature: []const u8, comptime format: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(self.arena, format, args);
        try self.problems.append(self.arena, .{ .feature = feature, .message = message });
    }
};

/// Parses a config.json document. Invalid JSON and a root that is not an object are errors.
/// Every other problem is returned in the report.
pub fn parse_config(arena: Allocator, json_bytes: []const u8) !report_t {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json_bytes, .{});
    const fields = switch (root) {
        .object => |object| object,
        else => return error.InvalidConfig,
    };
    var collector: collector_t = .{ .arena = arena };

    const model_type = (try string_field(&collector, fields, "model_type")) orelse
        return .{ .spec = null, .problems = collector.problems.items };
    const row = find_family(model_type) orelse {
        try collector.add("model_type", "unsupported model_type '{s}'", .{model_type});
        return .{ .spec = null, .problems = collector.problems.items };
    };
    const family = row.family;
    try check_architecture(&collector, fields, row);

    const vocab_size = try positive_field(&collector, fields, "vocab_size");
    const hidden_size = try positive_field(&collector, fields, "hidden_size");
    const intermediate_size = try positive_field(&collector, fields, "intermediate_size");
    const num_hidden_layers = try positive_field(&collector, fields, "num_hidden_layers");
    const num_attention_heads = try positive_field(&collector, fields, "num_attention_heads");
    const num_key_value_heads = (try uint_field(&collector, fields, "num_key_value_heads")) orelse num_attention_heads;
    if (num_key_value_heads == 0 or num_attention_heads % num_key_value_heads != 0) {
        try collector.add("num_key_value_heads", "must divide num_attention_heads", .{});
    }

    const head_dim = if (try uint_field(&collector, fields, "head_dim")) |explicit|
        explicit
    else if (family == .Qwen3)
        Qwen3DefaultHeadDim
    else if (num_attention_heads > 0 and hidden_size % num_attention_heads == 0)
        hidden_size / num_attention_heads
    else blk: {
        try collector.add("head_dim", "hidden_size must divide by num_attention_heads when head_dim is absent", .{});
        break :blk 0;
    };

    const rms_norm_eps = (try float_field(&collector, fields, "rms_norm_eps")) orelse 0;
    const rope_theta = if (fields.contains("rope_theta"))
        (try float_field(&collector, fields, "rope_theta")) orelse 0
    else
        DefaultRopeTheta;
    const rope_scaling = try parse_rope_scaling(&collector, fields);

    const hidden_act = (try optional_string_field(&collector, fields, "hidden_act")) orelse "silu";
    if (!std.mem.eql(u8, hidden_act, "silu")) {
        try collector.add("hidden_act", "only silu is supported, not '{s}'", .{hidden_act});
    }
    if (try bool_field(&collector, fields, "use_mrope", false)) {
        try collector.add("use_mrope", "multimodal rotary embeddings are not supported", .{});
    }
    if (try bool_field(&collector, fields, "rope_interleaved", false)) {
        try collector.add("rope_interleaved", "only the rotate-half RoPE layout is supported", .{});
    }
    if (fields.get("partial_rotary_factor")) |value| switch (value) {
        .float => |factor| if (factor != 1.0) try collector.add("partial_rotary_factor", "only full rotary embeddings are supported", .{}),
        .integer => |factor| if (factor != 1) try collector.add("partial_rotary_factor", "only full rotary embeddings are supported", .{}),
        else => try collector.add("partial_rotary_factor", "must be a number", .{}),
    };
    for ([_][]const u8{ "num_local_experts", "num_experts", "num_experts_per_tok" }) |key| {
        if (fields.get(key)) |value| switch (value) {
            .null => {},
            .integer => |count| if (count != 0) try collector.add(key, "mixture-of-experts models are not supported", .{}),
            else => try collector.add(key, "mixture-of-experts models are not supported", .{}),
        };
    }
    try check_layer_types(&collector, fields);

    const tie_word_embeddings = try bool_field(&collector, fields, "tie_word_embeddings", false);
    const attention_bias = switch (family) {
        .Llama, .Qwen3 => try bool_field(&collector, fields, "attention_bias", false),
        .Qwen2, .Mistral => false,
    };
    const mlp_bias = switch (family) {
        .Llama => try bool_field(&collector, fields, "mlp_bias", false),
        .Qwen2, .Qwen3, .Mistral => false,
    };
    const sliding_window = try parse_sliding_window(&collector, fields, family);

    const spec: spec_t = .{
        .family = family,
        .vocab_size = vocab_size,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .num_hidden_layers = num_hidden_layers,
        .num_attention_heads = num_attention_heads,
        .num_key_value_heads = num_key_value_heads,
        .head_dim = head_dim,
        .rms_norm_eps = rms_norm_eps,
        .rope_theta = rope_theta,
        .rope_scaling = rope_scaling,
        .qkv_bias = switch (family) {
            .Llama, .Qwen3 => attention_bias,
            .Qwen2 => true,
            .Mistral => false,
        },
        .o_bias = switch (family) {
            .Llama, .Qwen3 => attention_bias,
            .Qwen2, .Mistral => false,
        },
        .mlp_bias = mlp_bias,
        .qk_norm = family == .Qwen3,
        .tie_word_embeddings = tie_word_embeddings,
        .sliding_window = sliding_window,
    };
    const problems = collector.problems.items;
    return .{ .spec = if (problems.len == 0) spec else null, .problems = problems };
}

fn check_architecture(collector: *collector_t, fields: ObjectMap, row: family_row_t) !void {
    const items = switch (fields.get("architectures") orelse {
        try collector.add("architectures", "required key is missing", .{});
        return;
    }) {
        .array => |array| array.items,
        else => {
            try collector.add("architectures", "must be a list", .{});
            return;
        },
    };
    if (items.len != 1) {
        try collector.add("architectures", "must list exactly one architecture", .{});
        return;
    }
    const name = switch (items[0]) {
        .string => |text| text,
        else => {
            try collector.add("architectures", "must contain strings", .{});
            return;
        },
    };
    if (!std.mem.eql(u8, name, row.architecture)) {
        try collector.add("architectures", "'{s}' does not match model_type '{s}', which expects '{s}'", .{ name, row.model_type, row.architecture });
    }
}

fn string_field(collector: *collector_t, fields: ObjectMap, key: []const u8) !?[]const u8 {
    const value = fields.get(key) orelse {
        try collector.add(key, "required key is missing", .{});
        return null;
    };
    return switch (value) {
        .string => |text| text,
        else => {
            try collector.add(key, "must be a string", .{});
            return null;
        },
    };
}

/// Returns null when the key is absent. Present values must be strings.
fn optional_string_field(collector: *collector_t, fields: ObjectMap, key: []const u8) !?[]const u8 {
    const value = fields.get(key) orelse return null;
    return switch (value) {
        .string => |text| text,
        else => {
            try collector.add(key, "must be a string", .{});
            return null;
        },
    };
}

/// Returns null when the key is absent or invalid. Invalid values are reported.
fn uint_field(collector: *collector_t, fields: ObjectMap, key: []const u8) !?u64 {
    const value = fields.get(key) orelse return null;
    switch (value) {
        .integer => |number| {
            if (number >= 0) return @as(u64, @intCast(number));
            try collector.add(key, "must not be negative", .{});
        },
        else => try collector.add(key, "must be an integer", .{}),
    }
    return null;
}

fn positive_field(collector: *collector_t, fields: ObjectMap, key: []const u8) !u64 {
    if (!fields.contains(key)) {
        try collector.add(key, "required key is missing", .{});
        return 0;
    }
    const value = (try uint_field(collector, fields, key)) orelse return 0;
    if (value == 0) {
        try collector.add(key, "must be positive", .{});
        return 0;
    }
    return value;
}

fn float_field(collector: *collector_t, fields: ObjectMap, key: []const u8) !?f64 {
    const value = fields.get(key) orelse {
        try collector.add(key, "required key is missing", .{});
        return null;
    };
    return switch (value) {
        .float => |number| number,
        .integer => |number| @as(f64, @floatFromInt(number)),
        else => {
            try collector.add(key, "must be a number", .{});
            return null;
        },
    };
}

fn bool_field(collector: *collector_t, fields: ObjectMap, key: []const u8, default: bool) !bool {
    const value = fields.get(key) orelse return default;
    return switch (value) {
        .bool => |flag| flag,
        else => {
            try collector.add(key, "must be true or false", .{});
            return default;
        },
    };
}

fn parse_rope_scaling(collector: *collector_t, fields: ObjectMap) !rope_scaling_t {
    if (fields.contains("rope_parameters")) {
        try collector.add("rope_parameters", "only rope_scaling is read in this version", .{});
    }
    const value = fields.get("rope_scaling") orelse return .Default;
    const scaling = switch (value) {
        .null => return .Default,
        .object => |object| object,
        else => {
            try collector.add("rope_scaling", "must be an object or null", .{});
            return .Default;
        },
    };
    const rope_type_value = scaling.get("rope_type") orelse scaling.get("type") orelse return .Default;
    const rope_type = switch (rope_type_value) {
        .string => |text| text,
        else => {
            try collector.add("rope_scaling", "rope type must be a string", .{});
            return .Default;
        },
    };
    if (std.mem.eql(u8, rope_type, "default")) return .Default;
    if (!std.mem.eql(u8, rope_type, "llama3")) {
        try collector.add("rope_scaling", "rope type '{s}' is not supported in this version", .{rope_type});
        return .Default;
    }
    return .{ .Llama3 = .{
        .factor = try scaling_float(collector, scaling, "factor"),
        .low_freq_factor = try scaling_float(collector, scaling, "low_freq_factor"),
        .high_freq_factor = try scaling_float(collector, scaling, "high_freq_factor"),
        .original_max_position_embeddings = try scaling_uint(collector, scaling, "original_max_position_embeddings"),
    } };
}

fn scaling_float(collector: *collector_t, scaling: ObjectMap, comptime key: []const u8) !f64 {
    const value = scaling.get(key) orelse {
        try collector.add("rope_scaling." ++ key, "required key is missing", .{});
        return 0;
    };
    return switch (value) {
        .float => |number| number,
        .integer => |number| @as(f64, @floatFromInt(number)),
        else => {
            try collector.add("rope_scaling." ++ key, "must be a number", .{});
            return 0;
        },
    };
}

fn scaling_uint(collector: *collector_t, scaling: ObjectMap, comptime key: []const u8) !u64 {
    const value = scaling.get(key) orelse {
        try collector.add("rope_scaling." ++ key, "required key is missing", .{});
        return 0;
    };
    return switch (value) {
        .integer => |number| if (number >= 0) @as(u64, @intCast(number)) else 0,
        else => {
            try collector.add("rope_scaling." ++ key, "must be a non-negative integer", .{});
            return 0;
        },
    };
}

fn check_layer_types(collector: *collector_t, fields: ObjectMap) !void {
    const value = fields.get("layer_types") orelse return;
    const items = switch (value) {
        .null => return,
        .array => |array| array.items,
        else => {
            try collector.add("layer_types", "must be a list", .{});
            return;
        },
    };
    for (items) |item| {
        const kind = switch (item) {
            .string => |text| text,
            else => "",
        };
        if (!std.mem.eql(u8, kind, "full_attention")) {
            try collector.add("layer_types", "only full attention is supported in this version", .{});
            return;
        }
    }
}

/// Sliding windows follow the transformers configs. Qwen2 and Qwen3 use a window only when
/// use_sliding_window is true, and that is rejected here. Mistral uses sliding_window directly,
/// and its config class defaults the value to 4096 when the key is absent.
fn parse_sliding_window(collector: *collector_t, fields: ObjectMap, family: Family) !?u64 {
    switch (family) {
        .Qwen2, .Qwen3 => {
            if (try bool_field(collector, fields, "use_sliding_window", false)) {
                try collector.add("use_sliding_window", "sliding-window attention is not supported in this version", .{});
            }
            return null;
        },
        .Llama => {
            if (fields.get("sliding_window")) |value| switch (value) {
                .null => {},
                else => try collector.add("sliding_window", "sliding-window attention is not supported for llama", .{}),
            };
            return null;
        },
        .Mistral => {
            const value = fields.get("sliding_window") orelse return 4096;
            return switch (value) {
                .null => null,
                else => (try uint_field(collector, fields, "sliding_window")),
            };
        },
    }
}

test "parses a Qwen2.5-style config" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","vocab_size":151936,
        \\ "hidden_size":896,"intermediate_size":4864,"num_hidden_layers":24,
        \\ "num_attention_heads":14,"num_key_value_heads":2,"rms_norm_eps":1e-06,
        \\ "rope_theta":1000000.0,"hidden_act":"silu","tie_word_embeddings":true,
        \\ "use_sliding_window":false,"sliding_window":32768,"use_mrope":false}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(usize, 0), report.problems.len);
    const spec = report.spec.?;
    try std.testing.expectEqual(Family.Qwen2, spec.family);
    try std.testing.expectEqual(@as(u64, 64), spec.head_dim);
    try std.testing.expect(spec.qkv_bias);
    try std.testing.expect(!spec.o_bias);
    try std.testing.expectEqual(@as(?u64, null), spec.sliding_window);
}

test "Qwen3 uses the explicit head_dim and QK-norm" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","vocab_size":151936,
        \\ "hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,
        \\ "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,
        \\ "rms_norm_eps":1e-06,"rope_theta":1000000,"attention_bias":false,
        \\ "tie_word_embeddings":true,"sliding_window":null,"use_sliding_window":false}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(usize, 0), report.problems.len);
    const spec = report.spec.?;
    try std.testing.expectEqual(@as(u64, 128), spec.head_dim);
    try std.testing.expect(spec.qk_norm);
    try std.testing.expect(!spec.qkv_bias);
}

test "rope_theta defaults to 10000 when the key is absent" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":32000,
        \\ "hidden_size":4096,"intermediate_size":11008,"num_hidden_layers":32,
        \\ "num_attention_heads":32,"rms_norm_eps":1e-05,"hidden_act":"silu",
        \\ "tie_word_embeddings":false}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(usize, 0), report.problems.len);
    try std.testing.expectEqual(@as(f64, 10000.0), report.spec.?.rope_theta);
}

test "Mistral v0.1 keeps its 4096 sliding window" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["MistralForCausalLM"],"model_type":"mistral","vocab_size":32000,
        \\ "hidden_size":4096,"intermediate_size":14336,"num_hidden_layers":32,
        \\ "num_attention_heads":32,"num_key_value_heads":8,"rms_norm_eps":1e-05,
        \\ "rope_theta":10000.0,"sliding_window":4096}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(?u64, 4096), report.spec.?.sliding_window);
}

test "Mistral without a sliding_window key uses the config default" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["MistralForCausalLM"],"model_type":"mistral","vocab_size":8,
        \\ "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-05,"rope_theta":1e6}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(?u64, 4096), report.spec.?.sliding_window);
}

test "each unsupported feature yields exactly one problem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const base =
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":8,
        \\ "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-05,
        \\ "rope_theta":10000.0,"hidden_act":"gelu"}
    ;
    const report = try parse_config(arena.allocator(), base);
    try std.testing.expect(report.spec == null);
    try std.testing.expectEqual(@as(usize, 1), report.problems.len);
    try std.testing.expectEqualStrings("hidden_act", report.problems[0].feature);
}

test "rejects an architecture that disagrees with model_type" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["LlamaForCausalLM"],"model_type":"qwen2","vocab_size":8,
        \\ "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-06,"rope_theta":1e6}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqualStrings("architectures", report.problems[0].feature);
}

test "rejects an unsupported rope type and names it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":8,
        \\ "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-05,
        \\ "rope_theta":10000.0,"rope_scaling":{"rope_type":"linear","factor":2.0}}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(usize, 1), report.problems.len);
    try std.testing.expectEqualStrings("rope_scaling", report.problems[0].feature);
}

test "reads llama3 RoPE scaling" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":8,
        \\ "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-05,
        \\ "rope_theta":500000.0,"rope_scaling":{"factor":32.0,"high_freq_factor":4.0,
        \\ "low_freq_factor":1.0,"original_max_position_embeddings":8192,"rope_type":"llama3"}}
    ;
    const report = try parse_config(arena.allocator(), json);
    const scaling = report.spec.?.rope_scaling.Llama3;
    try std.testing.expectEqual(@as(f64, 32.0), scaling.factor);
    try std.testing.expectEqual(@as(u64, 8192), scaling.original_max_position_embeddings);
}

test "rejects multimodal RoPE, interleaved RoPE and mixture-of-experts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const json =
        \\{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","vocab_size":8,
        \\ "hidden_size":4,"intermediate_size":8,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-06,
        \\ "rope_theta":1e6,"use_mrope":true,"rope_interleaved":true,"num_experts":4}
    ;
    const report = try parse_config(arena.allocator(), json);
    try std.testing.expectEqual(@as(usize, 3), report.problems.len);
}
