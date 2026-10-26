//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! The tensors a configuration requires, compared with the tensors in a checkpoint's safetensors
//! headers. Problems are grouped by tensor pattern, so a feature missing from every layer is
//! reported once, with a count. No weight data is read.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors");
const config = @import("config.zig");

/// One kind of tensor. Per-layer templates apply to every layer, with `{i}` as the layer index.
pub const template_t = struct {
    /// Suffix after "model.layers.{i}." for per-layer tensors, or the full name otherwise.
    pattern: []const u8,
    /// Name shown in problems, such as "model.layers.{i}.self_attn.q_norm.weight".
    feature: []const u8,
    per_layer: bool,
    shape: []const u64,
    optional: bool,
};

/// Tensors from one shard file, with the file name used by the index.
pub const shard_t = struct {
    file: []const u8,
    tensors: []const safetensors.tensor_info_t,
};

pub const result_t = struct {
    problems: []const config.problem_t,
    /// Facts that are not errors, such as an lm_head.weight that the tied embeddings do not need.
    notes: []const config.problem_t,
};

const GroupKind = enum { Missing, Shape, Dtype, Unexpected, Duplicate };

/// All occurrences of one problem, so that one feature gives one problem.
const group_t = struct {
    kind: GroupKind,
    feature: []const u8,
    count: usize,
    sample: []const u8,
    detail: []const u8,
};

/// Lists the tensors that `spec` requires. Optional tensors are marked, not omitted.
pub fn templates(arena: Allocator, spec: config.spec_t) ![]template_t {
    var list: std.ArrayList(template_t) = .empty;
    const hidden = spec.hidden_size;
    const width = spec.intermediate_size;
    const q_width = spec.num_attention_heads * spec.head_dim;
    const kv_width = spec.num_key_value_heads * spec.head_dim;

    try add_template(arena, &list, "model.embed_tokens.weight", false, &.{ spec.vocab_size, hidden }, false);
    try add_template(arena, &list, "model.norm.weight", false, &.{hidden}, false);
    try add_template(arena, &list, "lm_head.weight", false, &.{ spec.vocab_size, hidden }, spec.tie_word_embeddings);

    try add_template(arena, &list, "input_layernorm.weight", true, &.{hidden}, false);
    try add_template(arena, &list, "self_attn.q_proj.weight", true, &.{ q_width, hidden }, false);
    try add_template(arena, &list, "self_attn.k_proj.weight", true, &.{ kv_width, hidden }, false);
    try add_template(arena, &list, "self_attn.v_proj.weight", true, &.{ kv_width, hidden }, false);
    try add_template(arena, &list, "self_attn.o_proj.weight", true, &.{ hidden, q_width }, false);
    if (spec.qkv_bias) {
        try add_template(arena, &list, "self_attn.q_proj.bias", true, &.{q_width}, false);
        try add_template(arena, &list, "self_attn.k_proj.bias", true, &.{kv_width}, false);
        try add_template(arena, &list, "self_attn.v_proj.bias", true, &.{kv_width}, false);
    }
    if (spec.o_bias) {
        try add_template(arena, &list, "self_attn.o_proj.bias", true, &.{hidden}, false);
    }
    if (spec.qk_norm) {
        try add_template(arena, &list, "self_attn.q_norm.weight", true, &.{spec.head_dim}, false);
        try add_template(arena, &list, "self_attn.k_norm.weight", true, &.{spec.head_dim}, false);
    }
    try add_template(arena, &list, "post_attention_layernorm.weight", true, &.{hidden}, false);
    try add_template(arena, &list, "mlp.gate_proj.weight", true, &.{ width, hidden }, false);
    try add_template(arena, &list, "mlp.up_proj.weight", true, &.{ width, hidden }, false);
    try add_template(arena, &list, "mlp.down_proj.weight", true, &.{ hidden, width }, false);
    if (spec.mlp_bias) {
        try add_template(arena, &list, "mlp.gate_proj.bias", true, &.{width}, false);
        try add_template(arena, &list, "mlp.up_proj.bias", true, &.{width}, false);
        try add_template(arena, &list, "mlp.down_proj.bias", true, &.{hidden}, false);
    }
    return list.items;
}

fn add_template(
    arena: Allocator,
    list: *std.ArrayList(template_t),
    pattern: []const u8,
    per_layer: bool,
    dims: []const u64,
    optional: bool,
) !void {
    const feature = if (per_layer)
        try std.fmt.allocPrint(arena, "model.layers.{{i}}.{s}", .{pattern})
    else
        pattern;
    try list.append(arena, .{
        .pattern = pattern,
        .feature = feature,
        .per_layer = per_layer,
        .shape = try arena.dupe(u64, dims),
        .optional = optional,
    });
}

const layer_name_t = struct {
    layer: u64,
    suffix: []const u8,
};

/// Splits "model.layers.<n>.<suffix>". Returns null for every other name.
fn split_layer_name(name: []const u8) ?layer_name_t {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseInt(u64, rest[0..dot], 10) catch return null;
    return .{ .layer = layer, .suffix = rest[dot + 1 ..] };
}

fn find_template(list: []const template_t, pattern: []const u8, per_layer: bool) ?usize {
    for (list, 0..) |template, index| {
        if (template.per_layer == per_layer and std.mem.eql(u8, template.pattern, pattern)) return index;
    }
    return null;
}

fn supported_dtype(dtype: safetensors.Dtype) bool {
    return switch (dtype) {
        .Bf16, .F16, .F32 => true,
        else => false,
    };
}

fn shape_text(arena: Allocator, shape: []const u64) ![]const u8 {
    var text: []const u8 = "[";
    for (shape, 0..) |dim, index| {
        const separator: []const u8 = if (index == 0) "" else ", ";
        text = try std.fmt.allocPrint(arena, "{s}{s}{d}", .{ text, separator, dim });
    }
    return std.fmt.allocPrint(arena, "{s}]", .{text});
}

fn add_group(
    arena: Allocator,
    groups: *std.ArrayList(group_t),
    kind: GroupKind,
    feature: []const u8,
    sample: []const u8,
    detail: []const u8,
) !void {
    for (groups.items) |*group| {
        if (group.kind == kind and std.mem.eql(u8, group.feature, feature)) {
            group.count += 1;
            return;
        }
    }
    try groups.append(arena, .{ .kind = kind, .feature = feature, .count = 1, .sample = sample, .detail = detail });
}

fn to_problem(arena: Allocator, group: group_t) !config.problem_t {
    const message = switch (group.kind) {
        .Missing => try std.fmt.allocPrint(arena, "missing from the checkpoint: {d} tensor(s), first {s}", .{ group.count, group.sample }),
        .Shape => try std.fmt.allocPrint(arena, "shape mismatch ({s}): {d} tensor(s), first {s}", .{ group.detail, group.count, group.sample }),
        .Dtype => try std.fmt.allocPrint(arena, "dtype {s} is not supported (expected BF16, F16 or F32): {d} tensor(s), first {s}", .{ group.detail, group.count, group.sample }),
        .Unexpected => try std.fmt.allocPrint(arena, "not expected for this configuration: {d} tensor(s), first {s}", .{ group.count, group.sample }),
        .Duplicate => try std.fmt.allocPrint(arena, "appears more than once, first {s}", .{group.sample}),
    };
    return .{ .feature = group.feature, .message = message };
}

/// Compares the tensors a checkpoint holds with the tensors `spec` requires.
pub fn check(arena: Allocator, spec: config.spec_t, present: []const safetensors.tensor_info_t) !result_t {
    const list = try templates(arena, spec);
    const seen = try arena.alloc([]u32, list.len);
    for (list, seen) |template, *counts| {
        const slots: usize = if (template.per_layer) @intCast(spec.num_hidden_layers) else 1;
        counts.* = try arena.alloc(u32, slots);
        @memset(counts.*, 0);
    }

    var groups: std.ArrayList(group_t) = .empty;
    var notes: std.ArrayList(config.problem_t) = .empty;

    for (present) |tensor| {
        var index: ?usize = null;
        var layer: usize = 0;
        if (split_layer_name(tensor.name)) |parsed| {
            if (find_template(list, parsed.suffix, true)) |found| {
                if (parsed.layer < spec.num_hidden_layers) {
                    index = found;
                    layer = @intCast(parsed.layer);
                }
            }
            if (index == null) {
                const feature = try std.fmt.allocPrint(arena, "model.layers.{{i}}.{s}", .{parsed.suffix});
                try add_group(arena, &groups, .Unexpected, feature, tensor.name, "");
                continue;
            }
        } else {
            index = find_template(list, tensor.name, false);
            if (index == null) {
                try add_group(arena, &groups, .Unexpected, tensor.name, tensor.name, "");
                continue;
            }
            if (std.mem.eql(u8, tensor.name, "lm_head.weight") and spec.tie_word_embeddings) {
                try notes.append(arena, .{
                    .feature = "lm_head.weight",
                    .message = "present although the embeddings are tied; its values are compared when the weights are read",
                });
            }
        }

        const template = list[index.?];
        seen[index.?][layer] += 1;
        if (seen[index.?][layer] == 2) {
            try add_group(arena, &groups, .Duplicate, template.feature, tensor.name, "");
        }
        if (!std.mem.eql(u64, tensor.shape, template.shape)) {
            const detail = try std.fmt.allocPrint(arena, "actual {s}, expected {s}", .{
                try shape_text(arena, tensor.shape),
                try shape_text(arena, template.shape),
            });
            try add_group(arena, &groups, .Shape, template.feature, tensor.name, detail);
        }
        if (!supported_dtype(tensor.dtype)) {
            try add_group(arena, &groups, .Dtype, template.feature, tensor.name, safetensors.dtype_name(tensor.dtype));
        }
    }

    for (list, seen) |template, counts| {
        if (template.per_layer) {
            for (counts, 0..) |count, layer| {
                if (count != 0 or template.optional) continue;
                const sample = try std.fmt.allocPrint(arena, "model.layers.{d}.{s}", .{ layer, template.pattern });
                try add_group(arena, &groups, .Missing, template.feature, sample, "");
            }
        } else if (counts[0] == 0 and !template.optional) {
            try add_group(arena, &groups, .Missing, template.feature, template.pattern, "");
        }
    }

    var problems: std.ArrayList(config.problem_t) = .empty;
    for (groups.items) |group| {
        try problems.append(arena, try to_problem(arena, group));
    }
    return .{ .problems = problems.items, .notes = notes.items };
}

const IndexKind = enum { ShardMissing, NotInShard, NotIndexed };

/// Index problems grouped by shard, so that one missing shard is one problem.
const index_group_t = struct {
    kind: IndexKind,
    shard: []const u8,
    count: usize,
    sample: []const u8,
};

fn add_index_group(
    arena: Allocator,
    groups: *std.ArrayList(index_group_t),
    kind: IndexKind,
    shard: []const u8,
    sample: []const u8,
) !void {
    for (groups.items) |*group| {
        if (group.kind == kind and std.mem.eql(u8, group.shard, shard)) {
            group.count += 1;
            return;
        }
    }
    try groups.append(arena, .{ .kind = kind, .shard = shard, .count = 1, .sample = sample });
}

/// Checks that a sharded index and the shard headers agree: every indexed tensor is in the named
/// shard, and every tensor in a shard is indexed under that shard.
pub fn check_index(arena: Allocator, index: safetensors.index_t, shards: []const shard_t) ![]const config.problem_t {
    var groups: std.ArrayList(index_group_t) = .empty;
    for (index.weight_map) |entry| {
        const shard = find_shard(shards, entry.shard) orelse {
            try add_index_group(arena, &groups, .ShardMissing, entry.shard, entry.tensor);
            continue;
        };
        if (!contains_tensor(shard.tensors, entry.tensor)) {
            try add_index_group(arena, &groups, .NotInShard, entry.shard, entry.tensor);
        }
    }
    for (shards) |shard| {
        for (shard.tensors) |tensor| {
            if (!indexed_in(index, tensor.name, shard.file)) {
                try add_index_group(arena, &groups, .NotIndexed, shard.file, tensor.name);
            }
        }
    }

    var problems: std.ArrayList(config.problem_t) = .empty;
    for (groups.items) |group| {
        const message = switch (group.kind) {
            .ShardMissing => try std.fmt.allocPrint(arena, "shard '{s}' is listed for {d} tensor(s) but was not found, first '{s}'", .{ group.shard, group.count, group.sample }),
            .NotInShard => try std.fmt.allocPrint(arena, "{d} tensor(s) are listed under shard '{s}' but are not in it, first '{s}'", .{ group.count, group.shard, group.sample }),
            .NotIndexed => try std.fmt.allocPrint(arena, "{d} tensor(s) in shard '{s}' are missing from the index, first '{s}'", .{ group.count, group.shard, group.sample }),
        };
        try problems.append(arena, .{ .feature = "model.safetensors.index.json", .message = message });
    }
    return problems.items;
}

fn find_shard(shards: []const shard_t, file: []const u8) ?shard_t {
    for (shards) |shard| {
        if (std.mem.eql(u8, shard.file, file)) return shard;
    }
    return null;
}

fn contains_tensor(tensors: []const safetensors.tensor_info_t, name: []const u8) bool {
    for (tensors) |tensor| {
        if (std.mem.eql(u8, tensor.name, name)) return true;
    }
    return false;
}

fn indexed_in(index: safetensors.index_t, tensor: []const u8, file: []const u8) bool {
    for (index.weight_map) |entry| {
        if (std.mem.eql(u8, entry.tensor, tensor) and std.mem.eql(u8, entry.shard, file)) return true;
    }
    return false;
}
