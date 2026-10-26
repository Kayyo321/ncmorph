//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Tests for manifest.zig with synthetic configurations and tensor lists.

const std = @import("std");
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors");
const config = @import("config.zig");
const manifest = @import("manifest.zig");

fn test_spec(family: config.Family, qk_norm: bool, qkv_bias: bool, tie: bool) config.spec_t {
    return .{
        .family = family,
        .vocab_size = 16,
        .hidden_size = 8,
        .intermediate_size = 12,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 4,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1e6,
        .rope_scaling = .Default,
        .qkv_bias = qkv_bias,
        .o_bias = false,
        .mlp_bias = false,
        .qk_norm = qk_norm,
        .tie_word_embeddings = tie,
        .sliding_window = null,
    };
}

/// Lists the tensors a complete checkpoint for `spec` holds. `skip_suffix` leaves out one
/// per-layer tensor in every layer.
fn present_from_templates(arena: Allocator, spec: config.spec_t, skip_suffix: ?[]const u8) ![]safetensors.tensor_info_t {
    const list = try manifest.templates(arena, spec);
    var present: std.ArrayList(safetensors.tensor_info_t) = .empty;
    for (list) |template| {
        if (template.optional) continue;
        if (template.per_layer) {
            if (skip_suffix) |skip| {
                if (std.mem.eql(u8, skip, template.pattern)) continue;
            }
            var layer: u64 = 0;
            while (layer < spec.num_hidden_layers) : (layer += 1) {
                const name = try std.fmt.allocPrint(arena, "model.layers.{d}.{s}", .{ layer, template.pattern });
                try present.append(arena, .{ .name = name, .dtype = .Bf16, .shape = template.shape, .data_begin = 0, .data_end = 0 });
            }
        } else {
            try present.append(arena, .{ .name = template.pattern, .dtype = .Bf16, .shape = template.shape, .data_begin = 0, .data_end = 0 });
        }
    }
    return present.items;
}

test "a complete checkpoint has no problems" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const spec = test_spec(.Qwen2, false, true, true);
    const present = try present_from_templates(arena.allocator(), spec, null);
    const result = try manifest.check(arena.allocator(), spec, present);
    try std.testing.expectEqual(@as(usize, 0), result.problems.len);
    try std.testing.expectEqual(@as(usize, 0), result.notes.len);
}

test "a feature missing from every layer is one problem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const spec = test_spec(.Qwen3, true, false, true);
    const present = try present_from_templates(arena.allocator(), spec, "self_attn.q_norm.weight");
    const result = try manifest.check(arena.allocator(), spec, present);
    try std.testing.expectEqual(@as(usize, 1), result.problems.len);
    try std.testing.expectEqualStrings("model.layers.{i}.self_attn.q_norm.weight", result.problems[0].feature);
}

test "a shape mismatch is one problem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const spec = test_spec(.Llama, false, false, true);
    const present = try present_from_templates(arena.allocator(), spec, null);
    present[0].shape = &.{ 1, 1 };
    const result = try manifest.check(arena.allocator(), spec, present);
    try std.testing.expectEqual(@as(usize, 1), result.problems.len);
    try std.testing.expectEqualStrings("model.embed_tokens.weight", result.problems[0].feature);
}

test "an unexpected tensor is one problem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const spec = test_spec(.Llama, false, false, true);
    const base = try present_from_templates(arena.allocator(), spec, null);
    var present: std.ArrayList(safetensors.tensor_info_t) = .empty;
    try present.appendSlice(arena.allocator(), base);
    try present.append(arena.allocator(), .{ .name = "model.layers.0.mlp.extra.weight", .dtype = .F32, .shape = &.{1}, .data_begin = 0, .data_end = 0 });
    const result = try manifest.check(arena.allocator(), spec, present.items);
    try std.testing.expectEqual(@as(usize, 1), result.problems.len);
}

test "a tied lm_head is a note, not a problem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const spec = test_spec(.Qwen3, true, false, true);
    const base = try present_from_templates(arena.allocator(), spec, null);
    var present: std.ArrayList(safetensors.tensor_info_t) = .empty;
    try present.appendSlice(arena.allocator(), base);
    try present.append(arena.allocator(), .{ .name = "lm_head.weight", .dtype = .Bf16, .shape = &.{ 16, 8 }, .data_begin = 0, .data_end = 0 });
    const result = try manifest.check(arena.allocator(), spec, present.items);
    try std.testing.expectEqual(@as(usize, 0), result.problems.len);
    try std.testing.expectEqual(@as(usize, 1), result.notes.len);
}

test "an unsupported dtype is one problem" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const spec = test_spec(.Llama, false, false, true);
    const present = try present_from_templates(arena.allocator(), spec, null);
    present[0].dtype = .U8;
    const result = try manifest.check(arena.allocator(), spec, present);
    try std.testing.expectEqual(@as(usize, 1), result.problems.len);
}

test "index and shards must agree" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const index: safetensors.index_t = .{
        .weight_map = &.{
            .{ .tensor = "a", .shard = "s1" },
            .{ .tensor = "b", .shard = "s2" },
        },
        .total_size = null,
    };
    const s1 = [_]safetensors.tensor_info_t{
        .{ .name = "a", .dtype = .F32, .shape = &.{1}, .data_begin = 0, .data_end = 4 },
    };
    const s2 = [_]safetensors.tensor_info_t{
        .{ .name = "b", .dtype = .F32, .shape = &.{1}, .data_begin = 0, .data_end = 4 },
        .{ .name = "c", .dtype = .F32, .shape = &.{1}, .data_begin = 4, .data_end = 8 },
    };
    const shards = [_]manifest.shard_t{
        .{ .file = "s1", .tensors = &s1 },
        .{ .file = "s2", .tensors = &s2 },
    };
    const problems = try manifest.check_index(arena.allocator(), index, &shards);
    try std.testing.expectEqual(@as(usize, 1), problems.len);
}
