//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Expected NCM bias tensors for independent generator bias flags.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const bias_flags_t = struct {
    qkv_bias: bool = false,
    o_bias: bool = false,
    mlp_bias: bool = false,
    norm_bias: bool = false,
};

pub const tensor_t = struct {
    name: []const u8,
    shape: []const u64,
};

/// Builds the layer weight manifest from the source model's explicit dimensions.
pub fn layer_weight_tensors(
    allocator: Allocator,
    hidden_size: u64,
    query_heads: u64,
    head_dim: u64,
    key_value_heads: u64,
    intermediate_size: u64,
    qk_norm: bool,
) ![]tensor_t {
    const query_width = try std.math.mul(u64, query_heads, head_dim);
    const key_value_width = try std.math.mul(u64, key_value_heads, head_dim);
    const fused_width = try std.math.add(u64, query_width, try std.math.mul(u64, 2, key_value_width));
    var tensors: std.ArrayList(tensor_t) = .empty;
    errdefer free_pending_tensors(allocator, &tensors);

    try append_tensor(allocator, &tensors, "transformer.blocks.0.attn.c_attn.weight", &.{ fused_width, hidden_size });
    try append_tensor(allocator, &tensors, "transformer.blocks.0.attn.c_proj.weight", &.{ hidden_size, query_width });
    if (qk_norm) {
        try append_tensor(allocator, &tensors, "transformer.blocks.0.attn.q_norm.weight", &.{head_dim});
        try append_tensor(allocator, &tensors, "transformer.blocks.0.attn.k_norm.weight", &.{head_dim});
    }
    try append_tensor(allocator, &tensors, "transformer.blocks.0.mlp.w_gate.weight", &.{ intermediate_size, hidden_size });
    try append_tensor(allocator, &tensors, "transformer.blocks.0.mlp.w_up.weight", &.{ intermediate_size, hidden_size });
    try append_tensor(allocator, &tensors, "transformer.blocks.0.mlp.c_proj.weight", &.{ hidden_size, intermediate_size });
    try append_tensor(allocator, &tensors, "transformer.blocks.0.ln_1.weight", &.{hidden_size});
    try append_tensor(allocator, &tensors, "transformer.blocks.0.ln_2.weight", &.{hidden_size});
    return tensors.toOwnedSlice(allocator);
}

/// Builds only the optional bias portion of the NCM tensor manifest.
pub fn bias_tensors(
    allocator: Allocator,
    flags: bias_flags_t,
    layer_count: usize,
    hidden_size: u64,
    query_width: u64,
    key_value_width: u64,
    intermediate_size: u64,
) ![]tensor_t {
    var tensors: std.ArrayList(tensor_t) = .empty;
    errdefer free_pending_tensors(allocator, &tensors);
    for (0..layer_count) |layer_index| {
        if (flags.qkv_bias) try append_layer_tensor(allocator, &tensors, layer_index, "attn.c_attn.bias", &.{query_width + 2 * key_value_width});
        if (flags.o_bias) try append_layer_tensor(allocator, &tensors, layer_index, "attn.c_proj.bias", &.{hidden_size});
        if (flags.mlp_bias) {
            try append_layer_tensor(allocator, &tensors, layer_index, "mlp.w_gate.bias", &.{intermediate_size});
            try append_layer_tensor(allocator, &tensors, layer_index, "mlp.w_up.bias", &.{intermediate_size});
            try append_layer_tensor(allocator, &tensors, layer_index, "mlp.c_proj.bias", &.{hidden_size});
        }
        if (flags.norm_bias) {
            try append_layer_tensor(allocator, &tensors, layer_index, "ln_1.bias", &.{hidden_size});
            try append_layer_tensor(allocator, &tensors, layer_index, "ln_2.bias", &.{hidden_size});
        }
    }
    if (flags.norm_bias) try append_tensor(allocator, &tensors, "transformer.ln_f.bias", &.{hidden_size});
    return tensors.toOwnedSlice(allocator);
}

fn layer_name(allocator: Allocator, layer_index: usize, suffix: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "transformer.blocks.{d}.{s}", .{ layer_index, suffix });
}

fn append_tensor(
    allocator: Allocator,
    tensors: *std.ArrayList(tensor_t),
    name: []const u8,
    shape: []const u64,
) !void {
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    const owned_shape = try allocator.dupe(u64, shape);
    errdefer allocator.free(owned_shape);
    try tensors.append(allocator, .{ .name = owned_name, .shape = owned_shape });
}

fn append_layer_tensor(
    allocator: Allocator,
    tensors: *std.ArrayList(tensor_t),
    layer_index: usize,
    suffix: []const u8,
    shape: []const u64,
) !void {
    const name = try layer_name(allocator, layer_index, suffix);
    defer allocator.free(name);
    try append_tensor(allocator, tensors, name, shape);
}

fn free_pending_tensors(allocator: Allocator, tensors: *std.ArrayList(tensor_t)) void {
    for (tensors.items) |tensor| {
        allocator.free(tensor.name);
        allocator.free(tensor.shape);
    }
    tensors.deinit(allocator);
}

pub fn deinit_tensors(allocator: Allocator, tensors: []tensor_t) void {
    for (tensors) |tensor| {
        allocator.free(tensor.name);
        allocator.free(tensor.shape);
    }
    allocator.free(tensors);
}

fn allocation_probe(allocator: Allocator) !void {
    const tensors = try bias_tensors(allocator, .{ .mlp_bias = true, .norm_bias = true }, 2, 8, 8, 4, 12);
    deinit_tensors(allocator, tensors);
}

test "NCM bias flags add only their own manifest tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const dimensions = .{ 8, 8, 4, 12 };
    const layer_count = 2;

    const no_bias = try bias_tensors(allocator, .{}, layer_count, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    try std.testing.expectEqual(@as(usize, 0), no_bias.len);

    const qkv = try bias_tensors(allocator, .{ .qkv_bias = true }, layer_count, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    try std.testing.expectEqual(@as(usize, 2), qkv.len);
    try std.testing.expectEqualStrings("transformer.blocks.0.attn.c_attn.bias", qkv[0].name);
    try std.testing.expectEqualSlices(u64, &.{16}, qkv[0].shape);

    const output = try bias_tensors(allocator, .{ .o_bias = true }, layer_count, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    try std.testing.expectEqual(@as(usize, 2), output.len);
    try std.testing.expectEqualStrings("transformer.blocks.0.attn.c_proj.bias", output[0].name);
    try std.testing.expectEqualSlices(u64, &.{8}, output[0].shape);

    const mlp = try bias_tensors(allocator, .{ .mlp_bias = true }, layer_count, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    try std.testing.expectEqual(@as(usize, 6), mlp.len);
    try std.testing.expectEqualStrings("transformer.blocks.0.mlp.w_gate.bias", mlp[0].name);
    try std.testing.expectEqualSlices(u64, &.{12}, mlp[0].shape);
    try std.testing.expectEqualStrings("transformer.blocks.0.mlp.c_proj.bias", mlp[2].name);
    try std.testing.expectEqualSlices(u64, &.{8}, mlp[2].shape);

    const norm = try bias_tensors(allocator, .{ .norm_bias = true }, layer_count, dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    try std.testing.expectEqual(@as(usize, 5), norm.len);
    try std.testing.expectEqualStrings("transformer.blocks.0.ln_1.bias", norm[0].name);
    try std.testing.expectEqualStrings("transformer.blocks.0.ln_2.bias", norm[1].name);
    try std.testing.expectEqualStrings("transformer.ln_f.bias", norm[4].name);
}

test "NCM layer manifest honors Qwen3 explicit head and MLP dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const small = try layer_weight_tensors(allocator, 1024, 16, 128, 8, 3072, true);
    try std.testing.expectEqualSlices(u64, &.{ 4096, 1024 }, small[0].shape);
    try std.testing.expectEqualSlices(u64, &.{ 1024, 2048 }, small[1].shape);
    try std.testing.expectEqualSlices(u64, &.{128}, small[2].shape);
    try std.testing.expectEqualSlices(u64, &.{ 3072, 1024 }, small[4].shape);
    try std.testing.expectEqualSlices(u64, &.{ 1024, 3072 }, small[6].shape);

    const large = try layer_weight_tensors(allocator, 2560, 32, 128, 8, 9728, true);
    try std.testing.expectEqualSlices(u64, &.{ 6144, 2560 }, large[0].shape);
    try std.testing.expectEqualSlices(u64, &.{ 2560, 4096 }, large[1].shape);
    try std.testing.expectEqualSlices(u64, &.{ 9728, 2560 }, large[4].shape);
    try std.testing.expectEqualSlices(u64, &.{ 2560, 9728 }, large[6].shape);
}

test "NCM bias manifest releases all allocations on failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocation_probe, .{});
}
