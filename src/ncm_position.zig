//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! NCM RoPE table gathering for explicit, restartable position ids.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("kernels");

/// Writes positions for a prefix followed by target levels that each restart at zero.
pub fn fill_level_positions(prefix_length: usize, level_lengths: []const usize, positions: []usize) !void {
    var sequence_length = prefix_length;
    for (level_lengths) |level_length| sequence_length = try std.math.add(usize, sequence_length, level_length);
    if (positions.len != sequence_length) return error.InvalidPositionBuffer;

    for (0..prefix_length) |position| positions[position] = position;
    var offset = prefix_length;
    for (level_lengths) |level_length| {
        for (0..level_length) |position| positions[offset + position] = position;
        offset += level_length;
    }
}

/// Builds Llama 3 RoPE values at the requested positions using the B4 kernel.
pub fn default_tables_at_positions(
    allocator: Allocator,
    head_dim: usize,
    rope_theta: f32,
    positions: []const usize,
    cosine: []f32,
    sine: []f32,
) !void {
    try tables_at_positions(allocator, head_dim, positions, cosine, sine, .{
        .rope_type = .Default,
        .rope_theta = rope_theta,
    });
}

/// Builds Llama 3 RoPE values at the requested positions using the B4 kernel.
pub fn llama3_tables_at_positions(
    allocator: Allocator,
    head_dim: usize,
    rope_theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position_embeddings: usize,
    positions: []const usize,
    cosine: []f32,
    sine: []f32,
) !void {
    try tables_at_positions(allocator, head_dim, positions, cosine, sine, .{
        .rope_type = .Llama3,
        .rope_theta = rope_theta,
        .factor = factor,
        .low_freq_factor = low_freq_factor,
        .high_freq_factor = high_freq_factor,
        .original_max_position_embeddings = original_max_position_embeddings,
    });
}

const rope_spec_t = struct {
    rope_type: enum { Default, Llama3 },
    rope_theta: f32,
    factor: f32 = 1,
    low_freq_factor: f32 = 1,
    high_freq_factor: f32 = 4,
    original_max_position_embeddings: usize = 8192,
};

fn tables_at_positions(
    allocator: Allocator,
    head_dim: usize,
    positions: []const usize,
    cosine: []f32,
    sine: []f32,
    spec: rope_spec_t,
) !void {
    const output_len = try std.math.mul(usize, positions.len, head_dim);
    if (head_dim == 0 or head_dim % 2 != 0 or cosine.len != output_len or sine.len != output_len) {
        return error.InvalidRopeTableSize;
    }
    if (positions.len == 0) return;

    var maximum_position: usize = 0;
    for (positions) |position| maximum_position = @max(maximum_position, position);
    const position_count = try std.math.add(usize, maximum_position, 1);
    const table_len = try std.math.mul(usize, position_count, head_dim);
    const mut_cosine = try allocator.alloc(f32, table_len);
    defer allocator.free(mut_cosine);
    const mut_sine = try allocator.alloc(f32, table_len);
    defer allocator.free(mut_sine);
    const inverse_frequencies = try allocator.alloc(f32, head_dim / 2);
    defer allocator.free(inverse_frequencies);

    switch (spec.rope_type) {
        .Default => kernels.rope_default(head_dim, spec.rope_theta, position_count, inverse_frequencies, mut_cosine, mut_sine),
        .Llama3 => {
            kernels.rope_llama3(
                head_dim,
                spec.rope_theta,
                spec.factor,
                spec.low_freq_factor,
                spec.high_freq_factor,
                spec.original_max_position_embeddings,
                inverse_frequencies,
            );
            kernels.rope_tables(inverse_frequencies, position_count, mut_cosine, mut_sine);
        },
    }

    for (positions, 0..) |position, index| {
        const source_offset = position * head_dim;
        const target_offset = index * head_dim;
        @memcpy(cosine[target_offset..][0..head_dim], mut_cosine[source_offset..][0..head_dim]);
        @memcpy(sine[target_offset..][0..head_dim], mut_sine[source_offset..][0..head_dim]);
    }
}

test "NCM Llama 3 positions reuse B4 frequencies across position restarts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const positions = [_]usize{ 0, 1, 2, 0, 1, 0 };
    var actual_cosine: [positions.len * 64]f32 = undefined;
    var actual_sine: [positions.len * 64]f32 = undefined;
    var inverse_frequencies: [32]f32 = undefined;
    var reference_cosine: [3 * 64]f32 = undefined;
    var reference_sine: [3 * 64]f32 = undefined;

    kernels.rope_llama3(64, 500_000, 32, 1, 4, 8192, &inverse_frequencies);
    kernels.rope_tables(&inverse_frequencies, 3, &reference_cosine, &reference_sine);
    try llama3_tables_at_positions(
        allocator,
        64,
        500_000,
        32,
        1,
        4,
        8192,
        &positions,
        &actual_cosine,
        &actual_sine,
    );

    for (positions, 0..) |position, index| {
        const actual_offset = index * 64;
        const reference_offset = position * 64;
        try std.testing.expectEqualSlices(f32, reference_cosine[reference_offset..][0..64], actual_cosine[actual_offset..][0..64]);
        try std.testing.expectEqualSlices(f32, reference_sine[reference_offset..][0..64], actual_sine[actual_offset..][0..64]);
    }
}

test "NCM position ids restart for each target level" {
    const levels = [_]usize{ 1, 2, 3 };
    var positions: [8]usize = undefined;
    try fill_level_positions(2, &levels, &positions);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 0, 0, 1, 0, 1, 2 }, &positions);
}
