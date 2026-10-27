//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! NCM-side normalization adapter using the source model's RMSNorm epsilon.

const std = @import("std");
const kernels = @import("kernels");

/// Applies RMSNorm independently to each row with the supplied source epsilon.
pub fn rms_norm_rows(
    input: []const f32,
    weight: []const f32,
    output: []f32,
    row_count: usize,
    row_width: usize,
    epsilon: f32,
) void {
    rms_norm_rows_with_bias(input, weight, null, output, row_count, row_width, epsilon);
}

/// Applies RMSNorm and an optional learned affine bias to each row.
pub fn rms_norm_rows_with_bias(
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    output: []f32,
    row_count: usize,
    row_width: usize,
    epsilon: f32,
) void {
    std.debug.assert(row_width != 0);
    std.debug.assert(input.len == row_count * row_width);
    std.debug.assert(weight.len == row_width);
    std.debug.assert(output.len == input.len);
    if (bias) |values| std.debug.assert(values.len == row_width);

    for (0..row_count) |row_index| {
        const offset = row_index * row_width;
        kernels.rms_norm(
            input[offset..][0..row_width],
            weight,
            output[offset..][0..row_width],
            epsilon,
        );
        if (bias) |values| {
            for (output[offset..][0..row_width], values) |*value, offset_value| value.* += offset_value;
        }
    }
}

test "NCM RMSNorm rows match the B2 kernel at source epsilons" {
    var prng = std.Random.Xoshiro256.init(0x4E43_4D32);
    const random = prng.random();
    var input: [3 * 17]f32 = undefined;
    var weight: [17]f32 = undefined;
    var got: [3 * 17]f32 = undefined;
    var expected: [3 * 17]f32 = undefined;

    for (&input) |*value| value.* = random.float(f32) * 2 - 1;
    for (&weight) |*value| value.* = random.float(f32) * 2 - 1;

    for ([_]f32{ 1e-5, 1e-6 }) |epsilon| {
        rms_norm_rows(&input, &weight, &got, 3, 17, epsilon);
        for (0..3) |row_index| {
            const offset = row_index * 17;
            kernels.rms_norm(
                input[offset..][0..17],
                &weight,
                expected[offset..][0..17],
                epsilon,
            );
        }
        try std.testing.expectEqualSlices(f32, &expected, &got);
    }
}
