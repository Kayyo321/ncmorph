//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Block-diagonal NCM attention masks.

const std = @import("std");

/// Fills a row-major allow mask: prefix queries see prefix keys, and each
/// target level sees prefix keys plus only the positions in its own level.
pub fn block_diagonal_mask(
    prefix_length: usize,
    level_lengths: []const usize,
    prefix_pad_mask: []const bool,
    output: []bool,
) !void {
    if (prefix_pad_mask.len != 0 and prefix_pad_mask.len != prefix_length) return error.InvalidPrefixMask;
    var target_length: usize = 0;
    for (level_lengths) |level_length| target_length = try std.math.add(usize, target_length, level_length);
    const sequence_length = try std.math.add(usize, prefix_length, target_length);
    const mask_length = try std.math.mul(usize, sequence_length, sequence_length);
    if (output.len != mask_length) return error.InvalidMaskSize;
    @memset(output, false);

    for (0..prefix_length) |query| {
        for (0..prefix_length) |key| {
            output[query * sequence_length + key] = !is_padded(prefix_pad_mask, key);
        }
    }

    var level_offset: usize = prefix_length;
    for (level_lengths) |level_length| {
        for (level_offset..level_offset + level_length) |query| {
            for (0..prefix_length) |key| {
                output[query * sequence_length + key] = !is_padded(prefix_pad_mask, key);
            }
            for (level_offset..level_offset + level_length) |key| {
                output[query * sequence_length + key] = true;
            }
        }
        level_offset += level_length;
    }
}

fn is_padded(prefix_pad_mask: []const bool, index: usize) bool {
    return prefix_pad_mask.len != 0 and prefix_pad_mask[index];
}

test "NCM block-diagonal mask isolates targets and excludes padded prefix keys" {
    const level_lengths = [_]usize{ 1, 2 };
    const prefix_pad_mask = [_]bool{ false, true };
    var mask: [25]bool = undefined;
    try block_diagonal_mask(2, &level_lengths, &prefix_pad_mask, &mask);

    try std.testing.expect(mask[0 * 5 + 0]);
    try std.testing.expect(!mask[0 * 5 + 1]);
    try std.testing.expect(!mask[0 * 5 + 2]);
    try std.testing.expect(mask[2 * 5 + 0]);
    try std.testing.expect(!mask[2 * 5 + 1]);
    try std.testing.expect(mask[2 * 5 + 2]);
    try std.testing.expect(!mask[2 * 5 + 3]);
    try std.testing.expect(mask[3 * 5 + 3]);
    try std.testing.expect(mask[3 * 5 + 4]);
    try std.testing.expect(mask[4 * 5 + 3]);
}
