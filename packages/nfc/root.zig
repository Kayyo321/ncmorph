//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Unicode 9.0.0 NFC normalization for tokenizer input.

const std = @import("std");
const tables = @import("tables.zig");

pub const UnicodeVersion = tables.Version;

const Allocator = std.mem.Allocator;
const Codepoints = std.ArrayList(u21);

/// Returns a newly allocated NFC-normalized UTF-8 string. The caller owns it.
pub fn normalize(allocator: Allocator, input: []const u8) (Allocator.Error || error{InvalidUtf8})![]u8 {
    var decoded = std.unicode.Utf8View.init(input) catch return error.InvalidUtf8;
    var decomposed: Codepoints = .empty;
    defer decomposed.deinit(allocator);

    var iterator = decoded.iterator();
    while (iterator.nextCodepoint()) |codepoint| {
        try append_decomposed(allocator, &decomposed, codepoint);
    }

    try canonical_order(allocator, decomposed.items);
    const normalized_length = canonical_compose(decomposed.items);

    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(allocator);
    for (decomposed.items[0..normalized_length]) |codepoint| {
        var encoded: [4]u8 = undefined;
        const length = std.unicode.utf8Encode(codepoint, &encoded) catch unreachable;
        try output.appendSlice(allocator, encoded[0..length]);
    }
    return output.toOwnedSlice(allocator);
}

fn append_decomposed(allocator: Allocator, output: *Codepoints, codepoint: u21) Allocator.Error!void {
    if (try hangul_decompose(allocator, output, codepoint)) return;
    if (find_decomposition(codepoint)) |entry| {
        const mapping = tables.DecompositionData[entry.offset .. @as(usize, entry.offset) + entry.len];
        for (mapping) |part| try append_decomposed(allocator, output, part);
    } else {
        try output.append(allocator, codepoint);
    }
}

fn find_decomposition(codepoint: u21) ?tables.decomposition_t {
    var lo: usize = 0;
    var hi: usize = tables.Decompositions.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const entry = tables.Decompositions[mid];
        if (codepoint < entry.codepoint) {
            hi = mid;
        } else if (codepoint > entry.codepoint) {
            lo = mid + 1;
        } else return entry;
    }
    return null;
}

fn combining_class(codepoint: u21) u8 {
    var lo: usize = 0;
    var hi: usize = tables.CombiningClassRanges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const range = tables.CombiningClassRanges[mid];
        if (codepoint < range.first) {
            hi = mid;
        } else if (codepoint > range.last) {
            lo = mid + 1;
        } else return range.value;
    }
    return 0;
}

fn canonical_order(allocator: Allocator, codepoints: []u21) Allocator.Error!void {
    const insertion_limit = 24;
    var scratch: ?[]u21 = null;
    defer if (scratch) |buffer| allocator.free(buffer);

    var index: usize = 0;
    while (index < codepoints.len) {
        if (combining_class(codepoints[index]) == 0) index += 1;
        const marks_start = index;
        while (index < codepoints.len and combining_class(codepoints[index]) != 0) : (index += 1) {}
        const marks_len = index - marks_start;
        if (marks_len < 2) continue;

        if (marks_len <= insertion_limit) {
            stable_insertion_order(codepoints[marks_start..index]);
        } else {
            if (scratch == null) scratch = try allocator.alloc(u21, codepoints.len);
            stable_counting_order(codepoints[marks_start..index], scratch.?[0..marks_len]);
        }
    }
}

fn stable_insertion_order(marks: []u21) void {
    var index: usize = 1;
    while (index < marks.len) : (index += 1) {
        const current_class = combining_class(marks[index]);
        var position = index;
        while (position > 0 and combining_class(marks[position - 1]) > current_class) : (position -= 1) {
            std.mem.swap(u21, &marks[position - 1], &marks[position]);
        }
    }
}

fn stable_counting_order(marks: []u21, scratch: []u21) void {
    var counts: [256]usize = undefined;
    @memset(&counts, 0);
    for (marks) |codepoint| counts[combining_class(codepoint)] += 1;

    var positions: [256]usize = undefined;
    var position: usize = 0;
    for (counts, 0..) |count, class| {
        positions[class] = position;
        position += count;
    }

    for (marks) |codepoint| {
        const class = combining_class(codepoint);
        scratch[positions[class]] = codepoint;
        positions[class] += 1;
    }
    @memcpy(marks, scratch);
}

fn canonical_compose(codepoints: []u21) usize {
    if (codepoints.len == 0) return 0;
    var output_length: usize = 1;
    var starter_position: usize = 0;
    var starter = codepoints[0];
    var last_class: u8 = 0;

    for (codepoints[1..]) |codepoint| {
        const current_class = combining_class(codepoint);
        if (combining_class(starter) == 0 and (last_class == 0 or last_class < current_class)) {
            if (compose_pair(starter, codepoint)) |composed| {
                codepoints[starter_position] = composed;
                starter = composed;
                continue;
            }
        }

        if (current_class == 0) {
            starter_position = output_length;
            starter = codepoint;
        }
        codepoints[output_length] = codepoint;
        output_length += 1;
        last_class = current_class;
    }
    return output_length;
}

fn compose_pair(first: u21, second: u21) ?u21 {
    if (hangul_compose(first, second)) |result| return result;
    var lo: usize = 0;
    var hi: usize = tables.Compositions.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const entry = tables.Compositions[mid];
        if (first < entry.first or (first == entry.first and second < entry.second)) {
            hi = mid;
        } else if (first > entry.first or second > entry.second) {
            lo = mid + 1;
        } else return entry.composed;
    }
    return null;
}

fn hangul_decompose(allocator: Allocator, output: *Codepoints, codepoint: u21) Allocator.Error!bool {
    const s_base: u21 = 0xAC00;
    const l_base: u21 = 0x1100;
    const v_base: u21 = 0x1161;
    const t_base: u21 = 0x11A7;
    const l_count: u21 = 19;
    const v_count: u21 = 21;
    const t_count: u21 = 28;
    const n_count = v_count * t_count;
    const s_count = l_count * n_count;
    if (codepoint < s_base or codepoint >= s_base + s_count) return false;
    const index = codepoint - s_base;
    try output.append(allocator, l_base + index / n_count);
    try output.append(allocator, v_base + (index % n_count) / t_count);
    const trailing = index % t_count;
    if (trailing != 0) try output.append(allocator, t_base + trailing);
    return true;
}

fn hangul_compose(first: u21, second: u21) ?u21 {
    const l_base: u21 = 0x1100;
    const v_base: u21 = 0x1161;
    const t_base: u21 = 0x11A7;
    const s_base: u21 = 0xAC00;
    const l_count: u21 = 19;
    const v_count: u21 = 21;
    const t_count: u21 = 28;
    const n_count = v_count * t_count;
    const s_count = l_count * n_count;
    if (first >= l_base and first < l_base + l_count and second >= v_base and second < v_base + v_count) {
        return s_base + (first - l_base) * n_count + (second - v_base) * t_count;
    }
    if (first >= s_base and first < s_base + s_count and (first - s_base) % t_count == 0 and second > t_base and second < t_base + t_count) {
        return first + second - t_base;
    }
    return null;
}

test "Unicode 9.0.0 version and canonical vectors" {
    try std.testing.expectEqualStrings("9.0.0", UnicodeVersion);
    const allocator = std.testing.allocator;
    const vectors = .{
        .{ "e\u{0301}", "é" },
        .{ "A\u{030A}\u{0301}", "Ǻ" },
        .{ "\u{212B}", "Å" },
        .{ "\u{1100}\u{1161}\u{11A8}", "각" },
        .{ "\u{1E0A}\u{0323}", "Ḍ\u{0307}" },
        .{ "\u{0344}", "\u{0308}\u{0301}" },
    };
    inline for (vectors) |vector| {
        const actual = try normalize(allocator, vector[0]);
        defer allocator.free(actual);
        try std.testing.expectEqualStrings(vector[1], actual);
    }
}

test "empty input and valid already-normalized text" {
    const allocator = std.testing.allocator;
    const empty = try normalize(allocator, "");
    defer allocator.free(empty);
    try std.testing.expectEqualStrings("", empty);
    const value = try normalize(allocator, "plain text 😀");
    defer allocator.free(value);
    try std.testing.expectEqualStrings("plain text 😀", value);
}

test "invalid UTF-8 is rejected" {
    const result = normalize(std.testing.allocator, &.{ 0xC0, 0xAF });
    try std.testing.expectError(error.InvalidUtf8, result);
}
