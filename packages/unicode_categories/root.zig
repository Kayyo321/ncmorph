//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! General-category lookup for Unicode code points, with predicates for the major groups
//! L, M, N, P, S, Z and C. The tables are generated from UnicodeData.txt by
//! tools/gen_unicode_categories.py. Zig 0.16.0's standard library has no such tables.

const std = @import("std");
const tables = @import("tables.zig");
const tables_16 = @import("tables_16.zig");

pub const Category = tables.Category;
pub const UnicodeVersion = tables.Version;
pub const Category16 = tables_16.Category;
pub const UnicodeVersion16 = tables_16.Version;

/// Returns the general category of `cp`. Code points that are not listed are unassigned (Cn).
pub fn general_category(cp: u21) Category {
    var lo: usize = 0;
    var hi: usize = tables.Ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const range = tables.Ranges[mid];
        if (cp < range.first) {
            hi = mid;
        } else if (cp > range.last) {
            lo = mid + 1;
        } else {
            return range.category;
        }
    }
    return .Cn;
}

/// Returns the Unicode 16.0.0 general category of `cp`.
pub fn general_category_16(cp: u21) Category16 {
    var lo: usize = 0;
    var hi: usize = tables_16.Ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const range = tables_16.Ranges[mid];
        if (cp < range.first) {
            hi = mid;
        } else if (cp > range.last) {
            lo = mid + 1;
        } else {
            return range.category;
        }
    }
    return .Cn;
}

/// Unicode 16.0.0 group L: letters (`\p{L}`).
pub fn is_letter_16(cp: u21) bool {
    return switch (general_category_16(cp)) {
        .Lu, .Ll, .Lt, .Lm, .Lo => true,
        else => false,
    };
}

/// Unicode 16.0.0 group N: numbers (`\p{N}`).
pub fn is_number_16(cp: u21) bool {
    return switch (general_category_16(cp)) {
        .Nd, .Nl, .No => true,
        else => false,
    };
}

/// Unicode group L: letters (\p{L}).
pub fn is_letter(cp: u21) bool {
    return switch (general_category(cp)) {
        .Lu, .Ll, .Lt, .Lm, .Lo => true,
        else => false,
    };
}

/// Unicode group M: marks (\p{M}).
pub fn is_mark(cp: u21) bool {
    return switch (general_category(cp)) {
        .Mn, .Mc, .Me => true,
        else => false,
    };
}

/// Unicode group N: numbers (\p{N}).
pub fn is_number(cp: u21) bool {
    return switch (general_category(cp)) {
        .Nd, .Nl, .No => true,
        else => false,
    };
}

/// Unicode group P: punctuation (\p{P}).
pub fn is_punctuation(cp: u21) bool {
    return switch (general_category(cp)) {
        .Pc, .Pd, .Ps, .Pe, .Pi, .Pf, .Po => true,
        else => false,
    };
}

/// Unicode group S: symbols (\p{S}).
pub fn is_symbol(cp: u21) bool {
    return switch (general_category(cp)) {
        .Sm, .Sc, .Sk, .So => true,
        else => false,
    };
}

/// Unicode group Z: separators (\p{Z}).
pub fn is_separator(cp: u21) bool {
    return switch (general_category(cp)) {
        .Zs, .Zl, .Zp => true,
        else => false,
    };
}

/// Unicode group C: other, including controls, format characters, surrogates,
/// private use and unassigned code points (\p{C}).
pub fn is_other(cp: u21) bool {
    return switch (general_category(cp)) {
        .Cc, .Cf, .Cs, .Co, .Cn => true,
        else => false,
    };
}

test "ASCII categories" {
    try std.testing.expectEqual(Category.Lu, general_category('A'));
    try std.testing.expectEqual(Category.Ll, general_category('a'));
    try std.testing.expectEqual(Category.Nd, general_category('7'));
    try std.testing.expectEqual(Category.Zs, general_category(' '));
    try std.testing.expectEqual(Category.Cc, general_category('\n'));
    try std.testing.expectEqual(Category.Po, general_category('!'));
    try std.testing.expectEqual(Category.Sc, general_category('$'));
    try std.testing.expectEqual(Category.Sm, general_category('+'));
    try std.testing.expectEqual(Category.Sk, general_category('^'));
    try std.testing.expectEqual(Category.Ps, general_category('('));
}

test "non-ASCII categories" {
    try std.testing.expectEqual(Category.Ll, general_category(0x00E9)); // e with acute
    try std.testing.expectEqual(Category.Mn, general_category(0x0301)); // combining acute accent
    try std.testing.expectEqual(Category.No, general_category(0x00B2)); // superscript two
    try std.testing.expectEqual(Category.Lo, general_category(0x4E00)); // CJK ideograph
    try std.testing.expectEqual(Category.So, general_category(0x1F600)); // grinning face
    try std.testing.expectEqual(Category.Co, general_category(0xE000)); // private use
    try std.testing.expectEqual(Category.Cs, general_category(0xD800)); // surrogate
    try std.testing.expectEqual(Category.Cn, general_category(0x10FFFF)); // unassigned
    try std.testing.expectEqual(Category.Cn, general_category(0xFFFE)); // noncharacter
}

test "group predicates" {
    try std.testing.expect(is_letter('z'));
    try std.testing.expect(is_letter(0x4E00));
    try std.testing.expect(!is_letter('9'));
    try std.testing.expect(is_number('9'));
    try std.testing.expect(is_mark(0x0301));
    try std.testing.expect(is_punctuation('.'));
    try std.testing.expect(is_symbol('$'));
    try std.testing.expect(is_punctuation('%'));
    try std.testing.expect(is_separator(' '));
    try std.testing.expect(is_other('\t'));
}

test "the table covers every code point in order" {
    try std.testing.expectEqual(@as(u21, 0), tables.Ranges[0].first);
    try std.testing.expectEqual(@as(u21, 0x10FFFF), tables.Ranges[tables.Ranges.len - 1].last);
    for (tables.Ranges, 0..) |range, i| {
        try std.testing.expect(range.first <= range.last);
        if (i > 0) try std.testing.expectEqual(tables.Ranges[i - 1].last + 1, range.first);
    }
}

test "the table records the Unicode version" {
    try std.testing.expectEqualStrings("18.0.0", UnicodeVersion);
}

test "Unicode 16 ranges cover every code point in order" {
    try std.testing.expectEqual(@as(u21, 0), tables_16.Ranges[0].first);
    try std.testing.expectEqual(@as(u21, 0x10FFFF), tables_16.Ranges[tables_16.Ranges.len - 1].last);
    for (tables_16.Ranges, 0..) |range, i| {
        try std.testing.expect(range.first <= range.last);
        if (i > 0) try std.testing.expectEqual(tables_16.Ranges[i - 1].last + 1, range.first);
    }
}

test "Unicode 16 sample categories and version" {
    try std.testing.expectEqualStrings("16.0.0", UnicodeVersion16);
    try std.testing.expectEqual(Category16.Ll, general_category_16(0x00E9));
    try std.testing.expectEqual(Category16.Mn, general_category_16(0x0301));
    try std.testing.expectEqual(Category16.No, general_category_16(0x00B2));
    try std.testing.expectEqual(Category16.Cn, general_category_16(0x0558));
    try std.testing.expect(is_letter_16(0x4E00));
    try std.testing.expect(is_number_16(0x2165));
}
