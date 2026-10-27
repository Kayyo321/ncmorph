//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Hand-written pre-tokenizer matchers for the pinned tokenizer pipelines.

const std = @import("std");
const unicode_categories = @import("unicode_categories");

pub const Matcher = enum {
    qwen_split,
    llama3_split,
    gpt2_byte_level,
    digits_individual,
    digits_contiguous,
};

const codepoint_t = struct {
    value: u21,
    end: usize,
};

/// Splits text into the isolated matches produced by a pinned pre-tokenizer.
/// Returned pieces borrow the input bytes. Input must be valid UTF-8.
pub fn split(allocator: std.mem.Allocator, input: []const u8, matcher: Matcher) (std.mem.Allocator.Error || error{InvalidUtf8})![][]const u8 {
    var pieces: std.ArrayList([]const u8) = .empty;
    defer pieces.deinit(allocator);

    var piece_start: usize = 0;
    var scan: usize = 0;
    while (scan < input.len) {
        if (try match_at(input, scan, matcher)) |match_end| {
            if (scan > piece_start) try pieces.append(allocator, input[piece_start..scan]);
            try pieces.append(allocator, input[scan..match_end]);
            piece_start = match_end;
            scan = match_end;
        } else {
            scan = (try codepoint_at(input, scan)).end;
        }
    }
    if (piece_start < input.len) try pieces.append(allocator, input[piece_start..]);
    return try pieces.toOwnedSlice(allocator);
}

fn codepoint_at(input: []const u8, start: usize) error{InvalidUtf8}!codepoint_t {
    if (start >= input.len) return error.InvalidUtf8;
    const width = std.unicode.utf8ByteSequenceLength(input[start]) catch return error.InvalidUtf8;
    const end = start + width;
    if (end > input.len) return error.InvalidUtf8;
    const value = std.unicode.utf8Decode(input[start..end]) catch return error.InvalidUtf8;
    return .{ .value = value, .end = end };
}

fn is_space(value: u21) bool {
    return (value >= 0x0009 and value <= 0x000D) or value == 0x0085 or switch (unicode_categories.general_category_16(value)) {
        .Zs, .Zl, .Zp => true,
        else => false,
    };
}

fn is_newline(value: u21) bool {
    return value == '\r' or value == '\n';
}

fn is_letter_or_number(value: u21) bool {
    return unicode_categories.is_letter_16(value) or unicode_categories.is_number_16(value);
}

fn is_ascii_letter(value: u8) u8 {
    return if (value >= 'A' and value <= 'Z') value + ('a' - 'A') else value;
}

fn contraction_end(input: []const u8, start: usize, case_insensitive: bool) ?usize {
    const suffixes = [_][]const u8{ "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    for (suffixes) |suffix| {
        var cursor = start;
        var matches = true;
        for (suffix) |expected| {
            if (cursor >= input.len) {
                matches = false;
                break;
            }

            // Unicode simple case folding maps U+017F LONG S to ASCII s.
            // The two-byte UTF-8 sequence must advance as one folded character.
            if (case_insensitive and expected == 's' and
                input.len - cursor >= 2 and input[cursor] == 0xC5 and input[cursor + 1] == 0xBF)
            {
                cursor += 2;
                continue;
            }

            const actual = input[cursor];
            const matches_char = actual < 0x80 and
                (if (case_insensitive) is_ascii_letter(actual) == expected else actual == expected);
            if (!matches_char) {
                matches = false;
                break;
            }
            cursor += 1;
        }
        if (matches) return cursor;
    }
    return null;
}

fn scan_run(input: []const u8, start: usize, predicate: *const fn (u21) bool) error{InvalidUtf8}!usize {
    var cursor = start;
    while (cursor < input.len) {
        const cp = try codepoint_at(input, cursor);
        if (!predicate(cp.value)) break;
        cursor = cp.end;
    }
    return cursor;
}

fn is_letter(value: u21) bool {
    return unicode_categories.is_letter_16(value);
}

fn is_number(value: u21) bool {
    return unicode_categories.is_number_16(value);
}

fn is_nonword_nonspace(value: u21) bool {
    return !is_space(value) and !is_letter_or_number(value);
}

fn match_qwen_or_llama3(input: []const u8, start: usize, digits_max: usize) error{InvalidUtf8}!?usize {
    if (contraction_end(input, start, true)) |end| return end;

    // [^\r\n\p{L}\p{N}]?\p{L}+
    var letter_start = start;
    const first = try codepoint_at(input, start);
    if (!is_newline(first.value) and !is_letter_or_number(first.value)) letter_start = first.end;
    const letters_end = try scan_run(input, letter_start, is_letter);
    if (letters_end > letter_start) return letters_end;

    // Qwen uses one numeric code point per match; Llama 3 uses one to three.
    if (is_number(first.value)) {
        var end = first.end;
        var count: usize = 1;
        while (count < digits_max and end < input.len) {
            const cp = try codepoint_at(input, end);
            if (!is_number(cp.value)) break;
            end = cp.end;
            count += 1;
        }
        return end;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    var punctuation_start = start;
    if (first.value == ' ' and first.end < input.len) punctuation_start = first.end;
    const punctuation_end = try scan_run(input, punctuation_start, is_nonword_nonspace);
    if (punctuation_end > punctuation_start) {
        var end = punctuation_end;
        while (end < input.len) {
            const cp = try codepoint_at(input, end);
            if (!is_newline(cp.value)) break;
            end = cp.end;
        }
        return end;
    }

    // \s*[\r\n]+. Greedy \s* backtracks to the final adjacent CR/LF run.
    if (is_space(first.value)) {
        var whitespace_end = start;
        var last_newline_end: ?usize = null;
        var previous_was_newline = false;
        while (whitespace_end < input.len) {
            const cp = try codepoint_at(input, whitespace_end);
            if (!is_space(cp.value)) break;
            if (is_newline(cp.value)) {
                last_newline_end = cp.end;
                previous_was_newline = true;
            } else if (previous_was_newline) {
                // A later newline after non-newline whitespace starts a new run.
                previous_was_newline = false;
            }
            whitespace_end = cp.end;
        }
        if (last_newline_end) |end| return end;

        // \s+(?!\S), including its greedy backtrack by one code point when
        // non-whitespace text follows the run.
        if (whitespace_end == input.len) return whitespace_end;
        if (whitespace_end > start) {
            var end = start;
            while (end < whitespace_end) {
                const cp = try codepoint_at(input, end);
                if (cp.end == whitespace_end) break;
                end = cp.end;
            }
            if (end > start) return end;
        }

        // Final alternative: \s+.
        return whitespace_end;
    }

    return null;
}

fn match_gpt2(input: []const u8, start: usize) error{InvalidUtf8}!?usize {
    if (contraction_end(input, start, false)) |end| return end;

    const first = try codepoint_at(input, start);
    var word_start = start;
    if (first.value == ' ' and first.end < input.len) word_start = first.end;
    const letters_end = try scan_run(input, word_start, is_letter);
    if (letters_end > word_start) return letters_end;
    const numbers_end = try scan_run(input, word_start, is_number);
    if (numbers_end > word_start) return numbers_end;
    const punctuation_end = try scan_run(input, word_start, is_nonword_nonspace);
    if (punctuation_end > word_start) return punctuation_end;

    // \s+(?!\S) matches trailing whitespace, or all but the final whitespace
    // code point before following non-whitespace text.
    if (is_space(first.value)) {
        const whitespace_end = try scan_run(input, start, is_space);
        if (whitespace_end == input.len) return whitespace_end;
        var end = start;
        while (end < whitespace_end) {
            const cp = try codepoint_at(input, end);
            if (cp.end == whitespace_end) break;
            end = cp.end;
        }
        if (end > start) return end;
        return whitespace_end;
    }
    return null;
}

fn match_digits(input: []const u8, start: usize, individual: bool) error{InvalidUtf8}!?usize {
    const first = try codepoint_at(input, start);
    if (!is_number(first.value)) return null;
    return if (individual) first.end else try scan_run(input, start, is_number);
}

fn match_at(input: []const u8, start: usize, matcher: Matcher) error{InvalidUtf8}!?usize {
    return switch (matcher) {
        .qwen_split => try match_qwen_or_llama3(input, start, 1),
        .llama3_split => try match_qwen_or_llama3(input, start, 3),
        .gpt2_byte_level => try match_gpt2(input, start),
        .digits_individual => try match_digits(input, start, true),
        .digits_contiguous => try match_digits(input, start, false),
    };
}

fn expect_pieces(expected: []const []const u8, actual: []const []const u8) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_piece, actual_piece| {
        try std.testing.expectEqualStrings(expected_piece, actual_piece);
    }
}

test "Qwen and Llama 3 split behavior uses pinned Unicode categories" {
    const allocator = std.testing.allocator;
    const qwen = try split(allocator, "ab 123,γ e\u{301}\n  !", .qwen_split);
    defer allocator.free(qwen);
    try expect_pieces(&.{ "ab", " ", "1", "2", "3", ",γ", " e", "\u{301}\n", " ", " !" }, qwen);

    const llama3 = try split(allocator, "ab 1234,γ", .llama3_split);
    defer allocator.free(llama3);
    try expect_pieces(&.{ "ab", " ", "123", "4", ",γ" }, llama3);
}

test "GPT-2 byte-level regex splits words, numbers, punctuation and whitespace" {
    const allocator = std.testing.allocator;
    const pieces = try split(allocator, "I'm 12🙂  x\t", .gpt2_byte_level);
    defer allocator.free(pieces);
    try expect_pieces(&.{ "I", "'m", " 12", "🙂", " ", " x", "\t" }, pieces);
}

test "case-insensitive contractions fold Unicode long s by code point" {
    const allocator = std.testing.allocator;
    const input = "A'ſoo";
    const qwen = try split(allocator, input, .qwen_split);
    defer allocator.free(qwen);
    try expect_pieces(&.{ "A", "'ſ", "oo" }, qwen);

    const llama3 = try split(allocator, input, .llama3_split);
    defer allocator.free(llama3);
    try expect_pieces(&.{ "A", "'ſ", "oo" }, llama3);
}

test "Digits matcher separates Unicode numeric code points" {
    const allocator = std.testing.allocator;
    const individual = try split(allocator, "a12½Ⅻz", .digits_individual);
    defer allocator.free(individual);
    try expect_pieces(&.{ "a", "1", "2", "½", "Ⅻ", "z" }, individual);

    const contiguous = try split(allocator, "a12½Ⅻz", .digits_contiguous);
    defer allocator.free(contiguous);
    try expect_pieces(&.{ "a", "12½Ⅻ", "z" }, contiguous);
}

test "empty input and invalid UTF-8" {
    const allocator = std.testing.allocator;
    const empty = try split(allocator, "", .qwen_split);
    defer allocator.free(empty);
    try std.testing.expectEqual(@as(usize, 0), empty.len);
    try std.testing.expectError(error.InvalidUtf8, split(allocator, &.{0xFF}, .qwen_split));
}
