#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Exhaustively compare the Zig Unicode 16 lookup with pinned UnicodeData.txt."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "packages" / "unicode_categories"
DEFAULT_SOURCE = Path("/tmp/ncmorph-unicode-16.txt")
EXPECTED_SIZE = 2_175_362
EXPECTED_SHA256 = "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f"

TEST_SOURCE = r'''//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
const std = @import("std");
const categories = @import("root.zig");
const source = @embedFile("UNICODE_DATA_FILE");

fn equal_bytes(left: []const u8, right: []const u8) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_byte, right_byte| {
        if (left_byte != right_byte) return false;
    }
    return true;
}

fn field(line: []const u8, wanted: usize) []const u8 {
    var start: usize = 0;
    var index: usize = 0;
    for (line, 0..) |byte, position| {
        if (byte == ';') {
            if (index == wanted) return line[start..position];
            index += 1;
            start = position + 1;
        }
    }
    return if (index == wanted) line[start..] else "";
}

fn parse_hex(text: []const u8) u21 {
    var result: u21 = 0;
    for (text) |byte| {
        const digit: u21 = if (byte >= '0' and byte <= '9')
            byte - '0'
        else if (byte >= 'A' and byte <= 'F')
            byte - 'A' + 10
        else if (byte >= 'a' and byte <= 'f')
            byte - 'a' + 10
        else
            unreachable;
        result = result * 16 + digit;
    }
    return result;
}

fn category_from_source(text: []const u8) categories.Category16 {
    const enum_info = @typeInfo(categories.Category16).@"enum";
    inline for (enum_info.field_names, 0..) |field_name, field_index| {
        if (equal_bytes(text, field_name)) return @enumFromInt(enum_info.field_values[field_index]);
    }
    unreachable;
}

fn ends_with(text: []const u8, suffix: []const u8) bool {
    return text.len >= suffix.len and equal_bytes(text[text.len - suffix.len ..], suffix);
}

fn compare_source_span(first: u21, last: u21, expected: categories.Category16, cursor: *u32) !void {
    while (cursor.* < first) : (cursor.* += 1) {
        try compare_one(cursor.*, .Cn);
    }

    var codepoint: u32 = first;
    while (true) {
        try compare_one(codepoint, expected);
        if (codepoint == last) break;
        codepoint += 1;
    }
    cursor.* = @as(u32, last) + 1;
}

fn compare_one(codepoint: u32, expected: categories.Category16) !void {
    const actual = categories.general_category_16(@intCast(codepoint));
    if (actual != expected) {
        std.debug.print("category mismatch at U+{X:0>4}: source={s}, Zig={s}\n", .{
            codepoint,
            @tagName(expected),
            @tagName(actual),
        });
        return error.TestExpectedEqual;
    }
}

test "Unicode 16.0.0 lookup matches UnicodeData for every code point" {
    try std.testing.expect(equal_bytes(source[0..5], "0000;"));
    var cursor: u32 = 0;
    var source_index: usize = 0;
    var range_first: ?u21 = null;
    var range_category: ?categories.Category16 = null;

    while (source_index < source.len) {
        var line_end = source_index;
        while (line_end < source.len and source[line_end] != '\n') : (line_end += 1) {}
        const line = source[source_index..line_end];
        source_index = if (line_end < source.len) line_end + 1 else line_end;
        if (line.len == 0) continue;

        const codepoint = parse_hex(field(line, 0));
        const name = field(line, 1);
        const category = category_from_source(field(line, 2));
        if (ends_with(name, ", First>")) {
            if (range_first != null) return error.NestedUnicodeRange;
            range_first = codepoint;
            range_category = category;
        } else if (ends_with(name, ", Last>")) {
            const first = range_first orelse return error.RangeEndWithoutStart;
            const first_category = range_category orelse unreachable;
            if (category != first_category) return error.RangeCategoryMismatch;
            try compare_source_span(first, codepoint, category, &cursor);
            range_first = null;
            range_category = null;
        } else {
            if (range_first != null) return error.UnclosedUnicodeRange;
            try compare_source_span(codepoint, codepoint, category, &cursor);
        }
    }

    if (range_first != null) return error.UnclosedUnicodeRange;
    while (cursor <= 0x10FFFF) : (cursor += 1) {
        try compare_one(cursor, .Cn);
    }
}
'''

def check_source(path: Path) -> None:
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != EXPECTED_SIZE or digest != EXPECTED_SHA256:
        raise ValueError(
            f"Unicode 16 source mismatch: expected {EXPECTED_SIZE} bytes and SHA-256 "
            f"{EXPECTED_SHA256}; got {len(content)} bytes and SHA-256 {digest}"
        )


def run(source_path: Path, zig: str) -> int:
    check_source(source_path)
    source_file = tempfile.NamedTemporaryFile(
        mode="wb", prefix=".unicode16-source-", suffix=".txt", dir=PACKAGE_DIR, delete=False
    )
    test_file = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".unicode16-check-", suffix=".zig", dir=PACKAGE_DIR, delete=False
    )
    source_temp_path = Path(source_file.name)
    test_path = Path(test_file.name)
    try:
        source_file.write(source_path.read_bytes())
        source_file.close()
        test_file.write(TEST_SOURCE.replace("UNICODE_DATA_FILE", source_temp_path.name))
        test_file.close()
        with tempfile.TemporaryDirectory(prefix="ncmorph-unicode16-zig-cache-") as cache_dir:
            environment = os.environ.copy()
            environment["ZIG_GLOBAL_CACHE_DIR"] = cache_dir
            result = subprocess.run(
                [zig, "test", str(test_path)],
                cwd=ROOT,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        print(result.stdout, end="")
        if result.returncode == 0:
            print("matched UnicodeData.txt for all 1,114,112 code points")
        return result.returncode
    finally:
        source_temp_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--zig", default="zig")
    args = parser.parse_args()
    try:
        return run(args.source, args.zig)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

