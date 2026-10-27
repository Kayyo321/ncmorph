#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Compare the Zig pre-tokenizers with pinned tokenizers on the fixed corpus."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PinnedVersion = "0.23.3"
CorpusPath = Path(__file__).parent / "corpus" / "tokenizer_cases.jsonl"


def check_version() -> None:
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"requires Python 3.13; found {sys.version.split()[0]}")
    try:
        actual = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"requires tokenizers=={PinnedVersion}; package is missing") from error
    if actual.split("+", maxsplit=1)[0] != PinnedVersion:
        raise RuntimeError(f"requires tokenizers=={PinnedVersion}; found {actual}")


def read_corpus(path: Path) -> list[dict[str, str]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line:
            continue
        case = json.loads(line)
        if not isinstance(case, dict) or set(case) != {"name", "text"}:
            raise ValueError(f"{path}:{line_number} must contain exactly name and text")
        if not isinstance(case["name"], str) or not isinstance(case["text"], str):
            raise ValueError(f"{path}:{line_number} has invalid name or text")
        cases.append(case)
    if not cases:
        raise ValueError(f"corpus is empty: {path}")
    return cases


def read_pretokenizer(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["pre_tokenizer"]


def sequence_parts(pre_tokenizer: dict[str, object]) -> list[dict[str, object]]:
    if pre_tokenizer.get("type") != "Sequence":
        raise ValueError("pinned pre-tokenizer must be a Sequence")
    parts = pre_tokenizer.get("pretokenizers")
    if not isinstance(parts, list):
        raise ValueError("pinned pre-tokenizer Sequence has no pretokenizers list")
    return parts


def build_reference(tokenizer_root: Path, corpus_path: Path) -> dict[str, object]:
    from tokenizers import Regex, pre_tokenizers

    qwen_parts = sequence_parts(read_pretokenizer(tokenizer_root / "qwen2.json"))
    qwen3_parts = sequence_parts(read_pretokenizer(tokenizer_root / "qwen3.json"))
    llama3_parts = sequence_parts(read_pretokenizer(tokenizer_root / "llama3.json"))
    llama32_parts = sequence_parts(read_pretokenizer(tokenizer_root / "llama32.json"))
    smollm_parts = sequence_parts(read_pretokenizer(tokenizer_root / "smollm.json"))
    qwen_split = qwen_parts[0]
    qwen3_split = qwen3_parts[0]
    llama3_split = llama3_parts[0]
    llama32_split = llama32_parts[0]
    smollm_digits, smollm_byte_level = smollm_parts
    if qwen_split.get("type") != "Split" or llama3_split.get("type") != "Split":
        raise ValueError("pinned Qwen and Llama 3 pre-tokenizers must start with Split")
    if qwen_split != qwen3_split:
        raise ValueError("pinned Qwen2.5 and Qwen3 Split definitions differ")
    if llama3_split != llama32_split:
        raise ValueError("pinned Llama 3.0 and Llama 3.2 Split definitions differ")
    if smollm_digits.get("type") != "Digits" or smollm_byte_level.get("type") != "ByteLevel":
        raise ValueError("pinned SmolLM2 pre-tokenizer must be Digits followed by ByteLevel")
    qwen_regex = qwen_split["pattern"]["Regex"]
    llama3_regex = llama3_split["pattern"]["Regex"]
    if qwen_split.get("behavior") != "Isolated" or llama3_split.get("behavior") != "Isolated":
        raise ValueError("pinned Qwen and Llama 3 Split behavior must be Isolated")

    matchers = {
        "qwen_split": pre_tokenizers.Split(Regex(qwen_regex), "isolated"),
        "llama3_split": pre_tokenizers.Split(Regex(llama3_regex), "isolated"),
        "gpt2_byte_level": pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=False,
            use_regex=smollm_byte_level["use_regex"],
        ),
        "digits_individual": pre_tokenizers.Digits(individual_digits=True),
        "digits_contiguous": pre_tokenizers.Digits(individual_digits=False),
    }
    result = {}
    for case in read_corpus(corpus_path):
        expected: dict[str, object] = {"text": case["text"]}
        for name, matcher in matchers.items():
            pieces = matcher.pre_tokenize_str(case["text"])
            if name == "gpt2_byte_level":
                expected[name] = [case["text"][start:end] for _, (start, end) in pieces]
            else:
                expected[name] = [piece for piece, _ in pieces]
        result[case["name"]] = expected
    return result


def zig_test_source(reference_path: Path) -> str:
    return f'''const std = @import("std");
const pretokenizers = @import("pretokenizers");

test "T0 pre-tokenizer parity on the fixed corpus" {{
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const test_allocator = arena.allocator();
    const fixture = try std.json.parseFromSliceLeaky(
        std.json.Value,
        test_allocator,
        @embedFile("{reference_path}"),
        .{{}},
    );
    const MatcherCase = struct {{ name: []const u8, matcher: pretokenizers.Matcher }};
    const matchers = [_]MatcherCase{{
        .{{ .name = "qwen_split", .matcher = .qwen_split }},
        .{{ .name = "llama3_split", .matcher = .llama3_split }},
        .{{ .name = "gpt2_byte_level", .matcher = .gpt2_byte_level }},
        .{{ .name = "digits_individual", .matcher = .digits_individual }},
        .{{ .name = "digits_contiguous", .matcher = .digits_contiguous }},
    }};
    var case_iterator = fixture.object.iterator();
    var comparison_count: usize = 0;
    while (case_iterator.next()) |case_entry| {{
        const case = case_entry.value_ptr.object;
        const input = case.get("text").?.string;
        for (matchers) |matcher_case| {{
            const actual = try pretokenizers.split(test_allocator, input, matcher_case.matcher);
            const expected = case.get(matcher_case.name).?.array.items;
            try std.testing.expectEqual(expected.len, actual.len);
            for (expected, actual) |expected_piece, actual_piece| {{
                try std.testing.expectEqualStrings(expected_piece.string, actual_piece);
                comparison_count += 1;
            }}
        }}
    }}
    try std.testing.expect(comparison_count > 0);
}}
'''


def run_zig_parity(repo_root: Path, reference: dict[str, object]) -> None:
    zig = shutil.which("zig")
    if zig is None:
        raise RuntimeError("zig was not found in PATH")
    with tempfile.TemporaryDirectory(prefix="ncmorph-pretokenizers-", dir="/tmp") as temp_name:
        temp_dir = Path(temp_name)
        reference_path = temp_dir / "t0_pieces.json"
        reference_path.write_text(
            json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        zig_test_path = temp_dir / "t0_parity_test.zig"
        zig_test_path.write_text(zig_test_source(reference_path), encoding="utf-8")
        cache_dir = temp_dir / "zig-cache"
        global_cache_dir = temp_dir / "zig-global-cache"
        command = [
            zig,
            "test",
            "--cache-dir",
            str(cache_dir),
            "--global-cache-dir",
            str(global_cache_dir),
            "--dep",
            "pretokenizers",
            f"-Mroot={zig_test_path}",
            "--dep",
            "unicode_categories",
            f"-Mpretokenizers={repo_root / 'packages/pretokenizers/root.zig'}",
            f"-Municode_categories={repo_root / 'packages/unicode_categories/root.zig'}",
        ]
        subprocess.run(command, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CorpusPath)
    args = parser.parse_args()
    try:
        check_version()
        reference = build_reference(args.tokenizer_root, args.corpus)
        run_zig_parity(Path(__file__).resolve().parent.parent, reference)
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"matched {len(reference)} corpus cases against tokenizers=={PinnedVersion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
