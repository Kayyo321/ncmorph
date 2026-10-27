#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Compare the Zig NFC normalizer with the pinned tokenizers NFC normalizer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizer_golden_helper import check_version, read_corpus

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(__file__).resolve().parent / "corpus" / "tokenizer_cases.jsonl"
NFC_DIR = ROOT / "packages" / "nfc"


def zig_byte_array(data: bytes) -> str:
    return "&[_]u8{" + ", ".join(f"0x{value:02X}" for value in data) + "}"


def generate_zig_test(cases: list[dict[str, str]], nfc) -> str:
    lines = [
        "//",
        "// Ko, (Baczynski-Bruce) 2026; Apache 2.0",
        "//",
        'const std = @import("std");',
        'const nfc = @import("root.zig");',
        "",
    ]
    for case in cases:
        expected = nfc.normalize_str(case["text"]).encode("utf-8")
        name = case["name"].replace('"', "'")
        lines.extend(
            [
                f'test "tokenizers NFC: {name}" {{',
                f"    const actual = try nfc.normalize(std.testing.allocator, {zig_byte_array(case['text'].encode('utf-8'))});",
                "    defer std.testing.allocator.free(actual);",
                f"    try std.testing.expectEqualSlices(u8, {zig_byte_array(expected)}, actual);",
                "}",
                "",
            ]
        )
    return "\n".join(lines)


def run(corpus_path: Path, zig: str) -> int:
    check_version()
    from tokenizers import normalizers

    cases = read_corpus(corpus_path)
    cases.append(
        {
            "name": "long_descending_combining_classes",
            "text": "\u00A0" + "\u0315\u0301\u0327\u0334" * 500,
        }
    )
    source = generate_zig_test(cases, normalizers.NFC())
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".zig", prefix=".t0_nfc_", dir=NFC_DIR, delete=False
    ) as test_file:
        test_path = Path(test_file.name)
        test_file.write(source)

    try:
        with tempfile.TemporaryDirectory(prefix="ncmorph-nfc-zig-cache-") as cache_dir:
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
            print(f"tokenizers==0.23.3 NFC parity passed for {len(cases)} inputs, including the stress case")
        return result.returncode
    finally:
        test_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--zig", default="zig")
    args = parser.parse_args()
    try:
        return run(args.corpus, args.zig)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
