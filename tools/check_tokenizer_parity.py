#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Run the T1/T2 gate against runtime-generated tokenizers==0.23.3 references."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizer_golden_helper import PinnedVersion, TokenizerFiles


CorpusPath = Path(__file__).parent / "corpus" / "tokenizer_cases.jsonl"
DefaultPython = "/tmp/ncmorph-b0-venv/bin/python"
Qwen3MalformedSequences = {
    "incomplete_three_byte": [158, 224],
    "incomplete_four_byte": [172, 253],
    "invalid_surrogate_prefix": [169, 224],
    "invalid_overlong_prefix": [156, 224],
}
DirectDecodeReferenceScript = r'''import importlib.metadata
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

if importlib.metadata.version("tokenizers").split("+", maxsplit=1)[0] != "0.23.3":
    raise RuntimeError("direct decode reference requires tokenizers==0.23.3")

tokenizer_root = Path(sys.argv[1])
reference_path = Path(sys.argv[2])
tokenizer_files = json.loads(sys.argv[3])
qwen3_malformed_sequences = json.loads(sys.argv[4])
reference = json.loads(reference_path.read_text(encoding="utf-8"))
direct_decodes = {}
for family, (file_name, _repository, _revision) in tokenizer_files.items():
    if family == "qwen3":
        cases = [
            {"name": name, "ids": ids}
            for name, ids in qwen3_malformed_sequences.items()
        ]
    elif family in {"llama-2-7b", "mistral-7b-v0.1", "mistral-7b-v0.3"}:
        tokenizer = Tokenizer.from_file(str(tokenizer_root / file_name))
        cases = []
        for name, byte_tokens in {
            "incomplete_three_byte_byte_fallback": ["<0xE2>", "<0x82>"],
            "incomplete_four_byte_byte_fallback": ["<0xF0>", "<0x9F>"],
        }.items():
            ids = [tokenizer.token_to_id(token) for token in byte_tokens]
            if any(token_id is None for token_id in ids):
                raise ValueError(f"{family} is missing required byte-fallback tokens: {byte_tokens}")
            cases.append({"name": name, "ids": ids})
    else:
        continue

    tokenizer = Tokenizer.from_file(str(tokenizer_root / file_name))
    for case in cases:
        ids = case["ids"]
        case["decoded_text"] = tokenizer.decode(ids, skip_special_tokens=True)
        case["decoded_text_with_special_tokens"] = tokenizer.decode(ids, skip_special_tokens=False)
    direct_decodes[family] = {"file_name": file_name, "cases": cases}

reference["direct_decodes"] = direct_decodes
reference_path.write_text(
    json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
'''


def zig_string(value: str) -> str:
    encoded = value.encode("utf-8")
    pieces = []
    for byte in encoded:
        if byte == 0x22:
            pieces.append('\\"')
        elif byte == 0x5C:
            pieces.append("\\\\")
        elif byte == 0x0A:
            pieces.append("\\n")
        elif byte == 0x0D:
            pieces.append("\\r")
        elif byte == 0x09:
            pieces.append("\\t")
        elif 0x20 <= byte < 0x7F:
            pieces.append(chr(byte))
        else:
            pieces.append(f"\\x{byte:02X}")
    return '"' + "".join(pieces) + '"'


def zig_test_source(reference_path: Path, tokenizer_root: Path) -> str:
    family_entries = ",\n".join(
        f"        .{{ .name = {zig_string(family)}, .file_name = {zig_string(filename)} }}"
        for family, (filename, _repository, _revision) in TokenizerFiles.items()
    )
    return f'''const std = @import("std");
const bpe = @import("bpe");

test "T1/T2 tokenizer ids and decoded text match the pinned T0 reference" {{
    const allocator = std.testing.allocator;
    const reference = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        @embedFile({zig_string(str(reference_path))}),
        .{{}},
    );
    defer reference.deinit();
    const tokenizer_dir = try std.Io.Dir.openDirAbsolute(
        std.testing.io,
        {zig_string(str(tokenizer_root))},
        .{{}},
    );
    defer tokenizer_dir.close(std.testing.io);

    const Family = struct {{ name: []const u8, file_name: []const u8 }};
    const families = [_]Family{{
{family_entries}
    }};
    const cases = reference.value.object.get("cases").?.array.items;
    var comparisons: usize = 0;
    for (families) |family| {{
        const tokenizer_json = try tokenizer_dir.readFileAlloc(
            std.testing.io,
            family.file_name,
            allocator,
            .unlimited,
        );
        defer allocator.free(tokenizer_json);
        var tokenizer = try bpe.tokenizer_t.from_json(allocator, tokenizer_json);
        defer tokenizer.deinit();

        for (cases) |case| {{
            const text = case.object.get("text").?.string;
            const actual_ids = try tokenizer.encode(allocator, text);
            const expected_ids_json = case.object.get("ids").?.object.get(family.name).?.array.items;
            const expected_ids = try allocator.alloc(u32, expected_ids_json.len);
            for (expected_ids_json, expected_ids) |expected, *slot| {{
                slot.* = @intCast(expected.integer);
            }}
            defer allocator.free(expected_ids);
            defer allocator.free(actual_ids);
            try std.testing.expectEqualSlices(u32, expected_ids, actual_ids);

            const outputs = case.object.get("tokenizer_outputs").?.object.get(family.name).?.object;
            const expected_decoded = outputs.get("decoded_text").?.string;
            const actual_decoded = try tokenizer.decode(allocator, actual_ids, true);
            defer allocator.free(actual_decoded);
            try std.testing.expectEqualStrings(expected_decoded, actual_decoded);

            const expected_decoded_with_special_tokens =
                outputs.get("decoded_text_with_special_tokens").?.string;
            const actual_decoded_with_special_tokens =
                try tokenizer.decode(allocator, actual_ids, false);
            defer allocator.free(actual_decoded_with_special_tokens);
            try std.testing.expectEqualStrings(
                expected_decoded_with_special_tokens,
                actual_decoded_with_special_tokens,
            );
            comparisons += 1;
        }}
    }}
    try std.testing.expectEqual(families.len * cases.len, comparisons);
}}

test "direct malformed UTF-8 decode matches runtime T0 references" {{
    const allocator = std.testing.allocator;
    const reference = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        @embedFile({zig_string(str(reference_path))}),
        .{{}},
    );
    defer reference.deinit();
    const tokenizer_dir = try std.Io.Dir.openDirAbsolute(
        std.testing.io,
        {zig_string(str(tokenizer_root))},
        .{{}},
    );
    defer tokenizer_dir.close(std.testing.io);

    const direct_decodes = reference.value.object.get("direct_decodes").?.object;
    var families = direct_decodes.iterator();
    var comparisons: usize = 0;
    while (families.next()) |family_entry| {{
        const family = family_entry.value_ptr.object;
        const tokenizer_json = try tokenizer_dir.readFileAlloc(
            std.testing.io,
            family.get("file_name").?.string,
            allocator,
            .unlimited,
        );
        defer allocator.free(tokenizer_json);
        var tokenizer = try bpe.tokenizer_t.from_json(allocator, tokenizer_json);
        defer tokenizer.deinit();

        for (family.get("cases").?.array.items) |case| {{
            const ids_json = case.object.get("ids").?.array.items;
            const ids = try allocator.alloc(u32, ids_json.len);
            defer allocator.free(ids);
            for (ids_json, ids) |expected, *slot| slot.* = @intCast(expected.integer);

            const expected_decoded = case.object.get("decoded_text").?.string;
            const actual_decoded = try tokenizer.decode(allocator, ids, true);
            defer allocator.free(actual_decoded);
            try std.testing.expectEqualStrings(expected_decoded, actual_decoded);

            const expected_decoded_with_special_tokens =
                case.object.get("decoded_text_with_special_tokens").?.string;
            const actual_decoded_with_special_tokens = try tokenizer.decode(allocator, ids, false);
            defer allocator.free(actual_decoded_with_special_tokens);
            try std.testing.expectEqualStrings(
                expected_decoded_with_special_tokens,
                actual_decoded_with_special_tokens,
            );
            comparisons += 1;
        }}
    }}
    try std.testing.expect(comparisons > 0);
}}
'''


def generate_reference(
    python: Path,
    tokenizer_root: Path,
    corpus: Path,
    output: Path,
) -> None:
    helper_path = Path(__file__).with_name("tokenizer_golden_helper.py")
    subprocess.run(
        [
            str(python),
            str(helper_path),
            "--tokenizer-root",
            str(tokenizer_root),
            "--output",
            str(output),
            "--corpus",
            str(corpus),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            DirectDecodeReferenceScript,
            str(tokenizer_root),
            str(output),
            json.dumps(TokenizerFiles),
            json.dumps(Qwen3MalformedSequences),
        ],
        check=True,
    )


def run_zig_parity(repo_root: Path, tokenizer_root: Path, reference_path: Path) -> None:
    zig = shutil.which("zig")
    if zig is None:
        raise RuntimeError("zig was not found in PATH")

    with tempfile.TemporaryDirectory(prefix="ncmorph-tokenizer-parity-", dir="/tmp") as temp_name:
        temp_dir = Path(temp_name)
        local_reference_path = temp_dir / "t0_tokenizers.json"
        shutil.copyfile(reference_path, local_reference_path)
        zig_test_path = temp_dir / "t0_tokenizer_parity_test.zig"
        zig_test_path.write_text(
            zig_test_source(local_reference_path, tokenizer_root),
            encoding="utf-8",
        )
        command = [
            zig,
            "test",
            "--cache-dir",
            str(temp_dir / "zig-cache"),
            "--global-cache-dir",
            str(temp_dir / "zig-global-cache"),
            "--dep",
            "bpe",
            f"-Mroot={zig_test_path}",
            "--dep",
            "pretokenizers",
            "--dep",
            "nfc",
            f"-Mbpe={repo_root / 'packages/bpe/root.zig'}",
            "--dep",
            "unicode_categories",
            f"-Mpretokenizers={repo_root / 'packages/pretokenizers/root.zig'}",
            f"-Mnfc={repo_root / 'packages/nfc/root.zig'}",
            f"-Municode_categories={repo_root / 'packages/unicode_categories/root.zig'}",
        ]
        subprocess.run(command, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("NCMORPH_GOLDEN_PYTHON", DefaultPython)))
    parser.add_argument("--corpus", type=Path, default=CorpusPath)
    args = parser.parse_args()
    try:
        for filename, _repository, _revision in TokenizerFiles.values():
            tokenizer_path = args.tokenizer_root / filename
            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"missing pinned tokenizer file: {tokenizer_path}")
        with tempfile.TemporaryDirectory(prefix="ncmorph-tokenizer-reference-", dir="/tmp") as temp_name:
            reference_path = Path(temp_name) / "t0_tokenizers.json"
            generate_reference(args.python, args.tokenizer_root, args.corpus, reference_path)
            run_zig_parity(Path(__file__).resolve().parent.parent, args.tokenizer_root, reference_path)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 2
    print(
        f"matched ids and decoded text for {len(TokenizerFiles)} tokenizer files "
        f"against runtime-generated tokenizers=={PinnedVersion} references"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
