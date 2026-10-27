#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Cross-check the Zig safetensors writer with the pinned Python reader."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]


def zig_emitter_source(output_path: str) -> str:
    return f'''//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
const std = @import("std");
const safetensors = @import("safetensors");

pub fn main(init: std.process.Init) !void {{
    const tensors = [_]safetensors.tensor_data_t{{
        .{{ .name = "zz_u8", .dtype = .U8, .shape = &.{{3}}, .data = &.{{ 0, 127, 255 }} }},
        .{{ .name = "z_half", .dtype = .F16, .shape = &.{{2}}, .data = &.{{ 0, 56, 0, 190 }} }},
        .{{ .name = "q_bf16", .dtype = .Bf16, .shape = &.{{2}}, .data = &.{{ 128, 63, 0, 64 }} }},
        .{{ .name = "z_f32", .dtype = .F32, .shape = &.{{2}}, .data = &.{{ 0, 0, 160, 63, 0, 0, 32, 192 }} }},
        .{{ .name = "a_f32", .dtype = .F32, .shape = &.{{1}}, .data = &.{{ 0, 0, 0, 64 }} }},
    }};
    const metadata = [_]safetensors.metadata_entry_t{{
        .{{ .key = "source_revision", .value = "fixture-123" }},
        .{{ .key = "ncmorph_format", .value = "1" }},
    }};
    const bytes = try safetensors.serialize(init.arena.allocator(), &tensors, &metadata);
    try std.Io.Dir.cwd().writeFile(init.io, .{{ .sub_path = "{output_path}", .data = bytes }});
}}
'''


def check_round_trip(path: Path) -> None:
    payload = path.read_bytes()
    header_length = int.from_bytes(payload[:8], "little")
    assert header_length % 8 == 0
    header_bytes = payload[8 : 8 + header_length]
    header = json.loads(header_bytes)
    assert header_bytes.rstrip(b" ") == json.dumps(header, separators=(",", ":")).encode()
    assert list(header) == [
        "__metadata__",
        "a_f32",
        "z_f32",
        "q_bf16",
        "z_half",
        "zz_u8",
    ]
    assert header["__metadata__"] == {
        "ncmorph_format": "1",
        "source_revision": "fixture-123",
    }

    expected = {
        "a_f32": (torch.float32, (1,), bytes([0, 0, 0, 64])),
        "z_f32": (torch.float32, (2,), bytes([0, 0, 160, 63, 0, 0, 32, 192])),
        "q_bf16": (torch.bfloat16, (2,), bytes([128, 63, 0, 64])),
        "z_half": (torch.float16, (2,), bytes([0, 56, 0, 190])),
        "zz_u8": (torch.uint8, (3,), bytes([0, 127, 255])),
    }
    loaded = load_file(str(path), device="cpu")
    assert loaded.keys() == expected.keys()
    for name, (dtype, shape, data) in expected.items():
        tensor = loaded[name]
        assert tensor.dtype == dtype, (name, tensor.dtype)
        assert tuple(tensor.shape) == shape, (name, tuple(tensor.shape))
        assert tensor.contiguous().view(torch.uint8).numpy().tobytes() == data, name


def main() -> None:
    if safetensors.__version__ != "0.8.0":
        raise SystemExit(f"expected safetensors 0.8.0, found {safetensors.__version__}")
    zig = shutil.which("zig")
    if zig is None:
        raise SystemExit("zig was not found on PATH")

    with tempfile.TemporaryDirectory(prefix="ncmorph-safetensors-writer-") as temporary:
        temp_dir = Path(temporary)
        source_path = temp_dir / "writer_check.zig"
        output_path = temp_dir / "generated.safetensors"
        source_path.write_text(zig_emitter_source(output_path.name), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(temp_dir / "global-cache"))
        result = subprocess.run(
            [
                zig,
                "run",
                "--cache-dir",
                str(temp_dir / "cache"),
                "--global-cache-dir",
                str(temp_dir / "global-cache"),
                "--dep",
                "safetensors",
                f"-Mroot={source_path}",
                f"-Msafetensors={ROOT / 'packages/safetensors/root.zig'}",
            ],
            cwd=temp_dir,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise SystemExit(result.stderr or result.stdout)
        check_round_trip(output_path)
    print("Zig writer round-trips through safetensors 0.8.0: 5 tensors, 4 dtypes, metadata and ordering verified")


if __name__ == "__main__":
    main()
