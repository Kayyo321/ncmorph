#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Convert the pinned released WMT14 tokenizer under D41's isolated-load procedure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PinnedSize = 645_790_351
PinnedSha256 = "74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702"
SourceRepository = "hazyresearch/ncm-checkpoints"
SourceRevision = "d342ce8aaea58830eef3b8ad473664673e41f692"
SourceFilename = "tokenizer.pt"
SchemaPath = Path(__file__).resolve().parents[1] / "research" / "converter" / "wmt14_tokenizer_keys.md"
RepositoryRoot = Path(__file__).resolve().parents[1]
TensorMap = {
    "bool": ("Bool", "BOOL"),
    "uint8": ("U8", "U8"),
    "int8": ("I8", "I8"),
    "uint16": ("U16", "U16"),
    "int16": ("I16", "I16"),
    "int32": ("I32", "I32"),
    "int64": ("I64", "I64"),
    "float16": ("F16", "F16"),
    "bfloat16": ("Bf16", "BF16"),
    "float32": ("F32", "F32"),
    "float64": ("F64", "F64"),
}
NoticeText = """Released WMT14 tokenizer tensors

Copyright (c) 2025 hermann kumbong

These tensors are licensed under the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


@dataclass(frozen=True)
class tensor_spec_t:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool = True


def load_schema(path: Path = SchemaPath) -> dict[str, tensor_spec_t]:
    """Read the checked-in, read-derived 96-entry WMT14 state-dict schema."""
    schema: dict[str, tensor_spec_t] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("| ") or not line[2:3].isdigit():
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 7:
            raise ValueError(f"invalid WMT14 schema table row at {path}:{line_number}")
        name, kind_text, shape_text, dtype = cells[1], cells[2], cells[4], cells[5]
        shape_values = shape_text.strip("() ")
        try:
            shape = tuple(int(value.strip()) for value in shape_values.split(",") if value.strip())
        except ValueError as error:
            raise ValueError(f"invalid WMT14 shape at {path}:{line_number}: {shape_text}") from error
        kind = "buffer" if "buffer" in kind_text else "parameter"
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError(f"invalid WMT14 tensor shape at {path}:{line_number}: {shape_text}")
        if dtype not in TensorMap:
            raise ValueError(f"unsupported WMT14 dtype at {path}:{line_number}: {dtype}")
        if name in schema:
            raise ValueError(f"duplicate WMT14 state-dict key in schema: {name}")
        schema[name] = tensor_spec_t(
            name, kind, shape, dtype, "requires_grad=False" not in kind_text
        )
    if len(schema) != 96:
        raise ValueError(f"WMT14 schema must have 96 tensor entries; found {len(schema)}")
    parameter_count = sum(spec.kind == "parameter" for spec in schema.values())
    buffer_count = sum(spec.kind == "buffer" for spec in schema.values())
    if (parameter_count, buffer_count) != (63, 33):
        raise ValueError(
            "WMT14 schema entry counts differ from the derivation: "
            f"{parameter_count} parameters and {buffer_count} buffers"
        )
    parameter_scalars = sum(
        _shape_product(spec.shape) for spec in schema.values() if spec.kind == "parameter"
    )
    buffer_scalars = sum(
        _shape_product(spec.shape) for spec in schema.values() if spec.kind == "buffer"
    )
    if (parameter_scalars, buffer_scalars) != (144_002_816, 17_391_616):
        raise ValueError(
            "WMT14 schema scalar counts differ from the derivation: "
            f"{parameter_scalars} parameters and {buffer_scalars} buffers"
        )
    return schema


def _shape_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pinned_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"released tokenizer pickle not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != PinnedSize:
        raise ValueError(f"refusing to load tokenizer.pt with size {actual_size}; expected {PinnedSize}")
    actual_hash = file_sha256(path)
    if actual_hash != PinnedSha256:
        raise ValueError(f"refusing to load tokenizer.pt with SHA-256 {actual_hash}; expected {PinnedSha256}")


def _ensure_mapping_matches_schema(state_dict: Any, schema: dict[str, tensor_spec_t], torch_module) -> None:
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint 'model' entry must be a state-dict mapping")
    expected_names = set(schema)
    actual_names = set(state_dict)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(f"state-dict keys differ; missing={missing}, unexpected={unexpected}")
    for name, spec in schema.items():
        tensor = state_dict[name]
        if not isinstance(tensor, torch_module.Tensor):
            raise ValueError(f"state-dict value is not a tensor: {name}")
        expected_dtype = getattr(torch_module, spec.dtype)
        actual_dtype_name = str(tensor.dtype).removeprefix("torch.")
        if actual_dtype_name not in TensorMap:
            raise ValueError(f"unsupported stored dtype for {name}: {tensor.dtype}")
        allowed_dtype_names = (
            {"float16", "bfloat16", "float32"}
            if spec.dtype == "float32"
            else {spec.dtype}
        )
        if actual_dtype_name not in allowed_dtype_names:
            raise ValueError(f"dtype mismatch for {name}: expected {expected_dtype}, found {tensor.dtype}")
        if tuple(tensor.shape) != spec.shape:
            raise ValueError(f"shape mismatch for {name}: expected {spec.shape}, found {tuple(tensor.shape)}")


def _nested_module(root, parts: tuple[str, ...], torch_module):
    current = root
    for part in parts:
        child = current._modules.get(part)
        if child is None:
            child = torch_module.nn.Module()
            current.add_module(part, child)
        current = child
    return current


def _make_schema_module(schema: dict[str, tensor_spec_t], torch_module, state_dict=None):
    """Build a meta-device module whose registered state matches the derived names/shapes."""
    root = torch_module.nn.Module()
    for name, spec in schema.items():
        *parents, leaf = name.split(".")
        module = _nested_module(root, tuple(parents), torch_module)
        dtype = state_dict[name].dtype if state_dict is not None else getattr(torch_module, spec.dtype)
        meta_tensor = torch_module.empty(spec.shape, dtype=dtype, device="meta")
        if spec.kind == "buffer":
            module.register_buffer(leaf, meta_tensor, persistent=True)
        else:
            module.register_parameter(
                leaf,
                torch_module.nn.Parameter(meta_tensor, requires_grad=spec.requires_grad),
            )
    return root


def validate_state_dict(state_dict: Any, schema: dict[str, tensor_spec_t], torch_module) -> None:
    """Check every derived name, shape and dtype and invoke strict state loading."""
    _ensure_mapping_matches_schema(state_dict, schema, torch_module)
    module = _make_schema_module(schema, torch_module, state_dict)
    module.load_state_dict(state_dict, strict=True, assign=True)


def extract_checkpoint(checkpoint: Path, raw_dir: Path, manifest_path: Path) -> None:
    """Unsafe internal stage: load only after the caller verified the exact pinned file."""
    checkpoint = _outside_repository(checkpoint)
    verify_pinned_file(checkpoint)
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"D41 requires Python 3.13; found {sys.version.split()[0]}")
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("D41 requires torch==2.13.0; package is missing") from error
    if torch_version.split("+", maxsplit=1)[0] != "2.13.0":
        raise RuntimeError(f"D41 requires torch==2.13.0; found {torch_version}")
    import torch

    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("released tokenizer pickle must contain a 'model' mapping")
    state_dict = payload["model"]
    schema = load_schema()
    validate_state_dict(state_dict, schema, torch)

    raw_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (name, spec) in enumerate(schema.items()):
        tensor = state_dict[name].detach().cpu().contiguous()
        stored_dtype = str(tensor.dtype).removeprefix("torch.")
        raw_name = f"tensor-{index:03d}.bin"
        raw_path = raw_dir / raw_name
        raw_path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())
        entries.append(
            {
                "name": name,
                "dtype": TensorMap[stored_dtype][0],
                "dtype_name": TensorMap[stored_dtype][1],
                "shape": list(spec.shape),
                "payload": raw_name,
            }
        )
    manifest_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def _zig_writer_source(entries: list[dict[str, Any]]) -> str:
    tensor_rows = []
    read_rows = []
    for index, entry in enumerate(entries):
        shape_text = ", ".join(str(dimension) for dimension in entry["shape"])
        tensor_rows.append(
            "    .{ .name = " + json.dumps(entry["name"]) +
            f", .dtype = .{entry['dtype']}, .shape = &.{{ {shape_text} }}, .data = payload_{index} }},"
        )
        read_rows.append(
            f'    const payload_{index} = try cwd.readFileAlloc(init.io, '
            + json.dumps(entry["payload"])
            + ", arena, .unlimited);"
        )
    return """//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
const std = @import("std");
const safetensors = @import("safetensors");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const cwd = std.Io.Dir.cwd();
""" + "\n".join(read_rows) + """
    const tensors = [_]safetensors.tensor_data_t{
""" + "\n".join(tensor_rows) + """
    };
    const metadata = [_]safetensors.metadata_entry_t{
        .{ .key = "ncmorph_format", .value = "1" },
        .{ .key = "source_repository", .value = "hazyresearch/ncm-checkpoints" },
        .{ .key = "source_revision", .value = "d342ce8aaea58830eef3b8ad473664673e41f692" },
    };
    const bytes = try safetensors.serialize(arena, &tensors, &metadata);
    const parsed = try safetensors.parse_header(arena, bytes, bytes.len);
    if (parsed.tensors.len != tensors.len) return error.RoundTripMismatch;
    for (parsed.tensors) |tensor_info| {
        var matched = false;
        for (tensors) |source_tensor| {
            if (std.mem.eql(u8, tensor_info.name, source_tensor.name)) {
                const data_start = safetensors.PrefixLen + parsed.header_len + tensor_info.data_begin;
                const data_end = safetensors.PrefixLen + parsed.header_len + tensor_info.data_end;
                if (!std.mem.eql(u8, bytes[data_start..data_end], source_tensor.data)) {
                    return error.RoundTripMismatch;
                }
                matched = true;
                break;
            }
        }
        if (!matched) return error.RoundTripMismatch;
    }
    try cwd.writeFile(init.io, .{ .sub_path = "tokenizer.safetensors", .data = bytes });
}
"""


def write_with_w1(raw_dir: Path, manifest_path: Path, zig_executable: str) -> Path:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = raw_dir / "writer.zig"
    source_path.write_text(_zig_writer_source(entries), encoding="utf-8")
    command = [
        zig_executable,
        "run",
        "--cache-dir",
        str(raw_dir / "zig-cache"),
        "--global-cache-dir",
        str(raw_dir / "zig-global-cache"),
        "--dep",
        "safetensors",
        f"-Mroot={source_path}",
        f"-Msafetensors={RepositoryRoot / 'packages' / 'safetensors' / 'root.zig'}",
    ]
    result = subprocess.run(command, cwd=raw_dir, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout or "Zig safetensors writer failed")
    output = raw_dir / "tokenizer.safetensors"
    if not output.is_file():
        raise RuntimeError("W1 writer completed without producing tokenizer.safetensors")
    return output


def _run_extraction(checkpoint: Path, raw_dir: Path, manifest_path: Path) -> None:
    """Run pickle extraction in a separate process so its tensor heap can be released."""
    extract_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_extract-checkpoint",
        str(checkpoint),
        "--_raw-dir",
        str(raw_dir),
        "--_manifest",
        str(manifest_path),
        "--allow-pickle-load",
    ]
    extraction = subprocess.run(extract_cmd, check=False, capture_output=True, text=True)
    if extraction.returncode:
        raise RuntimeError(extraction.stderr or extraction.stdout or "checkpoint validation failed")


def _outside_repository(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == RepositoryRoot or RepositoryRoot in resolved.parents:
        raise ValueError("converter input/output must be outside the ncmorph repository")
    return resolved


def convert(checkpoint: Path, output_dir: Path, allow_pickle_load: bool, zig_executable: str = "zig") -> None:
    if not allow_pickle_load:
        raise ValueError(
            "pickle loading is disabled; use --allow-pickle-load only in the D41 throwaway, "
            "offline, credential-free environment after owner approval"
        )
    checkpoint = _outside_repository(checkpoint)
    output_dir = _outside_repository(output_dir)
    verify_pinned_file(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ncmorph-wmt14-convert-") as temp_name:
        temp_dir = Path(temp_name)
        raw_dir = temp_dir / "raw"
        manifest_path = temp_dir / "manifest.json"
        _run_extraction(checkpoint, raw_dir, manifest_path)
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        safetensors_path = write_with_w1(raw_dir, manifest_path, zig_executable)
        output_size = safetensors_path.stat().st_size
        output_hash = file_sha256(safetensors_path)
        shutil.copyfile(safetensors_path, output_dir / "tokenizer.safetensors")

    dtype_counts: dict[str, int] = {}
    for entry in entries:
        dtype_name = entry["dtype_name"]
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    metadata = {
        "format": 1,
        "tokenizer_route": "wmt14",
        "source": {"repository": SourceRepository, "revision": SourceRevision},
        "tokenizer_config": {
            "K": 11,
            "L": 64,
            "level_sizes": [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64],
            "codebook_sizes": [4096] * 11,
            "width": 384,
            "vocabulary_size": 151680,
        },
        "released_pickle": {"filename": SourceFilename, "size_bytes": PinnedSize, "sha256": PinnedSha256},
        "tokenizer": {
            "filename": "tokenizer.safetensors",
            "size_bytes": output_size,
            "sha256": output_hash,
            "tensor_count": len(entries),
            "stored_dtypes": dtype_counts,
            "tensor_dtypes": {entry["name"]: entry["dtype_name"] for entry in entries},
        },
    }
    (output_dir / "ncm.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "NOTICE").write_text(NoticeText, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-pickle-load", action="store_true")
    parser.add_argument("--zig", default="zig")
    parser.add_argument("--_extract-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_raw-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_manifest", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args._extract_checkpoint is not None:
            if args._raw_dir is None or args._manifest is None:
                raise ValueError("internal extraction stage requires raw and manifest paths")
            if not args.allow_pickle_load:
                raise ValueError("internal pickle extraction requires --allow-pickle-load")
            extract_checkpoint(args._extract_checkpoint, args._raw_dir, args._manifest)
        else:
            if args.checkpoint is None or args.output_dir is None:
                raise ValueError("--checkpoint and --output-dir are required")
            convert(args.checkpoint, args.output_dir, args.allow_pickle_load, args.zig)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
