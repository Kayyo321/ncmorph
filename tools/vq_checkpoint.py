#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Atomic, pickle-free storage for resumable VQ training state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


FormatVersion = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity_hash(identity: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_pointer(root: Path) -> dict[str, Any] | None:
    pointer = root / "latest.json"
    if not pointer.exists():
        return None
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"VQ checkpoint pointer is corrupt: {error}") from error
    if not isinstance(value, dict) or value.get("format") != FormatVersion:
        raise ValueError("VQ checkpoint pointer has an unsupported format")
    return value


def _remove_unpublished_generations(root: Path, pointer: dict[str, Any] | None) -> None:
    """Discard generations left behind when publishing latest.json was interrupted."""
    keep = set()
    if pointer is not None:
        for key in ("generation", "previous_generation"):
            name = pointer.get(key)
            if isinstance(name, str) and name.startswith("generation-"):
                keep.add(name)
    for candidate in root.glob("generation-[0-9]*"):
        if candidate.is_dir() and candidate.name not in keep:
            shutil.rmtree(candidate)


def _validate_generation(root: Path, generation: str,
                        expected_state_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    folder = root / generation
    try:
        state_bytes = (folder / "state.json").read_bytes()
        state_hash = hashlib.sha256(state_bytes).hexdigest()
        if expected_state_sha256 is not None and state_hash != expected_state_sha256:
            raise ValueError(f"VQ checkpoint generation {generation} JSON checksum mismatch")
        metadata = json.loads(state_bytes)
        tensor_path = folder / "state.safetensors"
        actual_hash = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"VQ checkpoint generation {generation} is incomplete or corrupt: {error}") from error
    if metadata.get("format") != FormatVersion:
        raise ValueError(f"VQ checkpoint generation {generation} has an unsupported format")
    if metadata.get("tensor_sha256") != actual_hash:
        raise ValueError(f"VQ checkpoint generation {generation} tensor checksum mismatch")
    try:
        tensors = load_file(str(tensor_path), device="cpu")
    except Exception as error:
        raise ValueError(f"VQ checkpoint generation {generation} safetensors are corrupt: {error}") from error
    if set(tensors) != set(metadata.get("tensor_names", [])):
        raise ValueError(f"VQ checkpoint generation {generation} tensor manifest mismatch")
    return metadata, tensors


def save_generation(root: Path, identity: dict[str, Any], tensors: dict[str, torch.Tensor],
                    scalar_state: dict[str, Any], generation_number: int) -> Path:
    """Publish one complete generation; keep the newest two valid generations."""
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() == Path(__file__).resolve().parent.parent or Path(__file__).resolve().parent.parent in root.resolve().parents:
        raise ValueError("VQ checkpoints must be outside the ncmorph repository")
    previous_pointer = _read_pointer(root)
    _remove_unpublished_generations(root, previous_pointer)
    generation = f"generation-{generation_number:08d}"
    destination = root / generation
    if destination.exists():
        raise FileExistsError(f"VQ checkpoint generation already exists: {destination}")
    tensor_copy = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in tensors.items()}
    with tempfile.TemporaryDirectory(prefix=".vq-checkpoint-", dir=root) as temporary:
        staged = Path(temporary) / generation
        staged.mkdir()
        tensor_path = staged / "state.safetensors"
        save_file(tensor_copy, str(tensor_path))
        tensor_hash = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        metadata = {"format": FormatVersion, "generation": generation,
                    "identity": identity, "identity_sha256": identity_hash(identity),
                    "scalar_state": scalar_state, "tensor_names": sorted(tensor_copy),
                    "tensor_sha256": tensor_hash}
        (staged / "state.json").write_bytes(_canonical_bytes(metadata) + b"\n")
        _fsync_file(tensor_path)
        _fsync_file(staged / "state.json")
        # Read the staged generation through the same validation path used by resume.
        state_hash = hashlib.sha256((staged / "state.json").read_bytes()).hexdigest()
        checked, _ = _validate_generation(Path(temporary), generation, state_hash)
        if checked["identity_sha256"] != identity_hash(identity):
            raise ValueError("staged VQ checkpoint identity validation failed")
        _fsync_directory(staged)
        os.replace(staged, destination)
        _fsync_directory(root)

    previous = previous_pointer.get("generation") if previous_pointer else None
    pointer_value = {"format": FormatVersion, "generation": generation,
                     "state_sha256": state_hash,
                     "previous_generation": previous,
                     "previous_state_sha256": previous_pointer.get("state_sha256") if previous_pointer else None}
    pointer_tmp = root / ".latest.json.tmp"
    with pointer_tmp.open("wb") as stream:
        stream.write(_canonical_bytes(pointer_value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pointer_tmp, root / "latest.json")
    _fsync_directory(root)

    keep = {generation, previous}
    for candidate in root.glob("generation-[0-9]*"):
        if candidate.is_dir() and candidate.name not in keep:
            shutil.rmtree(candidate)
    _fsync_directory(root)
    return destination


def load_latest(root: Path, expected_identity: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    pointer = _read_pointer(root)
    if pointer is None:
        raise FileNotFoundError(f"no resumable VQ checkpoint in {root}")
    generation = pointer.get("generation")
    if not isinstance(generation, str) or not generation.startswith("generation-"):
        raise ValueError("VQ checkpoint pointer has an invalid generation name")
    state_hash = pointer.get("state_sha256")
    if not isinstance(state_hash, str):
        raise ValueError("VQ checkpoint pointer is missing the JSON checksum")
    metadata, tensors = _validate_generation(root, generation, state_hash)
    if metadata.get("generation") != generation:
        raise ValueError("VQ checkpoint generation name does not match its metadata")
    if metadata.get("identity_sha256") != identity_hash(metadata.get("identity", {})):
        raise ValueError("VQ checkpoint identity metadata is corrupt")
    if metadata["identity_sha256"] != identity_hash(expected_identity):
        raise ValueError("VQ checkpoint identity mismatch (model, schedule, or training data changed)")
    return metadata, tensors
