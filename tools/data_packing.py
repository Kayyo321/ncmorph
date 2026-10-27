#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Pack local OpenWebText parquet documents into deterministic token windows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import mmap
import os
import random
import struct
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator


PyArrowVersion = "26.0.0"
TokenizersVersion = "0.23.3"
ValidationDivisor = 1_000
Seed = 42
TokenFormat = struct.Struct("<I")
WindowIndexFormat = struct.Struct("<Q")
IndexInitializationChunk = 65_536
CheckpointDocuments = 1_000
BatchMaxDocuments = 128
BatchMaxTextBytes = 8 * 1024 * 1024


def check_versions() -> None:
    """Require the corpus reader and tokenizer versions pinned by PROJECT.md."""
    for package, expected in (("pyarrow", PyArrowVersion), ("tokenizers", TokenizersVersion)):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"A1 requires {package}=={expected}; package is missing") from error
        if actual.split("+", maxsplit=1)[0] != expected:
            raise RuntimeError(f"A1 requires {package}=={expected}; found {actual}")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"A1 requires Python 3.13; found {sys.version.split()[0]}")


def parquet_documents(parquet_root: Path, start_document_index: int = 0) -> Iterator[str]:
    """Yield text rows in stable shard and row order from local parquet files."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(f"A1 requires pyarrow=={PyArrowVersion}; package is missing") from error

    parquet_paths = sorted(parquet_root.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet shards found in {parquet_root}")
    skipped_documents = 0
    for parquet_path in parquet_paths:
        parquet_file = parquet.ParquetFile(parquet_path)
        if "text" not in parquet_file.schema_arrow.names:
            raise ValueError(f"{parquet_path} has no 'text' column")
        for batch in parquet_file.iter_batches(columns=["text"]):
            for value in batch.column(0).to_pylist():
                if skipped_documents < start_document_index:
                    skipped_documents += 1
                    continue
                if not isinstance(value, str):
                    raise ValueError(f"{parquet_path} contains a non-text document")
                yield value


def document_batches(
    documents: Iterable[str], max_documents: int, max_text_bytes: int
) -> Iterator[list[str]]:
    """Yield ordered document batches bounded by count and UTF-8 text size."""
    if max_documents <= 0 or max_text_bytes <= 0:
        raise ValueError("document batch limits must be positive")

    batch: list[str] = []
    batch_text_bytes = 0
    pending_document: str | None = None
    source_documents = iter(documents)
    while True:
        if pending_document is None:
            try:
                document = next(source_documents)
            except StopIteration:
                if batch:
                    yield batch
                return
        else:
            document = pending_document
            pending_document = None

        document_text_bytes = len(document.encode("utf-8"))
        if batch and (
            len(batch) >= max_documents
            or batch_text_bytes + document_text_bytes > max_text_bytes
        ):
            pending_document = document
            yield batch
            batch = []
            batch_text_bytes = 0
            continue

        batch.append(document)
        batch_text_bytes += document_text_bytes
        if len(batch) >= max_documents or batch_text_bytes >= max_text_bytes:
            yield batch
            batch = []
            batch_text_bytes = 0


def _encode_document_batch(tokenizer, documents: list[str]):
    """Encode a batch while preserving the tokenizer's single-input padding semantics."""
    padding = tokenizer.padding
    if padding is not None and padding.get("length") is None:
        # Dynamic padding uses the longest item in encode_batch, while encode() sees
        # only one document. Keep the established per-document output in that case.
        return [tokenizer.encode(document) for document in documents]
    return tokenizer.encode_batch(documents)


class window_spool_t:
    """Store fixed-size windows compactly so the full corpus stays off the heap."""

    def __init__(self, file: BinaryIO, window_length: int):
        self.file = file
        self.window_length = window_length
        self.count = 0

    def append(self, window: list[int]) -> None:
        for token_id in window:
            self.file.write(TokenFormat.pack(token_id))
        self.count += 1


class compact_window_order_t:
    """A mutable sequence of window indexes backed by a temporary mmap file."""

    def __init__(self, count: int):
        if count < 0:
            raise ValueError("window count must not be negative")
        self.count = count
        self.file = tempfile.TemporaryFile()
        self.mapping: mmap.mmap | None = None
        if count == 0:
            return
        storage_size = count * WindowIndexFormat.size
        self.file.truncate(storage_size)
        self.mapping = mmap.mmap(self.file.fileno(), storage_size)
        for start in range(0, count, IndexInitializationChunk):
            end = min(start + IndexInitializationChunk, count)
            chunk_count = end - start
            chunk_format = struct.Struct(f"<{chunk_count}Q")
            chunk_format.pack_into(self.mapping, start * WindowIndexFormat.size, *range(start, end))

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> int:
        normalized_index = self._normalize_index(index)
        assert self.mapping is not None
        return WindowIndexFormat.unpack_from(self.mapping, normalized_index * WindowIndexFormat.size)[0]

    def __setitem__(self, index: int, value: int) -> None:
        normalized_index = self._normalize_index(index)
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("window index does not fit an unsigned 64-bit value")
        assert self.mapping is not None
        WindowIndexFormat.pack_into(self.mapping, normalized_index * WindowIndexFormat.size, value)

    def __iter__(self) -> Iterator[int]:
        for index in range(self.count):
            yield self[index]

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("window order indexes must be integers")
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError("window order index out of range")
        return index

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.close()
        self.file.close()

    def __enter__(self) -> compact_window_order_t:
        return self

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.close()


def _write_spooled_epoch(
    spool: window_spool_t, data_path: Path, index_path: Path, order: Iterable[int]
) -> None:
    window_bytes = spool.window_length * TokenFormat.size
    temporary_data_path = data_path.with_name(data_path.name + ".partial")
    temporary_index_path = index_path.with_name(index_path.name + ".partial")
    with temporary_data_path.open("wb") as data_file, temporary_index_path.open(
        "w", encoding="ascii", newline="\n"
    ) as index_file:
        for window_index in order:
            spool.file.seek(window_index * window_bytes)
            packed = spool.file.read(window_bytes)
            if len(packed) != window_bytes:
                raise OSError("temporary window spool ended before the requested window")
            data_file.write(packed)
            index_file.write(hashlib.sha256(packed).hexdigest())
            index_file.write("\n")
        data_file.flush()
        os.fsync(data_file.fileno())
        index_file.flush()
        os.fsync(index_file.fileno())
    temporary_data_path.replace(data_path)
    temporary_index_path.replace(index_path)
    _fsync_directory(data_path.parent)


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_checkpoint(checkpoint_path: Path, checkpoint: dict) -> None:
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".partial")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as checkpoint_file:
        checkpoint_file.write(json.dumps(checkpoint, sort_keys=True) + "\n")
        checkpoint_file.flush()
        os.fsync(checkpoint_file.fileno())
    temporary_path.replace(checkpoint_path)
    _fsync_directory(checkpoint_path.parent)


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def pack_dataset(
    documents: Iterable[str] | Callable[[int], Iterable[str]], tokenizer, eos_id: int, window_length: int,
    output_dir: Path, epochs: int = 1, seed: int = Seed,
    input_identity: object = "provided-iterable",
) -> dict[str, object]:
    """Stream documents to temporary window spools and write packed output files."""
    if window_length <= 0:
        raise ValueError("window length must be positive")
    if not 0 <= eos_id <= 0xFFFFFFFF:
        raise ValueError("EOS id must fit in an unsigned 32-bit token")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    repository_root = Path(__file__).resolve().parent.parent
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir == repository_root or repository_root in resolved_output_dir.parents:
        raise ValueError("packed data output must be outside the ncmorph repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / ".packing-work"
    work_dir.mkdir(exist_ok=True)
    checkpoint_path = work_dir / "checkpoint.json"
    train_spool_path = work_dir / "train.spool"
    validation_spool_path = work_dir / "validation.spool"
    tokenizer_hash = hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest()
    identity = {
        "input": input_identity,
        "tokenizer_sha256": tokenizer_hash,
        "eos_id": eos_id,
        "window_length": window_length,
        "epochs": epochs,
        "seed": seed,
    }
    train_documents = 0
    validation_documents = 0
    train_buffer: list[int] = []
    validation_buffer: list[int] = []
    next_document_index = 0
    if checkpoint_path.exists():
        if input_identity == "provided-iterable":
            raise ValueError("resuming packing requires an explicit input identity")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != identity:
            raise ValueError("packing resume identity does not match inputs, tokenizer, or parameters")
        next_document_index = checkpoint["next_document_index"]
        train_documents = checkpoint["training_documents"]
        validation_documents = checkpoint["validation_documents"]
        train_buffer = checkpoint["training_buffer"]
        validation_buffer = checkpoint["validation_buffer"]
        for path, committed_windows in (
            (train_spool_path, checkpoint["training_windows"]),
            (validation_spool_path, checkpoint["validation_windows"]),
        ):
            expected_size = committed_windows * window_length * TokenFormat.size
            if not path.is_file() or path.stat().st_size < expected_size:
                raise ValueError(f"packing spool is shorter than its committed checkpoint: {path.name}")
            with path.open("r+b") as spool_file:
                spool_file.truncate(expected_size)
    else:
        train_spool_path.open("wb").close()
        validation_spool_path.open("wb").close()
    (output_dir / "summary.json").unlink(missing_ok=True)
    _fsync_directory(output_dir)
    with train_spool_path.open("r+b") as train_file, validation_spool_path.open("r+b") as validation_file:
        train_file.seek(0, 2)
        validation_file.seek(0, 2)
        train_spool = window_spool_t(train_file, window_length)
        validation_spool = window_spool_t(validation_file, window_length)
        train_spool.count = train_file.tell() // (window_length * TokenFormat.size)
        validation_spool.count = validation_file.tell() // (window_length * TokenFormat.size)
        source_documents = iter(documents(next_document_index) if callable(documents) else documents)
        if not callable(documents):
            for _ in range(next_document_index):
                next(source_documents)
        for document_batch in document_batches(
            source_documents, BatchMaxDocuments, BatchMaxTextBytes
        ):
            encoded_batch = _encode_document_batch(tokenizer, document_batch)
            if len(encoded_batch) != len(document_batch):
                raise ValueError("tokenizer batch result count does not match document count")
            for encoding in encoded_batch:
                document_index = next_document_index
                is_validation = document_index % ValidationDivisor == 0
                if is_validation:
                    validation_documents += 1
                    buffer = validation_buffer
                    spool = validation_spool
                else:
                    train_documents += 1
                    buffer = train_buffer
                    spool = train_spool

                for token_id in encoding.ids:
                    if not 0 <= token_id <= 0xFFFFFFFF:
                        raise ValueError(f"token id does not fit an unsigned 32-bit token: {token_id}")
                    buffer.append(token_id)
                    if len(buffer) == window_length:
                        spool.append(buffer)
                        buffer.clear()
                buffer.append(eos_id)
                if len(buffer) == window_length:
                    spool.append(buffer)
                    buffer.clear()
                next_document_index += 1
                if next_document_index % CheckpointDocuments == 0:
                    train_file.flush()
                    validation_file.flush()
                    os.fsync(train_file.fileno())
                    os.fsync(validation_file.fileno())
                    _write_checkpoint(checkpoint_path, {
                        "identity": identity,
                        "next_document_index": next_document_index,
                        "training_documents": train_documents,
                        "validation_documents": validation_documents,
                        "training_windows": train_spool.count,
                        "validation_windows": validation_spool.count,
                        "training_buffer": train_buffer,
                        "validation_buffer": validation_buffer,
                    })

        train_file.flush()
        validation_file.flush()
        os.fsync(train_file.fileno())
        os.fsync(validation_file.fileno())
        completed = {
            "identity": identity,
            "next_document_index": next_document_index,
            "training_documents": train_documents,
            "validation_documents": validation_documents,
            "training_windows": train_spool.count,
            "validation_windows": validation_spool.count,
            "training_buffer": train_buffer,
            "validation_buffer": validation_buffer,
            "complete": True,
        }
        _write_checkpoint(checkpoint_path, completed)

        shuffle_rng = random.Random(seed)
        for epoch in range(epochs):
            with compact_window_order_t(train_spool.count) as order:
                shuffle_rng.shuffle(order)
                _write_spooled_epoch(
                    train_spool,
                    output_dir / f"train_epoch_{epoch:04d}.u32le",
                    output_dir / f"train_epoch_{epoch:04d}.index",
                    order,
                )
        _write_spooled_epoch(
            validation_spool,
            output_dir / "validation.u32le",
            output_dir / "validation.index",
            range(validation_spool.count),
        )

    summary = {
        "identity": identity,
        "documents": train_documents + validation_documents,
        "training_documents": train_documents,
        "validation_documents": validation_documents,
        "training_windows_per_epoch": train_spool.count,
        "validation_windows": validation_spool.count,
        "window_length": window_length,
        "seed": seed,
        "epochs": epochs,
    }
    summary_temporary_path = output_dir / "summary.json.partial"
    with summary_temporary_path.open("w", encoding="utf-8", newline="\n") as summary_file:
        summary_file.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        summary_file.flush()
        os.fsync(summary_file.fileno())
    summary_temporary_path.replace(output_dir / "summary.json")
    _fsync_directory(output_dir)
    checkpoint_path.unlink(missing_ok=True)
    train_spool_path.unlink(missing_ok=True)
    validation_spool_path.unlink(missing_ok=True)
    for partial_path in work_dir.glob("*.partial"):
        partial_path.unlink(missing_ok=True)
    work_dir.rmdir()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--eos-id", type=int, required=True)
    parser.add_argument("--window-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=Seed)
    args = parser.parse_args()
    try:
        check_versions()
        if not args.tokenizer.is_file():
            raise FileNotFoundError(f"tokenizer file not found: {args.tokenizer}")
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        parquet_paths = sorted(args.parquet_dir.glob("*.parquet"))
        input_identity = [_file_identity(path) for path in parquet_paths]
        summary = pack_dataset(
            lambda start: parquet_documents(args.parquet_dir, start),
            tokenizer,
            args.eos_id,
            args.window_length,
            args.output_dir,
            args.epochs,
            args.seed,
            input_identity,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
