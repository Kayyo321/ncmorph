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
import random
import struct
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator


PyArrowVersion = "26.0.0"
TokenizersVersion = "0.23.3"
ValidationDivisor = 1_000
Seed = 42
TokenFormat = struct.Struct("<I")
WindowIndexFormat = struct.Struct("<Q")
IndexInitializationChunk = 65_536


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


def parquet_documents(parquet_root: Path) -> Iterator[str]:
    """Yield text rows in stable shard and row order from local parquet files."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(f"A1 requires pyarrow=={PyArrowVersion}; package is missing") from error

    parquet_paths = sorted(parquet_root.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet shards found in {parquet_root}")
    for parquet_path in parquet_paths:
        parquet_file = parquet.ParquetFile(parquet_path)
        if "text" not in parquet_file.schema_arrow.names:
            raise ValueError(f"{parquet_path} has no 'text' column")
        for batch in parquet_file.iter_batches(columns=["text"]):
            for value in batch.column(0).to_pylist():
                if not isinstance(value, str):
                    raise ValueError(f"{parquet_path} contains a non-text document")
                yield value


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
    with data_path.open("wb") as data_file, index_path.open(
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


def pack_dataset(
    documents: Iterable[str], tokenizer, eos_id: int, window_length: int,
    output_dir: Path, epochs: int = 1, seed: int = Seed,
) -> dict[str, int]:
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
    train_documents = 0
    validation_documents = 0
    train_buffer: list[int] = []
    validation_buffer: list[int] = []
    with tempfile.TemporaryFile() as train_file, tempfile.TemporaryFile() as validation_file:
        train_spool = window_spool_t(train_file, window_length)
        validation_spool = window_spool_t(validation_file, window_length)
        for document_index, document in enumerate(documents):
            is_validation = document_index % ValidationDivisor == 0
            if is_validation:
                validation_documents += 1
                buffer = validation_buffer
                spool = validation_spool
            else:
                train_documents += 1
                buffer = train_buffer
                spool = train_spool

            token_ids = tokenizer.encode(document).ids
            for token_id in token_ids:
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
        "documents": train_documents + validation_documents,
        "training_documents": train_documents,
        "validation_documents": validation_documents,
        "training_windows_per_epoch": train_spool.count,
        "validation_windows": validation_spool.count,
        "window_length": window_length,
        "seed": seed,
        "epochs": epochs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        summary = pack_dataset(
            parquet_documents(args.parquet_dir),
            tokenizer,
            args.eos_id,
            args.window_length,
            args.output_dir,
            args.epochs,
            args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
