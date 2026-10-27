#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Synthetic parquet tests for deterministic OpenWebText packing."""

from __future__ import annotations

import hashlib
import json
import random
import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, processors

import data_packing
from data_packing import compact_window_order_t, pack_dataset, parquet_documents


def make_test_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2, "document": 3, "body": 4, "[BOS]": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", 5)],
    )
    return tokenizer


class serial_encoding_tokenizer_t:
    """Expose the same tokenizer while producing batch results with encode()."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.padding = tokenizer.padding

    def to_str(self) -> str:
        return self.tokenizer.to_str()

    def encode(self, document: str):
        return self.tokenizer.encode(document)

    def encode_batch(self, documents: list[str]):
        return [self.tokenizer.encode(document) for document in documents]


class DataPackingTests(unittest.TestCase):
    def test_compact_window_order_matches_list_shuffle(self) -> None:
        count = 100_000
        expected_order = list(range(count))
        random.Random(42).shuffle(expected_order)

        with compact_window_order_t(count) as compact_order:
            random.Random(42).shuffle(compact_order)
            self.assertEqual(len(compact_order), count)
            self.assertEqual(list(compact_order), expected_order)

        with compact_window_order_t(0) as empty_order:
            random.Random(42).shuffle(empty_order)
            self.assertEqual(list(empty_order), [])

    def test_synthetic_parquet_split_postprocessing_and_reproducibility(self) -> None:
        documents = ["hello world" if index % 2 == 0 else "document body" for index in range(1001)]
        tokenizer = make_test_tokenizer()
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            parquet_root = root / "parquet"
            parquet_root.mkdir()
            pq.write_table(pa.table({"text": documents}), parquet_root / "part-000.parquet")

            summaries = []
            outputs = []
            input_identity = {"synthetic_documents_sha256": hashlib.sha256(
                "\n".join(documents).encode("utf-8")
            ).hexdigest()}
            run_tokenizers = (
                ("run-a", tokenizer),
                ("run-b", tokenizer),
                ("run-serial", serial_encoding_tokenizer_t(tokenizer)),
            )
            for run_name, run_tokenizer in run_tokenizers:
                output_dir = root / run_name
                summary = pack_dataset(
                    parquet_documents(parquet_root),
                    run_tokenizer,
                    eos_id=9,
                    window_length=4,
                    output_dir=output_dir,
                    epochs=2,
                    seed=42,
                    input_identity=input_identity,
                )
                summaries.append(summary)
                outputs.append(output_dir)

            self.assertEqual(summaries[0], summaries[1])
            self.assertEqual(summaries[0], summaries[2])
            self.assertEqual(summaries[0]["documents"], 1001)
            self.assertEqual(summaries[0]["training_documents"], 999)
            self.assertEqual(summaries[0]["validation_documents"], 2)
            self.assertEqual(summaries[0]["training_windows_per_epoch"], 999)
            self.assertEqual(summaries[0]["validation_windows"], 2)
            self.assertEqual(
                summaries[0]["identity"],
                {
                    "input": input_identity,
                    "tokenizer_sha256": hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest(),
                    "eos_id": 9,
                    "window_length": 4,
                    "epochs": 2,
                    "seed": 42,
                },
            )
            self.assertEqual(
                tokenizer.encode("hello world").ids,
                [5, 1, 2],
                "the post-processor must be included in document tokenization",
            )

            filenames = (
                "train_epoch_0000.u32le",
                "train_epoch_0000.index",
                "train_epoch_0001.u32le",
                "train_epoch_0001.index",
                "validation.u32le",
                "validation.index",
                "summary.json",
            )
            for filename in filenames:
                self.assertEqual(
                    (outputs[0] / filename).read_bytes(),
                    (outputs[1] / filename).read_bytes(),
                    filename,
                )
                self.assertEqual(
                    (outputs[0] / filename).read_bytes(),
                    (outputs[2] / filename).read_bytes(),
                    filename,
                )
            self.assertNotEqual(
                (outputs[0] / "train_epoch_0000.u32le").read_bytes(),
                (outputs[0] / "train_epoch_0001.u32le").read_bytes(),
                "the seeded shuffle advances between epochs",
            )

            hashes = (outputs[0] / "train_epoch_0000.index").read_text().splitlines()
            self.assertTrue(all(len(value) == 64 for value in hashes))
            raw_data = (outputs[0] / "train_epoch_0000.u32le").read_bytes()
            self.assertEqual(len(raw_data), len(hashes) * 4 * struct.calcsize("<I"))
            first_window = struct.unpack("<4I", raw_data[:16])
            packed_window = struct.pack("<4I", *first_window)
            self.assertEqual(hashes[0], hashlib.sha256(packed_window).hexdigest())
            self.assertEqual(json.loads((outputs[0] / "summary.json").read_text()), summaries[0])

    def test_document_batches_bound_count_and_utf8_bytes(self) -> None:
        batches = list(data_packing.document_batches(
            ["a", "é", "abc", "1234567", "z"], max_documents=2, max_text_bytes=5
        ))
        self.assertEqual(batches, [["a", "é"], ["abc"], ["1234567"], ["z"]])
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            text_bytes = sum(len(document.encode("utf-8")) for document in batch)
            self.assertTrue(text_bytes <= 5 or (len(batch) == 1 and text_bytes > 5))

    def test_dynamic_batch_padding_preserves_single_document_ids(self) -> None:
        tokenizer = make_test_tokenizer()
        tokenizer.enable_padding(pad_id=0, pad_token="[UNK]")
        documents = ["hello world", "hello"]
        serial_ids = [tokenizer.encode(document).ids for document in documents]
        parallel_ids = [encoding.ids for encoding in tokenizer.encode_batch(documents)]
        self.assertNotEqual(parallel_ids, serial_ids)
        encoded = data_packing._encode_document_batch(tokenizer, documents)
        self.assertEqual([encoding.ids for encoding in encoded], serial_ids)

    def test_interrupted_resume_matches_uninterrupted_bytes(self) -> None:
        documents = ["hello world" if index % 2 == 0 else "document body" for index in range(2005)]
        tokenizer = make_test_tokenizer()
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "resumed"

            original_encode_batch = data_packing._encode_document_batch
            encoded_batch_count = 0

            class interrupted_encodings_t(list):
                def __iter__(self):
                    for index, encoding in enumerate(super().__iter__()):
                        if index == 20:
                            raise RuntimeError("simulated interruption inside encoded batch")
                        yield encoding

            def interrupt_inside_batch(current_tokenizer, batch_documents):
                nonlocal encoded_batch_count
                encoded_batch_count += 1
                encodings = original_encode_batch(current_tokenizer, batch_documents)
                if encoded_batch_count == 9:
                    return interrupted_encodings_t(encodings)
                return encodings

            with patch.object(data_packing, "_encode_document_batch", interrupt_inside_batch):
                with self.assertRaisesRegex(RuntimeError, "inside encoded batch"):
                    pack_dataset(
                        lambda start_index: iter(documents[start_index:]), tokenizer, 9, 7,
                        output_dir, epochs=2, input_identity={"synthetic": "corpus-v1"},
                    )
            (output_dir / ".packing-work" / "checkpoint.json.partial").write_text("stale")

            resumed_summary = pack_dataset(
                lambda start_index: iter(documents[start_index:]), tokenizer, 9, 7,
                output_dir, epochs=2, input_identity={"synthetic": "corpus-v1"},
            )
            complete_summary = pack_dataset(
                documents, serial_encoding_tokenizer_t(tokenizer), 9, 7, root / "complete", epochs=2,
                input_identity={"synthetic": "corpus-v1"},
            )
            self.assertEqual(resumed_summary, complete_summary)
            self.assertEqual(resumed_summary["identity"]["input"], {"synthetic": "corpus-v1"})
            self.assertEqual(
                resumed_summary["identity"]["tokenizer_sha256"],
                hashlib.sha256(tokenizer.to_str().encode("utf-8")).hexdigest(),
            )
            self.assertFalse((output_dir / ".packing-work").exists())
            for path in sorted((root / "complete").iterdir()):
                self.assertEqual(path.read_bytes(), (output_dir / path.name).read_bytes(), path.name)

    def test_resume_rejects_mismatched_input_identity(self) -> None:
        tokenizer = make_test_tokenizer()
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "packed"

            def interrupted_documents(start_index: int):
                for index in range(start_index, 1500):
                    if index == 1200:
                        raise RuntimeError("simulated interruption")
                    yield "hello world"

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                pack_dataset(
                    interrupted_documents, tokenizer, 9, 4, output_dir,
                    input_identity={"synthetic": "corpus-v1"},
                )
            with self.assertRaisesRegex(ValueError, "resume identity"):
                pack_dataset(
                    lambda start_index: iter(["hello world"]), tokenizer, 9, 4,
                    output_dir, input_identity={"synthetic": "corpus-v2"},
                )

    def test_interruption_before_first_checkpoint_restarts_cleanly(self) -> None:
        documents = ["hello world"] * 337
        tokenizer = make_test_tokenizer()
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "resumed"

            def interrupted_documents(start_index: int):
                for index in range(start_index, len(documents)):
                    if index == 200:
                        raise RuntimeError("simulated early interruption")
                    yield documents[index]

            with self.assertRaisesRegex(RuntimeError, "early interruption"):
                pack_dataset(
                    interrupted_documents, tokenizer, 9, 7, output_dir,
                    input_identity={"synthetic": "early-v1"},
                )
            work_dir = output_dir / ".packing-work"
            self.assertGreater((work_dir / "train.spool").stat().st_size, 0)
            self.assertFalse((work_dir / "checkpoint.json").exists())
            got = pack_dataset(
                lambda start_index: iter(documents[start_index:]), tokenizer, 9, 7,
                output_dir, input_identity={"synthetic": "early-v1"},
            )
            want = pack_dataset(
                documents, tokenizer, 9, 7, root / "complete",
                input_identity={"synthetic": "early-v1"},
            )
            self.assertEqual(got, want)
            for path in sorted((root / "complete").iterdir()):
                self.assertEqual(path.read_bytes(), (output_dir / path.name).read_bytes(), path.name)

    def test_interruption_during_output_rebuilds_from_completed_spools(self) -> None:
        documents = ["hello world"] * 1005
        tokenizer = make_test_tokenizer()
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "resumed"
            original_writer = data_packing._write_spooled_epoch
            write_count = 0

            def interrupted_writer(*args, **kwargs):
                nonlocal write_count
                write_count += 1
                if write_count == 2:
                    raise RuntimeError("simulated output interruption")
                original_writer(*args, **kwargs)

            with patch.object(data_packing, "_write_spooled_epoch", interrupted_writer):
                with self.assertRaisesRegex(RuntimeError, "output interruption"):
                    pack_dataset(
                        documents, tokenizer, 9, 7, output_dir, epochs=2,
                        input_identity={"synthetic": "output-v1"},
                    )
            got = pack_dataset(
                documents, tokenizer, 9, 7, output_dir, epochs=2,
                input_identity={"synthetic": "output-v1"},
            )
            want = pack_dataset(
                documents, tokenizer, 9, 7, root / "complete", epochs=2,
                input_identity={"synthetic": "output-v1"},
            )
            self.assertEqual(got, want)
            for path in sorted((root / "complete").iterdir()):
                self.assertEqual(path.read_bytes(), (output_dir / path.name).read_bytes(), path.name)


if __name__ == "__main__":
    unittest.main()
