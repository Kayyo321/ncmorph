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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, processors

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
            for run_name in ("run-a", "run-b"):
                output_dir = root / run_name
                summary = pack_dataset(
                    parquet_documents(parquet_root),
                    tokenizer,
                    eos_id=9,
                    window_length=4,
                    output_dir=output_dir,
                    epochs=2,
                    seed=42,
                )
                summaries.append(summary)
                outputs.append(output_dir)

            self.assertEqual(summaries[0], summaries[1])
            self.assertEqual(summaries[0]["documents"], 1001)
            self.assertEqual(summaries[0]["training_documents"], 999)
            self.assertEqual(summaries[0]["validation_documents"], 2)
            self.assertEqual(summaries[0]["training_windows_per_epoch"], 999)
            self.assertEqual(summaries[0]["validation_windows"], 2)
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


if __name__ == "__main__":
    unittest.main()
