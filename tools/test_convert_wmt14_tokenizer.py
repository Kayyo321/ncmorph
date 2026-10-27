#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Synthetic safe tests for the WMT14 state schema and W1 writer bridge."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

import convert_wmt14_tokenizer as converter
from convert_wmt14_tokenizer import (
    PinnedSha256,
    PinnedSize,
    convert,
    load_schema,
    tensor_spec_t,
    validate_state_dict,
    verify_pinned_file,
    write_with_w1,
)


class Wmt14ConverterTests(unittest.TestCase):
    def test_schema_matches_the_read_derived_key_inventory(self) -> None:
        schema = load_schema()
        self.assertEqual(len(schema), 96)
        self.assertEqual(sum(spec.kind == "parameter" for spec in schema.values()), 63)
        self.assertEqual(sum(spec.kind == "buffer" for spec in schema.values()), 33)
        self.assertEqual(schema["encoder.wte.weight"].shape, (151680, 384))
        self.assertEqual(schema["vq.codebook_hit_10"].dtype, "int64")
        self.assertFalse(schema["vq.embeddings.10.weight"].requires_grad)

    def test_strict_schema_module_checks_synthetic_mapping(self) -> None:
        schema = {
            "layer.weight": tensor_spec_t("layer.weight", "parameter", (2, 2), "float32"),
            "vq.hits": tensor_spec_t("vq.hits", "buffer", (2,), "int64"),
        }
        tensors = {
            "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16),
            "vq.hits": torch.tensor([7, 9], dtype=torch.int64),
        }
        validate_state_dict(tensors, schema, torch)
        with self.assertRaisesRegex(ValueError, "missing=.*vq.hits"):
            validate_state_dict({"layer.weight": tensors["layer.weight"]}, schema, torch)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_state_dict(
                {**tensors, "vq.hits": torch.zeros(3, dtype=torch.int64)}, schema, torch
            )
        with self.assertRaisesRegex(ValueError, "dtype mismatch.*torch.float64"):
            validate_state_dict(
                {**tensors, "layer.weight": tensors["layer.weight"].to(torch.float64)},
                schema,
                torch,
            )

    def test_w1_bridge_writes_and_zig_reader_checks_synthetic_tensors(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-wmt14-synthetic-") as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            payloads = {
                "weight.bin": bytes([0, 0, 128, 63, 0, 0, 0, 64]),
                "hits.bin": bytes([2, 0, 0, 0, 4, 0, 0, 0]),
            }
            for filename, payload in payloads.items():
                (raw_dir / filename).write_bytes(payload)
            manifest = [
                {
                    "name": "layer.weight",
                    "dtype": "F32",
                    "dtype_name": "F32",
                    "shape": [2],
                    "payload": "weight.bin",
                },
                {
                    "name": "vq.hits",
                    "dtype": "I32",
                    "dtype_name": "I32",
                    "shape": [2],
                    "payload": "hits.bin",
                },
            ]
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = write_with_w1(raw_dir, manifest_path, "zig")
            self.assertTrue(output.is_file())
            with safe_open(output, framework="pt", device="cpu") as file:
                self.assertEqual(file.keys(), ["layer.weight", "vq.hits"])
                self.assertEqual(
                    file.get_tensor("layer.weight").view(torch.uint8).numpy().tobytes(),
                    payloads["weight.bin"],
                )
                self.assertEqual(
                    file.get_tensor("vq.hits").view(torch.uint8).numpy().tobytes(),
                    payloads["hits.bin"],
                )

    def test_pinned_file_guard_rejects_synthetic_file_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokenizer.pt"
            path.write_bytes(b"synthetic placeholder; never unpickled")
            with self.assertRaisesRegex(ValueError, f"expected {PinnedSize}"):
                verify_pinned_file(path)
            self.assertEqual(len(PinnedSha256), 64)

    def test_pickle_load_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "pickle loading is disabled"):
                convert(Path(temporary) / "missing.pt", Path(temporary) / "out", False)

    def test_complete_convert_lifecycle_with_synthetic_source_extraction(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-wmt14-lifecycle-") as temporary:
            root = Path(temporary)
            checkpoint = root / "tokenizer.pt"
            checkpoint.write_bytes(b"synthetic source placeholder; never unpickled")
            output_dir = root / "converted"

            def write_synthetic_extraction(_checkpoint, raw_dir, manifest_path) -> None:
                raw_dir.mkdir(parents=True)
                entries = [
                    {
                        "name": "layer.weight",
                        "dtype": "F32",
                        "dtype_name": "F32",
                        "shape": [2],
                        "payload": "weight.bin",
                    },
                    {
                        "name": "vq.hits",
                        "dtype": "I32",
                        "dtype_name": "I32",
                        "shape": [2],
                        "payload": "hits.bin",
                    },
                ]
                (raw_dir / "weight.bin").write_bytes(bytes([0, 0, 128, 63, 0, 0, 0, 64]))
                (raw_dir / "hits.bin").write_bytes(bytes([2, 0, 0, 0, 4, 0, 0, 0]))
                manifest_path.write_text(json.dumps(entries), encoding="utf-8")

            with (
                patch.object(converter, "verify_pinned_file") as pinned_hash_check,
                patch.object(converter, "_run_extraction", side_effect=write_synthetic_extraction) as pickle_load,
            ):
                convert(checkpoint, output_dir, allow_pickle_load=True)

            pinned_hash_check.assert_called_once_with(checkpoint.resolve())
            pickle_load.assert_called_once()
            safetensors_path = output_dir / "tokenizer.safetensors"
            self.assertTrue(safetensors_path.is_file())
            metadata = json.loads((output_dir / "ncm.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["tokenizer"]["tensor_count"], 2)
            self.assertEqual(metadata["tokenizer"]["stored_dtypes"], {"F32": 1, "I32": 1})
            self.assertEqual(metadata["tokenizer"]["size_bytes"], safetensors_path.stat().st_size)
            self.assertEqual(metadata["tokenizer"]["sha256"], hashlib.sha256(safetensors_path.read_bytes()).hexdigest())
            self.assertTrue((output_dir / "NOTICE").is_file())
            with safe_open(safetensors_path, framework="pt", device="cpu") as file:
                self.assertEqual(file.keys(), ["layer.weight", "vq.hits"])
                self.assertEqual(file.get_tensor("layer.weight").tolist(), [1.0, 2.0])
                self.assertEqual(file.get_tensor("vq.hits").tolist(), [2, 4])


if __name__ == "__main__":
    unittest.main()
