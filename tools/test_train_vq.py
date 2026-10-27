#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Synthetic end-to-end tests for the trained VQ tokenizer and W1 export."""

from __future__ import annotations

import json
import copy
import hashlib
import os
import random
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

from train_vq import _hash_data, _optimizer_parameter_groups, _train_batch_memory_bounded, train, vqvae_t
from vq_checkpoint import load_latest, save_generation


class TrainVqTests(unittest.TestCase):
    def test_memory_bounded_step_matches_full_batch_ema_loss_and_gradients(self) -> None:
        torch.manual_seed(13)
        random.seed(17)
        full = vqvae_t(vocab_size=19, length=16, width=8, levels=[1, 4, 16],
                       codebook_size=4, heads=2, dropout=0.0)
        bounded = copy.deepcopy(full)
        token_ids = torch.randint(0, 19, (6, 16))
        full.train()
        full_optimizer = torch.optim.AdamW(full.parameters(), lr=0.0)
        full_optimizer.zero_grad(set_to_none=True)
        random.seed(29)
        torch.manual_seed(31)
        _, full_loss, _ = full(token_ids)
        full_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in full.parameters() if p.requires_grad], 1.0)

        bounded.train()
        bounded_optimizer = torch.optim.AdamW(bounded.parameters(), lr=0.0)
        bounded_optimizer.zero_grad(set_to_none=True)
        max_encoder_batch = 0
        max_decoder_batch = 0

        def record_encoder(_module, args):
            nonlocal max_encoder_batch
            max_encoder_batch = max(max_encoder_batch, args[0].shape[0])

        def record_decoder(_module, args):
            nonlocal max_decoder_batch
            max_decoder_batch = max(max_decoder_batch, args[0].shape[0])

        encoder_hook = bounded.encoder.register_forward_pre_hook(record_encoder)
        decoder_hook = bounded.decoder.register_forward_pre_hook(record_decoder)
        random.seed(29)
        torch.manual_seed(31)
        bounded_loss = _train_batch_memory_bounded(
            bounded, bounded_optimizer, [p for p in bounded.parameters() if p.requires_grad],
            token_ids, microbatch_size=2, device="cpu")
        encoder_hook.remove()
        decoder_hook.remove()

        self.assertLessEqual(max_encoder_batch, 2)
        self.assertLessEqual(max_decoder_batch, 2)
        self.assertLessEqual(abs(bounded_loss.item() - full_loss.item()), 1e-4 * max(1.0, abs(full_loss.item())))
        for name, parameter in full.named_parameters():
            actual = dict(bounded.named_parameters())[name]
            if parameter.grad is None:
                self.assertIsNone(actual.grad, name)
            else:
                torch.testing.assert_close(actual.grad, parameter.grad, rtol=1e-4, atol=1e-5, msg=name)
        for name, value in full.state_dict().items():
            if name.startswith("vq.ema_") or name.startswith("vq.embeddings.") or name.startswith("vq.codebook_hit_"):
                torch.testing.assert_close(dict(bounded.state_dict())[name], value, rtol=1e-4, atol=1e-5, msg=name)
        full.eval()
        bounded.eval()
        torch.manual_seed(37)
        full_logits, _, _ = full(token_ids)
        torch.manual_seed(37)
        bounded_logits, _, _ = bounded(token_ids)
        output_scale = max(1.0, full_logits.abs().max().item())
        self.assertLessEqual((full_logits - bounded_logits).abs().max().item(), 1e-4 * output_scale)

    def test_dropout_replay_reuses_encoder_masks_and_advances_decoder_rng(self) -> None:
        torch.manual_seed(41)
        random.seed(43)
        template = vqvae_t(vocab_size=19, length=16, width=8, levels=[1, 4, 16],
                           codebook_size=4, heads=2, dropout=0.3)
        token_ids = torch.randint(0, 19, (4, 16))

        def run_once():
            model = copy.deepcopy(template).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
            outputs = []
            decoder_rng_states = []
            hook = model.encoder.register_forward_hook(
                lambda _module, _args, output: outputs.append(output.detach().clone()))
            decoder_hook = model.decoder.register_forward_pre_hook(
                lambda _module, _args: decoder_rng_states.append(torch.get_rng_state().clone()))
            torch.manual_seed(47)
            random.seed(53)
            loss = _train_batch_memory_bounded(
                model, optimizer, [p for p in model.parameters() if p.requires_grad],
                token_ids, microbatch_size=2, device="cpu")
            after = torch.get_rng_state().clone()
            hook.remove()
            decoder_hook.remove()
            return model, loss, outputs, decoder_rng_states, after

        first, first_loss, outputs, decoder_rng_states, after = run_once()
        second, second_loss, _, _, _ = run_once()
        self.assertEqual(len(outputs), 4)
        for cached, replayed in zip(outputs[:2], outputs[2:], strict=True):
            torch.testing.assert_close(replayed, cached, rtol=0.0, atol=0.0)
        self.assertEqual(len(decoder_rng_states), 2)
        self.assertFalse(torch.equal(decoder_rng_states[0], decoder_rng_states[1]),
                         "decoder chunks must not reuse dropout RNG state")
        self.assertFalse(torch.equal(decoder_rng_states[-1], after),
                         "the final decoder dropout must advance the global RNG")
        self.assertEqual(first_loss.item(), second_loss.item())
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0.0, atol=0.0, msg=name)

    def test_adamw_weight_decay_only_covers_matrices(self) -> None:
        vector = torch.nn.Parameter(torch.ones(8))
        matrix = torch.nn.Parameter(torch.ones(8, 8))
        groups = _optimizer_parameter_groups([vector, matrix])
        self.assertEqual(groups[0]["params"], [matrix])
        self.assertEqual(groups[0]["weight_decay"], 0.01)
        self.assertEqual(groups[1]["params"], [vector])
        self.assertEqual(groups[1]["weight_decay"], 0.0)

    def test_model_exports_reference_attribute_names_and_configured_shapes(self) -> None:
        model = vqvae_t(vocab_size=17, length=16, width=8, levels=[1, 4, 16], codebook_size=4, heads=2)
        state = model.state_dict()
        self.assertEqual(state["encoder.wte.weight"].shape, (17, 8))
        self.assertEqual(state["vq.embeddings.2.weight"].shape, (4, 8))
        self.assertEqual(state["vq.learned_down_l1.0.weight"].shape, (8, 8, 4))
        self.assertEqual(state["decoder.middle_block.attn.q_norm.weight"].shape, (4,))
        self.assertEqual(state["decoder.output_layer.weight"].shape, (17, 8))
        model.train()
        value = model.encoder(torch.tensor([[1, 2, 3, 4] * 4]))
        value = model.pre_quant_norm(value.transpose(1, 2)).transpose(1, 2)
        _, quantization_loss, _ = model.vq(value)
        quantization_loss.backward()
        self.assertIsNotNone(model.vq.learned_down_l1[0].weight.grad)
        self.assertGreater(model.vq.learned_down_l1[0].weight.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(torch.all(getattr(model.vq, f"codebook_hit_{i}") <= 1) for i in range(3)))

        model.vq.zero_grad(set_to_none=True)
        detached_encoder_value = torch.randn(2, 16, 8, requires_grad=True)
        reconstruction, _, _ = model.vq(detached_encoder_value)
        torch.nn.functional.mse_loss(reconstruction, detached_encoder_value.detach()).backward()
        self.assertIsNone(detached_encoder_value.grad)
        self.assertIsNotNone(model.vq.learned_down_l1[0].weight.grad)

    def test_learned_up_l1_uses_reference_layer_names_in_forward_order(self) -> None:
        quantizer = vqvae_t(vocab_size=17, length=16, width=8, levels=[1, 4, 16],
                            codebook_size=4, heads=2).vq
        patterns = [torch.tensor([0.1, 0.2, 0.4, 0.8]), torch.tensor([0.7, 0.3, 0.2, 0.1])]
        with torch.no_grad():
            for layer_index, layer in enumerate(quantizer.learned_up_l1):
                layer.weight.zero_()
                for channel in range(8):
                    layer.weight[channel, channel, :] = patterns[layer_index]
                layer.bias.copy_(torch.linspace(-0.1, 0.1, 8) * (layer_index + 1))

        value = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 8)
        value_bcl = value.transpose(1, 2)
        actual = quantizer._upsample(value, 0)
        natural = quantizer.learned_up_l1[1](quantizer.learned_up_l1[0](value_bcl)).transpose(1, 2)
        reversed_order = quantizer.learned_up_l1[0](quantizer.learned_up_l1[1](value_bcl)).transpose(1, 2)
        torch.testing.assert_close(actual, natural, rtol=0.0, atol=0.0)
        self.assertFalse(torch.allclose(actual, reversed_order), "distinct named layers must expose an order swap")

        actual.square().mean().backward()
        actual_grads = [layer.weight.grad.detach().clone() for layer in quantizer.learned_up_l1]
        quantizer.zero_grad(set_to_none=True)
        natural.square().mean().backward()
        for layer_index, layer in enumerate(quantizer.learned_up_l1):
            torch.testing.assert_close(layer.weight.grad, actual_grads[layer_index], rtol=0.0, atol=0.0)

    def test_vq_ema_stays_float32_under_bfloat16_autocast(self) -> None:
        model = vqvae_t(vocab_size=17, length=16, width=8, levels=[1, 4, 16], codebook_size=4, heads=2)
        model.train()
        value = torch.randn(2, 16, 8)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            reconstruction, loss, _ = model.vq(value)
        self.assertEqual(reconstruction.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertTrue(torch.isfinite(loss))

    def test_short_training_writes_w1_safetensors_and_ncm_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-test-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            packed = data_dir / "train_epoch_0000.u32le"
            packed.write_bytes(struct.pack("<32I", *([index % 13 for index in range(32)])))
            (data_dir / "summary.json").write_text(
                json.dumps({"window_length": 16, "seed": 42, "training_windows_per_epoch": 2, "epochs": 1}),
                encoding="utf-8",
            )
            output_path = train(
                data_dir=data_dir,
                output_dir=output_dir,
                vocab_size=13,
                source_family="synthetic",
                source_repository="synthetic/tokenizer-test",
                source_revision="test-only",
                length=16,
                levels=[1, 4, 16],
                width=8,
                codebook_size=4,
                heads=2,
                batch_size=2,
                updates=1,
                warmup_steps=2,
                max_steps=1,
                device="cpu",
            )
            self.assertTrue(output_path.is_file())
            metadata = json.loads((output_dir / "ncm.json").read_text(encoding="utf-8"))
            expected_names = {entry["name"] for entry in metadata["tensor_schema"]}
            self.assertEqual(metadata["tokenizer_config"]["level_sizes"], [1, 4, 16])
            with safe_open(output_path, framework="pt", device="cpu") as file:
                self.assertEqual(set(file.keys()), expected_names)
                self.assertEqual(file.get_tensor("encoder.wte.weight").shape, (13, 8))
                self.assertEqual(file.get_tensor("vq.embeddings.2.weight").shape, (4, 8))
                self.assertEqual(file.get_tensor("decoder.output_layer.weight").shape, (13, 8))
                self.assertEqual(file.metadata()["source_repository"], "synthetic/tokenizer-test")
                self.assertEqual(file.metadata()["source_revision"], "test-only")

    def test_interrupted_resume_matches_uninterrupted_training_exactly(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-resume-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            packed = data_dir / "train_epoch_0000.u32le"
            packed.write_bytes(struct.pack("<64I", *([index % 13 for index in range(64)])))
            (data_dir / "summary.json").write_text(
                json.dumps({"window_length": 16, "seed": 42, "training_windows_per_epoch": 4, "epochs": 1}),
                encoding="utf-8",
            )
            parameters = dict(
                data_dir=data_dir, vocab_size=13, source_family="synthetic",
                source_repository="synthetic/tokenizer-test", source_revision="test-only",
                length=16, levels=[1, 4, 16], width=8, codebook_size=4, heads=2,
                batch_size=2, updates=4, warmup_steps=2, max_steps=None, device="cpu",
                microbatch_size=1, checkpoint_every=1,
            )
            interrupted_dir = root / "interrupted"
            train(output_dir=interrupted_dir, stop_after=2, **parameters)
            train(output_dir=interrupted_dir, resume=True, **parameters)
            complete_dir = root / "complete"
            train(output_dir=complete_dir, **parameters)

            interrupted_meta, interrupted_tensors = load_latest(
                interrupted_dir / "checkpoints", _checkpoint_identity(parameters))
            complete_meta, complete_tensors = load_latest(
                complete_dir / "checkpoints", _checkpoint_identity(parameters))
            self.assertEqual(interrupted_meta["scalar_state"], complete_meta["scalar_state"])
            self.assertEqual(set(interrupted_tensors), set(complete_tensors))
            for name in interrupted_tensors:
                self.assertTrue(torch.equal(interrupted_tensors[name], complete_tensors[name]), name)
            generations = sorted((interrupted_dir / "checkpoints").glob("generation-*"))
            self.assertEqual([path.name for path in generations], ["generation-00000003", "generation-00000004"])
            pointer = json.loads((interrupted_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["previous_generation"], "generation-00000003")

    def test_checkpoint_rejects_corruption_and_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-invalid-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "train_epoch_0000.u32le").write_bytes(struct.pack("<16I", *([1] * 16)))
            (data_dir / "summary.json").write_text(
                json.dumps({"window_length": 16, "seed": 42, "training_windows_per_epoch": 1, "epochs": 1}),
                encoding="utf-8",
            )
            output_dir = root / "output"
            parameters = dict(
                data_dir=data_dir, output_dir=output_dir, vocab_size=3, source_family="synthetic",
                source_repository="synthetic/tokenizer-test", source_revision="test-only", length=16,
                levels=[1, 4, 16], width=8, codebook_size=4, heads=2, batch_size=1, updates=1,
                warmup_steps=1, max_steps=None, device="cpu", microbatch_size=1,
            )
            train(**parameters)
            identity = _checkpoint_identity(parameters)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_latest(output_dir / "checkpoints", {**identity, "training": {"changed": True}})
            state_path = output_dir / "checkpoints" / "generation-00000001" / "state.safetensors"
            state_path.write_bytes(state_path.read_bytes()[:-1] + b"X")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_latest(output_dir / "checkpoints", identity)

    def test_checkpoint_rejects_tampered_scalar_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-scalar-corruption-") as temporary:
            root = Path(temporary) / "checkpoints"
            identity = {"model": "tiny"}
            save_generation(root, identity, {"weight": torch.ones(1)}, {"step": 1}, 1)
            state_path = root / "generation-00000001" / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["scalar_state"]["step"] = 9
            state_path.write_text(json.dumps(state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON checksum mismatch"):
                load_latest(root, identity)

    def test_resume_rejects_out_of_range_update_count(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-position-corruption-") as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "train_epoch_0000.u32le").write_bytes(struct.pack("<16I", *([1] * 16)))
            (data_dir / "summary.json").write_text(
                json.dumps({"window_length": 16, "seed": 42, "training_windows_per_epoch": 1, "epochs": 1}),
                encoding="utf-8",
            )
            output_dir = root / "output"
            parameters = dict(
                data_dir=data_dir, output_dir=output_dir, vocab_size=3, source_family="synthetic",
                source_repository="synthetic/tokenizer-test", source_revision="test-only", length=16,
                levels=[1, 4, 16], width=8, codebook_size=4, heads=2, batch_size=1, updates=1,
                warmup_steps=1, max_steps=None, device="cpu", microbatch_size=1,
            )
            train(**parameters)
            checkpoint_dir = output_dir / "checkpoints"
            state_path = checkpoint_dir / "generation-00000001" / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["scalar_state"]["step"] = 2
            state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
            state_path.write_bytes(state_bytes)
            pointer_path = checkpoint_dir / "latest.json"
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer["state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
            pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "update count is outside the schedule"):
                train(**parameters, resume=True)

    def test_failed_first_checkpoint_publication_recovers_orphan(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-first-save-") as temporary:
            root = Path(temporary) / "checkpoints"
            identity = {"model": "tiny"}
            tensors = {"weight": torch.tensor([1.0, 2.0])}
            replace = os.replace

            def fail_pointer_replace(source, destination):
                if Path(destination).name == "latest.json":
                    raise OSError("injected pointer publication failure")
                return replace(source, destination)

            with patch("vq_checkpoint.os.replace", side_effect=fail_pointer_replace):
                with self.assertRaisesRegex(OSError, "injected pointer"):
                    save_generation(root, identity, tensors, {"step": 1}, 1)
            self.assertTrue((root / "generation-00000001").is_dir())
            self.assertFalse((root / "latest.json").exists())

            save_generation(root, identity, tensors, {"step": 1}, 1)
            metadata, restored = load_latest(root, identity)
            self.assertEqual(metadata["scalar_state"]["step"], 1)
            self.assertTrue(torch.equal(restored["weight"], tensors["weight"]))
            self.assertEqual([path.name for path in root.glob("generation-*")], ["generation-00000001"])

    def test_failed_resumed_checkpoint_publication_preserves_previous_generation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ncmorph-vq-resume-save-") as temporary:
            root = Path(temporary) / "checkpoints"
            identity = {"model": "tiny"}
            first_tensors = {"weight": torch.tensor([1.0])}
            second_tensors = {"weight": torch.tensor([2.0])}
            save_generation(root, identity, first_tensors, {"step": 1}, 1)
            replace = os.replace

            def fail_pointer_replace(source, destination):
                if Path(destination).name == "latest.json":
                    raise OSError("injected pointer publication failure")
                return replace(source, destination)

            with patch("vq_checkpoint.os.replace", side_effect=fail_pointer_replace):
                with self.assertRaisesRegex(OSError, "injected pointer"):
                    save_generation(root, identity, second_tensors, {"step": 2}, 2)
            current_metadata, current_tensors = load_latest(root, identity)
            self.assertEqual(current_metadata["scalar_state"]["step"], 1)
            self.assertTrue(torch.equal(current_tensors["weight"], first_tensors["weight"]))
            self.assertTrue((root / "generation-00000002").is_dir())

            save_generation(root, identity, second_tensors, {"step": 2}, 2)
            resumed_metadata, resumed_tensors = load_latest(root, identity)
            self.assertEqual(resumed_metadata["scalar_state"]["step"], 2)
            self.assertTrue(torch.equal(resumed_tensors["weight"], second_tensors["weight"]))
            self.assertTrue((root / "generation-00000001").is_dir())
            pointer = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["previous_generation"], "generation-00000001")


def _checkpoint_identity(parameters):
    # The trainer's identity deliberately excludes the output location and invocation stop limit.
    summary = json.loads((parameters["data_dir"] / "summary.json").read_text(encoding="utf-8"))
    return {
        "model": {"vocab_size": parameters["vocab_size"], "length": parameters["length"],
                  "levels": parameters["levels"], "width": parameters["width"],
                  "codebook_size": parameters["codebook_size"], "heads": parameters["heads"], "dropout": 0.3},
        "training": {"source_family": parameters["source_family"],
                     "source_repository": parameters["source_repository"],
                     "source_revision": parameters["source_revision"], "batch_size": parameters["batch_size"],
                     "updates": parameters["max_steps"] or parameters["updates"],
                     "warmup_steps": parameters["warmup_steps"], "seed": 42,
                     "microbatch_size": parameters["microbatch_size"], "device": "cpu",
                     "precision": "bf16_mixed_f32_master"},
        "data": {"summary": summary,
                 "files": _hash_data(sorted(parameters["data_dir"].glob("train_epoch_*.u32le")))},
    }


if __name__ == "__main__":
    unittest.main()
