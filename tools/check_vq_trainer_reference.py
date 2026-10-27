#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Compare tiny VQ inference with the pinned NCM tokenizer implementation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional

RepositoryRoot = Path(__file__).resolve().parent.parent
ReferenceRoot = Path("/home/dev/ncmorph-testing/ncm")
ReferenceRevision = "c70274273f489abc73832b2b98584a8483b6b203"
Width = 8
VocabSize = 17
SequenceLength = 16
Levels = [1, 4, 16]
CodebookSize = 4
BatchSize = 2


def check_reference_revision() -> None:
    revision = subprocess.run(
        ["git", "-C", str(ReferenceRoot), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != ReferenceRevision:
        raise RuntimeError(f"NCM reference revision mismatch: expected {ReferenceRevision}, found {revision}")


def capture_assignments(quantizer, value: torch.Tensor) -> tuple[tuple, list[torch.Tensor]]:
    assignments = []
    original_embedding = functional.embedding

    def recording_embedding(indices, weight, *args, **kwargs):
        assignments.append(indices.detach().clone())
        return original_embedding(indices, weight, *args, **kwargs)

    functional.embedding = recording_embedding
    try:
        result = quantizer(value)
    finally:
        functional.embedding = original_embedding
    return result, assignments


def make_reference_quantizer(quantizer_type):
    from train_vq import _vq_t

    current_quantizer = _vq_t(Width, Levels, CodebookSize)
    reference_quantizer = quantizer_type(
        [CodebookSize] * len(Levels),
        Width,
        SimpleNamespace(use_manual_levels=True, manual_levels=Levels),
        quant_resi={"enabled": True, "ratio": 0.5, "share_mode": 0},
        learned_l1_sampling=True,
        learned_l1_cascade=True,
        learned_l1_cascade_base=4,
    )
    reference_quantizer.load_state_dict(current_quantizer.state_dict(), strict=True)
    return current_quantizer, reference_quantizer


def check_training_parity(quantizer_type) -> None:
    torch.manual_seed(20261027)
    current_quantizer, reference_quantizer = make_reference_quantizer(quantizer_type)
    current_quantizer.train()
    reference_quantizer.train()
    base_value = torch.linspace(-1.25, 1.25, BatchSize * SequenceLength * Width).reshape(
        BatchSize, SequenceLength, Width
    )
    current_value = base_value.clone().requires_grad_(True)
    reference_value = base_value.clone().requires_grad_(True)

    (current_quantized, current_loss, _), current_assignments = capture_assignments(
        current_quantizer, current_value
    )
    reference_result, reference_assignments = capture_assignments(reference_quantizer, reference_value)
    reference_quantized, reference_loss = reference_result[:2]

    if len(current_assignments) != len(Levels) or len(reference_assignments) != len(Levels):
        raise AssertionError("training forward did not perform one codebook lookup per level")
    for level_index, (current_indices, reference_indices) in enumerate(
        zip(current_assignments, reference_assignments, strict=True)
    ):
        torch.testing.assert_close(current_indices, reference_indices, rtol=0, atol=0,
                                   msg=f"training level {level_index} assignments differ")

    torch.testing.assert_close(current_quantized, reference_quantized, rtol=1e-5, atol=1e-6,
                               msg="training quantized reconstruction differs")
    torch.testing.assert_close(current_loss, reference_loss, rtol=1e-5, atol=1e-6,
                               msg="training VQ loss differs")
    for level_index in range(len(Levels)):
        for suffix in ("ema_cluster_size", "ema_w", "codebook_hit"):
            name = f"{suffix}_{level_index}"
            torch.testing.assert_close(
                getattr(current_quantizer, name), getattr(reference_quantizer, name),
                rtol=1e-5, atol=1e-6, msg=f"training buffer {name} differs",
            )
        torch.testing.assert_close(
            current_quantizer.embeddings[level_index].weight,
            reference_quantizer.embeddings[level_index].weight,
            rtol=1e-5, atol=1e-6, msg=f"training codebook {level_index} differs",
        )

    current_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(current_value.grad, reference_value.grad, rtol=1e-5, atol=1e-6,
                               msg="training input gradients differ")
    reference_parameters = dict(reference_quantizer.named_parameters())
    for name, current_parameter in current_quantizer.named_parameters():
        reference_parameter = reference_parameters[name]
        if current_parameter.grad is None or reference_parameter.grad is None:
            if current_parameter.grad is not reference_parameter.grad:
                raise AssertionError(f"training gradient presence differs for {name}")
        else:
            torch.testing.assert_close(current_parameter.grad, reference_parameter.grad,
                                       rtol=1e-5, atol=1e-6,
                                       msg=f"training parameter gradient differs for {name}")


def main() -> None:
    check_reference_revision()
    sys.path.insert(0, str(RepositoryRoot / "tools"))
    sys.path.insert(0, str(ReferenceRoot))

    from train_vq import vqvae_t
    from models.quant import MultiscaleResidualVectorQuantizer
    from models.vae import Decoder, Encoder

    torch.manual_seed(20261026)
    current = vqvae_t(
        vocab_size=VocabSize,
        length=SequenceLength,
        width=Width,
        levels=Levels,
        codebook_size=CodebookSize,
        heads=2,
        dropout=0.0,
    ).eval()

    reference_encoder = Encoder(
        VocabSize,
        SequenceLength,
        Width,
        2,
        compression_factor=1,
        dropout=0.0,
        enc_layers=0,
        use_pos_emb=True,
    ).eval()
    reference_quantizer = MultiscaleResidualVectorQuantizer(
        [CodebookSize] * len(Levels),
        Width,
        SimpleNamespace(use_manual_levels=True, manual_levels=Levels),
        quant_resi={"enabled": True, "ratio": 0.5, "share_mode": 0},
        learned_l1_sampling=True,
        learned_l1_cascade=True,
        learned_l1_cascade_base=4,
    ).eval()
    reference_decoder = Decoder(
        VocabSize,
        Width,
        2,
        compression_factor=1,
        dropout=0.0,
        use_qk_norm=True,
        dec_layers=1,
        use_pos_emb=False,
        context_length=SequenceLength,
    ).eval()

    reference_encoder.load_state_dict(current.encoder.state_dict(), strict=True)
    reference_quantizer.load_state_dict(current.vq.state_dict(), strict=True)
    reference_decoder.load_state_dict(current.decoder.state_dict(), strict=True)

    token_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        current_encoded = current.encoder(token_ids)
        current_encoded = current.pre_quant_norm(current_encoded.transpose(1, 2)).transpose(1, 2)
        reference_encoded = reference_encoder(token_ids)
        reference_encoded = functional.group_norm(
            reference_encoded.transpose(1, 2), num_groups=8, weight=None, bias=None, eps=1e-5
        ).transpose(1, 2)

        (current_quantized, current_vq_loss, current_levels), current_assignments = capture_assignments(
            current.vq, current_encoded
        )
        reference_result, reference_assignments = capture_assignments(reference_quantizer, reference_encoded)
        reference_quantized, reference_vq_loss = reference_result[:2]

        current_logits = current.decoder(current_quantized)
        reference_logits = reference_decoder(reference_quantized)

    if len(current_assignments) != len(Levels) or len(reference_assignments) != len(Levels):
        raise AssertionError(
            f"expected {len(Levels)} codebook lookups per quantizer, got "
            f"{len(current_assignments)} and {len(reference_assignments)}"
        )
    for level_index, (current_indices, reference_indices) in enumerate(
        zip(current_assignments, reference_assignments, strict=True)
    ):
        torch.testing.assert_close(current_indices, reference_indices, rtol=0, atol=0,
                                   msg=f"level {level_index} assignments differ")
    torch.testing.assert_close(current_encoded, reference_encoded, rtol=1e-5, atol=1e-6,
                               msg="pre-quant encoder features differ")
    torch.testing.assert_close(current_quantized, reference_quantized, rtol=1e-5, atol=1e-6,
                               msg="quantized reconstruction differs")
    torch.testing.assert_close(current_vq_loss, reference_vq_loss, rtol=1e-5, atol=1e-6,
                               msg="VQ loss differs")
    torch.testing.assert_close(current_logits, reference_logits, rtol=1e-5, atol=1e-6,
                               msg="decoder logits differ")

    print(
        "PASS: pinned NCM inference parity "
        f"(revision={ReferenceRevision}, batch={BatchSize}, length={SequenceLength}, "
        f"width={Width}, levels={Levels}, assignments={len(Levels)}, "
        f"max_abs_logits={torch.max(torch.abs(current_logits - reference_logits)).item():.8g})"
    )
    check_training_parity(MultiscaleResidualVectorQuantizer)
    print("PASS: pinned NCM training quantizer parity (single deterministic batch, no fine-drop truncation)")


if __name__ == "__main__":
    main()
