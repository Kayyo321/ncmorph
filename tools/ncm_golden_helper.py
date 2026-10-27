#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Write tiny mask and RoPE-position references from the pinned NCM class."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

PinnedNcmCommit = "c70274273f489abc73832b2b98584a8483b6b203"


def build_reference(reference_dir: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PinnedNcmCommit:
        raise RuntimeError(f"NCM reference must be {PinnedNcmCommit}, found {commit}")

    sys.path.insert(0, str(reference_dir))
    # NextConceptTransformer imports VQVAE only for type-level construction; these
    # mask/position cases do not construct or invoke the VQ implementation.
    vqvae_module = types.ModuleType("models.vqvae")
    vqvae_module.VQVAE = type("VQVAE", (), {})
    sys.modules["models.vqvae"] = vqvae_module

    import torch
    from models.next_concept import NextConceptTransformer

    cases = [
        {"name": "three_levels", "levels": [1, 2, 3], "prefix_length": 2, "seed": 3407, "qk_norm": False},
        {"name": "two_levels_qk_norm", "levels": [1, 3], "prefix_length": 1, "seed": 3419, "qk_norm": True},
    ]
    outputs = []
    for case in cases:
        torch.manual_seed(case["seed"])
        levels = case["levels"]
        prefix_length = case["prefix_length"]
        hidden_size = 8
        embed_dim = 4
        vocab_sizes = [7] * len(levels)
        config = types.SimpleNamespace(
            n_embd=hidden_size,
            n_layer=2,
            n_head=2,
            n_kv_heads=1,
            dropout=0.0,
            bias=True,
            vocab_size=vocab_sizes,
            attn_pattern="block_diagonal",
            prefix_len=prefix_length,
            use_rope=True,
            rope_base=10000,
            use_level_encoding=True,
            use_swiglu=True,
            use_qk_norm=case["qk_norm"],
            query_level0=True,
            post_upsample_conv=types.SimpleNamespace(enabled=True, kernel_size=3),
            predict_level0=False,
            tie_heads_to_codebook=False,
            shared_output_head=False,
            prefix_pad_token_id=31,
        )
        vqvae = types.SimpleNamespace(
            embed_dim=embed_dim,
            in_vocab_size=32,
            vq=types.SimpleNamespace(levels=levels, vqvae_vocab_sizes=vocab_sizes),
            encoder=types.SimpleNamespace(wte=torch.nn.Embedding(32, embed_dim)),
        )
        model = NextConceptTransformer(vqvae, config).eval()
        conv = model.post_upsample_convs[0]
        with torch.no_grad():
            conv.weight.copy_(torch.arange(conv.weight.numel(), dtype=torch.float32).reshape_as(conv.weight) * 0.01 - 0.08)
            conv.bias.copy_(torch.arange(embed_dim, dtype=torch.float32) * 0.02 - 0.03)
        conv_input = torch.arange(levels[0] * embed_dim, dtype=torch.float32).reshape(1, levels[0], embed_dim) * 0.1 - 0.2
        conv_output = model._apply_post_upsample_conv_single(conv_input, 0)
        total_positions = prefix_length + sum(levels)
        base_mask = model.attn_mask_for_masking[0, 0].eq(0)
        pad_mask = torch.zeros((1, prefix_length), dtype=torch.bool)
        pad_mask[0, -1] = True
        pad_bias = model._prefix_pad_bias(pad_mask, total_positions, model.attn_mask_for_masking.dtype, torch.device("cpu"))
        padded_mask = (model.attn_mask_for_masking + pad_bias)[0, 0].eq(0)
        positions = list(range(prefix_length))
        per_level_positions = []
        for level_length in levels:
            positions.extend(range(level_length))
            per_level_positions.append(list(range(prefix_length)) + list(range(level_length)))

        outputs.append(
            {
                "name": case["name"],
                "seed": case["seed"],
                "prefix_length": prefix_length,
                "level_lengths": levels,
                "positions": positions,
                "per_level_positions": per_level_positions,
                "mask": base_mask.tolist(),
                "padded_mask": padded_mask.tolist(),
                "rope_cos": model.rope_cos.tolist(),
                "rope_sin": model.rope_sin.tolist(),
                "rope_cos_gen": [getattr(model, f"rope_cos_gen_{index}").tolist() for index in range(len(levels))],
                "rope_sin_gen": [getattr(model, f"rope_sin_gen_{index}").tolist() for index in range(len(levels))],
                "post_upsample_conv": {
                    "weight": conv.weight.detach().flatten().tolist(),
                    "bias": conv.bias.detach().tolist(),
                    "input": conv_input.flatten().tolist(),
                    "output": conv_output.flatten().tolist(),
                    "row_count": levels[0],
                    "width": embed_dim,
                },
            }
        )

    return {"ncm_commit": commit, "cases": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("zig-out/golden/ncm_masks_positions.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_reference(args.reference_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
