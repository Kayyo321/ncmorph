#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Generate fixed-ID VQ forward references from the pinned NCM implementation."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn import functional as functional
from safetensors.torch import load_file, save_file

RepositoryRoot = Path(__file__).resolve().parent.parent
PinnedNcmCommit = "c70274273f489abc73832b2b98584a8483b6b203"
PinnedTorchVersion = "2.13.0"
DefaultReferenceDir = Path("/home/dev/ncmorph-testing/ncm")
DefaultWmt14Weights = Path("/home/dev/ncmorph-testing/phase8/converted/tokenizer.safetensors")


class config_t(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def check_pins(reference_dir: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PinnedNcmCommit:
        raise RuntimeError(f"NCM reference revision mismatch: expected {PinnedNcmCommit}, found {revision}")
    actual_torch = torch.__version__.split("+", maxsplit=1)[0]
    if actual_torch != PinnedTorchVersion:
        raise RuntimeError(f"VQ reference requires torch=={PinnedTorchVersion}; found {torch.__version__}")


def make_tiny_trained_model():
    sys.path.insert(0, str(RepositoryRoot / "tools"))
    from train_vq import vqvae_t

    torch.manual_seed(730201)
    random.seed(730207)
    model = vqvae_t(vocab_size=17, length=16, width=8, levels=[1, 4, 16],
                    codebook_size=4, heads=2, dropout=0.0)
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
         [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    torch.manual_seed(730211)
    random.seed(730213)
    optimizer.zero_grad(set_to_none=True)
    _logits, loss, _vq_loss = model(token_ids)
    loss.backward()
    optimizer.step()
    return model, token_ids


def make_reference_vqvae(vqvae_type, config: dict[str, Any]):
    levels = SimpleNamespace(use_manual_levels=True, manual_levels=config["levels"])
    quantizer_config = config_t({
        "clss": "multiscale_residual_vector_quantizer",
        "decay": 0.99,
        "epsilon": 1e-5,
        "commitment_cost": 0.25,
        "learned_l1_sampling": True,
        "learned_l1_cascade": True,
        "learned_l1_cascade_base": 4,
        "learned_l1_cascade_norm": config.get("cascade_norm", False),
        "learned_l1_cascade_norm_final": config.get("cascade_norm_final", False),
        "learned_all_sampling": False,
        "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0, "learnable_ratio": False},
        "aux": {"fine_drop": {"prob": 0.0, "min_keep": 1}},
        "levels": levels,
    })
    model = vqvae_type(
        in_vocab_size=config["vocab_size"],
        context_length=config["length"],
        embed_dim=config["width"],
        vqvae_vocab_size=[config["codebook_size"]] * len(config["levels"]),
        n_layers=config["heads"],
        vector_quantizer_config=quantizer_config,
        compression_factor=1,
        dropout=0.0,
        pre_quant_groupnorm=8,
        pre_quant_dropout=0.0,
        use_qk_norm=True,
        enc_layers=0,
        dec_layers=1,
        enc_positional=True,
        dec_positional=False,
    )
    return model


def tensor_values(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(torch.float32).cpu().contiguous().view(-1).tolist()


def build_reference(vqvae_type, model_state: dict[str, torch.Tensor], token_ids: torch.Tensor,
                    config: dict[str, Any], case_name: str, weights_file: Path,
                    include_logits: bool, selected_vocab_ids: list[int] | None = None,
                    logits_file: Path | None = None) -> dict[str, object]:
    model = make_reference_vqvae(vqvae_type, config)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    with torch.inference_mode():
        encoded = model._encode_pre_quant(token_ids)
        code_ids, _z_hats, _z_levels = model.vq.z_to_idx(encoded, return_zhats=True, return_zlevels=True)
        quantized, vq_loss, _per_level_util, _global_util, _ppl = model.vq(encoded)
        logits = model.decoder(quantized) if include_logits else None
        if selected_vocab_ids is None or logits is not None:
            selected_logits = None
        else:
            selected_indices = torch.tensor(selected_vocab_ids, dtype=torch.long)
            decoder_hidden = model.decoder.ln_f(model.decoder.middle_block(quantized))
            selected_logits = functional.linear(
                decoder_hidden,
                model.decoder.output_layer.weight.index_select(0, selected_indices),
            )

    level_ids = []
    offset = 0
    for level_size in config["levels"]:
        level_ids.append(code_ids[:, offset : offset + level_size].cpu().tolist())
        offset += level_size
    result: dict[str, object] = {
        "reference_commit": PinnedNcmCommit,
        "case_name": case_name,
        "weights_file": str(weights_file.resolve()),
        "config": config,
        "token_ids": token_ids.cpu().tolist(),
        "level_ids": level_ids,
        "encoded_shape": list(encoded.shape),
        "encoded": tensor_values(encoded),
        "reconstruction_shape": list(quantized.shape),
        "reconstruction": tensor_values(quantized),
        "vq_loss": float(vq_loss.cpu()),
    }
    if logits is not None:
        result["logits_shape"] = list(logits.shape)
        if logits_file is None:
            result["logits"] = tensor_values(logits)
        else:
            logits_file.parent.mkdir(parents=True, exist_ok=True)
            logits.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False).tofile(logits_file)
            result["logits_file"] = str(logits_file.resolve())
            result["logits_dtype"] = "F32_LE"
            result["logits_layout"] = "C-order [batch, sequence, vocabulary]"
        if selected_vocab_ids is not None:
            selected_indices = torch.tensor(selected_vocab_ids, dtype=torch.long)
            selected_logits = logits.index_select(-1, selected_indices)
    if selected_logits is not None:
        result["selected_vocab_ids"] = selected_vocab_ids
        result["selected_logits_shape"] = list(selected_logits.shape)
        result["selected_logits"] = tensor_values(selected_logits)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DefaultReferenceDir)
    parser.add_argument("--wmt14-weights", type=Path, default=DefaultWmt14Weights)
    parser.add_argument("--output-dir", type=Path, default=RepositoryRoot / "zig-out" / "golden" / "vq_forward")
    args = parser.parse_args()
    check_pins(args.reference_dir)
    if not args.wmt14_weights.is_file():
        raise FileNotFoundError(f"converted WMT14 tokenizer safetensors not found: {args.wmt14_weights}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference_dir))
    # These reference modules are only used by the configuration entry point,
    # which this helper bypasses by constructing the model directly.
    hydra_module = types.ModuleType("hydra")
    def get_reference_class(name: str):
        module_name, class_name = name.rsplit(".", maxsplit=1)
        return getattr(importlib.import_module(module_name), class_name)

    hydra_module.utils = types.SimpleNamespace(get_class=get_reference_class)
    sys.modules.setdefault("hydra", hydra_module)
    omegaconf_module = types.ModuleType("omegaconf")
    omegaconf_module.OmegaConf = types.SimpleNamespace(create=lambda value: value)
    sys.modules.setdefault("omegaconf", omegaconf_module)
    from models.vqvae import VQVAE

    tiny_model, tiny_ids = make_tiny_trained_model()
    tiny_weights_path = args.output_dir / "tiny_trained.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in tiny_model.state_dict().items()},
              str(tiny_weights_path))
    tiny_config = {"vocab_size": 17, "length": 16, "width": 8, "levels": [1, 4, 16],
                   "codebook_size": 4, "heads": 2}
    tiny_reference = build_reference(
        VQVAE, load_file(str(tiny_weights_path), device="cpu"), tiny_ids, tiny_config,
        "tiny_trained", tiny_weights_path, include_logits=True,
    )

    wmt_config = {
        "vocab_size": 151680,
        "length": 64,
        "width": 384,
        "levels": [1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64],
        "codebook_size": 4096,
        "heads": 6,
        "cascade_norm": True,
        "cascade_norm_final": True,
    }
    wmt_ids = torch.tensor(
        [[(position * 7919 + batch * 104729 + 17) % wmt_config["vocab_size"] for position in range(64)]
         for batch in range(2)],
        dtype=torch.long,
    )
    selected_vocab_ids = [0, 1, 7, 42, 50256, 8191, 65535, 131071, 151679]
    wmt_logits_path = args.output_dir / "wmt14_released_logits.f32le"
    wmt_reference = build_reference(
        VQVAE, load_file(str(args.wmt14_weights), device="cpu"), wmt_ids, wmt_config,
        "wmt14_released", args.wmt14_weights, include_logits=True,
        selected_vocab_ids=selected_vocab_ids,
        logits_file=wmt_logits_path,
    )
    for name, result in (("tiny_trained", tiny_reference), ("wmt14_released", wmt_reference)):
        output_path = args.output_dir / f"{name}.json"
        output_path.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")
    print(f"wrote {tiny_weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
