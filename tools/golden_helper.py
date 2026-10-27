#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Generate deterministic tiny transformers checkpoints and reference outputs."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path


PinnedVersions = {
    "torch": "2.13.0",
    "transformers": "5.20.2",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
}
BaseSeed = 271828
TokenIds = [1, 4, 7, 11]
VocabSize = 64


def check_versions() -> bool:
    if sys.version_info[:2] != (3, 13):
        print(
            f"golden helper requires Python 3.13, found {sys.version.split()[0]}",
            file=sys.stderr,
        )
        return False

    ok = True
    for package, expected in PinnedVersions.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None

        # Local build suffixes such as torch's +cpu remain compatible with the pin.
        if actual is None or actual.split("+", maxsplit=1)[0] != expected:
            found = actual if actual is not None else "missing"
            print(
                f"golden helper requires {package}=={expected}, found {found}",
                file=sys.stderr,
            )
            ok = False
    return ok


def build_cases():
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    llama_defaults = {
        "vocab_size": VocabSize,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "use_cache": False,
    }
    qwen_defaults = {
        "vocab_size": VocabSize,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "use_cache": False,
    }

    return [
        (
            "llama_default",
            LlamaForCausalLM,
            LlamaConfig(
                **llama_defaults,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            ),
        ),
        (
            "llama_llama3",
            LlamaForCausalLM,
            LlamaConfig(
                **llama_defaults,
                rope_parameters={
                    "rope_type": "llama3",
                    "rope_theta": 500000.0,
                    "factor": 4.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 2.0,
                    "original_max_position_embeddings": 64,
                },
            ),
        ),
        (
            "qwen2",
            Qwen2ForCausalLM,
            Qwen2Config(
                **qwen_defaults,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=True,
                rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
                attention_dropout=0.0,
            ),
        ),
        (
            "qwen3",
            Qwen3ForCausalLM,
            Qwen3Config(
                **qwen_defaults,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
                rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
            ),
        ),
        (
            "mistral",
            MistralForCausalLM,
            MistralConfig(
                **llama_defaults,
                sliding_window=3,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            ),
        ),
        (
            "qwen3_attention_bias",
            Qwen3ForCausalLM,
            Qwen3Config(
                **qwen_defaults,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                tie_word_embeddings=True,
                attention_bias=True,
                rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
            ),
        ),
    ]


def tensor_values(tensor, torch):
    return tensor.detach().to(dtype=torch.float32, device="cpu").tolist()


def saved_tensors(case_dir: Path, load_file):
    index_path = case_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return load_file(case_dir / "model.safetensors", device="cpu")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard_names = sorted(set(index["weight_map"].values()))
    tensors = {}
    for shard_name in shard_names:
        tensors.update(load_file(case_dir / shard_name, device="cpu"))
    if set(tensors) != set(index["weight_map"]):
        raise RuntimeError(f"{case_dir.name} sharded tensors do not match its index")
    return tensors


def generate_rope_tables(output_dir: Path, torch) -> None:
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from transformers import __version__ as transformers_version

    tables = {}
    positions = torch.arange(4096, dtype=torch.long).unsqueeze(0)
    for head_dim in (64, 128):
        for rope_theta in (10000.0, 500000.0):
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
            table_name = f"default_head_dim_{head_dim}_theta_{int(rope_theta)}"
            config = LlamaConfig(
                vocab_size=VocabSize,
                hidden_size=head_dim * 4,
                intermediate_size=head_dim * 4,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=head_dim,
                max_position_embeddings=131072,
                rope_parameters=rope_parameters,
            )
            tables[table_name] = make_rope_table(
                torch, LlamaRotaryEmbedding(config), head_dim, positions, rope_parameters
            )

    llama3_parameters = {
        "rope_type": "llama3",
        "rope_theta": 500000.0,
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    llama3_config = LlamaConfig(
        vocab_size=VocabSize,
        hidden_size=64 * 4,
        intermediate_size=64 * 4,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=131072,
        rope_parameters=llama3_parameters,
    )
    tables["llama3_head_dim_64_theta_500000"] = make_rope_table(
        torch, LlamaRotaryEmbedding(llama3_config), 64, positions, llama3_parameters
    )

    output = {
        "transformers_version": transformers_version,
        "position_start": 0,
        "position_end_inclusive": 4095,
        "tables": tables,
    }
    (output_dir / "rope_tables.json").write_text(
        json.dumps(output, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def make_rope_table(torch, rotary_embedding, head_dim: int, position_ids, rope_parameters: dict):
    hidden_states = torch.empty((1, position_ids.shape[1], head_dim), dtype=torch.float32)
    with torch.inference_mode():
        cosine, sine = rotary_embedding(hidden_states, position_ids=position_ids)
    return {
        "head_dim": head_dim,
        "rope_parameters": rope_parameters,
        "inv_freq": tensor_values(rotary_embedding.inv_freq, torch),
        "cos": tensor_values(cosine[0], torch),
        "sin": tensor_values(sine[0], torch),
    }


def generate(output_dir: Path, seed: int) -> None:
    import torch
    from safetensors.torch import load_file

    from transformers import __version__ as transformers_version
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_rope_tables(output_dir, torch)
    input_ids = torch.tensor([TokenIds], dtype=torch.long)
    position_ids = torch.arange(len(TokenIds), dtype=torch.long).unsqueeze(0)

    for case_index, (case_name, model_type, config) in enumerate(build_cases()):
        torch.manual_seed(seed + case_index)
        if case_name == "qwen3_attention_bias":
            config._attn_implementation = "eager"
        model = model_type(config).to(device="cpu", dtype=torch.float32).eval()
        if case_name == "qwen3_attention_bias":
            attention = model.model.layers[0].self_attn
            with torch.no_grad():
                attention.q_proj.bias.copy_(torch.linspace(-0.07, 0.08, attention.q_proj.bias.numel()))
                attention.k_proj.bias.copy_(torch.linspace(0.05, -0.04, attention.k_proj.bias.numel()))
                attention.v_proj.bias.copy_(torch.linspace(-0.03, 0.06, attention.v_proj.bias.numel()))
                attention.o_proj.bias.copy_(torch.linspace(0.02, -0.05, attention.o_proj.bias.numel()))
                attention.q_norm.weight.copy_(torch.linspace(0.7, 1.3, attention.q_norm.weight.numel()))
                attention.k_norm.weight.copy_(torch.linspace(1.2, 0.8, attention.k_norm.weight.numel()))
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in (
            case_dir / "model.safetensors",
            case_dir / "model.safetensors.index.json",
            *case_dir.glob("model-*-of-*.safetensors"),
        ):
            stale_file.unlink(missing_ok=True)
        save_options = {"max_shard_size": "8KB"} if case_name == "mistral" else {}
        model.save_pretrained(case_dir, safe_serialization=True, **save_options)

        layer_0_outputs = []
        attention_outputs = []

        def capture_layer_0_output(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"{case_name} layer 0 did not return a tensor")
            layer_0_outputs.append(output.detach().clone())

        hook = model.model.layers[0].register_forward_hook(capture_layer_0_output)
        attention_hook = None
        if case_name == "qwen3_attention_bias":
            def capture_attention_output(_module, _inputs, output):
                if not isinstance(output, tuple) or not isinstance(output[0], torch.Tensor):
                    raise RuntimeError(f"{case_name} attention did not return a tensor output")
                attention_outputs.append(output[0].detach().clone())

            attention_hook = model.model.layers[0].self_attn.register_forward_hook(capture_attention_output)
        try:
            with torch.inference_mode():
                embedded = model.model.embed_tokens(input_ids)
                cosine, sine = model.model.rotary_emb(embedded, position_ids=position_ids)
                mask_config = copy.deepcopy(model.config)
                mask_config._attn_implementation = "eager"
                mask_function = (
                    create_sliding_window_causal_mask
                    if getattr(mask_config, "sliding_window", None) is not None
                    else create_causal_mask
                )
                attention_mask = mask_function(
                    config=mask_config,
                    inputs_embeds=embedded,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=position_ids,
                    allow_is_causal_skip=False,
                )[0, 0].eq(0)
                outputs = model(input_ids, position_ids=position_ids, use_cache=False, output_hidden_states=True)
        finally:
            hook.remove()
            if attention_hook is not None:
                attention_hook.remove()

        if len(layer_0_outputs) != 1:
            raise RuntimeError(f"{case_name} captured layer 0 {len(layer_0_outputs)} times")
        layer_0_output = layer_0_outputs[0]
        if outputs.hidden_states is None or len(outputs.hidden_states) < 2:
            raise RuntimeError(f"{case_name} did not return hidden states")

        with torch.inference_mode():
            # Verify the captured tensor is the decoder block output before the
            # model's final normalization, not Transformers' tied final state.
            final_norm_output = model.model.norm(layer_0_output)
            torch.testing.assert_close(outputs.hidden_states[-1], final_norm_output)

            bf16_model = copy.deepcopy(model).eval()
            for parameter in bf16_model.parameters():
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
            bf16_model.config._attn_implementation = "eager"
            if case_name == "llama_default":
                bf16_dir = output_dir / "llama_default_bf16"
                bf16_dir.mkdir(parents=True, exist_ok=True)
                bf16_model.save_pretrained(bf16_dir, safe_serialization=True)
            bf16_outputs = bf16_model(
                input_ids,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=False,
            )

        reference = {
            "case": case_name,
            "seed": seed + case_index,
            "transformers_version": transformers_version,
            "token_ids": TokenIds,
            "position_ids": position_ids.tolist(),
            "rope": {
                "inv_freq": tensor_values(model.model.rotary_emb.inv_freq, torch),
                "cos": tensor_values(cosine, torch),
                "sin": tensor_values(sine, torch),
            },
            "layer_0_output": tensor_values(layer_0_output, torch),
            "hidden_states": [tensor_values(state, torch) for state in outputs.hidden_states],
            "logits": tensor_values(outputs.logits, torch),
            "bf16_logits": tensor_values(bf16_outputs.logits, torch),
            "attention_mask": attention_mask.tolist(),
            "tensor_bits": {
                name: [int(value) & 0xFFFFFFFF for value in tensor.contiguous().view(torch.int32).reshape(-1).tolist()]
                for name, tensor in saved_tensors(case_dir, load_file).items()
            },
        }
        if case_name == "qwen3_attention_bias":
            if len(attention_outputs) != 1:
                raise RuntimeError(f"{case_name} captured attention {len(attention_outputs)} times")
            attention = model.model.layers[0].self_attn
            attention_input = model.model.layers[0].input_layernorm(embedded)
            reference["attention_fixture"] = {
                "head_dim": attention.head_dim,
                "num_attention_heads": config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "rms_norm_epsilon": config.rms_norm_eps,
                "input": tensor_values(attention_input, torch)[0],
                "query_weight": tensor_values(attention.q_proj.weight, torch),
                "query_bias": tensor_values(attention.q_proj.bias, torch),
                "key_weight": tensor_values(attention.k_proj.weight, torch),
                "key_bias": tensor_values(attention.k_proj.bias, torch),
                "value_weight": tensor_values(attention.v_proj.weight, torch),
                "value_bias": tensor_values(attention.v_proj.bias, torch),
                "query_norm_weight": tensor_values(attention.q_norm.weight, torch),
                "key_norm_weight": tensor_values(attention.k_norm.weight, torch),
                "output_weight": tensor_values(attention.o_proj.weight, torch),
                "output_bias": tensor_values(attention.o_proj.bias, torch),
                "cosine": tensor_values(cosine, torch)[0],
                "sine": tensor_values(sine, torch)[0],
                "attention_mask": attention_mask.tolist(),
                "expected_output": tensor_values(attention_outputs[0], torch)[0],
            }
        (case_dir / "reference.json").write_text(
            json.dumps(reference, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        del model


def require_onednn_enabled(torch) -> None:
    backend = torch.backends.mkldnn
    if not backend.is_available():
        raise RuntimeError("the pinned PyTorch runtime does not have oneDNN available")
    if not backend.enabled:
        raise RuntimeError("oneDNN is available but disabled in the pinned PyTorch runtime")


def generate_real_checkpoint_reference(checkpoint_dir: Path) -> None:
    import torch

    require_onednn_enabled(torch)
    from transformers import AutoModelForCausalLM, __version__ as transformers_version

    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.to(device="cpu").eval()
    attention_backend = model.config._attn_implementation
    if attention_backend != "eager":
        raise RuntimeError(f"expected eager attention, got {attention_backend!r}")
    torch_config = torch.__config__.show()
    onednn_match = re.search(r"Intel\(R\) MKL-DNN v([^\s]+) \(Git Hash ([^)]+)\)", torch_config)
    cpu_capability_match = re.search(r"CPU capability usage: ([^\s]+)", torch_config)
    if onednn_match is None or cpu_capability_match is None:
        raise RuntimeError("could not identify oneDNN version/hash and CPU capability from torch.__config__.show()")
    input_ids = torch.tensor([TokenIds], dtype=torch.long, device="cpu")
    position_ids = torch.arange(len(TokenIds), dtype=torch.long, device="cpu").unsqueeze(0)
    diagnostics = {}
    diagnostic_hooks = []

    def capture_diagnostic(name: str, _module, _inputs, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"SmolLM2 {name} diagnostic was not a tensor")
        diagnostics[name] = tensor.detach().clone()

    def capture_diagnostic_input(name: str, _module, inputs) -> None:
        diagnostics[name] = inputs[0].detach().clone()

    diagnostic_layers = model.model.layers[:12]
    diagnostic_stages = {
        "input_norm": lambda layer: layer.input_layernorm,
        "attention_output": lambda layer: layer.self_attn,
        "post_attention_norm": lambda layer: layer.post_attention_layernorm,
        "gate_projection": lambda layer: layer.mlp.gate_proj,
        "up_projection": lambda layer: layer.mlp.up_proj,
        "mlp_output": lambda layer: layer.mlp.down_proj,
        "block_output": lambda layer: layer,
    }
    for layer_index, layer in enumerate(diagnostic_layers):
        for stage_name, get_module in diagnostic_stages.items():
            name = f"layer_{layer_index}_{stage_name}"
            module = get_module(layer)
            diagnostic_hooks.append(module.register_forward_hook(lambda mod, inputs, output, name=name: capture_diagnostic(name, mod, inputs, output)))
        name = f"layer_{layer_index}_gate_up_product"
        diagnostic_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda mod, inputs, name=name: capture_diagnostic_input(name, mod, inputs)))
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
    for hook in diagnostic_hooks:
        hook.remove()
    with torch.inference_mode():
        for layer_index, _layer in enumerate(diagnostic_layers):
            diagnostics[f"layer_{layer_index}_block_input"] = outputs.hidden_states[layer_index].detach().clone()
        for layer_index, layer in enumerate(diagnostic_layers):
            diagnostics[f"layer_{layer_index}_attention_residual"] = diagnostics[f"layer_{layer_index}_block_input"] + diagnostics[f"layer_{layer_index}_attention_output"]
    first_layer = diagnostic_layers[0]
    if getattr(first_layer.self_attn, "q_norm", None) is None:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        with torch.inference_mode():
            for layer_index, layer in enumerate(diagnostic_layers):
                attention = layer.self_attn
                head_dim = attention.head_dim
                stage_input = diagnostics[f"layer_{layer_index}_input_norm"]
                query = attention.q_proj(stage_input).view(1, len(TokenIds), attention.config.num_attention_heads, head_dim).transpose(1, 2)
                key = attention.k_proj(stage_input).view(1, len(TokenIds), attention.config.num_key_value_heads, head_dim).transpose(1, 2)
                value = attention.v_proj(stage_input).view(1, len(TokenIds), attention.config.num_key_value_heads, head_dim).transpose(1, 2)
                cosine, sine = model.model.rotary_emb(diagnostics[f"layer_{layer_index}_block_input"], position_ids=position_ids)
                query, key = apply_rotary_pos_emb(query, key, cosine, sine)
                key = key.repeat_interleave(attention.config.num_attention_heads // attention.config.num_key_value_heads, dim=1)
                value = value.repeat_interleave(attention.config.num_attention_heads // attention.config.num_key_value_heads, dim=1)
                scores = (query @ key.transpose(-1, -2)) * attention.scaling
                causal_mask = torch.full_like(scores, float("-inf")).triu(diagonal=1)
                scores = scores + causal_mask
                probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
                context = (probabilities @ value).transpose(1, 2).reshape(1, len(TokenIds), -1)
                diagnostics[f"layer_{layer_index}_attention_context"] = context.detach().clone()
    else:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

        with torch.inference_mode():
            for layer_index, layer in enumerate(diagnostic_layers):
                attention = layer.self_attn
                stage_input = diagnostics[f"layer_{layer_index}_input_norm"]
                input_shape = stage_input.shape[:-1]
                hidden_shape = (*input_shape, -1, attention.head_dim)
                query = attention.q_norm(attention.q_proj(stage_input).view(hidden_shape)).transpose(1, 2)
                key = attention.k_norm(attention.k_proj(stage_input).view(hidden_shape)).transpose(1, 2)
                value = attention.v_proj(stage_input).view(hidden_shape).transpose(1, 2)
                cosine, sine = model.model.rotary_emb(
                    diagnostics[f"layer_{layer_index}_block_input"],
                    position_ids=position_ids,
                )
                query, key = apply_rotary_pos_emb(query, key, cosine, sine)
                key = repeat_kv(key, attention.num_key_value_groups)
                value = repeat_kv(value, attention.num_key_value_groups)
                scores = torch.matmul(query, key.transpose(2, 3)) * attention.scaling
                scores = scores + torch.full_like(scores, float("-inf")).triu(diagonal=1)
                probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
                context = torch.matmul(probabilities, value).transpose(1, 2).contiguous()
                context = context.reshape(*input_shape, -1)
                diagnostics[f"layer_{layer_index}_attention_context"] = context.detach().clone()

    reference = {
        "reference_schema_version": 2,
        "reference_method": "Transformers 5.20.2 AutoModelForCausalLM / PyTorch 2.13.0+cpu, CPU BF16, eager, eval, inference_mode, use_cache=False",
        "transformers_version": transformers_version,
        "transformers_commit": "18329f1065382e54db698e242cf98bd3f3e5a766",
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "device": str(next(model.parameters()).device),
        "attention_backend": attention_backend,
        "torch_num_threads": torch.get_num_threads(),
        "onednn_enabled": torch.backends.mkldnn.enabled,
        "onednn_version": onednn_match.group(1),
        "onednn_git_version": onednn_match.group(2),
        "cpu_architecture": platform.machine(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_build_cpu_capability": cpu_capability_match.group(1),
        "model_type": model.config.model_type,
        "weight_dtype": str(next(model.parameters()).dtype),
        "token_ids": TokenIds,
        "position_ids": position_ids.tolist(),
        "hidden_states": [tensor_values(state, torch) for state in outputs.hidden_states],
        "diagnostics": {name: tensor_values(tensor, torch) for name, tensor in diagnostics.items()},
        "logits": tensor_values(outputs.logits, torch),
    }
    output_path = checkpoint_dir / "ncmorph_reference.json"
    output_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path} ({len(TokenIds)} positions, {model.config.vocab_size} logits per position)")


def main(argv: list[str] | None = None) -> int:
    if not check_versions():
        return 2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("zig-out/golden"),
        help="directory for generated configs, safetensors, and JSON references",
    )
    parser.add_argument(
        "--real-checkpoint-dir",
        type=Path,
        help="run a local real checkpoint and write ncmorph_reference.json beside it",
    )
    parser.add_argument("--seed", type=int, default=BaseSeed)
    args = parser.parse_args(argv)
    if args.real_checkpoint_dir is not None:
        generate_real_checkpoint_reference(args.real_checkpoint_dir)
    else:
        generate(args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
