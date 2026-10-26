# ncmorph

Scope, architecture, decisions and open questions are not written yet.

## Source file header

Every source file should contain:

```
/*
 * Ko, (Baczynski-Bruce) 2026; Apache 2.0
*/
```

Zig has no block comments. A `/* */` block does not parse in Zig 0.16.0, which was checked with `zig ast-check`. In `.zig` files the same header is therefore written as `//` lines, placed at the top of the file and above any `//!` module doc comment (also checked with `zig ast-check`):

```zig
//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Module doc comment goes here.
```

Other languages keep their native comment syntax: `#` lines for Python and shell, and the block form for C and C++. Data and configuration files are not source files and do not need the header. The `//` form for Zig was accepted by the project owner on 2026-10-26.

## Scope

ncmorph converts an open-weight, decoder-only transformer language model into a Next-Concept Modeling (NCM) model. It is a Zig library and a command-line tool. Zig is the primary language. Other languages are used only where a tool needs them, and each exception is recorded in the decision log.

Out of scope for the first version:

- Vision, multimodal and mixture-of-experts models.
- Quantized checkpoints (GPTQ, AWQ, GGUF). Input is full-precision safetensors.
- Downloading models. ncmorph reads a local directory. Model access, license acceptance and tokens stay with the user.
- Serving, batching and distributed training.
- Quality or speed claims that have no measurement in this repository.

## NCM reference

- Paper: arXiv:2610.14162, Next-Concept Modeling (Kumbong et al., HazyResearch, Stanford, 2026).
- Reference code: github.com/HazyResearch/ncm (MIT License), commit `c70274273f489abc73832b2b98584a8483b6b203`.

Facts checked in the pinned reference:

- The generator is `NextConceptTransformer` in `models/next_concept.py`. Blocks are in `models/common.py`.
- Every block norm and the final norm is `LayerNorm(bias=config.bias)`. `RMSNorm` is used only for QK-norm.
- `head_dim` is `n_embd // n_head`, and the fused `c_attn` is sized `n_embd + 2 * kv_dim`. Query and output widths therefore equal `n_embd`.
- The default MLP is GELU with hidden width `mlp_ratio * n_embd`. The gated variants fix their width at `round(8 * n_embd / 3)`, rounded to a multiple of 8.
- RoPE uses the rotate-half layout, which matches Hugging Face `rotate_half`. Positions restart at 0 for each level.
- Attention masks: prefix positions attend to each other, targets attend to the whole prefix, and targets attend within their own level (`block_diagonal`) or to earlier levels (`block_causal`).
- Output heads are tied to the VQ codebooks.
- With `raw_token_prefix`, prefix token ids are embedded through the VQ encoder's token table (`encoder.wte`) in `generate`, unless `contextual_prefix` is set.

## Supported source families

Llama, Qwen and Mistral checkpoints share one decoder structure: pre-norm blocks, RMSNorm, grouped-query attention, rotary position embeddings and a SwiGLU MLP. They differ only in configuration flags. ncmorph therefore supports them with one decoder core, configured by flags. There is no family-specific code path.

Evidence:

- Transformers pinned to `v5.20.2`, commit `18329f1065382e54db698e242cf98bd3f3e5a766` (2026-10-19). Files read: `models/{llama,qwen2,qwen3,mistral}/modeling_*.py`, the matching `configuration_*.py`, and `modeling_rope_utils.py`.
- Checkpoints inspected through `config.json` and safetensors headers only. No weights were downloaded:
  - `Qwen/Qwen2.5-0.5B` (single file)
  - `Qwen/Qwen3-0.6B` (single file)
  - `HuggingFaceTB/SmolLM2-135M`, which uses the Llama architecture (single file)
  - `mistralai/Mistral-7B-v0.3` (sharded; layer 0 is in shard 1 of 3)
  - `mistralai/Mistral-7B-v0.1` (config only)
  - `unsloth/Llama-3.2-1B` and `NousResearch/Llama-3.2-1B` (config only). The gated `meta-llama/Llama-3.2-1B` returned HTTP 401 without access.

### Flags by family

| Flag | Llama | Qwen2 | Qwen3 | Mistral |
|---|---|---|---|---|
| Normalization | RMSNorm, `rms_norm_eps` | RMSNorm, `rms_norm_eps` | RMSNorm, `rms_norm_eps` | RMSNorm, `rms_norm_eps` |
| q/k/v bias | `attention_bias` (default false) | true, fixed in modeling | `attention_bias` (false in Qwen3-0.6B) | false, fixed in modeling |
| o bias | `attention_bias` | false, fixed in modeling | `attention_bias` | false, fixed in modeling |
| MLP bias | `mlp_bias` (default false) | false, fixed | false, fixed | false, fixed |
| QK-norm | no | no | yes: `q_norm` and `k_norm`, width `head_dim` | no |
| `head_dim` | `head_dim`, or hidden / heads | hidden / heads (64 in Qwen2.5-0.5B) | explicit (128 in Qwen3-0.6B, where q width is 2048 while hidden is 1024; the config class defaults to 128 when the key is absent) | hidden / heads (128 in v0.3) |
| RoPE | `rope_theta`; `rope_scaling` of type `llama3` in Llama 3.2 | `rope_theta` (1e6) | `rope_theta` (1e6) | `rope_theta` (1e4 in v0.1, 1e6 in v0.3) |
| Sliding window | none | ignored unless `use_sliding_window` is true | `null` | v0.1: 4096, applied to all layers; v0.3: `null` |
| Output head | `tie_word_embeddings` | tied (0.5B) | tied in config, but the file still contains `lm_head.weight` | untied |
| Activation | `silu` | `silu` | `silu` | `silu` |

### What ncmorph accepts in the first version

- `model_type` and `architectures` agree: `llama` with `LlamaForCausalLM`, `qwen2` with `Qwen2ForCausalLM`, `qwen3` with `Qwen3ForCausalLM`, `mistral` with `MistralForCausalLM`.
- The flags above, with RoPE types `default` and `llama3`. The `llama3` frequency adjustment follows `_compute_llama3_parameters` in `modeling_rope_utils.py`.
- Weights stored as `BF16`, `F16` or `F32`. BF16 is widened to F32 on load. Zig 0.16.0's `std.zig` exposes no bf16 type (checked by grep), so the widening is done by bit manipulation.
- A single `model.safetensors`, or a sharded layout with `model.safetensors.index.json`.

### What ncmorph rejects in the first version, naming the feature

- Any other `model_type`, and any `architectures` value that disagrees with `model_type`.
- `use_mrope: true`, mixture-of-experts keys, and any `hidden_act` other than `silu`.
- `rope_interleaved: true`. Only the rotate-half layout is implemented.
- RoPE types other than `default` and `llama3`. Another type is added only together with tests against transformers.
- Sliding windows where the NCM sequence length exceeds the window. Source-decoder mode can run them.
- `use_sliding_window: true` for Qwen2 and Qwen3, and any `layer_types` value other than `full_attention`.
- Pickle files (`.bin`, `.pt`, `.pth`) are never loaded. Quantized checkpoints fail the manifest check. Known configuration keys that change computation are rejected, and the manifest check catches any tensor that does not fit the configuration.

### Manifest validation

Before any weight is read, ncmorph derives the expected tensor list from the flags and compares it with the safetensors header or headers:

- Every expected tensor is present, with the expected shape and a supported dtype.
- No unexpected tensors are present.
- `lm_head.weight` is optional when `tie_word_embeddings` is true. Qwen3-0.6B's file contains it anyway. If it is present, it must equal `model.embed_tokens.weight` after widening to F32, or ncmorph reports an error.
- For sharded checkpoints, the index maps every expected tensor to exactly one shard.

Per layer `i`, the expected tensors are `self_attn.{q,k,v,o}_proj.weight`, the biases the flags require, `self_attn.{q,k}_norm.weight` when QK-norm is on, `mlp.{gate,up,down}_proj.weight`, `input_layernorm.weight` and `post_attention_layernorm.weight`. Global tensors are `model.embed_tokens.weight`, `model.norm.weight`, and `lm_head.weight` unless the embeddings are tied.

The `inspect` command runs this check without reading weight data.

## NCM dialect: what the generator must change

| Need | Reference today | Change |
|---|---|---|
| RMSNorm in blocks and in the final norm | LayerNorm with bias only | Add an RMSNorm option. The reference `RMSNorm` class already follows the Hugging Face operation order: float32 accumulation, cast, then weight multiply. |
| Explicit `head_dim`, non-square q and o | `head_dim = n_embd // n_head`; q and o are `n_embd` wide | Separate `head_dim`. q width is `n_heads * head_dim`, and o maps it back to `n_embd`. |
| Separate bias flags | One `bias` flag for every linear layer and LayerNorm | `qkv_bias`, `o_bias`, `mlp_bias` and `norm_bias`. |
| Explicit MLP width | Gated width fixed by formula | Accept `intermediate_size`. |
| llama3 RoPE scaling | Plain base only | Add the `llama3` frequency adjustment. |
| Source-mode attention | Block-diagonal targets and restarted positions | Source mode uses a causal mask and one RoPE sequence. It is used only for reduction tests. |
| Sliding window | Not implemented | Reject when the NCM sequence exceeds the window. The source decoder implements it. |
| Token input | VQ token table, sized by `in_vocab_size` (50304 in the xsum config) | Size the table to the source vocabulary. Initialize it from `embed_tokens` when dimensions match (open decision 3). |
| Output | Tied codebook heads | New heads. The source LM head is not used in NCM mode. |
| QK-norm, GQA, RoPE layout | Present. GQA maps query head `h` to KV head `h / group`, which matches Hugging Face `repeat_kv`. RoPE uses rotate-half, which matches Hugging Face. | No change. Confirmed by the reduction tests. |

## Library packages

Generic Zig code lives in `packages/`, one module per package. Packages do not import NCM or ncmorph code, and a grep check in [AGENTS.md](AGENTS.md) enforces this. A package is added only when a milestone needs it, and it comes with tests.

Zig 0.16.0's standard library has no bf16 type and no Unicode general-category tables (both checked by grep). The packages below fill those gaps and the other project-specific pieces:

| Package | Contents | Needed by |
|---|---|---|
| `bf16` | BF16 type: round-to-nearest-even conversion from F32, and widening to F32. Verified against PyTorch. | M0b |
| `unicode_categories` | General-category lookup for the Unicode groups L, M, N, P, S, Z and C. Generated from `UnicodeData.txt` by a script in `tools/`. | T1 |
| `safetensors` | Header reader with the reference validation rules, sharded index parsing, and later a writer. | M0a (reader), M4 (writer) |
| `hf_config` | `config.json` parsing, family detection, flags and the expected tensor manifest. | M0a |
| `pretokenizers` | Hand-written splitters for the GPT-2, Qwen2 and SmolLM2 pre-tokenizer patterns, built on `unicode_categories`. | T1 |
| `bpe` | Byte-level BPE, and Metaspace with byte fallback. | T1, T2 |
| `kernels` | f32 matmul, RMSNorm, softmax, SiLU, RoPE (`default` and `llama3`), causal and sliding-window masks, and grouped-query attention. | M0b |

## Milestones and gates

Each gate must pass before the next step starts.

- **M0a `inspect`.** Parse `config.json`, detect the family, validate the flags, derive the manifest, and check it against the headers. No weights are read. Gate: accepts the header fixtures for the checkpoints listed above, and rejects synthetic variants with one error per feature, for example a Qwen3 fixture with `q_norm` removed.
- **M0b source decoder, F32.** One decoder block, used everywhere: RMSNorm, RoPE (`default` and `llama3`), GQA, optional QKV bias, optional QK-norm, explicit `head_dim`, SwiGLU, causal and sliding-window masks, and a tied or untied head. Gate: logits match transformers `v5.20.2` on tiny random models generated per family with fixed seeds. The golden files are generated at test time by the pinned Python helper.
- **M0c real checkpoints.** SmolLM2-135M first, then Qwen2.5-0.5B and Qwen3-0.6B. Gate: logits match on the same token ids, within the M0b tolerance. These tests read checkpoints from a path given in an environment variable and are skipped when it is unset. Checkpoints are never committed. Model size is limited to about 1B parameters until a BF16 compute path exists.
- **M1 NCM dialect.** Extend the NCM block with the flags above. Gate: in source mode, the NCM body reproduces the M0 hidden states. Unit tests cover the NCM masks and per-level positions.
- **M2 transplant.** Map every source tensor that has an exact counterpart, and initialize the rest explicitly. Gate: transplanted tensors are bit-identical to the source after widening. A transplant report lists every tensor as exact, initialized or new, with a checksum.
- **M3 adaptation.** Undecided. Its method and budget depend on the research (section Research findings) and on measurement. No gate is set until then.
- **M4 export and reload.** Gate: reloading an exported checkpoint gives the same logits as before export.

## Tokenizers

Tokenizer types, read from each `tokenizer.json`:

| Checkpoint | Model | Normalizer | Pre-tokenizer | Decoder |
|---|---|---|---|---|
| Qwen2.5-0.5B | BPE, no byte fallback | NFC | Split (regex), then ByteLevel | ByteLevel |
| SmolLM2-135M | BPE, no byte fallback | none | Digits, then ByteLevel | ByteLevel |
| Mistral-7B-v0.3 | BPE with byte fallback | none | Metaspace | Replace, ByteFallback, Fuse, Strip |

- **T0.** Token ids come from a pinned reference tokenizer, run through a Python helper. Tests use fixed id files. This is the first path.
- **T1.** Byte-level BPE in Zig, covering Qwen and SmolLM2. The blocker is that the regex pre-tokenizers use Unicode categories. Zig 0.16.0's standard library has UTF-8 utilities (`std/unicode.zig`), but grep found no general-category tables. Such tables must be generated or vendored, and their source and license recorded.
- **T2.** Metaspace with byte fallback, covering Mistral v0.3.

## Risks

- Tokenizer parity (T1) is the largest engineering risk, because of the Unicode tables.
- Memory: widening BF16 to F32 doubles the size of the weights. 7B-class checkpoints need either a BF16 compute path or streamed, memory-mapped loading. Neither is decided.
- Numerical parity depends on reduction order. Tolerances are measured, not assumed.
- Whether adaptation can recover quality is unknown. See the research findings.

## Open decisions

Each decision has a proposed default. The project owner decides.

1. **Family scope.** Llama (including the `llama3` RoPE type), Qwen2, Qwen3, and Mistral v0.1 and v0.3. Default: yes.
2. **Local input only.** ncmorph reads checkpoints from a local directory. Default: yes.
3. **VQ embedding dimension.** Keep the reference 384, or set it to the source hidden size so `embed_tokens` can be transplanted exactly. Default: keep 384 for M0 to M2, then measure.
4. **Tokenizer order.** T0, then T1, then T2. Default: yes.
5. **Training language for M3.** Default: decide after the research and a feasibility spike.
6. **Model size for M0.** About 1B parameters in F32. Default: yes.
7. **Test fixtures.** Golden models and outputs are generated at test time by a pinned Python helper in `tools/`. The repository commits only the helper, its pinned requirements, and text fixtures such as configs and header JSON. No `.safetensors` files are committed. Default: yes.

## Open questions

These are UNVERIFIED and must be checked before use:

- Llama 2 and Llama 3.0 checkpoints have not been inspected. Llama 3.2 was inspected through mirror configs only.
- Whether the tied `lm_head.weight` in Qwen3-0.6B equals `embed_tokens` exactly.
- Whether the QK-norm epsilon and shapes match in every Qwen3 size. Only 0.6B was inspected.
- The exact regex patterns of each family's pre-tokenizer.

## Decision log

- 2026-10-26: NCM means Next-Concept Modeling (HazyResearch/ncm), not any other expansion.
- 2026-10-26: Zig files use `//` for the required header, because Zig has no block comments. Accepted by the project owner.
- 2026-10-26 (accepted with the M0a start): Llama, Qwen2, Qwen3 and Mistral share one decoder core configured by flags. See Supported source families.
- 2026-10-26: Generic Zig code incubates in `packages/` in this repository, with no NCM imports. It moves to its own repository only after M0b and M0c pass and its APIs stop changing. Agreed with the project owner.
- 2026-10-26: Library boundary: readers, tokenizers and kernels are generic. The NCM generator, VQ tokenizer, morph logic and CLI stay in ncmorph. Agreed with the project owner.
- 2026-10-26: The library adds a BF16 type and Unicode general-category tables, as the project owner requested.
- 2026-10-26: Coding style for all code in this repository is in AGENTS.md: no camelCase, PascalCase for constants, enums and preprocessor macros, snake_case otherwise, and type definitions end in `_t`. Requested by the project owner.
- 2026-10-26: M0a (`inspect`) is implemented. Six real checkpoints, checked from their headers only, pass. Four deliberately broken copies fail, with one error per feature.

## Research findings

Pending. The deep-research run (`wf_3a9de5f1-8b7`) was still fetching sources when this section was last updated.

## References

- Transformers `v5.20.2`, commit `18329f1065382e54db698e242cf98bd3f3e5a766`.
- HazyResearch/ncm, commit `c70274273f489abc73832b2b98584a8483b6b203`.
- arXiv:2610.14162.
- The model repositories and configs listed under Supported source families.

Zig has no block comments. The Zig language reference says there are no multiline comments, and `/* */` is a parse error. In `.zig` files, write the same text as a line comment:

```
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
```
