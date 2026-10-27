# ncmorph

ncmorph converts an open-weight, decoder-only language model into a Next-Concept Modeling (NCM) model. This file is the specification. It states the scope, the pinned sources, the facts the implementation depends on, every decision with its basis, and the gates that define done. It contains no open questions. [AGENTS.md](AGENTS.md) holds the rules for working in the repository, and [ROADMAP.md](ROADMAP.md) lists the implementation steps in order. The evidence for each fact is in [research/](research/), and [research/checks.md](research/checks.md) records how each one was checked.

## Source file header

The owner asked for every source file to begin with:

```
/*
 * Ko, (Baczynski-Bruce) 2026; Apache 2.0
*/
```

Zig 0.16.0 has no block comments, and the block form does not parse. `zig ast-check` exits 1 on it, with `expected type expression, found '/'`, and exits 0 on the line form (checked 2026-10-26, see [checks.md](research/checks.md) section 7). So every `.zig` file begins with the same text as `//` lines, placed before any `//!` module comment:

```zig
//
// Ko, (Baczynski-Bruce) 2026; Apache 2.0
//
//! Module doc comment goes here.
```

Every other language keeps the owner's block form in its own comment syntax: `#` lines for Python and shell, and the block form for C and C++. Data and configuration files do not carry the header. This is decision D42.

## Scope

ncmorph converts a decoder-only transformer language model into an NCM model. It is a Zig library and a command-line tool. Zig is the primary language. Python is used only where a tool needs PyTorch or the pinned Hugging Face libraries, and those tools live in `tools/` (decision D5).

In scope: the source families in the table under "Source checkpoints". Out of scope, and rejected with a named error where the input would reach the code:

- Vision, multimodal and mixture-of-experts models.
- Quantized checkpoints (GPTQ, AWQ, bitsandbytes) and GGUF files (decision D46). ncmorph reads and writes safetensors only.
- Downloads. ncmorph reads a local directory. Model access, licence acceptance and tokens stay with the user (decision D2).
- Serving, batching and distributed training.
- Claims of quality or speed that the repository has not measured.
- `block_causal` attention (decision D8).

## Pinned sources

Every version here is pinned, and a step that uses a version uses the one shown.

| Item | Pin | Evidence |
|---|---|---|
| Zig | 0.16.0 through M0c, 0.17.0 from Z1 onward (decision D28) | `zig version`; `build.zig.zon` `minimum_zig_version = "0.17.0"` |
| transformers (decoder reference) | v5.20.2, commit `18329f1065382e54db698e242cf98bd3f3e5a766`, committed 2026-10-19T17:34:25Z | `software_sources.md` section 0; PyPI `transformers==5.20.2` uploaded 2026-10-19 |
| NCM reference | HazyResearch/ncm, commit `c70274273f489abc73832b2b98584a8483b6b203`, MIT | `research/ncm_facts.md` section 0; `checks.md` section 6 |
| Python for tools | 3.13 (local interpreter 3.13.5) | Every package below accepts 3.10 or later; `pyarrow` 26.0.0 needs 3.11 or later |
| torch | `torch==2.13.0` (the reference's `requirements.txt` leaves torch unpinned, so parity is measured under this pin) | PyPI, uploaded 2026-07-08 |
| transformers (Python) | `transformers==5.20.2` | PyPI, uploaded 2026-10-19 |
| tokenizers | `tokenizers==0.23.3`: tag v0.23.3, commit `301277072263fb3830eb66eec7e47adadf76f1dc`, published 2026-10-09. The transformers dependency range is `tokenizers>=0.23.1,<0.24.0` | `software_sources.md` section 0; `dependency_versions_table.py` line 72; PyPI first upload 2026-10-09 |
| safetensors | `safetensors==0.8.0`: tag v0.8.0, commit `a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6`, published 2026-06-09. The transformers range is `safetensors>=0.8.0` | `software_sources.md` section 0; `dependency_versions_table.py` line 58; PyPI first upload 2026-06-09 |
| pyarrow | `pyarrow==26.0.0` (reads the OpenWebText shards) | PyPI latest, uploaded 2026-10-09 |
| Judge model | `openai-community/gpt2-large` at `32b71b12589c2f8d625668d2335a01cac3249519`, licence `mit` | `config/eval/owt.yaml` line 193; Hugging Face API response |
| Corpus | `Skylion007/openwebtext` at `79d93d786212f7344586290adb811d4ae6a1762c`, licence `cc0-1.0` | Hugging Face API response; `checks.md` section 11 |
| Unicode 9.0.0 data (NFC tables) | `UnicodeData.txt` SHA-256 `68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f`; `CompositionExclusions.txt` SHA-256 `5623df16856ad4007c60bdfff6f054e087521becd24cb4006be69c3a1d851aee` | unicode.org, read 2026-10-26 (D56) |
| Unicode 16.0.0 data (categories for regexes and Digits) | `UnicodeData.txt` SHA-256 `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f` | unicode.org, read 2026-10-26 (D57) |

The helper for golden tests and the trainers install these pins from one requirements file in `tools/`. The helper refuses to run when a pin is missing (step B0 of [ROADMAP.md](ROADMAP.md)).

## Source checkpoints

Every checkpoint is named by repository and revision. Header fixtures for each one are in `packages/hf_config/testdata/`, and their `provenance.json` files hold the same revisions.

| Fixture | Repository | Revision | Licence tag |
|---|---|---|---|
| llama-2-7b | `NousResearch/Llama-2-7b-hf` (mirror of gated `meta-llama/Llama-2-7b-hf`) | `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` | none on the mirror; the original is `llama2` |
| llama-3.0-8b | `NousResearch/Meta-Llama-3-8B` (mirror of gated `meta-llama/Meta-Llama-3-8B`) | `315b20096dc791d381d514deb5f8bd9c8d6d3061` | `other` on the mirror; the original is `llama3` |
| llama-3.2-1b | `unsloth/Llama-3.2-1B` (mirror of gated `meta-llama/Llama-3.2-1B`) | `9535bd9b1d1dea6acafbdc4813b728796aeb28da` | `llama3.2` |
| mistral-7b-v0.1 | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `apache-2.0` |
| mistral-7b-v0.3 | `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` | `apache-2.0` |
| qwen2.5-0.5b | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | `apache-2.0` |
| qwen3-0.6b | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | `apache-2.0` |
| qwen3-1.7b | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `apache-2.0` |
| qwen3-4b | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `apache-2.0` |
| qwen3-8b | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | `apache-2.0` |
| qwen3-14b | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | `apache-2.0` |
| qwen3-32b | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | `apache-2.0` |
| smollm2-135m | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `apache-2.0` |

Licence tags are read from the Hugging Face API responses saved on 2026-10-26. The licence texts are in "Licences and notices". Real conversions use the gated originals, which the user must have access to. The mirrors are used only for header fixtures, and their weights are never downloaded.

## Flags by family

Each row was read from the fixture `config.json` on 2026-10-26 (`checks.md` section 2). A `—` means the key is absent in that config.

| Flag | Llama (2, 3.0, 3.2, SmolLM2) | Qwen2.5 | Qwen3 (dense) | Mistral (v0.1, v0.3) |
|---|---|---|---|---|
| `model_type` / architecture | `llama` / `LlamaForCausalLM` | `qwen2` / `Qwen2ForCausalLM` | `qwen3` / `Qwen3ForCausalLM` | `mistral` / `MistralForCausalLM` |
| Normalization | RMSNorm, `rms_norm_eps` 1e-5 | RMSNorm, 1e-6 | RMSNorm, 1e-6 | RMSNorm, 1e-5 |
| q, k, v bias | `attention_bias` (false in the fixtures that set it; absent in Llama 2) | true (fixed in the model code) | `attention_bias` false in all six | false (fixed) |
| o bias | `attention_bias` | false (fixed) | `attention_bias` false | false (fixed) |
| MLP bias | `mlp_bias` false (Llama 3.2) | false (fixed) | false (fixed) | false (fixed) |
| QK-norm | no | no | yes: `q_norm` and `k_norm`, width `head_dim` | no |
| `head_dim` | explicit in Llama 3.2-1B (64); otherwise hidden / heads: 128 in Llama 2 7B and Llama 3.0 8B, 64 in SmolLM2-135M | hidden / heads = 64 (896 / 14) | explicit, 128 in all six dense sizes. q width is heads × 128, which differs from hidden in 0.6B (2048 against 1024), 4B (4096 against 2560) and 32B (8192 against 5120) | hidden / heads = 128 |
| RoPE | `rope_theta`: 500000 (Llama 3.0 and 3.2), 100000 (SmolLM2), absent in Llama 2 (then 10000). Llama 3.2-1B adds `rope_scaling` of type `llama3` (factor 32, low 1, high 4, original 8192) | `rope_theta` 1e6 | `rope_theta` 1e6 in all six | `rope_theta` 1e4 (v0.1), 1e6 (v0.3) |
| Sliding window | none | `sliding_window` 32768 with `use_sliding_window` false, so it is ignored | `null` in all six | v0.1: 4096 (applies to every layer). v0.3: `null` |
| Head | `tie_word_embeddings`: true in SmolLM2-135M and Llama 3.2-1B; false in Llama 2 7B and Llama 3.0 8B | tied | tied in 0.6B, 1.7B and 4B; untied in 8B, 14B and 32B | untied |
| `lm_head.weight` stored | SmolLM2 no, Llama 3.2 no; Llama 2 and 3.0 yes | no | yes in 0.6B, 1.7B, 8B, 14B and 32B; no in 4B | yes |
| Activation | `silu` | `silu` | `silu` | `silu` |

Two facts from the table are used later. In Qwen3-0.6B, `lm_head.weight` and `model.embed_tokens.weight` are byte-identical (both 311,164,928 bytes, SHA-256 `8f29acf519434862d95613b2b4f6b9d14933a5e4d16baebf8ac0b33b410acfb6`, `checks.md` section 3). The Mistral v0.1 sliding window follows the mask rule in "Source decoder: exact computation".

## What the loader accepts and rejects

Accepted in the first version:

- `model_type` and the architecture name agree, as in the flags table.
- RoPE types `default` and `llama3`. The `llama3` frequency adjustment follows `_compute_llama3_parameters` (`research/transformers_formulas.md`).
- A missing `rope_theta` means 10000 (decision D11).
- Per-layer `self_attn.rotary_emb.inv_freq` buffers are optional. They are accepted and not loaded, because transformers v5.20.2 drops them on load and recomputes the frequencies from `rope_theta`.
- Weights stored as BF16, F16 or F32. BF16 is widened to F32 on load.
- A single `model.safetensors`, or a sharded layout with `model.safetensors.index.json`.

Rejected, each with a named error that says which feature was refused:

- Any other `model_type`, and any architecture name that disagrees with `model_type`.
- `use_mrope: true`, mixture-of-experts keys, and any `hidden_act` other than `silu`.
- `rope_interleaved: true`. Only the rotate-half layout is implemented.
- `rope_parameters`, the transformers v5 key for RoPE settings. It is rejected until step B0a reads it. The rejection was found in review, and the check is in `packages/hf_config/config.zig`.
- RoPE types other than `default` and `llama3`.
- `use_sliding_window: true` for Qwen2 or Qwen3, and any `layer_types` value other than `full_attention`.
- Pickle files (`.bin`, `.pt`, `.pth`). They are never loaded (the one exception is the converter in step V1, under decision D41).

## Manifest validation

Before any weight is read, ncmorph derives the expected tensor list from the flags and compares it with the header or headers:

- Every expected tensor is present, with the expected shape and a supported dtype.
- No tensor is unexpected, except the optional per-layer `self_attn.rotary_emb.inv_freq` buffers.
- `lm_head.weight` is optional when `tie_word_embeddings` is true. When it is present, its shape and dtype are checked by `inspect`. When weights are loaded (M0c), its values must equal `model.embed_tokens.weight` after widening, or the load fails.
- For sharded checkpoints, the index maps every expected tensor to exactly one shard, and each shard's header lists the tensors the index assigns to it.

Per layer `i`, the expected tensors are `self_attn.{q,k,v,o}_proj.weight`, the biases the flags require, `self_attn.{q,k}_norm.weight` when QK-norm is on, `mlp.{gate,up,down}_proj.weight`, `input_layernorm.weight` and `post_attention_layernorm.weight`. The global tensors are `model.embed_tokens.weight`, `model.norm.weight`, and `lm_head.weight` unless the head is tied.

## Source decoder: exact computation

The source decoder reproduces transformers v5.20.2 for every family above. The formulas are copied, with line numbers, in [research/transformers_formulas.md](research/transformers_formulas.md).

- **RMSNorm** (`LlamaRMSNorm`, `modeling_llama.py` lines 52 to 69): the input is cast to f32, `variance = mean(x²)` over the last dimension, `y = x · rsqrt(variance + eps)`, then `y` is cast back to the input dtype and multiplied by the weight. The weight multiply comes after the cast.
- **RoPE, type `default`** (`LlamaRotaryEmbedding`, lines 72 to 116): `inv_freq_i = 1 / base^(2i/d)` for `i` in `0 .. d/2 − 1`, computed as `1.0 / (base ** (arange(0, d, 2, float32) / d))`. Positions are multiplied by `inv_freq` in f32, `emb = concat(angles, angles)`, and `cos` and `sin` are taken in f32 and cast to the activation dtype. Positions start at 0.
- **RoPE, type `llama3`** (`_compute_llama3_parameters`, `modeling_rope_utils.py` lines 580 to 662): with `base = rope_theta`, `factor`, `low_freq_factor`, `high_freq_factor` and `old_len = original_max_position_embeddings`, take `inv_freq` as above, then `low_wl = old_len / low_freq_factor`, `high_wl = old_len / high_freq_factor`, `wavelen = 2π / inv_freq`. Where `wavelen > low_wl`, the frequency is `inv_freq / factor`. Let `smooth = (old_len / wavelen − low_freq_factor) / (high_freq_factor − low_freq_factor)`, and `smoothed = (1 − smooth) · inv_freq' / factor + smooth · inv_freq'`. In the medium band (`not (wavelen < high_wl)` and `not (wavelen > low_wl)`) the result is `smoothed`. The attention scaling factor is 1.0.
- **Rotation**: `rotate_half(x) = concat(−x₂, x₁)`, where `x₁` and `x₂` are the two halves of the head dimension. `q' = q·cos + rotate_half(q)·sin`, and the same for `k`.
- **QK-norm** (Qwen3 only): an RMSNorm over `head_dim` on each head of `q` and `k`, before RoPE.
- **Grouped-query attention**: query head `h` reads KV head `h / (n_heads / n_kv_heads)` (`repeat_kv`). Scores are scaled by `head_dim^(−1/2)`, masked, and softmaxed in f32.
- **Attention masks**: causal keeps `kv ≤ q` (`masking_utils.py` line 76). Sliding window keeps `q − W < kv ≤ q` (line 95, and `sliding_window_causal_mask_function`, lines 130 to 134). A missing `sliding_window` key in Mistral's config gives 4096 (`configuration_mistral.py` line 80). An explicit `null` gives a plain causal mask (`modeling_mistral.py` line 353). Decision D38.
- **MLP** (`LlamaMLP`, lines 152 to 165): SwiGLU, `down(silu(gate(x)) · up(x))`, with width `intermediate_size`.
- **Block**: `h = x + attn(input_layernorm(x))`, then `out = h + mlp(post_attention_layernorm(h))`.
- **Head**: `model.norm` on the last hidden state, then a linear map to the vocabulary. The head is `embed_tokens` when `tie_word_embeddings` is true, and `lm_head` otherwise.

The decoder preserves the source activation dtype through the full forward pass. F32 inputs use f32 operations. BF16 inputs use f32 accumulation for matrix products followed by BF16 rounding at the tensor-operation output, BF16 rounding after elementwise operations, and the Llama RMSNorm sequence below. The source reference and checkpoint dtype determine the mode; BF16 weights do not silently imply F32 activations. D53 remains the acceptance bound for both modes.

## Numeric types

- A BF16 value widens to F32 by placing its 16 bits in the high half of a 32-bit word. The widening is exact.
- Narrowing from F32 to BF16 rounds to nearest, ties to even. Every NaN becomes `0x7FC0`. The sign of `-0` is kept, and subnormals are not flushed. `packages/bf16` matches PyTorch 2.13.0 on 4,119 vectors (done, M0a).
- Conversion copies BF16 bytes without widening. Widening happens per tensor only where a comparison needs F32 values.

## Tokenizers

Each family's text tokenizer is reproduced from its `tokenizer.json`. The exact pipelines, at the revisions above, are in [research/tokenizer_pipelines.json](research/tokenizer_pipelines.json). Their summary, which the implementation follows, is:

| Family | Normalizer | Pre-tokenizer | Model | Decoder | Post-processor |
|---|---|---|---|---|---|
| Qwen2.5, Qwen3 | NFC | Split (Qwen regex, `Isolated`), then ByteLevel with `use_regex` false, `add_prefix_space` false, `trim_offsets` false | BPE; `byte_fallback` false; `fuse_unk` false | ByteLevel (`use_regex` false) | ByteLevel (`use_regex` false), as stored |
| Llama 3.0, Llama 3.2 | none | Split (Llama 3 regex, `Isolated`), then ByteLevel with `use_regex` false, `trim_offsets` true | BPE; `ignore_merges` true; `byte_fallback` false | ByteLevel (`add_prefix_space` true, `use_regex` true) | Sequence: ByteLevel (`add_prefix_space` true, `use_regex` true), then TemplateProcessing with `<\|begin_of_text\|>` (id 128000) |
| SmolLM2 | none | Digits (`individual_digits` true), then ByteLevel (`use_regex` true) | BPE; `byte_fallback` false | ByteLevel (`add_prefix_space` true, `use_regex` true) | none |
| Llama 2 | Prepend `▁`, then Replace `" "` → `▁` | none | BPE; `byte_fallback` true; `fuse_unk` true; `unk_token` `<unk>` | Replace `▁` → space, ByteFallback, Fuse, Strip (space, 1, 0) | TemplateProcessing with `<s>` |
| Mistral v0.1, v0.3 | none | Metaspace (`▁`, `prepend_scheme` `first`, `split` false) | BPE; `byte_fallback` true; `fuse_unk` true; `unk_token` `<unk>` | as Llama 2 | TemplateProcessing with `<s>` |

The regex strings, exactly as stored:

- Qwen2.5 and Qwen3 Split (`Isolated`):

  ```
  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```

- Llama 3.0 and 3.2 Split (`Isolated`): the same with `\p{N}{1,3}` in place of `\p{N}`.
- GPT-2 ByteLevel regex, used when `use_regex` is true (`tokenizers/src/pre_tokenizers/byte_level.rs` line 44 at v0.23.3):

  ```
  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  ```

The regex engine is Oniguruma 6.9.10 with Unicode 16.0.0 properties (`research/software_sources.md`, section B). An alternation takes the first alternative that matches at the current position, not the longest match. `\s` is the Oniguruma definition: U+0009 to U+000D, U+0085, and the categories Zl, Zp and Zs. Decision D54 keeps these matchers hand-written, and decision D57 fixes the Unicode tables.

**Byte-to-unicode table** (`byte_level.rs` lines 15 to 39). The bytes `0x21`–`0x7E`, `0xA1`–`0xAC` and `0xAE`–`0xFF` map to the code point with the same value. The other 68 bytes map, in byte order, to `256 + n`: `0x00`–`0x20` to `U+0100`–`U+0120`, `0x7F`–`0xA0` to `U+0121`–`U+0142`, and `0xAD` to `U+0143`. SmolLM2 confirms this: `Ġ` (U+0120) is id 216, and `Ċ` (U+010A) is id 198.

**Semantics the code needs** (sources in `software_sources.md`):

- ByteLevel with `use_regex` true splits each piece with the regex, using `Isolated` delimiters. With `use_regex` false the piece is kept whole and each UTF-8 byte is mapped through the table. `add_prefix_space` prepends a space to each piece that does not already start with one. The decoder inverts the table, and if a character is not in the table it uses the token's raw UTF-8 bytes. The bytes are decoded with `String::from_utf8_lossy` (`byte_level.rs` lines 155 to 172).
- Digits with `individual_digits` true makes each numeric character its own piece. `char::is_numeric` covers the general categories `Nd`, `Nl` and `No` (`digits.rs` lines 29 to 37).
- Metaspace replaces each ASCII space with `▁`. `first` prefixes only the piece at original offset 0, `always` prefixes every piece, and `never` prefixes nothing. Each prefix is added only when the piece does not already start with `▁`. With `split` true the pieces are split before each `▁` (`metaspace.rs`, lines 118 to 142).
- NFC uses the `unicode-normalization-alignments` 0.1.12 tables, which are Unicode 9.0.0 (`normalizers/unicode.rs` lines 27 to 31; `software_sources.md`, section B). Decision D56 reproduces those tables from the Unicode 9.0.0 data. NFC is the only normalizer in these families. The regexes and Digits depend on Unicode general categories instead (D57).
- Prepend adds `▁` only when the input is non-empty. Replace substitutes every match of the escaped pattern (`prepend.rs` lines 19 to 21; `replace.rs` lines 70 and 82 to 85).
- BPE: each pre-tokenized piece is looked up character by character. A character that is not in the vocabulary becomes its UTF-8 bytes looked up as `<0xNN>` (uppercase hex, two digits) when `byte_fallback` is true. Otherwise it becomes `unk_token`, and consecutive unknowns fuse when `fuse_unk` is true. Merges apply in rank order, where rank is the index in `merges`, and ties go to the leftmost position (`models/bpe/model.rs` lines 466 to 550, `word.rs` lines 28 to 35). With `ignore_merges` true, a whole piece that is in the vocabulary becomes one token (lines 559 to 567). Merges may be pairs (`[["a","b"], …]`) or strings (`"a b"`), and both are accepted. Llama 3.2 stores pairs. Llama 3.0 and Mistral store strings.
- Decoders: ByteFallback turns each `<0xNN>` token into one byte, and a run of byte tokens is decoded as UTF-8 with one `�` per undecodable byte. Fuse joins tokens. Strip removes up to `start` leading and `stop` trailing `content` characters per token.

**Staged plan.** Phase T0 uses the pinned Python `tokenizers==0.23.3` for every family, and it is the reference the Zig tokenizers are checked against. T1 implements the byte-level families in Zig (Qwen, Llama 3, SmolLM2). T2 implements the Metaspace and SentencePiece-style families (Llama 2, Mistral). Each Zig tokenizer is accepted only when its ids equal the T0 ids on the fixed corpus (decision D4).

**Family EOS and BOS ids** (`config.json`, fixture): Llama 2 and Mistral: BOS 1, EOS 2. Llama 3.0 and 3.2: BOS 128000, EOS 128001. Qwen2.5: BOS and EOS 151643. Qwen3: BOS 151643, EOS 151645. SmolLM2: BOS and EOS 0. The EOS id is the value the unconditional prefix and the document separator use (decisions D22 and D24).

## NCM reference facts

Each fact is cited to the pinned commit. The full fact sheet with more citations is [research/ncm_facts.md](research/ncm_facts.md).

**Generation** (paper Algorithm 2, `models/next_concept.py` `generate`, lines 521 to 703). The coarsest level is generated first. For each level `k`: the previous reconstruction is resized to `ℓ_k` positions, projected to the hidden width, passed through one forward pass, and all `ℓ_k` codes of the level are sampled in parallel. The reconstruction then accumulates `Rₖ(eₖ(rₖ))`. The head for the level sees its own positions (`ncm_facts.md` A2).

**Attention** (`NextConceptTransformer`, mask construction at lines 266 to 292): the prefix attends to the prefix only. Each level attends to the prefix and to itself, and not to other levels. Pad prefix tokens are masked as keys (lines 418 to 425 and 572 to 579). Prompt positions cannot attend to target blocks (paper Section 3.4).

**Positions**: the prefix takes positions `0 … P−1`, and every level restarts at `0 … ℓ_j − 1` (lines 94 to 117). RoPE is the only active positional mechanism (`use_positional_encoding` is read by no code, decision D34). QK-norm is applied before RoPE (`models/common.py` lines 76 to 110).

**Level embedding and projection**: `transformer.wte` is a Linear from the VQ embedding width to `n_embd` (lines 131 to 133). `transformer.wle` is a learned level embedding, an `nn.Embedding` with one row per level (line 199). Every released configuration sets `use_level_encoding: true`.

**Prefix**: with `raw_token_prefix` true (every released configuration), prefix ids go through the VQ encoder's token table, `vqvae.encoder.wte` (`nn.Embedding(in_vocab_size, embed_dim)`), and then through `transformer.wte` (lines 546 to 558). The generator has no token table of its own (decision D58). The unconditional prefix is one EOS id (`scripts/eval_shard.py` line 157). The classifier-free guidance null prefix is `[EOS, PAD, …]` (lines 534 to 535).

**Level 0 in the released configurations** (`ncm_facts.md` A4): `query_level0` gives level 0 learned query slots, `l0_queries` (lines 239 to 242). `predict_level0` reads level 0 from the prefix instead (lines 249 to 264). The OWT and TinyStories configurations set neither, and they draw level-0 codes i.i.d. from `inference.level0_unigram` (`scripts/eval_shard.py` line 156). Decision D15 chooses learned queries for every family and no unigram.

**Sampling in the released configurations**: `_sample_level_raw` (lines 487 to 499) gives argmax at temperature 0, and otherwise samples from `logits/T` after the top-k and nucleus masks. The OWT configuration uses temperature 1.0, top-k 0 and top-p 1.0 (`owt.yaml` lines 176 to 178). The final token pick is `argmax` over the decoder logits, unless a frequency or adjacency penalty is configured (`scripts/eval_shard.py` lines 99 to 101; paper Algorithm 2, line 11).

**Tied heads** (`_TiedCodebookHead`, lines 9 to 34): `z = Linear(n_embd → d, bias false)(h)`; `E` is the level's codebook, frozen and stored as a buffer; `E_sq = Σ E²` per code; `d_scale = mean(E_sq)`; `neg_d2 = (2 z·Eᵀ − E_sq) / d_scale`; `logits = neg_d2 · exp(log_inv_T)`, plus an optional per-code bias. `log_inv_T` is a learned scalar that starts at 0. `owt.yaml` sets `tie_heads_to_codebook`, `tie_heads_learn_scale` and `tie_heads_bias` to true (lines 64 to 66). Without tying, a head is one Linear per level.

**Corruption** (paper Section 3.4 and Algorithm 1 lines 2 to 3): input codes are replaced by other codes from the same level's codebook, and the targets stay clean. The reference contains no training code (`ncm_facts.md` C5), and the `aux.code_corruption` block in the lm1b configurations is read by no pinned file. The schedule therefore comes from the paper and decision D17.

**Evaluation**: `evaluation/gen_ppl.py` and `scripts/eval_merge.py`, read in this pass (`checks.md` section 6).

**Known differences between the paper and the shipped code** are listed with the decisions that settle them (D43). The XSum level count differs (paper 8, `xsum.yaml` 9), and the OWT codebook differs (paper 4,096, `owt.yaml` 8,192). Each difference is settled by a decision below (D15, D16, D19, D43).

## Released NCM checkpoints

The Hugging Face repository `hazyresearch/ncm-checkpoints` (tagged MIT, not gated) holds the `lm1b`, `xsum`, `wmt14`, `tinystories` and `sudoku` checkpoints. Its `wmt14` folder holds `tokenizer.pt` (645,790,351 bytes, SHA-256 `74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702`), `generator.pt` (480,582,539 bytes, SHA-256 `ae580cbbb400a27b30c616634745d4c9cfa8b66806f936c6c26041e5b9640d4f`) and `config.yaml` (2,177 bytes). The repository's revision read on 2026-10-26 is `d342ce8aaea58830eef3b8ad473664673e41f692`. The hashes come from the Hugging Face tree API (`research/checks.md`, section 15). Its `owt` folder holds only `TODO.md`, so no OWT tokenizer or generator is published (`research/deep_research_digest.md`, claim C13).

The checkpoints are PyTorch pickles. The reference loads its generator with `torch.load(..., weights_only=False)` (`scripts/eval_shard.py` line 52). The tokenizer is loaded by `tokenization/vqvae_tokenizer.py` line 18: `vqvae.load_state_dict(torch.load(config.vqvae.checkpoint_path, weights_only=False, map_location='cpu')['model'], strict=True)`, with the path set at `scripts/eval_shard.py` line 45. ncmorph never loads a `.pt` file. The one conversion that does is the converter in step V1, which runs outside ncmorph under decision D41.

## Tokenizer route per family

Decision D10 says a released NCM tokenizer is reused only when its token set and its pre-tokenizer both match the source exactly. Otherwise the source gets its own trained tokenizer. The owner confirmed D10 on 2026-10-26, and chose reading A of "vocabulary" (decision D14).

| Source | Token set | Route | Basis |
|---|---|---|---|
| Qwen3, all six dense sizes | Qwen3's (ids 0 to 151,668; 151,669 ids) | **Reuse** the released WMT14 tokenizer | `wmt14.yaml` line 46 names `Qwen/Qwen3-0.6B` as the text tokenizer, and its table has 151,680 rows (line 53). That is 151,669 rounded up to a multiple of 64. The Qwen3 tokenizer files are the same in all six sizes (`checks.md` section 4). |
| Qwen2.5-0.5B | 151,665 ids. It lacks Qwen3's four added tokens 151665 to 151668 | **Train** | Decision D14 |
| Llama 2 7B, Llama 3.0 8B, Llama 3.2 1B, Mistral v0.1, Mistral v0.3, SmolLM2-135M | Their own vocabularies: Llama 2 32,000; Llama 3.0 and 3.2 128,256; Mistral v0.1 32,000; Mistral v0.3 32,768; SmolLM2 49,152 | **Train**, one per checkpoint | Decision D10. No released NCM tokenizer uses these vocabularies. |

Qwen2.5's token set differs from Qwen3's in the four added ids. Its base vocabulary and merges are identical to Qwen3's (`checks.md` section 4). Its normalizer, pre-tokenizer, post-processor and decoder are identical as well. The difference is therefore only in the added tokens, and the decision follows D14 on that basis.

Consequence of reuse, recorded in full: the WMT14 tokenizer was trained on WMT14 text, not on OpenWebText. Qwen3's generators are trained on OpenWebText windows that the WMT14 tokenizer encodes. Decision D44 requires a reconstruction gate on OpenWebText before that generator is trained. If the gate fails, the Qwen3 path trains a tokenizer with the trained-family recipe.

## Trained VQ tokenizer

Trained tokenizers keep the structure of the released OWT configuration, with the sizes of the paper's OWT column (decision D19). The reference modules are `models/vqvae.py`, `models/vae.py` and `models/quant.py` at the pinned commit. [research/reference/vq_forward_semantics.md](research/reference/vq_forward_semantics.md) gives the forward pass, the losses and their line citations, and it is the reference for the implementation.

| Setting | Value | Basis |
|---|---|---|
| Sequence length `L` | 1,024 | paper Table 5 OWT; `owt.yaml` line 79 |
| Levels `K` | 32, with level sizes `k²` for `k = 1 … 32` (1, 4, 9, …, 1,024) | paper Table 5; `owt.yaml` lines 139 to 172 |
| Codebook size `V_k` | 4,096 at every level | paper Table 5 (`owt.yaml` says 8,192; decision D43) |
| Width `d` | 384 | paper Table 5; `owt.yaml` line 77 |
| Encoder | token embedding `nn.Embedding(in_vocab_size, embed_dim)` (`vae.py` line 126; rows = source `vocab_size`, D23), a learned positional embedding, no blocks. Dropout 0.3 (`owt.yaml` line 81) runs before the pre-quant norm (`vae.py` line 171) | paper Table 5 (encoder blocks 0); `ncm_facts.md` C1; `vq_forward_semantics.md` section 2 |
| Pre-quant norm | GroupNorm with 8 groups and no affine parameters, applied to the `B × 384 × 1024` tensor; pre-quant dropout 0.0 | `owt.yaml` lines 82 and 84; `vqvae.py` lines 59 to 67 and 82 to 86 |
| Decoder | one pre-LN block, bidirectional attention through SDPA with `is_causal` false, 6 heads (`n_layers` in the config is the head count, not the depth), GELU MLP of width 4·d, QK-norm, output `Linear(d → vocab)` without bias, untied | paper Table 5 (decoder blocks 1, heads 6); `ncm_facts.md` C1; `owt.yaml` lines 76 and 83 |
| Nested dropout | probability 0.5 per forward call, `min_keep` 1 | paper Table 5; `owt.yaml` lines 134 to 136; `quant.py` lines 457 to 461 |
| Level-0 resampling | learned: five stride-4 `Conv1d(384, 384, kernel 4)` layers for the down step (weights identity divided by 4, biases zero) and five `ConvTranspose1d` layers for the up step (identity weights, biases zero), so the cascade factors are 4, 4, 4, 4, 4 (product 1,024) | `owt.yaml` lines 124 to 126; `quant.py` lines 145 to 162 and 185 to 204 |
| Levels 1 to 30 | down by `F.interpolate` with mode `area` to `ℓ_i` positions; up by `F.interpolate` with mode `linear` to 1,024 positions. Default PyTorch `align_corners` | `quant.py` lines 399 and 408 |
| Level 31 | no resampling | `quant.py` lines 474 to 475 and 509 to 510 |
| Residual mixing `quant_resi` | `(1 − r)·h + r·Conv1d(d, d, k = 3)(h)`, with `r = 0.5`, fixed. The convolution keeps PyTorch's default initialization, so the output is not the identity at initialization. One module per level (`share_mode` 0) | `owt.yaml` lines 128 to 132; `quant.py` lines 11 to 38 and 139 to 140 |
| Zero-initialized residual blocks | not active for OWT. They exist only under `learned_all_sampling` (false in `owt.yaml`, line 127) | `quant.py` lines 42 to 53 and 207 to 213, 330 to 339 |
| Codebook initialization and gradient | `N(0, 1)`, with `requires_grad` false. The codebook moves only by EMA | `quant.py` lines 297 to 300 |
| Codebook EMA | decay `γ = 0.99`, `ε = 1e-5`. Cluster sizes: `ema_cs ← γ·ema_cs + (1 − γ)·counts`. Smoothed sizes: `(ema_cs + ε) / (n + Vε) · n`. Sums: `ema_w ← γ·ema_w + (1 − γ)·Σz`. Weight: `ema_w / smoothed` | `owt.yaml` lines 121 and 122; `quant.py` lines 302 to 303 and 489 to 494 |
| Dead-code restart | off. The threshold defaults to 0.0 (`quant.py` line 244), and no OWT setting turns it on | `quant.py` lines 244 and 437 to 446 |
| Commitment `β` | 0.25. The paper states no numeric value | `owt.yaml` line 123; paper lines 394 to 397 |
| Post-upsample convolution | generator side only: zero-initialized Conv1d with kernel 3 on each target level from `_lvl_first`; in the pinned OWT configuration `query_level0: false`, so `_lvl_first = 1` and levels `j ≥ 1` are processed. The source-conditioned NCM dialect sets `query_level0: true`, so its level 0 is included (D73). | `owt.yaml` lines 60 to 63; `next_concept.py` lines 83 to 90 and 189 to 195 |

**Nearest code.** The distance is `‖x‖² + ‖E_i‖² − 2 x·E_iᵀ` per position (`quant.py` lines 388 to 392), and the code is the argmin.

**Objective** (decision D52). At every level `i`, the reference adds `β·MSE(sg[ẑ_i], z) + MSE(ẑ_i, sg[z])` with `ẑ_i` the full cumulative reconstruction (`quant.py` lines 512 to 524). The sum is divided by `K`, and the division is the final line of the loop. The reconstruction term, which the reference does not contain, is the cross-entropy between the decoder's logits and the input token ids, evaluated on the decoder input that nested dropout selects (paper Eq. 9). The decoder receives the cumulative reconstruction after `m` levels, where `m` is uniform on `{min_keep, …, K−1}` when the Bernoulli draw fires (`quant.py` lines 525 to 530). The reconstruction is then passed through the straight-through estimator `ẑ = z + sg(ẑ − z)` (line 531).

**Codes.** `VQVAE.encode` runs the encoder, the pre-quant norm and `Q.z_to_idx`. Per level it downsamples the residual, takes the nearest code, upsamples and applies `quant_resi`, and subtracts (`ncm_facts.md` C4). Indices are concatenated with level 0 first.

**Utilization counters.** The reference's `codebook_hit` buffers mark codes visited since the buffer was created and are never reset (`quant.py` lines 522 and 534 to 541). The utilization of decision D37 is therefore measured on the validation split with fresh counts, and never read from those buffers.

**Reference defect.** `VQVAE.reconstruct` (`models/vqvae.py` line 116) unpacks three of the five values that `self.vq` returns (`quant.py` lines 543 to 549), so it raises as written. ncmorph's converter and parity tests call `encode` and `forward` only. The reference's `quant.py` line 548 returns the literal 0.1 where a perplexity is expected, and nothing in ncmorph reads that value.

**Exported names.** The trained VQ keeps the reference attribute names, so the converter and the trainer write the same tensor names, listed in [research/converter/wmt14_tokenizer_keys.md](research/converter/wmt14_tokenizer_keys.md).

## Released WMT14 tokenizer conversion (V1)

Reuse for Qwen3 needs the released `tokenizer.pt` converted to `tokenizer.safetensors`. The converter is in `tools/`, outside ncmorph's runtime, and it follows these rules (decision D41):

1. The file is loaded only in a throwaway environment with no network access and no credentials. The load uses `torch.load(..., weights_only=False, map_location='cpu')`, which the reference also uses (`tokenization/vqvae_tokenizer.py` line 18). The file is a pickle, so this step is the one place that unpickles it, and it does so only under this rule.
2. Before loading, the converter checks the file's size (645,790,351 bytes) and SHA-256 (`74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702`, from the repository's LFS metadata), and it refuses any other file. It then takes the file's `'model'` mapping, as the reference's loader does, keeps tensors only, and records both values in `ncm.json`.
3. The mapping must hold exactly the 96 keys derived in `research/converter/wmt14_tokenizer_keys.md`: 63 parameter entries with 144,002,816 scalars, and 33 persistent buffer entries with 17,391,616 scalars. The check is `strict=True` loading into a module built from the derived structure, with no `vqvae.` prefix. The VQ module keeps the reference attribute names, so each key is written under its own name and nothing is mapped. Any key that is missing or unexpected fails the converter. A name is never renamed to fit.
4. Every tensor's shape is checked against the derivation. The encoder table has 151,680 rows and width 384, and the decoder output has 151,680 rows. The tokenizer has 11 levels (`wmt14.yaml` lines 61 to 71 and 95 to 106).
5. The converter keeps each tensor's stored dtype (D59), and writes `tokenizer.safetensors` with the writer of step W1 (D39). The export's `ncm.json` records the dtype, the file's size and its SHA-256, and the tokenizer route.
6. The notices of the export follow "Licences and notices". The released tensors are MIT material.

The derivation in `wmt14_tokenizer_keys.md` was written by reading the pinned code, because running it was refused in the session that wrote it. Its check is the converter run under rule 1, which the owner approves (ROADMAP, "Steps that need the owner").

The owner-approved isolated conversion produced 96 tensors (85 F32 and 11 I64), with all stored tensor bytes verified on readback. `tokenizer.safetensors` is 645,767,080 bytes with SHA-256 `c5a7a9689b1c50478419efcc6908477d3c4b3d334c7299ee5e15ce51628eff35` (see `research/checks.md`, Phase 8 conversion).

## NCM generator (ncmorph's dialect)

Every field below is fixed. The basis is in the decision list.

| Field | Value | Decision | Pinned reference evidence |
|---|---|---|---|
| Blocks | source `num_hidden_layers` | D9 | `models/common.py` `Block`; `models/next_concept.py` `NextConceptTransformer.__init__` |
| Width `n_embd` | source `hidden_size` | D9 | `models/common.py` `SelfAttention.__init__`, `MLP.__init__`, `Block.__init__` |
| Heads, KV heads, head dimension | source `num_attention_heads`, `num_key_value_heads`, and source `head_dim` (or hidden / heads) | D9 | `models/common.py` `SelfAttention.__init__` (reference uses `n_head`, optional `n_kv_heads`, and derives head width) |
| MLP | SwiGLU with width `intermediate_size` | D9 | `models/common.py` `MLP.__init__` and `MLP.forward` |
| Norms | RMSNorm with source `rms_norm_eps`, for `ln_1`, `ln_2` and `ln_f` | D9, D71 | The pinned `models/common.py` `Block.__init__` uses LayerNorm; its `RMSNorm` defaults to 1e-6 and is used for optional QK-norm. N2 adds the source-epsilon RMSNorm option for source-mode parity. |
| QK-norm | only for Qwen3 | D13 | `models/common.py` `SelfAttention.__init__` and `SelfAttention.forward` |
| RoPE | source type (`default` or `llama3`) and base | D20 | `models/common.py` `apply_rotary_emb`; `NextConceptTransformer` uses supplied RoPE cos/sin before attention |
| Attention pattern | `block_diagonal` | D8 | `models/next_concept.py` `NextConceptTransformer.__init__` constructs `target_mask` by level equality |
| Level embedding `wle` | learned, shape `K × n_embd` for this dialect because `query_level0` is true; pinned reference shape is `len(levels) - (0 if query_level0 else 1)` rows | D15, D30, D70 | `models/next_concept.py` `NextConceptTransformer.__init__` creates `wle` with that conditional size; `config/eval/owt.yaml` has `query_level0: false` |
| Level-0 slots | `query_level0` true, with learned `l0_queries` of shape `1 × ℓ_0 × d`, trunc-normal with std `sqrt(1/n_embd)` | D15, D30 | `models/next_concept.py` `NextConceptTransformer.__init__` and `build_l0_query_input` |
| Input projection `transformer.wte` | Linear from VQ width `d` to `n_embd`, with bias | D9, D30 | `models/next_concept.py` `NextConceptTransformer.forward` and `generate`; `config/eval/owt.yaml` `model.bias: true` |
| Biases | independent `qkv_bias`, `o_bias`, `mlp_bias` and `norm_bias`; QKV bias is fused in `c_attn`, MLP bias applies to `w_gate`, `w_up` and `c_proj`, and norm bias applies to `ln_1`, `ln_2` and `ln_f` | D72 | Pinned `models/common.py` `SelfAttention.__init__`, `MLP.__init__`, and `Block.__init__` use one shared `config.bias`; this dialect splits it to follow source checkpoint flags. |
| Output heads | one per predicted level, codebook-tied, with learned scale and per-code bias | D18 | `models/next_concept.py` `_TiedCodebookHead`, `NextConceptTransformer.__init__`; `config/eval/owt.yaml` `tie_heads_bias: true` |
| Post-upsample convolution | per target level beginning at `_lvl_first` (`0` when `query_level0` is true, `1` otherwise); this dialect uses `query_level0: true`, so level 0 is included | D73 | `models/next_concept.py` `NextConceptTransformer.__init__` sets `_lvl_first` and creates one convolution per target level; `_apply_post_upsample_convs_multi` iterates `levels[_lvl_first:]` |
| Prefix | one EOS id, `P = 1`, through the VQ encoder's table (D58) | D24 | `models/next_concept.py` `generate` (`raw_token_prefix` path); `config/eval/owt.yaml` `prefix_len: 1`, `raw_token_prefix: true` |
| Classifier-free guidance | not implemented | D33 | `models/next_concept.py` `generate` has CFG branches; this dialect deliberately omits them |
| Decoding penalties | not implemented | D33 | `scripts/eval_shard.py` `pick_tokens`; this dialect deliberately omits them |

Parameter counts follow from the source configuration. None is stated here, because none was measured.

## M2 transplant mapping

This is the decision record for M2.1 (D74). Source matrices and NCM matrices use the same `[output, input]` layout, so exact entries are copied without transposition. The fused QKV matrix and bias are assembled by concatenating the source Q, K and V rows in that order. `exact` means bit-identical after the source dtype is widened to F32; it describes tensor values and does not imply identical normalization semantics (D71).

| Source tensor | NCM tensor | Status |
|---|---|---|
| `model.layers.i.input_layernorm.weight` | `transformer.blocks.i.ln_1.weight` | exact |
| `model.layers.i.self_attn.{q,k,v}_proj.weight` | concatenated rows of `transformer.blocks.i.attn.c_attn.weight`, in q, k, v order | exact |
| `model.layers.i.self_attn.{q,k,v}_proj.bias` (when present) | concatenated rows of `transformer.blocks.i.attn.c_attn.bias`, in q, k, v order | exact |
| `model.layers.i.self_attn.o_proj.weight` | `transformer.blocks.i.attn.c_proj.weight` | exact |
| `model.layers.i.self_attn.o_proj.bias` (when present) | `transformer.blocks.i.attn.c_proj.bias` | exact |
| `model.layers.i.self_attn.{q,k}_norm.weight` (Qwen3) | `transformer.blocks.i.attn.{q,k}_norm.weight` | exact |
| `model.layers.i.post_attention_layernorm.weight` | `transformer.blocks.i.ln_2.weight` | exact |
| `model.layers.i.mlp.gate_proj.weight` | `transformer.blocks.i.mlp.w_gate.weight` | exact |
| `model.layers.i.mlp.up_proj.weight` | `transformer.blocks.i.mlp.w_up.weight` | exact |
| `model.layers.i.mlp.down_proj.weight` | `transformer.blocks.i.mlp.c_proj.weight` | exact |
| `model.layers.i.mlp.{gate,up,down}_proj.bias` (when present) | `transformer.blocks.i.mlp.{w_gate,w_up,c_proj}.bias`, respectively | exact |
| `model.norm.weight` | `transformer.ln_f.weight` | exact |
| `model.embed_tokens.weight` | no counterpart; source width is hidden size, while the prefix table is the VQ encoder table (D3, D58) | not used |
| `lm_head.weight` | no counterpart; NCM prediction heads are codebook-tied (D18) | not used |
| `model.layers.i.self_attn.rotary_emb.inv_freq` (when present) | no stored counterpart; runtime RoPE frequencies are computed from `rope_theta` | not used |

Evidence: the source tensor names and shapes are checked by `packages/hf_config/manifest.zig` and its checkpoint header fixtures. The pinned reference at `c70274273f489abc73832b2b98584a8483b6b203` defines the destination components in `models/common.py` (`SelfAttention`, `MLP`, `Block`) and `models/next_concept.py` (`NextConceptTransformer.__init__`). `SelfAttention.forward` splits `c_attn(x)` into Q, K and V in that order. Destination shapes are checked by `src/ncm_manifest.zig`, and `src/ncm_model.zig` consumes the fused matrix and bias in the same order. Extra or new NCM tensors are initialized under D30 and accounted for by M2.2/M2.3; they are not source mappings.

## Training

**Data.** OpenWebText, `Skylion007/openwebtext` at `79d93d786212f7344586290adb811d4ae6a1762c`, split `train`: 8,013,769 documents in 80 parquet shards, column `text`, with 39,769,491,688 bytes of text (`checks.md` section 11). Documents are split by index (decision D22):

- Validation: every document whose index is divisible by 1,000 (0.1 percent). These are excluded from training.
- Training: all other documents.

**Packing.** Each training document is tokenized with its family's tokenizer and post-processor, so BOS is added where the template adds it. Documents are joined with the family's EOS id from `config.json` (see "Family EOS and BOS ids"). The stream is cut into windows of `L` tokens, and the last partial window of each epoch is dropped. Windows are shuffled with seed 42 at each epoch. Epochs repeat until the update count is reached.

**Tokenizer recipe** (trained families, decision D21). The paper's OWT column of Table 5:

| Setting | Value |
|---|---|
| Updates | 78,432 |
| Global batch | 768 sequences of 1,024 tokens |
| Peak learning rate / floor | 3e-4 / 3e-5 |
| Warmup | 1,000 updates, linear from 0 |
| Schedule after warmup | cosine from peak to floor (paper Appendix F.1.2) |
| AdamW β1 / β2 | 0.9 / 0.95 |
| Weight decay | 0.01, on matrices with two or more dimensions only |
| Gradient clip (norm) | 1.0 |
| Codebook EMA decay | 0.99 |
| Dropout / nested dropout | 0.3 / 0.5 |
| Precision | BF16 mixed precision with f32 master weights (paper Appendix C) |
| Seed | 42 |

**Generator recipe** (trained families, decision D20). The paper's OWT column of Table 6, applied to the source-derived generator:

| Setting | Value |
|---|---|
| Updates | 250,000 |
| Global batch | 256 sequences of `L` tokens |
| Prefix | `P = 1`, one EOS id |
| Peak learning rate / floor | 3e-4 / 3e-5 |
| Warmup | 500 updates, linear from 0 |
| Schedule after warmup | cosine (paper Appendix F.1.2) |
| AdamW β1 / β2 | 0.9 / 0.95 |
| Weight decay | 0.1, on matrices with two or more dimensions only |
| Dropout | 0.1 |
| Gradient clip (norm) | 1.0 |
| Loss | `L_NCM` (Eq. 10) with `α = 0`, over all `K` levels, from one forward pass (D16) |
| Input corruption | uniform code replacement, rate `r_k = 1 − (k − 1)/(K − 1)` at level `k` (D17) |
| Precision | BF16 mixed precision |
| Seed | 42 |

The Qwen3 generators use the same recipe with `L = 64`, the length of the WMT14 tokenizer (D27).

**Initialization of new tensors** (decision D30). The reference's weight choices are kept: `l0_queries` is trunc-normal with std `sqrt(1/n_embd)` (`next_concept.py` line 242, using `init_std` from line 86). `transformer.wle` keeps PyTorch's `nn.Embedding` default, because the reference sets no initializer for it (line 199). The other new Linear and Conv weights keep PyTorch's default initialization, because the reference never calls `_init_weights` (`ncm_facts.md` B). The reference's normal initialization of `c_proj` (line 305) applies to models trained from scratch, so ncmorph does not use it, since every source block is transplanted exactly (M2). Biases start at 0, including the input projection bias; this adapts the reference, whose Linear default initializes that bias uniformly. RMSNorm weights start at 1.

**Transplant** (M2): each source block and norm is copied exactly (BF16 bytes widened to F32), and every new tensor is initialized as stated above. The source `embed_tokens` and `lm_head` have no NCM counterpart and are not used (D3, D18, D58). This is an initialization. Whether it is useful is measured by the M3 gate, and no quality is assumed.

**Budgets.** The budgets are the paper's, and they are the spec. The paper's OWT budget counts `updates × batch × L` positions, which is 78,432 × 768 × 1,024 ≈ 61.7 B for the tokenizer and 250,000 × 256 × 1,024 ≈ 65.5 B for the generator (paper Section C.2). The time they take depends on the hardware, and no time is claimed here.

## Sampling and evaluation

The evaluation follows the reference's OWT protocol (decision D35). The metric is generative perplexity, scored by the judge model.

- **Samples**: seeds `[42, 0, 1, 2, 3]`, 256 samples per seed, sampling batch size 8, fp32, no autocast. `torch.manual_seed(seed)` runs once per seed (`scripts/eval_shard.py` line 153).
- **Sampling**: temperature 1.0, `top_k` 0, `top_p` 1.0. Level codes are sampled from the generator (D31). The final token pick is argmax (D32).
- **Levels**: all `K` levels come from the generator, one forward pass per level, with learned queries for level 0 (D15). The unigram level-0 mode of the reference is not used (D36).
- **Text**: the ids are cut at the first EOS only when `cut_at_eos` is true, which it is not for OWT. Then every EOS and PAD id is removed, and the rest is decoded with the family's tokenizer and `skip_special_tokens` (`scripts/eval_shard.py` lines 130 to 134).
- **Perplexity**: the judge tokenizer encodes `prompt + text` with truncation at 1,024 tokens. The prompt is empty. The NLL of each token is taken from the logits at the previous position, and the first token has no prediction, so every other token is scored (`evaluation/gen_ppl.py` lines 56 and 80 to 124). For each seed, `ppl = exp(Σ NLL / Σ tokens)` over all samples of that seed (lines 134 to 140). The judge batch size is 32 (`scripts/eval_merge.py` line 99).
- **Report**: the mean over the five seeds, and the standard deviation with `ddof = 1` (`scripts/eval_merge.py` lines 105 to 106).
- **Entropy**: the natural-log unigram entropy of the family's tokenizer encoding of each decoded sample, averaged over samples (`scripts/eval_merge.py` lines 90 to 101; paper Section 4.2).

Two reference details are not reproduced exactly, and the report says so. The reference runs the seeds in two shards (`scripts/eval_shard.py` line 83), so its random streams differ from ncmorph's single run. The reference's default for OWT uses the unigram level-0 mode, which D36 replaces. No claim of matching the paper's numbers is made, and the report states the protocol used.

## Gates

A gate is passed when every check it names holds and its result is recorded in PROJECT.md or in [research/checks.md](research/checks.md).

- **M0a `inspect`. Done.** Accepts the 13 header fixtures of "Source checkpoints" and rejects synthetic variants with one error per feature (for example a Qwen3 fixture with `q_norm` removed).
- **M0b source decoder, F32.** The decoder of "Source decoder: exact computation", in F32. Logits match transformers v5.20.2 on tiny random models of each family, with the bounds of D53.
- **M0c real checkpoints.** SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B and Llama 3.2-1B (D12). Logits match the reference on the same token ids, within the D53 bound. The tests read checkpoints from the directory named by `NCMORPH_CHECKPOINT_DIR`, and they skip when it is unset. Llama 3.2-1B is the largest, at about 1.24 billion parameters (D6).
- **M1 NCM dialect.** In source mode, the NCM body reproduces the M0b hidden states within the bound. In NCM mode, masks and per-level positions equal the pinned reference's on random tiny NCM models built from the pinned classes (`models/next_concept.py`).
- **M2 transplant.** Every transplanted tensor is bit-identical to its source after widening. The report lists every NCM tensor (exact, initialized or new) with the SHA-256 of its widened F32 bytes, and every source tensor (exact, initialized or not used).
- **T1 and T2 tokenizers.** Ids equal the T0 reference on the fixed corpus for every family in the tokenizer table, and decoded text equals T0's decoder output.
- **V1 released WMT14 tokenizer.** The converter of "Released WMT14 tokenizer conversion" runs, and the converted tensors read back with bytes equal to the loaded tensors. The Zig VQ encoder, quantizer and decoder match the Python reference on fixed ids (V3).
- **V2 trained tokenizers.** On the validation split, token-exact reconstruction with the full hierarchy is at least 99.9 percent, and codebook utilization is at least 80 percent (D37). The paper's OWT values are 99.99 and 83.10 percent (Table 1). The same check applies to the reused WMT14 tokenizer on OpenWebText windows of 64 tokens (D44).
- **M3 generator training.** The run completes its update count with no non-finite loss. The validation loss at the last update is lower than at update 1,000. The evaluation protocol completes for all five seeds, with the mean and the standard deviation recorded.
- **M4 export and reload.** Reloading an exported directory gives logits within the D53 bound of the in-memory model, on the same ids. For models of at most 1.3 billion parameters this check runs in Zig. For larger models it runs in Python on the exported files (D6).

## Export format

An exported NCM model is a directory of four files (decision D26):

| File | Contents |
|---|---|
| `ncm.safetensors` | generator tensors: `transformer.wte.weight` and `.bias`, `transformer.wle.weight`, `transformer.blocks.{i}.ln_1.weight`, `transformer.blocks.{i}.attn.c_attn.weight` and `.bias` (q, k and v rows in that order), `transformer.blocks.{i}.attn.c_proj.weight` and `.bias`, `transformer.blocks.{i}.attn.q_norm.weight` and `k_norm.weight` (Qwen3 only), `transformer.blocks.{i}.ln_2.weight`, `transformer.blocks.{i}.mlp.w_gate.weight`, `w_up.weight`, `c_proj.weight`, `transformer.ln_f.weight`, `l0_queries`, `lm_heads.{j}.proj.weight`, `lm_heads.{j}.log_inv_T`, `lm_heads.{j}.bias` (when present), and `post_upsample_convs.{j}.weight` and `.bias` |
| `tokenizer.safetensors` | VQ tensors with the reference attribute names: `encoder.wte.weight` (the prefix token table, D58), `encoder.wpe.weight`, the `decoder.*` tensors, `vq.embeddings.{j}.weight`, and the level-0 resampling and `quant_resi` tensors (`wmt14_tokenizer_keys.md` lists the released set) |
| `ncm.json` | source family and checkpoint revision, `K`, level sizes, `L`, codebook sizes, `d`, every generator flag from the table above, the tokenizer route (`wmt14` or `trained`), and for the released tokenizer the file's size, SHA-256 and stored dtypes (D59) |
| `NOTICE` | the licence and attribution text for each part, as set out in "Licences and notices" |

The safetensors `__metadata__` map holds `ncmorph_format` (value `1`), `source_repository` and `source_revision`.

**Writer rules** (`safetensors/src/tensor.rs` at v0.8.0; `software_sources.md`, section C):

- Layout: an 8-byte little-endian `u64` header length `n`, then `n` bytes of compact JSON, then the data. The JSON is padded with spaces (`0x20`) to a multiple of 8 bytes (lines 254 to 255). `n` is the padded length (line 291).
- Order: entries are sorted by dtype in descending order of declaration, then by name ascending (line 230). Data follows the same order with contiguous offsets and no per-tensor padding.
- Each entry is `{"dtype": …, "shape": […], "data_offsets": [start, end]}`, in that field order. `__metadata__`, when present, is a map of strings to strings.
- Dtype names are the upstream names (`BF16`, `F16`, `F32`, and the others).
- The reader accepts a header of at most `MAX_HEADER_SIZE = 100000000` bytes (line 10). It rejects fewer than 8 bytes, a header that is not UTF-8 JSON, offsets that are not contiguous, and any byte count that does not equal shape × bit size / 8. It uses checked arithmetic for every offset and size sum, and an overflow is an error.

## Licences and notices

- **ncmorph's code**: Apache-2.0, as the README states.
- **NCM-derived code and the released tokenizer tensors**: MIT, copyright line `Copyright (c) 2025 hermann kumbong`, with the permission and warranty text (`LICENSE` at the pinned commit, read 2026-10-26). MIT requires the notice to stay with every copy or substantial portion. An independent implementation from the paper is outside that condition (`research/deep_research_digest.md`, claim C12).
- **Llama 2, Llama 3 and Llama 3.2 sources**: the generator tensors are derivatives of Llama Materials, so the Llama Community License applies to them. The licence texts were read on 2026-10-26 (`checks.md` section 8). The clauses that bind an export are:
  - Llama 3.2: a copy of the agreement goes with the materials; "Built with Llama" is shown on any related product or documentation; a distributed AI model derived from the materials has "Llama" at the beginning of its name; the notice file says "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved."; the Acceptable Use Policy applies; a licence from Meta is needed above 700 million monthly active users.
  - Llama 3: the same, with "Built with Meta Llama 3", a name beginning with "Llama 3", and the notice "Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.". Its Section 1.b.v bars using the materials or their outputs to improve other large language models, except Llama 3 and its derivatives.
  - Llama 2: a copy of the agreement goes to any third party that receives the materials, and the notice says "Llama 2 is licensed under the LLAMA 2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.". The 700 million user limit applies.
- **Qwen and Mistral sources**: Apache-2.0. The export keeps the source's notice.
- **Export `NOTICE`** (decision D45): the export writes a `NOTICE` file that contains the licence line of each source family and the notice text from the table above. For a Llama source it also contains the "Built with" statement and the name rule, and it says that the source licence applies to the whole export. ncmorph does not ship the Llama agreement text. The file says where to obtain it, which is the gated source repository.
- **Distribution**: ncmorph distributes no source weights and no converted weights. The released NCM tokenizer tensors are the MIT material above.

## Risks

- **Compute.** The paper's OWT budgets are about 62 B and 66 B positions. Their time depends on the hardware, which is the owner's. The budgets are fixed here, and no time is claimed.
- **Reuse of the WMT14 tokenizer.** It was trained on WMT14 text, and Qwen3's generators are trained on OpenWebText through it. Decision D44 measures that before training.
- **Dormant codes.** A code that is never selected keeps its `N(0, 1)` start, and under EMA it stays far from the data scale for many steps (`research/reference/vq_forward_semantics.md`, section 5; derived from the equations, not measured). The reference's dead-code restart is off for OWT. Utilization is therefore measured at the end of training (D37), and a low value is a failed gate, not a reason to change the recipe.
- **Reference differences.** The paper and the shipped configurations disagree on several values, and each disagreement is settled by a decision (D43).
- **Numerical parity.** F32 in Zig and BF16 in the reference differ. The bounds of D53 account for that, and the largest measured difference is recorded after each run. The reference's VQ operators (`F.interpolate` modes and GroupNorm) are PyTorch's, and parity is measured under the pinned torch (D5).
- **Unpickling.** The released tokenizer is a pickle. Only the converter loads it, under D41.

## Decisions

Each decision names its basis. Numbers are final for this file. The decision log below gives the dates.

- **D1. Family scope.** Llama 2, Llama 3.0, Llama 3.2, SmolLM2, Qwen2.5, Qwen3 (all dense sizes), Mistral v0.1 and v0.3. Basis: the 13 fixtures pass the manifest check, and all of them are the same decoder core with flags (transformers v5.20.2 model files).
- **D2. Local input only.** ncmorph reads checkpoints and data from local directories and downloads nothing. Basis: model access and licence acceptance belong to the user.
- **D3. Source embeddings and the input table.** The source `embed_tokens` is not transplanted, because its width is the source hidden size, and the NCM generator never reads a source token table. The prefix ids go through the VQ encoder's table (D58). That table belongs to the tokenizer: it is learned with a trained tokenizer, or it is the released table for Qwen3. The source `lm_head` is not used, because the NCM heads are codebook-tied (D18). Basis: `next_concept.py` lines 546 to 558; paper Table 5 (`d` = 384); ROADMAP step M2.1.
- **D4. Tokenizer order.** T0 (the pinned Python tokenizer) is the reference from M0c. T1 (byte-level) and T2 (SentencePiece-style) follow M2 in the roadmap. Basis: the model-core gates do not depend on tokenizers.
- **D5. Training language.** Python with `torch==2.13.0` in `tools/`. Zig does inference, export and verification. Basis: the reference has no training code (`ncm_facts.md` C5), and the trainers need PyTorch. The reference's `requirements.txt` leaves torch unpinned, so the pin is set here.
- **D6. Size limit and activation dtype for the Zig forward pass.** Models of at most 1.3 billion parameters run in Zig for M0b, M0c, M1 and the M4 reload; F32 fixtures use F32 arithmetic, BF16 real checkpoints use BF16 arithmetic per D64, and F16 storage is widened and currently uses F32 arithmetic. Larger models are converted and exported by copying BF16 bytes, and they are trained and reloaded in Python. Basis: the memory of an F32 copy, which is about twice the BF16 size; the user explicitly selected source BF16 arithmetic for BF16 real-checkpoint parity, and D53 remains unchanged.
- **D7. Golden references.** They are generated at test time by the pinned Python helper in `tools/`. The repository commits the helper, its requirements file and text fixtures only. Basis: AGENTS.md data rules.
- **D8. Attention pattern.** `block_diagonal`. All ten released configurations set it (`research/deep_research_digest.md`, claim C10; `ncm_facts.md` A2 step 5). `block_causal` is not implemented, because no released configuration uses it and its decoding differs from its training (`ncm_facts.md` B).
- **D9. Generator shape.** Blocks, width, heads, KV heads, head dimension and MLP width equal the source's. Basis: the M2 gate needs bit-identical transplants.
- **D10. Tokenizer per source.** A released NCM tokenizer is reused only when its token set and its pre-tokenizer both match the source exactly. Otherwise a tokenizer is trained for that checkpoint. Basis: the owner's confirmation on 2026-10-26. Each released configuration pairs its own pre-tokenizer with its own VQ input size (`config/eval/*.yaml`, `pre_tokenizer` and `in_vocab_size`).
- **D11. RoPE base default and stored frequencies.** A missing `rope_theta` means 10000. The stored `inv_freq` buffers are accepted and not loaded. Basis: transformers' `RotaryEmbeddingConfigMixin.default_theta` and `_adjust_missing_and_unexpected_keys` at the pinned commit (`checks.md` section 9, carried from an earlier pass).
- **D12. Real checkpoints in M0c.** SmolLM2-135M, Qwen2.5-0.5B, Qwen3-0.6B and Llama 3.2-1B, read from the directory named by the environment variable `NCMORPH_CHECKPOINT_DIR`, with one directory per fixture. Basis: they cover tied heads, QKV bias, QK-norm and `llama3` RoPE, and each fits D6.
- **D13. QK-norm.** `use_qk_norm` is true only for Qwen3. Basis: only Qwen3 has `q_norm` and `k_norm` tensors (fixtures).
- **D14. "Vocabulary" means the token set.** The owner chose this reading on 2026-10-26. Table sizes, including padding, are not compared. Consequences: Qwen3 reuses the WMT14 tokenizer (basis in "Tokenizer route per family"); Qwen2.5 is trained, because its token set lacks ids 151665 to 151668; the other families are trained.
- **D15. Level 0 is predicted by the generator.** `query_level0` is true for every family. The generator predicts all `K` levels from learned queries `l0_queries`, and there is no unigram draw and no prefix read-out. Basis: paper Algorithms 1 and 2 predict every level from `ẑ⁽⁰⁾ = 0`, and the paper never describes a unigram (`ncm_facts.md` A4). With a zero input, the first level's positions would be identical, so learned queries are needed. Three released configurations already use learned queries (`xsum`, `lm1b-*`, `sudoku-*`). OWT and TinyStories use the unigram (`owt.yaml` line 60).
- **D16. Loss weights.** `α = 0`, so every code position has the same weight (Eq. 10). Basis: paper Appendix A.2 identifies `α = 0` with the ELBO generation term, and the paper does not state the `α` of its main runs.
- **D17. Corruption schedule.** Codes are replaced uniformly from the same level's codebook, at rate `r_k = 1 − (k − 1)/(K − 1)`: 1.0 at level 1, 0.0 at level `K`, and a mean of 0.5 over the levels. Basis: paper Section 3.4 (per-level replacement) and Appendix F.3 (rates decrease linearly across levels); Appendix F.2.1 selects a mean rate of 0.50. The paper gives the trend and not the endpoints, so the endpoints are fixed here.
- **D18. Output heads.** Codebook-tied, one per level, with a learned scale (`log_inv_T`, initialized to 0) and a learned per-code bias. Basis: `owt.yaml` lines 64 to 66, the released OWT configuration.
- **D19. Trained tokenizer shape.** The `owt.yaml` structure, with the paper's OWT column for the sizes: `K = 32`, levels `k²`, `V_k = 4,096`, `d = 384`, `L = 1,024`, one decoder block, six heads, dropout 0.3. Basis: the architecture details the paper omits come from `owt.yaml`, and the sizes come from paper Table 5 (D43).
- **D20. Generator recipe and RoPE.** The paper's OWT column of Table 6 for training, with cosine decay after warmup. The generator's RoPE type and base equal the source's. Basis: Table 6 and Appendix F.1.2 (cosine). The paper does not state the warmup shape, so warmup is linear. The transplanted attention keeps its frequencies, which the source-mode test checks exactly.
- **D21. Tokenizer recipe.** The paper's OWT column of Table 5, with a cosine decay after warmup, BF16 mixed precision and seed 42. Basis: Table 5 and Appendix C. The warmup shape is not stated in the paper, so warmup is linear.
- **D22. Corpus, split and packing.** OpenWebText (CC0-1.0), `train` split. Validation is every document with index divisible by 1,000. Packing and shuffling as in "Training". Basis: the OWT row of the paper, and no validation split is published.

- **D77. Packed training-window storage.** Store fixed-length windows outside the repository as contiguous little-endian unsigned 32-bit token IDs, with `L` IDs per window; a text index records the SHA-256 of each window's bytes in output order. Assign global document indices in lexicographic shard order and row order within each shard. Basis: token IDs fit in 32 bits for the pinned source vocabularies, and fixed-width windows allow seeded shuffling without holding the corpus in memory. The pinned OpenWebText parquet metadata contains 8,013,769 documents: 8,005,755 training and 8,014 validation documents under D22 (80 shards; see `research/checks.md`, Phase 8).

- **D78. Memory-bounded VQ training.** A global batch may be processed in smaller pieces when all assignments and EMA statistics for each level are accumulated over the whole batch before that level's single EMA update. Retain the assignments and replay the batch for gradients against the updated codebooks, with one dropout cutoff and one optimizer update per global batch. Keep the dropout realization for replay. Basis: the pinned `models/quant.py` `forward` assigns against the old codebook and updates EMA once before looking up the new embeddings; a naive per-piece EMA update would change this behavior. Floating-point reduction order may differ from a monolithic batch, so a tiny comparison must verify the implementation within D53 rather than claim bitwise equality.

- **D79. Resumable VQ training.** Save training checkpoints at optimizer-step boundaries outside the repository using safetensors for tensor state and JSON for scalar state, never pickle. Preserve the model, EMA and optimizer state, all random streams, current data order and position, update count, schedule and data identity; reject a mismatched resume. Publish a validated complete generation atomically and retain a previous generation. Basis: D21's full budget is long-running on the owner's hardware, and restoring all state preserves the training sequence across interruptions. A tiny interrupted-versus-uninterrupted test must verify resume behavior.

- **D80. Resumable corpus packing.** Keep persistent window spools and atomic document-boundary progress outside the repository, recording the input and tokenizer identity, packing parameters, document position, partial windows and committed spool lengths. On restart reject mismatched inputs and truncate uncommitted spool tails; regenerate output from the committed spools with D77's order and hashes. Publish the summary only after all output is complete. Basis: packing the pinned corpus takes long enough to be interrupted, and anonymous temporary spools otherwise lose all progress. Test interrupted and uninterrupted runs for byte-identical output.

- **D81. Packed-data provenance.** Retain the input file identities, tokenizer digest and packing parameters in the completed packing summary, and bind the summary identity to resumable training state. For data packed before this decision, require an independently recorded JSON provenance record through the trainer's `--provenance-file` option instead of inventing missing identities. Basis: the output windows alone cannot establish which source tokenizer and corpus produced them, and the recovery checkpoint is removed when packing completes.

- **D82. Ordered batch tokenization.** Tokenize bounded batches with the pinned tokenizer's batch API, then consume results in document order using the unchanged split, window and checkpoint rules. Verify byte-identical output and interruption recovery against single-document tokenization. Basis: batch tokenization can use the tokenizer's parallel implementation while keeping D77 and D80 semantics.

- **D83. System erff for VQ GELU.** Extend D55's system C math dependency to `erff` for the VQ decoder's exact GELU, using `0.5*x*(1+erf(x/sqrt(2)))`. Basis: pinned `models/vae.py` uses `nn.GELU()` and the system function provides the error function for its non-approximate form. The same dynamically linked glibc LGPL-2.1-or-later license check recorded in D55 applies; no library is vendored.
- **D23. Input table rows.** A trained tokenizer's input table has the source's configuration `vocab_size` rows, with no padding. The released WMT14 table has 151,680 rows, which is the Qwen3 token count rounded up to 64. Basis: the source's ids are dense, and the table must cover them.
- **D24. Unconditional prefix.** One EOS id from the family's `config.json`, so `P = 1`. Basis: paper Table 6 (`P = 1` for OWT) and `scripts/eval_shard.py` line 157.
- **D25. Decoding.** Remove EOS and PAD ids, decode with the family's tokenizer and `skip_special_tokens`, and do not cut at EOS for OWT. Basis: `scripts/eval_shard.py` lines 130 to 134 and `owt.yaml` line 181.
- **D26. Export.** The four-file format of "Export format". Basis: the reference defines no export, so this one is defined here, and the M4 gate tests it.
- **D27. Trained families and the Qwen3 route.** Llama 2 7B, Llama 3.0 8B, Llama 3.2 1B, SmolLM2-135M, Qwen2.5-0.5B, Mistral v0.1 and Mistral v0.3 are trained, one tokenizer each. The six Qwen3 dense sizes reuse the WMT14 tokenizer (`L = 64`, `K = 11`, levels `[1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64]`, `V = 4,096`, `d = 384`), with the generator recipe of D20 at `L = 64`. Basis: D10 and D14. Consequence: WMT14 text (see Risks).
- **D28. Zig version.** Zig 0.16.0 for every step up to M0c. Step Z1 then moves the project to the current stable release, which was 0.17.0 on 2026-10-26, and re-checks every standard-library call the code uses. Basis: every check in this repository was run on 0.16.0 (`zig version`), and 0.17.0 was published on 2026-10-01 (ziglang.org/download, read 2026-10-26).
- **D29. Standard-library names.** The names used are `std.crypto.hash.sha2.Sha256.hash`, `std.Random.Xoshiro256` with `random()` and `floatNorm`, `std.mem.readInt` and `writeInt`, `std.math.exp`, `std.testing.expectApproxEqAbs` and `expectApproxEqRel`, `std.Io.Dir.createFile`, `std.Io.File.writer`, `std.Io.Writer.print`, `std.unicode.utf8Decode` and `utf8Encode`, and `std.json.parseFromSliceLeaky`. Basis: checked against the 0.16.0 standard library (AGENTS.md; carried from an earlier pass). Step Z1 re-checks them.
- **D30. Initialization of new tensors.** The reference's weight choices are kept, with zero biases as an explicit adaptation. `l0_queries` is trunc-normal with std `sqrt(1/n_embd)` (`next_concept.py` line 242, with `init_std` from line 86). `transformer.wle` keeps PyTorch's `nn.Embedding` default (line 199), because the reference sets no initializer for it. Other new Linear and Conv weights keep PyTorch's default initialization, because the reference never calls `_init_weights` (`ncm_facts.md` B). The reference's `c_proj` normal initialization (line 305) is for models trained from scratch, and it is not used here. Basis: those lines of the pinned code.
- **D31. Level sampling.** Each level's codes are sampled at temperature 1.0 with `top_k` 0 and `top_p` 1.0. Basis: `owt.yaml` lines 176 to 178.
- **D32. Final token pick.** Argmax over the decoder logits, with no penalties. Basis: paper Algorithm 2 line 11 and `scripts/eval_shard.py` lines 99 to 101, which return argmax when no penalty is set.
- **D33. Classifier-free guidance and penalties are not implemented.** `cfg_scale` is 0.0 in every released configuration (`owt.yaml` line 179), and the OWT configuration sets no penalty key. The code paths exist (`models/next_concept.py` lines 528 to 540 and `scripts/eval_shard.py` lines 99 to 124), and ncmorph does not port them.
- **D34. `use_positional_encoding` is ignored.** Basis: no code reads it (`research/deep_research_digest.md`, claim C7). RoPE is the only positional mechanism.
- **D35. Evaluation protocol.** The protocol of "Sampling and evaluation": judge `gpt2-large` at the pinned commit, five seeds, 256 samples per seed, sampling batch 8, judge batch 32, truncation 1,024, and the pooled perplexity per seed. Basis: `config/eval/owt.yaml` lines 183 to 193, `scripts/eval_merge.py` lines 99 and 105 to 106, and `evaluation/gen_ppl.py`.
- **D36. Unigram level 0 is not used.** The reference's default OWT evaluation draws level 0 from a unigram (`scripts/eval_shard.py` line 70). ncmorph follows D15 instead. Basis: D15, and the paper's Algorithm 2.
- **D37. Gate thresholds and the utilization definition.** The VQ gate is 99.9 percent token-exact reconstruction and 80 percent codebook utilization, both rounded down from the paper's OWT values (99.99 and 83.10 percent, Table 1). Utilization is the mean over levels of the fraction of codes used on the validation split, counted fresh, and never read from the reference's cumulative counters (`quant.py` lines 522 and 534 to 541). The M3 gate checks completion and the direction of the loss, not a quality target, because the paper gives no quality target for an adapted model.
- **D38. Sliding window.** A query at position `q` attends to keys `kv` with `q − W < kv ≤ q`. Basis: `masking_utils.py` lines 76 and 95, at the pinned commit. The Mistral v0.1 window is 4096, and it applies to every layer (`configuration_mistral.py` line 80).
- **D39. Safetensors reader and writer.** The rules of "Writer rules", taken from `safetensors/src/tensor.rs` at v0.8.0. Basis: the upstream reader and writer at the pinned tag.
- **D40. Tokenizer pipelines.** The exact pipelines in `research/tokenizer_pipelines.json`, at the revisions of "Source checkpoints". Basis: each family's `tokenizer.json` at its pinned revision.
- **D41. The released tokenizer is read only by the converter.** The converter loads the pickle in a throwaway environment with `weights_only=False`, as the reference does. It checks the file's size and SHA-256 against the pins of the released checkpoint before loading. It takes the `'model'` mapping, requires exactly the 96 keys of the derivation under `strict=True` loading, keeps tensors only, and records the file's size and SHA-256. Basis: `tokenization/vqvae_tokenizer.py` line 18 and `scripts/eval_shard.py` lines 52 and 62 show the reference's load path, and AGENTS.md forbids ncmorph itself from loading pickles.
- **D42. Source file header.** The owner's block form for every language that has block comments. Zig files use the `//` form, which is the same text, because `zig ast-check` rejects the block form. Basis: `zig ast-check` exit codes in `checks.md` section 7.
- **D43. Paper or shipped configuration.** Where they disagree, the training settings, the VQ codebook size (4,096) and the trained tokenizer's level sizes follow the paper's OWT column. Architecture details the paper omits (post-upsample convolution, tied heads, quantizer resampling, pre-quant GroupNorm) follow `owt.yaml`. The generator's shape follows the source (D9), so the reference generator's OWT shape (12 × 768 in the paper, 16 × 1,024 in `owt.yaml`) does not apply. The XSum configuration is not used. The commitment β is 0.25 from `owt.yaml` line 123, since the paper states no numeric value. Basis: the paper is the only published description of the OWT training settings, and no OWT checkpoint is published, so the configuration cannot be checked against weights.
- **D44. Reuse gate for the Qwen3 route.** Before a Qwen3 generator is trained, the released WMT14 tokenizer must pass the V2 reconstruction gate on OpenWebText windows of 64 tokens, using the Qwen3 ids. If it fails, a Qwen3 tokenizer is trained with the trained-family recipe at `L = 64`. Basis: D14 depends on the match, and the gate measures the transfer to OpenWebText.
- **D45. Notices.** The export's `NOTICE` file, set out in "Licences and notices". Basis: the Llama licence texts (read 2026-10-26), the MIT notice (read at the pinned commit), and the Apache-2.0 tags (read from the API responses).
- **D46. Out of scope: GGUF and quantized formats.** Basis: the task is safetensors in and safetensors out, and GGUF is a separate container that llama.cpp uses. Its layout was not verified, and it is not needed.
- **D47. Compute and budgets.** The paper's budgets are the spec. Time depends on the owner's hardware, and no speed is claimed. Basis: `ncm_facts.md` D (token accounting), and the rule that no unmeasured speed is stated.
- **D48. Where each language runs.** Zig runs the source decoder, the NCM generator's forward pass, the VQ encoder, quantizer and decoder forward pass, the tokenizers from T1 and T2, and export. Python runs the golden helper, the converter, the VQ and generator trainers, the T0 tokenizer, and the judge. Basis: D5 and the roadmap's phases.
- **D49. Packages.** Generic code lives in `packages/`, and it imports nothing from ncmorph or NCM. A package moves to its own repository only at the trigger in ROADMAP phase 11. Basis: the owner asked whether the libraries belong in a separate repository, and the M0a start waited for a consensus on the layout (session transcript).
- **D50. Gates run where they can.** M0c, M1 and the Zig parts of M2 and M4 run on models of at most 1.3 billion parameters. The Python parts run on any size. Basis: D6.
- **D51. The Llama name rule.** Export names are not changed by ncmorph. The `NOTICE` file states the name rule for distributed models. Basis: Section 1.b.i of the Llama 3.2 and Llama 3 licence texts places the naming duty on the person who distributes the model.
- **D52. VQ objective and the cutoff.** The VQ terms follow the reference code: `β·MSE(sg[ẑ_i], z) + MSE(ẑ_i, sg[z])` at every level, summed and divided by `K`, with the full cumulative reconstructions (`models/quant.py` lines 512 to 524). The reconstruction term is the cross-entropy between the decoder logits and the input token ids `x`, on the decoder input that nested dropout selects (paper Eq. 9). Nested dropout draws one Bernoulli per forward call (`quant.py` lines 457 to 461); on a hit the decoder receives `m` levels, with `m` uniform on `{min_keep, …, K−1}` (lines 525 to 530). Basis: the reference code is the only executable form of the VQ terms and produced the released tokenizers; the paper's Eq. 9 is the only statement of the cross-entropy, and `x` there is the token sequence.
- **D53. Numeric bounds for the golden tests.** (1) Matrix multiply: for each output, `|got − ref| ≤ (in + 1) · 2⁻²⁴ · Σₖ |xₖ · wₖ|`, with `ref` computed in f64 from the same f32 inputs. (2) Elementwise: relative error at most `(n + 8) · 2⁻²⁴` for RMSNorm and softmax, and `4 · 2⁻²⁴` for SiLU, `n` being the vector length. (3) RoPE cos and sin: absolute error at most `1e-6`, for positions 0 to 4095. (4) `llama3` inverse frequencies: relative error at most `1e-6`. (5) Hidden states and logits: `|Δ| ≤ 1e-4 · max(1, max|value|)` per position. Basis: planning values fixed before any comparison was run. The first two forms follow the rounding model of recursive f32 summation with unit roundoff `2⁻²⁴`. The largest measured differences are recorded after each run, and the bounds do not change after a run. B9's maximum absolute logit errors were `2.980232239e-8` for Llama default RoPE, `5.960464478e-8` for Llama3 RoPE, `5.960464478e-8` for Qwen2, `5.960464478e-8` for Qwen3, and `2.980232239e-8` for Mistral; the overall maximum was `5.960464478e-8`. N6's overall maximum hidden-state difference against Transformers was `4.768371582e-7`. N8's maximum hidden-state difference against M0 was `0.0`.
- **D54. Pre-tokenizer matchers are hand-written.** The regexes of the pre-tokenizers are implemented as hand-written matchers in `packages/pretokenizers`, with the leftmost-alternative semantics of Oniguruma. Basis: AGENTS.md requires a PROJECT entry and a licence check for any dependency. The patterns are fixed, and each uses only `(?i:…)` contractions, `\p{L}`, `\p{N}`, `\s`, character classes, and the quantifiers `*`, `+`, `?` and `{1,3}`.
- **D55. System libm for RoPE parity.** ncmorph's code depends on the Zig standard library, the packages in this repository, and the target system's C math library for the `f32` `powf`, `cosf` and `sinf` operations in B3 and B4. The build links the system math library on targets where it is separate; no library is vendored. Targets must provide these C math functions. Basis: the pinned Transformers 5.20.2 CPU tensor power path uses `std::pow` with f32 operands (`torch/include/ATen/cpu/vec/vec_base.h:689`); testing against all pinned default RoPE golden tables showed that system `powf` plus `cosf`/`sinf` matches within D53, while Zig's f32 `std.math.pow` does not. The current Linux host supplies glibc 2.41, licensed LGPL-2.1-or-later (`/usr/share/doc/libc6/copyright`); it is dynamically linked and not distributed by ncmorph. Basis for the exception: reproducing the pinned host f32 math is necessary for B3/B4 parity, and this narrow system-library use replaces no other dependency.
- **D56. NFC tables from Unicode 9.0.0.** The NFC normalizer for the Qwen families uses tables generated from the Unicode 9.0.0 data, with the hashes in "Pinned sources". Basis: the reference's normalizer is `unicode-normalization-alignments` 0.1.12, whose `UNICODE_VERSION` is 9.0.0 (`software_sources.md`, section B, `tables.rs`). Reproducing it exactly needs those tables, not the current Unicode data.
- **D57. Unicode 16.0.0 categories for the regexes.** The pre-tokenizer regexes and Digits use general categories from Unicode 16.0.0, with the hash in "Pinned sources". Basis: Oniguruma 6.9.10 ships Unicode 16.0.0 (`software_sources.md`, section B), and Rust 1.88's `char::is_numeric` uses Unicode 16.0.0 data (the same section). The Unicode 18.0.0 table from M0a stays for its own API.
- **D58. The prefix table is the VQ encoder's.** The prefix token table is `encoder.wte` in `tokenizer.safetensors`. The generator has no token table of its own, and the source `embed_tokens` is not used (D3). Basis: `models/next_concept.py` lines 546 to 558 send raw prefix ids through `vqvae.encoder.wte`.
- **D59. The converter keeps the stored dtype.** The converter accepts F32, BF16 and F16 tensors, keeps each one's stored dtype, records it in `ncm.json`, and never converts a dtype on disk. Basis: the stored dtype of `tokenizer.pt` is not known until the file is read, and copying bytes keeps the conversion exact (`research/converter/wmt14_tokenizer_keys.md`, section 8, note 5).
- **D60. Package split.** The repository is named `ncmorph-packages`, licensed Apache-2.0 with the same header, and starts at version `0.1.0`. Basis: the owner asked about a separate repository in the session transcript; this decision fixes the name, licence and version so that the split needs no further decision.
- **D61. Matrix multiply reduction block.** The blocked matrix multiply uses an inner reduction block of 32 elements. This is a fixed implementation choice so the plain reference can test a non-divisible tail (35 elements); it is a modest power-of-two chunk and does not change the reduction order. Basis: B1 requires a fixed block size and a tail case but specifies no value; 32 keeps each chunk bounded while making the tail test explicit.
- **D63. Real-checkpoint parity input and reference method.** M0c uses token ids `[1, 4, 7, 11]` at positions `[0, 1, 2, 3]`, without tokenization. The pinned Transformers 5.20.2 helper loads each local model on CPU with `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True)`, switches to evaluation mode, runs under `torch.inference_mode()` with `use_cache=False`, then writes F32 logits to `ncmorph_reference.json` beside the checkpoint. Schema version 2 records and the loader checks the Transformers and PyTorch revisions, eager backend, CPU/BF16 mode, oneDNN revision, thread count, CPU architecture and capability; absent or mismatched provenance invalidates the artifact and prints the regeneration command. Zig reads checkpoints under `NCMORPH_CHECKPOINT_DIR`, one named subdirectory per model, and reads this uncommitted reference file from each same directory. Basis: `[1, 4, 7, 11]` is the existing B9 fixed list and is in range for all four C2–C5 vocabularies; positions begin at zero per the source decoder specification; each config records BF16 weights; the pinned Transformers 5.20.2 `from_pretrained` accepts explicit `dtype`, `attn_implementation` and `local_files_only` (`modeling_utils.py`); PyTorch runtime details are verified in `research/checks.md`, section 34; and AGENTS.md requires no tokenization in C1. The expected reference output is local-only, BF16-weight eager Transformers inference on identical ids, compared against Zig BF16 arithmetic under D53.
- **D64. Preserve BF16 activation arithmetic for BF16 checkpoints.** When all loaded model tensors use BF16, the source decoder computes with BF16 activations and weights. Linear and attention matrix products accumulate in F32 and round their output to BF16; the pinned CPU PyTorch 2.13.0 probe confirms the result matches F32 accumulation followed by BF16 rounding and differs from rounding each product and partial sum. BF16 elementwise add/multiply and SiLU outputs are BF16. SiLU follows the pinned CPU implementation's F32 `x / (1 + exp(-x))` scalar kernel before BF16 output conversion (`aten/src/ATen/native/cpu/Activation.cpp`, `silu_kernel`, lines 1098–1112). RMSNorm computes variance and normalization in F32, casts normalized values to BF16, then multiplies BF16 scale weights with BF16 output. RoPE uses BF16 cosine/sine operands and BF16-rounded multiply and add results. Softmax computes in F32 and casts probabilities to BF16, per the pinned Llama eager attention implementation. F32 tiny-model references continue to use F32. Basis: Transformers 5.20.2 `modeling_llama.py` RMSNorm, `apply_rotary_pos_emb`, `LlamaMLP`, and eager attention source plus direct operations on the pinned CPU PyTorch 2.13.0+cpu runtime (evidence in `research/checks.md`, section 22); the prior F32 diagnostic passed D53 while BF16 reference logits did not, showing that changing the reference dtype would conceal activation-rounding differences. D53 is unchanged.
- **D65. Eager attention backend for BF16 decoder parity.** The real-checkpoint helper and the dedicated tiny BF16 Qwen3 attention case explicitly select Transformers' eager attention implementation, because the Zig decoder implements that operation sequence. The other pre-existing F32 tiny golden cases retain their original configured backends; B9's F32 outputs are unchanged. Basis: the pinned PyTorch 2.13.0 CPU probe found BF16 logits differed by up to `0.0009765625` between Transformers SDPA and eager for the same tiny model, beyond D53's `1e-4` hidden/logit tolerance. The eager choice is scoped to BF16 references where the backend discrepancy was measured. See `research/checks.md`, section 24.
- **D66. Continue BF16 parity debugging.** For this session, authorized debugging may continue after a real-checkpoint D53 failure, preserving every failed result and leaving D53 unchanged; C3–C5 remain sequential and may run only after the preceding real gate passes. No general BF16 dot-product reduction order is specified by this decision. Basis: the pinned CPU PyTorch implementation has multiple shape- and backend-dependent paths (including the `BlasKernel.cpp` fallback and vectorized GEMV); the session authorization permits evidence-based debugging while keeping D53 fixed. See `research/checks.md`, sections 26, 27 and 29.
- **D67. BF16 linear reduction for the pinned oneDNN path.** The BF16 scalar linear kernel emulates the pair operation documented for AVX-512 BF16 `VDPBF16PS`: widen each BF16 pair, perform the instruction's two F32 fused multiply-add accumulations with round-to-nearest-even, and flush F32 subnormal inputs and outputs to signed zero. For the pinned Torch 2.13.0+cpu / oneDNN 3.12.0 build at commit `80afa71049cd69a3df32adcccb623b12cd7baa22`, the captured SmolLM2 projections dispatch to `brg_matmul:avx512_core_bf16`. For non-transposed source matrices, the AVX-512 heuristic selects K blocks of 512 when K≤1024 and 1024 when K>1024; when F32 accumulation is written to BF16 output and a block is split, oneDNN accumulates block partials in F32 before BF16 conversion. Zig follows that rule. This decision is scoped to the verified pinned backend, ISA and captured operations; it does not claim parity for other PyTorch backends, oneDNN versions, CPU ISAs or blocking heuristics. It adds no runtime dependency and copies no oneDNN source. The inspected oneDNN source is Apache-2.0-licensed. Basis: exact pinned oneDNN source in `/tmp/onednn-v3.12`, `cpu/matmul/cpu_matmul_list.cpp`, `cpu/x64/matmul/brgemm_matmul_utils.cpp`, `cpu/x64/matmul/brgemm_matmul.cpp`, and `cpu/x64/brgemm/jit_brgemm_kernel.cpp`; Intel Architecture Instruction Set Extensions Programming Reference, VDPBF16PS operation pseudocode; a native-intrinsic comparison on 16,000,000 lanes and captured projection comparisons (see `research/checks.md`, section 30). D53 remains unchanged.
- **D68. RMSNorm variance reduction for the pinned CPU reference.** Compute the F32 sum of squares using the pinned PyTorch 2.13.0 CPU contiguous-inner reduction order: the AVX2 `Vectorized<float>` width of eight, with the source `SumKernel.cpp` four-level/four-row cascade and scalar lane fold, then divide by the row width. This order is used for F32 and BF16 RMSNorm so tiny and real references follow the same pinned CPU behavior. Basis: the pinned `SumKernel.cpp` source and installed wheel symbols identify DEFAULT and AVX2 `sum_kernel_impl` variants but no AVX-512 specialization; simulating that source algorithm with eight lanes reproduces all four captured Qwen2.5 layer-2 sum-of-squares outputs bit-for-bit, while a 16-lane simulation does not. This is scoped to the pinned CPU reduction implementation and does not claim equivalent dispatch for other PyTorch builds or backends. No PyTorch code is copied and no dependency is added. See `research/checks.md`, section 31.
- **D69. Zig 0.17.0 upgrade.** Z1 moves the supported toolchain to the official Zig 0.17.0 release. The build minimum is 0.17.0. The only required source migrations found were replacing removed `Build.args` access with `Run.addPassthruArgs()` and replacing removed array multiplication syntax with an explicit array initializer. All D29 standard-library names remain available in 0.17.0. Basis: the official [0.17.0 release notes](https://ziglang.org/download/0.17.0/release-notes.html), source inspection in the checksum-verified 0.17.0 distribution, a clean-cache full test run, format and repository checks, and sequential C2–C5 gates (evidence in `research/checks.md`, section 35). No D53 tolerance or phase-5 scope changed.
- **D70. Scope the level-embedding row count to the selected level-0 mode.** The NCM dialect sets `query_level0: true` (D15), so its level embedding has K rows. The pinned class instead allocates one row per predicted level: `len(levels) - (0 if query_level0 else 1)`. Basis: `models/next_concept.py` `NextConceptTransformer.__init__` at the pinned commit, and `config/eval/owt.yaml` showing `query_level0: false`; the table now distinguishes the dialect from the reference's other mode.
- **D71. Source RMSNorm in the NCM dialect.** Use RMSNorm and the source model's `rms_norm_eps` for the transplanted decoder norms. This deliberately adapts the pinned generator's default LayerNorm, whose epsilon is fixed at 1e-6; the pinned `RMSNorm` is otherwise only used for optional QK-norm. Basis: exact source-model hidden-state parity is the M1 gate, while the source decoder's norm and epsilon are pinned in `research/transformers_formulas.md`; ROADMAP N2 implements and tests this adapter.
- **D72. Independent NCM bias flags.** Split the pinned generator's shared `config.bias` into `qkv_bias`, `o_bias`, `mlp_bias` and `norm_bias`, following the source checkpoint flags. QKV is one fused bias; MLP bias covers both input projections and the output projection; norm bias adds per-width bias tensors to `ln_1`, `ln_2` and `ln_f`. The VQ input projection retains its bias under D9. Basis: pinned `models/common.py` uses the same shared flag for all attention/MLP linears and LayerNorms, while source-family flags are independently defined in PROJECT.md; synthetic manifest cases exercise each group alone.
- **D73. Generator post-upsample convolution levels.** Apply one Conv1d residual to each target level from `_lvl_first` through the last, where the pinned generator sets `_lvl_first = 0` for `query_level0: true` and `1` otherwise. This dialect uses `query_level0: true`, so its level-0 target is included. Keep this generator rule distinct from the trained VQ quantizer's convolution configuration in the tokenizer table. Basis: pinned `models/next_concept.py` `NextConceptTransformer.__init__` and `_apply_post_upsample_convs_multi`; a test-time pinned convolution output is compared against the Zig operation.
- **D74. Source-to-generator transplant mapping.** Use the mapping in "M2 transplant mapping" for every source decoder tensor represented by the generator. Assemble fused QKV weights and biases by concatenating the source Q, K and V outputs in that order; copy linear matrices without transposing because both source and destination use `[output, input]` shape. Copy the source normalization scales exactly after widening while retaining the source-epsilon RMSNorm behavior of D71. Source token embeddings, the source language-model head and optional rotary frequency buffers have no stored generator counterpart. Basis: the pinned `models/common.py` (`SelfAttention`, `MLP`, `Block`) and `models/next_concept.py` (`NextConceptTransformer.__init__`), source names and shapes in `packages/hf_config/manifest.zig` and checkpoint fixtures, destination shape declarations in `src/ncm_manifest.zig`, and QKV consumption in `src/ncm_model.zig`.

- **D75. Repeatable transplant initialization.** Use `std.Random.Xoshiro256`, with seed 42 by default (and an explicit seed recorded in each transplant report), in a fixed target inventory order. Draw `transformer.wle.weight` from the standard normal distribution. Draw new Linear weights uniformly from `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`; initialize biases to zero as D30 requires. Draw `l0_queries` from a normal distribution with standard deviation `sqrt(1/n_embd)`, rejecting values outside the absolute interval `[-2, 2]`. Post-upsample convolution weights and biases are zero (D73); RMSNorm weights are one. This reproduces the specified distributions, not PyTorch's random bit stream. Basis: the pinned `models/next_concept.py` constructor, PyTorch 2.13.0 [`Linear.reset_parameters`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/modules/linear.py), [`Embedding.reset_parameters`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/modules/sparse.py), and [`trunc_normal_`](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/nn/init.py); seed 42 follows D20 and meets M2.2's repeatability requirement. The pinned constructor does not specify an initialization seed.

- **D76. Added-token decoding and case folding.** Reproduce the pinned tokenizer's normalized added-token spelling when resolving IDs and filtering special tokens, and its Unicode 16 case folding for the case-insensitive contraction alternatives, including U+017F for `s`. Basis: tokenizers v0.23.3 `tokenizer/mod.rs` (`TokenizerImpl::decode`) and `added_vocabulary.rs` (`simple_id_to_token` and normalized cache), and Oniguruma's Unicode 16 case-fold table; see `research/checks.md`, Phase 7.

## Decision log

- 2026-10-26: NCM means Next-Concept Modeling (HazyResearch/ncm), not any other expansion.
- 2026-10-26: The owner specified the block-comment header for every source file. Zig does not parse it, so Zig uses the `//` form (D42).
- 2026-10-26: Llama, Qwen2, Qwen3 and Mistral share one decoder core, configured by flags. SmolLM2 is a Llama-architecture checkpoint and uses the same core.
- 2026-10-26: Generic Zig code lives in `packages/` in this repository, with no NCM imports, and it moves out at the trigger in ROADMAP phase 11 (D49, D60).
- 2026-10-26: The library adds a BF16 type and Unicode general-category tables, as the owner requested.
- 2026-10-26: Coding style is in AGENTS.md. Its rules are the owner's, and they apply to every language in the repository.
- 2026-10-26: M0a (`inspect`) is implemented in commit 3a771d5. The header fixtures were added after it, for Llama 2 7B, Llama 3.0 8B and the dense Qwen3 sizes 1.7B, 4B, 8B, 14B and 32B.
- 2026-10-26 (commit c833749): the stale intro and the duplicate Zig header note were fixed.
- 2026-10-26: Llama 2 exposed two gaps, both fixed (D11): a missing `rope_theta` now means 10000, and per-layer `inv_freq` buffers are accepted.
- 2026-10-26: The owner confirmed D10 (reuse only on an exact match of token set and pre-tokenizer, otherwise train), and chose reading A of "vocabulary" (D14).
- 2026-10-26: Qwen2.5 differs from Qwen3 by four added tokens, so it is trained, and Qwen3 reuses the WMT14 tokenizer (D14).
- 2026-10-26: The Qwen3-0.6B tied head is byte-identical (SHA-256 `8f29acf5…`), which confirms the tied-head rule of the manifest check (`checks.md` section 3).
- 2026-10-26: The deep-research run `wf_3a9de5f1-8b7` completed, with 13 claims confirmed and 12 refuted. Its synthesis step had written open questions into this file, and they are closed by D43, D45 and D28 (`research/deep_research_digest.md`, section 4).
- 2026-10-26: Z1 moved the project from Zig 0.16.0 to the official Zig 0.17.0 release after C5. The minimum compiler version, toolchain instructions and two required source migrations are recorded in D69.
- 2026-10-26: The reference's OWT default draws level 0 from a unigram, and its final token pick is argmax. Both are settled by D15, D32 and D36.
- 2026-10-26: The state-dict key list of the released WMT14 tokenizer (96 entries) is derived by reading the pinned code. Running the reference was refused in this session, so the converter's run is the check (D41, V1).
- 2026-10-26: The VQ forward pass and losses were derived by reading the pinned code (`research/reference/vq_forward_semantics.md`). They correct two drafts of this file: `quant_resi` is not zero-initialized (D19), and the reference's VQ terms are not evaluated at the sampled cutoff (D52).
- 2026-10-26: The reference's `VQVAE.reconstruct` is broken (`models/vqvae.py` line 116). ncmorph does not call it (see "Trained VQ tokenizer", Reference defect).
- 2026-10-26: B1 fixes the later blocked matmul reduction chunk at 32 elements (D61), with a 35-element random test input to exercise a tail.
- 2026-10-26: C1 fixes real-checkpoint parity to token ids `[1, 4, 7, 11]`, positions `[0, 1, 2, 3]`, and pinned local CPU BF16 Transformers inference (D63).
- 2026-10-26: Real-checkpoint parity keeps BF16 activation arithmetic for BF16 checkpoints, with F32 accumulation and BF16 outputs at pinned Transformers tensor-operation boundaries; F32 tiny references remain F32 (D64).
- 2026-10-26: Decoder parity references use Transformers eager attention, and F.linear bias joins the F32 accumulator before BF16 rounding; tiny BF16 helper converts parameters while preserving F32 RoPE buffers (D65, evidence in `research/checks.md` section 24).
- 2026-10-26: The owner authorized evidence-based BF16 parity retries without changing D53; no universal BF16 dot reduction order was selected because the pinned CPU has multiple shape-dependent backends (D66; evidence in `research/checks.md` sections 26, 27 and 29).
- 2026-10-26: BF16 linear follows the source-verified VDPBF16PS pair-FMA semantics and the pinned oneDNN v3.12.0 AVX-512 K-block selection (512 through K=1024, otherwise 1024 for non-transposed inputs), only within the stated pinned-backend scope; no oneDNN code or runtime dependency is added (D67; evidence in `research/checks.md` section 30).
- 2026-10-26: RMSNorm's F32 sum of squares follows the source-verified pinned PyTorch CPU AVX2 cascade reduction (D68; evidence in `research/checks.md` section 31).
- 2026-10-27: N1 checked every NCM dialect field against the pinned implementation. Corrected the level-embedding row count to state the `query_level0` condition (D70), and documented the source-RMSNorm adaptation against the reference's LayerNorm baseline (D71).
- 2026-10-27: N3 splits the pinned generator's shared bias flag into the four source-facing groups and checks their NCM manifest tensors independently (D72).
- 2026-10-26: The numbering of this file is final. D15 to D61 replace the numbering of an earlier draft, and no earlier number is reused.

## Research record

- [research/checks.md](research/checks.md): every check, its method and its result. Entries marked "carried" were checked in an earlier pass.
- [research/ncm_facts.md](research/ncm_facts.md): the NCM fact sheet with line citations to the pinned commit.
- [research/software_sources.md](research/software_sources.md): tokenizers v0.23.3 and safetensors v0.8.0, with line citations.
- [research/tokenizer_pipelines.json](research/tokenizer_pipelines.json): the exact tokenizer pipelines of the eight families.
- [research/transformers_formulas.md](research/transformers_formulas.md): the transformers v5.20.2 formulas, extracted with line numbers.
- [research/deep_research_digest.md](research/deep_research_digest.md): the completed deep-research run.
- [research/converter/wmt14_tokenizer_keys.md](research/converter/wmt14_tokenizer_keys.md): the derived state-dict keys of the WMT14 tokenizer, from reading the code (step V1).
- [research/reference/vq_forward_semantics.md](research/reference/vq_forward_semantics.md): the VQ forward pass, losses, resampling and EMA, from reading the code (steps V2 and V3).
- [research/README.md](research/README.md): the index of the research folder, and how to repeat the checks.


## References

- Paper: arXiv:2610.14162v1 (Kumbong et al., HazyResearch, 2026). Local text: Sections 3.4 and 4, Algorithms 1 and 2, Equations 9 and 10, Tables 1, 5 and 6, Appendices A, C, F.
- HazyResearch/ncm, commit `c70274273f489abc73832b2b98584a8483b6b203`: `config/eval/owt.yaml`, `config/eval/wmt14.yaml`, `config/eval/xsum.yaml`, `models/next_concept.py`, `models/quant.py`, `models/vqvae.py`, `models/vae.py`, `models/common.py`, `tokenization/vqvae_tokenizer.py`, `scripts/eval_shard.py`, `scripts/eval_merge.py`, `evaluation/gen_ppl.py`, `LICENSE`.
- transformers v5.20.2, commit `18329f1065382e54db698e242cf98bd3f3e5a766`: `modeling_rope_utils.py`, `models/llama/modeling_llama.py`, `masking_utils.py`, `dependency_versions_table.py`, `configuration_mistral.py`, `modeling_mistral.py`.
- tokenizers v0.23.3, commit `301277072263fb3830eb66eec7e47adadf76f1dc`: `pre_tokenizers/byte_level.rs`, `pre_tokenizers/digits.rs`, `pre_tokenizers/metaspace.rs`, `normalizers/unicode.rs`, `models/bpe/model.rs`, `decoders/byte_fallback.rs`.
- safetensors v0.8.0, commit `a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6`: `safetensors/src/tensor.rs`.
- Checkpoints and tokenizer files: the revisions in "Source checkpoints" and the Hugging Face repository `hazyresearch/ncm-checkpoints`.
- Llama Community Licence texts for Llama 2 (2023-07-18), Llama 3 (2024-04-18) and Llama 3.2 (2024-09-25), read 2026-10-26.
- Zig 0.16.0 (local compiler) and Zig 0.17.0 (ziglang.org/download, 2026-10-01).
- Corpus `Skylion007/openwebtext` at `79d93d786212f7344586290adb811d4ae6a1762c`, and judge `openai-community/gpt2-large` at `32b71b12589c2f8d625668d2335a01cac3249519`.
- Unicode data 9.0.0 and 16.0.0 (`UnicodeData.txt`, `CompositionExclusions.txt`), read from unicode.org on 2026-10-26; hashes in "Pinned sources".
