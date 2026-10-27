# ncmorph roadmap

This file lists every implementation step in the order it must be done, and it records what is finished. A new contributor or agent starts at the first step whose status is not done, once the steps it depends on are done.

- Scope, pinned sources, decisions and gates: [PROJECT.md](PROJECT.md)
- Rules for agents and the verification list: [AGENTS.md](AGENTS.md)
- The research behind the decisions, with checks and sources: [research/README.md](research/README.md)

The roadmap has no dates and no durations. The order of the steps and their "Done when" checks are the plan.

## How to use this file

1. Read the files in the "Read first" order of [AGENTS.md](AGENTS.md).
2. Start the first step whose status is not done. A step starts only when every step on its "Depends on" line is done.
3. A step is done when each of its "Done when" checks passes and the verification list in AGENTS.md has been run on that state. Then mark the step done here, and record each decision the step made in PROJECT.md.
4. If a step needs a fact that PROJECT.md does not state, research the fact first, and record its source and check in [research/checks.md](research/checks.md). Then add a decision entry to PROJECT.md with its basis, before code depends on it. No step leaves a question open.
5. Bounds and tolerances are set in PROJECT.md (decision D53), before any comparison is run. After a run, the largest difference found is recorded. The bound does not change after a run.
6. Steps that run pinned reference code, or that run the converter on the released pickle, need the owner's permission in the session that runs them. See "Steps that need the owner".
7. Commit only what the owner asks for. Committed: source code, tools, configs, text fixtures and hand-written expectations. Not committed: weights, datasets, caches, logs, and reference outputs that a helper generates.

## Status at a glance

| Phase | Milestone | Status |
|---|---|---|
| 1 | M0a `inspect` | Done |
| 2 | M0b source decoder in F32 | Done |
| 3 | M0c real checkpoints | Done |
| 4 | Z1 Zig upgrade | Done after M0c |
| 5 | M1 NCM dialect | Done |
| 6 | M2 transplant | Next: step M2.1 |
| 7 | Tokenizers T0, T1, T2 | Not started |
| 8 | Data, writer and VQ tokenizers | Not started. The Qwen3 route and the trained families are decided (D10, D14) |
| 9 | M3 generator training | Not started |
| 10 | M4 export and reload | Not started |
| 11 | Package split | Waiting for its trigger |

Phases run in this order because each depends on the one before. Within a phase, the "Depends on" line of each step gives its order. The Zig upgrade (phase 4) comes after M0c, so that every M0 result is measured on one toolchain (D28).

## Steps that need the owner

- **V1 converter run.** The released `tokenizer.pt` is a pickle, and loading it is the one place that unpickles a file (D41). The owner runs the converter, or approves the run in the session that runs it.
- **Runs of pinned NCM reference code.** Steps N7, V3 and V4 import or run the pinned NCM reference code. The owner approves each session that does. A session without that approval records the step as not run and does not run the code some other way. The key list in [research/converter/wmt14_tokenizer_keys.md](research/converter/wmt14_tokenizer_keys.md) was derived by reading the code, because running it was refused in the session that wrote it.
- **Data.** The owner downloads the OpenWebText parquet shards into a local directory (D2). ncmorph reads that directory and downloads nothing (step A1).
- **Compute.** The full VQ and generator runs (steps V2 and A2 at their full budgets) run on the owner's hardware. The budgets are fixed in PROJECT.md (D47).

## Phase 1: M0a `inspect` (done)

Gate: `inspect` accepts the 13 header fixtures listed in PROJECT.md ("Source checkpoints"), and rejects synthetic variants with one error per feature. No weight data is read.

Steps that are done:

- **1.1 Safetensors reader** (`packages/safetensors`): reads the 8-byte header length, the JSON header and the optional `__metadata__`. It checks contiguous offsets, byte size equal to shape times bit size, and file size equal to prefix plus header plus data. It parses `model.safetensors.index.json` and rejects unsafe shard names.
- **1.2 BF16 type** (`packages/bf16`): round to nearest, ties to even, from F32. Every NaN becomes `0x7FC0`. Widening to F32 is a bit shift. Checked against PyTorch 2.13.0 on 4,119 vectors, including subnormals and signed zero.
- **1.3 Unicode general categories** (`packages/unicode_categories`): generated from `UnicodeData.txt` for Unicode 18.0.0 by `tools/gen_unicode_categories.py`. Python's `unicodedata` (Unicode 15.1.0) differs only on characters assigned after 15.1 and on two recategorized characters. The tokenizer path needs Unicode 16.0.0 tables as well (step K0, decision D57).
- **1.4 Configuration and manifest** (`packages/hf_config`): family detection, flags, rejected features, the expected tensor list, shape and dtype checks, the tied-head note, and the index check. Every problem is collected, and each feature is reported once.
- **1.5 `inspect` command** (`src/main.zig`): reads `config.json`, the index when present, and each shard header. Exit status 0 when the checkpoint matches, 1 on problems, 2 on usage or I/O errors.
- **1.6 Header fixtures for 13 checkpoints** (`packages/hf_config/testdata`): each folder holds `config.json`, the exact header bytes of every shard, `sizes.json` and `provenance.json`. The Llama 2 7B, Llama 3.0 8B and Qwen3 1.7B, 4B, 8B, 14B and 32B fixtures were added after commit 3a771d5 and stay in the working tree until they are committed.
- **1.7 Fixes found by the new fixtures.** Llama 2's `config.json` has no `rope_theta`, so the default of 10000 is used (D11). Llama 2 files store per-layer `self_attn.rotary_emb.inv_freq` buffers, which are accepted as optional (D11).
- **1.8 Known gap, closed by step B0a.** `inspect` rejects `rope_parameters`, the key that transformers v5 writes for RoPE settings. The 13 fixtures use the older keys, so they pass.

Verification record for this state: see the entry for this pass in [research/checks.md](research/checks.md), section 14.

## Phase 2: M0b source decoder in F32

Goal: one decoder core, configured by flags, that runs a source model on the CPU in f32. Its logits must match transformers v5.20.2 on tiny random models of each family.

Gate: logits match transformers v5.20.2 on tiny random models, one per family in scope, generated with fixed seeds, within the bounds of decision D53.

**B0a. Accept the transformers v5 `rope_parameters` key.** Status: done.
- Depends on: phase 1.
- Why: the installed transformers 5.14.1 writes `rope_parameters` (for example `{"rope_theta": 10000.0, "rope_type": "default"}`) and no top-level `rope_theta` when it saves a config. `packages/hf_config/config.zig` rejects that key with "only rope_scaling is read in this version" (lines 315 and 316). Step B0 confirms that the pinned v5.20.2 writes the same key.
- Work: read `rope_parameters` when `rope_scaling` is absent, with its `rope_type`, `rope_theta` and the llama3 fields. If both keys are present and disagree, report a named error. A config in either form gives the same spec.
- Done when: a unit test shows that a config in the `rope_parameters` form gives the same spec as its legacy form; `inspect` on a directory holding such a config reports no error for the key; `zig build test` passes with the new test counted; `zig fmt --check .` is clean.

**B0. Golden helper, pins and bounds.** Status: done.
- Depends on: B0a.
- Work: the helper is `tools/golden_helper.py` (its name is fixed here). Its environment is built from `tools/requirements-golden.txt`, which lists the pins of PROJECT.md: `torch==2.13.0`, `transformers==5.20.2`, `tokenizers==0.23.3`, `safetensors==0.8.0`, on Python 3.13. The helper checks the installed versions and refuses to run when one is missing or different. For fixed seeds it writes tiny random models per family (`config.json` and safetensors, in a build cache, not committed) and the reference values as JSON: RoPE tables, layer outputs, and logits for fixed token ids.
- Done when: two runs with the same seed give identical files; the helper refuses to run with a wrong pin; and B0 records the version of `rope_parameters` that the pinned transformers writes (step B0a depends on it).

**B1. Matrix multiply** (`packages/kernels`). Status: done.
- Depends on: phase 1.
- Work: row-major f32 `y = x Wᵀ`, with `x` of shape `[T, in]`, `W` stored as `[out, in]` (as in the safetensors header), and `y` of shape `[T, out]`. Write the plain loop first. It stays in the tests as the reference for the blocked version, which comes second.
- Wiring: create `packages/kernels/root.zig` with the header and a `//!` doc comment. Register it in `build.zig` with `b.addModule("kernels", ...)` and add it to `library_modules`. Without that, `zig build test` runs no tests for the package, because the test loop covers only `library_modules`.
- Done when: (a) hand-checked small matrices pass; (b) for fixed seeds (`std.Random.Xoshiro256`, with the seed written in the test), including `T = 1` and an inner dimension that the block size does not divide, every output satisfies the D53 bound against an f64 reference computed in the test; (c) the package imports nothing from ncmorph or NCM (the grep check in AGENTS.md); (d) `zig build test` passes with more than 40 tests.

**B2. Elementwise and reduction kernels** (`packages/kernels`). Status: done.
- Depends on: B1.
- Work: RMSNorm, SiLU, softmax with the maximum subtracted, and the embedding row lookup. RMSNorm accumulates the mean of squares in f32, multiplies by `rsqrt(mean + eps)`, casts back, then multiplies by the weight. That is the order of `LlamaRMSNorm` (`research/transformers_formulas.md`, lines 52 to 69).
- Done when: (a) exact small cases pass: softmax of two equal logits gives 0.5 each, SiLU of 0 is 0, and RMSNorm of a constant vector is exact; (b) random inputs with fixed seeds satisfy the D53 bounds against an f64 reference.

**B3. RoPE, type `default`** (`packages/kernels`). Status: done.
- Depends on: B0.
- Work: rotate-half layout. Inverse frequencies `base^(−2i/d)` for `i` in `0 .. d/2 − 1`, computed in f32 in the order of `LlamaRotaryEmbedding` (`research/transformers_formulas.md`, lines 72 to 116). Positions are multiplied by the inverse frequencies in f32. The reference is the helper's f32 output, not an f64 port: an f64 computation of the same angles differs from that f32 result by more than the bound at large positions.
- Done when: cos and sin match the helper within the D53 bound for positions 0 to 4095, head dimensions 64 and 128, and bases 10,000 and 500,000.

**B4. RoPE, type `llama3`** (`packages/kernels`). Status: done.
- Depends on: B3.
- Work: port `_compute_llama3_parameters` (`research/transformers_formulas.md`, lines 5 to 92, from `modeling_rope_utils.py` lines 580 to 662) in f32, in the same operation order.
- Done when: inverse frequencies match the helper within the D53 bound at every index, for the Llama 3.2-1B parameters (`factor` 32, `low_freq_factor` 1, `high_freq_factor` 4, `original_max_position_embeddings` 8192, `rope_theta` 500000).

**B5. Attention** (`packages/kernels` and the decoder). Status: done.
- Depends on: B2, B3.
- Work: query heads have width `n_heads · head_dim`; key and value heads have width `n_kv_heads · head_dim`. Query head `h` reads KV head `h / (n_heads / n_kv_heads)`. Optional q, k and v biases. Optional QK-norm: an RMSNorm over `head_dim` on each head's q and k, before RoPE (Qwen3). The output projection maps `n_heads · head_dim` back to the hidden size.
- Done when: one attention layer of a tiny model with grouped-query heads, biases and QK-norm matches the helper's output within the D53 bound.

**B6. Causal and sliding-window masks.** Status: done.
- Depends on: B5.
- Work: causal mask for every family. Sliding-window mask for Mistral v0.1, with the rule `q − W < kv ≤ q` (decision D38), where `W` comes from `sliding_window` in the config (4096 in the real config).
- Done when: the masks equal the helper's masks for a sequence longer than a small test window.

**B7. Block, model and head.** Status: done.
- Depends on: B6.
- Work: pre-norm block: `h = x + attn(input_layernorm(x))`, then `out = h + mlp(post_attention_layernorm(h))`. Final norm. SwiGLU MLP: `down(silu(gate(x)) · up(x))`, with width `intermediate_size`. The head is tied to `embed_tokens` when `tie_word_embeddings` is true, and otherwise it is `lm_head`.
- Done when: a tiny model runs forward and returns logits of shape `[T, vocab_size]`.

**B8. Weight loading.** Status: done.
- Depends on: B7, B0a, phase 1.
- Work: validate the manifest before any weight is read, then read each tensor into an f32 buffer, one tensor at a time. BF16, F16 and F32 are accepted, and BF16 is widened with `packages/bf16`. Memory mapping is not needed in M0, which accepts checkpoints of up to 1.3 billion parameters (D6).
- Done when: a fixture whose manifest check fails is rejected before any tensor is read, and a tiny model written by the helper loads with every tensor equal, bit for bit, to the helper's values.

**B9. Gate tests.** Status: done.
- Depends on: B0, B8.
- Work: one tiny random model per case, compared with the helper's logits within the D53 bound: Llama with default RoPE; Llama with `llama3` RoPE; Qwen2 with QKV bias; Qwen3 with QK-norm, explicit `head_dim` and a tied head; Mistral with a sliding window and an untied head.
- Done when: every case passes. The largest difference is recorded in PROJECT.md after the run, and the bound does not change.

## Phase 3: M0c real checkpoints

Gate: logits match transformers v5.20.2 on the same token ids, within the D53 bound. The tests read checkpoints from the directory named by the environment variable `NCMORPH_CHECKPOINT_DIR` (D12), and they skip when it is unset. Checkpoints are never committed. If a real checkpoint exceeds the bound, record the measured difference and stop. Do not change the bound.

**C1. Token-id path (T0).** Status: done.
- Depends on: phase 2.
- Work: the helper runs a checkpoint on a fixed id list and writes the reference logits to a file that is not committed. The Zig test reads the ids and the logits and compares. The test never tokenizes text. Tokenization is phase 7.
- Done when: the helper and the test run on SmolLM2-135M, and the reference output is not committed.

**C2. SmolLM2-135M.** Status: done. Depends on: C1. Llama architecture, single file, tied head, RoPE base 100000. After matching the pinned oneDNN K-block reduction, the eager/BF16 gate passed with max logit difference `0.0` and maximum row tolerance `0.002725` (section 30 of `research/checks.md`). Done when: logits match within the bound, and the largest difference is recorded.

**C3. Qwen2.5-0.5B.** Status: done. Depends on: C1. QKV bias, tied head. After applying the source-verified pinned CPU RMSNorm sum cascade, the regenerated eager/BF16 gate passed with maximum logit difference `0.0` and maximum row tolerance `0.0016375` (section 31 of `research/checks.md`). The earlier pre-fix failures and stale-reference result remain documented. Done when: as C2.

**C4. Qwen3-0.6B.** Status: done. Depends on: C1. QK-norm, explicit `head_dim` 128, and `lm_head.weight` stored next to a tied `embed_tokens`. The helper's Qwen3 context diagnostic now includes QK-norm; the eager/BF16 gate passed with maximum logit difference `0.0` and maximum row tolerance `0.001575` (section 32 of `research/checks.md`). The load checks `lm_head.weight` equals `model.embed_tokens.weight` after widening. The byte identity is recorded in [research/checks.md](research/checks.md), section 3. Done when: as C2.

**C5. Llama 3.2-1B.** Status: done. Depends on: C1. `llama3` RoPE on real weights. The copied, hash-verified checkpoint passed the eager/BF16 gate with maximum logit difference `0.0` and maximum row tolerance `0.00185` (section 33 of `research/checks.md`). The model has about 1.24 billion parameters, so it is the largest checkpoint M0 accepts (D6). Done when: as C2.

## Phase 4: Z1 Zig upgrade

**Z1. Move to the current stable Zig.** Status: done. Depends on: C5.
- Work: decision D28 moves the project to the then-current stable Zig release. On 2026-10-26 that is 0.17.0. Read its release notes first. Set the new version in `build.zig.zon` (`minimum_zig_version`) and in the toolchain section of AGENTS.md. Run `zig build test` and `zig fmt --check .` on the new version. Grep every standard-library name listed in D29 in the new standard library (`zig env` prints its `std_dir`), and fix any name that was renamed or removed.
- Done when: all tests pass and the format check is clean on the new version, every name in D29 exists in its standard library, and the change is recorded in PROJECT.md's decision log. Completed with D69; `Build.args` and array multiplication syntax were migrated per the official release notes (evidence in `research/checks.md`, section 35).

## Phase 5: M1 NCM dialect

Gate: in source mode, the NCM body reproduces the M0b hidden states on tiny models of each family, within the D53 bound. In NCM mode, masks and per-level positions equal the pinned reference's on random tiny NCM models.

Facts this phase relies on, all from the pinned reference (commit `c70274273f489abc73832b2b98584a8483b6b203`), with their full citations in [research/ncm_facts.md](research/ncm_facts.md):
- `models/common.py` (`SelfAttention`, `MLP`, `Block`): the fused `c_attn` is split into q, k and v, then `c_proj`, `q_norm` and `k_norm`, `w_gate`, `w_up` and `c_proj` for the gated MLP.
- `models/next_concept.py` (`NextConceptTransformer`): `transformer.wte` is a Linear from the VQ width to `n_embd` (lines 131 to 133), then the blocks, then `ln_f`. The codebook-tied heads are `_TiedCodebookHead` (lines 9 to 34).

**N1. Confirm the NCM field table.** Status: done. Depends on: Z1.
- Work: every row of the "NCM generator (ncmorph's dialect)" table in PROJECT.md names a decision (D8, D9, D13, D15, D18, D20, D24, D30 to D33) and the pinned code path it follows. Each row was checked against the pinned file it names. D70 corrects the row-count description for `wle` by distinguishing this dialect's `query_level0: true` from the pinned reference's conditional allocation. D71 records that source-mode parity intentionally adapts the reference's LayerNorm baseline to source-epsilon RMSNorm.
- Done: each row has its decision and pinned citation, and each cited path was read at commit `c70274273f489abc73832b2b98584a8483b6b203`.

**N2. RMSNorm option.** Status: done. Depends on: N1. Added `src/ncm_norm.zig`, which applies the B2 RMSNorm kernel row-wise with the caller-supplied source `rms_norm_eps`; epsilon remains runtime-configurable. A seeded test compares the NCM adapter exactly with B2 on random inputs at both fixture epsilon values, 1e-5 and 1e-6. Registered the module in the build and test graph.

**N3. Separate bias flags.** Status: done. Depends on: N1. Decision D72 defines `qkv_bias`, `o_bias`, `mlp_bias` and `norm_bias` separately from the reference's shared `bias` flag (`config/eval/owt.yaml` line 9). `src/ncm_manifest.zig` builds the optional bias manifest tensors, and its synthetic two-layer test enables each flag independently and checks names, shapes and counts.

**N4. Explicit `head_dim` and `intermediate_size`.** Status: done. Depends on: N1. `src/ncm_manifest.zig` calculates q width as `n_heads · head_dim`, maps the attention output projection from that width to `n_embd`, and uses source `intermediate_size` for the MLP tensors. Synthetic shape tests use verified Qwen3-0.6B and Qwen3-4B fixture dimensions: q widths 2048 and 4096, respectively, with MLP widths 3072 and 9728.

**N5. `llama3` RoPE in NCM positions.** Status: done. Depends on: B4. `src/ncm_position.zig` builds Llama 3 inverse frequencies with B4 and gathers cosine/sine rows at explicit position ids, allowing per-level restarts. The test checks exact equality against B4-generated tables for positions `[0, 1, 2, 0, 1, 0]`.

**N6. Source mode.** Status: done. Depends on: N2 to N5. Added `src/ncm_body.zig`, a causal single-sequence source-mode body using the shared kernels. `tools/golden_helper.py` now writes every Transformers hidden state for the five tiny family cases, and the loader gate compares embedding, intermediate, and final-normalized states against those independently generated references. The overall maximum absolute difference was `4.768371582e-7`, within D53.

**N7. NCM mode.** Status: done. Depends on: N6. `src/ncm_model.zig` executes the continuous NCM body: prefix ids pass through the supplied VQ encoder table and `transformer.wte`; `l0_queries` replace the level-0 input; `wle` is added per target level; optional post-upsample Conv1d residuals follow D73; the mask, padding rules and restarted RoPE positions drive attention; and RMSNorm, fused QKV/GQA attention and SwiGLU blocks run to final hidden states. Its synthetic execution test verifies prefix lookup, level embedding, query replacement, and finite outputs with a fully padded prefix across two transformer layers. `src/ncm_mask.zig` and `src/ncm_position.zig` are compared against test-time references from the pinned `NextConceptTransformer` on two tiny model configurations. Base/padded masks, flattened/per-level positions and the `rope_cos`, `rope_sin`, `rope_cos_gen_k` and `rope_sin_gen_k` tables match the pinned model. A pinned post-upsample convolution output is also compared numerically. The authorized reference run is implemented by `tools/ncm_golden_helper.py`, which checks the reference commit before importing the pinned class.

**N8. Gate test.** Status: done. Depends on: N6, N7. In addition to N6's independent Transformers hidden-state goldens, the N8 gate maps each family’s split decoder Q/K/V tensors into the actual fused `ncm_model` executor, runs it with identity WTE, source-causal mask and source sliding-window/RoPE settings, and compares embedding, every intermediate block state and final-normalized states against `decoder.forward_with_trace`. All states matched exactly across five tiny families; maximum absolute difference was `0.0`.

## Phase 6: M2 transplant

Gate: every transplanted tensor is bit-identical to its source after widening. The report lists every NCM tensor as exact, initialized or new, with a checksum. It also lists every source tensor that has no counterpart as not used.

**M2.1. Mapping table.** Status: not started. Depends on: N8. This table is the decision record for the transplant. Record it in PROJECT.md before the transplant code is written. It is the table below, after its entries are confirmed against the pinned names.

| Source tensor | NCM tensor | Status |
|---|---|---|
| `model.layers.i.input_layernorm.weight` | `transformer.blocks.i.ln_1.weight` | exact |
| `model.layers.i.self_attn.{q,k,v}_proj.weight` | rows of `transformer.blocks.i.attn.c_attn.weight`, in q, k, v order | exact |
| `model.layers.i.self_attn.{q,k,v}_proj.bias` (when present) | rows of `transformer.blocks.i.attn.c_attn.bias` | exact |
| `model.layers.i.self_attn.o_proj.weight` | `transformer.blocks.i.attn.c_proj.weight` | exact |
| `model.layers.i.self_attn.o_proj.bias` (when present) | `transformer.blocks.i.attn.c_proj.bias` | exact |
| `model.layers.i.self_attn.{q,k}_norm.weight` (Qwen3) | `transformer.blocks.i.attn.{q,k}_norm.weight` | exact |
| `model.layers.i.post_attention_layernorm.weight` | `transformer.blocks.i.ln_2.weight` | exact |
| `model.layers.i.mlp.gate_proj.weight` | `transformer.blocks.i.mlp.w_gate.weight` | exact |
| `model.layers.i.mlp.up_proj.weight` | `transformer.blocks.i.mlp.w_up.weight` | exact |
| `model.layers.i.mlp.down_proj.weight` | `transformer.blocks.i.mlp.c_proj.weight` | exact |
| `model.norm.weight` | `transformer.ln_f.weight` | exact |
| `model.embed_tokens.weight` | no counterpart. Its width is the source hidden size, and the VQ width is 384 (D3). The NCM prefix table is the VQ encoder's (D58) | not used |
| `lm_head.weight` | no counterpart. The NCM heads are codebook-tied (D18) | not used |
| `model.layers.i.self_attn.rotary_emb.inv_freq` | computed from `rope_theta` | not used |

**M2.2. Transplant engine.** Status: not started. Depends on: M2.1. Copy every exact tensor after widening. Initialize every other NCM tensor with the policy of D30, seeded and recorded in PROJECT.md. Done when: exact tensors are bit-identical to their source after widening, and rerunning with the same seed gives the same initialized tensors.

**M2.3. Transplant report.** Status: not started. Depends on: M2.2. One row per NCM tensor: NCM name, status, source name, shape, dtype, and the SHA-256 of the widened f32 bytes. The hash uses `std.crypto.sha2.Sha256` (D29). Source tensors without a counterpart appear with status "not used". Done when: each NCM tensor and each source tensor appears exactly once, and recomputing the checksums from the loaded tensors matches the report.

**M2.4. Gate test.** Status: not started. Depends on: M2.3. Exact tensors are compared byte for byte. The number of exact tensors equals the number of source tensors that the mapping table maps to an NCM tensor.

## Phase 7: Tokenizers T0, T1 and T2

Goal: text in, token ids out, in Zig, for every family in PROJECT.md's tokenizer table. A Zig tokenizer is accepted only when its ids equal the T0 reference on the fixed corpus.

**K0. Unicode 16.0.0 general categories.** Status: not started. Depends on: phase 1.
- Why: the regex pre-tokenizers and Digits must use the Unicode 16.0.0 tables that the reference's regex engine and Rust's `char::is_numeric` use (D57).
- Work: generate a Unicode 16.0.0 category table with `tools/gen_unicode_categories.py` from the 16.0.0 `UnicodeData.txt` (SHA-256 `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`, 2,175,362 bytes, read 2026-10-26). The Unicode 18.0.0 table stays for the M0a API.
- Done when: the 16.0.0 table gives the same general category as the 16.0.0 file for every code point (a test checks the file's ranges and a sample), and the file's hash is recorded in PROJECT.md (D57).

**K1. T0 reference helper.** Status: not started. Depends on: phase 2.
- Work: the Python helper runs `tokenizers==0.23.3` for each family and writes the ids of a fixed corpus, at test time, not committed. The corpus is a hand-written text file under `tools/corpus/`, committed. It holds ASCII text, non-ASCII letters, digits, emoji, combining marks and several whitespace variants.
- Done when: the helper reproduces the ids of each family for the corpus, and two runs give the same output.

**K2. Pre-tokenizers** (`packages/pretokenizers`). Status: not started. Depends on: K0.
- Work: hand-written matchers for the Qwen Split (`Isolated`), the Llama 3 Split (`Isolated`), the GPT-2 ByteLevel regex, and Digits with `individual_digits` true and false (D54). Their semantics are: the leftmost alternative that matches at the current position; `\s` as the Oniguruma definition (U+0009 to U+000D, U+0085, and the categories Zl, Zp and Zs; `research/software_sources.md`, section B); and `\p{L}` and `\p{N}` from K0.
- Done when: the pieces equal the T0 pieces on the corpus, including non-ASCII text.

**K3. NFC normalizer tables.** Status: not started. Depends on: phase 2.
- Work: generate the NFC tables from the Unicode 9.0.0 data: `UnicodeData.txt` (SHA-256 `68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f`, 1,686,443 bytes) and `CompositionExclusions.txt` (SHA-256 `5623df16856ad4007c60bdfff6f054e087521becd24cb4006be69c3a1d851aee`, 8,355 bytes), both read 2026-10-26. The reference uses `unicode-normalization-alignments` 0.1.12, which is Unicode 9.0.0 (D56).
- Done when: the normalizer equals T0's NFC on the corpus, including composed and decomposed Latin letters, Hangul and combining sequences.

**K4. Byte-level BPE** (`packages/bpe`). Status: not started. Depends on: K2, K3.
- Work: merges from `tokenizer.json` (as pairs or as strings), the byte-to-unicode table (`byte_level.rs` lines 15 to 39), `ignore_merges` (Llama 3), and the ByteLevel, Fuse and Strip decoders.
- Done when: ids and decoded text equal T0's for Qwen2.5-0.5B, Qwen3-0.6B (both with NFC), Llama 3.0 8B and Llama 3.2-1B (with `ignore_merges`), and SmolLM2-135M (Digits, then ByteLevel).

**K5. Gate T1.** Status: not started. Depends on: K4. Ids match T0 on the corpus for all five byte-level checkpoints, and decoding returns the text T0's decoder returns.

**K6. SentencePiece-style tokenizers** (`packages/bpe`). Status: not started. Depends on: K1.
- Work: the Llama 2 normalizer (Prepend `▁`, then Replace `" "` with `▁`); the Mistral Metaspace pre-tokenizer (replacement `▁`, `prepend_scheme` `first`, `split` false); BPE with `byte_fallback` and `fuse_unk`; the ByteFallback, Fuse and Strip decoders; and the TemplateProcessing post-processor, which adds `<s>`.
- Done when: ids and decoded text equal T0's for Llama 2, Mistral v0.1 and Mistral v0.3 on the corpus, including bytes outside the vocabulary.

**K7. Gate T2.** Status: not started. Depends on: K6. Ids and decoded text match T0 for the three checkpoints.

## Phase 8: Data, writer and VQ tokenizers

Goal: a VQ tokenizer for each source (D10), and the released WMT14 tokenizer for the Qwen3 route (D14). The tables and rules are in PROJECT.md.

**A1. Data pipeline** (`tools/data_packing.py`, Python). Status: not started. Depends on: K1 (for the tokenizer), phase 1.
- Work: read the owner's local OpenWebText parquet shards with `pyarrow==26.0.0`. Apply D22: validation documents are those whose index is divisible by 1,000, and training documents are the rest. Tokenize each document with its family's tokenizer and post-processor. Join documents with the family's EOS id. Cut the stream into windows of `L` tokens, drop the last partial window of each epoch, and shuffle the windows with seed 42 at each epoch. Write a text index of window hashes for the tests. The data itself is never written into the repository.
- Done when: two runs with seed 42 give identical window hashes, and the validation and training document counts are recorded in PROJECT.md.

**W1. Safetensors writer** (`packages/safetensors`). Status: not started. Depends on: phase 1.
- Work: write files that follow the writer rules of PROJECT.md ("Writer rules", D39) and `safetensors/src/tensor.rs` at v0.8.0. Use checked arithmetic for every size and offset.
- Done when: a round trip through the Zig reader and the Python `safetensors==0.8.0` reader gives identical tensors, and the writer rejects duplicate names and unsupported dtypes.

**V1. Released WMT14 tokenizer conversion** (`tools/convert_wmt14_tokenizer.py`). Status: not started. Depends on: W1. Needs the owner's approval, as set out above.
- Work: follow decision D41 exactly. Run in a throwaway environment with no network and no credentials. Load `tokenizer.pt` with `torch.load(..., weights_only=False, map_location='cpu')`, as the reference does (`tokenization/vqvae_tokenizer.py`, line 18), and take its `'model'` mapping. Check that the mapping holds exactly the 96 keys of [research/converter/wmt14_tokenizer_keys.md](research/converter/wmt14_tokenizer_keys.md), with no prefix, and that `strict=True` loading into a module built from the derived structure succeeds. Keep each tensor's stored dtype (D59). Write `tokenizer.safetensors` with W1's writer. Record the file's size and SHA-256 in `ncm.json`. Any mismatch fails the converter. A name is never renamed to fit.
- Done when: the converter run completes with exactly 96 tensors, every tensor reads back from the output with the same bytes as the loaded tensor, and the file's size and SHA-256 are recorded in PROJECT.md's released-checkpoint section.

**V2. Trained VQ tokenizer** (`tools/train_vq.py`, Python). Status: not started. Depends on: A1, W1.
- Work: the architecture of PROJECT.md ("Trained VQ tokenizer", the OWT column of Table 5 for sizes, `owt.yaml` for the rest), the recipe of D21, the loss of D52, and the EMA of the same table. Train on the OWT windows of A1. Write `tokenizer.safetensors` with the reference attribute names, in the stored dtype of training (F32), and `ncm.json`.
- Done when: a short run writes the export, the tensor names and shapes match the list the config implies (checked in Python), and the full run is scheduled on the owner's hardware (D47).

**V3. Zig VQ forward pass and loader** (`packages/vq`). Status: not started. Depends on: V1 (for the released tokenizer), V2 (for trained tokenizers).
- Work: the encoder (token table, learned positional embedding, pre-quant GroupNorm), the quantizer (per-level down and up sampling, `quant_resi`, nearest code, residual), the decoder (one bidirectional block with QK-norm and GELU, then the output head), and a loader that checks every name and shape. Inference only. The EMA and the losses stay in Python.
- Done when: on fixed ids, the Zig encoder, quantizer and decoder outputs equal the Python reference's within the D53 bound, for the released WMT14 tokenizer and for one tiny trained VQ. The comparison of the released tokenizer runs the pinned reference code, and needs the owner's approval.

**V4. Reuse gate for the Qwen3 route.** Status: not started. Depends on: V1, V3.
- Work: decision D44. Measure the released WMT14 tokenizer on OpenWebText windows of 64 tokens, with the Qwen3 ids. Measure token-exact reconstruction with the full hierarchy, and codebook utilization (D37). Record both in [research/checks.md](research/checks.md).
- Done when: if the measurement passes the V5 thresholds, the route stays (D14). If it does not, a Qwen3 tokenizer is trained with V2 at `L = 64`. The route is confirmed in PROJECT.md either way.

**V5. Trained VQ gate.** Status: not started. Depends on: V2 at its full budget, V3.
- Work: for each trained tokenizer, measure reconstruction and utilization on the validation split (D37).
- Done when: each trained tokenizer passes at least 99.9 percent token-exact reconstruction and 80 percent utilization, and the numbers are recorded in PROJECT.md.

## Phase 9: M3 generator training

Gate: as set out in PROJECT.md ("Gates", M3).

**A2. Generator trainer** (`tools/train_generator.py`, Python). Status: not started. Depends on: A1, V5, M2.2.
- Work: the recipe of D20, the corruption of D17, the loss of D16 (Eq. 10 with `α = 0`), the initialization of D30, BF16 mixed precision with f32 master weights, AdamW with weight decay on matrices only, cosine decay after linear warmup, and seed 42. The frozen tokenizer comes from V5.
- Done when: on a fixed small batch the loss falls over a fixed number of steps, recorded in the test, and two runs with the same seed give the same losses on CPU for the tiny model.

**A3. M3 gate.** Status: not started. Depends on: A2.
- Work: the full recipe runs on the owner's hardware (D47). Gate: the run completes its update count with no non-finite loss; the validation loss at the last update is lower than at update 1,000; and the evaluation protocol of D35 completes for all five seeds. Record the mean and the standard deviation, and the run's configuration, in PROJECT.md.
- Done when: the gate passes, and the results are recorded.

## Phase 10: M4 export and reload

Gate: reloading an exported directory gives logits within the D53 bound of the in-memory model, on the same token ids.

**E1. Export.** Status: not started. Depends on: W1, A3 (for a real model) or a tiny model from the helper (for tests).
- Work: write the four files of PROJECT.md ("Export format", D26): `ncm.safetensors`, `tokenizer.safetensors`, `ncm.json` and `NOTICE` (D45). The `NOTICE` file holds the licence lines of PROJECT.md ("Licences and notices"), including the Llama name rule and the source licence where the source is Llama.
- Done when: the export of a tiny model has the four files, each safetensors header passes the reader's validation, and the fields of `ncm.json` match the generator table of PROJECT.md.

**E2. Reload.** Status: not started. Depends on: E1.
- Work: load the export, run the same ids, and compare the logits with the in-memory model. For models of at most 1.3 billion parameters the check runs in Zig. For larger models the check runs in Python on the exported files (D6 and D50).
- Done when: the logits are within the D53 bound for each model checked.

**E3. Gate M4.** Status: not started. Depends on: E2. Done when: the reload check passes on a tiny model from the helper, and on each trained model that the owner produces.

## Phase 11: Package split

Trigger: M0b and M0c have passed, Z1 is done, and no `packages/` API has changed across the last two gates. The owner approves the trigger.

Then move `bf16`, `unicode_categories`, `safetensors`, `hf_config`, `kernels`, `pretokenizers`, `bpe` and `vq` to their own repository, `ncmorph-packages`, licensed Apache-2.0 with the same header, at version `0.1.0` (D60). ncmorph depends on that repository, with a version pin recorded in PROJECT.md when the split is done.

## Rules for every step

- Run the verification list in AGENTS.md, and quote the results in the report.
- Commit only what the owner asks for. Source, configs, text fixtures and hand-written expectations are the committed kinds. No `.safetensors`, `.pt` or other weights, no datasets, no logs, and no reference output that a helper generates.
- Record every decision in PROJECT.md's decision list when it is made, and before the code that depends on it.
- Every statement about NCM behaviour cites the pinned file and the symbol or line.
- A fact that is not verified is researched and recorded in [research/checks.md](research/checks.md) before a step relies on it. No `UNVERIFIED:` marker remains in PROJECT.md or in this file.

## Open items

None. Every question a step needs is answered in PROJECT.md. A step that finds a gap records a decision entry, with its basis, before it continues.
