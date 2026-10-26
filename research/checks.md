# Research checks

This file is the log of the checks behind [PROJECT.md](../PROJECT.md). Each entry says how the fact was checked and what the result was. An entry marked **carried** was checked in an earlier pass; its evidence is in the file named in the entry, and it was not re-run in the pass that wrote this file (2026-10-26). Nothing here is an open question. Where a check could not be run, the entry says so, and the spec states what the implementation does about it.

Scratch paths below are under the session scratchpad, `/private/tmp/claude-501/-Users-jessicabruce--claude-spares-spare-e7phlw/19e02558-926e-417a-8cc2-e33c5915494f/scratchpad/`. They are not in the repository. The pinned NCM copy is `ncm_license_check/src/ncm-c70274273f489abc73832b2b98584a8483b6b203/` in that folder.

## 1. Revisions of the source checkpoints

Read from `packages/hf_config/testdata/<name>/provenance.json` (re-read this pass). The gated Llama originals are represented by the mirrors below; the spec uses the originals for real conversions.

| Fixture | Repository | Revision |
|---|---|---|
| llama-2-7b | `NousResearch/Llama-2-7b-hf` | `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` |
| llama-3.0-8b | `NousResearch/Meta-Llama-3-8B` | `315b20096dc791d381d514deb5f8bd9c8d6d3061` |
| llama-3.2-1b | `unsloth/Llama-3.2-1B` | `9535bd9b1d1dea6acafbdc4813b728796aeb28da` |
| mistral-7b-v0.1 | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| mistral-7b-v0.3 | `mistralai/Mistral-7B-v0.3` | `caa1feb0e54d415e2df31207e5f4e273e33509b1` |
| qwen2.5-0.5b | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| qwen3-0.6b | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| qwen3-1.7b | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| qwen3-4b | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| qwen3-8b | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| qwen3-14b | `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| qwen3-32b | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` |
| smollm2-135m | `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |

## 2. Flags from each `config.json`

Read with a Python script over the 13 fixture configs (re-run this pass). `—` means the key is absent.

| Fixture | model_type | hidden | inter | layers | heads / kv | head_dim | vocab | rms_norm_eps | rope_theta | rope_scaling | tie | attention_bias | mlp_bias | sliding_window | use_sliding_window | max_pos | dtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama-2-7b | llama | 4096 | 11008 | 32 | 32 / 32 | — | 32000 | 1e-5 | — | null | false | — | — | — | — | 4096 | float16 |
| llama-3.0-8b | llama | 4096 | 14336 | 32 | 32 / 8 | — | 128256 | 1e-5 | 500000 | null | false | false | — | — | — | 8192 | bfloat16 |
| llama-3.2-1b | llama | 2048 | 8192 | 16 | 32 / 8 | 64 | 128256 | 1e-5 | 500000 | llama3 (factor 32, high 4, low 1, original 8192) | true | false | false | — | — | 131072 | bfloat16 |
| mistral-7b-v0.1 | mistral | 4096 | 14336 | 32 | 32 / 8 | — | 32000 | 1e-5 | 10000 | — | false | — | — | 4096 | — | 32768 | bfloat16 |
| mistral-7b-v0.3 | mistral | 4096 | 14336 | 32 | 32 / 8 | — | 32768 | 1e-5 | 1000000 | — | false | — | — | null | — | 32768 | bfloat16 |
| qwen2.5-0.5b | qwen2 | 896 | 4864 | 24 | 14 / 2 | — | 151936 | 1e-6 | 1000000 | — | true | — | — | 32768 | false | 32768 | bfloat16 |
| qwen3-0.6b | qwen3 | 1024 | 3072 | 28 | 16 / 8 | 128 | 151936 | 1e-6 | 1000000 | null | true | false | — | null | false | 40960 | bfloat16 |
| qwen3-1.7b | qwen3 | 2048 | 6144 | 28 | 16 / 8 | 128 | 151936 | 1e-6 | 1000000 | null | true | false | — | null | false | 40960 | bfloat16 |
| qwen3-4b | qwen3 | 2560 | 9728 | 36 | 32 / 8 | 128 | 151936 | 1e-6 | 1000000 | null | true | false | — | null | false | 40960 | bfloat16 |
| qwen3-8b | qwen3 | 4096 | 12288 | 36 | 32 / 8 | 128 | 151936 | 1e-6 | 1000000 | null | false | false | — | null | false | 40960 | bfloat16 |
| qwen3-14b | qwen3 | 5120 | 17408 | 40 | 40 / 8 | 128 | 151936 | 1e-6 | 1000000 | null | false | false | — | null | false | 40960 | bfloat16 |
| qwen3-32b | qwen3 | 5120 | 25600 | 64 | 64 / 8 | 128 | 151936 | 1e-6 | 1000000 | null | false | false | — | null | false | 40960 | bfloat16 |
| smollm2-135m | llama | 576 | 1536 | 30 | 9 / 3 | — | 49152 | 1e-5 | 100000 | null | true | false | — | — | — | 8192 | bfloat16 |

Notes from the same read:

- `head_dim` is absent in every Llama, Mistral and Qwen2.5 fixture. The head dimension is then `hidden_size / num_attention_heads`: 128 for Llama 2 7B, Llama 3.0 8B, Mistral v0.1 and v0.3 (4096 / 32), 64 for Qwen2.5-0.5B (896 / 14) and SmolLM2-135M (576 / 9). Llama 3.2-1B sets `head_dim` 64 explicitly.
- `rope_theta` is absent in `llama-2-7b`. The spec default of 10000 is transformers' `RotaryEmbeddingConfigMixin.default_theta` (see section 9).
- `rope_parameters` (the transformers v5 key) is absent from all 13 fixtures. The loader rejects it until step B0a.
- The Qwen3 q width is `num_attention_heads × head_dim`: 0.6B 16 × 128 = 2048 against hidden 1024; 4B 32 × 128 = 4096 against hidden 2560; 32B 64 × 128 = 8192 against hidden 5120.

## 3. Where `lm_head.weight` is stored

Parsed from each fixture's header bytes (`.header` files, 8-byte length prefix plus JSON) or from its index file (re-run this pass).

| Fixture | `lm_head.weight` stored | `model.embed_tokens.weight` stored |
|---|---|---|
| llama-2-7b | yes | yes |
| llama-3.0-8b | yes | yes |
| llama-3.2-1b | no | yes |
| mistral-7b-v0.1 | yes | yes |
| mistral-7b-v0.3 | yes | yes |
| qwen2.5-0.5b | no | yes |
| qwen3-0.6b | yes | yes |
| qwen3-1.7b | yes | yes |
| qwen3-4b | no | yes |
| qwen3-8b | yes | yes |
| qwen3-14b | yes | yes |
| qwen3-32b | yes | yes |
| smollm2-135m | no | yes |

For the tied models whose `lm_head.weight` is stored (Qwen3-0.6B and Qwen3-1.7B), the manifest check compares it with `embed_tokens` after widening (PROJECT.md, "Manifest validation").

**Qwen3-0.6B tied head (byte identity), re-run this pass.** The scratch copies `qwen3-0.6b-tensors/lm_head.bin` and `embed.bin` each hold 311,164,928 bytes. `shasum -a 256` gives `8f29acf519434862d95613b2b4f6b9d14933a5e4d16baebf8ac0b33b410acfb6` for both, and `cmp` reports them identical. The tensors were fetched at revision `c1899de289a04d12100db370d81485cdf75e47ca` in an earlier pass (carried).

## 4. Qwen2.5 against Qwen3 tokenizers

Re-run this pass with `scratchpad/scripts/compare_qwen_tokenizers_v2.py` and `scratchpad/scripts/merges_check_v3.py`.

- Qwen2.5 `tokenizer.json`: `model.vocab` has 151,643 entries (largest id 151,642). It has 22 added tokens (largest id 151,664). Total ids: 151,665.
- Qwen3 (`tokenizer_config.json`, `added_tokens_decoder`): 26 added tokens, ids 151,643 to 151,668. Total ids: 151,669.
- Every Qwen2.5 added token appears in Qwen3 with the same id and content. The four Qwen3 tokens with no Qwen2.5 counterpart are 151,665 `<tool_response>`, 151,666 `</tool_response>`, 151,667 `<think>` and 151,668 `</think>`.
- Merges: Qwen2.5 `model.merges` has 151,387 entries. Qwen3 `merges.txt` has 151,387 lines after its `#version: 0.2` header. They are identical in content and order.
- The normalizer, pre-tokenizer, post-processor and decoder sections are identical in the two files.
- Correction: the first run of the v2 comparison reported 96 merges missing from Qwen3. That result is wrong. The script dropped every line starting with `#`, which removes real merges such as `# include`. Only the header line should be dropped, and `merges_check_v3.py` does that. The v2 output must not be quoted.
- Verified this pass (Hugging Face tree API, git blob ids at the pinned revisions): `tokenizer.json` `cd71f61a15a5`, `tokenizer_config.json` `417d038a63fa`, `vocab.json` `4783fe10ac3a` and `merges.txt` `31349551d90c` are identical in all six dense Qwen3 revisions: 0.6B, 1.7B, 4B, 8B, 14B and 32B.
- Carried: the WMT14 tokenizer is built on `Qwen/Qwen3-0.6B` (`config/eval/wmt14.yaml` line 46), so its token set is Qwen3's token set. Its released file is `hazyresearch/ncm-checkpoints/wmt14/tokenizer.pt`, 645,790,351 bytes, tagged MIT. These two facts come from the earlier pass.

## 5. Paper values (arXiv:2610.14162v1)

Read from the local text `ncm_2610_14162.txt` (re-read this pass; line numbers are of that file).

- Table 5 (tokenizers), lines 1268 to 1422: text vocabulary 50,304 / 30,528 / 8,192 / 50,304 / 151,680 / 12; L 1,024 / 128 / 128 / 48 / 64 / 89; K 32 / 12 / 12 / 8 / 11 / 12; V_k 4,096 / 8,192 / 4,096 / 4,096 / 4,096 / 16; d 384 / 384 / 384 / 384 / 384 / 256; encoder blocks 0 in every column; decoder blocks 1 / 4 / 1 / 1 / 1 / 1; decoder heads 6 / 6 / 6 / 6 / 6 / 4; dropout 0.3 / 0.1 / 0.3 / 0.1 / 0.1 / 0.3; updates 78,432 / 25,957 / 30,000 / 120,001 / 60,000 / 10,000; global batch 768 / 2,048 / 4,096 / 4,096 / 4,096 / 1,024; peak LR 3e-4 / 3e-4 / 5e-4 / 5e-4 / 5e-4 / 5e-4; floor 3e-5 / 3e-5 / 5e-5 / 5e-5 / 5e-5 / 5e-5; warmup 1,000 / 1,000 / 1,000 / 1,000 / 1,000 / 500; β2 0.95 / 0.99 / 0.95 / 0.95 / 0.95 / 0.99; weight decay 0.01 / 0.05 / 0.01 / 0.01 / 0.01 / 0.01; clip 1; codebook EMA decay 0.99 / 0.99 / 0.95 / 0.95 / 0.95 / 0.99; nested dropout 0.5 / 0.7 and 0.3 / 0.5 / 0.5 / 0.5 / 0.5; parameters 111.2M / 111.40M / 36.43M / 60.63M / 144.00M / 14.90M.
- Table 6 (generator), lines 1449 to 1548: hidden width 768 / 768 / 768 / 768 / 768 / 512; heads 12 / 12 / 12 / 12 / 12 / 8; blocks 12 / 12 / 12 / 12 / 12 / 8; prefix P 1 / 1 / 1 / 512 / 96 / 91; RoPE base 1e5 / 1e4 / 1e5 / 1e5 / 1e5 / 1e4; dropout 0.1 / 0.1 / 0.1 / 0.3 / 0.1 / 0.1; updates 250,000 / 100,000 / 50,000 / 350,000 / 150,000 / 50,000; global batch 256 / 512 / 2,048 / 512 / 2,048 / 1,024; peak LR 3e-4 in every column; floor 3e-5 in every column; warmup 500 / 1,000 / 500 / 500 / 500 / 1,000; β2 0.95 / 0.95 / 0.95 / 0.95 / 0.95 / 0.99; weight decay 0.1 / 0.1 / 0.1 / 0.1 / 0.01 / 0.01; clip 1; parameters 94.65M / 97.07M / 88.61M / 88.02M / 119.96M / 25.46M.
- Table 1 (lines 480 to 489): OWT reconstruction 99.99%, utilization 83.10%.
- Budget accounting (lines 1554 to 1558): OWT tokenizer and generator together are 78,432 × 768 × 1,024 + 250,000 × 256 × 1,024 ≈ 127.22 B positions (61.68 B + 65.54 B).
- Algorithm 1 (lines 407 to 428) and Algorithm 2 (lines 430 to 453), with Algorithm 2 line 11 "Select output tokens by argmax". Eq. 10 is in the paragraph before Algorithm 1, and its weights are `w_k = ℓ_k^(−α) / Σ_j ℓ_j^(1−α)`.
- Conditioning corruption: codes are replaced by other codes of the same level's codebook, and the targets stay clean ("Conditioning Corruption", Section 3.4). Appendix F.3 holds the per-level rates (the fact sheet, `research/ncm_facts.md` C2, gives the schedule as linear across levels).
- Conditional generation (Section 3.4): prompt positions attend to each other, each target block attends to the prompt and itself, and prompt positions cannot attend to target blocks.

## 6. Reference configs and code (pinned commit `c70274273f489abc73832b2b98584a8483b6b203`)

Re-read this pass.

**`config/eval/owt.yaml`** (`owt.yaml` line numbers): `n_layer` 16 (7); `n_head` 16 (8); `n_embd` 1024 (11); `prefix_len` 1 (13); `raw_token_prefix` true (14); `predict_level0` false (48); `rope_base` 100000 (53); `ctx_encoder.enabled` false (56); `query_level0` false (60); `tie_heads_to_codebook` true (64), `tie_heads_learn_scale` true (65), `tie_heads_bias` true (66); `tokenizer.vqvae`: `n_layers` 6 (76), `embed_dim` 384 (77), `in_vocab_size` 50304 (78), `context_length` 1024 (79), `dropout` 0.3 (81), `pre_quant_groupnorm` 8 (82), `use_qk_norm` true (83), `pre_quant_dropout` 0.0 (84), `vqvae_vocab_size` 8192 for 32 levels (85 to 118); `decay` 0.99 (121), `epsilon` 1e-5 (122), `commitment_cost` 0.25 (123), `learned_l1_sampling` true (124), `learned_l1_cascade` true (125), `learned_l1_cascade_base` 4 (126), `learned_all_sampling` false (127), `quant_resi` enabled, ratio 0.5, `share_mode` 0, `learnable_ratio` false (128 to 132), `fine_drop` prob 0.5 and `min_keep` 1 (134 to 136), `manual_levels` 1, 4, 9, …, 1024 (139 to 172); `generation` `temperature` 1.0 (176), `top_k` 0 (177), `top_p` 1.0 (178), `cfg_scale` 0.0 (179), `level0_unigram` (180), `cut_at_eos` false (181); `eval` `n_samples` 256 (183), `seeds` 42, 0, 1, 2, 3 (184 to 189), `batch_size` 8 (190), `n_shards` 2 (191), `metric` genppl (192), `ref_model` gpt2-large (193).

Note that `n_layers: 6` (line 76) is the decoder's head count, not its depth. The paper's Table 5 calls it "decoder attn. heads" (6). The depth is `dec_layers`, which the fact sheet reads as 1 for OWT (`research/ncm_facts.md` C1).

**`config/eval/wmt14.yaml`**: `n_layer` 12 (9), `n_embd` 768 (13), `prefix_len` 96 (15), `predict_level0` true (29), `rope_base` 100000 (34), `pre_tokenizer.model_id` `Qwen/Qwen3-0.6B` (46), `vqvae.n_layers` 6 (51), `embed_dim` 384 (52), `in_vocab_size` 151680 (53), `manual_levels` from line 95.

**Level-0 flags across the released configs** (re-read this pass):

| Config | `query_level0` | `predict_level0` | `tie_heads_to_codebook` |
|---|---|---|---|
| owt | false | false | true |
| tinystories | false | false | true |
| xsum | true | false | true |
| lm1b-small, lm1b-medium, lm1b-large | true | false | not set |
| wmt14 | not set | true | not set |
| sudoku-easy, sudoku-medium, sudoku-hard | true | not set | false |

**`scripts/eval_shard.py`** (re-read this pass): line 34 `temp` defaults to 1.0. Line 35 `top_k` falls back to 1 if the config has no key. Lines 38 to 40 `freq` (frequency penalty) defaults to 0.0 and `adj` comes from the config. Line 52 loads the generator with `torch.load(..., weights_only=False)`. Line 70 `L0_MODE = a.l0 or "unigram"`, so the reference's default level-0 mode is the unigram. Line 83 splits the seeds across shards with `[a.shard::a.nshards]`. Line 84 `n` is `n_samples`. Lines 99 to 101: `pick_tokens` returns `logits.argmax(-1)` when there is no frequency and no adjacency penalty. Line 127 `CUT_AT_EOS` defaults to true. Lines 130 to 134 `dec` cuts at the first EOS when that flag is set, removes EOS and PAD, and calls `decode(skip_special_tokens=True)`. Lines 140 to 146 `run`: `generate`, then `idx_to_zhats`, then the decoder, then `pick_tokens`, then decode. Line 153 calls `torch.manual_seed(seed)` for each seed.

**`scripts/eval_merge.py`** (re-read this pass): the `genppl` path calls `generation_perplexity(..., [""] * len(gen), gen, batch_size=32)` at line 99. The judge batch size is therefore 32, and the sampling batch size is `CFG.eval.batch_size` (8, `eval_shard.py` line 43).

**`evaluation/gen_ppl.py`** (re-read this pass): line 56 the default truncation length is 1024. Lines 80 to 105 build the scored-token mask. With an empty prompt, the first token has no prediction, so every other token is scored. Lines 121 to 124 compute the NLL. Lines 134 to 140 pool the NLL: `mean_ppl = exp(Σ nll / Σ tokens)`.

**`models/next_concept.py`** (re-read this pass): lines 9 to 34 `_TiedCodebookHead`: `z = Linear(n_embd → d, bias false)(h)`; `E` is the codebook (a frozen buffer unless trainable); `E_sq = Σ E²` per code; `d_scale = mean(E_sq)`; `neg_d2 = (2 z·Eᵀ − E_sq) / d_scale`; `out = neg_d2 · exp(log_inv_T)`, plus an optional per-code bias. Line 86 `init_std = sqrt(1 / n_embd)`. Lines 88 to 91 the `query_level0` flag and the transformer's target length `T`. Line 127 `wpe` is initialized with `init_std`. Lines 239 to 242 `l0_queries` has shape `1 × ℓ₀ × embed_dim`, initialized with `init_std`. Line 521 the `generate` signature. Line 581 the unconditional path (`idx is None`).

**`models/quant.py`** (re-read this pass): line 299 codebook weights are `normal_()` (N(0, 1)); line 300 `requires_grad_(False)`; lines 302 to 303 the EMA buffers `ema_cluster_size_i` and `ema_w_i`. Lines 489 to 494 the EMA update of the cluster sizes: `ema_cs ← decay·ema_cs + (1 − decay)·counts`, with `counts` the sum of the one-hot encodings. Lines 448 to 495 `forward`, which runs the level loop.

**Fact sheet**: `research/ncm_facts.md` was checked against these files in an earlier pass (carried). Its contradictions list is section H, and the spec's decisions cover each item there.

## 7. Zig

- `zig version` prints `0.16.0` (re-run this pass).
- ziglang.org/download, fetched this pass: latest stable is **0.17.0**, dated 2026-10-01. The master branch is `0.18.0-dev.610+720939f8c`, dated 2026-10-25. The project pins 0.16.0 (decision D28 in PROJECT.md).
- `zig ast-check` results (re-run this pass, files in `scratchpad/hdr_check/`):
  - `/* */` block header, then `pub fn main`: exit 1, `error: expected type expression, found '/'` (the message starts at 1:1).
  - `//` header lines, then `//!` doc comment, then code: exit 0.
  - `//` header lines only, then code: exit 0.
  - `//!` doc comment first, then `//` header lines, then code: exit 0.
- `packages/bf16/root.zig` already uses the `//` header followed by `//!` lines, and `build.zig.zon` sets `minimum_zig_version = "0.16.0"`, with no dependencies (both read this pass).

## 8. Licences

Read this pass from the primary pages (fetched through the built-in browser, 2026-10-26):

- **Llama 3.2 Community License**, version release 2024-09-25. Section 1.b.i: a distributed Llama Materials product, or a product that contains them, must provide a copy of the agreement and prominently display "Built with Llama". An AI model that is created, trained, fine-tuned or otherwise improved with the materials, and that is distributed, must include "Llama" at the beginning of its name. Section 1.b.iii: a "Notice" file with "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." Section 1.b.iv: the Acceptable Use Policy applies. Section 2: a licence from Meta is needed above 700 million monthly active users. Section 5.a: the name "Llama" is licensed only as Section 1.b.i requires.
- **Meta Llama 3 Community License**, version release 2024-04-18. The same structure, with "Built with Meta Llama 3", the model name starting with "Llama 3", and the notice "Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." Section 1.b.v: the materials and their outputs may not be used to improve any other large language model, except Meta Llama 3 and its derivatives.
- **Llama 2 Community License**, version release 2023-07-18. Section 1.b.i: a copy of the agreement goes to any third party that receives the materials or a derivative. Section 1.b.iii: the notice "Llama 2 is licensed under the LLAMA 2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." Section 1.b.v: the same no-improvement restriction, excepting Llama 2 and its derivatives. Section 2: the 700 million user limit.
- Licence tags (Hugging Face API responses saved in scratch, read this pass): `license:apache-2.0` for Qwen3-0.6B, Qwen2.5-0.5B, Mistral-7B-v0.1, Mistral-7B-v0.3 and SmolLM2-135M. `license:llama3.2` for `unsloth/Llama-3.2-1B`. `license:llama2` for `meta-llama/Llama-2-7b-hf`. `license:other` for `NousResearch/Meta-Llama-3-8B`. No licence tag for `NousResearch/Llama-2-7b-hf`.
- NCM code: MIT, `Copyright (c) 2025 hermann kumbong` (carried; read by the deep-research pass from `LICENSE` at the pinned commit, `research/deep_research_digest.md`).

## 9. transformers v5.20.2 and the Llama 3 RoPE formula

Carried from the earlier pass. The formulas, with the line numbers of `modeling_rope_utils.py`, `modeling_llama.py` and `masking_utils.py`, are in [transformers_formulas.md](transformers_formulas.md). The pin is `18329f1065382e54db698e242cf98bd3f3e5a766` (carried). PyPI `transformers==5.20.2` was uploaded 2026-10-19 (re-read this pass).

## 10. Tokenizers and safetensors pins

- Re-read this pass (`software_sources.md` in this folder): tokenizers tag **v0.23.3**, commit `301277072263fb3830eb66eec7e47adadf76f1dc`, published 2026-10-09 (carried, and PyPI `tokenizers==0.23.3` first upload 2026-10-09, re-read this pass). The annotated tag object's full hash is not recorded in this folder, so the spec pins the commit.
- Re-read this pass: safetensors tag **v0.8.0**, commit `a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6`, published 2026-06-09 (carried; PyPI `safetensors==0.8.0` first upload 2026-06-09, re-read this pass).
- Transformers' dependency table, `dependency_versions_table.py` lines 58 and 72: `safetensors>=0.8.0` and `tokenizers>=0.23.1,<0.24.0` (carried).
- The `tokenizers` source files cited in `software_sources.md` are identical in v0.23.1, v0.23.2 and v0.23.3 (carried). The workspace `Cargo.toml` still reads 0.23.2 in the v0.23.3 tag, which does not affect the pin.

## 11. Corpus, judge and Python packages

Re-read this pass from the Hugging Face and PyPI responses saved in scratch.

- `Skylion007/openwebtext` at `79d93d786212f7344586290adb811d4ae6a1762c`: licence `cc0-1.0`. Config `plain_text`, split `train`: 8,013,769 examples, 39,769,491,688 bytes of text, 24,193,092,408 bytes to download. Column `text`.
- `openai-community/gpt2-large` at `32b71b12589c2f8d625668d2335a01cac3249519`: licence `mit`.
- PyPI: `torch` 2.13.0, uploaded 2026-07-08, requires Python ≥3.10. `transformers` 5.20.2, uploaded 2026-10-19, requires Python ≥3.10. `pyarrow` 26.0.0, uploaded 2026-10-09, requires Python ≥3.11.

## 12. Deep-research run

`wf_3a9de5f1-8b7` finished on 2026-10-26 at 16:22 (109 agents, 0 errors). It fetched 27 sources, extracted 122 claims, verified 25 of them, and found 13 confirmed and 12 refuted, with 0 unverified. The result is in [deep_research_digest.md](deep_research_digest.md). The workflow's synthesis step also wrote into `PROJECT.md` and added four open questions (O-XSUM, O-OWTCFG, O-LIC, O-GAP). Those questions conflict with the owner's instruction that no open questions stay in the project, so the rewritten PROJECT.md closes each of them with a decision.

## 13. VQ state-dict keys and VQ forward semantics (2026-10-26)

- **Key list.** [converter/wmt14_tokenizer_keys.md](converter/wmt14_tokenizer_keys.md) lists 96 state-dict entries for the WMT14 tokenizer: 63 parameter entries with 144,002,816 scalars, and 33 persistent buffer entries with 17,391,616 scalars. Its UNRESOLVED list is empty. It was derived by reading `models/vqvae.py`, `models/vae.py` and `models/quant.py` at the pinned commit, with no code run. Its PyTorch rules P1 to P7 come from PyTorch's documented API. No PyTorch source is installed locally, so the converter checks the key set against the real file (step V1).
- **Load contract**, read this pass (`tokenization/vqvae_tokenizer.py` lines 14 to 22): `vqvae = VQVAE(**config.vqvae)`, then `vqvae.load_state_dict(torch.load(config.vqvae.checkpoint_path, weights_only=False, map_location='cpu')['model'], strict=True)`.
- **VQ semantics.** [reference/vq_forward_semantics.md](reference/vq_forward_semantics.md) was derived by reading the pinned code. Its lines were re-checked this pass: `quant.py` 11 to 38 (`Phi` keeps the default Conv1d initialization, and its output is `0.5·h + 0.5·Conv(h)`), 42 to 53 and 205 to 215 (the zero-initialized block, inactive for OWT), 330 to 339 (used only under `learned_all_sampling`), `next_concept.py` 183 to 195 (`post_upsample_convs`, zero-initialized, enabled by `owt.yaml` 61 to 63), `quant.py` 457 to 461 (one Bernoulli per forward call), 512 to 531 (commitment and codebook terms at every level, divided by `num_levels`; the truncation indices `kmin = max(min_keep − 1, 0)` and `kmax = len − 2`; the straight-through estimator), `vqvae.py` 110 to 118 (`reconstruct` unpacks three values where `self.vq` returns five, so it raises), and `owt.yaml` 120 to 124 (decay 0.99, epsilon 1e-5, commitment_cost 0.25).
- **Restart.** `quant.py` line 244 sets `restart_threshold=0.0`. The restart function is at lines 437 to 446.
- **Initialization of `wle`.** `next_concept.py` line 199 creates `nn.Embedding` without an initializer. `_init_weights` is never called (`ncm_facts.md`, section B), and `c_proj` is re-initialized at line 305.

## 14. Verification record for this pass (2026-10-26)

Commands were run from the repository root, with these results.

- `zig version`: `0.16.0`.
- `zig build test --summary all`: exit 0. The first line of the log is `Build Summary: 11/11 steps succeeded; 40/40 tests passed`. The log is in the session scratchpad as `zig_build_test_final.log`. No Zig source file was changed in this pass, so the count is the same as the M0a record.
- `zig fmt --check .`: exit 0.
- Header check (AGENTS.md, item 3): no output.
- Weight check (AGENTS.md, item 4): no output.
- Isolation check (AGENTS.md, item 5): no output.
- Link check over PROJECT.md, ROADMAP.md, README.md, AGENTS.md and the research folder: 13 files and 68 relative links, none broken.
- `git status --short`: modified AGENTS.md, PROJECT.md, README.md, `packages/hf_config/config.zig`, `fixtures_test.zig` and `manifest.zig`. Untracked: ROADMAP.md, the seven fixture folders listed in M0a, and `research/`. Nothing was committed or pushed.

Not run in this pass, by design:

- The pinned NCM reference code and the released pickle. Running them was refused in this session, and the procedure in [converter/PROCEDURE.md](converter/PROCEDURE.md) says how they are run.
- The Python golden helper, which step B0 creates.
- The Zig kernels, tokenizers, VQ code and writer, which phases 2 to 8 of the roadmap create.

## 15. Released checkpoint pins and Unicode data (2026-10-26)

Read through public endpoints this pass. The requests were `https://huggingface.co/api/models/hazyresearch/ncm-checkpoints` (the repository revision), `https://huggingface.co/api/models/hazyresearch/ncm-checkpoints/tree/main/wmt14` (file sizes and LFS hashes), and `https://huggingface.co/api/models/<repo>/tree/<revision>` for each Qwen3 dense size.

- Repository `hazyresearch/ncm-checkpoints`: revision `d342ce8aaea58830eef3b8ad473664673e41f692` (main, last modified 2026-10-16). Its tags include `license:mit`.
- `wmt14/tokenizer.pt`: 645,790,351 bytes, LFS SHA-256 `74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702`.
- `wmt14/generator.pt`: 480,582,539 bytes, LFS SHA-256 `ae580cbbb400a27b30c616634745d4c9cfa8b66806f936c6c26041e5b9640d4f`.
- `wmt14/config.yaml`: 2,177 bytes. `wmt14/README.md`: 218 bytes.
- Qwen3 tokenizer files at the six pinned revisions (git blob ids, first 12 hex digits): `tokenizer.json` `cd71f61a15a5`, `tokenizer_config.json` `417d038a63fa`, `vocab.json` `4783fe10ac3a`, `merges.txt` `31349551d90c`. They are identical in 0.6B, 1.7B, 4B, 8B, 14B and 32B.
- Unicode data, fetched from unicode.org on 2026-10-26 with `curl` (HTTP 200 for all four):
  - 9.0.0 `UnicodeData.txt`: 1,686,443 bytes, SHA-256 `68dfc414d28257b9b5d6ddbb8b466c768c00ebdf6cbf7784364a9b6cad55ee8f`.
  - 9.0.0 `CompositionExclusions.txt`: 8,355 bytes, SHA-256 `5623df16856ad4007c60bdfff6f054e087521becd24cb4006be69c3a1d851aee`.
  - 16.0.0 `UnicodeData.txt`: 2,175,362 bytes, SHA-256 `ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f`.
  - 16.0.0 `CompositionExclusions.txt`: 9,007 bytes, SHA-256 `89e83cf9cc8bef6c1f8bf77e42cf6f0341dfa42e66261f4dbe9b492e7a23c8ee`.
- PyPI `tokenizers==0.23.3` first upload 2026-10-09 (Python ≥3.10). PyPI `safetensors==0.8.0` first upload 2026-06-09 (Python ≥3.10). Both re-read this pass.
- Judge `openai-community/gpt2-large`, licence `mit`, and corpus `Skylion007/openwebtext`, licence `cc0-1.0`: from the saved API responses, re-read this pass.
- Reference statistics, read this pass: `scripts/eval_merge.py` line 99 (`batch_size=32` for the judge), lines 90 to 101 (the entropy, used at line 100), and lines 105 to 106 (`std(ddof=1)`).

## 16. Golden helper and `rope_parameters` (2026-10-26)

Ran `tools/golden_helper.py` with Python 3.13.5, torch 2.13.0+cpu, transformers 5.20.2, tokenizers 0.23.3 and safetensors 0.8.0. The helper accepts torch's `+cpu` local build suffix for the pinned 2.13.0 version. Its base seed is 271828, incremented by one in case order; the fixed token ids are `[1, 4, 7, 11]`. It generated tiny CPU checkpoints and JSON references for Llama default RoPE, Llama3 RoPE, Qwen2, Qwen3 and Mistral. Its `rope_tables.json` also holds the f32 `inv_freq`, `cos` and `sin` values for every position 0 to 4095 at head dimensions 64 and 128 with default bases 10,000 and 500,000, plus Llama3 inverse frequencies with the pinned parameters. A second run with the same default seed produced identical file bytes (`diff -r` returned no output). A controlled call with a mocked transformers version of 5.19.0 was refused with exit status 2.

Each generated Transformers 5.20.2 `config.json` uses the flat `rope_parameters` form and omits a top-level `rope_theta`. Default RoPE is serialized with `rope_type` and `rope_theta`; llama3 additionally includes `factor`, `low_freq_factor`, `high_freq_factor` and `original_max_position_embeddings`. This matches the schema accepted in B0a. The format and per-layer standardization behavior were checked in the pinned `transformers/modeling_rope_utils.py` (`RopeParameters`, `standardize_rope_params` and `nested_rope_parameter_keys`). Generated checkpoints and references live in `/tmp` and are not committed.

`nested_rope_parameter_keys` recognizes nested parameter maps by the config's declared layer types, and `standardize_rope_params` later selects parameters per layer type. The config reader has no layer-aware RoPE selection, so B0a reports nested maps as an unsupported `rope_parameters` feature instead of treating them as the default type. Running Zig `inspect` on all five generated directories returned `the checkpoint matches its configuration`, including both flat Llama3 parameters and the Qwen3 `head_dim` case.

## 17. System C math for RoPE parity (2026-10-26)

The installed pinned PyTorch 2.13.0 CPU wheel contains `torch/include/ATen/cpu/vec/vec_base.h`; `Vectorized<T>::pow` (line 689) applies `std::pow(values[i], exp[i])` to the f32 lane values used by the helper. The matching Zig target's system C math functions were checked without importing or executing NCM code: `ctypes.CDLL(None).powf`, `cosf` and `sinf` were called with f32 inputs and outputs rounded at each f32 operation. Compared with the B0 helper's pinned Transformers 5.20.2 `rope_tables.json`, all inverse frequencies matched bit-for-bit for the default tables at head dimensions 64 and 128 with bases 10,000 and 500,000. Across positions 0 through 4095, duplicated half-dimension cosine and sine values had maximum absolute difference `5.960464477539063e-08`; no value exceeded D53's `1e-6` bound. Zig 0.16.0 `std.math.pow(f32, ...)` instead differs by one ULP for head dimension 64, index 1, base 10,000, which produces a sine error above `1e-6` at later positions.

The supported host at the time of the check is Debian 13 with glibc 2.41. Debian's `/usr/share/doc/libc6/copyright` identifies the GNU C Library as LGPL-2.1-or-later; `/usr/share/common-licenses/LGPL-2.1` contains its licence text. The implementation calls system C math symbols and does not copy, bundle or statically vendor glibc. Zig 0.16's `std/Build/Module.zig` exposes `link_libc` and `linkSystemLibrary`, which are used to express the system runtime and separate `m` library dependency. Other targets need their platform C math library to provide the same symbols; their numeric parity remains gated by the golden tests.

The Zig B3 golden test then exercised every f32 cosine and sine entry for those four tables and passed; its reported maximum absolute error was `5.9605e-08` against D53's `1e-6` bound. The test reads the ignored, generated `zig-out/golden/rope_tables.json` fixture; no generated output is tracked.

The B4 test passed against every one of the 32 helper inverse frequencies for the pinned Llama 3.2-1B parameters (`head_dim=64`, `rope_theta=500000`, `factor=32`, `low_freq_factor=1`, `high_freq_factor=4`, `original_max_position_embeddings=8192`). Maximum relative error was `0.0`, within D53's `1e-6` bound. The inputs and expected values came from the helper's Transformers 5.20.2 `rope_tables.json`; no NCM reference was run.

## 18. Attention and mask golden helper data (2026-10-26)

Extended `tools/golden_helper.py` with an additional `qwen3_attention_bias` tiny case using the pinned Qwen3 implementation, grouped-query attention (`num_attention_heads=2`, `num_key_value_heads=1`, `head_dim=8`), Q/K/V/O biases, and the model's Q/K RMSNorm. Before saving and running the model, the helper now sets every attention bias to a deterministic nonzero ramp and Q/K norm weights to deterministic nonunit ramps. It records the normalized attention input, projection and norm weights, RoPE values, Transformers materialized causal mask, and the post-output-projection attention result. It also adds an `attention_mask` field generated by Transformers 5.20.2 `create_causal_mask` for full-attention cases and `create_sliding_window_causal_mask` for Mistral, forcing the eager interface to materialize masks. Existing B0 reference fields for the original five cases were compared before and after helper regeneration and were unchanged. Loading the generated safetensors with pinned `safetensors==0.8.0` confirmed each of the four biases is nonzero and matches the JSON values, and both norm scales are nonunit and match the JSON values.

The Zig B5 attention test used the generated Qwen3 fixture and matched all post-projection output values within D53's per-position bound; maximum absolute error was `3.725e-09`. The B6 test compared every cell of the materialized Transformers masks for six tiny cases (all five original model families plus the Qwen3 bias/QK-norm case). For Mistral, the four-token sequence is longer than its three-token window; its final query admits keys 1, 2 and 3 and excludes key 0, matching `q − W < kv ≤ q`. Every mask cell matched exactly. Golden-backed tests skip with an explicit `SKIP golden acceptance` message when their generated fixture is absent, so ordinary `zig build test` works on a clean checkout without Python installed. Full B3–B6 acceptance is exercised by first running `/tmp/ncmorph-b0-venv/bin/python tools/golden_helper.py --output-dir zig-out/golden`, then `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache /tmp/ncmorph-zig-0.16.0/zig build test --summary all`; with generated files present these tests run, rather than skip. The helper output remains under ignored `zig-out/`; no weights or generated references are tracked.

## 19. Source decoder forward pass (2026-10-26)

`src/decoder.zig` composes the pinned f32 kernels into one shared decoder forward path: embedding lookup, rotary tables, causal/sliding mask, pre-norm attention/residual, SwiGLU MLP/residual, final RMSNorm and tied/untied output projection. B7's shape test builds a one-layer synthetic Llama spec and checks finite logits with shape `[T, vocab_size]`. `zig build test` passed with generated fixtures (56/56 tests), and with `zig-out/golden` temporarily absent it passed 52 tests and skipped the four fixture-backed B3–B6 tests; the temporary move was restored immediately. The generated fixtures remain ignored and no NCM reference code was run.

## 20. Source decoder loader and M0b gate (2026-10-26)

`src/loader.zig` reads config and all safetensors headers, checks the manifest and sharded index, then reads one tensor at a time in 64 KiB chunks into F32 arrays. It supports F32, F16 and BF16, ignores the optional per-layer `inv_freq` buffers, and compares a stored tied head to embeddings bitwise after widening. Tests exercise the three dtype decoders, tied-head equal and unequal cases, and a valid safetensors file with an invalid manifest; the latter reports `InvalidManifest` with zero tensor-data reads. The helper records each stored tensor's flattened F32 bit patterns in `reference.json`; B8 tests loaded all tensors bit-for-bit for six fixtures: Llama default (12 tensors), Llama3 (12), Qwen2 (14), Qwen3 (13), Mistral (12), and Qwen3 attention-bias (17). Mistral is emitted as three safetensors shards with an index, and this load passed the same exact-value test.

B9 compared `[T, vocab_size]` logits with the pinned Transformers 5.20.2 helper for all five required families. Maximum absolute errors were: Llama default `2.980232239e-8`, Llama3 `5.960464478e-8`, Qwen2 `5.960464478e-8`, Qwen3 `5.960464478e-8`, Mistral `2.980232239e-8`; the overall maximum `5.960464478e-8` was within D53's per-position bound. The final generated-fixture test run passed 61/61 tests. No NCM reference code ran, and generated weights/references remain ignored under `zig-out/`.

## 21. M0c real-checkpoint C2 attempt (2026-10-26)

Recorded D63 before implementation: use ids `[1, 4, 7, 11]` at positions `[0, 1, 2, 3]`, with no tokenization; generate the pinned Transformers 5.20.2 reference locally on CPU using `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, local_files_only=True)`, evaluation mode, `torch.inference_mode()` and `use_cache=False`. The helper ran against SmolLM2-135M's local config and safetensors. `/home/dev/ncmorph-testing` is read-only, so `/tmp/ncmorph-c-checkpoints/smollm2-135m` contains symlinks to the original config and weights and the generated, uncommitted `ncmorph_reference.json` (3,355,058 bytes). The pinned Python helper loaded all 272 tensors and wrote four positions of 49,152 logits each. The config metadata says the checkpoint is BF16; no remote files were fetched.

Command: `/tmp/ncmorph-b0-venv/bin/python tools/golden_helper.py --real-checkpoint-dir /tmp/ncmorph-c-checkpoints/smollm2-135m` completed successfully. Command: `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints /tmp/ncmorph-zig-0.16.0/zig build test --summary all` failed in the first real-checkpoint case, SmolLM2-135M, and the ordered test stopped before Qwen2.5 or Qwen3 acceptance. At token index 3, vocabulary index 48,518, the maximum absolute logit difference was `4.570385932922`. The reference row's maximum absolute logit was `18.5`, so D53's tolerance at that position was `0.00185`; the observed difference exceeds it by about 2,470×. D53 was not changed. The run completed loader and decoder execution; it did not mark C1–C4 done. Qwen2.5/Qwen3 Zig gates have not been run. This result blocks Phase 3 pending review of the reference-precision interpretation; no F32 reference substitution or further real-checkpoint gate was attempted.

## 22. BF16 operation semantics for decoder parity (2026-10-26)

Recorded decision D64 before implementation. The installed pins are torch `2.13.0+cpu` and Transformers `5.20.2`. In the pinned `transformers/models/llama/modeling_llama.py`, `LlamaRMSNorm.forward` (lines 61–66) casts input to F32 for square/mean/rsqrt and normalized multiply, casts normalized values back to `input_dtype`, then multiplies the module weight. `apply_rotary_pos_emb` (lines 115–118) performs `(q * cos) + (rotate_half(q) * sin)` and the same for key, with each tensor operation returning the tensor dtype. `eager_attention_forward` (lines 185–199) computes QK matmul then scalar scaling, applies the mask, computes softmax explicitly in F32 and casts to query dtype, and computes the probability/value matmul. `LlamaMLP.forward` (lines 152–165) applies SiLU, multiplies by the up projection, then applies the down projection. Decoder blocks add the attention and MLP results to residual tensors in the activation dtype (`modeling_llama.py` lines 294–313).

The pinned CPU runtime directly confirmed that BF16 multiplication, addition, SiLU, matmul, and softmax-cast all return BF16 tensors. A deterministic BF16 vector dot product returned `0.1630859375`, equal to F32 products/sum followed by BF16 rounding, and different from rounding each product/addition (`0.16015625`). The CPU wheel uses MKL and reports `USE_CUDA=0`; no CUDA or GPU path was involved. The prior F32 diagnostic against the same stored weights passed D53 (max `0.00132465362549`, tolerance `0.00204430084229`), whereas BF16 Transformers output differed from current Zig F32 by `4.570385932922` (D53 tolerance `0.00185`). This evidence supports preserving activation dtype, rather than changing the real reference or tolerance.

Follow-up source check: the pinned PyTorch v2.13.0 CPU `silu_kernel` in `aten/src/ATen/native/cpu/Activation.cpp` lines 1098–1112 uses `x / (1 + exp(-x))` for reduced floating types with F32 intermediates before conversion to BF16. A pinned-runtime probe over 100,001 F32 values compared the BF16 outputs: `torch.nn.functional.silu(x)` differed at 2 positions from `x * (1 / (1 + exp(-x)))`, but only 1 from `x / (1 + exp(-x))`. `silu_bf16` follows the pinned division form.

## 23. BF16 decoder attempt and stop point (2026-10-26)

D64 was recorded before adding `bf16_compute` to decoder weights. The loader selects this mode only when every non-`inv_freq` tensor header is BF16; F32 behavior remains the default. The BF16 kernel path rounds projections to BF16 after F32 accumulation, carries BF16 tensor values through norm/rotary/attention/MLP/residual operations, and returns BF16-rounded logits. The golden helper now adds `bf16_logits` by deep-copying each tiny F32 model, converting model parameters to BF16, and running the same fixed ids in evaluation/inference mode. Existing F32 tiny outputs are unchanged.

The ordinary F32 B9 results remained unchanged. The new tiny BF16 comparison failed on the first `llama_default` case at token 1, vocabulary 5: Transformers expected `0.12060546875`, Zig returned `0.1201171875`, absolute difference `0.00048828125`; D53 tolerance was `0.0001`. SmolLM2 was then run once with the BF16 implementation before the final SiLU expression correction: maximum logit difference `4.4375` at token 3, vocabulary 48,518; maximum row tolerance `0.002725`. This exceeds D53. The preceding BF16 implementation run had max difference `4.570385932922` at that same index. Per the roadmap gate rule, no Qwen checkpoint was run. The SiLU expression was subsequently corrected to the pinned PyTorch v2.13.0 CPU `silu_kernel` division form based on the checked source and a pinned-runtime probe; no real-checkpoint gate was rerun after that correction. Therefore C2 remains blocked and no claim of BF16 parity is made. D53 has not changed.

## 24. BF16 parity backend and linear bias order (2026-10-26)

Recorded D65 before implementation. Pinned Transformers 5.20.2 `PreTrainedModel.from_pretrained` reads and forwards the explicit `attn_implementation` keyword (`modeling_utils.py` lines 1418–1420), and the pinned Llama `eager_attention_forward` sequence is exactly the QK matmul, scale, mask, F32 softmax cast to query dtype, and probability/value matmul implemented by this decoder. A deterministic tiny Llama CPU probe under PyTorch `2.13.0+cpu` showed identical F32 logits for SDPA and eager to `1.4901161193847656e-8`, but BF16 logits differed by `0.0009765625`; D53's logit tolerance is `1e-4` per position. Therefore D63 now requests eager explicitly. The real SmolLM2 loading probe confirmed `from_pretrained(dtype=torch.bfloat16, attn_implementation="eager")` yields BF16 parameters and an F32 `model.rotary_emb.inv_freq` nonpersistent buffer. The tiny BF16 helper must therefore convert parameters only and retain F32 rotary buffers, instead of calling `model.to(dtype=torch.bfloat16)` which also narrows buffers.

Bias ordering probe: pinned CPU `torch.nn.functional.linear` for BF16 inputs/weights/bias matched F32 accumulation plus BF16 bias followed by one BF16 output cast. A seed-0 width-32 example returned `-2.0625`; separate BF16 matmul output rounding followed by BF16 bias addition returned `-2.078125`. This establishes that BF16 `F.linear` bias must be added before final output rounding. The probe used the local CPU wheel (`USE_CUDA=0`); no GPU or NCM code ran.

## 25. SmolLM2 retry after reference and BF16 operation fixes (2026-10-26)

The five tiny BF16 golden comparisons now all pass with maximum absolute logit error `0.0`; the five F32 B9 comparisons remain unchanged with overall maximum `5.960464478e-8`. The BF16 helper casts only model parameters, keeps the F32 `rotary_emb.inv_freq` buffer (verified against `from_pretrained`), and selects eager attention. BF16 linear now adds the bias to the F32 accumulated sum before its BF16 output conversion. The C-gate test accepts `NCMORPH_CHECKPOINT_CASE` to run one named case, so this retry selected only SmolLM2.

The eager/BF16 reference was regenerated using `/tmp/ncmorph-b0-venv/bin/python tools/golden_helper.py --real-checkpoint-dir /tmp/ncmorph-c-checkpoints/smollm2-135m`. The single real gate command was `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints NCMORPH_CHECKPOINT_CASE=smollm2-135m /tmp/ncmorph-zig-0.16.0/zig build test --summary all`. It exceeded D53 at token 0, vocabulary index 19,618: maximum absolute logit difference `0.521484375`; maximum row tolerance was `0.002725`. The gate stopped immediately; no Qwen checkpoint was read or run. No tolerance was changed. C2 remains blocked pending review of this remaining mismatch.

## 26. BF16 dot reduction order investigation (2026-10-26)

The pinned CPU wheel reports `torch==2.13.0+cpu`, `USE_CUDA=0`, CPU capability `AVX512`, MKL 2024.2 and oneDNN 3.12.0. On the isolated width-576 adversarial input `x = [1] + [2^-24] * 574 + [1/256]`, `w = ones(576)`, both `torch.nn.functional.linear(x, w)` and `x @ w.T` return BF16 `1.0078125`; serial F32 accumulation returns `1.00390625`, which rounds to BF16 `1.0`. Four independent F32 sums, with product `i` accumulated into `partial[i % 4]` and partials combined in ascending order, return the BF16 result observed from PyTorch.

The pinned v2.13.0 source `aten/src/ATen/native/cpu/BlasKernel.cpp`, `sum` lines 74–94, defines four independent partial sums, distributes consecutive terms to the four accumulators, adds a tail to the first, then combines partials 1 through 3 into partial 0. The same source's reduced-precision `gemm_notrans_` lines 144–157 calls this helper for each BF16 dot. The wheel also contains optimized routes: `CPUBlas.cpp` dispatches BF16 GEMM through oneDNN/MKL where available, and `ReducedPrecisionFloatGemvFastPathKernel.cpp` uses vector accumulators and a distinct reduction tree for GEMV. The width-576 probe establishes that the four-way schedule matches that isolated case but does not establish which optimized backend path produced the result. Section 27 records the counterexample in the actual SmolLM2 layer-0 O projection; therefore no general accumulator schedule was adopted from this probe.

## 27. BF16 first-block localization and reduction counterexample (2026-10-26)

To localize the authorized C2 retries, `tools/golden_helper.py` now records BF16 Transformers hidden states and first-block diagnostic stages. `decoder.forward_with_trace` records the corresponding Zig stages only for the acceptance diagnostic. The pinned Transformers 5.20.2 run reports the same block input and input RMSNorm output as Zig at every one of the four positions in SmolLM2 layer 0. At position 0, the entire layer-0 block output is bit-identical. At position 2, the first reported discrepancy is the attention output (`0.000122070313` max); the post-attention residual and RMSNorm differ by `0.000244140625`, then gate projection by `0.0078125`, and the MLP/block output by `0.03125`. The widening hidden-state differences continue in later blocks, demonstrating that an early BF16 rounding difference is amplified.

Pinned PyTorch CPU probes show that layer-0 Q/K RoPE values exactly match when computed with the Zig rotate-half operation sequence; token 2 QK scores, scaled scores, probabilities and context also match when the dot accumulates serially in F32 and rounds at the BF16 output boundary. A direct capture of the layer-0 O projection inputs and weight from the real model showed serial F32 accumulation followed by BF16 rounding matches every pinned PyTorch output bit-for-bit for this operation. Four independent F32 accumulators, although matching the isolated width-576 adversarial case in section 26, differ on two layer-0 O-projection outputs (max `0.000122070313`, both at token 2). This confirms that a source fallback's schedule cannot be applied to every optimized CPU shape. The four-way implementation and its test were reverted; production BF16 matrix multiply remains serial F32 accumulation at this point.

The corrected-precision C2 retry after applying four-way accumulation failed at token 3, vocabulary index 48,518, with maximum logit error `1.21875` against maximum row tolerance `0.002725`. The per-token trace locates the first hidden mismatch at layer 0, token 2. This result is preserved as failed evidence; D53 is unchanged, and no Qwen model was run. The remaining concrete discrepancy is the layer-0 O projection for token 2, where the Zig serial F32 result needs comparison against the exact captured reference values to determine whether the attention input or projection's matrix kernel is responsible.

## 28. Long BF16 dot schedule experiment (2026-10-26)

The restored serial implementation produces bit-identical C2 block-0 outputs. Its first difference is block 1's 1536-wide MLP down projection, where the reference hidden input, weight and output were captured from the pinned CPU model: a serial-F32 schedule differs in three BF16 outputs (max `0.015625`), while eight and 64 interleaved F32 accumulators match every output bit. A four-way schedule has one differing output (`9.536743e-7`). In the first two blocks' captured linear operations, the same eight-way candidate matches the 1536-wide down projections in both blocks; the corresponding source fallback (`BlasKernel.cpp`, four independent sums) does not. Narrower 576-wide operations do not share one consistent winner: serial matches the first-block O projection exactly, and eight-way differs on two of its outputs. This is evidence for a shape-dependent oneDNN/MKL reduction path, not a universal PyTorch accumulator rule.

As a bounded diagnostic experiment, a local Zig run temporarily used eight independent F32 accumulators only for BF16 dots with `input_width >= 1024`, keeping the already matching 576-wide path serial. It was not recorded as a general backend fact or adopted policy. It improved the captured first two 1536-wide down projections but did not pass C2; section 29 records the result. The experiment was reverted. D53 remains unchanged and no Qwen gate is authorized before C2 passes.

## 29. BF16 stage trace and schedule follow-up (2026-10-26)

For a bounded diagnosis, the helper and decoder temporarily captured query/key/value projections, post-RoPE query/key, scores, probabilities and contexts through layer 4. With the temporary eight-lane reduction only on widths at least 1024, hidden states after layers 0–3 matched the pinned reference exactly. At layer 4, Q/K/V, scaled scores, probabilities, attention context and attention output also matched exactly; the first remaining difference was the MLP gate projection (maximum `0.0009765625`), followed by the gate/up product (`0.000122070313`), MLP down projection (`0.001953125`) and block output (`0.0009765625`). Layer 1's gate projection differed by `7.629395e-6`, while its block output remained exact under the wider down-projection experiment. This shows that fixing the captured layer-1 down projection did not establish parity for all BF16 linears.

A pinned CPU probe captured all 30 SmolLM2 gate projections and compared serial versus eight-interleaved F32 accumulation before BF16 output rounding. Results varied by layer: eight lanes matched layer 1 and layer 4, but had three differing outputs at layer 6 where serial had one, one differing output at layer 8 where serial had none, and two differing outputs at layer 15 for both. Therefore neither schedule is a verified general rule, even for the same named projection family and shape. The temporary wide-dot implementation and expanded internal trace instrumentation were reverted; standard acceptance diagnostics remain at their earlier scope. No source-supported CPU-backend accumulation schedule was established. The C2 attempt still failed D53 at token 0, vocabulary index 19,618, with maximum logit difference `0.521484375` against maximum row tolerance `0.002725`; no Qwen model was run.

## 30. Pinned oneDNN AVX-512 BF16 K blocking (2026-10-26)

The authorized C2 trace first diverged at layer 11's up projection, output index 763: scalar VDPBF16PS emulation produced BF16 `0x3fa7`, while pinned Transformers CPU produced `0x3fa8`. The captured input and weight have shape `[4,576]` and `[1536,576]`. `DNNL_VERBOSE=1` reports both layer 0 and layer 11 up projections as `brg_matmul:avx512_core_bf16`, with the same BF16 source/weight/destination layouts and `4x576:576x1536` dimensions. A direct AVX-512 `_mm512_dpbf16_ps` scalar-per-output loop reproduces Zig's one-ULP mismatch, so the instruction's pair operation alone does not reproduce this output.

The pinned oneDNN source identifies the missing reduction boundary. In `src/cpu/x64/matmul/brgemm_matmul_utils.cpp`, `compute_blocking_heuristic_avx512` sets `use_extended_k_blk` only for K>1024 and a non-transposed source, then selects `k_blk = min(K, 1024)` when extended and `min(K, 512)` otherwise (lines 883–886 in the checked source). `is_buffer_c_required` buffers C when accumulation and destination types differ and K exceeds a chunk or has a K-block remainder (lines 831–839). In this BF16 linear path that means K=576 is accumulated in 512- and 64-element F32 partials before BF16 conversion; K=1536 uses 1024- and 512-element partials. The registered implementation is `brg_matmul:avx512_core_bf16`; the JIT `dot_product` selects `vdpbf16ps` for BF16 on AVX-512 (the same file contains this dispatch in `jit_brgemm_kernel.cpp`, around line 2282). The inspected source carries oneDNN's Apache-2.0 license; no source was copied and no runtime dependency was added.

Using a temporary copy of the Zig kernel with exactly this source-derived K-block rule, all 14 captured projections from layers 0–1 matched the pinned Torch outputs bit-for-bit, including the K=1536 down projections. A focused capture of layer 0 and layer 11 up projections also matched every output (0/12,288 mismatches); the previous scalar-without-512-split version had 1 mismatch in layer 11. The generated data and temporary harness stayed under `/tmp`. This establishes the K-block behavior for the pinned CPU backend, not a universal reduction schedule.

After applying that rule, the C2 command `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints NCMORPH_CHECKPOINT_CASE=smollm2-135m /tmp/ncmorph-zig-0.16.0/zig build test --summary all` passed all 68 tests. SmolLM2's four-token, full-vocabulary logit comparison had maximum absolute difference `0.0` against maximum row tolerance `0.002725`; the test checks every token/vocabulary element and fails if any exceeds its per-row D53 bound. No tolerance changed. The output also printed a large final raw-block hidden-state difference because Transformers' `capture_outputs` (pinned `transformers/utils/output_capturing.py`) replaces `hidden_states[-1]` with `last_hidden_state`, which is after the final RMSNorm. That diagnostic compared incompatible stages, was not an acceptance assertion, and was corrected to omit only this unmatched pre-norm final block output. A direct pinned-model hook confirmed the raw final block output differs from the tied final hidden-state tensor by `6987.625`; all earlier hidden-state diagnostic indices correspond to raw block outputs. The C2 roadmap gate is therefore satisfied by its specified full-logit criterion; C3 is next in order.

## 31. Pinned PyTorch CPU RMSNorm reduction (2026-10-26)

After regenerating the Qwen2.5-0.5B eager BF16 reference with layer diagnostics, C3's first discrepancy is Qwen2 layer index 2 input RMSNorm. Its block input is exactly equal to the reference. The helper's source expression in Transformers 5.20.2 (`Qwen2RMSNorm.forward`) widens to F32, squares, takes `.mean(-1, keepdim=True)`, adds epsilon, applies `torch.rsqrt`, multiplies, and casts to BF16. Re-evaluating this expression in pinned Torch 2.13.0+cpu reproduces the diagnostic exactly. At token 1, Torch's mean is `0.07995878159999847`; a scalar left-to-right F32 sum/divide in Zig gives `0.07995861768722534`, causing six normalized BF16 values to differ by up to `0.0078125`. The C3 full-logit gate after eager-reference regeneration remains failed at max difference `0.259765625` versus D53 max row tolerance `0.0016375`.

The pinned PyTorch v2.13.0 sources explain the CPU operation: `aten/src/ATen/native/ReduceOps.cpp`, `mean_out` (lines 1276–1333), implements CPU mean as `sum_out(...).div_(dim_prod)`. `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp` delegates float sum to `cascade_sum` in `aten/src/ATen/native/cpu/SumKernel.cpp` (`sum_kernel_impl`, lines 565–578). For contiguous inner reduction, `SumKernel.cpp` uses `vectorized_inner_sum`, `row_sum` and `multi_row_sum` (lines 287–423); its four-level and four-row accumulation structure is not a serial sum. The installed pinned wheel reports AVX512 as the general CPU capability, but `nm -C /tmp/ncmorph-b0-venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.so` exposes DEFAULT and AVX2 `sum_kernel_impl` variants but no AVX-512 specialization. The captured squared input is contiguous shape `[1, 4, 896]`; outputs were invariant under Torch thread counts 1, 2, 4 and 8.

A scratch simulation of the exact source accumulation structure over the four captured rows matched pinned `torch.sum` bit-for-bit with eight F32 lanes (the AVX2 `Vectorized<float>` width); the corresponding 16-lane AVX512 simulation did not. This explains the variance and normalized BF16 discrepancy with source-backed arithmetic and measured tensor inputs, rather than a shape-only schedule. The Zig implementation follows the pinned CPU sum order for RMSNorm in both F32 and BF16 modes. It does not copy PyTorch source and introduces no dependency. PyTorch v2.13.0's checked `LICENSE` is BSD-style, with source redistribution notice and disclaimer terms; no third-party code is copied. The scope remains the pinned CPU reference backend; other PyTorch builds or device backends can dispatch differently.

The source-order emulation was implemented as the pinned eight-lane sum cascade for RMSNorm in both F32 and BF16 paths, with synthetic tests for a 896-element reduction and a non-vector-width tail. C3 was then rerun, in order, with the Qwen2.5 reference regenerated by the pinned eager/BF16 helper. The command `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints NCMORPH_CHECKPOINT_CASE=qwen2.5-0.5b /tmp/ncmorph-zig-0.16.0/zig build test --summary all` passed 69/69 tests. All traced block outputs and all four positions' full-vocabulary logits matched exactly (maximum logit difference `0.0`, max row tolerance `0.0016375`). The prior post-regeneration failure (`0.259765625`, token 1/vocabulary 75694) is preserved as pre-fix evidence; the earlier `0.6953125` result used a stale reference lacking the new diagnostics. D53 is unchanged. C3 passes; C4's result is in section 32.

## 32. Qwen3 real-checkpoint reference diagnostics and C4 (2026-10-26)

The first C4 run stopped at `InvalidFixture`: the helper had no `attention_context` diagnostics for Qwen3 because it generated that stage only for models without `q_norm`. No model parity assertion ran in that attempt. The pinned Transformers v5.20.2 `Qwen3Attention.forward` applies q/k projection, per-head q_norm/k_norm, rotary embedding, grouped-query repetition, causal eager attention, and the output projection; `eager_attention_forward` returns context before the output projection. The helper now derives Qwen3 context using the pinned `Qwen3RMSNorm`, `apply_rotary_pos_emb` and `repeat_kv` functions and the same eager matmul/softmax sequence. A scratch validation projected the generated context through every captured layer's pinned `o_proj`: all 12 layer outputs matched `attention_output` bit-for-bit (0 mismatches).

After regenerating `/tmp/ncmorph-c-checkpoints/qwen3-0.6b/ncmorph_reference.json` with the updated helper, the command `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints NCMORPH_CHECKPOINT_CASE=qwen3-0.6b /tmp/ncmorph-zig-0.16.0/zig build test --summary all` passed 69/69 tests. Every captured diagnostic stage through layer 11 matched; all four full-vocabulary logit rows matched exactly, with maximum difference `0.0` and maximum row tolerance `0.001575`. The reference contains 151,936 logits per position. The existing tied-head loader check runs on this checkpoint as part of acceptance. D53 is unchanged. C4 passes; C5 was not run at the time of this entry.

## 33. Llama 3.2-1B real-checkpoint C5 (2026-10-26)

The authorized official Llama 3.2-1B checkpoint was copied from `/home/dev/ncmorph-testing/llama-3.2-1b` into `/tmp/ncmorph-c-checkpoints/llama-3.2-1b`; source and copy `model.safetensors` SHA-256 both equal `68a2e4be76fa709455a60272fba8e512c02d81c46e6c671cc9449e374fd6809a`. The pinned helper generated an eager/BF16 reference of 128,256 logits per position. The C2-C5 acceptance test was extended with this case and model type `llama` to execute the C5 gate.

The command `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-cache NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints NCMORPH_CHECKPOINT_CASE=llama-3.2-1b /tmp/ncmorph-zig-0.16.0/zig build test --summary all` passed 69/69 tests. The four full-vocabulary logit rows and all captured diagnostics through layer 11 matched exactly; maximum logit error was `0.0` versus maximum row tolerance `0.00185`. The 15 reported hidden-state outputs also matched exactly. D53 is unchanged. C5 passes.

## 34. Real-checkpoint reference provenance schema (2026-10-26)

The pinned environment was checked with `/tmp/ncmorph-b0-venv/bin/python`: Python 3.13; `transformers==5.20.2` at commit `18329f1065382e54db698e242cf98bd3f3e5a766`; `torch==2.13.0+cpu` at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`; oneDNN enabled, version 3.12.0, git hash `80afa71049cd69a3df32adcccb623b12cd7baa22`; x86_64, runtime CPU capability AVX512; and `torch.__config__.show()` reports CPU capability usage AVX512. The real-reference helper sets PyTorch's thread count to one, loads BF16 on CPU, explicitly requests eager attention, checks the selected config backend is still eager, and uses eval plus `torch.inference_mode()` with `use_cache=False`. D63, D65, D67 and D68 describe why these reference/runtime details matter to the computed parity.

The helper now writes schema version 2 provenance fields for the Transformers pin and source commit, PyTorch wheel version and source commit, CPU device, BF16 dtype, selected attention backend, PyTorch thread count, oneDNN availability/version/hash, CPU architecture, runtime capability and build capability. The C-gate loader rejects a missing or mismatched field before reading any checkpoint weights and prints a command to regenerate the reference with the pinned helper. A unit test accepts a complete matching record, rejects one missing `attention_backend`, and rejects an SDPA backend marker. This prevents an older fixture created under the default SDPA implementation from being mistaken for the eager D63 reference. D53 is unchanged.

The four authorized references were regenerated individually by `/tmp/ncmorph-b0-venv/bin/python tools/golden_helper.py --real-checkpoint-dir <checkpoint-dir>` and gated sequentially with the pinned Zig 0.16.0 compiler. C2 SmolLM2 passed 71/71 tests, maximum logit difference `0.0` (maximum row tolerance `0.002725`); C3 Qwen2.5 passed 71/71, difference `0.0` (tolerance `0.0016375`); C4 Qwen3 passed 71/71, difference `0.0` (tolerance `0.001575`); and C5 Llama 3.2-1B passed 71/71, difference `0.0` (tolerance `0.00185`). Each gate compares all four positions and the full vocabulary. All reported captured diagnostic stages matched exactly. No real-checkpoint gate changed D53.

## 35. Zig 0.17.0 upgrade (Z1)

The official Zig 0.17.0 Linux x86_64 archive was staged at `/tmp/zig-x86_64-linux-0.17.0.tar.xz`; its SHA-256 is `1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026`. The compiler is `/tmp/zig-x86_64-linux-0.17.0/zig`, and `zig version` prints `0.17.0`. The official [Zig 0.17.0 release notes](https://ziglang.org/download/0.17.0/release-notes.html) document removal of array multiplication syntax and the build-run passthrough argument migration. The build script's former `if (b.args) |args| run_cmd.addArgs(args)` did not compile because `Build.args` is removed; the corresponding 0.17 `Run.addPassthruArgs()` API is at `lib/std/Build/Step/Run.zig:593`. The decoder's test-only repeated-array initializer was rewritten as an explicit four-element initializer. No other compile fixes were required.

The D29 names were checked against the compiler's `std_dir`, `/tmp/zig-x86_64-linux-0.17.0/lib/std`: `crypto/sha2.zig` has `Sha256` at line 23 and `hash` at line 108; `Random.zig` has `Xoshiro256` at line 24 and `floatNorm` at 343; `Random/Xoshiro256.zig` has `random` at 20; `mem.zig` has `readInt` at 1957 and `writeInt` at 2081; `math.zig` has `exp` at 334; `testing.zig` has `expectApproxEqAbs` at 258 and `expectApproxEqRel` at 294; `Io/Dir.zig` has `createFile` at 641; `Io/File.zig` has `writer` at 597; `Io/Writer.zig` has `print` at 698; `unicode.zig` has `utf8Encode` at 44 and `utf8Decode` at 101; and `json.zig` re-exports `parseFromSliceLeaky` at 84, implemented in `json/static.zig:89`. Every listed name remains available. `build.zig.zon` now requires at least 0.17.0, and AGENTS.md records the target and the two migration points.

A clean-cache command with checkpoint variables unset,
`env -u NCMORPH_CHECKPOINT_DIR -u NCMORPH_CHECKPOINT_CASE ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-0.17-cache ZIG_LOCAL_CACHE_DIR=/tmp/ncmorph-zig-0.17-local-fresh /tmp/zig-x86_64-linux-0.17.0/zig build test --summary all`, completed with `Build Summary: 17/17 steps succeeded; 70/71 tests passed (1 skipped)` and `test success`. The one skip is the explicit real-checkpoint acceptance skip when the checkpoint directory is absent. B9 ran all five family cases; its reported overall maximum absolute logit error was `0.00000004470348358`.

The four C gates were each executed with a separate, initially empty local Zig cache so the environment-specific checkpoint test could not be served from a prior cached run. Every command used `ZIG_GLOBAL_CACHE_DIR=/tmp/ncmorph-zig-0.17-cache`, `NCMORPH_CHECKPOINT_DIR=/tmp/ncmorph-c-checkpoints` and the indicated `NCMORPH_CHECKPOINT_CASE`, followed by `/tmp/zig-x86_64-linux-0.17.0/zig build test --summary all`:

- `smollm2-135m`, local cache `/tmp/ncmorph-zig-0.17-local-gates`: explicitly named in the live output; 71/71 tests passed, maximum logit difference `0.0` versus row-tolerance maximum `0.002725`.
- `qwen2.5-0.5b`, local cache `/tmp/ncmorph-zig-0.17-local-c3`: 71/71 passed, difference `0.0` versus `0.0016375`.
- `qwen3-0.6b`, local cache `/tmp/ncmorph-zig-0.17-local-c4`: 71/71 passed, difference `0.0` versus `0.001575`.
- `llama-3.2-1b`, local cache `/tmp/ncmorph-zig-0.17-local-c5b`: 71/71 passed, difference `0.0` versus `0.00185`.

All four runs report `C2-C5 overall maximum absolute logit difference: 0.0`; full-vocabulary comparisons and captured diagnostics passed. An intermediate C3 attempt reusing the C2 cache returned cached steps and is not counted; C3 was then run successfully using its separate fresh cache. The first C5 attempt was interrupted before completion; the successful C5 result above is the separate fresh-cache run. D53 is unchanged.

The final `zig fmt --check .` exited 0. The required header check printed nothing; the weight check printed nothing (grep exit 1, meaning no listed weight files); the package isolation check printed nothing (grep exit 1, meaning no `ncm`/`ncmorph` matches in Zig package sources). The command sequence is recorded in the task receipt. No Git operation was performed.

## 36. oneDNN enabled-state provenance guard (2026-10-26)

`tools/golden_helper.py` previously used `torch.backends.mkldnn.is_available()` both as a runtime guard and as the `onednn_enabled` provenance value. Availability only establishes that the backend is present; it does not reflect the independent `torch.backends.mkldnn.enabled` switch. The helper now checks both conditions before loading the checkpoint: unavailable oneDNN and available-but-disabled oneDNN have distinct errors. Provenance records the actual `.enabled` boolean. No reference was regenerated and no real model was loaded for this fix.

The regression command `/tmp/ncmorph-b0-venv/bin/python tools/test_golden_helper.py -v` passed `test_disabled_onednn_fails_before_checkpoint_loading`. The test supplies a backend where `is_available()` is true and `enabled` is false, expects the disabled-backend error, and asserts `AutoModelForCausalLM.from_pretrained` was never called. A pinned-runtime check reported `oneDNN available=True enabled=True`. The existing four external C references (`smollm2-135m`, `qwen2.5-0.5b`, `qwen3-0.6b`, `llama-3.2-1b`) each record `onednn_enabled=true`, version `3.12.0`, and git hash `80afa71049cd69a3df32adcccb623b12cd7baa22`; their provenance remains valid. D53 and all checkpoint gates are unchanged.
