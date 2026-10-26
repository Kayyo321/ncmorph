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
