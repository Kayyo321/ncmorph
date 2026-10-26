# NCM fact sheet (arXiv:2610.14162v1 and HazyResearch/ncm @ c70274273f489abc73832b2b98584a8483b6b203)

## 0. Provenance and conventions
- Paper text read in full (`ncm_2610_14162.txt`, 2434 lines); reference list scanned for dataset entries. Cited as P§, Eq., Alg., Table.
- `ncm_pinned/models_next_concept.py` and `ncm_pinned/next_concept.py` are byte-identical; both hash to the pinned blob 25ee56d5bbcaa21558768b279490ad21f5a12cb9. I used `models_next_concept.py`.
- Five pinned files are missing from `ncm_pinned/`: `models/quant.py`, `scripts/eval_merge.py`, `utils/registry.py`, `tokenization/sudoku_tokenizer.py`, `requirements.txt`. I read them from `scratchpad/ncm_license_check/src/ncm-c70274273f489abc73832b2b98584a8483b6b203/`; all 39 blobs there match `tree_pinned.json`. The 10 `configs/*.yaml` also match.
- Names: NC, Q, VQ, VAE, CM = models/next_concept.py, quant.py, vqvae.py, vae.py, common.py; ES, EM = scripts/eval_shard.py, eval_merge.py; GP = evaluation/gen_ppl.py; `cfg:x` = config/eval/x.yaml. Paper level k is 1-based; code level j=k-1 is 0-based.
- The pinned repo is inference and evaluation only (C5). Training numbers come from the paper alone.

## A. Generation (sampling)

**A1. The coarsest level (j=0, fewest codes) is generated first; the finest (L codes) last.** First-level size (cfg `manual_levels`, C1): 1 code (owt, tinystories, lm1b, wmt14, sudoku), 4 codes (xsum).
- Why: Eq. 4 factorizes p(r_1:K) as a product over levels, each conditioned on the cumulative reconstruction of the earlier levels (P§3.2). Level k quantizes the residual left by levels <k (Eq. 8), so the order is fixed by the tokenizer. P§3.3: nested dropout pushes information into the early levels. P§4.6: coarse steps are cheap because they process few tokens.
- Code: `generate` (NC:520-703) returns level-0 codes first; `Q.idx_to_zhats` (Q:635-668) decodes in that order.

**A2. Procedure** (all ℓ_k codes of a level sampled in parallel, Eq. 5; one forward pass per level):
1. Embed the prefix (A3). Get level-0 codes (A4).
2. For j=0..K-2: ẑ += Phi_j(Up_j(codebook_j[idx])); next input = `F.interpolate(ẑ, size=ℓ_{j+1}, mode="area")` (adaptive average pool). `Q.get_next_autoregressive_input` Q:697-709, called NC:665.
3. cfg:owt and cfg:lm1b-* only: zero-initialised residual Conv1d(k=3) per level on that input (`post_upsample_conv`, NC:183-195, 667-668). Not in the paper.
4. h = `transformer.wte`(input) + `transformer.wle`[level] (NC:670). wte is Linear(embed_dim→n_embd) (NC:131-133); wle is a learned level embedding (NC:197-200; `use_level_encoding: true` in every cfg).
5. Blocks run over [prefix_emb | h] (NC:501-518). Mask is block-diagonal (NC:266-292): the prefix sees only the prefix (bidirectional); a level sees the prefix and its own level only. Fig. 11d: attention stays inside the current block because its input already summarizes earlier levels. Pad prefix tokens are masked as keys (NC:418-425, 572-579).
6. RoPE positions: prefix 0..P-1; every level restarts at 0..ℓ_j-1 (NC:94-117). All cfg: `use_rope: true`; base in D. QK RMSNorm is applied before RoPE (CM:76-110).
7. Head j (`lm_heads[j]`, NC:518) on the level's slice; sample (A5). After the last level: `idx_to_zhats` → ẑ^(K) → VQ decoder logits → token pick → detokenize (ES:142-146).

**A3. Prefix token ids.** Paper (P§3.4): prompt tokens go through the frozen VQ-VAE encoder's token embedding table, then a projection to the hidden width.
- Code (NC:546-558): every cfg sets `raw_token_prefix: true`, so `z_prefix = vqvae.encoder.wte(prefix_tokens)[:, :P]` (NC:555; nn.Embedding(in_vocab_size, embed_dim), VAE:126; no position, no norm), then `prefix_emb = transformer.wte(z_prefix)` (NC:558).
- `contextual_prefix` (NC:551-553) calls `Encoder.forward_nopos` (VAE:181-195: wte + fixed sinusoidal PE base 10000 + conv stack + blocks + ln_e). No cfg sets it, and it needs encoder blocks (`enc_layers`>0, default 0): dead in shipped cfgs. If `raw_token_prefix` were false: `encode_z` (NC:557). `ctx_encoder` is unused (G).
- Unconditional prefix = one EOS id (ES:157; Table 4: null prefix). Conditional: ids[:P] padded with PAD (ES:94-96). CFG null prefix = [EOS, PAD, ...] (NC:534-535).

**A4. Level 0.** Paper Alg. 2 (P§3.4): ẑ^(0)=0; for k=1..K: resize ẑ^(k-1) to ℓ_k positions, project to hidden width, run one forward pass, sample all ℓ_k codes in parallel, update ẑ^(k). So the generator predicts the coarsest level from a zero input, K steps in all. The paper never mentions a unigram, learned queries or a prefix read-out.
- `query_level0` (NC:88-91, 239-242, 408-410, 581-627): level 0 gets its own target slots, with learned parameter `l0_queries` (1×ℓ_0×embed_dim, trunc-normal std sqrt(1/n_embd)) replacing ẑ^(0); head `lm_heads[0]`. Heads, `wle`, T are sized K, not K-1. `l0_autoregressive` (NC:89, 594-621) is an unused option.
- `predict_level0` (NC:249-264, 427-435, 629-641): no target slots. One prefix-only pass; final hidden states of the last ℓ_0 non-pad prefix positions go through `lm_head_l0` (Linear n_embd→V_0 without bias; tied head if `tie_heads_to_codebook`). Needs P≥ℓ_0.
- Neither flag: K-1 heads; level-0 codes must be passed as `idx` (NC:629-630 asserts otherwise). ES:156 draws them i.i.d. from `inference.level0_unigram` stored in generator.pt (`--l0` modes unigram|active|uniform|smooth:eps, ES:25, 70-78). Why: not stated.
- If `idx` is passed, both level-0 branches are skipped (NC:581, 629): lm1b cfgs set `query_level0: true` and `level0_unigram`, so eval uses the unigram.

| cfg | query_level0 | predict_level0 | level0_unigram | level-0 source at eval | K | transformer passes |
|---|---|---|---|---|---|---|
| xsum | true | false | none | learned queries | 9 | 9 |
| wmt14 | unset | true | none | prefix read-out | 11 | 11 |
| sudoku-* | true | unset | none | learned queries | 12 | 12 |
| owt | false | false | set | unigram | 32 | 31 |
| tinystories | false | false | set | unigram | 12 | 11 |
| lm1b-* | true | false | set | unigram | 12 | 11 |

**A5. Sampling.** `generate` defaults: temperature=1.0, top_k=50, top_p=0.9 (NC:521). ES fallbacks: top_k=1, top_p=0.0 (ES:35-36). `_sample_level_raw` (NC:487-499): temperature 0 → argmax; else logits/T, top-k mask (if >0), nucleus mask (always run), multinomial. Checked by hand: top_k=1 with top_p=0.0 is greedy; top_k=0 with top_p=1.0 filters nothing.
- cfg values (temperature/top_k/top_p): xsum and wmt14 1.0/1/0.0 (greedy); owt, tinystories, lm1b-* 1.0/0/1.0 (pure sampling); sudoku-* 0.0/0/1.0 (argmax).
- Paper states none of these. It mentions only greedy decoding at 8 NFE (XSum, P§F.2.2) and, in Alg. 2, parallel sampling of a level's codes.
- CFG (NC:477-485, 528-540): batch doubled [doc; null]; t = scale·j/(K-1); logits = (1+t)·cond − t·uncond; both branches get the same sampled codes. P§F.4: guidance combines conditional and unconditional logits per scale with weight w, following VAR. w=0.5 (ROUGE), 1.0 (BLEU peak); README:87: guided WMT14 uses `--cfg 1.0`. The linear ramp is code-only.

## B. Training objective

- Targets: all codes of all K levels, predicted in one forward pass, teacher-forced, with cross-entropy (Alg. 1; Fig. 4 caption). The tokenizer is trained first, then frozen while the generator trains (P§3).
- Eq. 10: L_NCM = −Σ_k w_k Σ_i log p_θ(r_k^i | z̃^(k-1)), w_k = ℓ_k^(−α) / Σ_j ℓ_j^(1−α). α=0: every code equal (ELBO generation term / N, P§A.2); α=1: every level equal. α in main runs: not stated. P§F.2.3 (TinyStories): α 0→1.5 raises the coarse-level loss share 19%→63%; Gen. PPL 18.9-19.0 for α≤1, 19.4 at 1.5.
- Corruption (Alg. 1 lines 2-3; P§F.3): some input codes are replaced by uniform draws from the same level's codebook, at a per-level rate; targets stay clean. Rates decrease linearly across levels (P§F.3; Fig. 13 caption: shifted linear schedule for nonzero rates); selected TinyStories mean rate 0.50 (P§F.2.1); per-level values not stated. Corrupted codes enter only through z̃^(k-1) (Alg. 1 line 7). Code: not implemented; `aux.code_corruption` (mode uniform, random_rate true, low 0.0, high 1.0, skip_level0 true) appears only in cfg:lm1b-* and no pinned .py reads it. UNVERIFIED: relation to the paper schedule.
- Label smoothing: not stated (paper, repo). Masking of PAD/EOS target tokens: not stated.
- Mask: block-diagonal (A2 step 5). Table 9 (TinyStories Gen. PPL): 15.41 vs 17.84 sliding-window vs 19.63 block-causal; code also has `block_causal` (NC:276).
- Heads (paper silent; NC:214-237). Tied (xsum, owt, tinystories), `_TiedCodebookHead` NC:9-34: logits = exp(log_inv_T)·(2 z·Eᵀ − ‖E‖²)/mean(‖E‖²), z = Linear(n_embd→d, no bias)(h), E = that level's frozen VQ codebook (non-persistent buffer), learned scalar `log_inv_T`, owt adds a learned per-code bias; one head per level. Shared (lm1b): one Linear(n_embd→8192, no bias) for all levels plus per-level bias vectors (NC:48-62, 217-222). Untied (wmt14: Linear per level, no bias; sudoku: Linear with bias).
- Tokenizer losses during generator training: none. The VQ-VAE is held in a tuple so it stays out of the module tree and state dict (NC:80-81). P§A.2: with tokenizer and decoder frozen, only the generation term depends on θ.
- Data path in code: `VQVAE.get_ncm_inputs` (VQ:120-124) returns inputs `Q.idx_to_ncm_input(idx)` (Q:670-694: cumulative ẑ after each level, area-resized to the next level's length, concatenated) and targets `idx[:, ℓ_0:]` (level 0 excluded). UNVERIFIED: how the absent trainer builds level-0 targets for query_level0 or predict_level0.
- Init (NC:303-305): every `c_proj.weight` normal(0, 0.02/sqrt(2·n_layer)). `_init_weights` (CM:219) is never called.

## C. VQ tokenizer

**C1. Architecture** (Table 5; VAE, VQ).
- Encoder: 0 blocks = token embedding wte (+ learned wpe unless `enc_positional: false`, lm1b) + dropout (VAE:102-179), so z_i depends on token i only until the norm. Then pre-quant `GroupNorm(groups, d, affine=False)` over (B,d,L) (VQ:59-67; groups 8, lm1b 4) and dropout `pre_quant_dropout` (xsum, wmt14 0.25, else 0.0). Paper silent on both.
- Decoder: `dec_layers` pre-LN blocks (1; lm1b 4): LayerNorm eps 1e-6, bidirectional SDPA attention, GELU MLP ×4, QK RMSNorm per head (`use_qk_norm: true` in all cfg), `output_layer` Linear(d→vocab, no bias), untied from wte (VAE:29-100, 198-258). Decoder positional embedding only if `dec_positional` (lm1b). cfg `n_layers` is the head count (6; sudoku 4), not depth (VAE:224).
- Levels (cfg `manual_levels`): owt k² for k=1..32; lm1b and tinystories [1,8,16,24,32,40,52,64,76,92,108,128]; wmt14 [1,4,8,14,20,26,32,40,48,56,64]; xsum [4,6,10,14,20,26,32,40,48]; sudoku [1,2,4,6,9,13,18,26,37,50,67,89]. All levels share one codebook size: owt 8192 (paper 4,096), lm1b 8192, sudoku 16, others 4096.

**C2. Quantizer** (`MultiscaleResidualVectorQuantizer`, Q:228-342, 448-549).
- Codebooks nn.Embedding(V_k, d), init N(0,1), `requires_grad False`, EMA only in training (Q:297-304, 485-500): ema_cs = decay·ema_cs + (1−decay)·counts; smoothed = (ema_cs+ε)/(n+Vε)·n; ema_w = decay·ema_w + (1−decay)·Σz; weight = ema_w/smoothed. decay 0.99 (tinystories 0.95), ε=1e-5, counts and sums all-reduced under DDP.
- Down_k and R_k (paper leaves open). Level 0 (`learned_l1_sampling: true`, all cfg): learned strided Conv1d down (init 1/stride on identity = area mean) and ConvTranspose1d up (init nearest-repeat), stride L/ℓ_0 (Q:145-162, 310-328). `learned_l1_cascade` splits it into factors ≤4: owt 4,4,4,4,4; tinystories 4,4,4,2; xsum 4,3; wmt14 4,4,4; sudoku 89 in one stage (computed from Q:165-178); GroupNorm(8, affine=False) between stages when `learned_l1_cascade_norm` (xsum, wmt14), also after the last when `..._norm_final`. lm1b: no cascade, one stride-128 conv. Levels 1..K-2: down = area interpolate, up = linear interpolate to L. Last level: no resampling. `learned_all_sampling` false everywhere.
- Every level's upsampled output passes Phi (`quant_resi`, Q:11-38, 380-386): (1−r)·h + r·Conv1d(d,d,k=3)(h), r=0.5 fixed, share_mode 0 (one Phi per level, chosen by nearest tick of j/(K-1)).
- Straight-through: in-loop at level 0 only (a code comment says higher levels diverge within about 100 steps, Q:503-505), plus ẑ = z + sg(ẑ − z) before the decoder (Q:531).

**C3. Objective.**
- Paper Eq. 9: L_tok = CE(x | ẑ^(k)) + β·MSE(z, sg[ẑ^(k)]) + MSE(sg[z], ẑ^(k)), k = nested-dropout cutoff. β value only in cfg `commitment_cost: 0.25`.
- Code (Q:464-531): at every level i, vq_loss += 0.25·mse(sg[ẑ_i], z) + mse(ẑ_i, sg[z]) with ẑ_i the cumulative reconstruction after level i; divide by K. Nested dropout (`fine_drop`): one Bernoulli(prob) per forward call; on a hit, keep K'~Uniform{min_keep..K-1} levels for the decoder input (never all K). prob 0.5, min_keep 1 (lm1b: prob 1.0, `dual`, w_full 0.7, w_trunc 0.3 per the Table 5 note; Q never reads `dual` or the weights: UNVERIFIED).
- The CE term is outside the repo; `VQVAE.forward` (VQ:91-95) returns decoder logits and vq_loss (plus utilizations).
- Table 1 targets (reconstruction %/utilization %): OWT 99.99/83.10; LM1B 99.86/89.31; TinyStories 99.98/90.12; XSum 99.79/86.20; WMT14 98.79/61.89; Sudoku 100/80.73, 83.85, 80.73 (easy, medium, hard).

**C4. Codes.** `VQVAE.encode` (VQ:97-101): eval-mode encoder → pre-quant norm → `Q.z_to_idx` (Q:573-632): per level down-sample the residual, nearest code, up-sample + Phi, subtract. Indices concatenated, level 0 first.

**C5. Training code in the pinned repo: none.** I grepped every pinned .py file for optimizer|AdamW|backward|clip_grad|scheduler|DataLoader|autocast|wandb: no training-loop hits (only `cross_entropy` in EM:131 for scoring). README's repository layout lists evaluate.py, scripts/, config/eval/, models/, tokenization/. No CE loss or batch builder exists.

## D. Training hyperparameters

Source: P Appendix C, Tables 5 (tokenizer) and 6 (generator). Common: AdamW, β1 0.9, gradient-clip norm 1, seed 42, BF16 mixed precision and torch.compile for all experiments (P§C). Columns: OWT, LM1B, TinyStories, XSum, WMT14, Sudoku.

| Tokenizer (Table 5) | OWT | LM1B | TS | XSum | WMT14 | Sudoku |
|---|---|---|---|---|---|---|
| updates | 78,432 | 25,957 | 30,000 | 120,001 | 60,000 | 10,000 |
| global batch (seqs) | 768 | 2,048 | 4,096 | 4,096 | 4,096 | 1,024 |
| peak LR / floor | 3e-4/3e-5 | 3e-4/3e-5 | 5e-4/5e-5 | 5e-4/5e-5 | 5e-4/5e-5 | 5e-4/5e-5 |
| warmup | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 500 |
| β2 | 0.95 | 0.99 | 0.95 | 0.95 | 0.95 | 0.99 |
| weight decay | 0.01 | 0.05 | 0.01 | 0.01 | 0.01 | 0.01 |
| EMA decay | 0.99 | 0.99 | 0.95 | 0.95 | 0.95 | 0.99 |
| dropout | 0.3 | 0.1 | 0.3 | 0.1 | 0.1 | 0.3 |
| nested dropout | 0.5 | 0.7/0.3 | 0.5 | 0.5 | 0.5 | 0.5 |
| params | 111.2M | 111.40M | 36.43M | 60.63M | 144.00M | 14.90M |

Tokens per update (derived) = global batch × L, prefix excluded: tokenizer 786,432/262,144/524,288/196,608/262,144/91,136; generator 262,144/65,536/262,144/24,576/131,072/91,136.

Tokenizer shape: text vocab 50,304/30,528/8,192/50,304/151,680/12; sequence length L 1,024/128/128/48/64/89; K 32/12/12/8/11/12; V_k 4,096/8,192/4,096/4,096/4,096/16; d 384 (Sudoku 256); decoder blocks 1 (LM1B 4); heads 6 (Sudoku 4).

| Generator (Table 6) | OWT | LM1B | TS | XSum | WMT14 | Sudoku |
|---|---|---|---|---|---|---|
| updates | 250,000 | 100,000 | 50,000 | 350,000 | 150,000 | 50,000 |
| global batch (seqs) | 256 | 512 | 2,048 | 512 | 2,048 | 1,024 |
| warmup | 500 | 1,000 | 500 | 500 | 500 | 1,000 |
| β2 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.99 |
| weight decay | 0.1 | 0.1 | 0.1 | 0.1 | 0.01 | 0.01 |
| dropout | 0.1 | 0.1 | 0.1 | 0.3 | 0.1 | 0.1 |
| prefix P / RoPE base | 1/1e5 | 1/1e4 | 1/1e5 | 512/1e5 | 96/1e5 | 91/1e4 |
| params | 94.65M | 97.07M | 88.61M | 88.02M | 119.96M | 25.46M |

Generator: peak LR 3e-4 and floor 3e-5 for all; 12 blocks, width 768, 12 heads (Sudoku 8/512/8); head dim 64.
- Token accounting (P§C.2): counts padded full-length positions; LM1B 25,957×2,048×128 + 100,000×512×128 ≈ 13.36B; OWT ≈ 127.22B (61.68B + 65.54B). Recomputed: both match. XSum/WMT14: Table 6 is the conditional stage; the generator is first trained on unpaired text, then on source-target pairs (P§C).
- Not stated: LR decay shape (only a floor; F.2.2 mentions a constant-LR recipe for one ablation), warmup shape, Adam ε, epochs, GPU count and time, weight EMA, unpaired-stage settings, α, corruption rates, data packing.

## E. Evaluation

**E1. Unconditional (owt, tinystories, lm1b).**
- Judge GPT-2-large (`eval.ref_model`; P§4.2). LM1B also under Mistral-7B, Llama-3.1-8B, Gemma-2-9B, Qwen-3-14B (P§E.1, Table 7).
- Samples: seeds [42,0,1,2,3] × `n_samples` 256 (cfg; P§E.1 and F.3: five sampling seeds, 256 samples each); `torch.manual_seed(seed)` per seed (ES:153); batch 8 (tinystories 128); fp32, no autocast. Report mean over seeds, std ddof=1 (EM:105-107).
- Text: `dec` (ES:130-134): cut at first EOS if `cut_at_eos` (default true, owt false), drop EOS/PAD, `decode(skip_special_tokens=True)`.
- `genppl` (owt, tinystories; EM:79-107): `generation_perplexity` with empty prompts (GP:19-148): judge tokenizer, truncation 1,024; NLL of every token except the first, pooled over all tokens of the seed's 256 samples: ppl = exp(ΣNLL/Σtokens); batch 32. Entropy = natural-log unigram entropy of `pre_tokenizer.encode(decoded text)` per sample, averaged (EM:90-101); P§4.2: mean per-sample unigram entropy.
- `genppl_flm` (lm1b; EM:110-151; README:87 says LM1B is scored under the FLM protocol): decode raw ids with the BERT tokenizer keeping [CLS]/[PAD]; judge tokenizer truncated to 128, pad=eos; ppl = exp(Σ NLL·valid / Σ valid), valid = non-EOS tokens plus the first EOS; batch 16. Entropy over raw ids including specials.
- Targets, Table 8 (Gen. PPL / H): TinyStories 15.41/3.98 (12 NFE), LM1B 93.05/4.31 (12), OWT 83.12/5.42 (32).

**E2. Conditional.**
- XSum ROUGE-1/2/L: `rouge_score`, use_stemmer=True, mean F-measure ×100 over 11,334 test docs (EM:35-55). Target (Table 2): 35.0/12.8/28.0 at 8 NFE.
- WMT14: `sacrebleu.corpus_bleu(hyps,[refs])` defaults, plus lowercase BLEU, chrF, length ratio; 3,003 docs (EM:24-32). Target 22.6 at 11 NFE.
- Sudoku: solved % = decoded string equals solution; also cell accuracy and valid-grid % (EM:58-76); 2,000 boards per level; clues 40/35/30 (P§4.4); targets 99.5/96.5/86.1 at 12 NFE.
- NFE counts generator calls; each classifier-free-guidance step counts as two (P§4.3). MBR (P§F.4): K candidates, keep the one most similar to the rest; K=16 XSum (128 NFE), K=12 WMT14 (132 NFE). Similarity function and sampling settings: not stated; no code.
- Timing (P§4.6): wall-clock for 1,024 sequences at batch size 16 on one H100 SXM 80GB. No timing code in repo.

## F. Datasets (P Table 4; sizes and licences not stated in the paper)

| Dataset (paper ref) | Text tokenizer | Context | L | Eval size (cfg `n_docs`, `n_samples`) |
|---|---|---|---|---|
| OpenWebText [16] | GPT-2 | null | 1,024 | 5×256 samples |
| LM1B [17] | bert-base-uncased | null | 128 | 5×256 |
| TinyStories [40] | custom BPE, 8,192 | null | 128 | 5×256 |
| XSum [44] | GPT-2 | article 512 | 48 | 11,334 test docs |
| WMT14 De-En [41] | Qwen3-0.6B | German 96 | 64 | 3,003 |
| Sudoku [50] | 12 symbols (FMLM+ [49]) | puzzle 91 | 89 | 2,000 per difficulty |

- Train and validation sizes and splits: not stated (only mentions of held-out text, Table 1, and the full XSum test set, Table 10).
- Licences: no dataset licence stated. Code is MIT (README, LICENSE). Tokenized data and checkpoints are on HF `hazyresearch/ncm-datasets` and `ncm-checkpoints` (README:38-54); licence not stated.

## G. Initialization from a pretrained LM

- No. P§1 claims quality in K ≪ L steps without distillation. P§F.2.2 tried a frozen pretrained T5-small prefix encoder (+35.3M parameters, slightly higher ROUGE) and rejected it: "We therefore use the encoder from our VQ-VAE in all reported experiments."
- Qwen3-0.6B, GPT-2, bert-base-uncased serve only as tokenizers (Table 4); GPT-2-large and others only as evaluation judges.
- Code has unused options: `ctx_encoder.pretrained` (T5EncoderModel.from_pretrained), `init_from_gpt2`, `init_from_vqvae` (NC:135-181); all cfg have `ctx_encoder.enabled: false` or no key.

## H. Contradictions and silent deviations (paper vs pinned code)

1. First level and NFE. Paper Alg. 2: ẑ^(0)=0 and the generator predicts level 1; Table 8 NFE 12/12/32. Code: unconditional evals draw level 0 from a stored unigram (ES:156, NC:88-91), so 11/11/31 transformer passes (cfg:tinystories, lm1b-*, owt).
2. Token pick. Paper Alg. 2 line 11: output tokens by argmax. Code ES:99-124 `pick_tokens`: if `frequency_penalty`>0 or `adjacency_penalty` is set, sequential greedy pick with a count penalty against `inference.token_unigram`×T, a decaying bonus below that allowance, and a penalty on the previous 4 tokens. cfg:xsum 2.0/deficit 1.0/5,4,0.5; cfg:lm1b-* 11.5/2.0/10,4,0.5. Undisclosed; README says cfgs equal the shipped ones (README:93).
3. Nested dropout and VQ loss. Paper (P§3.3, Eq. 9): both the reconstruction CE and the VQ terms are evaluated at the sampled cutoff. Code Q:519-530: VQ terms at every level, averaged over K, independent of the cutoff; the cutoff only picks the decoder input, drawn once per call, never all K.
4. XSum levels. Table 5 K=8, Table 2 NFE 8 (MBR 16×8=128) vs cfg:xsum 9 levels (xsum.yaml:94-102) and 9 passes.
5. OWT size. Tables 5-6: V_k 4,096, 12 blocks, width 768, 12 heads, 94.65M vs owt.yaml:7-11 (16/1024/16) and owt.yaml:85-117 (8192 codes). UNVERIFIED which produced the paper's OWT results.
6. EMA decay. Table 5 XSum and WMT14 0.95 vs xsum.yaml:75 and wmt14.yaml:75 0.99.
7. Sudoku-easy dropout. Table 6 0.1 vs sudoku-easy.yaml:12 0.2 (medium, hard 0.1).
8. Gen. PPL protocol. Paper describes one; EM has two (`genppl`, `genppl_flm`) with different truncation, pooling and entropy inputs.
9. Paper omits: pre-quant GroupNorm and dropout, QK-norm, Phi conv, learned level-0 samplers and cascade, `post_upsample_conv`, tied or shared heads, level-0 unigram, CFG ramp.
10. Schedules. P§F.1.2 lists a cosine schedule; utils/registry.py:15-23 has linear, cubic, exp, sqrt, sine, log, quadratic.

## Not found (place searched)

- Generator and tokenizer training code, data packing: all pinned .py files, README, grep for training symbols.
- LR decay and warmup shape, Adam ε, weight EMA, GPU count and time, XSum/WMT14 unpaired-stage settings: P Tables 5-6, §C; grep "cosine|schedule|decay".
- α, per-level corruption rates, label smoothing, PAD/EOS loss masking: P§3.4, §F.2.3, §F.3; code.
- Temperature, top-k, top-p in the paper; MBR similarity and sampling settings: P§4, §F.4; no MBR code.
- Dataset sizes, splits, licences: P Table 4, App. B; README.
- How `level0_unigram` and `token_unigram` are computed (ES:59-71 only loads them); meaning of `dual`, `w_full`, `w_trunc` and `code_corruption` (unread by pinned .py).
- OWT conditional prompts (Fig. 8) vs Table 6 P=1: no config or table.
- Timing script for the 21×/31× claims; codebook-perplexity code for Fig. 16 (Q:548 returns a constant 0.1).
