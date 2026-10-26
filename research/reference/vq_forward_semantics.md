# VQ tokenizer (OWT): forward pass and training losses

## 0. Header

- Reference: HazyResearch/ncm at commit c70274273f489abc73832b2b98584a8483b6b203 (read-only copy).
- `R/` is that copy's root: `/private/tmp/claude-501/-Users-jessicabruce--claude-spares-spare-e7phlw/19e02558-926e-417a-8cc2-e33c5915494f/scratchpad/ncm_license_check/src/ncm-c70274273f489abc73832b2b98584a8483b6b203/`. Code citations are `path:line` relative to `R/`.
- `paper:N` is line N of `/private/tmp/claude-501/-Users-jessicabruce--claude-spares-spare-e7phlw/19e02558-926e-417a-8cc2-e33c5915494f/scratchpad/ncm_2610_14162.txt` (a text extraction, so these are not PDF page numbers).
- Date: 2026-10-26.
- Read in full: `models/vqvae.py` (1-165), `models/vae.py` (1-272), `models/quant.py` (1-737), `config/eval/owt.yaml` (1-193), `utils/registry.py` (1-23).
- Read in part: `tokenization/vqvae_tokenizer.py` (1-30, plus grep), `evaluate.py` (1-25), `scripts/eval_shard.py` (27-31), `models/next_concept.py` (181-197), paper lines 312-450, 1196-1215, 1236-1430, 1470-1490, 1850-1862, and grep hits over the whole paper text.
- Not read: `utils/misc.py` (imported at `models/vqvae.py:8`, used only to print parameter counts), `models/common.py` (not imported by the VQ path), and the rest of `models/next_concept.py` (generator).
- Method: the Grep and Glob tools were not available in this session, so grep, find, sed, awk and cat were run through Bash. No Python and no reference code was executed. After writing, every cited line was printed from its file and checked against the claim.

## 0.1 Where the brief and the code differ

1. quant_resi is not zero-initialized for OWT. `Phi` (`models/quant.py:11-38`) keeps PyTorch's default Conv1d init. With ratio r = 0.5 its output is 0.5·h + 0.5·Conv(h) even at initialization (`quant.py:36-38`).
2. The zero-initialized Conv1d(k=3) in the brief is the generator's `post_upsample_conv` (`models/next_concept.py:189-195`, enabled by `config/eval/owt.yaml:61-63`), not a tokenizer module. The tokenizer's zero-init variants (`quant.py:52-53`, `quant.py:207-213`) are inactive for OWT (§2.7).
3. Eq. (9) evaluates the VQ terms only at the sampled nested-dropout cutoff (`paper:390-392`). The code evaluates them at all 32 levels and divides by 32 (`quant.py:515-524`).
4. The reconstruction cross-entropy is not in the pinned tree, which has no tokenizer training loop (§6, §10).
5. Codebook size: `owt.yaml` sets 8192 per level (`owt.yaml:86-117`). Paper Table 5, OWT column, gives 4,096 (`paper:1296-1302`).
6. Paper Algorithm 1 (`paper:407-423`) is the generator's training step, not the tokenizer's. The paper has no tokenizer algorithm box.
7. The paper gives no numeric value for β (`paper:394-397`; Table 5 has no β row). `owt.yaml:123` sets 0.25.

## 1. Config resolution

- `tokenization/vqvae_tokenizer.py:17-18` calls `VQVAE(**config.vqvae)` and then `load_state_dict(..., strict=True)`. Every key under `tokenizer.vqvae` (`owt.yaml:75-174`) is a constructor kwarg. `checkpoint_path` (`owt.yaml:118`) is absorbed by `**kwargs` (`models/vqvae.py:32`, `:35`).
- The YAML is loaded alone with `OmegaConf.load` (`evaluate.py:19`, `scripts/eval_shard.py:29`). No defaults file is merged.
- `vector_quantizer_config.clss` resolves via `utils/registry.py:11-12` to `models.quant.MultiscaleResidualVectorQuantizer` (`models/vqvae.py:50-51`; `from_config` at `quant.py:224-226`).
- Derived OWT shapes: L = 1024 (`owt.yaml:79`), C = d = 384 (`owt.yaml:77`), in_vocab = 50304 (`owt.yaml:78`), n_layers = 6 (`owt.yaml:76`, used as the decoder head count, `models/vae.py:224-226`). There are 32 levels with ℓ_k = (k+1)² for k = 0..31 (`owt.yaml:139-171` lists 1, 4, ..., 1024). Each level has V = 8192 codes (`owt.yaml:86-117`). Σ_k ℓ_k = 11,440 codes per 1024-token sequence (derived: Σ_{j=1}^{32} j² = 11,440).

## 2. Forward pass (training mode)

Entry point: `models/vqvae.py:91-95` (`VQVAE.forward`) calls `_encode_pre_quant` (`:80-89`), then `self.vq(z)` (`:93`), then `self.decoder(z_q)` (`:94`). It returns logits, `vq_loss`, per-level utilization, global utilization, and `ppl` (`:95`).

### 2.1 Encoder (`models/vae.py:102-179`, OWT instance)

- Token embedding `wte = Embedding(50304, 384)` (`vae.py:126`, lookup at `:167`).
- Positional embedding `wpe = Embedding(1024, 384)` (`vae.py:127`, added at `:168-170`). Enabled by `enc_positional=True` (`vqvae.py:30`, passed at `:47`).
- Dropout p = 0.3 (`vae.py:128`) applied to `wte(x) + wpe(0..T-1)` (`vae.py:171`).
- No blocks: `enc_layers=0` (`vqvae.py:24`; loop empty at `vae.py:130-136`). No conv blocks (`vqvae.py:26`; `vae.py:138-144`). `ln_e` is None (`vae.py:145`). No compression: `compression_factor=1` (`owt.yaml:80`) skips `down` (`vae.py:148-157`).
- Output: `z_enc = Dropout_0.3(wte(x) + wpe(t))`, shape B×1024×384 (`vae.py:171-179`).

### 2.2 Pre-quantization norm and dropout (`models/vqvae.py:59-71`, `:80-89`)

- `z = encoder(x)` (`vqvae.py:81`).
- GroupNorm with 8 groups over 384 channels, `affine=False`, applied to the transposed B×384×1024 tensor (`vqvae.py:66`, `:82`, `:86`). Each group has 48 channels. Statistics are per sample and per group, over all 1024 positions (PyTorch GroupNorm semantics; external). `affine=False` is deliberate, per the comment at `vqvae.py:59`.
- `pre_quant_dropout = 0.0` is falsy, so `pre_quant_drop` is None (`vqvae.py:69-71`, `:87-88`). There is no pre-quant dropout.
- Order matters: the encoder dropout (`vae.py:171`) runs before GroupNorm, so training-mode normalization statistics include dropped activations.
- The quantizer receives z as float32, B×1024×384 (`quant.py:449-450`).

### 2.3 Quantizer setup (`models/quant.py:229-342`)

- Levels come from the manual list (`quant.py:285-286`), with a length check against the 32 codebook sizes (`:290-292`). ℓ_0 = 1 and ℓ_31 = 1024 (`owt.yaml:140`, `:171`).
- Codebooks: 32 × `Embedding(8192, 384)` (`quant.py:294-296`), initialized N(0, 1) (`:299`), `requires_grad_(False)` (`:298-300`).
- EMA buffers per level: `ema_cluster_size_i` (zeros), `ema_w_i` (copy of the initial codebook), `codebook_hit_i` (zeros) (`:301-304`).
- `quant_resi` is a `PhiNonShared` holding 32 `Phi` modules (`:306`; built by `_build_quant_resi`, `:108-143`).
- The learned level-0 sampler and cascade are built at `:311-328`. The level-wise refine convs (`:330-342`) are inactive because `learned_all_sampling: false` (`owt.yaml:127`).

### 2.4 Multiscale residual loop (`models/quant.py:448-549`)

Notation: r^(i) = residual entering level i; ẑ^(i) = cumulative reconstruction after level i; sg = stop-gradient (`.detach()`).

Initialization: r^(0) = sg(z) (`quant.py:451`, `:453`); ẑ^(-1) = 0 (`:454`).

| Step for level i = 0..31 (ℓ_i = (i+1)²) | Code | Shape |
|---|---|---|
| u_i = D_i(r^(i)): downsample residual to ℓ_i positions | `:472-476`; `_down_for_quant` at `:395-402`; i = 31 uses r^(31) unchanged | B×ℓ_i×384 |
| idx_i = argmin_k d(u_i, e_{i,k}) | `:478-480`; distance `:388-393` | B×ℓ_i integers in [0, 8192) |
| q_i = e_i[idx_i] (STE at i = 0 only) | `:502`; STE `:503-505` | B×ℓ_i×384 |
| Q_i = Phi_i(U_i(q_i)): upsample to 1024, then residual conv | `:507-512`; `_up_to_full` `:404-411`; `_apply_quant_resi` `:380-386` | B×1024×384 |
| ẑ^(i) = ẑ^(i-1) + Q_i | `:514` | B×1024×384 |
| r^(i+1) = r^(i) − Q_i | `:517` | B×1024×384 |

The EMA codebook update for level i runs at `:485-500`, before q_i is read at `:502` (§5).

After the loop, `z_dec = z + sg(ẑ_sel − z)` (`:531`), where ẑ_sel is the full ẑ^(31) or a truncated ẑ^(K) (§3). It is passed to the decoder at `models/vqvae.py:94`.

Per-stage details:

- Downsampling `D_i` (`_down_for_quant`, `quant.py:395-402`). For i = 0 with learned sampling, `learned_down_l1` maps 1024 positions to 1 (`:397-398`). For 1 ≤ i ≤ 30, `F.interpolate(mode="area")` maps the residual to ℓ_i positions (`:399`; PyTorch adaptive average pooling, external). For i = 31 there is no downsampling (`:474-475`).
- Distance (`quant.py:388-393`): d(u, e) = ‖u‖² + ‖e‖² − 2·uᵀe, computed by matmul. Its argmin equals the nearest squared-Euclidean code (the paper's Eq. 1/7 form, `paper:317-323`, `paper:371-373`). Tie handling is torch.argmin's (external).
- Upsampling `U_i` (`_up_to_full`, `quant.py:404-411`). For i = 0, `learned_up_l1` maps length 1 to 1024 (`:406-407`). For 1 ≤ i ≤ 30, `F.interpolate(mode="linear", size=1024)` (`:408`). `align_corners` is not passed, so PyTorch's default applies (external). For i = 31 it is the identity (`:509-510`).

### 2.5 Straight-through estimators (two of them)

- Level 0, only when learned sampling is on (`quant.py:503-505`): q_0 ← u_0 + sg(e_0[idx_0] − u_0). The forward value is the codebook vector. The gradient reaches u_0, and through it `learned_down_l1`. The comment at `:503` says the STE is kept at level 0 only because it "diverges within ~100 steps" at higher levels. That is the authors' claim, not checkable here.
- Decoder STE (`quant.py:531`): z_dec = z + sg(ẑ_sel − z). The forward value is ẑ_sel, and the gradient to z is the identity. Decoder gradients therefore reach the encoder only, never Phi, the learned samplers, or the codebooks. This is the reconstruction-to-encoder STE described at `paper:398-399`.

### 2.6 Codebook read order

- The indices are chosen with the codebook before this step's EMA write (`quant.py:478-480`). The vectors are read after it (`quant.py:498` writes, `:502` reads). So the quantized vectors come from the updated codebook, while the indices come from the old one.

### 2.7 quant_resi: `Phi` (`models/quant.py:11-38`, `:380-386`, `:108-143`)

- `Phi` subclasses `nn.Conv1d(384 → 384, kernel 3, stride 1, padding 1)` (`quant.py:11-20`). Its `__init__` does not zero the weights, so PyTorch's default Conv1d init applies (external).
- Forward: out = (1 − r)·h + r·Conv(h), with r = 0.5 (`quant.py:36-38`). r comes from `owt.yaml:130` through `_build_quant_resi` (`quant.py:112`, `:130`, `:137`).
- Layout: `_apply_quant_resi` transposes B×L×C to B×C×L, applies module `at = level_idx/31` (`quant.py:384-385`), and transposes back (`:383`, `:386`). `PhiNonShared` picks the nearest tick (`quant.py:89`, `:91-92`). For 32 levels, level i picks module i (derived: the nearest tick to i/31 is (2i+1)/64).
- `arch` defaults to `'blend'` (`quant.py:115`), so `PhiConvStack` is not built. That class zero-inits its last conv (`quant.py:52-53`) but is inactive here. `share_mode: 0` gives one module per level (`quant.py:139-140`).
- `_make_zeroinit_residual_conv` (`quant.py:207-213`) is used only under `learned_all_sampling` (`quant.py:330-339`), which is off.
- Phi is applied to every level, including level 0 (`quant.py:512`).

### 2.8 Decoder (`models/vae.py:198-258`, OWT instance)

- No upsampling: `compression_factor=1` skips `up` (`vae.py:236-245`). No decoder positional embedding: `dec_positional` defaults to False (`vqvae.py:31`; `vae.py:219`). No extra blocks: `dec_layers=1` gives zero extra blocks (`vqvae.py:28`; `vae.py:229-231`).
- `middle_block` = `Block` with 384 channels, 6 heads, bias True, dropout 0.3, and QK-norm (`vae.py:224-228`).
- Block (`vae.py:92-100`): pre-LN residual. x ← x + Attn(LN_1(x)); x ← x + MLP(LN_2(x)).
- LayerNorm (`vae.py:6-13`) uses eps = 1e-6. Block LNs have bias (`vae.py:92`, `:94`, bias from `:226`). The final `ln_f` has no bias (`vae.py:234`, default `bias=False` at `:7`).
- Attention (`vae.py:29-70`): `c_attn` is Linear(384 → 1152, bias). Heads are 6 × 64 (`vae.py:55-58`). Q and K get per-head RMSNorm with eps 1e-6 (`vae.py:40-44`, `:60-62`, `RMSNorm` at `:16-26`). `attn_window` is 0, so there is no mask (`vae.py:46`, `:64`). SDPA gets `dropout_p = 0.3` in training (`vae.py:65-67`) and no `is_causal` argument, so attention is bidirectional (PyTorch default; external). Output goes through `c_proj` and residual dropout (`vae.py:68-69`).
- MLP (`vae.py:73-86`): Linear 384 → 1536, `nn.GELU()` (exact by default; external), Linear 1536 → 384, dropout 0.3 (`vae.py:76-79`, `:81-85`).
- Output: `ln_f` (`vae.py:256`), then `output_layer = Linear(384 → 50304, bias=False)` (`vae.py:233`, `:257`). Logits are B×1024×50304 (`vqvae.py:94-95`).
- Unused: `Decoder.drop` (`vae.py:220`, never called in `:248-258`), `SelfAttention.attn_dropout` (`vae.py:35`), and the first `ln_f` assignment (`vae.py:221`, overwritten at `:234`).

## 3. Nested dropout (`models/quant.py:429-431`, `:457-462`, `:514-531`)

- Probability: `fine_drop.prob = 0.5` (`owt.yaml:135` → `quant.py:264`). It is constant, because `prob_start` is None and `anneal_steps` is 0 (`quant.py:266-268`, `:430-431`).
- Draw: once per forward call, not per sample. `do_fine_drop = training AND p > 0 AND U < p` with one scalar U (`quant.py:460-461`). Eval never truncates (`:460`). The same cutoff applies to every row and every position of the batch.
- Cutoff (only if drawn and `len(z_hat_keep) ≥ 2`, `:525`):
  - kmin = max(min_keep − 1, 0) = 0 (`:526`), with min_keep = 1 from `owt.yaml:136` via `quant.py:265`.
  - kmax = len(z_hat_keep) − 2 = 30 (`:527`). The list has 32 entries, appended once per level (`:515-516`).
  - K ~ uniform on {0, ..., 30} (`:528-529`).
  - z_hat = z_hat_keep[K] = Σ_{i ≤ K} Q_i (`:530`).
- Result: when the draw fires, the decoder keeps levels 0..K, which is 1 to 31 levels. The finest level (index 31) is never kept in a dropped draw, because kmax excludes the full stack.
- Distribution (derived): P(full stack) = 0.5, and P(K = j) = 0.5/31 for j = 0..30.
- Removed levels: levels K+1..31 are dropped from the decoder input by leaving them out of the cumulative sum. They are not multiplied by zero. Their indices are still computed (`:478-480`), and their VQ terms still count (`:519-520` run before the cutoff at `:525`).
- The step counter `_fd_step` (`:269`, `:458-459`) matters only for annealing, which is off.

## 4. Learned level-0 sampler and cascade (`models/quant.py:145-204`, `:311-328`, `:397-398`, `:406-407`, `:504-505`)

- Switches: `learned_l1_sampling: true` (`owt.yaml:124` → `quant.py:271`). `learned_l1_cascade: true` (`owt.yaml:125` → `:272`). `learned_l1_cascade_base: 4` (`owt.yaml:126` → `:273`). The inter-stage and final norms are off (`owt.yaml:172-173` → `:274-275`; gated at `:189-193`).
- Stride: total = L // ℓ_0 = 1024 (`quant.py:312-314`). Divisibility is asserted (`:315-316`).
- Stage plan, `_plan_cascade_strides(1024, base=4)` (`:165-178`): repeated division by 4 gives 1024 → 256 → 64 → 16 → 4 → 1, so the strides are [4, 4, 4, 4, 4].
- Down (`_make_cascaded_down_l1`, `:185-194`, using `:145-152`): five Conv1d(384, 384, kernel 4, stride 4, bias). Each weight is identity/4 and each bias is 0 (`:148-151`). The input is B×384×1024 and the output is B×384×1 (`:397-398`).
- Up (`_make_cascaded_up_l1`, `:197-204`, using `:155-162`): five ConvTranspose1d(384, 384, kernel 4, stride 4, bias). Weights are identity (1.0) and biases are 0 (`:158-161`). Output is B×384×1024 (`:406-407`).
- Initialization: the composed down is a mean over all 1024 positions (area pool to one position). The composed up is nearest-neighbour repeat. The comment at `quant.py:310` says the same (derived from the identity/4 init).
- All ten convs are trainable (default `requires_grad=True`).
- What the sampler outputs: one 384-vector per sequence, which is the level-0 input u_0 (ℓ_0 = 1). The up-sampler expands the level-0 code vector to 1024 positions.
- What it does not do: it does not set the number of positions per level. ℓ_k comes only from the manual list (`quant.py:285-286`, `owt.yaml:139-171`). The sampler maps L to ℓ_0, and the cascade plan depends only on the total stride 1024 and base 4.
- Gradient sources: D_0 and U_0 get gradient only through the VQ term at `quant.py:520` (via Q_0, and via the level-0 STE at `:504-505` for D_0). The commitment term detaches ẑ (`:519`), and the decoder path is detached (`:531`). So the decoder's reconstruction loss never trains D_0 or U_0.
- Levels 1..31 have no learned resampling.

## 5. Codebook EMA (`models/quant.py:294-304`, `:413-427`, `:436-446`, `:485-500`)

Per level i, training forward only (`quant.py:485`):

- n_k = number of rows assigned to code k in this batch (`:489`, `encodings.sum(0)`).
- s_k = Σ of the detached inputs u assigned to code k (`:490`, `encodings.t() @ u`).
- All-reduce of n and s only if `torch.distributed` is initialized (`:491-493`). It is a no-op on one device.
- γ = decay = 0.99 (`owt.yaml:121` → `quant.py:252-261`). `_get_ema_decay_level` returns the same value (`:424-427`). `ema_scale_by_world_size: false` (`owt.yaml:174`) means no rescaling (`:416-422`).
- N_k ← γ·N_k + (1 − γ)·n_k (`:494`).
- n = Σ_k N_k (`:495`).
- Laplace smoothing: N̂_k = (N_k + ε) / (n + V·ε) · n, with ε = 1e-5 (`owt.yaml:122` → `quant.py:254`, `:236`) and V = 8192 (`:496`). Σ_k N̂_k = n.
- m_k ← γ·m_k + (1 − γ)·s_k (`:497`).
- e_k ← m_k / N̂_k (`:498`). This writes the embedding weight data.
- If `restart_threshold > 0`, dead-code restart runs (`:499-500`).

Initialization:

- e_k^(0) ~ N(0, I) (`quant.py:299`).
- N_k^(0) = 0 (`:302`).
- m_k^(0) = e_k^(0) (`:303`).
- codebook_hit = 0 (`:304`).
- k-means init is off (`quant.py:246`; `:280-283`, `:345-378` inactive).

requires_grad: only the codebook embeddings are False (`quant.py:298-300`). The encoder, decoder, Phi and the samplers remain trainable. The EMA and hit buffers are persistent (`:302-304`, no `persistent=False`), so they are part of the state dict.

Derived consequences (from `:494-498`, not run):

- A code never selected has N_k = 0, so N̂_k ≈ ε·n/(n + Vε) ≈ ε whenever n ≫ Vε. Then e_k ≈ γ^t · m_k^(0) / ε ≈ 10^5 · 0.99^t · m_k^(0). Such a code sits about 10^5 times farther out than the data scale, and it returns to data scale only after order 10^3 steps. It is dormant, not permanently dead.
- For a code with a steady count n_k per step and mean x̄, e_k ≈ x̄ + γ^t · m_k^(0) / (n_k (1 − γ^t)). The random initial vector keeps weight γ^t / (n_k (1 − γ^t)). When n_k is near 1, which happens at levels where B·ℓ_i is comparable to V, the initial vector dominates for on the order of 10^2 steps. The batch size is not in the eval config.

Dead-code restart is off for OWT. `restart_threshold` defaults to 0.0 (`quant.py:244`, stored at `:276`), and the gate is `> 0` (`:499`). `owt.yaml` does not set it (`owt.yaml:119-174`). When it is on, `_restart_dead_codes` (`:436-446`) replaces codes with N < threshold by random inputs from the batch and resets their count to the threshold.

`codebook_hit` counters are never reset (`:522`, `:534-541`). The per-level utilization returned is therefore cumulative since the buffer was created (derived from the absence of any reset in the tree).

## 6. Loss terms (`models/quant.py:464`, `:514-524`; `models/vqvae.py:93`)

Per level i, after ẑ^(i) is formed (`quant.py:514`):

- Commitment term: L_c,i = β · MSE(sg(ẑ^(i)), z), with β = `commitment_cost` = 0.25 (`owt.yaml:123` → `quant.py:234`, `:253`). Computed at `quant.py:519`. The gradient reaches the encoder only.
- Codebook (quantization) term: L_q,i = MSE(ẑ^(i), sg(z)), weight 1 (`quant.py:520`; sg(z) = `z_no_grad`, `:451`). The gradient reaches Phi_j for j ≤ i, U_0 and D_0 (through the level-0 STE). It never reaches the codebook (`quant.py:300`) or the encoder.
- Summation: `vq_loss` starts at 0 (`:464`) and accumulates L_c,i + L_q,i for i = 0..31 (`:519-520`). It is then divided by `num_levels` = 32 (`:524`). Each MSE is the default mean over all B·1024·384 elements (`:519-520`; PyTorch default, external).
- The value is returned as element 2 of the tuple (`quant.py:543-549`, returned to `vqvae.py:93`). It is not added to any cross-entropy in this tree.
- Reconstruction loss at the decoder: NOT IN THIS TREE. The decoder returns logits, B×1024×50304 (`vqvae.py:94-95`, `vae.py:257`). No loss over them exists in the reference. The only cross-entropy is `scripts/eval_merge.py:131`, which scores generation perplexity. The paper's form is token cross-entropy averaged over the L positions, evaluated at the sampled cutoff (`paper:392`, with `paper:390-391`). The labels are most plausibly the input token ids, from `tokenization/vqvae_tokenizer.py:21-28` and `:69-70` (padding with the pad id at `:25`). This is inference, not established by code (§10).

Gradient routing summary: commitment (`:519`) reaches z only. The codebook term (`:520`) reaches Phi, U_0 and D_0. The decoder path reaches z only (`:531`). So no decoder gradient reaches Phi or the samplers.

## 7. Defaults the OWT config does not set

| Parameter (default, file:line) | Effect for OWT |
|---|---|
| `enc_layers = 0` (`vqvae.py:24`) | No encoder blocks. Matches paper Table 5 "Encoder blocks 0" (`paper:1310-1316`). |
| `enc_attn_window = 0`, `enc_conv_layers = 0`, `enc_conv_kernel = 3` (`vqvae.py:25-27`) | No encoder attention or conv. |
| `dec_layers = 1` (`vqvae.py:28`) | One decoder block. Matches `paper:1317-1323`. |
| `pre_quant_norm_type = 'group'` (`vqvae.py:29`) | GroupNorm (`vqvae.py:62-67`). |
| `enc_positional = True` (`vqvae.py:30`) | Encoder wpe on. |
| `dec_positional = False` (`vqvae.py:31`) | Decoder wpe off (`vae.py:219`). |
| `pre_quant_dropout = None` (`vqvae.py:22`) | Owt sets 0.0, same result. |
| `compression_factor = 4` (`vqvae.py:19`) | Owt overrides to 1 (`owt.yaml:80`). |
| `dropout = 0.1` (`vqvae.py:20`); `use_qk_norm = False` (`vqvae.py:23`) | Owt overrides to 0.3 and true. |
| `arch = 'blend'` (`quant.py:115`); `hidden_mult = 1` (`:116`); `kernel_size = 3` (`:117`) | Phi is built (`:132-137`). Phi's kernel is hard-coded to 3 (`:13`). |
| `share_mode`, `learnable_ratio` (`quant.py:113-114`) | Owt sets 0 and false. |
| `restart_threshold = 0.0` (`quant.py:244`) | Restart off (`:499`). |
| `kmeans_init = False`, `kmeans_iters = 10` (`quant.py:246-247`) | Off. |
| `fine_drop.prob = 0.0`, `min_keep = 2` (`quant.py:264-265`) | Owt sets 0.5 and 1. |
| `prob_start = None`, `anneal_steps = 0` (`quant.py:266-268`) | No annealing. |
| `learned_all_sampling = False` (`quant.py:239`) | Owt sets false. |
| `learned_l1_sampling = False` (`quant.py:238`); `learned_l1_cascade = False` (`:240`); `learned_l1_cascade_base = 4` (`:241`) | Owt sets true, true, 4. |
| `learned_l1_cascade_norm`, `_norm_final = False` (`quant.py:242-243`) | Owt sets false. |
| `ema_scale_by_world_size = False` (`quant.py:245`) | Owt sets false. |
| `decay = 0.99` (`quant.py:235`); `epsilon = 1e-5` (`:236`); `commitment_cost = 0.25` (`:234`) | Equal to the owt values. |
| `pre_quant_norm_type` (`quant.py:248`) | Accepted by `MultiscaleResidualVectorQuantizer` but unused in its body. |
| `Encoder` kernel_size 8, stride 2, padding 3 (`vae.py:110-113`) | Unused; not passed by `VQVAE`. |
| `LayerNorm` bias = False (`vae.py:7`) | Applies to `Decoder.ln_f` (`vae.py:234`). |
| `attn_window = 0` (`vae.py:46`) | Full (bidirectional) attention. |

## 8. Code versus paper

Paper locations: §3.3 is `paper:364-399`. Eq. (9) is at `paper:392-393`. Algorithm 1 is at `paper:407-423`. The nested-dropout text is at `paper:383-391`.

Matches:

- Commitment term: β times MSE with stop-gradient on the quantized side (`paper:392` vs `quant.py:519`).
- Codebook term: MSE with stop-gradient on z (`paper:392` vs `quant.py:520`).
- Decoder sees the truncated reconstruction at the cutoff (`quant.py:525-531` vs `paper:390-392`).
- Straight-through to the encoder (`paper:398-399` vs `quant.py:531`).
- Table 5 OWT column vs `owt.yaml`: text vocab 50,304 (`paper:1275-1281` vs `owt.yaml:78`); L = 1,024 (`paper:1282-1288` vs `owt.yaml:79`); K = 32 (`paper:1289-1295`; 32 entries at `owt.yaml:139-171`); d = 384 (`paper:1303-1309` vs `owt.yaml:77`); encoder blocks 0 (`paper:1310-1316`); decoder blocks 1 (`paper:1317-1323`); decoder heads 6 (`paper:1324-1330` vs `owt.yaml:76`, which feeds the head count at `vae.py:224-226`); dropout 0.3 (`paper:1331-1337` vs `owt.yaml:81`); EMA decay 0.99 (`paper:1409-1415` vs `owt.yaml:121`); nested dropout 0.5 (`paper:1416-1422` vs `owt.yaml:135`).

Differences:

1. Which ẑ the VQ terms use. Eq. (9) uses the sampled cutoff only (`paper:390-392`). The code uses every level's ẑ^(i) and averages (`quant.py:515-524`, `:524` divides by 32). Eq. (9) has one commitment term and one codebook term, with no averaging over levels.
2. β. The paper gives no number (`paper:394-397`; Table 5 has no β row, `paper:1275-1425`). The code uses 0.25 (`owt.yaml:123`). Not comparable.
3. Codebook size. The paper's OWT column says V_k = 4,096 (`paper:1296-1302`). `owt.yaml:86-117` says 8192. Loading is strict (`vqvae_tokenizer.py:18`), so the shipped checkpoint must match whichever size was used. The checkpoint is not in the tree.
4. Cutoff distribution. The paper samples a cutoff k in {1..K} (`paper:384-388`). Table 5 gives 0.5 as the "probability of dropping the finest levels" (`paper:1423-1425`). The code draws a Bernoulli(0.5) gate, then a uniform truncation over 31 points (`quant.py:460-461`, `:525-530`). It never drops to the full 32 levels in the truncated branch. The paper does not specify the distribution.
5. R_k. The paper describes a reconstruction operator that maps codes to length L (`paper:374-377`). The code's R_k = Phi ∘ U (`quant.py:507-512`). It uses a learned level-0 sampler (`:311-328`), linear interpolation at levels 1..30 (`:408`), and a per-level residual conv (`:11-38`). None of these three is in the paper text read.
6. Down_k. The paper says the residual is "downsampled" without a method (`paper:371-372`). The code uses area pooling for levels 1..30 (`quant.py:399`) and a learned five-stage conv for level 0 (`:397-398`).
7. Encoder. The paper says the encoder maps x to L latent vectors (`paper:366-370`). The code adds a pre-quant GroupNorm (`vqvae.py:66`, `:86`) and dropout before it (`vae.py:171`). Neither is in the paper text read.
8. Codebook update. Table 5 lists an EMA decay (`paper:1409-1415`), but the text does not describe EMA. With EMA the codebook term never trains the codebook (`quant.py:300`). Whether the paper intends gradients to the codebook is not stated.
9. Level sizes. The paper text read does not give OWT level sizes. Its level-schedule discussion (`paper:1854-1856`) covers an ablation with K = 12. The code uses (k+1)² (`owt.yaml:139-171`). Not checkable.
10. Precision. The paper says BF16 mixed precision (`paper:1264`). The tree has no autocast. The VQ forward casts to float32 (`quant.py:449`).

Algorithm 1 (`paper:407-423`) is the generator's training step. It uses the generator loss of Eq. (10) (`paper:404-406`). The paper has no tokenizer algorithm to compare against. The generator is in `models/next_concept.py`, which was not read.

## 9. Known defects in the reference (do not copy)

1. `models/vqvae.py:116` unpacks three values from `self.vq(z)`, which returns five (`models/quant.py:543-549`). `VQVAE.reconstruct` (`vqvae.py:113-118`) and its callers (`tokenization/vqvae_tokenizer.py:56`, `:63`) would raise as written. Read-only finding; not executed.
2. `models/quant.py:548` returns the literal 0.1 where a perplexity would be expected.
3. `models/vqvae.py:146-158` (`__main__`) passes a bare list as `levels`, where `quant.py:285` expects a config node with `use_manual_levels`. The smoke test fails as written.

## 10. UNRESOLVED

1. Reconstruction cross-entropy. No tokenizer training code exists in `R/`: a grep for backward, optimizer, AdamW, lr_scheduler and cross_entropy found only `scripts/eval_merge.py:131`. So the label source, the padding mask (`vqvae_tokenizer.py:25`), the reduction (the paper's form is a mean over L, `paper:392`) and the weight relative to the VQ loss are all unknown.
2. β for the shipped OWT tokenizer. The paper gives no number (`paper:394-397`). The code's 0.25 (`owt.yaml:123`) comes from the eval config and is not verified against training.
3. OWT codebook size: 4,096 in `paper:1296-1302` versus 8192 in `owt.yaml:86-117`. Which one the shipped `tokenizer.pt` uses is not in the tree.
4. Cutoff distribution. The paper does not specify it (`paper:384-388`), and the training code is absent. The code's choice (`quant.py:525-530`) may not match the shipped model.
5. OWT level sizes. The paper text read does not list them. The code's (k+1)² list is at `owt.yaml:139-171`.
6. PyTorch semantics the description relies on (external). `requirements.txt:1` leaves torch unpinned, so these must be checked against the torch actually used. They are: `F.interpolate` "area" (adaptive average pooling; window bounds follow PyTorch's implementation) and "linear" with default `align_corners` (`quant.py:408` passes none); GroupNorm statistics over (channels-per-group × L) (`vqvae.py:66`, `:86`); exact `nn.GELU` (`vae.py:77`); `nn.Dropout` scaling; SDPA's default `is_causal=False` (`vae.py:65-67`); and default Conv, Linear and Embedding initialization.
7. Training-time initialization. `grep` found no `.apply(` call in `models/`, `utils/`, `tokenization/`, `scripts/` or `evaluation/`. The training script is absent, so a re-initialization there cannot be ruled out.
8. Training precision. The paper says BF16 mixed precision (`paper:1264`). No autocast appears in the tree.
9. Paper training settings for the OWT tokenizer (`paper:1346-1407`): 78,432 updates, global batch 768, peak LR 3e-4, floor 3e-5, 1,000 warmup, AdamW betas (0.9, 0.95), weight decay 0.01, gradient clip 1. None is in the tree, so they cannot be verified.
10. Whether the shipped OWT tokenizer was trained with dead-code restart on. The eval config has none (`owt.yaml:119-174`), and the training config is absent.
