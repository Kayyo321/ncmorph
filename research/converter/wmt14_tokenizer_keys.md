# WMT14 VQ tokenizer: registered state_dict keys and shapes

## 1. Header

- **Reference:** HazyResearch/ncm at commit `c70274273f489abc73832b2b98584a8483b6b203`. The commit is taken from PROJECT.md:42 and the pinned copy's directory name. The copy has no `.git` directory, so the hash is not checked against its history.
- **Date:** 2026-10-26
- **Class derived:** `VQVAE` (`models/vqvae.py:10`). `VQVAETokenizer.from_config` builds it as `VQVAE(**config.vqvae)` (`tokenization/vqvae_tokenizer.py:14-19`, call at line 17).
- **Pinned copy (REF):** `/private/tmp/claude-501/-Users-jessicabruce--claude-spares-spare-e7phlw/19e02558-926e-417a-8cc2-e33c5915494f/scratchpad/ncm_license_check/src/ncm-c70274273f489abc73832b2b98584a8483b6b203`. Every file path below is relative to REF unless it is absolute.
- **Method:** Read-only. This session has no Grep or Glob tool, so searches used read-only `grep` and `find` through Bash. No interpreter was run, no reference file was imported or executed, and `VQVAE` was not constructed.

Files read (REF-relative, line ranges):

- `models/vqvae.py` 1-165 (whole file)
- `models/vae.py` 1-272 (whole file)
- `models/quant.py` 1-737 (whole file)
- `models/__init__.py` and `utils/__init__.py`: empty (0 lines each)
- `utils/registry.py` 1-23 (whole file; quantizer mapping at line 12)
- `utils/misc.py` 1-30 (whole file; `count_params` at lines 4-14)
- `tokenization/vqvae_tokenizer.py` 1-86 (whole file)
- `tokenization/tokenizer.py` 1-31 (whole file)
- `config/eval/wmt14.yaml` 1-117 (whole file; the VQ block is `tokenizer.vqvae` at lines 50-106)
- `models/next_concept.py` 78-84 and 255-262 (only to confirm where the VQ-VAE is held)
- `scripts/eval_shard.py` 26-52 and `evaluate.py` 12-30 (config loading, checked for merged defaults)
- `requirements.txt` 1-9 (torch is not pinned)
- `models/common.py`: not read. An import grep shows only `models/next_concept.py:5` imports it, so it is not on the VQ path.

Grep checks (REF): imports in `models/*.py`, `utils/*.py` and `tokenization/*.py`; references to `VQVAE`, `vqvae`, `from_config`, `OmegaConf` and `hydra`; config merges and tokenizer overrides; `tie`, `set_default_dtype`, `bfloat16`, `float16`, `.half()` and `.to(torch.` on the VQ path. None of the tying or dtype patterns matched.

## 2. WMT14 VQ config values used

Source: `config/eval/wmt14.yaml`, block `tokenizer.vqvae`. The config is loaded with `OmegaConf.load` only (`evaluate.py:19`, `scripts/eval_shard.py:29`). The only edits are `checkpoint_path` and the pre-tokenizer paths (`scripts/eval_shard.py:45, 48`), so no defaults are merged in.

| Key | Value | yaml line | Used at (code) |
|---|---|---|---|
| n_layers | 6 | 51 | `models/vqvae.py:17`; decoder head count `models/vae.py:224-226`, so head dim = 384 / 6 = 64 |
| embed_dim | 384 | 52 | `models/vqvae.py:15` |
| in_vocab_size | 151680 | 53 | `models/vqvae.py:13`; encoder `wte` rows and decoder `output_layer` rows |
| context_length | 64 | 54 | `models/vqvae.py:14`; encoder `wpe` rows |
| compression_factor | 1 | 55 | `models/vqvae.py:19`; no `Encoder.down` (`models/vae.py:149`) and no `Decoder.up` (`models/vae.py:237`) |
| dropout | 0.1 | 56 | `models/vqvae.py:20`; no state entries |
| pre_quant_groupnorm | 8 | 57 | `models/vqvae.py:21, 60, 66`: `GroupNorm(8, 384, affine=False)` |
| use_qk_norm | true | 58 | `models/vqvae.py:23`; `q_norm` and `k_norm` in the decoder block (`models/vae.py:40-44`) |
| pre_quant_dropout | 0.25 | 59 | `models/vqvae.py:22, 69-71`; no state entries |
| vqvae_vocab_size | 4096 x 11 | 60-71 | `models/vqvae.py:16`; `models/quant.py:292-304` |
| checkpoint_path | tokenizer.pt | 72 | `models/vqvae.py:32` (absorbed by `**kwargs`); read at `tokenization/vqvae_tokenizer.py:18` |
| vector_quantizer_config.clss | multiscale_residual_vector_quantizer | 74 | `models/vqvae.py:50`; `utils/registry.py:12` maps it to `models.quant.MultiscaleResidualVectorQuantizer` |
| vector_quantizer_config.decay | 0.99 | 75 | `models/quant.py:256-261` (no state entries) |
| vector_quantizer_config.epsilon | 1.0e-05 | 76 | `models/quant.py:254` (no state entries) |
| vector_quantizer_config.commitment_cost | 0.25 | 77 | `models/quant.py:253` (no state entries) |
| vector_quantizer_config.learned_l1_sampling | true | 78 | `models/quant.py:271, 311`: `learned_down_l1` and `learned_up_l1` are created |
| vector_quantizer_config.learned_l1_cascade | true | 79 | `models/quant.py:272, 317`: cascaded stages |
| vector_quantizer_config.learned_l1_cascade_base | 4 | 80 | `models/quant.py:273, 318`; stride plan `models/quant.py:165-178` gives [4, 4, 4] |
| vector_quantizer_config.learned_l1_cascade_norm | true | 81 | `models/quant.py:274, 319`: GroupNorm between stages (`models/quant.py:189-190, 201-202`) |
| vector_quantizer_config.learned_l1_cascade_norm_final | true | 82 | `models/quant.py:275, 321`: final GroupNorm (`models/quant.py:192-193`) |
| vector_quantizer_config.learned_all_sampling | false | 83 | `models/quant.py:270, 330-342`: refine modules are None |
| vector_quantizer_config.quant_resi.enabled | true | 85 | `models/quant.py:109` |
| vector_quantizer_config.quant_resi.ratio | 0.5 | 86 | `models/quant.py:112, 130`: scalar, so one value per level; `Phi(384, 0.5)` at `models/quant.py:137` |
| vector_quantizer_config.quant_resi.share_mode | 0 | 87 | `models/quant.py:113, 139-140`: `PhiNonShared`, one Phi per level |
| vector_quantizer_config.quant_resi.learnable_ratio | false | 88 | `models/quant.py:114, 135-137`: no `ratio_logit` parameter |
| vector_quantizer_config.aux.fine_drop.prob | 0.5 | 91 | `models/quant.py:263-264`: train-time only; no state entries |
| vector_quantizer_config.aux.fine_drop.min_keep | 1 | 92 | `models/quant.py:265`; no state entries |
| vector_quantizer_config.levels.use_manual_levels | true | 94 | `models/quant.py:285-286` |
| vector_quantizer_config.levels.manual_levels | 1, 4, 8, 14, 20, 26, 32, 40, 48, 56, 64 (11 values) | 96-106 | `models/quant.py:286, 290-292`: count must equal the vocab-size count (11 = 11); L = 64, level-0 stride = 64 (`models/quant.py:312-314`) |

Keys not in the yaml, so the signature defaults apply:

- `enc_layers=0` (`models/vqvae.py:24`), `enc_attn_window=0` (25), `enc_conv_layers=0` (26), `enc_conv_kernel=3` (27), `dec_layers=1` (28), `pre_quant_norm_type='group'` (29), `enc_positional=True` (30), `dec_positional=False` (31).
- Quantizer defaults: `restart_threshold=0.0` (`models/quant.py:244`), `ema_scale_by_world_size=False` (245), `kmeans_init=False` (246), `kmeans_iters=10` (247). `pre_quant_norm_type` (`models/quant.py:248`) is accepted but unused, and `VQVAE` does not pass it.

Symbols used in section 3:

- D = embed_dim = 384
- V_i = vqvae_vocab_size[i] = 4096, for i = 0..10
- C = in_vocab_size = 151680
- T = context_length = 64
- H = n_layers = 6 (decoder head count); Hd = D / H = 64
- L = levels[-1] = 64, levels[0] = 1, so the level-0 stride is 64
- S_0 = S_1 = S_2 = 4 (cascade strides for base 4 over stride 64)

Note: `model.vocab_size` (`config/eval/wmt14.yaml:18-28`, 10 entries) belongs to the generator. The VQ uses its own 11-entry list at lines 61-71.

## 3. Registered state_dict entries (96)

Row order follows the order `state_dict()` visits entries. `VQVAE` children are created in the order encoder, vq, decoder. Inside each module, direct parameters and persistent buffers come before child modules.

Dtype: no reference code sets a dtype, so constructor tensors are float32, except `codebook_hit_i` (int64, `models/quant.py:304`). The on-disk dtype is not determined by this note.

Decoder block (rows 82-95) is built at `models/vae.py:228` from `_cfg` (`models/vae.py:225-227`): n_embd = D, n_head = n_layers = 6, bias = True, use_qk_norm = True. It has no `attn_window`, so `attn_window` = 0 (`models/vae.py:46`) and no mask is applied.

| # | state_dict key | kind | shape formula | WMT14 shape | dtype | registered at (file:line) |
|---|---|---|---|---|---|---|
| 1 | encoder.wte.weight | parameter | (C, D) | (151680, 384) | float32 | models/vae.py:126 (nn.Embedding); encoder built at models/vqvae.py:42-48 |
| 2 | encoder.wpe.weight | parameter | (T, D) | (64, 384) | float32 | models/vae.py:127 (nn.Embedding; use_pos_emb = enc_positional = True, models/vqvae.py:30, 47) |
| 3 | vq.ema_cluster_size_0 | buffer | (V_0,) | (4096,) | float32 | models/quant.py:302 (register_buffer) |
| 4 | vq.ema_w_0 | buffer | (V_0, D) | (4096, 384) | float32 | models/quant.py:303 (register_buffer) |
| 5 | vq.codebook_hit_0 | buffer | (V_0,) | (4096,) | int64 | models/quant.py:304 (register_buffer) |
| 6 | vq.ema_cluster_size_1 | buffer | (V_1,) | (4096,) | float32 | models/quant.py:302 |
| 7 | vq.ema_w_1 | buffer | (V_1, D) | (4096, 384) | float32 | models/quant.py:303 |
| 8 | vq.codebook_hit_1 | buffer | (V_1,) | (4096,) | int64 | models/quant.py:304 |
| 9 | vq.ema_cluster_size_2 | buffer | (V_2,) | (4096,) | float32 | models/quant.py:302 |
| 10 | vq.ema_w_2 | buffer | (V_2, D) | (4096, 384) | float32 | models/quant.py:303 |
| 11 | vq.codebook_hit_2 | buffer | (V_2,) | (4096,) | int64 | models/quant.py:304 |
| 12 | vq.ema_cluster_size_3 | buffer | (V_3,) | (4096,) | float32 | models/quant.py:302 |
| 13 | vq.ema_w_3 | buffer | (V_3, D) | (4096, 384) | float32 | models/quant.py:303 |
| 14 | vq.codebook_hit_3 | buffer | (V_3,) | (4096,) | int64 | models/quant.py:304 |
| 15 | vq.ema_cluster_size_4 | buffer | (V_4,) | (4096,) | float32 | models/quant.py:302 |
| 16 | vq.ema_w_4 | buffer | (V_4, D) | (4096, 384) | float32 | models/quant.py:303 |
| 17 | vq.codebook_hit_4 | buffer | (V_4,) | (4096,) | int64 | models/quant.py:304 |
| 18 | vq.ema_cluster_size_5 | buffer | (V_5,) | (4096,) | float32 | models/quant.py:302 |
| 19 | vq.ema_w_5 | buffer | (V_5, D) | (4096, 384) | float32 | models/quant.py:303 |
| 20 | vq.codebook_hit_5 | buffer | (V_5,) | (4096,) | int64 | models/quant.py:304 |
| 21 | vq.ema_cluster_size_6 | buffer | (V_6,) | (4096,) | float32 | models/quant.py:302 |
| 22 | vq.ema_w_6 | buffer | (V_6, D) | (4096, 384) | float32 | models/quant.py:303 |
| 23 | vq.codebook_hit_6 | buffer | (V_6,) | (4096,) | int64 | models/quant.py:304 |
| 24 | vq.ema_cluster_size_7 | buffer | (V_7,) | (4096,) | float32 | models/quant.py:302 |
| 25 | vq.ema_w_7 | buffer | (V_7, D) | (4096, 384) | float32 | models/quant.py:303 |
| 26 | vq.codebook_hit_7 | buffer | (V_7,) | (4096,) | int64 | models/quant.py:304 |
| 27 | vq.ema_cluster_size_8 | buffer | (V_8,) | (4096,) | float32 | models/quant.py:302 |
| 28 | vq.ema_w_8 | buffer | (V_8, D) | (4096, 384) | float32 | models/quant.py:303 |
| 29 | vq.codebook_hit_8 | buffer | (V_8,) | (4096,) | int64 | models/quant.py:304 |
| 30 | vq.ema_cluster_size_9 | buffer | (V_9,) | (4096,) | float32 | models/quant.py:302 |
| 31 | vq.ema_w_9 | buffer | (V_9, D) | (4096, 384) | float32 | models/quant.py:303 |
| 32 | vq.codebook_hit_9 | buffer | (V_9,) | (4096,) | int64 | models/quant.py:304 |
| 33 | vq.ema_cluster_size_10 | buffer | (V_10,) | (4096,) | float32 | models/quant.py:302 |
| 34 | vq.ema_w_10 | buffer | (V_10, D) | (4096, 384) | float32 | models/quant.py:303 |
| 35 | vq.codebook_hit_10 | buffer | (V_10,) | (4096,) | int64 | models/quant.py:304 |
| 36 | vq.embeddings.0.weight | parameter, requires_grad=False | (V_0, D) | (4096, 384) | float32 | models/quant.py:295 (nn.Embedding in ModuleList, lines 294-296); requires_grad_(False) at models/quant.py:300 does not remove it |
| 37 | vq.embeddings.1.weight | parameter, requires_grad=False | (V_1, D) | (4096, 384) | float32 | models/quant.py:295 |
| 38 | vq.embeddings.2.weight | parameter, requires_grad=False | (V_2, D) | (4096, 384) | float32 | models/quant.py:295 |
| 39 | vq.embeddings.3.weight | parameter, requires_grad=False | (V_3, D) | (4096, 384) | float32 | models/quant.py:295 |
| 40 | vq.embeddings.4.weight | parameter, requires_grad=False | (V_4, D) | (4096, 384) | float32 | models/quant.py:295 |
| 41 | vq.embeddings.5.weight | parameter, requires_grad=False | (V_5, D) | (4096, 384) | float32 | models/quant.py:295 |
| 42 | vq.embeddings.6.weight | parameter, requires_grad=False | (V_6, D) | (4096, 384) | float32 | models/quant.py:295 |
| 43 | vq.embeddings.7.weight | parameter, requires_grad=False | (V_7, D) | (4096, 384) | float32 | models/quant.py:295 |
| 44 | vq.embeddings.8.weight | parameter, requires_grad=False | (V_8, D) | (4096, 384) | float32 | models/quant.py:295 |
| 45 | vq.embeddings.9.weight | parameter, requires_grad=False | (V_9, D) | (4096, 384) | float32 | models/quant.py:295 |
| 46 | vq.embeddings.10.weight | parameter, requires_grad=False | (V_10, D) | (4096, 384) | float32 | models/quant.py:295 |
| 47 | vq.quant_resi.0.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | models/quant.py:137 builds Phi; Conv1d kernel 3 at models/quant.py:13; container PhiNonShared at models/quant.py:139-140 |
| 48 | vq.quant_resi.0.bias | parameter | (D,) | (384,) | float32 | as row 47 (Conv1d bias=True default) |
| 49 | vq.quant_resi.1.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 50 | vq.quant_resi.1.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 51 | vq.quant_resi.2.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 52 | vq.quant_resi.2.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 53 | vq.quant_resi.3.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 54 | vq.quant_resi.3.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 55 | vq.quant_resi.4.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 56 | vq.quant_resi.4.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 57 | vq.quant_resi.5.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 58 | vq.quant_resi.5.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 59 | vq.quant_resi.6.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 60 | vq.quant_resi.6.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 61 | vq.quant_resi.7.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 62 | vq.quant_resi.7.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 63 | vq.quant_resi.8.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 64 | vq.quant_resi.8.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 65 | vq.quant_resi.9.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 66 | vq.quant_resi.9.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 67 | vq.quant_resi.10.weight | parameter | (D, D, 3) | (384, 384, 3) | float32 | as row 47 |
| 68 | vq.quant_resi.10.bias | parameter | (D,) | (384,) | float32 | as row 47 |
| 69 | vq.learned_down_l1.0.weight | parameter | (D, D, S_0) | (384, 384, 4) | float32 | models/quant.py:320 (cascade branch, lines 317-321); `_make_cascaded_down_l1` models/quant.py:185-194; nn.Conv1d at models/quant.py:146 |
| 70 | vq.learned_down_l1.0.bias | parameter | (D,) | (384,) | float32 | as row 69 |
| 71 | vq.learned_down_l1.2.weight | parameter | (D, D, S_1) | (384, 384, 4) | float32 | as row 69 (index 1 is a GroupNorm with no state; see section 4) |
| 72 | vq.learned_down_l1.2.bias | parameter | (D,) | (384,) | float32 | as row 69 |
| 73 | vq.learned_down_l1.4.weight | parameter | (D, D, S_2) | (384, 384, 4) | float32 | as row 69 (index 3 is a GroupNorm with no state) |
| 74 | vq.learned_down_l1.4.bias | parameter | (D,) | (384,) | float32 | as row 69 |
| 75 | vq.learned_up_l1.0.weight | parameter | (D, D, S_0) | (384, 384, 4) | float32 | models/quant.py:322; `_make_cascaded_up_l1` models/quant.py:197-204; nn.ConvTranspose1d at models/quant.py:156 |
| 76 | vq.learned_up_l1.0.bias | parameter | (D,) | (384,) | float32 | as row 75 |
| 77 | vq.learned_up_l1.2.weight | parameter | (D, D, S_1) | (384, 384, 4) | float32 | as row 75 (index 1 is a GroupNorm with no state) |
| 78 | vq.learned_up_l1.2.bias | parameter | (D,) | (384,) | float32 | as row 75 |
| 79 | vq.learned_up_l1.4.weight | parameter | (D, D, S_2) | (384, 384, 4) | float32 | as row 75 (index 3 is a GroupNorm with no state) |
| 80 | vq.learned_up_l1.4.bias | parameter | (D,) | (384,) | float32 | as row 75 |
| 81 | decoder.ln_f.weight | parameter | (D,) | (384,) | float32 | models/vae.py:234 (`LayerNorm(embed_dim)`); weight at models/vae.py:9; bias is None (models/vae.py:10, bias=False default at models/vae.py:7) |
| 82 | decoder.middle_block.ln_1.weight | parameter | (D,) | (384,) | float32 | models/vae.py:92 (LayerNorm); weight at models/vae.py:9 |
| 83 | decoder.middle_block.ln_1.bias | parameter | (D,) | (384,) | float32 | models/vae.py:92; bias at models/vae.py:10 (bias=True from models/vae.py:226) |
| 84 | decoder.middle_block.attn.c_attn.weight | parameter | (3D, D) | (1152, 384) | float32 | models/vae.py:93 (SelfAttention); nn.Linear at models/vae.py:33 |
| 85 | decoder.middle_block.attn.c_attn.bias | parameter | (3D,) | (1152,) | float32 | models/vae.py:33 |
| 86 | decoder.middle_block.attn.c_proj.weight | parameter | (D, D) | (384, 384) | float32 | models/vae.py:34 (nn.Linear) |
| 87 | decoder.middle_block.attn.c_proj.bias | parameter | (D,) | (384,) | float32 | models/vae.py:34 |
| 88 | decoder.middle_block.attn.q_norm.weight | parameter | (Hd,), Hd = D / H | (64,) | float32 | models/vae.py:41-43 (use_qk_norm, line 41); RMSNorm weight at models/vae.py:19 |
| 89 | decoder.middle_block.attn.k_norm.weight | parameter | (Hd,) | (64,) | float32 | models/vae.py:44; RMSNorm weight at models/vae.py:19 |
| 90 | decoder.middle_block.ln_2.weight | parameter | (D,) | (384,) | float32 | models/vae.py:94 (LayerNorm); weight at models/vae.py:9 |
| 91 | decoder.middle_block.ln_2.bias | parameter | (D,) | (384,) | float32 | models/vae.py:94; bias at models/vae.py:10 |
| 92 | decoder.middle_block.mlp.c_fc.weight | parameter | (4D, D) | (1536, 384) | float32 | models/vae.py:95 (MLP); nn.Linear at models/vae.py:76 |
| 93 | decoder.middle_block.mlp.c_fc.bias | parameter | (4D,) | (1536,) | float32 | models/vae.py:76 |
| 94 | decoder.middle_block.mlp.c_proj.weight | parameter | (D, 4D) | (384, 1536) | float32 | models/vae.py:78 (nn.Linear) |
| 95 | decoder.middle_block.mlp.c_proj.bias | parameter | (D,) | (384,) | float32 | models/vae.py:78 |
| 96 | decoder.output_layer.weight | parameter | (C, D) | (151680, 384) | float32 | models/vae.py:233 (nn.Linear, bias=False) |

Counts: 2 encoder entries (rows 1-2), 78 vq entries (rows 3-80: 33 buffers, 11 embeddings, 22 quant_resi, 6 learned_down_l1, 6 learned_up_l1), and 16 decoder entries (rows 81-96).

## 4. Entries the module has that are NOT in state_dict

| Item | Kind | Set at (file:line) | Why it has no state_dict entry |
|---|---|---|---|
| vq._fd_step | buffer, persistent=False | models/quant.py:269 | Non-persistent buffer (rule P2 in section 7) |
| vq._km_done | buffer | models/quant.py:281-283 | Not created: kmeans_init is False (default models/quant.py:246; not set in wmt14.yaml) |
| vq._km_buf | plain attribute, None | models/quant.py:279 | kmeans_init is False |
| vq.learned_down_refine, vq.learned_up_refine | plain attribute, None | models/quant.py:341-342 | learned_all_sampling is False (wmt14.yaml:83) |
| vq.levels | plain attribute, ListConfig [1..64] | models/quant.py:286 | Not a tensor |
| vq.vqvae_vocab_sizes | plain attribute, list of 11 ints | models/quant.py:292 | Not a tensor |
| vq scalars: embed_dim, commitment_cost, epsilon, base_decay, base_decay_per_level, ema_scale_by_world_size, fine_drop_prob, fine_drop_min_keep, fine_drop_prob_start, fine_drop_anneal_steps, restart_threshold, kmeans_init, kmeans_iters, learned_* flags, quant_resi_enabled, _qr_num_levels | plain attributes | models/quant.py:252-278, 307-308 | Not tensors |
| Phi._ratio_const | plain float 0.5 | models/quant.py:28 | Not a tensor. No ratio_logit parameter, because learnable_ratio is False (models/quant.py:23-26 not executed) |
| PhiNonShared.ticks (vq.quant_resi.ticks) | plain numpy array | models/quant.py:89 | Not a tensor |
| vq.learned_down_l1.1, .3, .5 | GroupNorm(8, 384, affine=False) | models/quant.py:190, 193 (via `_cascade_norm`, models/quant.py:182) | affine=False: no weight or bias (rule P1) |
| vq.learned_up_l1.1, .3 | GroupNorm(8, 384, affine=False) | models/quant.py:202 (via models/quant.py:182) | affine=False: no weight or bias (rule P1) |
| pre_quant_norm | GroupNorm(8, 384, affine=False) | models/vqvae.py:66 | affine=False: no weight or bias (rule P1) |
| pre_quant_drop | Dropout(0.25) | models/vqvae.py:70 | No parameters or buffers |
| encoder.drop, decoder.drop | Dropout(0.1) | models/vae.py:128, 220 | No parameters or buffers |
| SelfAttention attn_dropout, resid_dropout; MLP gelu, dropout | Dropout / GELU | models/vae.py:35-36, 77, 79 | No parameters or buffers |
| encoder.blocks | empty ModuleList | models/vae.py:130-136 | enc_layers = 0 by default (models/vqvae.py:24), so no children |
| encoder.conv_blocks | empty ModuleList | models/vae.py:138-143 | enc_conv_layers = 0 by default (models/vqvae.py:26), so no children |
| decoder.extra_blocks | empty ModuleList | models/vae.py:229-231 | dec_layers = 1 by default (models/vqvae.py:28, 55), so range(0) is empty |
| encoder.ln_e | None | models/vae.py:145 | enc_layers = 0 and enc_conv_layers = 0 |
| encoder.down | None | models/vae.py:148-149 | compression_factor = 1 (models/vqvae.py:19 via wmt14.yaml:55), so no conv |
| decoder.wpe | None | models/vae.py:219 | dec_positional = False (models/vqvae.py:31, 56) |
| decoder.up | None | models/vae.py:236-237 | compression_factor = 1 |
| first decoder.ln_f (LayerNorm instance) | module, replaced | models/vae.py:221, replaced at models/vae.py:234 | Overwritten before any state_dict call; its weight is not registered (rule P5) |
| decoder.ln_f.bias | None | models/vae.py:10 | LayerNorm built with bias=False (models/vae.py:234 passes no bias) |
| VQVAE plain attributes: embed_dim, vqvae_vocab_size, in_vocab_size, context_length, _pre_quant_is_layer | plain attributes | models/vqvae.py:36-39, 64, 67, 73 | Not tensors |
| VQVAE checkpoint_path (kwargs) | not an attribute | models/vqvae.py:32, 35 | Absorbed by `**kwargs` and unused by the module; read at tokenization/vqvae_tokenizer.py:18 |
| Tokenizer, VQVAETokenizer | not nn.Module | tokenization/tokenizer.py:4; tokenization/vqvae_tokenizer.py:7, 10 | Plain Python classes with no state_dict. The VQVAE is the attribute `self.vqvae` |
| VQVAE inside NextConceptTransformer | not a child module | models/next_concept.py:80-81 | Held in a tuple, which keeps the VQ-VAE out of the generator's module tree and state_dict (the reference comment at line 80 says so) |

## 5. UNRESOLVED

None. Every key, shape and count above follows from code in REF. The only assumptions are the PyTorch rules P1-P6 in section 7, which were not checked against torch source in this session.

## 6. Totals

- **state_dict entries for WMT14: 96** = 63 parameter entries + 33 buffer entries.
- **Parameter scalars: 144,002,816** = encoder 58,269,696 (2 entries) + vq 25,713,024 (45 entries) + decoder 60,020,096 (16 entries).
  - Encoder: 151680 x 384 + 64 x 384 = 58,245,120 + 24,576.
  - VQ parameters: embeddings 11 x 4096 x 384 = 17,301,504; quant_resi 11 x (384 x 384 x 3 + 384) = 4,870,272; learned_down_l1 3 x (384 x 384 x 4 + 384) = 1,770,624; learned_up_l1 the same, 1,770,624.
  - Decoder: ln_f 384; middle block 1,774,592 (c_attn 443,520; attn c_proj 147,840; q_norm + k_norm 128; mlp c_fc 591,360; mlp c_proj 590,208; ln_1 and ln_2 768 each); output_layer 151680 x 384 = 58,245,120.
- **Buffer scalars (counted separately): 17,391,616** = ema_cluster_size 11 x 4096 = 45,056; ema_w 11 x 4096 x 384 = 17,301,504; codebook_hit 11 x 4096 = 45,056.
- Parameter plus buffer scalars: 161,394,432.
- Cross-check, derived and not executed: `count_params` (`models/vqvae.py:75`, `utils/misc.py:9-12`) sums `model.parameters()` only, excluding names containing `wte` or `wpe`. It would print a total of 144.00M and a non-embedding count of 85.73M (144,002,816 - 58,269,696 = 85,733,120).

## 7. PyTorch rules relied on (not in the reference)

- **P1.** `nn.GroupNorm(..., affine=False)` has no weight or bias, so it adds no state entries. Applied at `models/vqvae.py:66` and `models/quant.py:182` (used at 190, 193, 202). There are 6 such modules: 1 pre-quant, 3 down-cascade, 2 up-cascade.
- **P2.** `register_buffer(..., persistent=False)` is excluded from state_dict. Applied at `models/quant.py:269`.
- **P3.** Parameters with `requires_grad=False` are still in state_dict. Applied at `models/quant.py:300`.
- **P4.** Weight layouts: `nn.Linear` (out, in); `nn.Embedding` (num, dim); `nn.Conv1d` (out, in/groups, k); `nn.ConvTranspose1d` (in, out/groups, k). Here in = out = 384, so both conv layouts give (384, 384, k).
- **P5.** Assigning a new child module to an existing attribute name replaces the old child. Applied at `models/vae.py:221` and `models/vae.py:234`.
- **P6.** Assigning None to a name that is not a registered child, parameter or buffer stores a plain attribute with no state entry.
- **P7.** `state_dict()` visits a module's direct parameters and persistent buffers before its child modules, and children in the order they were registered. This sets the row order in section 3; it does not change the key set.

Basis: PyTorch's documented API. No torch source was found: a search of /opt, /usr/local, /Library/Frameworks, ~/Documents, ~/ncmorph, ~/miniconda3, ~/anaconda3, ~/.pyenv, ~/.local and ~/.venvs for `torch/nn/modules/normalization.py` returned nothing. `requirements.txt:1` does not pin torch. Before the converter relies on the key set, compare it with a real state_dict from the torch version it uses.

## 8. Notes for the converter

1. **Load contract.** `tokenization/vqvae_tokenizer.py:18` reads `checkpoint_path` (`config/eval/wmt14.yaml:72`; file name `tokenizer.pt` per `wmt14.yaml:4`), takes the `'model'` key and calls `load_state_dict(..., strict=True)`. The `'model'` mapping must therefore hold exactly the 96 keys in section 3, with no `vqvae.` prefix.
2. **Pickle warning.** That loader is `torch.load(..., weights_only=False)`, which is pickle-based. AGENTS.md forbids loading `.pt` files written by `torch.save`. The converter must not reuse this loader. It should reject pickle `.pt` input with an error that tells the user to convert the file outside ncmorph.
3. **Level count.** The generator's `model.vocab_size` (`wmt14.yaml:18-28`, 10 entries) is not the VQ's. The VQ has 11 levels (`wmt14.yaml:61-71`; manual levels at lines 96-106).
4. **Not analyzed.** `models/next_concept.py:257-259` reads `vq.embeddings[0].weight` when `tie_heads_to_codebook` is set. The WMT14 model block does not set that key (`wmt14.yaml:9-41`), so that path is off for WMT14. Generator keys are out of scope for this note.
5. **Dtype on disk.** Not determined here. The constructors create float32 tensors, and the on-disk dtype belongs to `tokenizer.pt`.
