# Deep-research digest: workflow wf_3a9de5f1-8b7 (completed)

This digest replaces the partial digest of the same workflow, which was written while claim 25 was still in verification. The run is complete. The source is the workflow's result (109 agents, 0 errors, finished 2026-10-26 at 16:22 EDT). The full output is in the session scratchpad, at `tasks/wycideclb.output`.

The question asked six things: the parameters that NCM could take from an autoregressive model; how published work handles tokenizer mismatch; the safetensors and GGUF formats; Zig's version, build and standard-library conventions; practices for agent instruction files; and licensing. The run answered the first two, with abstract-level evidence for most precedents, and it answered the NCM facts and the licence of the NCM code. It did not verify items 3 to 5 or the licence terms of converted derivatives. The spec covers those items by checking them locally (see [checks.md](checks.md)), and by decision.

## 1. Counts

- Sources fetched: 27. Claims extracted: 122. Claims verified (3-vote, 2 of 3 refutes kill): 25.
- Verdicts: **13 confirmed, 12 refuted, 0 unverified.**
- Earlier partial run, for reference only: 11 refuted, 13 not refuted, 1 split. The final run turned the split into a refutation.

## 2. Verified claims, with final verdicts

| Claim | Verdict | Basis that the spec uses |
|---|---|---|
| The pinned XSum generator has 12 blocks, width 768, 12 heads, block-diagonal attention, RoPE base 1e5, QK-norm, level encodings and tied heads. The paper's Table 6 agrees for XSum (88.02M). | confirmed | `config/eval/xsum.yaml`; Table 6 |
| The VQ tokenizer, the 384-to-768 input projection, the level encodings and the codebook-tied heads have no LLM counterpart and must be trained. | confirmed (medium) | `models/next_concept.py` lines 131 to 133; `config/eval/owt.yaml` lines 64 to 66 |
| The paper does not initialize NCM from a pretrained autoregressive model. The only pretrained component it tests is a frozen T5-small prefix encoder, which it sets aside. | confirmed (medium) | paper Appendix F.2.2; `models/next_concept.py` lines 135 to 181 (opt-in paths, off in the released configs) |
| Published work converts autoregressive LLMs to diffusion or block diffusion by continued training, usually keeping the transformer blocks and changing the attention mask. DiffuLLaMA covers 127M to 7B parameters with under 200B tokens. | confirmed (medium) | arXiv:2410.17891 abstract (the 200B figure is a ceiling; the body reports 30B to 70B); arXiv:2509.26328; arXiv:2510.06303; arXiv:2508.15487 |
| Tokenizer mismatch is handled in published work by zero-shot tokenizer transfer. Standard embedding heuristics often perform at chance level. A hypernetwork that predicts embeddings from a tokenizer generalizes to XLM-R and Mistral-7B. | confirmed (medium) | arXiv:2405.07883 abstract and summary |
| The reference uses a different text tokenizer per task: GPT-2 for OWT and XSum, bert-base-uncased for LM1B, a custom BPE for TinyStories, Qwen3-0.6B for WMT14, and a Sudoku tokenizer. Prefix ids go through the VQ encoder's own table. | confirmed | `config/eval/*.yaml` `pre_tokenizer` blocks; `models/next_concept.py` lines 549 to 558; `scripts/eval_shard.py` line 91 |
| The paper's Table 5 gives 8 XSum levels; the pinned `xsum.yaml` has 9. The paper gives OWT 4,096 codes and a 12-block, 768-wide generator; `owt.yaml` gives 8,192 codes and 16 blocks at width 1024. | confirmed (high) | paper Tables 5 and 6; the two configs. Resolved by decisions D16, D19 and D43 in PROJECT.md. |
| The reference loads its generator with `torch.load(..., weights_only=False)`, which unpickles arbitrary objects. | confirmed (local read; the PyTorch security note was not checked) | `scripts/eval_shard.py` line 52. ncmorph never loads `.pt` files (AGENTS.md). |
| No verified source covers the safetensors header layout, dtypes, offsets and alignment, or the GGUF layout. | confirmed as a gap, then closed locally | Closed by reading the upstream source at v0.8.0 (`software_sources.md`, section C). GGUF is out of scope (PROJECT.md, Scope). |
| Zig 0.16.0 has no block comments. `zig ast-check` rejects a `/* */` header (exit 1) and accepts the `//` form (exit 0). | confirmed (local compiler) | Re-run this pass: see `checks.md` section 7. |
| The repository already uses several measures against agent error: a shared AGENTS.md imported by CLAUDE.md, a pinned reference commit and Zig version, an `UNVERIFIED:` marker rule, grep-checkable header, weight and isolation checks, and a verification list. | confirmed as local fact; the external guidance was a gap | `AGENTS.md`, `CLAUDE.md`. External guidance is not needed to implement the spec. |
| The NCM code is MIT-licensed. The LICENSE says `Copyright (c) 2025 hermann kumbong`, and the standard permission and warranty text. The notice must stay in copies or substantial portions. | confirmed (3-0) | `LICENSE` at the pinned commit; blob `8c9c7f4ff5fc026b93c55dcce141d142bd3b650c` |
| The licence terms that govern converted derivatives of the source models are unverified. Only the Hugging Face tags were read. | confirmed as a gap, then closed locally | Closed by reading the Llama 2, Llama 3 and Llama 3.2 licence texts this pass (`checks.md` section 8). The Apache-2.0 tags for Qwen and Mistral were read from the saved API responses. |

Two other verdicts changed the design:

- **ZeTT claims (C22 to C24) were confirmed**, with medium confidence. ZeTT has not been tested on NCM-style codebooks. The spec does not rely on it. The Qwen3 reuse decision (D14) and the trained tokenizers (D10) use no transfer method.
- **WECHSEL (C25) was refuted** (two of three voters refuted it). The spec does not use it.

## 3. Refuted claims

Twelve claims were refuted. Local checks confirmed the content of eight of them (generator table values; no autoregressive initialization in the paper; the frozen tokenizer design; raw prefix ids; opt-in GPT-2 and T5 paths; within-level parallel generation; Dream 7B's autoregressive initialization; the 9-level XSum tokenizer, which the config supports and the paper does not). The rest were refuted on wording or scope, and the spec does not depend on them. The four refutations that matter for the spec are:

- Table 4 lists five text tokenizers, one of them from Qwen3. The research question's "GPT-2 BPE pre-tokenizer" was only partly right.
- The statement that "the NCM generator is not initialized from an LLM" holds, but a frozen VQ codebook is copied into the heads (`tie_heads_to_codebook`).
- `use_positional_encoding` is never read by the code. RoPE is the only active positional mechanism.
- "Only abstract-level evidence" for AR-to-diffusion conversion is contradicted by the other precedents listed above.

## 4. Process issue

The synthesis step wrote into `PROJECT.md`. Its edit added four open questions (O-XSUM, O-OWTCFG, O-LIC and O-GAP) and replaced the research section. Those questions conflict with the owner's instruction that the project holds no open questions. The rewritten PROJECT.md closes each one by decision, and the earlier text is kept at `scratchpad/before_rewrite/PROJECT.md`. Research workflows must not write into `PROJECT.md`. They should return their findings here, and the spec is written from them.

## 5. What the run did not reach

- Safetensors and GGUF layouts, and Zig's release policy and standard-library APIs, were not in any verified claim. They are checked locally in `checks.md` and `software_sources.md`.
- Fast-dLLM v2, SDAR, Dream 7B and the block-diffusion paper were read at abstract level only.
- The Table 4 tokenizer list and the page-level NCM settings were refuted on wording and are checked against the paper text and the configs locally.
- Two sources returned errors or were not fetched: the NVD entry, and the Llama 3.2 licence page at its older URL. The current Llama licence pages were read directly this pass.
