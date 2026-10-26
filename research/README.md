# Research folder

The files here hold the evidence behind [PROJECT.md](../PROJECT.md). A fact in PROJECT.md that is not a decision cites a file here, or a line of the pinned code. Nothing here is a plan. Plans are in [ROADMAP.md](../ROADMAP.md).

| File | What it holds | Source and date |
|---|---|---|
| [checks.md](checks.md) | Every check, with its method and its result. Each entry says whether it was re-run in the pass that wrote the file, or carried from an earlier pass | Written 2026-10-26 |
| [ncm_facts.md](ncm_facts.md) | The NCM fact sheet, with line citations to the pinned commit: generation, training, the VQ tokenizer, hyperparameters, evaluation, and the contradictions between the paper and the code | Pinned commit `c70274273f489abc73832b2b98584a8483b6b203`; paper arXiv:2610.14162v1 |
| [software_sources.md](software_sources.md) | Source facts of tokenizers v0.23.3 and safetensors v0.8.0, with line numbers, and the sliding-window mask of transformers | Upstream tags, read 2026-10-26 |
| [tokenizer_pipelines.json](tokenizer_pipelines.json) | The exact normalizer, pre-tokenizer, post-processor, decoder and model flags of the eight families | Each family's `tokenizer.json` at its pinned revision |
| [transformers_formulas.md](transformers_formulas.md) | The transformers v5.20.2 formulas, extracted by reading the pinned files, with line numbers | transformers commit `18329f1065382e54db698e242cf98bd3f3e5a766` |
| [deep_research_digest.md](deep_research_digest.md) | The completed deep-research run: 25 claims, 13 confirmed and 12 refuted, with the corrections it made | Workflow `wf_3a9de5f1-8b7`, finished 2026-10-26 |
| [converter/wmt14_tokenizer_keys.md](converter/wmt14_tokenizer_keys.md) | The 96 state-dict keys of the released WMT14 tokenizer, derived by reading the pinned code | Pinned commit; derived 2026-10-26 |
| [converter/PROCEDURE.md](converter/PROCEDURE.md) | How the converter and the reference comparisons are run, and which approvals they need | Written 2026-10-26 |
| [reference/vq_forward_semantics.md](reference/vq_forward_semantics.md) | The VQ forward pass, the losses, the level-wise resampling, the EMA and the nested dropout, with line citations | Pinned commit; read 2026-10-26 |

## Pinned revisions used by the checks

- transformers v5.20.2, commit `18329f1065382e54db698e242cf98bd3f3e5a766`.
- NCM reference, commit `c70274273f489abc73832b2b98584a8483b6b203`.
- tokenizers v0.23.3, commit `301277072263fb3830eb66eec7e47adadf76f1dc`. safetensors v0.8.0, commit `a406ca3e7a90598be0cd05a50069cb9bf5ef6ba6`.
- Released checkpoints: `hazyresearch/ncm-checkpoints` at `d342ce8aaea58830eef3b8ad473664673e41f692`.
- Source checkpoint revisions: the table in PROJECT.md ("Source checkpoints"), and the `provenance.json` file of each fixture.

## How to repeat a check

Each check in [checks.md](checks.md) states its method and its input.

- The fixture checks read files in the repository, under `packages/hf_config/testdata/`. They need no network.
- The Hugging Face and PyPI checks use public metadata endpoints. Their responses are the sources listed in checks.md, sections 11 and 15.
- The NCM reference is not in the repository. Obtain it at the commit above, outside the repository. For example, clone `https://github.com/HazyResearch/ncm`, then check out `c70274273f489abc73832b2b98584a8483b6b203`. Its line numbers are the ones cited in the files here.
- The paper is arXiv:2610.14162v1. The line numbers in checks.md, section 5, refer to the plain text of that version.

## Raw downloads

Raw downloads and the scratch scripts were kept in the session scratchpad, outside the repository. A repeat of a check does not need them. Each check says how to fetch its input.
