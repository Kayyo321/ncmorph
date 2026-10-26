# ncmorph

ncmorph is a bridge that turns an open-weight transformer language model into a [Next-Concept Modeling (NCM)](https://arxiv.org/abs/2610.14162) model. The project is written primarily in [Zig](https://ziglang.org), with other languages used only where a tool or dependency requires them.

NCM generates text coarse to fine. A multiscale residual vector-quantized tokenizer represents a sequence as a hierarchy of discrete codes, and a transformer predicts that hierarchy level by level, generating all positions within a level in parallel. ncmorph carries an existing open-weight model into that form.

- [PROJECT.md](PROJECT.md) describes the scope, architecture and design decisions.
- [AGENTS.md](AGENTS.md) holds the rules for AI coding agents working in this repository.
- [LICENSE](LICENSE) is the Apache License 2.0.

## Requirements

- The Zig version pinned in `build.zig.zon`. Install that exact version, not the newest release.
- Open-weight models in Hugging Face `safetensors` format. Model weights are not part of this repository and must never be committed.

## Build and test

```bash
zig build
zig build test
```

## Inspect a checkpoint

`inspect` checks a local checkpoint against its `config.json`. It reads only the safetensors headers, never the weights.

```bash
zig build run -- inspect <checkpoint-dir>
```

## Contributing

Every source file begins with the copyright header defined in [PROJECT.md](PROJECT.md). Read [AGENTS.md](AGENTS.md) before making changes.

## Credits

ncmorph builds on **Next-Concept Modeling**, developed by Hermann Kumbong, Ishan S. Khare, Iddah Mlauzi, Andrew Woen, Charles Ding, E. Kelly Buchanan, Simran Arora, Azalia Mirhoseini and Christopher Ré in the HazyResearch group at Stanford.

- Paper: [arXiv:2610.14162](https://arxiv.org/abs/2610.14162)
- Reference code and checkpoints: [github.com/HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT License) and the [NCM collection on Hugging Face](https://huggingface.co/collections/hazyresearch/ncm)

Thank you to the NCM authors. Their research, and their decision to release the paper, code and weights openly, made ncmorph possible.

## License

ncmorph is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). Code derived from NCM keeps the notices required by its MIT License.
