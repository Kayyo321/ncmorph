# AGENTS.md

Instructions for AI coding agents working in this repository. Read this file and PROJECT.md before you change anything.

## Read first

1. [README.md](README.md): what ncmorph is and how to build it.
2. [PROJECT.md](PROJECT.md): scope, architecture, decisions and open questions. It is the source of truth for what this project is.
3. This file.

If a task conflicts with PROJECT.md, stop and ask. Do not change the scope on your own.

## Glossary

- **NCM**: Next-Concept Modeling (arXiv:2610.14162, HazyResearch). Never expand it as anything else, such as "Nearest Class Mean".
- **Source model**: the open-weight decoder-only LLM that ncmorph converts.
- **Level**: one scale of the multiscale residual VQ hierarchy. Positions within a level are generated in parallel.
- **Prefix**: conditioning positions that the generator sees before the target levels.
- **Reference**: the pinned HazyResearch/ncm commit recorded in PROJECT.md.

## Sources of truth

Before you state a fact (a Zig API, a file-format field, an NCM behavior, a hyperparameter or a number), check it in the first source that applies:

1. Local code you can open: this repository, the pinned Zig standard library (`zig env` prints its `std_dir`), and the reference at the pinned commit.
2. Specifications and the paper: the safetensors and GGUF specifications, and arXiv:2610.14162.
3. Your memory. Use it only for general programming ideas, and check those against step 1 or 2.

Do not rely on memory for Zig standard-library signatures, file byte layouts, tensor names or NCM hyperparameters.

If you cannot verify something, write `UNVERIFIED:` next to it in the code or in your report, and add it to "Open questions" in PROJECT.md. Do not fill the gap with a plausible guess.

## Toolchain

- Zig 0.16.0 is the target. Check the active version with `zig version`.
- Before you call any `std` function, grep the standard library at `zig env`'s `std_dir`. Do not trust remembered signatures.
- Zig 0.16 differs from older releases. These points were verified against the 0.16.0 standard library:
  - There is no `std.io` namespace. The I/O namespace is `std.Io`.
  - `std.ArrayList(T)` is unmanaged. Start from `.empty` or `initCapacity(gpa, n)`, and pass the allocator to `append`, `deinit` and `toOwnedSlice`.
  - `main` may take a `std.process.Init` (or `Init.Minimal`) parameter. Check `callMain` in `start.zig` before you write an entry point.
  - Comments are `//`, `///` and `//!` only. `/* */` is a parse error.
- Formatting: `zig fmt --check .` lists non-conforming files and exits non-zero. Run `zig fmt .` to fix them.

## Commands

Run these from the repository root.

- `zig build`: build the library and the CLI.
- `zig build test`: run all tests.
- `zig fmt --check .`: check formatting.
- `zig build run -- inspect <checkpoint-dir>`: check a local checkpoint against its config. No weights are read.

## Verification before you report

Run these and include the results in your report:

1. `zig build test`
2. `zig fmt --check .`
3. Header check. This must print nothing:

   ```bash
   for f in $(git ls-files --cached --others --exclude-standard '*.zig'); do grep -qF "Ko, (Baczynski-Bruce) 2026; Apache 2.0" "$f" || echo "missing header: $f"; done
   ```

4. Weight check. This must print nothing:

   ```bash
   git ls-files --cached --others --exclude-standard | grep -E '\.(safetensors|gguf|pt|pth|onnx|ckpt)$'
   ```

5. Isolation check. This must print nothing:

   ```bash
   grep -rniE "\bncm|ncmorph" packages --include='*.zig'
   ```

Never say that tests pass unless you ran them in this task. Quote the command and the last lines of its output. If you could not run something, say what blocked it.

## Code rules

- Every source file begins with the header defined in PROJECT.md. In `.zig` files it is written with `//` comments.
- One concern per change. Keep diffs small enough to review.
- Do not add a dependency without a PROJECT.md entry and a license check.
- Never load pickle-based weights (`.pt`, `.pth`, or `.bin` written by `torch.save`). Reject them with an error that tells the user to convert them outside ncmorph.
- If the source model has a feature that NCM cannot represent, return an error that names the feature. Do not approximate silently.
- Do not leave commented-out code, stub functions that return fake results, or placeholder numbers.
- Llama, Qwen and Mistral support is one decoder core configured by flags (PROJECT.md, "Flags by family"). Do not add a per-model code path. Add a flag and a test against transformers v5.20.2 instead.

## Coding style

- Never use camelCase, in any language in this repository.
- Use PascalCase only for constants, enums and preprocessor macros. For example: `MaxHeaderSize`, `Dtype`, `Dtype.F32`.
- Everything else uses snake_case: functions, variables, fields, parameters, modules and file names.
- Type definitions end in `_t`, for example `header_t`, `tensor_info_t` and `bf16_t`. In Zig this covers struct, union and other type aliases. Enum types are not type definitions, so they keep PascalCase (`Dtype`).
- Constants are module-level values. Local variables, including `const` locals, use snake_case.
- Zig standard-library names keep their own spelling when we call them, for example `std.mem.readInt`.

## Scope and hallucination guards

- Do not invent CLI flags, file formats, tensor names, config keys, benchmark results or speed claims. If something is not in the specification, the pinned code or a measured artifact, it does not exist yet.
- When you describe NCM behavior, cite the pinned file and symbol, for example `models/next_concept.py` (`NextConceptTransformer`).
- If a request is ambiguous, ask one specific question. If you proceed on a default, say which default you used.

## Data and weights

- Never commit weights, datasets, caches, logs or generated artifacts. `.gitignore` covers the usual paths. Check `git status` before you report.
- Golden tests generate their tiny models and reference outputs at test time with the pinned Python helper in `tools/`. Commit only text fixtures, such as configs, tensor-header JSON and expected values in JSON. Never commit `.safetensors` files; the weight check above enforces this.

## Git

- Do not commit, push, rewrite history or open pull requests unless the user asks.

## Report format

End every task with: changed files, the commands you ran with their results, what is unverified, and any new open questions.
