# Procedure: converter, reference comparisons and their approvals

This file says how the steps that touch the released pickle or the pinned NCM reference code are run, who runs them, and what each run must check. The steps are in [ROADMAP.md](../../ROADMAP.md): V1 (the converter), V3 and V4 (comparisons with the reference VQ), and N7 (reference masks and positions). The rules they follow are decisions D41 and D52 in [PROJECT.md](../../PROJECT.md).

## Why these steps need an approval

- The released `tokenizer.pt` is a PyTorch pickle. Loading it runs code that the file contains, so the load happens only in a throwaway environment, once, under D41.
- The pinned NCM reference is third-party Python. Running it is a comparison step, and the owner approves each session that runs it.

## What happened in the session that wrote this file (2026-10-26)

An attempt to construct the reference's VQ module, in order to list its state-dict keys, was refused by the automatic permission check, which treats running the reference code as external code. The attempt was not repeated, and no other route to run that code was tried. The keys were derived instead by reading the pinned source, with no code run: [converter/wmt14_tokenizer_keys.md](wmt14_tokenizer_keys.md). That derivation relies on PyTorch's documented rules, because no PyTorch source is installed locally. Step V1 therefore confirms the key set against the real file.

## The released pickle: V1

Inputs. The owner downloads the released file into a folder outside the repository:

- Repository: `hazyresearch/ncm-checkpoints`, at revision `d342ce8aaea58830eef3b8ad473664673e41f692`.
- Path: `wmt14/tokenizer.pt`.
- Size: 645,790,351 bytes.
- SHA-256: `74f04176a3cd985e4142761bd7464ea6edb44dc0025c40302b5e45ea0d865702` (the LFS hash in the repository's tree metadata, read 2026-10-26).

Environment. A throwaway environment with no network access and no credentials, on Python 3.13, with the versions pinned in PROJECT.md. The owner chooses the form of the environment, for example a separate user account or a container with networking off.

Steps, in order:

1. Check the file's size and SHA-256 against the values above. Stop on any mismatch, and load nothing.
2. Load the file with `torch.load(path, weights_only=False, map_location='cpu')`, the same call the reference uses (`tokenization/vqvae_tokenizer.py` line 18). Take the `'model'` mapping.
3. Check the mapping against the derivation. It must hold exactly the 96 keys listed in `wmt14_tokenizer_keys.md` (63 parameter entries with 144,002,816 scalars, and 33 persistent buffer entries with 17,391,616 scalars), with no prefix. Each shape and dtype is checked the same way. Load the mapping into ncmorph's own VQ module (step V3) with `strict=True`. Stop on any missing or unexpected key. Never rename a key to make it fit.
4. Keep each tensor in its stored dtype (D59). Write `tokenizer.safetensors` with the writer of step W1, and write `ncm.json` with the file's size, SHA-256 and the stored dtypes.
5. Read `tokenizer.safetensors` back with the Zig reader, and compare every tensor's bytes with the tensor loaded in step 2.
6. Record the run in [checks.md](../checks.md), section 15: the date, the file's size and SHA-256, the output's SHA-256, and the result of each check.

## Comparisons with the pinned reference: N7, V3 and V4

These import or run the pinned NCM reference (`models/vqvae.py`, `models/next_concept.py`). The rules:

- The owner approves each session that runs them.
- Do not call `VQVAE.reconstruct`. It is broken at `models/vqvae.py` line 116: it unpacks three values from `self.vq`, which returns five (`models/quant.py` lines 543 to 549). Use `VQVAE.forward`, which returns the logits, the VQ loss, the utilization values and the perplexity, and `VQVAE.encode`.
- Set torch to the pinned version before the run. The reference's `requirements.txt` leaves torch unpinned (`research/reference/vq_forward_semantics.md`, section 10).
- Record each run's commit, fixed input ids, seed and outputs in [checks.md](../checks.md).

## Reference details that these procedures depend on

- The reference loads its checkpoints with `weights_only=False` (`scripts/eval_shard.py` lines 52 and 62; `tokenization/vqvae_tokenizer.py` line 18).
- `models/quant.py` line 548 returns the literal 0.1 where a perplexity would be expected. ncmorph does not read that value.
- The `codebook_hit` counters are never reset (`models/quant.py` lines 522 and 534 to 541). Utilization is therefore counted fresh on the validation split (D37).
- The dead-code restart threshold defaults to 0.0 (`models/quant.py` line 244), so restart does nothing unless a setting raises it.

## Commands

Run from the repository root. Each block is one command. The converter's own command line is defined by step V1, and it is added to this file when V1 is done.

Check the interpreter version. It must report Python 3.13:

```bash
python3 -V
```

Create the golden environment (after step B0 creates `tools/requirements-golden.txt`):

```bash
python3 -m venv .venv-golden
```

Install the pinned packages into it:

```bash
.venv-golden/bin/pip install -r tools/requirements-golden.txt
```
