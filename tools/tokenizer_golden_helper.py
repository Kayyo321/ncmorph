#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
"""Generate T0 token ids with the pinned Hugging Face tokenizers package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


PinnedVersion = "0.23.3"
CorpusPath = Path(__file__).parent / "corpus" / "tokenizer_cases.jsonl"
TokenizerFiles = {
    "qwen2.5-0.5b": ("qwen2.json", "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    "qwen3": ("qwen3.json", "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "llama-3.0-8b": ("llama3.json", "NousResearch/Meta-Llama-3-8B", "315b20096dc791d381d514deb5f8bd9c8d6d3061"),
    "llama-3.2-1b": ("llama32.json", "unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da"),
    "smollm2-135m": ("smollm.json", "HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "llama-2-7b": ("llama2.json", "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"),
    "mistral-7b-v0.1": ("mistral01.json", "mistralai/Mistral-7B-v0.1", "27d67f1b5f57dc0953326b2601d68371d40ea8da"),
    "mistral-7b-v0.3": ("mistral03.json", "mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1"),
}


def read_corpus(path: Path) -> list[dict[str, str]]:
    cases = []
    seen_names = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line:
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if not isinstance(case, dict) or set(case) != {"name", "text"}:
            raise ValueError(f"{path}:{line_number} must contain exactly name and text")
        if not isinstance(case["name"], str) or not case["name"]:
            raise ValueError(f"{path}:{line_number} has an empty or invalid name")
        if case["name"] in seen_names:
            raise ValueError(f"{path}:{line_number} repeats case name {case['name']!r}")
        if not isinstance(case["text"], str):
            raise ValueError(f"{path}:{line_number} has non-text input")
        seen_names.add(case["name"])
        cases.append(case)
    if not cases:
        raise ValueError(f"corpus is empty: {path}")
    return cases


def check_version() -> None:
    try:
        actual = importlib.metadata.version("tokenizers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("T0 requires tokenizers==0.23.3; package is missing") from error
    if actual.split("+", maxsplit=1)[0] != PinnedVersion:
        raise RuntimeError(f"T0 requires tokenizers=={PinnedVersion}; found {actual}")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"T0 requires Python 3.13; found {sys.version.split()[0]}")


def make_pre_tokenizer(config: dict[str, object]):
    """Build one configured stage so T0 can expose every sequence boundary."""
    from tokenizers import Regex, pre_tokenizers

    kind = config.get("type")
    if kind == "Split":
        pattern = config["pattern"]
        if not isinstance(pattern, dict) or set(pattern) != {"Regex"}:
            raise ValueError(f"unsupported Split pattern: {pattern!r}")
        return pre_tokenizers.Split(
            Regex(pattern["Regex"]),
            behavior=config["behavior"].lower(),
            invert=config["invert"],
        )
    if kind == "ByteLevel":
        return pre_tokenizers.ByteLevel(
            add_prefix_space=config["add_prefix_space"],
            trim_offsets=config["trim_offsets"],
            use_regex=config["use_regex"],
        )
    if kind == "Digits":
        return pre_tokenizers.Digits(individual_digits=config["individual_digits"])
    if kind == "Metaspace":
        return pre_tokenizers.Metaspace(
            replacement=config["replacement"],
            prepend_scheme=config["prepend_scheme"],
            split=config["split"],
        )
    raise ValueError(f"unsupported pre-tokenizer type: {kind!r}")


def pre_tokenizer_trace(config: dict[str, object] | None, text: str) -> tuple[list[dict], list[str]]:
    """Return pieces after each configured stage and final pieces from the config."""
    if config is None:
        return [], [text]
    if config.get("type") == "Sequence":
        stages = config.get("pretokenizers")
        if not isinstance(stages, list):
            raise ValueError("Sequence.pre_tokenizers must be a list")
    else:
        stages = [config]

    current_pieces = [text]
    trace = []
    for stage_config in stages:
        if not isinstance(stage_config, dict):
            raise ValueError(f"invalid pre-tokenizer stage: {stage_config!r}")
        stage = make_pre_tokenizer(stage_config)
        current_pieces = [piece for value in current_pieces for piece, _ in stage.pre_tokenize_str(value)]
        trace.append({"type": stage_config["type"], "pieces": current_pieces})
    return trace, current_pieces


def generate(tokenizer_root: Path, output_path: Path, corpus_path: Path) -> None:
    from tokenizers import Tokenizer

    cases = read_corpus(corpus_path)
    tokenizers = {}
    case_results = {case["name"]: {} for case in cases}
    for family, (filename, repository, revision) in TokenizerFiles.items():
        tokenizer_path = tokenizer_root / filename
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"missing pinned tokenizer for {family}: {tokenizer_path}; "
                f"expected {repository} at {revision}"
            )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        for case in cases:
            normalized_text = (
                tokenizer.normalizer.normalize_str(case["text"])
                if tokenizer.normalizer is not None
                else case["text"]
            )
            stage_trace, traced_pieces = pre_tokenizer_trace(
                tokenizer_config.get("pre_tokenizer"),
                normalized_text,
            )
            if tokenizer.pre_tokenizer is None:
                pretokenized_pieces = [normalized_text]
            else:
                pretokenized_pieces = [
                    piece for piece, _ in tokenizer.pre_tokenizer.pre_tokenize_str(normalized_text)
                ]
            if traced_pieces != pretokenized_pieces:
                raise ValueError(
                    f"pre-tokenizer stage trace disagrees with tokenizers for {family} "
                    f"on corpus case {case['name']!r}: {traced_pieces!r} != {pretokenized_pieces!r}"
                )

            encoding = tokenizer.encode(case["text"])
            case_results[case["name"]][family] = {
                "ids": encoding.ids,
                "tokens": encoding.tokens,
                "normalized_text": normalized_text,
                "pretokenizer_stages": stage_trace,
                "pretokenized_pieces": pretokenized_pieces,
                "decoded_text": tokenizer.decode(encoding.ids, skip_special_tokens=True),
                "decoded_text_with_special_tokens": tokenizer.decode(
                    encoding.ids, skip_special_tokens=False
                ),
            }
        tokenizers[family] = {
            "repository": repository,
            "revision": revision,
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        }

    result = {
        "tokenizers_version": PinnedVersion,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "cases": [
            {
                "name": case["name"],
                "text": case["text"],
                "ids": {
                    family: outputs["ids"]
                    for family, outputs in case_results[case["name"]].items()
                },
                "tokenizer_outputs": {
                    family: {key: value for key, value in outputs.items() if key != "ids"}
                    for family, outputs in case_results[case["name"]].items()
                },
            }
            for case in cases
        ],
        "tokenizers": tokenizers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=CorpusPath)
    args = parser.parse_args()
    try:
        check_version()
        generate(args.tokenizer_root, args.output, args.corpus)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
