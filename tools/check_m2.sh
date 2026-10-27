#
# Ko, (Baczynski-Bruce) 2026; Apache 2.0
#
# Generate the pinned tiny checkpoints and run the transplant gate.
set -eu

python_command=${NCMORPH_GOLDEN_PYTHON:-python3}
"$python_command" tools/golden_helper.py --output-dir zig-out/golden
for case_name in llama_default llama_llama3 qwen2 qwen3 mistral qwen3_attention_bias llama_default_bf16; do
    if [ ! -f "zig-out/golden/$case_name/config.json" ]; then
        printf 'Missing generated checkpoint: %s\n' "$case_name" >&2
        exit 1
    fi
done
cache_dir=$(mktemp -d "${TMPDIR:-/tmp}/ncmorph-m2-cache.XXXXXX")
trap 'rm -rf "$cache_dir"' EXIT
zig build test --cache-dir "$cache_dir" --summary all
