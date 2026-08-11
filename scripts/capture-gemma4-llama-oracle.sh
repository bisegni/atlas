#!/usr/bin/env bash
# Capture a token-identical llama.cpp oracle through the Resident LegacyFused
# path, then record whether the faster Flash16 path has exact token parity.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
llama_simple=${LLAMA_SIMPLE:-llama-simple}
llama_tokenize=${LLAMA_TOKENIZE:-llama-tokenize}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12a-llama-oracle/${stamp}"
flash_mismatch=0

if [[ ! -f "$fixture" ]]; then
    echo "missing Gemma fixture: $fixture" >&2
    exit 2
fi
if ! command -v "$llama_simple" >/dev/null; then
    echo "missing llama.cpp binary: $llama_simple (set LLAMA_SIMPLE to its path)" >&2
    exit 2
fi
if ! command -v "$llama_tokenize" >/dev/null; then
    echo "missing llama.cpp tokenizer binary: $llama_tokenize (set LLAMA_TOKENIZE to its path)" >&2
    exit 2
fi

mkdir -p "$artifact_dir"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"
"$llama_simple" --version | tee "$artifact_dir/llama-version.txt"

# The embedded Hugging Face tokenizer uses Metaspace::Always. Retain the
# leading ASCII space and suppress llama.cpp's automatic BOS token so both
# engines consume identical prompt IDs.
capture_case() {
    local label=$1
    local atlas_prompt=$2
    local raw_prompt=$3
    local max_tokens=$4
    local expected_finish=$5
    local fixture_path=$6
    local case_dir="$artifact_dir/$label"
    local oracle_output oracle_visible atlas_prompt_ids llama_prompt_ids
    local legacy_visible legacy_finish flash_visible flash_finish
    local legacy_tokens flash_tokens first_divergence model_sha256 llama_version

    mkdir -p "$case_dir"
    printf '%s' "$raw_prompt" > "$case_dir/raw-prompt.txt"

    echo "[$label] verifying prompt token identity with llama.cpp..."
    "$llama_tokenize" \
        -m "$fixture" \
        --no-bos \
        --ids \
        --log-disable \
        --prompt "$raw_prompt" > "$case_dir/llama-prompt-token-ids.json"

    echo "[$label] capturing raw greedy llama.cpp completion..."
    "$llama_simple" \
        -m "$fixture" \
        --override-kv tokenizer.ggml.add_bos_token=bool:false \
        --cache-type-k q4_0 \
        --cache-type-v q4_0 \
        -n "$max_tokens" \
        --temp 0 \
        --top-k 1 \
        --special \
        --log-disable \
        "$raw_prompt" > "$case_dir/llama-completion.txt"

    echo "[$label] capturing Resident LegacyFused oracle..."
    cargo run --release -p atlas-cli -- generate \
        --model "$model_id" \
        --chat \
        --prompt "$atlas_prompt" \
        --max-new-tokens "$max_tokens" \
        --greedy \
        --q4-attention-mode legacy_fused \
        --json > "$case_dir/atlas-legacy.json"

    echo "[$label] capturing Resident Flash16 candidate..."
    cargo run --release -p atlas-cli -- generate \
        --model "$model_id" \
        --chat \
        --prompt "$atlas_prompt" \
        --max-new-tokens "$max_tokens" \
        --greedy \
        --q4-attention-mode flash16 \
        --json > "$case_dir/atlas-flash16.json"

    atlas_prompt_ids=$(jq -c '.prompt_token_ids' "$case_dir/atlas-legacy.json")
    llama_prompt_ids=$(jq -c . "$case_dir/llama-prompt-token-ids.json")
    if [[ "$atlas_prompt_ids" != "$llama_prompt_ids" ]]; then
        echo "[$label] Atlas and llama.cpp tokenized the prompt differently; fixture not changed" >&2
        printf 'Atlas:     %s\nllama.cpp: %s\n' "$atlas_prompt_ids" "$llama_prompt_ids" >&2
        exit 1
    fi

    oracle_output=$(<"$case_dir/llama-completion.txt")
    if [[ "$expected_finish" == eos ]]; then
        if [[ "$oracle_output" != *'<turn|>' ]]; then
            echo "[$label] llama.cpp did not terminate with <turn|>; refusing to promote a fixture" >&2
            exit 1
        fi
        oracle_visible=${oracle_output%'<turn|>'}
    else
        if [[ "$oracle_output" == *'<turn|>'* ]]; then
            echo "[$label] llama.cpp terminated early; expected a fixed max-token oracle" >&2
            exit 1
        fi
        oracle_visible=$oracle_output
    fi

    legacy_visible=$(jq -r '.visible_text' "$case_dir/atlas-legacy.json")
    legacy_finish=$(jq -r '.finish_reason' "$case_dir/atlas-legacy.json")
    if [[ "$legacy_finish" != "$expected_finish" ]]; then
        echo "[$label] LegacyFused finish mismatch: expected $expected_finish, got $legacy_finish" >&2
        exit 1
    fi
    if [[ "$legacy_visible" != "$oracle_visible" ]]; then
        echo "[$label] LegacyFused visible completion differs from llama.cpp; fixture not changed" >&2
        diff -u <(printf '%s' "$oracle_visible") <(printf '%s' "$legacy_visible") || true
        exit 1
    fi
    if ! jq -e '
        .executor == "resident"
        and .kv_cache_type == "q4_0"
        and .q4_attention_mode == "legacy_fused"
        and .attention_kernel == "attention_decode_fused_gemma4_simd_q4_0"
    ' "$case_dir/atlas-legacy.json" >/dev/null; then
        echo "[$label] LegacyFused capture did not use the required Resident Q4-KV path" >&2
        exit 1
    fi

    legacy_tokens=$(jq -c '.generated_token_ids' "$case_dir/atlas-legacy.json")
    flash_tokens=$(jq -c '.generated_token_ids' "$case_dir/atlas-flash16.json")
    flash_visible=$(jq -r '.visible_text' "$case_dir/atlas-flash16.json")
    flash_finish=$(jq -r '.finish_reason' "$case_dir/atlas-flash16.json")
    first_divergence=$(jq -n \
        --argjson legacy "$legacy_tokens" \
        --argjson flash "$flash_tokens" \
        '[range(0; ([$legacy | length, $flash | length] | min))
          | select($legacy[.] != $flash[.])]
         | first // (if ($legacy | length) == ($flash | length)
                     then null else ([$legacy | length, $flash | length] | min) end)')
    jq -n \
        --arg label "$label" \
        --argjson first_divergence "$first_divergence" \
        --slurpfile legacy "$case_dir/atlas-legacy.json" \
        --slurpfile flash16 "$case_dir/atlas-flash16.json" \
        '{label: $label, first_divergent_generated_token: $first_divergence,
          exact_token_parity: ($legacy[0].generated_token_ids == $flash16[0].generated_token_ids),
          exact_finish_parity: ($legacy[0].finish_reason == $flash16[0].finish_reason),
          legacy: $legacy[0], flash16: $flash16[0]}' > "$case_dir/flash16-parity.json"

    model_sha256=$(awk '{print $1}' "$artifact_dir/fixture-sha256.txt")
    llama_version=$(<"$artifact_dir/llama-version.txt")
    jq -n \
        --arg fixture_sha256 "$model_sha256" \
        --arg llama_version "$llama_version" \
        --arg expected_finish "$expected_finish" \
        --arg oracle_visible "$oracle_visible" \
        --argjson max_tokens "$max_tokens" \
        --argjson first_divergence "$first_divergence" \
        --slurpfile legacy "$case_dir/atlas-legacy.json" \
        --slurpfile flash16 "$case_dir/atlas-flash16.json" \
        '{fixture_sha256: $fixture_sha256, model_id: $legacy[0].model_id,
          prompt: $legacy[0].prompt, prompt_token_ids: $legacy[0].prompt_token_ids,
          generated_token_ids: $legacy[0].generated_token_ids,
          finish_reason: $legacy[0].finish_reason, visible_text: $legacy[0].visible_text,
          recorded_by: "atlas-cli generate --chat --greedy --json --q4-attention-mode legacy_fused, matched against llama.cpp",
          external_oracle: {status: "verified", engine: "llama.cpp", version: $llama_version,
            protocol: "leading metaspace plus --no-bos", visible_text: $oracle_visible,
            finish_reason: (if $expected_finish == "eos" then "<turn|>" else "max_tokens" end),
            max_new_tokens: $max_tokens},
          flash16_parity: {exact_token_parity: ($legacy[0].generated_token_ids == $flash16[0].generated_token_ids),
            exact_finish_parity: ($legacy[0].finish_reason == $flash16[0].finish_reason),
            first_divergent_generated_token: $first_divergence}}' \
        > "$fixture_path"

    if [[ "$legacy_tokens" != "$flash_tokens" || "$legacy_finish" != "$flash_finish" || "$legacy_visible" != "$flash_visible" ]]; then
        echo "[$label] Flash16 does not have exact LegacyFused parity; recorded diagnostic artifact" >&2
        flash_mismatch=1
    else
        echo "[$label] Flash16 has exact LegacyFused token, finish, and visible-text parity"
    fi
}

capture_case \
    canonical \
    hi \
    $' <|turn>user\nhi<turn|>\n<|turn>model\n' \
    32 \
    eos \
    fixtures/gemma4-e2b-resident-canonical.json
capture_case \
    long-cpp \
    'write an hello world main c++ function' \
    $' <|turn>user\nwrite an hello world main c++ function<turn|>\n<|turn>model\n' \
    64 \
    max_tokens \
    fixtures/gemma4-e2b-resident-long-cpp.json

echo "Oracle artifacts: $artifact_dir"
if (( flash_mismatch )); then
    echo "Flash16 remains the production default but is not accepted until this parity artifact is clean" >&2
    exit 1
fi
