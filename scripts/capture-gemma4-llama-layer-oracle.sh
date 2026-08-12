#!/usr/bin/env bash
# Build and run a compact `l_out-N` Gemma 4 oracle against llama.cpp b10360.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
llama_source=${LLAMA_CPP_SOURCE:-/private/tmp/atlas-llama-cpp}
timeout_seconds=${ATLAS_LLAMA_LAYER_ORACLE_TIMEOUT_SECONDS:-120}
source_revision=48d22e295
atlas_trace=${1:-}
prompt=${2:-$' <|turn>user\nwrite a simple c++ hello world function<turn|>\n<|turn>model\nHere is the simplest "Hello, World!" program'}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/gemma4-llama-layer-oracle/$stamp"

[[ -f "$model" ]] || { echo "missing Gemma fixture: $model" >&2; exit 2; }
[[ -n "$atlas_trace" ]] || {
    echo "usage: $0 <atlas-flash16-diagnosis.jsonl> [raw-context-prompt]" >&2
    exit 2
}
[[ -f "$llama_source/examples/eval-callback/eval-callback.cpp" ]] || {
    echo "missing llama.cpp source at $llama_source; clone b10360 first:" >&2
    echo "  git clone https://github.com/ggml-org/llama.cpp.git /private/tmp/atlas-llama-cpp" >&2
    echo "  git -C /private/tmp/atlas-llama-cpp checkout $source_revision" >&2
    exit 2
}
[[ "$(git -C "$llama_source" rev-parse --short=9 HEAD)" == "$source_revision" ]] || {
    echo "llama.cpp source is not $source_revision" >&2
    exit 2
}
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "ATLAS_LLAMA_LAYER_ORACLE_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
}

mkdir -p "$artifact_dir"
printf '%s' "$prompt" > "$artifact_dir/raw-prompt.txt"
shasum -a 256 "$model" > "$artifact_dir/model-sha256.txt"

# The source checkout is disposable. The installed llama.cpp binary is never
# modified; only the existing example target in this checkout is replaced.
cp scripts/gemma4-llama-layer-oracle.cpp "$llama_source/examples/eval-callback/eval-callback.cpp"
cmake -S "$llama_source" -B "$llama_source/build-atlas-layer-oracle" \
    -DLLAMA_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build "$llama_source/build-atlas-layer-oracle" --target llama-eval-callback -j2

oracle="$llama_source/build-atlas-layer-oracle/bin/llama-eval-callback"
[[ -x "$oracle" ]] || oracle="$llama_source/build-atlas-layer-oracle/examples/eval-callback/llama-eval-callback"
[[ -x "$oracle" ]] || { echo "layer oracle binary was not built" >&2; exit 1; }

echo "[layer-oracle] evaluating one fixed 27-token context (timeout: ${timeout_seconds}s)..." >&2
ATLAS_LLAMA_ORACLE_CAPTURE_LAYERS=1 "$oracle" -m "$model" -ngl 99 --cache-type-k f32 --cache-type-v f32 \
    --n-predict 0 \
    --ctx-size 64 --batch-size 64 --ubatch-size 64 --prompt "$prompt" \
    > "$artifact_dir/llama-layer-states.jsonl" 2> "$artifact_dir/llama-stderr.txt" &
oracle_pid=$!
(
    sleep "$timeout_seconds"
    if kill -0 "$oracle_pid" 2>/dev/null; then
        echo "layer oracle exceeded ${timeout_seconds}s; refusing incomplete output" >&2
        kill "$oracle_pid" 2>/dev/null || true
    fi
) &
watchdog_pid=$!
set +e
wait "$oracle_pid"
oracle_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
if (( oracle_status != 0 )); then
    echo "layer oracle failed (status $oracle_status); inspect $artifact_dir/llama-stderr.txt" >&2
    exit "$oracle_status"
fi

jq -s '
  {prompt_token_ids: ([.[] | select(.event == "prompt_tokens") | .token_ids] | first),
   logits: ([.[] | select(.event == "logits")] | first),
   layer_states: [.[] | select(.event == "layer_state")]}' \
    "$artifact_dir/llama-layer-states.jsonl" > "$artifact_dir/summary.json"
jq -e '
  (.prompt_token_ids | length) == 27
  and (.layer_states | length) == 35
  and .logits != null
  and ([.layer_states[] | .non_finite] | add) == 0
' "$artifact_dir/summary.json" >/dev/null || {
    echo "layer oracle did not emit the expected 27 prompt tokens and 35 finite layers" >&2
    exit 1
}
jq '{prompt_tokens: (.prompt_token_ids | length), layers: (.layer_states | length),
     non_finite: ([.layer_states[] | .non_finite] | add),
     logits,
     final_layer: .layer_states[-1]}' "$artifact_dir/summary.json"
if [[ -n "$atlas_trace" ]]; then
    [[ -f "$atlas_trace" ]] || { echo "missing Atlas layer trace: $atlas_trace" >&2; exit 2; }
    jq -n --slurpfile llama "$artifact_dir/summary.json" --slurpfile atlas "$atlas_trace" '
      ($atlas | map(select(.event == "flash16_token"))) as $atlas_tokens
      | ($atlas_tokens | map(select(.token_index == 10)) | first) as $atlas_token
      | ([236743,105,2364,107,5986,496,3606,505,2124,29104,1902,1292,106,107,105,4368,107]
         + ($atlas_tokens | map(select(.token_index < 10)) | sort_by(.token_index) | map(.token_id))) as $atlas_context_ids
      | ($llama[0].layer_states
         | map({key: (.name | sub("^l_out-"; "")), value: .})
         | from_entries) as $llama_layers
      | ($atlas_token.layer_states | map({key: (.layer_index | tostring), value: .}) | from_entries) as $atlas_layers
      | [range(0; 35) | (. | tostring) as $layer_key | ($atlas_layers[$layer_key] as $a | $llama_layers[$layer_key] as $l |
           {layer_index: ., atlas: {l2_norm: $a.l2_norm, max_abs: $a.max_abs, non_finite: $a.non_finite},
            llama: {l2_norm: $l.l2_norm, max_abs: $l.max_abs, non_finite: $l.non_finite},
            l2_relative_error: (($a.l2_norm - $l.l2_norm) / ([1.0, ($a.l2_norm | abs), ($l.l2_norm | abs)] | max) | abs),
            max_abs_relative_error: (($a.max_abs - $l.max_abs) / ([1.0, ($a.max_abs | abs), ($l.max_abs | abs)] | max) | abs)})] as $layers
      | {atlas_prompt_and_prefix_token_ids: $atlas_context_ids,
         llama_prompt_token_ids: $llama[0].prompt_token_ids,
         prompt_ids_match: ($atlas_context_ids == $llama[0].prompt_token_ids),
         atlas_logits: {top_token_id: $atlas_token.top_token_id, top_logit: $atlas_token.top_logit,
                        runner_up_token_id: $atlas_token.runner_up_token_id, runner_up_logit: $atlas_token.runner_up_logit},
         llama_logits: $llama[0].logits,
         greedy_token_matches: ($atlas_token.top_token_id == $llama[0].logits.top_token_id),
         layers: $layers,
         first_layer_over_1e_4: ($layers | map(select((.l2_relative_error > 0.0001) or (.max_abs_relative_error > 0.0001))) | first // null),
         first_layer_over_1e_3: ($layers | map(select((.l2_relative_error > 0.001) or (.max_abs_relative_error > 0.001))) | first // null)}
    ' > "$artifact_dir/atlas-comparison.json"
    jq '{prompt_ids_match, greedy_token_matches, atlas_logits, llama_logits,
         first_layer_over_1e_4, first_layer_over_1e_3}' "$artifact_dir/atlas-comparison.json"
fi
echo "artifacts: $artifact_dir"
