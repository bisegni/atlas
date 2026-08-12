#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
model=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
llama_source=${LLAMA_CPP_SOURCE:-/private/tmp/atlas-llama-cpp}
prompt=${1:-'write a simple c++ hello world function'}
max_tokens=${2:-512}
timeout_seconds=${ATLAS_LLAMA_RAW_TOKEN_TIMEOUT_SECONDS:-180}
capture_index=${ATLAS_LLAMA_RAW_TOKEN_CAPTURE_INDEX:-}
kv_cache_type=${ATLAS_GEMMA4_RAW_TOKEN_KV_CACHE_TYPE:-f32}
q4_attention_mode=${ATLAS_GEMMA4_RAW_TOKEN_ATTENTION_MODE:-flash16}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/gemma4-raw-token-oracle/$stamp"
[[ -f "$model" && -f "$llama_source/examples/eval-callback/eval-callback.cpp" ]] || exit 2
if [[ "$kv_cache_type" != "f32" ]]; then
  echo "[raw-token] only F32 KV is a llama.cpp cross-engine oracle; llama.cpp and Atlas use different Q4-KV quantization semantics. Use the Resident Flash16-vs-Legacy test for Q4 kernel parity." >&2
  exit 2
fi
mkdir -p "$artifact_dir"

raw_prompt=$' <|turn>user\n'
raw_prompt+="$prompt"
raw_prompt+=$'<turn|>\n<|turn>model\n'
printf '%s' "$raw_prompt" > "$artifact_dir/raw-prompt.txt"

echo "[raw-token] capturing Atlas Resident $kv_cache_type stream ($q4_attention_mode)..." >&2
cargo run --release -p atlas-cli -- generate --model gemma4-e2b-q4_0 --chat \
  --prompt "$prompt" --max-new-tokens "$max_tokens" --greedy \
  --kv-cache-type "$kv_cache_type" --q4-attention-mode "$q4_attention_mode" --json > "$artifact_dir/atlas.json"

cp scripts/gemma4-llama-layer-oracle.cpp "$llama_source/examples/eval-callback/eval-callback.cpp"
cmake -S "$llama_source" -B "$llama_source/build-atlas-layer-oracle" -DGGML_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF >/dev/null
cmake --build "$llama_source/build-atlas-layer-oracle" --target llama-eval-callback -j2 >/dev/null
oracle="$llama_source/build-atlas-layer-oracle/bin/llama-eval-callback"

echo "[raw-token] capturing $max_tokens raw llama.cpp selections..." >&2
run_oracle() {
  if [[ -n "$capture_index" ]]; then
    env ATLAS_LLAMA_ORACLE_CAPTURE_LAYERS=1 \
      "ATLAS_LLAMA_ORACLE_CAPTURE_TOKEN_INDEX=$capture_index" \
      "$oracle" -m "$model" -ngl 99 --cache-type-k "$kv_cache_type" --cache-type-v "$kv_cache_type" \
      --ctx-size $((max_tokens + 64)) --batch-size 64 --ubatch-size 64 \
      --n-predict "$max_tokens" --prompt "$raw_prompt"
  else
    "$oracle" -m "$model" -ngl 99 --cache-type-k "$kv_cache_type" --cache-type-v "$kv_cache_type" \
      --ctx-size $((max_tokens + 64)) --batch-size 64 --ubatch-size 64 \
      --n-predict "$max_tokens" --prompt "$raw_prompt"
  fi
}
run_oracle > "$artifact_dir/llama.jsonl" 2> "$artifact_dir/llama-stderr.txt" &
oracle_pid=$!
( sleep "$timeout_seconds"; kill -0 "$oracle_pid" 2>/dev/null && kill "$oracle_pid" 2>/dev/null || true ) &
watchdog_pid=$!
set +e
wait "$oracle_pid"
oracle_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
(( oracle_status == 0 )) || { echo "raw llama oracle failed or timed out" >&2; exit "$oracle_status"; }

jq -s '[.[] | select(.event == "generated_token") | .token_id]' "$artifact_dir/llama.jsonl" > "$artifact_dir/llama-token-ids.json"
jq -n --slurpfile atlas "$artifact_dir/atlas.json" --slurpfile llama "$artifact_dir/llama-token-ids.json" '
  ($atlas[0].generated_token_ids) as $atlas_ids | $llama[0] as $llama_ids
  | {first_divergent_generated_token: ([range(0; ([$atlas_ids|length, $llama_ids|length] | min)) | select($atlas_ids[.] != $llama_ids[.])] | first // (if ($atlas_ids|length) == ($llama_ids|length) then null else ([$atlas_ids|length, $llama_ids|length] | min) end)),
     atlas_tokens: ($atlas_ids|length), llama_tokens: ($llama_ids|length), atlas_finish: $atlas[0].finish_reason}' \
  > "$artifact_dir/result.json"
jq . "$artifact_dir/result.json"
echo "artifacts: $artifact_dir"
