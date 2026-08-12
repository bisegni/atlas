#!/usr/bin/env bash
# Compare one Flash16 Resident stream directly with llama.cpp, without running
# LegacyFused. llama.cpp b10360 cannot safely consume Gemma's raw turn-marker
# prompt in --no-conversation mode, so this uses its supported Jinja single
# turn. Its rendered completion must not be re-tokenized and presented as its
# generation token stream: formatting and special-token treatment can change
# the IDs. This is therefore a bounded visible-behavior check only.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt=${1:-'write a simple c++ hello world function'}
# Token 10 is the known candidate divergence; 32 leaves room to verify it
# while keeping an external-oracle diagnostic bounded.
max_tokens=${2:-32}
timeout_seconds=${ATLAS_LLAMA_DIAG_TIMEOUT_SECONDS:-120}
llama_cli=${LLAMA_CLI:-llama-cli}
llama_tokenize=${LLAMA_TOKENIZE:-llama-tokenize}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/flash16-llama-diagnosis/${stamp}"

[[ -f "$model" ]] || { echo "missing Gemma fixture: $model" >&2; exit 2; }
command -v "$llama_cli" >/dev/null || { echo "missing llama-cli" >&2; exit 2; }
command -v "$llama_tokenize" >/dev/null || { echo "missing llama-tokenize" >&2; exit 2; }
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || { echo "ATLAS_LLAMA_DIAG_TIMEOUT_SECONDS must be a positive integer" >&2; exit 2; }
mkdir -p "$artifact_dir"

# llama-tokenize needs this leading Metaspace sentinel to recognize the first
# turn marker as special; it yields the same IDs as Atlas's visible prompt.
raw_prompt=$' <|turn>user\n'
raw_prompt+="$prompt"
raw_prompt+=$'<turn|>\n<|turn>model\n'
printf '%s' "$raw_prompt" > "$artifact_dir/raw-prompt.txt"

echo "[diagnose] capturing at most $max_tokens llama tokens (timeout: ${timeout_seconds}s)..." >&2
"$llama_cli" -m "$model" \
  --override-kv tokenizer.ggml.add_bos_token=bool:false \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --conversation --single-turn --simple-io --no-display-prompt --special \
  --reasoning off --temp 0 --top-k 1 -n "$max_tokens" --log-disable --prompt "$prompt" \
  > "$artifact_dir/llama-completion.txt" 2> "$artifact_dir/llama-stderr.txt" &
llama_pid=$!
(
  sleep "$timeout_seconds"
  if kill -0 "$llama_pid" 2>/dev/null; then
    echo "llama-cli exceeded ${timeout_seconds}s; refusing an incomplete oracle" >&2
    kill "$llama_pid" 2>/dev/null || true
  fi
) &
watchdog_pid=$!
set +e
wait "$llama_pid"
llama_status=$?
set -e
kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
if (( llama_status != 0 )); then
  echo "llama-cli failed or timed out (status $llama_status); inspect $artifact_dir/llama-stderr.txt" >&2
  exit "$llama_status"
fi

# llama-cli writes a banner, echoed prompt, optional hidden thinking, and
# timing footer to stdout. Keep only its visible assistant completion.
llama_raw=$(<"$artifact_dir/llama-completion.txt")
if [[ "$llama_raw" != *"> $prompt"* ]]; then
  echo "llama-cli did not emit the expected single-turn prompt marker" >&2
  exit 1
fi
input_marker=$"> $prompt"$'\n'
llama_visible=${llama_raw#*"$input_marker"}
thinking_end=$'[End thinking]\n'
if [[ "$llama_visible" == *"$thinking_end"* ]]; then
  llama_visible=${llama_visible#*"$thinking_end"}
fi
llama_visible=${llama_visible#$'\n'}
if [[ "$llama_visible" == *"[Start thinking]"* || "$llama_visible" == *"[End thinking]"* ]]; then
  echo "llama-cli ignored --reasoning off; refusing to compare hidden reasoning as visible output" >&2
  exit 1
fi
timing_marker=$'\n\n[ Prompt:'
llama_visible=${llama_visible%%"$timing_marker"*}
printf '%s' "$llama_visible" > "$artifact_dir/llama-visible-completion.txt"

"$llama_tokenize" -m "$model" --no-bos --ids --log-disable --prompt "$raw_prompt" \
  > "$artifact_dir/llama-prompt-token-ids.json"
"$llama_tokenize" -m "$model" --no-bos --ids --log-disable \
  --prompt "$llama_visible" \
  > "$artifact_dir/llama-generated-token-ids.json"

cargo run --release -p atlas-cli -- generate --model gemma4-e2b-q4_0 --chat \
  --prompt "$prompt" --max-new-tokens "$max_tokens" --greedy \
  --q4-attention-mode flash16 --json > "$artifact_dir/atlas-flash16.json"

jq -n \
  --slurpfile atlas "$artifact_dir/atlas-flash16.json" \
  --slurpfile llama_prompt "$artifact_dir/llama-prompt-token-ids.json" \
  --slurpfile llama_generated "$artifact_dir/llama-generated-token-ids.json" \
  '{comparison_mode: "llama_jinja_single_turn",
    atlas_raw_prompt_token_ids_match_tokenizer: ($atlas[0].prompt_token_ids == $llama_prompt[0]),
    generated_token_parity: "unavailable_from_rendered_jinja_completion",
    atlas: $atlas[0], llama_rendered_completion_token_ids: $llama_generated[0]}' \
  > "$artifact_dir/result.json"

jq '{comparison_mode, atlas_raw_prompt_token_ids_match_tokenizer, generated_token_parity,
     atlas_kernel: .atlas.attention_kernel, atlas_finish: .atlas.finish_reason,
     atlas_tokens: (.atlas.generated_token_ids | length),
     llama_rendered_tokens: (.llama_rendered_completion_token_ids | length)}' "$artifact_dir/result.json"
echo "artifacts: $artifact_dir"
