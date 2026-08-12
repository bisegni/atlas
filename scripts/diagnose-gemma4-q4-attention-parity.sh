#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

prompt=${1:-'write a simple c++ hello world function'}
max_tokens=${2:-512}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/gemma4-q4-attention-parity/$stamp"
mkdir -p "$artifact_dir"

capture() {
  local mode=$1
  local output=$2
  echo "[q4-parity] capturing Resident Q4_0 $mode stream ($max_tokens tokens)..." >&2
  cargo run --release -p atlas-cli -- generate --model gemma4-e2b-q4_0 --chat \
    --prompt "$prompt" --max-new-tokens "$max_tokens" --greedy \
    --kv-cache-type q4_0 --q4-attention-mode "$mode" --json > "$output"
}

capture legacy_fused "$artifact_dir/legacy.json"
capture flash16 "$artifact_dir/flash16.json"

jq -n --slurpfile legacy "$artifact_dir/legacy.json" --slurpfile flash16 "$artifact_dir/flash16.json" '
  ($legacy[0]) as $legacy | ($flash16[0]) as $flash | $legacy.generated_token_ids as $legacy_ids | $flash.generated_token_ids as $flash_ids
  | {exact_token_parity: ($legacy_ids == $flash_ids),
     first_divergent_generated_token: ([range(0; ([$legacy_ids|length, $flash_ids|length] | min)) | select($legacy_ids[.] != $flash_ids[.])] | first // (if ($legacy_ids|length) == ($flash_ids|length) then null else ([$legacy_ids|length, $flash_ids|length] | min) end)),
     legacy: {kernel: $legacy.attention_kernel, tokens: ($legacy_ids|length), finish: $legacy.finish_reason},
     flash16: {kernel: $flash.attention_kernel, tokens: ($flash_ids|length), finish: $flash.finish_reason}}' \
  > "$artifact_dir/result.json"
jq . "$artifact_dir/result.json"
echo "artifacts: $artifact_dir"
jq -e '.exact_token_parity' "$artifact_dir/result.json" >/dev/null
