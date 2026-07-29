#!/usr/bin/env bash
# Fast, non-promotional localization of Q4 vocabulary-table divergence.
# It compares mixed Q4/Q6 Resident execution with one Q4 table at a time.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain why token embeddings and the output head must preserve greedy-token selection.'
runs=2
decode_tokens=64
max_context=256
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12a-q4-vocabulary-diagnosis/${stamp}"
mkdir -p "$artifact_dir"

if [[ ! -f "$fixture" ]]; then
    echo "missing Gemma fixture: $fixture" >&2
    exit 2
fi

run_mode() {
    local label=$1
    shift
    local mode_dir="$artifact_dir/$label"
    local run
    mkdir -p "$mode_dir"
    echo "Running ${runs} short Resident windows for ${label}..."
    for run in $(seq 1 "$runs"); do
        env "$@" cargo run --release -p atlas-cli -- benchmark \
            --model "$model_id" --kv-cache-type q4_0 --prompt "$prompt" \
            --warmup-decode-tokens 0 --decode-tokens "$decode_tokens" \
            --max-context "$max_context" \
            > "$mode_dir/run-${run}.json" 2> "$mode_dir/run-${run}.log"
    done
    jq -s --arg label "$label" '
      . as $records
      | {
          label: $label,
          records: $records,
          checks: {
            resident: all($records[]; .executor == "resident"),
            deterministic_stream: (($records | map(.generated_token_sha256) | unique | length) == 1),
            deterministic_eos: (($records | map(.first_eos_position) | unique | length) == 1),
            stable_resident_bytes: (($records | map(.resident_bytes) | unique | length) == 1),
            stable_kernels: (($records | map(.selected_kernels) | unique | length) == 1)
          }
        }
        | .pass = all(.checks[]; . == true)
    ' "$mode_dir"/run-*.json | tee "$mode_dir/summary.json"
}

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"

# Do not inherit a user-shell experiment into the mixed oracle.
run_mode mixed -u ATLAS_GEMMA4_WEIGHT_FORMAT -u ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT
run_mode q4_embeddings ATLAS_GEMMA4_WEIGHT_FORMAT=q4_embeddings
run_mode q4_lm_head ATLAS_GEMMA4_WEIGHT_FORMAT=q4_lm_head

jq -n \
    --slurpfile mixed "$artifact_dir/mixed/summary.json" \
    --slurpfile embeddings "$artifact_dir/q4_embeddings/summary.json" \
    --slurpfile lm_head "$artifact_dir/q4_lm_head/summary.json" '
      ($mixed[0]) as $baseline
      | ($embeddings[0]) as $embeddings
      | ($lm_head[0]) as $lm_head
      | {
          diagnostic: true,
          promotion: "not_applicable",
          configurations: {mixed: $baseline, q4_embeddings: $embeddings, q4_lm_head: $lm_head},
          comparison: {
            q4_embeddings_exact_stream: ($baseline.records[0].generated_token_sha256 == $embeddings.records[0].generated_token_sha256),
            q4_embeddings_same_eos: ($baseline.records[0].first_eos_position == $embeddings.records[0].first_eos_position),
            q4_lm_head_exact_stream: ($baseline.records[0].generated_token_sha256 == $lm_head.records[0].generated_token_sha256),
            q4_lm_head_same_eos: ($baseline.records[0].first_eos_position == $lm_head.records[0].first_eos_position),
            q4_embeddings_kernels: $embeddings.records[0].selected_kernels,
            q4_lm_head_kernels: $lm_head.records[0].selected_kernels
          }
        }
    ' | tee "$artifact_dir/q4-vocabulary-diagnosis-summary.json"

echo "Q4 vocabulary diagnosis: $artifact_dir/q4-vocabulary-diagnosis-summary.json"
