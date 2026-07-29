#!/usr/bin/env bash
# Compare Gemma Resident decode kernel configurations at a sustained context
# window. Each argument is one environment assignment (NAME=value) or
# `-` for the production default with no environment override.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline_env=${1:--}
candidate_env=${2:-ATLAS_GEMMA4_WEIGHT_FORMAT=all_q4}
model_id=gemma4-e2b-q4_0
cache_type=q4_0
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
long_warmup_tokens=1024
long_measure_tokens=512
long_context=2048
short_measure_tokens=128
short_context=4096
runs=5
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12a-decode-attention-ab/${stamp}"
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
mkdir -p "$artifact_dir"

if [[ ! -f "$fixture" ]]; then
    echo "missing Gemma fixture: $fixture" >&2
    exit 2
fi

run_command() {
    local env_assignment=$1
    shift
    if [[ "$env_assignment" == "-" ]]; then
        "$@"
    else
        env "$env_assignment" "$@"
    fi
}

run_configuration() {
    local label=$1
    local env_assignment=$2
    local mode_dir="$artifact_dir/$label"
    local run
    mkdir -p "$mode_dir"

    echo "Running ${label} long-context profile (${env_assignment})..."
    run_command "$env_assignment" cargo run --release -p atlas-cli -- profile \
        --model "$model_id" --kv-cache-type "$cache_type" \
        --decode-tokens "$long_warmup_tokens" --max-context "$long_context" \
        > "$mode_dir/profile.json" 2> "$mode_dir/profile.log"
    jq -e '
      .executor == "resident"
      and (.kv_cache_type == "q4_0")
      and ([.samples[].decode_position] | index(1024) != null)
    ' "$mode_dir/profile.json" >/dev/null

    for workload in long short; do
        local warmup=0
        local measured=$short_measure_tokens
        local max_context=$short_context
        if [[ "$workload" == "long" ]]; then
            warmup=$long_warmup_tokens
            measured=$long_measure_tokens
            max_context=$long_context
        fi
        echo "Running ${runs} ${label} ${workload}-context fixed Resident windows..."
        for run in $(seq 1 "$runs"); do
            run_command "$env_assignment" cargo run --release -p atlas-cli -- benchmark \
                --model "$model_id" --kv-cache-type "$cache_type" --prompt "$prompt" \
                --warmup-decode-tokens "$warmup" --decode-tokens "$measured" \
                --max-context "$max_context" \
                > "$mode_dir/${workload}-${run}.json" \
                2> "$mode_dir/${workload}-${run}.log"
        done
        cat "$mode_dir"/"$workload"-*.json > "$mode_dir/${workload}.jsonl"
        jq -s --arg workload "$workload" --arg env_assignment "$env_assignment" '
          def median: sort | .[length / 2 | floor];
          . as $records
          | ($records | map(.decode_tok_s)) as $decode
          | {
              workload: $workload,
              environment: $env_assignment,
              records: $records,
              median_decode_tok_s: ($decode | median),
              attention_kernel: $records[0].selected_kernels.attention,
              checks: {
                expected_runs: ($records | length == 5),
                resident_executor: all($records[]; .executor == "resident"),
                q4_kv_cache: all($records[]; .kv_cache_type == "q4_0"),
                deterministic_prompt: (($records | map(.prompt_token_sha256) | unique | length) == 1),
                deterministic_full_stream: (($records | map(.generated_token_sha256) | unique | length) == 1),
                deterministic_measured_stream: (($records | map(.measured_generated_token_sha256) | unique | length) == 1),
                deterministic_first_eos: (($records | map(.first_eos_position) | unique | length) == 1),
                deterministic_selected_kernels: (($records | map(.selected_kernels) | unique | length) == 1),
                stable_kv_residency: (($records | map(.kv_cache_bytes) | unique | length) == 1),
                stable_resident_accounting: (($records | map(.resident_bytes) | unique | length) == 1),
                positive_measurement_window: all($records[]; .measured_decode_tokens > 0 and .decode_command_buffers > 0)
              }
            }
          | .pass = all(.checks[]; . == true)
        ' "$mode_dir/${workload}.jsonl" | tee "$mode_dir/${workload}-summary.json"
    done
}

echo "Verifying the pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

run_configuration baseline "$baseline_env"
run_configuration candidate "$candidate_env"

summary="$artifact_dir/decode-attention-ab-summary.json"
jq -n \
    --arg baseline_env "$baseline_env" \
    --arg candidate_env "$candidate_env" \
    --slurpfile baseline_long "$artifact_dir/baseline/long-summary.json" \
    --slurpfile baseline_short "$artifact_dir/baseline/short-summary.json" \
    --slurpfile candidate_long "$artifact_dir/candidate/long-summary.json" \
    --slurpfile candidate_short "$artifact_dir/candidate/short-summary.json" \
    '
      ($baseline_long[0]) as $old_long
      | ($baseline_short[0]) as $old_short
      | ($candidate_long[0]) as $new_long
      | ($candidate_short[0]) as $new_short
      | {
          configurations: {
            baseline: {environment: $baseline_env, long: $old_long, short: $old_short},
            candidate: {environment: $candidate_env, long: $new_long, short: $new_short}
          },
          comparison: {
            same_long_prompt_tokens: ($old_long.records[0].prompt_token_sha256 == $new_long.records[0].prompt_token_sha256),
            same_long_full_stream: ($old_long.records[0].generated_token_sha256 == $new_long.records[0].generated_token_sha256),
            same_long_measured_stream: ($old_long.records[0].measured_generated_token_sha256 == $new_long.records[0].measured_generated_token_sha256),
            same_long_first_eos: ($old_long.records[0].first_eos_position == $new_long.records[0].first_eos_position),
            same_long_kv_cache_bytes: ($old_long.records[0].kv_cache_bytes == $new_long.records[0].kv_cache_bytes),
            same_short_prompt_tokens: ($old_short.records[0].prompt_token_sha256 == $new_short.records[0].prompt_token_sha256),
            same_short_full_stream: ($old_short.records[0].generated_token_sha256 == $new_short.records[0].generated_token_sha256),
            same_short_first_eos: ($old_short.records[0].first_eos_position == $new_short.records[0].first_eos_position),
            same_short_kv_cache_bytes: ($old_short.records[0].kv_cache_bytes == $new_short.records[0].kv_cache_bytes),
            candidate_all_q4: ($new_long.records[0].weight_format == "all_q4" and $new_short.records[0].weight_format == "all_q4"),
            candidate_q4_vocabulary_kernels: ($new_long.records[0].embedding_kernel == "embedding_lookup_q4_0" and $new_long.records[0].output_projection_kernel == "matvec_q4_0_16row" and $new_long.records[0].selected_kernels.q6_projection == "none"),
            candidate_selected_kernels_differ: ($old_long.records[0].selected_kernels != $new_long.records[0].selected_kernels),
            candidate_selected_kernels_are_consistent: ($new_long.records[0].selected_kernels == $new_short.records[0].selected_kernels),
            baseline_long_decode_tok_s: $old_long.median_decode_tok_s,
            candidate_long_decode_tok_s: $new_long.median_decode_tok_s,
            long_decode_speedup_percent: (($new_long.median_decode_tok_s / $old_long.median_decode_tok_s - 1) * 100),
            baseline_short_decode_tok_s: $old_short.median_decode_tok_s,
            candidate_short_decode_tok_s: $new_short.median_decode_tok_s,
            short_decode_speedup_percent: (($new_short.median_decode_tok_s / $old_short.median_decode_tok_s - 1) * 100)
          }
        }
      | .comparison.exact_long_stream_parity_pass =
          (.comparison.same_long_prompt_tokens and .comparison.same_long_full_stream and .comparison.same_long_measured_stream and .comparison.same_long_first_eos)
      | .comparison.exact_short_stream_parity_pass =
          (.comparison.same_short_prompt_tokens and .comparison.same_short_full_stream and .comparison.same_short_first_eos)
      | .comparison.required_long_decode_speedup_percent = 3
      | .comparison.long_context_improved = (.comparison.long_decode_speedup_percent >= .comparison.required_long_decode_speedup_percent)
      | .comparison.short_context_not_regressed = (.comparison.short_decode_speedup_percent >= 0)
      | .pass =
          ($old_long.pass and $old_short.pass and $new_long.pass and $new_short.pass
           and .comparison.exact_long_stream_parity_pass
           and .comparison.exact_short_stream_parity_pass
           and .comparison.same_long_kv_cache_bytes
           and .comparison.same_short_kv_cache_bytes
           and .comparison.candidate_all_q4
           and .comparison.candidate_q4_vocabulary_kernels
           and .comparison.candidate_selected_kernels_differ
           and .comparison.candidate_selected_kernels_are_consistent
           and .comparison.long_context_improved
           and .comparison.short_context_not_regressed)
    ' | tee "$summary"

echo "Combined decode kernel A/B artifact: $summary"
if jq -e '.pass == true' "$summary" >/dev/null; then
    echo "GEMMA DECODE KERNEL A/B: PASS"
else
    echo "GEMMA DECODE KERNEL A/B: NO PROMOTION" >&2
    exit 1
fi
