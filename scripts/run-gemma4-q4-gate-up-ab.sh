#!/usr/bin/env bash
# Isolated mixed-Q4/Q6 Gate/Up kernel A/B: --screen uses two runs.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline_env=ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=baseline
candidate_env=ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=simdgroup_tiled
runs=5
promotion=true
while (($#)); do
    case "$1" in
        --screen) runs=2; promotion=false ;;
        *) echo "usage: $0 [--screen]" >&2; exit 2 ;;
    esac
    shift
done

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12a-q4-gate-up-ab/${stamp}"
mkdir -p "$artifact_dir"

[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

clean_env=(
    -u ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_NORM_EXPERIMENT
    -u ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT
    -u ATLAS_GEMMA4_WEIGHT_FORMAT
    -u ATLAS_GEMMA4_FFN_GATE_UP_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT
)

run_mode() {
    local label=$1
    local env_assignment=$2
    local mode_dir="$artifact_dir/$label"
    mkdir -p "$mode_dir"
    for workload in long short; do
        local warmup=0 measured=128 context=4096
        [[ "$workload" == long ]] && { warmup=1024; measured=512; context=2048; }
        echo "Running ${runs} ${label} ${workload}-context Resident windows..."
        for run in $(seq 1 "$runs"); do
            if ! env "${clean_env[@]}" "$env_assignment" \
                cargo run --release -p atlas-cli -- benchmark \
                --model "$model_id" --kv-cache-type q4_0 --prompt "$prompt" \
                --warmup-decode-tokens "$warmup" --decode-tokens "$measured" \
                --max-context "$context" > "$mode_dir/$workload-$run.json" \
                2> "$mode_dir/$workload-$run.log"; then
                echo "${label} ${workload} run ${run} failed; log follows:" >&2
                cat "$mode_dir/$workload-$run.log" >&2
                exit 1
            fi
        done
        jq -s --arg workload "$workload" --argjson runs "$runs" '
          def median: sort | .[length / 2 | floor];
          . as $records | {
            workload: $workload, records: $records,
            median_decode_tok_s: ($records | map(.decode_tok_s) | median),
            checks: {
              expected_runs: ($records | length == $runs),
              resident: all($records[]; .executor == "resident"),
              mixed_weights: all($records[]; .weight_format == "mixed_q4_q6"),
              q4_kv: all($records[]; .kv_cache_type == "q4_0"),
              deterministic_prompt: (($records | map(.prompt_token_sha256) | unique | length) == 1),
              deterministic_stream: (($records | map(.generated_token_sha256) | unique | length) == 1),
              deterministic_measured_stream: (($records | map(.measured_generated_token_sha256) | unique | length) == 1),
              deterministic_eos: (($records | map(.first_eos_position) | unique | length) == 1),
              stable_resident: (($records | map(.resident_bytes) | unique | length) == 1),
              stable_kv: (($records | map(.kv_cache_bytes) | unique | length) == 1),
              stable_upload: (($records | map(.weight_upload_bytes) | unique | length) == 1),
              stable_readback: (($records | map(.readback_bytes) | unique | length) == 1),
              cold_upload_present: all($records[]; .weight_upload_bytes > 0)
            }
          } | .pass = all(.checks[]; . == true)
        ' "$mode_dir"/$workload-*.json | tee "$mode_dir/$workload-summary.json"
    done
}

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"
run_mode baseline "$baseline_env"
run_mode candidate "$candidate_env"

jq -n --arg baseline_env "$baseline_env" --arg candidate_env "$candidate_env" --argjson promotion "$promotion" \
  --slurpfile baseline_long "$artifact_dir/baseline/long-summary.json" \
  --slurpfile baseline_short "$artifact_dir/baseline/short-summary.json" \
  --slurpfile candidate_long "$artifact_dir/candidate/long-summary.json" \
  --slurpfile candidate_short "$artifact_dir/candidate/short-summary.json" '
    ($baseline_long[0]) as $bl | ($baseline_short[0]) as $bs |
    ($candidate_long[0]) as $cl | ($candidate_short[0]) as $cs |
    {baseline_environment: $baseline_env, candidate_environment: $candidate_env,
     baseline: {long: $bl, short: $bs}, candidate: {long: $cl, short: $cs},
     comparison: {
       exact_long: ($bl.records[0].generated_token_sha256 == $cl.records[0].generated_token_sha256 and $bl.records[0].measured_generated_token_sha256 == $cl.records[0].measured_generated_token_sha256 and $bl.records[0].first_eos_position == $cl.records[0].first_eos_position),
       exact_short: ($bs.records[0].generated_token_sha256 == $cs.records[0].generated_token_sha256 and $bs.records[0].measured_generated_token_sha256 == $cs.records[0].measured_generated_token_sha256 and $bs.records[0].first_eos_position == $cs.records[0].first_eos_position),
       stable_long_accounting: ($bl.records[0].resident_bytes == $cl.records[0].resident_bytes and $bl.records[0].kv_cache_bytes == $cl.records[0].kv_cache_bytes and $bl.records[0].weight_upload_bytes == $cl.records[0].weight_upload_bytes and $bl.records[0].readback_bytes == $cl.records[0].readback_bytes),
       stable_short_accounting: ($bs.records[0].resident_bytes == $cs.records[0].resident_bytes and $bs.records[0].kv_cache_bytes == $cs.records[0].kv_cache_bytes and $bs.records[0].weight_upload_bytes == $cs.records[0].weight_upload_bytes and $bs.records[0].readback_bytes == $cs.records[0].readback_bytes),
       identical_non_gate_up_kernels: ($bl.records[0].selected_kernels.q4_qkv_projection == $cl.records[0].selected_kernels.q4_qkv_projection and $bl.records[0].selected_kernels.ffn_down_projection == $cl.records[0].selected_kernels.ffn_down_projection and $bl.records[0].selected_kernels.output_projection == $cl.records[0].selected_kernels.output_projection),
       baseline_pipeline_selected: ($bl.records[0].selected_kernels.q4_gate_up_projection == "matmul_q4_0_gate_up_16row" and $bs.records[0].selected_kernels.q4_gate_up_projection == "matmul_q4_0_gate_up_16row"),
       candidate_pipeline_selected: ($cl.records[0].selected_kernels.q4_gate_up_projection == "matmul_q4_0_gate_up_16row_simdgroup_tiled" and $cs.records[0].selected_kernels.q4_gate_up_projection == "matmul_q4_0_gate_up_16row_simdgroup_tiled"),
       baseline_long_tok_s: $bl.median_decode_tok_s, candidate_long_tok_s: $cl.median_decode_tok_s,
       baseline_short_tok_s: $bs.median_decode_tok_s, candidate_short_tok_s: $cs.median_decode_tok_s,
       promotion_eligible: $promotion}
    }
    | .comparison.long_speedup_percent = ((.comparison.candidate_long_tok_s / .comparison.baseline_long_tok_s - 1) * 100)
    | .comparison.short_speedup_percent = ((.comparison.candidate_short_tok_s / .comparison.baseline_short_tok_s - 1) * 100)
    | .comparison.long_improved = (.comparison.long_speedup_percent >= 3)
    | .comparison.short_not_regressed = (.comparison.short_speedup_percent >= 0)
    | .pass = (all(.baseline[]; .pass) and all(.candidate[]; .pass) and all(.comparison["exact_long","exact_short","stable_long_accounting","stable_short_accounting","identical_non_gate_up_kernels","baseline_pipeline_selected","candidate_pipeline_selected","short_not_regressed"]; . == true) and (if .comparison.promotion_eligible then .comparison.long_improved else (.comparison.long_speedup_percent >= 0) end))
  ' | tee "$artifact_dir/q4-gate-up-ab-summary.json"

echo "Q4 Gate/Up A/B: $artifact_dir/q4-gate-up-ab-summary.json"
if jq -e '.pass == true' "$artifact_dir/q4-gate-up-ab-summary.json" >/dev/null; then
    [[ "$promotion" == true ]] && echo "Q4 GATE/UP A/B: PASS" || echo "Q4 GATE/UP SCREEN: PASS (not eligible for promotion)"
else
    echo "Q4 GATE/UP A/B: NO PROMOTION" >&2
    exit 1
fi
