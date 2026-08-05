#!/usr/bin/env bash
# Exact Resident A/B for packed Q4 16-row FFN-down + PLE projection weights.
# Normal mode is the five-run 2x campaign gate; --screen runs two samples and
# only rejects regressions before the expensive full gate.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline_env=ATLAS_GEMMA4_Q4_PACKED16_EXPERIMENT=baseline
candidate_env=ATLAS_GEMMA4_Q4_PACKED16_EXPERIMENT=ffn_down_ple
runs=5
require_double=true
while (($#)); do
    case "$1" in
        --screen) runs=2; require_double=false ;;
        *) echo "usage: $0 [--screen]" >&2; exit 2 ;;
    esac
    shift
done

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12.3-q4-packed16-ab/${stamp}"
mkdir -p "$artifact_dir"
[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

clean_env=(
    -u ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_PACKED16_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_NORM_EXPERIMENT
    -u ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT
    -u ATLAS_GEMMA4_WEIGHT_FORMAT
    -u ATLAS_GEMMA4_FFN_GATE_UP_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT
    -u ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT
    -u ATLAS_GEMMA4_TRACE_STAGES
    -u ATLAS_GEMMA4_TRACE_GELU
)

run_mode() {
    local label=$1
    local env_assignment=$2
    local mode_dir="$artifact_dir/$label"
    mkdir -p "$mode_dir"
    for workload in short long; do
        local warmup=0
        local measured=128
        local context=4096
        if [[ "$workload" == long ]]; then
            warmup=1024
            measured=512
            context=2048
        fi
        echo "Running ${runs} ${label} ${workload}-context Resident windows..."
        for run in $(seq 1 "$runs"); do
            env "${clean_env[@]}" "$env_assignment" cargo run --release -p atlas-cli -- benchmark \
                --model "$model_id" --kv-cache-type q4_0 --prompt "$prompt" \
                --warmup-decode-tokens "$warmup" --decode-tokens "$measured" \
                --max-context "$context" > "$mode_dir/$workload-$run.json" \
                2> "$mode_dir/$workload-$run.log" || {
                    cat "$mode_dir/$workload-$run.log" >&2
                    exit 1
                }
        done
        jq -s --arg workload "$workload" --argjson runs "$runs" '
          def median: sort | .[length / 2 | floor];
          . as $records | {
            workload: $workload, records: $records,
            median_decode_tok_s: ($records | map(.decode_tok_s) | median),
            median_measured_decode_gpu_ms: ($records | map(.measured_decode_gpu_ms) | median),
            checks: {
              expected_runs: ($records | length == $runs),
              production_timing: all($records[]; .diagnostic == false),
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
              stable_readback: (($records | map(.readback_bytes) | unique | length) == 1)
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

jq -n --arg baseline_env "$baseline_env" --arg candidate_env "$candidate_env" \
  --argjson require_double "$require_double" \
  --slurpfile baseline_short "$artifact_dir/baseline/short-summary.json" \
  --slurpfile baseline_long "$artifact_dir/baseline/long-summary.json" \
  --slurpfile candidate_short "$artifact_dir/candidate/short-summary.json" \
  --slurpfile candidate_long "$artifact_dir/candidate/long-summary.json" '
  ($baseline_short[0]) as $bs | ($baseline_long[0]) as $bl |
  ($candidate_short[0]) as $cs | ($candidate_long[0]) as $cl |
  {baseline_environment: $baseline_env, candidate_environment: $candidate_env,
   baseline: {short: $bs, long: $bl}, candidate: {short: $cs, long: $cl}, comparison: {
     exact_short: ($bs.records[0].generated_token_sha256 == $cs.records[0].generated_token_sha256 and $bs.records[0].measured_generated_token_sha256 == $cs.records[0].measured_generated_token_sha256 and $bs.records[0].first_eos_position == $cs.records[0].first_eos_position),
     exact_long: ($bl.records[0].generated_token_sha256 == $cl.records[0].generated_token_sha256 and $bl.records[0].measured_generated_token_sha256 == $cl.records[0].measured_generated_token_sha256 and $bl.records[0].first_eos_position == $cl.records[0].first_eos_position),
     stable_short_accounting: ($bs.records[0].resident_bytes == $cs.records[0].resident_bytes and $bs.records[0].kv_cache_bytes == $cs.records[0].kv_cache_bytes and $bs.records[0].weight_upload_bytes == $cs.records[0].weight_upload_bytes and $bs.records[0].readback_bytes == $cs.records[0].readback_bytes),
     stable_long_accounting: ($bl.records[0].resident_bytes == $cl.records[0].resident_bytes and $bl.records[0].kv_cache_bytes == $cl.records[0].kv_cache_bytes and $bl.records[0].weight_upload_bytes == $cl.records[0].weight_upload_bytes and $bl.records[0].readback_bytes == $cl.records[0].readback_bytes),
     baseline_pipeline_selected: ($bs.records[0].selected_kernels.q4_packed16_layout == "baseline" and $bl.records[0].selected_kernels.q4_packed16_layout == "baseline" and $bs.records[0].selected_kernels.ple_projection == "matvec_q4_0_16row" and $bl.records[0].selected_kernels.ple_projection == "matvec_q4_0_16row"),
     candidate_pipeline_selected: ($cs.records[0].selected_kernels.q4_packed16_layout == "ffn_down_ple" and $cl.records[0].selected_kernels.q4_packed16_layout == "ffn_down_ple" and $cs.records[0].selected_kernels.ffn_down_projection == "matvec_q4_0_16row_packed16" and $cl.records[0].selected_kernels.ffn_down_projection == "matvec_q4_0_16row_packed16" and $cs.records[0].selected_kernels.ple_projection == "matvec_q4_0_16row_packed16" and $cl.records[0].selected_kernels.ple_projection == "matvec_q4_0_16row_packed16"),
     baseline_short_tok_s: $bs.median_decode_tok_s, candidate_short_tok_s: $cs.median_decode_tok_s,
     baseline_long_tok_s: $bl.median_decode_tok_s, candidate_long_tok_s: $cl.median_decode_tok_s,
     baseline_short_gpu_ms: $bs.median_measured_decode_gpu_ms, candidate_short_gpu_ms: $cs.median_measured_decode_gpu_ms,
     baseline_long_gpu_ms: $bl.median_measured_decode_gpu_ms, candidate_long_gpu_ms: $cl.median_measured_decode_gpu_ms,
     require_double: $require_double
   }}
  | .comparison.short_speedup_percent = ((.comparison.candidate_short_tok_s / .comparison.baseline_short_tok_s - 1) * 100)
  | .comparison.long_speedup_percent = ((.comparison.candidate_long_tok_s / .comparison.baseline_long_tok_s - 1) * 100)
  | .comparison.short_double_target_met = (.comparison.candidate_short_tok_s >= 2 * .comparison.baseline_short_tok_s)
  | .comparison.long_double_target_met = (.comparison.candidate_long_tok_s >= 2 * .comparison.baseline_long_tok_s)
  | .comparison.no_regression = (.comparison.short_speedup_percent >= 0 and .comparison.long_speedup_percent >= 0)
  | .pass = (all(.baseline[]; .pass) and all(.candidate[]; .pass) and .comparison.exact_short and .comparison.exact_long and .comparison.stable_short_accounting and .comparison.stable_long_accounting and .comparison.baseline_pipeline_selected and .comparison.candidate_pipeline_selected and .comparison.no_regression and (if .comparison.require_double then (.comparison.short_double_target_met and .comparison.long_double_target_met) else true end))
  ' | tee "$artifact_dir/q4-packed16-ab-summary.json"

echo "Q4 packed16 A/B: $artifact_dir/q4-packed16-ab-summary.json"
jq -e '.pass == true' "$artifact_dir/q4-packed16-ab-summary.json" >/dev/null || {
    echo "Q4 PACKED16: NO PROMOTION" >&2
    exit 1
}
[[ "$require_double" == true ]] && echo "Q4 PACKED16: 2X TARGET MET" || echo "Q4 PACKED16: SCREEN PASS"
