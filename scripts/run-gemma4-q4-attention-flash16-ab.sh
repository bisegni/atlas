#!/usr/bin/env bash
# Speed-first screen for the single-dispatch flash16 attention kernel vs the
# production two-pass pipeline. Parity is not a gate anymore: stream hashes
# are recorded as drift diagnostics, and the screen verdict is speed.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

runs=1
while (($#)); do
    case "$1" in
        --runs)
            shift
            runs=${1:?--runs needs a positive integer}
            [[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo "--runs must be a positive integer" >&2; exit 2; }
            ;;
        *)
            echo "usage: $0 [--runs N]" >&2
            exit 2
            ;;
    esac
    shift
done

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12.3-q4-attention-flash16-ab/${stamp}"
mkdir -p "$artifact_dir/baseline" "$artifact_dir/candidate"

[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

clean_env=(
    -u ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT
    -u ATLAS_GEMMA4_QKV_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_PACKED16_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT
    -u ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT
    -u ATLAS_GEMMA4_QK_NORM_ROPE_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_NORM_EXPERIMENT
    -u ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT
    -u ATLAS_GEMMA4_WEIGHT_FORMAT
    -u ATLAS_GEMMA4_TRACE_STAGES
    -u ATLAS_GEMMA4_TRACE_GELU
    ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=disabled
    ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT=baseline
    ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT=baseline
    ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=baseline
)

run_mode() {
    local label=$1
    local attention_selector=$2
    local expected_kernel=$3
    local mode_dir="$artifact_dir/$label"

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
            if ! env "${clean_env[@]}" "ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=$attention_selector" \
                cargo run --release -p atlas-cli -- benchmark \
                    --model "$model_id" \
                    --kv-cache-type q4_0 \
                    --prompt "$prompt" \
                    --warmup-decode-tokens "$warmup" \
                    --decode-tokens "$measured" \
                    --max-context "$context" \
                    > "$mode_dir/$workload-$run.json" \
                    2> "$mode_dir/$workload-$run.log"; then
                echo "${label} ${workload} screen failed; log follows:" >&2
                cat "$mode_dir/$workload-$run.log" >&2
                exit 1
            fi
        done

        jq -s --arg workload "$workload" --argjson expected_runs "$runs" --arg expected_kernel "$expected_kernel" '
            def median: sort | .[length / 2 | floor];
            . as $records |
            {
              workload: $workload,
              records: $records,
              median: {
                decode_tok_s: ($records | map(.decode_tok_s) | median),
                measured_decode_gpu_ms: ($records | map(.measured_decode_gpu_ms) | median),
                dispatches_per_measured_token: ($records | map(.measured_decode_dispatch_calls / .measured_decode_tokens) | median)
              },
              checks: {
                expected_runs: ($records | length == $expected_runs),
                resident: all($records[]; .executor == "resident"),
                mixed_weights: all($records[]; .weight_format == "mixed_q4_q6"),
                q4_kv: all($records[]; .kv_cache_type == "q4_0"),
                selected_attention_kernel: all($records[]; .selected_kernels.attention == $expected_kernel),
                stable_resident: (($records | map(.resident_bytes) | unique | length) == 1),
                stable_kv: (($records | map(.kv_cache_bytes) | unique | length) == 1),
                stable_upload: (($records | map(.weight_upload_bytes) | unique | length) == 1),
                stable_readback: (($records | map(.readback_bytes) | unique | length) == 1)
              }
            }
            | .pass = all(.checks[]; . == true)
        ' "$mode_dir"/"$workload"-*.json > "$mode_dir/$workload-summary.json"
    done
}

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

run_mode baseline 2pass_no_value_barrier attention_decode_fused_gemma4_simd_q4_0_2pass_no_value_barrier
run_mode candidate flash16 attention_decode_gemma4_simd_q4_0_flash16

jq -n \
    --arg model_id "$model_id" \
    --arg fixture "$fixture" \
    --rawfile fixture_sha256 "$artifact_dir/fixture-sha256.txt" \
    --arg artifact_dir "$artifact_dir" \
    --argjson expected_runs "$runs" \
    --slurpfile baseline_short "$artifact_dir/baseline/short-summary.json" \
    --slurpfile baseline_long "$artifact_dir/baseline/long-summary.json" \
    --slurpfile candidate_short "$artifact_dir/candidate/short-summary.json" \
    --slurpfile candidate_long "$artifact_dir/candidate/long-summary.json" '
    ($baseline_short[0]) as $base_short |
    ($baseline_long[0]) as $base_long |
    ($candidate_short[0]) as $candidate_short |
    ($candidate_long[0]) as $candidate_long |
    {
      purpose: "speed-first single-dispatch flash16 attention experiment; parity is not a gate",
      model_id: $model_id,
      fixture: $fixture,
      fixture_sha256: ($fixture_sha256 | split(" ")[0]),
      artifact_dir: $artifact_dir,
      expected_benchmark_runs: $expected_runs,
      environment: {
        ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT: "disabled",
        baseline_attention: "2pass_no_value_barrier",
        candidate_attention: "flash16"
      },
      baseline: {short: $base_short, long: $base_long},
      candidate: {short: $candidate_short, long: $candidate_long},
      comparison: {
        baseline_long_tok_s: $base_long.median.decode_tok_s,
        candidate_long_tok_s: $candidate_long.median.decode_tok_s,
        baseline_short_tok_s: $base_short.median.decode_tok_s,
        candidate_short_tok_s: $candidate_short.median.decode_tok_s,
        baseline_long_gpu_ms: $base_long.median.measured_decode_gpu_ms,
        candidate_long_gpu_ms: $candidate_long.median.measured_decode_gpu_ms,
        baseline_short_gpu_ms: $base_short.median.measured_decode_gpu_ms,
        candidate_short_gpu_ms: $candidate_short.median.measured_decode_gpu_ms,
        baseline_long_dispatches: $base_long.median.dispatches_per_measured_token,
        candidate_long_dispatches: $candidate_long.median.dispatches_per_measured_token,
        baseline_short_dispatches: $base_short.median.dispatches_per_measured_token,
        candidate_short_dispatches: $candidate_short.median.dispatches_per_measured_token,
        long_speedup_percent: (($candidate_long.median.decode_tok_s / $base_long.median.decode_tok_s - 1) * 100),
        short_speedup_percent: (($candidate_short.median.decode_tok_s / $base_short.median.decode_tok_s - 1) * 100),
        long_gpu_time_delta_percent: (($candidate_long.median.measured_decode_gpu_ms / $base_long.median.measured_decode_gpu_ms - 1) * 100),
        short_gpu_time_delta_percent: (($candidate_short.median.measured_decode_gpu_ms / $base_short.median.measured_decode_gpu_ms - 1) * 100),
        baseline_stream_sha256: $base_long.records[0].generated_token_sha256,
        candidate_stream_sha256: $candidate_long.records[0].generated_token_sha256,
        stream_changed: ($base_long.records[0].generated_token_sha256 != $candidate_long.records[0].generated_token_sha256),
        resident: ($base_long.checks.resident and $base_short.checks.resident and $candidate_long.checks.resident and $candidate_short.checks.resident),
        candidate_kernel_selected: $candidate_long.checks.selected_attention_kernel,
        stable_accounting: ($candidate_long.checks.stable_resident and $candidate_long.checks.stable_kv and $candidate_long.checks.stable_upload and $candidate_long.checks.stable_readback)
      }
    }
    | .comparison.long_improved = (.comparison.long_speedup_percent >= 3)
    | .comparison.short_not_regressed = (.comparison.short_speedup_percent >= 0)
    | .pass = (
      .baseline.short.pass and .baseline.long.pass and .candidate.short.pass and .candidate.long.pass and
      .comparison.resident and .comparison.candidate_kernel_selected and .comparison.stable_accounting and
      .comparison.long_improved and .comparison.short_not_regressed
    )
' > "$artifact_dir/q4-attention-flash16-ab-summary.json"

summary="$artifact_dir/q4-attention-flash16-ab-summary.json"
jq '{pass, comparison}' "$summary"
echo "Q4 attention flash16 screen: $summary"
if jq -e '.pass == true' "$summary" >/dev/null; then
    echo "Q4 ATTENTION FLASH16: SCREEN PASS"
else
    echo "Q4 ATTENTION FLASH16: SCREEN FAILED" >&2
    exit 1
fi
