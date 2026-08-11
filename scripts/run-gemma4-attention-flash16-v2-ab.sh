#!/usr/bin/env bash
# Exact Flash16 attention A/B: the exact-compatible Flash16 kernel versus the
# Resident LegacyFused oracle. Both modes run with the same fixed workload so
# exact token/EOS parity and a positive median decode improvement decide
# whether Flash16 can return to the production default.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

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
artifact_dir="artifacts/phase-12.3-q4-attention-flash16-v2-ab/${stamp}"
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
    ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=mv_ext
    ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT=baseline
    ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT=baseline
    ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=mv_ext
    ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT=mv_ext
    ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=vec4
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
            if ! env "${clean_env[@]}" \
                cargo run --release -p atlas-cli -- benchmark \
                    --model "$model_id" \
                    --kv-cache-type q4_0 \
                    --prompt "$prompt" \
                    --warmup-decode-tokens "$warmup" \
                    --decode-tokens "$measured" \
                    --max-context "$context" \
                    --q4-attention-mode "$attention_selector" \
                    > "$mode_dir/$workload-$run.json" \
                    2> "$mode_dir/$workload-$run.log"; then
                echo "${label} ${workload} run ${run} failed; log follows:" >&2
                cat "$mode_dir/$workload-$run.log" >&2
                exit 1
            fi
        done

        jq -s --arg workload "$workload" --argjson runs "$runs" \
            --arg expected_kernel "$expected_kernel" '
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
                expected_runs: ($records | length == $runs),
                resident: all($records[]; .executor == "resident"),
                mixed_weights: all($records[]; .weight_format == "mixed_q4_q6"),
                q4_kv: all($records[]; .kv_cache_type == "q4_0"),
                selected_attention_kernel: all($records[]; .selected_kernels.attention == $expected_kernel),
                deterministic_stream: (($records | map(.generated_token_sha256) | unique | length) == 1),
                deterministic_eos: (($records | map(.first_eos_position) | unique | length) == 1),
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

echo "Running Flash16 exact kernel parity tests..."
cargo test --release -p atlas-metal --test attention_flash_correctness \
    > "$artifact_dir/attention-flash-correctness.log" 2>&1
parity_pass=false
grep -q "2 passed; 0 failed" "$artifact_dir/attention-flash-correctness.log" && parity_pass=true
[[ "$parity_pass" == true ]] || { echo "flash16 parity tests failed; log:" >&2; cat "$artifact_dir/attention-flash-correctness.log" >&2; exit 1; }

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

run_mode baseline legacy_fused attention_decode_fused_gemma4_simd_q4_0
run_mode candidate flash16 attention_decode_gemma4_simd_q4_0_flash16_exact

jq -n \
    --arg baseline_env "legacy_fused on mv_ext + q6 mv + gate-up mv + rms vec4 stack" \
    --arg candidate_env "flash16 exact on mv_ext + q6 mv + gate-up mv + rms vec4 stack" \
    --arg parity_log "$artifact_dir/attention-flash-correctness.log" \
    --argjson promotion "$promotion" \
    --slurpfile baseline_short "$artifact_dir/baseline/short-summary.json" \
    --slurpfile baseline_long "$artifact_dir/baseline/long-summary.json" \
    --slurpfile candidate_short "$artifact_dir/candidate/short-summary.json" \
    --slurpfile candidate_long "$artifact_dir/candidate/long-summary.json" '
    ($baseline_short[0]) as $bs | ($baseline_long[0]) as $bl |
    ($candidate_short[0]) as $cs | ($candidate_long[0]) as $cl |
    {
      baseline_environment: $baseline_env, candidate_environment: $candidate_env,
      kernel_parity_test: "attention_flash_correctness: Flash16 exact is bitwise equal to LegacyFused across Q4-KV cases",
      baseline: {long: $bl, short: $bs}, candidate: {long: $cl, short: $cs},
      comparison: {
        stream_drift_long: ($bl.records[0].generated_token_sha256 != $cl.records[0].generated_token_sha256),
        stream_drift_short: ($bs.records[0].generated_token_sha256 != $cs.records[0].generated_token_sha256),
        exact_long: ($bl.records[0].generated_token_sha256 == $cl.records[0].generated_token_sha256 and $bl.records[0].measured_generated_token_sha256 == $cl.records[0].measured_generated_token_sha256 and $bl.records[0].first_eos_position == $cl.records[0].first_eos_position),
        exact_short: ($bs.records[0].generated_token_sha256 == $cs.records[0].generated_token_sha256 and $bs.records[0].measured_generated_token_sha256 == $cs.records[0].measured_generated_token_sha256 and $bs.records[0].first_eos_position == $cs.records[0].first_eos_position),
        stable_long_accounting: ($bl.records[0].resident_bytes == $cl.records[0].resident_bytes and $bl.records[0].kv_cache_bytes == $cl.records[0].kv_cache_bytes and $bl.records[0].weight_upload_bytes == $cl.records[0].weight_upload_bytes and $bl.records[0].readback_bytes == $cl.records[0].readback_bytes),
        stable_short_accounting: ($bs.records[0].resident_bytes == $cs.records[0].resident_bytes and $bs.records[0].kv_cache_bytes == $cs.records[0].kv_cache_bytes and $bs.records[0].weight_upload_bytes == $cs.records[0].weight_upload_bytes and $bs.records[0].readback_bytes == $cs.records[0].readback_bytes),
        baseline_long_tok_s: $bl.median.decode_tok_s, candidate_long_tok_s: $cl.median.decode_tok_s,
        baseline_short_tok_s: $bs.median.decode_tok_s, candidate_short_tok_s: $cs.median.decode_tok_s,
        promotion_eligible: $promotion
      }
    }
    | .comparison.long_speedup_percent = ((.comparison.candidate_long_tok_s / .comparison.baseline_long_tok_s - 1) * 100)
    | .comparison.short_speedup_percent = ((.comparison.candidate_short_tok_s / .comparison.baseline_short_tok_s - 1) * 100)
    | .comparison.long_improved = (.comparison.long_speedup_percent > 0)
    | .comparison.short_not_regressed = (.comparison.short_speedup_percent >= 0)
    | .pass = (all(.baseline[]; .pass) and all(.candidate[]; .pass) and all(.comparison["stable_long_accounting","stable_short_accounting","short_not_regressed"]; . == true) and (if .comparison.promotion_eligible then .comparison.long_improved else (.comparison.long_speedup_percent >= 0) end))
  ' | tee "$artifact_dir/q4-attention-flash16-v2-ab-summary.json"

echo "Flash16 v2 A/B: $artifact_dir/q4-attention-flash16-v2-ab-summary.json"
if jq -e '.pass == true' "$artifact_dir/q4-attention-flash16-v2-ab-summary.json" >/dev/null; then
    [[ "$promotion" == true ]] && echo "FLASH16 V2 A/B: PASS" || echo "FLASH16 V2 SCREEN: PASS (not eligible for promotion)"
else
    echo "FLASH16 V2 A/B: NO PROMOTION" >&2
    exit 1
fi
