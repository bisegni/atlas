#!/usr/bin/env bash
# llama.cpp-style 32-row mv_ext Q4 matvec + Q6 LM-head A/B vs the production
# 16-row/8-row kernels (on top of the flash16 attention default). The Q4
# matvec parity gate is superseded per phase 12.3: stream hashes are recorded
# as diagnostics and correctness is proven by the kernel-level tolerance test
# crates/atlas-metal/tests/matvec_mv_ext_parity.rs. --with-rms-fused composes
# the phase 13.0 P1 RMS-input fusion into the candidate (parity proven by
# crates/atlas-metal/tests/matvec_rms_fused_parity.rs). --with-matvec-64row
# composes the phase 13.0 P4 64-row-per-threadgroup mv_ext variants
# (mv_ext_64: matvec_q4_0_64row_mv[_rms] + matvec_q6_k_64row_mv[_rms], 256
# threads per dispatch; same tolerance parity contract). --screen uses two
# runs and is not eligible for promotion.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

runs=5
promotion=true
compose_rms=false
compose_flash16_uw=false
compose_rms_matvec=false
compose_dispatch_fusion=false
compose_matvec_64row=false
compose_batch_tiled=false
while (($#)); do
    case "$1" in
        --screen) runs=2; promotion=false ;;
        --with-rms-vec4) compose_rms=true ;;
        --with-flash16-uw) compose_flash16_uw=true ;;
        --with-rms-fused) compose_rms_matvec=true ;;
        --with-dispatch-fusion) compose_dispatch_fusion=true ;;
        --with-matvec-64row) compose_matvec_64row=true ;;
        --with-batch-tiled) compose_batch_tiled=true ;;
        *) echo "usage: $0 [--screen] [--with-rms-vec4] [--with-flash16-uw] [--with-rms-fused] [--with-dispatch-fusion] [--with-matvec-64row] [--with-batch-tiled]" >&2; exit 2 ;;
    esac
    shift
done

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12.3-q4-matvec-mv-ext-ab/${stamp}"
mkdir -p "$artifact_dir/baseline" "$artifact_dir/candidate"

[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

clean_env=(
    -u ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT
    -u ATLAS_GEMMA4_QKV_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_BATCH_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT
    -u ATLAS_GEMMA4_Q4_PACKED16_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT
    -u ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT
    -u ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT
    -u ATLAS_GEMMA4_QK_NORM_ROPE_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT
    -u ATLAS_GEMMA4_RMS_MATVEC_EXPERIMENT
    -u ATLAS_GEMMA4_KV_APPEND_VNORM_EXPERIMENT
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
    ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT=baseline
)

run_mode() {
    local label=$1
    local q4_matvec_selector=$2
    local q6_selector=$3
    local gate_up_selector=$4
    local rms_selector=$5
    local attention_selector=$6
    local expected_attention=$7
    local rms_matvec_selector=$8
    local fusion_extra=$9
    local expected_fusion=${10}
    local batch_selector=${11}
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
        local fusion_env=($fusion_extra)
        for run in $(seq 1 "$runs"); do
            if ! env "${clean_env[@]}" ${fusion_env[@]+"${fusion_env[@]}"} \
                "ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=$attention_selector" \
                "ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT=$q4_matvec_selector" \
                "ATLAS_GEMMA4_Q4_BATCH_EXPERIMENT=$batch_selector" \
                "ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=$q6_selector" \
                "ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=$gate_up_selector" \
                "ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=$rms_selector" \
                "ATLAS_GEMMA4_RMS_MATVEC_EXPERIMENT=$rms_matvec_selector" \
                cargo run --release -p atlas-cli -- benchmark \
                    --model "$model_id" \
                    --kv-cache-type q4_0 \
                    --prompt "$prompt" \
                    --warmup-decode-tokens "$warmup" \
                    --decode-tokens "$measured" \
                    --max-context "$context" \
                    > "$mode_dir/$workload-$run.json" \
                    2> "$mode_dir/$workload-$run.log"; then
                echo "${label} ${workload} run ${run} failed; log follows:" >&2
                cat "$mode_dir/$workload-$run.log" >&2
                exit 1
            fi
        done

        jq -s --arg workload "$workload" --argjson runs "$runs" \
            --arg expected_q4 "$q4_matvec_selector" --arg expected_q6 "$q6_selector" \
            --arg expected_gate_up "$gate_up_selector" --arg expected_rms "$rms_selector" \
            --arg expected_rms_matvec "$rms_matvec_selector" \
            --arg expected_fusion "$expected_fusion" \
            --arg expected_attention "$expected_attention" \
            --arg expected_batch "$batch_selector" '
            def median: sort | .[length / 2 | floor];
            . as $records |
            {
              workload: $workload,
              records: $records,
              median: {
                decode_tok_s: ($records | map(.decode_tok_s) | median),
                measured_decode_gpu_ms: ($records | map(.measured_decode_gpu_ms) | median),
                dispatches_per_measured_token: ($records | map(.measured_decode_dispatch_calls / .measured_decode_tokens) | median),
                prefill_tok_s: ($records | map(.prefill_tok_s) | median)
              },
              checks: {
                expected_runs: ($records | length == $runs),
                resident: all($records[]; .executor == "resident"),
                mixed_weights: all($records[]; .weight_format == "mixed_q4_q6"),
                q4_kv: all($records[]; .kv_cache_type == "q4_0"),
                selected_attention_kernel: all($records[]; .selected_kernels.attention == $expected_attention),
                selected_batch_kernel: all($records[]; .selected_kernels.q4_batch_projection == ($expected_batch | if . == "tiled" then "matmul_q4_0_batch_16row_token_tiled" else "matmul_q4_0_batch_16row" end)),
                selected_matvec_kernel: all($records[]; .selected_kernels.q4_projection == ($expected_q4 | if . == "mv_ext_64" then "matvec_q4_0_64row_mv" elif . == "mv_ext" then "matvec_q4_0_32row_mv" else "matvec_q4_0_16row" end) and .selected_kernels.ffn_down_projection == ($expected_q4 | if . == "mv_ext_64" then "matvec_q4_0_64row_mv" elif . == "mv_ext" then "matvec_q4_0_32row_mv" else "matvec_q4_0_16row" end)),
                selected_qkv_kernel: all($records[]; .selected_kernels.q4_qkv_projection == ($expected_q4 | if . == "mv_ext_64" then "matmul_q4_0_qkv_32row_mv" elif . == "mv_ext" then "matmul_q4_0_qkv_32row_mv" else "matmul_q4_0_qkv_16row" end)),
                selected_gate_up_kernel: all($records[]; .selected_kernels.q4_gate_up_projection == ($expected_gate_up | if . == "mv_ext_64" then "matmul_q4_0_gate_up_32row_mv" elif . == "mv_ext" then "matmul_q4_0_gate_up_32row_mv" else "matmul_q4_0_gate_up_16row" end)),
                selected_q6_kernel: all($records[]; .selected_kernels.output_projection == ($expected_q6 | if . == "mv_ext_64" then "matvec_q6_k_64row_mv" elif . == "mv_ext" then "matvec_q6_k_32row_mv" else "matvec_q6_k_8row" end)),
                selected_rms_kernel: all($records[]; .selected_kernels.rms_norm == ($expected_rms | if . == "vec4" then "rms_norm_decode_f32_vec4" else "rms_norm_decode_f32" end)),
                selected_rms_fused_kernel: all($records[]; .selected_kernels.rms_fused_projection == ($expected_rms_matvec | if . == "fused" then "rms_input_matvec_fused" else "none" end)),
                selected_rms_epilogue_kernel: all($records[]; .selected_kernels.rms_epilogue == ($expected_fusion | if . == "fused" then "gemma4_rms_residual_f32" else "rms_norm_decode_f32+vector_add_f32" end)),
                selected_activation_kernel: all($records[]; .selected_kernels.ffn_gate_up_activation == ($expected_fusion | if . == "fused" then "gelu_multiply_f32" else "gelu_f32+vector_multiply_f32" end)),
                selected_ple_kernel: all($records[]; .selected_kernels.ple_composition == ($expected_fusion | if . == "fused" then "ple_gelu_multiply_offset_f32" else "gelu_f32+vector_multiply_offset_f32" end)),
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

echo "Running mv_ext kernel-level tolerance parity tests..."
cargo test --release -p atlas-metal --test matvec_mv_ext_parity \
    > "$artifact_dir/matvec-mv-ext-parity.log" 2>&1
parity_pass=false
grep -q "6 passed; 0 failed" "$artifact_dir/matvec-mv-ext-parity.log" && parity_pass=true
[[ "$parity_pass" == true ]] || { echo "mv_ext parity tests failed; log:" >&2; cat "$artifact_dir/matvec-mv-ext-parity.log" >&2; exit 1; }

echo "Running RMS-matvec fusion kernel-level tolerance parity tests..."
cargo test --release -p atlas-metal --test matvec_rms_fused_parity \
    > "$artifact_dir/matvec-rms-fused-parity.log" 2>&1
rms_fused_parity_pass=false
grep -q "6 passed; 0 failed" "$artifact_dir/matvec-rms-fused-parity.log" && rms_fused_parity_pass=true
[[ "$rms_fused_parity_pass" == true ]] || { echo "rms-fused parity tests failed; log:" >&2; cat "$artifact_dir/matvec-rms-fused-parity.log" >&2; exit 1; }

echo "Running dispatch-fusion kernel-level tolerance parity tests..."
cargo test --release -p atlas-metal --test dispatch_fusion_parity \
    > "$artifact_dir/dispatch-fusion-parity.log" 2>&1
dispatch_fusion_parity_pass=false
grep -q "3 passed; 0 failed" "$artifact_dir/dispatch-fusion-parity.log" && dispatch_fusion_parity_pass=true
[[ "$dispatch_fusion_parity_pass" == true ]] || { echo "dispatch-fusion parity tests failed; log:" >&2; cat "$artifact_dir/dispatch-fusion-parity.log" >&2; exit 1; }

echo "Running KV-append + V-norm fusion bitwise parity tests..."
cargo test --release -p atlas-metal --test kv_append_vnorm_fused_parity \
    > "$artifact_dir/kv-append-vnorm-parity.log" 2>&1
kv_append_vnorm_parity_pass=false
grep -q "3 passed; 0 failed" "$artifact_dir/kv-append-vnorm-parity.log" && kv_append_vnorm_parity_pass=true
[[ "$kv_append_vnorm_parity_pass" == true ]] || { echo "kv-append-vnorm parity tests failed; log:" >&2; cat "$artifact_dir/kv-append-vnorm-parity.log" >&2; exit 1; }

echo "Running token-tiled batch kernel tolerance parity tests..."
cargo test --release -p atlas-metal --test batch_matmul_parity \
    > "$artifact_dir/batch-matmul-parity.log" 2>&1
batch_matmul_parity_pass=false
grep -q "1 passed; 0 failed" "$artifact_dir/batch-matmul-parity.log" && batch_matmul_parity_pass=true
[[ "$batch_matmul_parity_pass" == true ]] || { echo "batch-matmul parity tests failed; log:" >&2; cat "$artifact_dir/batch-matmul-parity.log" >&2; exit 1; }

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

candidate_rms=baseline
[[ "$compose_rms" == true ]] && candidate_rms=vec4
candidate_rms_matvec=baseline
[[ "$compose_rms_matvec" == true ]] && candidate_rms_matvec=fused
candidate_mv="mv_ext"
[[ "$compose_matvec_64row" == true ]] && candidate_mv=mv_ext_64
candidate_fusion_env=""
[[ "$compose_dispatch_fusion" == true ]] && candidate_fusion_env="ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT=fused ATLAS_GEMMA4_FFN_GELU_MULTIPLY_EXPERIMENT=fused ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=fused ATLAS_GEMMA4_KV_APPEND_VNORM_EXPERIMENT=fused"
candidate_attention=flash16
candidate_attention_kernel=attention_decode_gemma4_simd_q4_0_flash16
[[ "$compose_flash16_uw" == true ]] && candidate_attention=flash16_uw && candidate_attention_kernel=attention_decode_gemma4_simd_q4_0_flash16_uw
candidate_batch=baseline
[[ "$compose_batch_tiled" == true ]] && candidate_batch=tiled
run_mode baseline baseline baseline baseline baseline flash16 attention_decode_gemma4_simd_q4_0_flash16 baseline "" baseline baseline
candidate_expected_fusion=baseline
[[ "$compose_dispatch_fusion" == true ]] && candidate_expected_fusion=fused
run_mode candidate "$candidate_mv" "$candidate_mv" mv_ext "$candidate_rms" "$candidate_attention" "$candidate_attention_kernel" "$candidate_rms_matvec" "$candidate_fusion_env" "$candidate_expected_fusion" "$candidate_batch"

candidate_env_str="$candidate_attention + ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT=$candidate_mv + ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=$candidate_mv + ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=mv_ext"
[[ "$compose_rms" == true ]] && candidate_env_str="$candidate_env_str + ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=vec4"
[[ "$compose_flash16_uw" == true ]] && candidate_env_str="$candidate_env_str + ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=flash16_uw"
[[ "$compose_rms_matvec" == true ]] && candidate_env_str="$candidate_env_str + ATLAS_GEMMA4_RMS_MATVEC_EXPERIMENT=fused"
[[ "$compose_dispatch_fusion" == true ]] && candidate_env_str="$candidate_env_str + ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT=fused + ATLAS_GEMMA4_FFN_GELU_MULTIPLY_EXPERIMENT=fused + ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=fused + ATLAS_GEMMA4_KV_APPEND_VNORM_EXPERIMENT=fused"
[[ "$compose_batch_tiled" == true ]] && candidate_env_str="$candidate_env_str + ATLAS_GEMMA4_Q4_BATCH_EXPERIMENT=tiled"

jq -n \
    --arg baseline_env "flash16 + production 16row/8row kernels" \
    --arg candidate_env "$candidate_env_str" \
    --arg parity_log "$artifact_dir/matvec-mv-ext-parity.log" \
    --arg parity_log_rms "$artifact_dir/matvec-rms-fused-parity.log" \
    --arg parity_log_dispatch "$artifact_dir/dispatch-fusion-parity.log" \
    --arg parity_log_kv_append "$artifact_dir/kv-append-vnorm-parity.log" \
    --arg parity_log_batch "$artifact_dir/batch-matmul-parity.log" \
    --argjson promotion "$promotion" \
    --argjson attention_changed "$compose_flash16_uw" \
    --argjson prefill_target 120 \
    --slurpfile baseline_short "$artifact_dir/baseline/short-summary.json" \
    --slurpfile baseline_long "$artifact_dir/baseline/long-summary.json" \
    --slurpfile candidate_short "$artifact_dir/candidate/short-summary.json" \
    --slurpfile candidate_long "$artifact_dir/candidate/long-summary.json" '
    ($baseline_short[0]) as $bs | ($baseline_long[0]) as $bl |
    ($candidate_short[0]) as $cs | ($candidate_long[0]) as $cl |
    {
      baseline_environment: $baseline_env, candidate_environment: $candidate_env,
      kernel_parity_test: "matvec_mv_ext_parity: 6 passed; 0 failed (max-abs < 1e-3)",
      kernel_parity_test_rms_fused: "matvec_rms_fused_parity: 6 passed; 0 failed (max-abs < 1e-3)",
      kernel_parity_test_dispatch_fusion: "dispatch_fusion_parity: 3 passed; 0 failed (max-abs < 1e-3)",
      kernel_parity_test_kv_append_vnorm: "kv_append_vnorm_fused_parity: 3 passed; 0 failed (bitwise)",
      kernel_parity_test_batch_tiled: "batch_matmul_parity: 1 passed; 0 failed (max-abs < 1e-3)",
      baseline: {long: $bl, short: $bs}, candidate: {long: $cl, short: $cs},
      comparison: {
        stream_drift_long: ($bl.records[0].generated_token_sha256 != $cl.records[0].generated_token_sha256),
        stream_drift_short: ($bs.records[0].generated_token_sha256 != $cs.records[0].generated_token_sha256),
        exact_long: ($bl.records[0].generated_token_sha256 == $cl.records[0].generated_token_sha256 and $bl.records[0].measured_generated_token_sha256 == $cl.records[0].measured_generated_token_sha256 and $bl.records[0].first_eos_position == $cl.records[0].first_eos_position),
        exact_short: ($bs.records[0].generated_token_sha256 == $cs.records[0].generated_token_sha256 and $bs.records[0].measured_generated_token_sha256 == $cs.records[0].measured_generated_token_sha256 and $bs.records[0].first_eos_position == $cs.records[0].first_eos_position),
        stable_long_accounting: ($bl.records[0].resident_bytes == $cl.records[0].resident_bytes and $bl.records[0].kv_cache_bytes == $cl.records[0].kv_cache_bytes and $bl.records[0].weight_upload_bytes == $cl.records[0].weight_upload_bytes and $bl.records[0].readback_bytes == $cl.records[0].readback_bytes),
        stable_short_accounting: ($bs.records[0].resident_bytes == $cs.records[0].resident_bytes and $bs.records[0].kv_cache_bytes == $cs.records[0].kv_cache_bytes and $bs.records[0].weight_upload_bytes == $cs.records[0].weight_upload_bytes and $bs.records[0].readback_bytes == $cs.records[0].readback_bytes),
        identical_attention_kernel: ($bl.records[0].selected_kernels.attention == $cl.records[0].selected_kernels.attention),
        candidate_attention_kernel_selected: ($bl.records[0].selected_kernels.attention != $cl.records[0].selected_kernels.attention and $cl.checks.selected_attention_kernel),
        baseline_long_tok_s: $bl.median.decode_tok_s, candidate_long_tok_s: $cl.median.decode_tok_s,
        baseline_short_tok_s: $bs.median.decode_tok_s, candidate_short_tok_s: $cs.median.decode_tok_s,
        baseline_prefill_tok_s: $bl.median.prefill_tok_s, candidate_prefill_tok_s: $cl.median.prefill_tok_s,
        prefill_gate_met: ($cl.median.prefill_tok_s >= $prefill_target),
        promotion_eligible: $promotion
      }
    }
    | .comparison.long_speedup_percent = ((.comparison.candidate_long_tok_s / .comparison.baseline_long_tok_s - 1) * 100)
    | .comparison.short_speedup_percent = ((.comparison.candidate_short_tok_s / .comparison.baseline_short_tok_s - 1) * 100)
    | .comparison.prefill_speedup_percent = ((.comparison.candidate_prefill_tok_s / .comparison.baseline_prefill_tok_s - 1) * 100)
    | .comparison.long_improved = (.comparison.long_speedup_percent >= 3)
    | .comparison.short_not_regressed = (.comparison.short_speedup_percent >= 0)
    | .comparison.prefill_improved = (.comparison.prefill_speedup_percent >= 3)
    | .pass = (all(.baseline[]; .pass) and all(.candidate[]; .pass) and all(.comparison["stable_long_accounting","stable_short_accounting","short_not_regressed"]; . == true) and ((if $attention_changed then .comparison.candidate_attention_kernel_selected else .comparison.identical_attention_kernel end)) and (if .comparison.promotion_eligible then .comparison.long_improved else (.comparison.long_speedup_percent >= 0) end))
  ' | tee "$artifact_dir/q4-matvec-mv-ext-ab-summary.json"

echo "Q4 matvec mv_ext A/B: $artifact_dir/q4-matvec-mv-ext-ab-summary.json"
if jq -e '.pass == true' "$artifact_dir/q4-matvec-mv-ext-ab-summary.json" >/dev/null; then
    [[ "$promotion" == true ]] && echo "MV_EXT A/B: PASS" || echo "MV_EXT SCREEN: PASS (not eligible for promotion)"
else
    echo "MV_EXT A/B: NO PROMOTION" >&2
    exit 1
fi
