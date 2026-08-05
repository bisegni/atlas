#!/usr/bin/env bash
# Exact mixed-Q4/Q6 LM-head A/B for the fixed 32+128 Resident decode window.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline_env=ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=baseline
candidate_env=ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=cacheopt
runs=3
while (($#)); do
    case "$1" in
        *) echo "usage: $0" >&2; exit 2 ;;
    esac
    shift
done

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain Resident inference.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12a-q6-lm-head-ab/${stamp}"
mkdir -p "$artifact_dir"
[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

run_mode() {
    local label=$1
    local env_assignment=$2
    local mode_dir="$artifact_dir/$label"
    mkdir -p "$mode_dir"
    for workload in fixed; do
        local warmup=32 measured=128 context=4096
        echo "Running ${runs} ${label} ${workload} Resident windows..."
        for run in $(seq 1 "$runs"); do
            if ! env -u ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT -u ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT \
                -u ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT -u ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT \
                -u ATLAS_GEMMA4_RMS_NORM_EXPERIMENT -u ATLAS_GEMMA4_WEIGHT_FORMAT \
                -u ATLAS_GEMMA4_FFN_GATE_UP_EXPERIMENT "$env_assignment" \
                cargo run --release -p atlas-cli -- benchmark --model "$model_id" --kv-cache-type q4_0 --prompt "$prompt" --warmup-decode-tokens "$warmup" --decode-tokens "$measured" --max-context "$context" > "$mode_dir/$workload-$run.json" 2> "$mode_dir/$workload-$run.log"; then
                echo "${label} ${workload} run ${run} failed; log follows:" >&2
                cat "$mode_dir/$workload-$run.log" >&2
                exit 1
            fi
        done
        jq -s --arg workload "$workload" --argjson runs "$runs" '
          def median: sort | .[length / 2 | floor];
          . as $records |
          {workload:$workload,records:$records,
           median_decode_tok_s:($records|map(.decode_tok_s)|median),
           median_measured_decode_gpu_ms:($records|map(.measured_decode_gpu_ms)|median),
           checks:{expected_runs:($records|length==$runs),
             resident:all($records[];.executor=="resident"),
             mixed_weights:all($records[];.weight_format=="mixed_q4_q6"),
             q4_kv:all($records[];.kv_cache_type=="q4_0"),
             warmup_window:all($records[];.warmup_decode_tokens==32),
             measured_window:all($records[];.measured_decode_tokens==128),
             deterministic_prompt:(($records|map(.prompt_token_sha256)|unique|length)==1),
             deterministic_stream:(($records|map(.generated_token_sha256)|unique|length)==1),
             deterministic_measured_stream:(($records|map(.measured_generated_token_sha256)|unique|length)==1),
             deterministic_eos:(($records|map(.first_eos_position)|unique|length)==1),
             stable_resident:(($records|map(.resident_bytes)|unique|length)==1),
             stable_kv:(($records|map(.kv_cache_bytes)|unique|length)==1),
             stable_upload:(($records|map(.weight_upload_bytes)|unique|length)==1),
             stable_readback:(($records|map(.readback_bytes)|unique|length)==1),
             selected_lm_head:all($records[];.selected_kernels.output_projection!="none")}}
          | .pass=all(.checks[];.==true)
        ' "$mode_dir"/$workload-*.json | tee "$mode_dir/$workload-summary.json"
    done
}

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"
run_mode baseline "$baseline_env"
run_mode candidate "$candidate_env"

jq -n --arg baseline_env "$baseline_env" --arg candidate_env "$candidate_env" --slurpfile bl "$artifact_dir/baseline/fixed-summary.json" --slurpfile cl "$artifact_dir/candidate/fixed-summary.json" '
 ($bl[0]) as $bl|($cl[0]) as $cl|
 {baseline_environment:$baseline_env,candidate_environment:$candidate_env,baseline:{fixed:$bl},candidate:{fixed:$cl},comparison:{
   exact_full_stream:($bl.records[0].generated_token_sha256==$cl.records[0].generated_token_sha256),
   exact_measured_stream:($bl.records[0].measured_generated_token_sha256==$cl.records[0].measured_generated_token_sha256),
   identical_eos:($bl.records[0].first_eos_position==$cl.records[0].first_eos_position),
   identical_non_q6_kernels:(($bl.records[0].selected_kernels|del(.q6_projection,.output_projection))==($cl.records[0].selected_kernels|del(.q6_projection,.output_projection))),
   baseline_pipeline_selected:($bl.records[0].selected_kernels.output_projection=="matvec_q6_k_8row"),
   candidate_pipeline_selected:($cl.records[0].selected_kernels.output_projection=="matvec_q6_k_8row_cacheopt"),
   stable_resident:($bl.records[0].resident_bytes==$cl.records[0].resident_bytes),
   stable_kv:($bl.records[0].kv_cache_bytes==$cl.records[0].kv_cache_bytes),
   no_upload_regression:($cl.records[0].weight_upload_bytes<=$bl.records[0].weight_upload_bytes),
   no_readback_regression:($cl.records[0].readback_bytes<=$bl.records[0].readback_bytes),
   baseline_tok_s:$bl.median_decode_tok_s,candidate_tok_s:$cl.median_decode_tok_s,
   baseline_gpu_ms:$bl.median_measured_decode_gpu_ms,candidate_gpu_ms:$cl.median_measured_decode_gpu_ms}}
 | .comparison.throughput_improvement_percent=((.comparison.candidate_tok_s/.comparison.baseline_tok_s-1)*100)
 | .comparison.gpu_time_reduction_percent=((1-.comparison.candidate_gpu_ms/.comparison.baseline_gpu_ms)*100)
 | .comparison.performance_gate=(.comparison.throughput_improvement_percent>=2 or .comparison.gpu_time_reduction_percent>=2)
 | .pass=($bl.pass and $cl.pass and .comparison.exact_full_stream and .comparison.exact_measured_stream and .comparison.identical_eos and .comparison.identical_non_q6_kernels and .comparison.baseline_pipeline_selected and .comparison.candidate_pipeline_selected and .comparison.stable_resident and .comparison.stable_kv and .comparison.no_upload_regression and .comparison.no_readback_regression and .comparison.performance_gate)
' | tee "$artifact_dir/q6-lm-head-ab-summary.json"

echo "Q6 LM-head A/B: $artifact_dir/q6-lm-head-ab-summary.json"
if jq -e '.pass == true' "$artifact_dir/q6-lm-head-ab-summary.json" >/dev/null; then
    echo "Q6 LM-HEAD A/B: PASS (promotion evidence ready)"
else
    echo "Q6 LM-HEAD A/B: NO PROMOTION" >&2
    exit 1
fi
