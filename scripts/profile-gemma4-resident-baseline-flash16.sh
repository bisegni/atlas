#!/usr/bin/env bash
# Establish a decision-grade, baseline-only Gemma Resident profile.
#
# This is deliberately not an A/B harness. It pins every known optimization
# selector to the production baseline, captures repeatable throughput metrics,
# then records exact per-dispatch diagnostic attribution for the same workload.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

runs=5
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
artifact_dir="artifacts/phase-12.3-baseline-profile/${stamp}"
mkdir -p "$artifact_dir/benchmark" "$artifact_dir/diagnostic"

[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

# Unset experimental switches inherited from an interactive shell, then pin the
# ones with explicit baseline selectors. This makes the artifact a usable oracle
# for every later candidate, rather than a profile of an unknown mixed pipeline.
baseline_env=(
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
    ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=flash16
    ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT=baseline
    ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT=baseline
    ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=baseline
)

run_benchmark_workload() {
    local workload=$1
    local warmup=0
    local measured=128
    local context=4096
    if [[ "$workload" == long ]]; then
        warmup=1024
        measured=512
        context=2048
    fi

    echo "Running ${runs} baseline ${workload}-context Resident benchmark windows..."
    for run in $(seq 1 "$runs"); do
        env "${baseline_env[@]}" cargo run --release -p atlas-cli -- benchmark \
            --model "$model_id" \
            --kv-cache-type q4_0 \
            --prompt "$prompt" \
            --warmup-decode-tokens "$warmup" \
            --decode-tokens "$measured" \
            --max-context "$context" \
            > "$artifact_dir/benchmark/${workload}-${run}.json" \
            2> "$artifact_dir/benchmark/${workload}-${run}.log"
    done

    jq -s --arg workload "$workload" --argjson expected_runs "$runs" '
        def median: sort | .[length / 2 | floor];
        . as $records |
        {
          workload: $workload,
          records: $records,
          median: {
            decode_tok_s: ($records | map(.decode_tok_s) | median),
            measured_decode_gpu_ms: ($records | map(.measured_decode_gpu_ms) | median),
            dispatches_per_measured_token: ($records | map(.measured_decode_dispatch_calls / .measured_decode_tokens) | median),
            threadgroups_per_measured_token: ($records | map(.measured_decode_threadgroups / .measured_decode_tokens) | median),
            command_buffers_per_measured_token: ($records | map(.measured_decode_command_buffers / .measured_decode_tokens) | median)
          },
          selected_kernels: $records[0].selected_kernels,
          accounting: {
            resident_bytes: $records[0].resident_bytes,
            kv_cache_bytes: $records[0].kv_cache_bytes,
            weight_upload_bytes: $records[0].weight_upload_bytes,
            readback_bytes: $records[0].readback_bytes
          },
          checks: {
            expected_runs: ($records | length == $expected_runs),
            resident: all($records[]; .executor == "resident"),
            mixed_weights: all($records[]; .weight_format == "mixed_q4_q6"),
            q4_kv: all($records[]; .kv_cache_type == "q4_0"),
            baseline_gate_up: all($records[]; .selected_kernels.q4_gate_up_projection == "matmul_q4_0_gate_up_16row"),
            baseline_activation: all($records[]; .selected_kernels.ffn_gate_up_activation == "gelu_f32+vector_multiply_f32"),
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
    ' "$artifact_dir"/benchmark/"$workload"-*.json \
        | tee "$artifact_dir/benchmark/${workload}-summary.json"
}

run_diagnostic_workload() {
    local workload=$1
    local warmup=0
    local measured=128
    local context=4096
    if [[ "$workload" == long ]]; then
        warmup=1024
        measured=512
        context=2048
    fi

    local output="$artifact_dir/diagnostic/${workload}-profile.json"
    echo "Recording exact per-dispatch baseline attribution for ${workload}-context..."
    env "${baseline_env[@]}" cargo run --release -p atlas-cli -- profile bottlenecks \
        --model "$model_id" \
        --kv-cache-type q4_0 \
        --prompt "$prompt" \
        --warmup-decode-tokens "$warmup" \
        --decode-tokens "$measured" \
        --max-context "$context" \
        --mode diagnostic \
        --gpu-counters auto \
        --output "$output" \
        > "$artifact_dir/diagnostic/${workload}.stdout.json" \
        2> "$artifact_dir/diagnostic/${workload}.log"

    jq '
        {
          mode,
          profile_status,
          scope_contract,
          decode_scope,
          benchmark_compatibility,
          reconciliation,
          coverage,
          profiler_overhead,
          gpu_counter_capture,
          measured_decode: .scope_counters.decode_measured,
          attention_dispatches: [.attention_dispatches[] | select(.scope == "decode_measured")],
          recommendations: .hotspots.decode_measured,
          top_operation_families: ([.operation_families[] | select(.scope == "decode_measured")] | sort_by(.gpu_ns) | reverse | .[:15]),
          top_kernels: ([.kernels[] | select(.scope == "decode_measured")] | sort_by(.gpu_ns) | reverse | .[:20]),
          top_layers: ([.layers[] | select(.scope == "decode_measured")] | sort_by(.gpu_ns) | reverse | .[:20])
        }
    ' "$output" > "$artifact_dir/diagnostic/${workload}-hotspots.json"
}

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

run_benchmark_workload short
run_benchmark_workload long
run_diagnostic_workload short
run_diagnostic_workload long

jq -n \
    --arg model_id "$model_id" \
    --arg fixture "$fixture" \
    --rawfile fixture_sha256 "$artifact_dir/fixture-sha256.txt" \
    --arg artifact_dir "$artifact_dir" \
    --argjson expected_runs "$runs" \
    --slurpfile short_benchmark "$artifact_dir/benchmark/short-summary.json" \
    --slurpfile long_benchmark "$artifact_dir/benchmark/long-summary.json" \
    --slurpfile short_hotspots "$artifact_dir/diagnostic/short-hotspots.json" \
    --slurpfile long_hotspots "$artifact_dir/diagnostic/long-hotspots.json" '
    ($short_benchmark[0]) as $short |
    ($long_benchmark[0]) as $long |
    ($short_hotspots[0]) as $short_diagnostic |
    ($long_hotspots[0]) as $long_diagnostic |
    {
      purpose: "baseline-only Resident performance attribution; not a candidate experiment",
      model_id: $model_id,
      fixture: $fixture,
      fixture_sha256: ($fixture_sha256 | split(" ")[0]),
      artifact_dir: $artifact_dir,
      expected_benchmark_runs: $expected_runs,
      environment: {
        ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT: "disabled",
        ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT: "baseline",
        ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT: "baseline",
        ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT: "baseline",
        ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT: "baseline",
        ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT: "baseline"
      },
      benchmark: {short: $short, long: $long},
      diagnostic: {short: $short_diagnostic, long: $long_diagnostic},
      decision_gate: {
        benchmark_is_production_measurement: true,
        diagnostic_is_attribution_only: true,
        required_coverage: 0.95,
        candidate_rule: "Do not create a candidate until a top measured-decode hotspot has coverage at or above 95 percent, a stable baseline kernel selection, and an explicit hypothesis tied to its dispatch, bandwidth, or occupancy evidence."
      }
    }
    | .diagnostic.short.measured_decode_coverage = {
        dispatch: (
          .diagnostic.short.measured_decode.categorized_dispatches /
          .diagnostic.short.measured_decode.dispatches
        ),
        gpu: (
          .diagnostic.short.measured_decode.categorized_gpu_ns /
          .diagnostic.short.measured_decode.attributed_gpu_duration_ns
        )
      }
    | .diagnostic.long.measured_decode_coverage = {
        dispatch: (
          .diagnostic.long.measured_decode.categorized_dispatches /
          .diagnostic.long.measured_decode.dispatches
        ),
        gpu: (
          .diagnostic.long.measured_decode.categorized_gpu_ns /
          .diagnostic.long.measured_decode.attributed_gpu_duration_ns
        )
      }
    | .pass = (
        .benchmark.short.pass and .benchmark.long.pass and
        .diagnostic.short.profile_status == "complete" and
        .diagnostic.long.profile_status == "complete" and
        (.diagnostic.short.measured_decode_coverage.dispatch >= .decision_gate.required_coverage) and
        (.diagnostic.long.measured_decode_coverage.dispatch >= .decision_gate.required_coverage) and
        (.diagnostic.short.measured_decode_coverage.gpu >= .decision_gate.required_coverage) and
        (.diagnostic.long.measured_decode_coverage.gpu >= .decision_gate.required_coverage) and
        (.diagnostic.short.benchmark_compatibility.executor == "resident") and
        (.diagnostic.long.benchmark_compatibility.executor == "resident")
    )
' | tee "$artifact_dir/baseline-profile-summary.json"

jq -r '
    def ms: (.gpu_ns / 1000000 | round / 1000);
    "# Gemma Resident baseline profile\n\n" +
    "- Result: " + (if .pass then "PASS" else "INCOMPLETE" end) + "\n" +
    "- Model: `" + .model_id + "`\n" +
    "- Benchmark repetitions per workload: " + (.expected_benchmark_runs | tostring) + "\n" +
    "- Artifact: `" + .artifact_dir + "`\n\n" +
    "## Production medians\n\n" +
    "| Workload | tok/s | GPU ms | dispatches/token | threadgroups/token | Resident bytes | KV bytes |\n" +
    "|---|---:|---:|---:|---:|---:|---:|\n" +
    (["short", "long"][] as $w | .benchmark[$w] as $b |
      "| " + $w + " | " + ($b.median.decode_tok_s | tostring) + " | " + ($b.median.measured_decode_gpu_ms | tostring) + " | " + ($b.median.dispatches_per_measured_token | tostring) + " | " + ($b.median.threadgroups_per_measured_token | tostring) + " | " + ($b.accounting.resident_bytes | tostring) + " | " + ($b.accounting.kv_cache_bytes | tostring) + " |") +
    "\n\n## Long-context measured-decode hotspots\n\n" +
    "| Rank | Target | Class | GPU ms | GPU share | Dispatches/token | Confidence |\n" +
    "|---:|---|---|---:|---:|---:|---|\n" +
    (.diagnostic.long.recommendations[] |
      "| " + (.rank | tostring) + " | " + .target + " | " + .classification + " | " + (.absolute_ms | tostring) + " | " + (.categorized_gpu_share | tostring) + " | " + (.dispatches_per_measured_token | tostring) + " | " + .confidence + " |") +
    "\n\n## Decision rule\n\n" + .decision_gate.candidate_rule + "\n"
' "$artifact_dir/baseline-profile-summary.json" > "$artifact_dir/baseline-profile-summary.md"

echo "Baseline profile summary: $artifact_dir/baseline-profile-summary.json"
echo "Baseline hotspot report:  $artifact_dir/baseline-profile-summary.md"
jq -e '.pass == true' "$artifact_dir/baseline-profile-summary.json" >/dev/null || {
    echo "BASELINE PROFILE: INCOMPLETE (inspect diagnostics and benchmark logs)" >&2
    exit 1
}
echo "BASELINE PROFILE: PASS"
