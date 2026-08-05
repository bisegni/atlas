#!/usr/bin/env bash
# Measure the production Gemma Resident baseline at increasing decode context.
#
# This is deliberately baseline-only. It does not select an experiment or make
# a promotion decision; it establishes whether a proposed optimization should
# target context-dependent attention or context-independent decode work.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

runs=5
contexts=(0 256 512 1024 1536)
while (($#)); do
    case "$1" in
        --runs)
            shift
            runs=${1:?--runs needs a positive integer}
            [[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo "--runs must be a positive integer" >&2; exit 2; }
            ;;
        --contexts)
            shift
            IFS=',' read -r -a contexts <<< "${1:?--contexts needs comma-separated warmup token counts}"
            ((${#contexts[@]})) || { echo "--contexts must not be empty" >&2; exit 2; }
            for context in "${contexts[@]}"; do
                [[ "$context" =~ ^[0-9]+$ ]] || { echo "invalid context warmup count: $context" >&2; exit 2; }
            done
            ;;
        *)
            echo "usage: $0 [--runs N] [--contexts 0,256,512,1024,1536]" >&2
            exit 2
            ;;
    esac
    shift
done

model_id=gemma4-e2b-q4_0
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12.3-context-sweep/${stamp}"
mkdir -p "$artifact_dir/runs"

[[ -f "$fixture" ]] || { echo "missing Gemma fixture: $fixture" >&2; exit 2; }

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
    ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=disabled
    ATLAS_GEMMA4_FFN_GATE_UP_KERNEL_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_GATE_UP_ACTIVATION_EXPERIMENT=baseline
    ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=baseline
    ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT=baseline
    ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=baseline
)

echo "Verifying pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    | tee "$artifact_dir/model-verify.json"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

for context in "${contexts[@]}"; do
    echo "Running ${runs} Resident baseline windows after ${context} warmup tokens..."
    for run in $(seq 1 "$runs"); do
        env "${baseline_env[@]}" cargo run --release -p atlas-cli -- benchmark \
            --model "$model_id" \
            --kv-cache-type q4_0 \
            --prompt "$prompt" \
            --warmup-decode-tokens "$context" \
            --decode-tokens 128 \
            --max-context 2048 \
            > "$artifact_dir/runs/context-${context}-${run}.json" \
            2> "$artifact_dir/runs/context-${context}-${run}.log"
    done
done

jq -s --argjson expected_runs "$runs" --argjson contexts "$(printf '%s\n' "${contexts[@]}" | jq -R 'tonumber' | jq -s .)" '
    def median: sort | .[length / 2 | floor];
    . as $all |
    {
      purpose: "production Resident baseline context sweep; not a candidate experiment",
      expected_runs: $expected_runs,
      contexts: [
        $contexts[] as $context |
        ($all | map(select(.warmup_decode_tokens == $context))) as $records |
        {
          warmup_decode_tokens: $context,
          runs: ($records | length),
          median: {
            decode_tok_s: ($records | map(.decode_tok_s) | median),
            measured_decode_gpu_ms: ($records | map(.measured_decode_gpu_ms) | median),
            dispatches_per_token: ($records | map(.measured_decode_dispatch_calls / .measured_decode_tokens) | median),
            threadgroups_per_token: ($records | map(.measured_decode_threadgroups / .measured_decode_tokens) | median),
            command_buffers_per_token: ($records | map(.measured_decode_command_buffers / .measured_decode_tokens) | median)
          },
          checks: {
            expected_runs: ($records | length == $expected_runs),
            resident: all($records[]; .executor == "resident"),
            q4_kv: all($records[]; .kv_cache_type == "q4_0"),
            baseline_attention: all($records[]; .selected_kernels.attention == "attention_decode_fused_gemma4_simd_q4_0_2pass_no_value_barrier"),
            deterministic_stream: (($records | map(.measured_generated_token_sha256) | unique | length) == 1),
            stable_resident: (($records | map(.resident_bytes) | unique | length) == 1),
            stable_kv: (($records | map(.kv_cache_bytes) | unique | length) == 1),
            stable_readback: (($records | map(.readback_bytes) | unique | length) == 1)
          }
        } | .pass = all(.checks[]; . == true)
      ]
    } | .pass = all(.contexts[]; .pass)
' "$artifact_dir"/runs/*.json | tee "$artifact_dir/context-sweep-summary.json"

jq -r '
    "# Gemma Resident baseline context sweep\n\n" +
    "| Warmup tokens | tok/s | GPU ms | dispatches/token | threadgroups/token | command buffers/token |\n" +
    "|---:|---:|---:|---:|---:|---:|\n" +
    (.contexts[] | "| " + (.warmup_decode_tokens | tostring) + " | " + (.median.decode_tok_s | tostring) + " | " + (.median.measured_decode_gpu_ms | tostring) + " | " + (.median.dispatches_per_token | tostring) + " | " + (.median.threadgroups_per_token | tostring) + " | " + (.median.command_buffers_per_token | tostring) + " |")
' "$artifact_dir/context-sweep-summary.json" > "$artifact_dir/context-sweep-summary.md"

echo "Context sweep summary: $artifact_dir/context-sweep-summary.json"
echo "Context sweep report:  $artifact_dir/context-sweep-summary.md"
jq -e '.pass == true' "$artifact_dir/context-sweep-summary.json" >/dev/null
