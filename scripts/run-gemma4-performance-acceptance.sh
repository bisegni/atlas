#!/usr/bin/env bash
# Run the Phase 12a Gemma Resident workload for each supported KV-cache type.
# The F32 result remains the only strict phase-acceptance gate; Q8/Q4 records
# are experimental comparisons collected with the identical prompt and process.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model_id=gemma4-e2b-q4_0
max_tokens=128
warm_runs=5
kv_cache_types=(f32 q8_0 q4_0)
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.'
shared_log=artifacts/chat-performance.jsonl
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/phase-12a-perf/${stamp}"
fixture=models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
mkdir -p "$artifact_dir"

if [[ ! -f "$fixture" ]]; then
    echo "missing Gemma fixture: $fixture" >&2
    exit 2
fi

# The old F32 attention pipeline is diagnostic-only, never an acceptance mode.
unset ATLAS_GEMMA4_ATTENTION_BASELINE

echo "Verifying the pinned Gemma fixture..."
cargo run --release -p atlas-cli -- model verify --model "$model_id" \
    2>&1 | tee "$artifact_dir/model-verify.log"
shasum -a 256 "$fixture" | tee "$artifact_dir/fixture-sha256.txt"

run_cache_type() {
    local cache_type=$1
    local mode_dir="$artifact_dir/$cache_type"
    local before_lines=0
    local after_lines new_records
    mkdir -p "$mode_dir"
    if [[ -f "$shared_log" ]]; then
        before_lines=$(wc -l < "$shared_log")
    fi

    echo "Running one cold and ${warm_runs} warm Resident turns with KV cache ${cache_type}..."
    {
        printf '%s\n' "$prompt"
        for _ in $(seq 1 "$warm_runs"); do
            printf '/reset\n%s\n' "$prompt"
        done
        printf '/quit\n'
    } | cargo run --release -p atlas-cli -- chat --model "$model_id" \
        --kv-cache-type "$cache_type" --max-tokens "$max_tokens" \
        2>&1 | tee "$mode_dir/chat.log"

    after_lines=$(wc -l < "$shared_log")
    new_records=$((after_lines - before_lines))
    if (( new_records != warm_runs + 1 )); then
        echo "${cache_type}: expected $((warm_runs + 1)) new performance records, found $new_records" >&2
        exit 1
    fi
    sed -n "$((before_lines + 1)),${after_lines}p" "$shared_log" > "$mode_dir/chat-performance.jsonl"

    echo "Profiling Resident decode with KV cache ${cache_type}..."
    cargo run --release -p atlas-cli -- profile --model "$model_id" \
        --kv-cache-type "$cache_type" 2>&1 | tee "$mode_dir/decode-profile.log"
}

for cache_type in "${kv_cache_types[@]}"; do
    run_cache_type "$cache_type"
done

# Chat deliberately stops at EOS/end-turn. Run a separate fixed-length
# workload so every cache mode performs the same 128 selections even when its
# greedy sequence encounters EOS at a different point.
run_fixed_benchmark() {
    local cache_type=$1
    local mode_dir="$artifact_dir/$cache_type"
    local run
    rm -f "$mode_dir"/benchmark-*.json "$mode_dir"/benchmark-*.log
    echo "Running six fixed 128-token Resident benchmarks with KV cache ${cache_type}..."
    for run in $(seq 1 6); do
        cargo run --release -p atlas-cli -- benchmark --model "$model_id" \
            --kv-cache-type "$cache_type" --prompt "$prompt" --decode-tokens "$max_tokens" \
            > "$mode_dir/benchmark-${run}.json" 2> "$mode_dir/benchmark-${run}.log"
    done
    cat "$mode_dir"/benchmark-*.json > "$mode_dir/benchmark.jsonl"
    jq -s --arg cache_type "$cache_type" '
      def median: sort | .[length / 2 | floor];
      . as $records
      | ($records | map(.prefill_tok_s)) as $prefill
      | ($records | map(.decode_tok_s)) as $decode
      | ($records | map(.kv_cache_bytes) | unique) as $kv_bytes
      | {
          cache_type: $cache_type,
          records: $records,
          median_prefill_tok_s: ($prefill | median),
          median_decode_tok_s: ($decode | median),
          kv_cache_bytes: $kv_bytes[0],
          checks: {
            six_records: ($records | length == 6),
            resident_executor: all($records[]; .executor == "resident"),
            selected_kv_cache_type: all($records[]; .kv_cache_type == $cache_type),
            fixed_decode_completion: all($records[]; .fixed_decode_tokens == 128 and .completed_decode_tokens == 128 and .decode_command_buffers == 127),
            first_eos_is_bounded: all($records[]; .first_eos_position == null or (.first_eos_position >= 1 and .first_eos_position <= .fixed_decode_tokens)),
            deterministic_token_stream: (($records | map(.generated_token_sha256) | unique | length) == 1),
            stable_kv_residency: ($kv_bytes | length == 1)
          }
        }
      | if $cache_type == "f32" then
          .checks += {
            prefill_median_at_least_50: (.median_prefill_tok_s >= 50),
            decode_median_at_least_40: (.median_decode_tok_s >= 40),
            bounded_resident_memory: all($records[]; .resident_bytes <= 3489602512)
          }
        else . end
      | .pass = all(.checks[]; . == true)
    ' "$mode_dir/benchmark.jsonl" | tee "$mode_dir/benchmark-summary.json"
}

for cache_type in "${kv_cache_types[@]}"; do
    run_fixed_benchmark "$cache_type"
done

echo "Checking canonical and packed-KV Resident gates..."
canonical_pass=true
if ! cargo test -p atlas-model --test phase_12a_gemma4_resident -- --ignored \
    2>&1 | tee "$artifact_dir/resident-token-and-packed-kv-gates.log"; then
    canonical_pass=false
fi

for cache_type in "${kv_cache_types[@]}"; do
    jq -s --arg cache_type "$cache_type" --argjson canonical_pass "$canonical_pass" '
      def median: sort | .[length / 2 | floor];
      . as $records
      | $records[1:] as $warm
      | ($warm | map(.prefill_tok_s)) as $prefill
      | ($warm | map(.decode_tok_s)) as $decode
      | {
          cache_type: $cache_type,
          records: $records,
          warm_summary: {
            runs: ($warm | length),
            prefill_tok_s: {median: ($prefill | median), range: [$prefill | min, max]},
            decode_tok_s: {median: ($decode | median), range: [$decode | min, max]},
            generated_tokens: ($warm | map(.generated_tokens)),
            decode_command_buffers: ($warm | map(.decode_command_buffers)),
            weight_upload_bytes: ($warm | map(.weight_upload_bytes)),
            resident_bytes: ($warm | map(.resident_bytes)),
            kv_cache_bytes: ($warm | map(.kv_cache_bytes)),
            attention_kernel: ($warm | map(.attention_kernel))
          },
          checks: {
            six_records: (($records | length) == 6),
            five_warm_runs: (($warm | length) == 5),
            resident_executor: all($warm[]; .executor == "resident"),
            selected_kv_cache_type: all($warm[]; .kv_cache_type == $cache_type),
            warm_zero_weight_upload: all($warm[]; .weight_upload_bytes == 0),
            # F32 is the pinned token-oracle mode, so its historical EOS
            # length is part of the phase gate. Packed KV modes are explicitly
            # experimental and may select a different greedy EOS token; require
            # a meaningful decode workload there without confusing it with F32
            # token equivalence.
            fixed_long_decode: (if $cache_type == "f32" then
                all($warm[]; .generated_tokens == 104 and .decode_command_buffers == 103)
              else
                all($warm[]; .generated_tokens >= 32 and .decode_command_buffers >= 31)
              end),
            canonical_and_packed_kv_tests: $canonical_pass
          }
        }
      | if $cache_type == "f32" then
          .checks += {
            simd_attention: all($warm[]; .attention_kernel == "attention_decode_fused_gemma4_simd_f32"),
            bounded_resident_memory: all($warm[]; .resident_bytes <= 3489602512),
            prefill_median_at_least_50: (($prefill | median) >= 50),
            decode_median_at_least_40: (($decode | median) >= 40)
          }
        else . end
      | .pass = all(.checks[]; . == true)
    ' "$artifact_dir/$cache_type/chat-performance.jsonl" \
        | tee "$artifact_dir/$cache_type/summary.json"
done

summary="$artifact_dir/acceptance-summary.json"
jq -n \
    --slurpfile f32 "$artifact_dir/f32/summary.json" \
    --slurpfile q8 "$artifact_dir/q8_0/summary.json" \
    --slurpfile q4 "$artifact_dir/q4_0/summary.json" \
    --slurpfile f32_benchmark "$artifact_dir/f32/benchmark-summary.json" \
    --slurpfile q8_benchmark "$artifact_dir/q8_0/benchmark-summary.json" \
    --slurpfile q4_benchmark "$artifact_dir/q4_0/benchmark-summary.json" \
    '{
      modes:{f32:$f32[0],q8_0:$q8[0],q4_0:$q4[0]},
      fixed_workload:{f32:$f32_benchmark[0],q8_0:$q8_benchmark[0],q4_0:$q4_benchmark[0]},
      phase_acceptance_pass:($f32[0].pass and $f32_benchmark[0].pass),
      experimental_modes_valid:(
        $q8[0].pass and $q4[0].pass and $q8_benchmark[0].pass and $q4_benchmark[0].pass
        and ($q8_benchmark[0].kv_cache_bytes < $f32_benchmark[0].kv_cache_bytes)
        and ($q4_benchmark[0].kv_cache_bytes < $f32_benchmark[0].kv_cache_bytes)
      )
    }' \
    | tee "$summary"

echo "Acceptance artifact: $summary"
if jq -e '.phase_acceptance_pass == true' "$summary" >/dev/null; then
    echo "GEMMA 4 RESIDENT PERFORMANCE ACCEPTANCE: PASS"
else
    echo "GEMMA 4 RESIDENT PERFORMANCE ACCEPTANCE: FAIL (F32 phase gate)" >&2
    exit 1
fi
