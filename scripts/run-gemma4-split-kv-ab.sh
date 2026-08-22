#!/usr/bin/env bash
# Matched Resident A/B for the opt-in split-KV decode attention candidate.
#
# `flash16_split_kv` is tolerance-level (the cross-threadgroup split + merge
# reorder the FP32 reduction, like the v4 default), so its greedy-stream hash
# is NOT equal to the baseline `flash16` — that is expected and is proven at
# the kernel level by `flash16_split_kv_parity.rs` (max-abs < 1e-3 vs the CPU
# oracle).  The A/B therefore gates on:
#   * candidate determinism — the same hash on all 5 candidate runs
#     (stable greedy stream, a drift diagnostic),
#   * unchanged first_eos_position, resident_bytes, kv_cache_bytes (residency
#     is identical across attention modes; the split scratch is unconditional),
#   * positive median decode tok/s on every workload.
# It does NOT require baseline/candidate hash equality.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

model_id=gemma4-e2b-q4_0
prompt='Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Keep the answer concise.'
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/split-kv-attention-ab/${stamp}"
mkdir -p "$artifact_dir"

run_mode() {
    local label=$1 mode=$2 workload=$3 warmup=$4 tokens=$5 context=$6
    local dir="$artifact_dir/$label/$workload"
    mkdir -p "$dir"
    for run in $(seq 1 5); do
        cargo run --release -p atlas-cli -- benchmark \
              --model "$model_id" --kv-cache-type q4_0 \
              --q4-attention-mode "$mode" --prompt "$prompt" \
              --warmup-decode-tokens "$warmup" --decode-tokens "$tokens" \
              --max-context "$context" > "$dir/$run.json"
    done
    jq -s '
      def median: sort | .[length / 2 | floor];
       { records: length,
        all_hashes: [.[].generated_token_sha256],
        distinct_hashes: ([.[].generated_token_sha256] | unique),
        deterministic: ([.[].generated_token_sha256] | unique | length == 1),
        median_decode_tok_s: ([.[].decode_tok_s] | median),
        executor: .[0].executor,
        q4_attention_mode: .[0].q4_attention_mode,
        attention_kernel: .[0].selected_kernels.attention,
        generated_token_sha256: .[0].generated_token_sha256,
        first_eos_position: .[0].first_eos_position,
        resident_bytes: .[0].resident_bytes,
        kv_cache_bytes: .[0].kv_cache_bytes }' "$dir"/*.json > "$dir/summary.json"
}

cargo run --release -p atlas-cli -- model verify --model "$model_id" \
      > "$artifact_dir/model-verify.json"

for workload in long short; do
    if [[ "$workload" == long ]]; then warmup=1024; tokens=512; context=2048
    else warmup=0; tokens=128; context=4096; fi
    run_mode baseline flash16 "$workload" "$warmup" "$tokens" "$context"
    run_mode candidate flash16_split_kv "$workload" "$warmup" "$tokens" "$context"
done

jq -n \
    --slurpfile base_long "$artifact_dir/baseline/long/summary.json" \
    --slurpfile cand_long "$artifact_dir/candidate/long/summary.json" \
    --slurpfile base_short "$artifact_dir/baseline/short/summary.json" \
    --slurpfile cand_short "$artifact_dir/candidate/short/summary.json" '
   def residually_parity($a; $b):
      $a.first_eos_position == $b.first_eos_position
     and $a.kv_cache_bytes == $b.kv_cache_bytes
     and $a.resident_bytes == $b.resident_bytes;
    def workload($base; $cand):
       { baseline: $base[0],
        candidate: {
           deterministic: $cand[0].deterministic,
           median_decode_tok_s: $cand[0].median_decode_tok_s,
           generated_token_sha256: $cand[0].generated_token_sha256,
           first_eos_position: $cand[0].first_eos_position,
           resident_bytes: $cand[0].resident_bytes,
           kv_cache_bytes: $cand[0].kv_cache_bytes } };
    { long: workload($base_long; $cand_long),
      short: workload($base_short; $cand_short) }
    | .long_speedup_percent = ((.long.candidate.median_decode_tok_s / .long.baseline.median_decode_tok_s - 1) * 100)
    | .short_speedup_percent = ((.short.candidate.median_decode_tok_s / .short.baseline.median_decode_tok_s - 1) * 100)
    | .deterministic = (.short.candidate.deterministic and .long.candidate.deterministic)
    | .residency_stable = (residually_parity(.short.baseline; .short.candidate)
                          and residually_parity(.long.baseline; .long.candidate))
    | .positive_speedup = (.long_speedup_percent > 0 and .short_speedup_percent > 0)
    | .pass = (.deterministic and .residency_stable and .positive_speedup)' \
    > "$artifact_dir/summary.json"

cat "$artifact_dir/summary.json"
