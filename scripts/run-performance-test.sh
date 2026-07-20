#!/usr/bin/env bash
# Run the reusable resident performance suite without mixing FP32, Q4, and Q8
# chat records in the append-only shared performance log.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

prompt='The capital of france is'
max_tokens=32
stamp=$(date -u +%Y%m%dT%H%M%SZ)
artifact_dir="artifacts/performance/${stamp}"
shared_log="artifacts/chat-performance.jsonl"
mkdir -p "$artifact_dir"

run_chat() {
    local model=$1
    local before_lines=0
    printf 'Running resident chat for %s...\n' "$model"
    if [[ -f "$shared_log" ]]; then
        before_lines=$(wc -l < "$shared_log")
    fi

    cargo run -p atlas-cli -- chat \
        --model "$model" \
        --prompt "$prompt" \
        --max-tokens "$max_tokens" \
        2>&1 | tee "$artifact_dir/${model}.chat.log"

    local after_lines
    after_lines=$(wc -l < "$shared_log")
    if (( after_lines != before_lines + 1 )); then
        echo "expected one new chat performance record for ${model}, found $((after_lines - before_lines))" >&2
        exit 1
    fi
    sed -n "$((before_lines + 1))p" "$shared_log" > "$artifact_dir/${model}.performance.jsonl"
}

cargo test -p atlas-model --test phase_06_executors \
    phase_12a_larger_q4_profile_reports_decode_kernel_costs \
    -- --ignored --nocapture \
    2>&1 | tee "$artifact_dir/larger-q4.kernel-profile.log"

cargo test -p atlas-model --test phase_06_executors \
    phase_12a_larger_q8_profile_reports_decode_kernel_costs \
    -- --ignored --nocapture \
    2>&1 | tee "$artifact_dir/larger-q8.kernel-profile.log"

run_chat larger-q4
run_chat larger-q8
# FP32 needs roughly 7.25 GB of resident memory for this fixture and can take
# much longer before the first streamed token. Run it last so its cold-start
# cost cannot hide the completed quantized evidence.
run_chat larger

printf 'Resident performance artifacts: %s\n' "$artifact_dir"
