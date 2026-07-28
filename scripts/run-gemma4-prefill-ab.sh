#!/usr/bin/env bash
# Compare two Gemma Resident prefill configurations on the exact same
# acceptance workload.  Each argument is either one environment assignment
# (NAME=value) or `-` to run with no extra environment variable.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

baseline_env=${1:-ATLAS_GEMMA4_PREFILL_TOKEN_MAJOR=1}
candidate_env=${2:--}
acceptance_script=scripts/run-gemma4-performance-acceptance.sh
stamp=$(date -u +%Y%m%dT%H%M%SZ)
comparison_dir="artifacts/phase-12a-prefill-ab/${stamp}"
mkdir -p "$comparison_dir"

if [[ ! -x "$acceptance_script" ]]; then
    echo "missing executable acceptance runner: $acceptance_script" >&2
    exit 2
fi

run_configuration() {
    local label=$1
    local env_assignment=$2
    local log="$comparison_dir/${label}.log"
    local status summary

    echo "Running ${label} prefill configuration (${env_assignment})..."
    set +e
    if [[ "$env_assignment" == "-" ]]; then
        bash "$acceptance_script" 2>&1 | tee "$log"
        status=${PIPESTATUS[0]}
    else
        env "$env_assignment" bash "$acceptance_script" 2>&1 | tee "$log"
        status=${PIPESTATUS[0]}
    fi
    set -e

    summary=$(awk -F ': ' '/^Acceptance artifact:/ {artifact=$2} END {print artifact}' "$log")
    if [[ -z "$summary" || ! -f "$summary" ]]; then
        echo "${label}: acceptance runner did not produce a readable summary" >&2
        exit 1
    fi
    cp "$summary" "$comparison_dir/${label}-acceptance-summary.json"
    printf '%s\n' "$status" > "$comparison_dir/${label}-exit-status.txt"
}

run_configuration token_major "$baseline_env"
run_configuration layer_major "$candidate_env"

baseline_status=$(<"$comparison_dir/token_major-exit-status.txt")
candidate_status=$(<"$comparison_dir/layer_major-exit-status.txt")
summary="$comparison_dir/prefill-ab-summary.json"

jq -n \
    --arg baseline_env "$baseline_env" \
    --arg candidate_env "$candidate_env" \
    --argjson baseline_exit "$baseline_status" \
    --argjson candidate_exit "$candidate_status" \
    --slurpfile baseline "$comparison_dir/token_major-acceptance-summary.json" \
    --slurpfile candidate "$comparison_dir/layer_major-acceptance-summary.json" \
    '
      def q4_chat($summary): $summary.modes.q4_0;
      def q4_fixed($summary): $summary.fixed_workload.q4_0;
      ($baseline[0]) as $old
      | ($candidate[0]) as $new
      | (q4_chat($old)) as $old_chat
      | (q4_chat($new)) as $new_chat
      | (q4_fixed($old)) as $old_fixed
      | (q4_fixed($new)) as $new_fixed
      | {
          configurations: {
            token_major: {
              environment: $baseline_env,
              runner_exit_status: $baseline_exit,
              acceptance_artifact: "token_major-acceptance-summary.json",
              result: $old
            },
            layer_major: {
              environment: $candidate_env,
              runner_exit_status: $candidate_exit,
              acceptance_artifact: "layer_major-acceptance-summary.json",
              result: $new
            }
          },
          comparison: {
            same_fixed_prompt_tokens:
              ($old_fixed.records[0].prompt_token_sha256 == $new_fixed.records[0].prompt_token_sha256),
            same_fixed_token_stream:
              ($old_fixed.records[0].generated_token_sha256 == $new_fixed.records[0].generated_token_sha256),
            same_fixed_first_eos:
              ($old_fixed.records[0].first_eos_position == $new_fixed.records[0].first_eos_position),
            token_major_fixed_sha256: $old_fixed.records[0].generated_token_sha256,
            layer_major_fixed_sha256: $new_fixed.records[0].generated_token_sha256,
            token_major_fixed_prompt_token_sha256: $old_fixed.records[0].prompt_token_sha256,
            layer_major_fixed_prompt_token_sha256: $new_fixed.records[0].prompt_token_sha256,
            token_major_fixed_first_eos: $old_fixed.records[0].first_eos_position,
            layer_major_fixed_first_eos: $new_fixed.records[0].first_eos_position,
            token_major_prefill_tok_s: $old_chat.warm_summary.prefill_tok_s.median,
            layer_major_prefill_tok_s: $new_chat.warm_summary.prefill_tok_s.median,
            prefill_speedup_percent:
              (($new_chat.warm_summary.prefill_tok_s.median / $old_chat.warm_summary.prefill_tok_s.median - 1) * 100),
            token_major_decode_tok_s: $old_chat.warm_summary.decode_tok_s.median,
            layer_major_decode_tok_s: $new_chat.warm_summary.decode_tok_s.median,
            decode_speedup_percent:
              (($new_chat.warm_summary.decode_tok_s.median / $old_chat.warm_summary.decode_tok_s.median - 1) * 100)
          }
        }
      | .comparison.exact_stream_parity_pass =
          (.comparison.same_fixed_prompt_tokens and .comparison.same_fixed_token_stream and .comparison.same_fixed_first_eos)
      | .pass =
          ($baseline_exit == 0 and $candidate_exit == 0 and .comparison.exact_stream_parity_pass)
    ' | tee "$summary"

echo "Combined A/B artifact: $summary"
if jq -e '.pass == true' "$summary" >/dev/null; then
    echo "GEMMA PREFILL A/B: PASS (exact fixed stream parity retained)"
else
    echo "GEMMA PREFILL A/B: DIFFERENCE DETECTED" >&2
    exit 1
fi
