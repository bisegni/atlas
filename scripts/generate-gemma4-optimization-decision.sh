#!/usr/bin/env bash
# Turn clean Phase 12.3 baseline artifacts into the mandatory candidate gate.
set -euo pipefail

usage() {
    echo "usage: $0 --baseline DIR --context-sweep DIR [--output DIR]" >&2
    exit 2
}

baseline_dir=
sweep_dir=
output_dir=
while (($#)); do
    case "$1" in
        --baseline) shift; baseline_dir=${1:-} ;;
        --context-sweep) shift; sweep_dir=${1:-} ;;
        --output) shift; output_dir=${1:-} ;;
        *) usage ;;
    esac
    shift
done
[[ -n "$baseline_dir" && -n "$sweep_dir" ]] || usage
[[ -f "$baseline_dir/baseline-profile-summary.json" ]] || { echo "missing baseline summary" >&2; exit 2; }
[[ -f "$sweep_dir/context-sweep-summary.json" ]] || { echo "missing context sweep summary" >&2; exit 2; }
output_dir=${output_dir:-"$baseline_dir"}
mkdir -p "$output_dir"

jq -n \
  --slurpfile baseline "$baseline_dir/baseline-profile-summary.json" \
  --slurpfile sweep "$sweep_dir/context-sweep-summary.json" '
  def coverage($d): {
    dispatch: ($d.measured_decode.categorized_dispatches / ($d.measured_decode.dispatches | if . == 0 then 1 else . end)),
    gpu: ($d.measured_decode.categorized_gpu_ns / ($d.measured_decode.attributed_gpu_duration_ns | if . == 0 then 1 else . end))
  };
  ($baseline[0]) as $b | ($sweep[0]) as $s |
  ($b.diagnostic.short) as $short | ($b.diagnostic.long) as $long |
  ($s.contexts | sort_by(.warmup_decode_tokens)) as $points |
  (($points[-1].median.measured_decode_gpu_ms - $points[0].median.measured_decode_gpu_ms) /
   (($points[-1].warmup_decode_tokens - $points[0].warmup_decode_tokens) | if . == 0 then 1 else . end)) as $gpu_ms_per_context_token |
  ([$short.top_operation_families[], $long.top_operation_families[]] | group_by(.key) |
    map({key: .[0].key, gpu_ns: map(.gpu_ns) | add}) | sort_by(.gpu_ns) | reverse | .[0]) as $top |
  {
    schema_version: 1,
    purpose: "Phase 12.3 hypothesis-gated Resident optimization admission record",
    fixture: {path: $b.fixture, sha256: $b.fixture_sha256},
    token_hashes: {short: $b.benchmark.short.records[0] | {prompt_token_sha256, generated_token_sha256, measured_generated_token_sha256, first_eos_position}, long: $b.benchmark.long.records[0] | {prompt_token_sha256, generated_token_sha256, measured_generated_token_sha256, first_eos_position}},
    kernel_plan: {short: $b.benchmark.short.selected_kernels, long: $b.benchmark.long.selected_kernels},
    resident_accounting: {short: $b.benchmark.short.accounting, long: $b.benchmark.long.accounting},
    measured_decode_coverage: {short: coverage($short), long: coverage($long)},
    context_sweep: {points: $points, gpu_ms_per_warmup_token: $gpu_ms_per_context_token},
    supported_counter_set: {short: $short.gpu_counter_capture, long: $long.gpu_counter_capture},
    largest_common_hotspot: $top,
    classification: "unclassified: counter and occupancy evidence must identify memory_cache_pressure, arithmetic_dequantization, occupancy_resource_exhaustion, or synchronization_dependency",
    amdahl_upper_bound: {campaign_target: "exceed the matched short-context baseline without changing production defaults", candidate_specific_bound: "must be computed from the classified common hotspot before implementation"},
    admission: {
      coverage_at_least_95_percent: ((coverage($short).dispatch >= .95 and coverage($short).gpu >= .95 and coverage($long).dispatch >= .95 and coverage($long).gpu >= .95)),
      stable_accounting: ($b.benchmark.short.checks.stable_resident and $b.benchmark.short.checks.stable_kv and $b.benchmark.short.checks.stable_upload and $b.benchmark.short.checks.stable_readback and $b.benchmark.long.checks.stable_resident and $b.benchmark.long.checks.stable_kv and $b.benchmark.long.checks.stable_upload and $b.benchmark.long.checks.stable_readback),
      approved: false,
      rationale: "Admit only an opt-in graph-cost candidate whose measured-decode coverage, accounting, and limiter-specific hypothesis are recorded. RMSNorm and attention require counter classification; MQA-tiled attention is permanently rejected."
    }
  }' > "$output_dir/optimization-decision.json"

jq -r '"# Phase 12.3 optimization decision\n\n" +
  "Status: **REJECT new candidate**\n\n" +
  "- Largest common hotspot: `" + .largest_common_hotspot.key + "`\n" +
  "- Context slope: " + (.context_sweep.gpu_ms_per_warmup_token | tostring) + " GPU ms/warmup token\n" +
  "- Rationale: " + .admission.rationale + "\n"' "$output_dir/optimization-decision.json" > "$output_dir/optimization-decision.md"

echo "Decision JSON: $output_dir/optimization-decision.json"
echo "Decision Markdown: $output_dir/optimization-decision.md"
