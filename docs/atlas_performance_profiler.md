# Atlas Performance Bottleneck Profiler

## Goal

Atlas needs a profiling tool that answers one practical question:

> Where is Atlas losing time compared with llama.cpp, and which optimization should be implemented first?

A benchmark that reports only total tokens per second is not enough. The profiler must attribute time to execution phases, layer families, Metal kernel families, command submission, synchronization, uploads, readbacks, allocations, and memory pressure.

The profiler should complement `docs/llama_cpp_matched_benchmark_instrumentation.md`:

- matched benchmark mode produces trustworthy comparable numbers;
- profiler mode explains the difference;
- the automatic comparison script combines both outputs into a prioritized optimization report.

## Required operating modes

### Benchmark mode

Low-overhead instrumentation suitable for performance comparisons.

Collect:

```text
prefill time
decode time
time to first token
host wall time
GPU elapsed time
command-buffer count
dispatch count
upload/readback bytes
resident and peak memory
```

### Diagnostic mode

Higher-overhead instrumentation used to locate bottlenecks.

Collect:

```text
per-layer time
per-operation-family time
per-kernel time
host encode time
CPU wait time
Metal command-buffer latency
allocation and buffer-resize events
estimated bytes read and written
```

Diagnostic-mode timings must not be used as final throughput numbers unless instrumentation overhead is measured and accepted.

### Measurement-scope accounting contract

The JSON report uses schema version `4`. Every recorded event and aggregate is
assigned an explicit scope. The default hotspot scope is
`decode_measured`; decode warmup is reported separately and is excluded from
the default rankings. The report exposes `prefill`, `decode_warmup`,
`decode_measured`, `decode_complete`, and `profiler_overhead` summaries.

Each kernel event also carries a conservative bound-span traffic estimate. The
estimate sums the remaining bytes of bound Resident buffers as reads and uses
the common third binding as the output-write estimate. It is intended for
relative hotspot ranking only; it is not a Metal bandwidth-counter result.

Dispatch metrics are deliberately
separate:

```text
dispatch_calls          Metal compute-dispatch invocations
threadgroups_dispatched total threadgroups submitted by those invocations
threads_dispatched      logical threads requested by those threadgroups
```

All values are phase-local telemetry deltas. A report also includes timed and
untimed dispatch calls, categorized and uncategorized calls, categorized and
uncategorized GPU nanoseconds, and explicit dispatch/GPU attribution coverage.
Attribution coverage below 75% is warned as incomplete attribution and coverage
below 90% emits an additional warning. Collection status remains `complete`
when all requested scopes and report sections were collected. Unknown
operations and kernels remain in the report as `other` or `unattributed`; they
are never removed from totals.

Production-boundary GPU elapsed time and exact per-dispatch attributed GPU
duration are separate fields. Categorized GPU share uses the exact attributed
duration denominator; it is not a host wall-time share.

Prefill and decode are independent workloads. Each phase reports wall time,
GPU time, CPU encoding, CPU wait, upload/readback time, unexplained time,
throughput, dispatch calls, threadgroups, memory traffic, and hotspot rankings.
The reconciliation table is intentionally additive only for non-overlapping
intervals; any remaining wall time is reported as `unexplained_ns`.

Diagnostic mode uses two passes: a production-boundary Resident pass supplies
the benchmark-compatible wall time and phase telemetry, while a separate
exact pass supplies per-dispatch attribution. Benchmark mode never enables
per-dispatch command-buffer timing. The report records clock domains and
timing boundaries, and never presents summed overlapping GPU durations as a
host wall-time share. Such shares are emitted as `null` with a status.

The measured decode boundary is the same as the normal fixed-workload
benchmark: after warmup completes and before encoding the first measured
token, through availability of the final measured token. Token selection and
readback are included. Physical command-buffer counts are reported as
observed; the first generated token is selected by the prefill command buffer,
so a 32-token warmup plus 128-token measured window can have one fewer decode
forward command buffer than logical tokens.

Run a diagnostic report with:

```zsh
cargo run --release -p atlas-cli -- profile bottlenecks \
  --model gemma4-e2b-q4_0 \
  --prompt 'Explain Resident inference.' \
  --warmup-decode-tokens 32 \
  --decode-tokens 128 \
  --mode diagnostic \
  --output artifacts/profiles/atlas-profile.json
```

The command writes both `atlas-profile.json` and `atlas-profile.md`.

## Execution hierarchy

Every timed event should belong to the following hierarchy:

```text
run
  phase: model_load | prefill | decode | token_selection
    token_position, when applicable
      layer_index, when applicable
        operation_family
          kernel_name
            dispatch
```

Recommended operation families:

```text
embedding
rms_norm
qkv_projection
attention_score
attention_value
kv_append
attention_output_projection
ffn_gate_up
ffn_activation_multiply
ffn_down
final_norm
output_projection
argmax_or_token_selection
copy
conversion
synchronization
other
```

The Resident Gemma diagnostic path also attributes helper kernels to these
families when their kernel names provide sufficient evidence:

```text
embedding_lookup_*       embedding
vector_add_*             residual
gelu_* / vector_multiply_* ffn_activation_multiply
kv_append_*              kv_append
softcap_*                logit_softcap
argmax_*                 argmax_or_token_selection
scalar_multiply_*        conversion
```

Kernel names that do not have a conservative semantic mapping remain
`other`/unknown and reduce operation coverage. They are retained in the raw
event stream and reconciliation totals.

## Event schema

```rust
pub struct ProfileEvent {
    pub run_id: u64,
    pub repetition: u32,
    pub phase: ProfilePhase,
    pub token_position: Option<u32>,
    pub layer_index: Option<u32>,
    pub operation_family: OperationFamily,
    pub kernel_name: Option<String>,

    pub host_encode_start_ns: u64,
    pub host_encode_end_ns: u64,
    pub gpu_start_ns: Option<u64>,
    pub gpu_end_ns: Option<u64>,

    pub command_buffer_id: Option<u64>,
    pub encoder_id: Option<u64>,
    pub dispatch_id: Option<u64>,

    pub bytes_read_estimate: Option<u64>,
    pub bytes_written_estimate: Option<u64>,
    pub upload_bytes: u64,
    pub readback_bytes: u64,

    pub selected_format: Option<String>,
    pub selected_kernel: Option<String>,
}
```

The production implementation may use compact numeric IDs internally and resolve names only when writing the report.

## Required aggregate reports

### Phase summary

```text
model load
prefill
decode
token selection
CPU wait
unattributed host overhead
```

For each phase report:

```text
host wall time
GPU elapsed time
percentage of total time
command buffers
dispatches
uploads
readbacks
allocations
```

### Operation-family summary

Example:

```text
Family                    GPU ms/token    % decode    Dispatch/token
QKV projection                 1.80          18.0          34
Attention                      1.20          12.0          51
FFN gate/up                    2.40          24.0          17
FFN down                       1.70          17.0          17
Output projection              1.90          19.0           1
Norm and other                 0.50           5.0          68
CPU wait/readback              0.50           5.0           -
```

### Per-layer summary

Report for every layer:

```text
QKV GPU time
attention GPU time
gate/up GPU time
down GPU time
dispatch count
estimated memory bytes
```

This exposes unusually slow layers, shape-specific kernel problems, and missed fusion opportunities.

### Kernel summary

For each kernel name:

```text
calls
GPU total time
GPU mean time
GPU median time
host encode time
percentage of phase
input shapes
quantization formats
threadgroup configuration
```

### Synchronization summary

Report:

```text
number of CPU waits
total CPU wait time
wait time per generated token
readback wait time
command-buffer queue latency
GPU idle gap estimate
```

Resident Metal telemetry reports command-buffer schedule time and GPU idle
gaps between completed command buffers when Metal supplies valid GPU start/end
timestamps. GPU-wait-for-CPU and transfer-specific waits remain nullable when
the API does not expose a reliable interval; the report must show them as
unavailable rather than zero.

### Memory and allocation summary

Report:

```text
resident weight bytes
converted weight bytes
KV-cache bytes
compute scratch bytes
peak working set
Metal buffer allocations during prefill
Metal buffer allocations during decode
buffer resizes during decode
pipeline creations during measured work
```

Steady-state decode should ideally have:

```text
zero weight uploads
zero conversions
zero pipeline creations
zero buffer resizes
zero avoidable allocations
```

## Bottleneck classification

The profiler should classify each major cost.

### Compute-bound candidate

Signals:

```text
high GPU occupancy or arithmetic work
low synchronization overhead
kernel dominates GPU time
small improvement from reducing bytes
```

Suggested action:

```text
optimize SIMD-group work
increase vectorization
reduce instructions
use fused arithmetic
improve reduction strategy
```

### Memory-bandwidth-bound candidate

Signals:

```text
large estimated bytes read
low arithmetic intensity
quantized matvec dominates
performance scales with weight size
```

Suggested action:

```text
better packed layout
faster dequantization
coalesced reads
reuse scales and inputs
more aggressive safe quantization
```

### Dispatch-bound candidate

Signals:

```text
very many short kernels
high host encode time
low GPU duration per dispatch
large dispatches-per-token count
```

Suggested action:

```text
fuse QKV
fuse gate/up
fuse activation and multiply
batch operations
reduce encoders and barriers
```

### Synchronization-bound candidate

Signals:

```text
large CPU wait time
GPU idle gaps
one wait per token
large readback latency
```

Suggested action:

```text
reduce waits
pipeline command buffers
move token reduction to GPU
read back only selected token data
avoid full-logit readback
```

### Allocation-bound candidate

Signals:

```text
allocations or resizes during decode
pipeline creation in measured work
repeated temporary-buffer construction
```

Suggested action:

```text
preallocate
reuse scratch buffers
cache pipelines
materialize converted weights once
```

### Prefill batching problem

Signals:

```text
prefill gap much larger than decode gap
small effective batch or micro-batch
repeated matvec instead of batch matmul
many prompt chunks and command buffers
```

Suggested action:

```text
increase batch size
implement batch projection kernels
improve chunking
dispatch layer-major matrix work
```

## Optimization priority score

The tool should rank optimization targets rather than merely listing timings.

Recommended first formula:

```text
priority_score =
    phase_time_share
  * measured_gap_factor
  * estimated_improvability
  * confidence
```

Where:

```text
phase_time_share = target time / total phase time
measured_gap_factor = max(1, Atlas phase time / llama.cpp phase time)
estimated_improvability = heuristic in [0, 1]
confidence = measurement stability in [0, 1]
```

Sensitivity-aware quantization may add:

```text
quantization_priority = priority_score / (numerical_risk + epsilon)
```

The tool must label heuristic scores as estimates. Promotion decisions continue to require real matched benchmarks and exact parity.

## Required final report

The user-facing report should begin with a concise diagnosis:

```text
Atlas decode is 2.02x slower than llama.cpp.
The largest measured costs are:
1. FFN gate/up: 24% of decode
2. Output projection: 19% of decode
3. QKV projection: 18% of decode
CPU wait/readback accounts for 5%.
```

Then produce prioritized actions:

```text
Priority 1: optimize/fuse FFN gate/up
Potential recoverable share: 10-18% of decode
Evidence: 17 dispatches/token, 2.4 ms/token, short GPU kernels

Priority 2: optimize Q6 output-projection kernel
Potential recoverable share: 8-15%
Evidence: 1.9 ms/token; Q4 is faster but parity-invalid

Priority 3: reduce normalization/dispatch overhead
Potential recoverable share: 3-6%
Evidence: 68 small dispatches/token
```

## JSON report

Suggested root schema:

```json
{
  "schema_version": 4,
  "engine": "atlas",
  "mode": "diagnostic",
  "profile_status": "complete",
  "scope_contract": {},
  "decode_scope": {},
  "measured_windows": {},
  "benchmark_compatibility": {},
  "profiler_overhead": {},
  "workload": {},
  "counters": {},
  "reconciliation": {},
  "coverage": {},
  "phases": [],
  "operation_families": [],
  "layers": [],
  "kernels": [],
  "synchronization": {},
  "memory": {},
  "recommendations": []
}
```

Recommendation entry:

```json
{
  "rank": 1,
  "target": "ffn_gate_up",
  "classification": "dispatch_bound",
  "priority_score": 0.81,
  "phase_time_share": 0.24,
  "estimated_recoverable_percent": [10.0, 18.0],
  "evidence": [
    "17 dispatches per decode token",
    "median GPU time 2.4 ms/token",
    "average dispatch duration is small"
  ],
  "suggested_actions": [
    "fuse gate and up projection",
    "fuse activation and multiply",
    "evaluate larger row groups"
  ]
}
```

## Proposed CLI

```bash
cargo run --release -p atlas-cli -- profile bottlenecks \
  --model models/model.gguf \
  --prompt-token-file artifacts/prompts/pp512.tokens.json \
  --decode-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --output artifacts/profiles/atlas-profile.json
```

Optional llama.cpp reference:

```bash
cargo run --release -p atlas-cli -- profile bottlenecks \
  --model models/model.gguf \
  --prompt-token-file artifacts/prompts/pp512.tokens.json \
  --decode-tokens 128 \
  --llama-reference artifacts/benchmarks/llama.json \
  --output artifacts/profiles/atlas-vs-llama-profile.json
```

## Implementation phases

### Phase 1

Implement low-overhead counters:

```text
phase timing
command buffers
dispatches
uploads/readbacks
resident memory
allocations
```

### Phase 2

Add operation-family attribution and per-layer aggregation.

### Phase 3

Add Metal GPU timestamps and kernel aggregation.

### Phase 4

Add synchronization and GPU idle-gap analysis.

### Phase 5

Add automatic bottleneck classification and recommendation ranking.

## Atlas implementation

The profiler is implemented in the `atlas-profiler` workspace crate and is
disabled by default. Metal runtime counters are collected at the shared
command and buffer boundaries, so normal Resident execution does not add
per-operation timers or change command submission. The current report command
is:

```zsh
cargo run --release -p atlas-cli -- profile bottlenecks \
  --model gemma4-e2b-q4_0 \
  --prompt 'Explain Resident inference.' \
  --warmup-decode-tokens 32 \
  --decode-tokens 128 \
  --mode diagnostic \
  --output artifacts/profiles/atlas-profile.json
```

This writes both `atlas-profile.json` and `atlas-profile.md`. Diagnostic mode
uses a production-boundary Resident pass for wall time plus a separate exact
per-dispatch pass for attribution; it must not be used as a throughput oracle.
The JSON schema is version `4`, and its default recommendations are scoped to
`decode_measured` only.

The JSON report preserves raw event values and aggregate counters for command
buffers, dispatches, host/GPU time, CPU wait time, uploads, readbacks,
allocations, resident memory, peak memory, and KV-cache memory. Markdown adds
human-readable phase summaries and ranked heuristic recommendations. A
recommendation is an investigation priority, not proof that an optimization
is safe or faster; matched Resident workloads and exact token parity remain
required for promotion.

Reports also contain a reconciliation section. `dispatch_calls` is the number
of Metal dispatch calls in the measured window; `threadgroups_dispatched` and
`threads_dispatched` are launch-size totals and are not interchangeable with
dispatch calls. Diagnostic mode reports timed versus untimed and categorized
versus uncategorized dispatches, plus dispatch and GPU timing coverage. A
profile with incomplete coverage is marked incomplete and includes warnings.
Prefill, decode warmup, measured decode, complete decode, and
host/synchronization summaries are reported separately. Categorized GPU share
is a GPU-duration ranking metric. `wall_time_share` is nullable and is only
populated when a valid non-overlapping host-wall contribution exists.
Kernel recommendations use conservative labels (`compute_candidate`,
`bandwidth_candidate`, `dispatch_overhead_candidate`,
`synchronization_candidate`, `mixed`, or `unknown`) and never claim a bound
without supporting measurements.

### Phase 6

Integrate matched llama.cpp results and calculate the remaining gap.

## Non-negotiable constraints

- Profiling must use the production Resident Metal executor.
- No Reference or CPU inference fallback.
- Benchmark and diagnostic modes must be clearly separated.
- Instrumentation overhead must be measured.
- No optimization should be declared successful without a clean matched benchmark.
- Exact generated-token and EOS parity remain required for candidate promotion.
