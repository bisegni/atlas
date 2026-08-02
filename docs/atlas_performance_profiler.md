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
  "schema_version": 1,
  "engine": "atlas",
  "mode": "diagnostic",
  "identity": {},
  "workload": {},
  "summary": {},
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

### Phase 6

Integrate matched llama.cpp results and calculate the remaining gap.

## Non-negotiable constraints

- Profiling must use the production Resident Metal executor.
- No Reference or CPU inference fallback.
- Benchmark and diagnostic modes must be clearly separated.
- Instrumentation overhead must be measured.
- No optimization should be declared successful without a clean matched benchmark.
- Exact generated-token and EOS parity remain required for candidate promotion.
