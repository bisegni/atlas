# llama.cpp-Matched Benchmark Instrumentation for Atlas

## Purpose

Atlas currently runs the same model and prompts more slowly than llama.cpp. To close that gap, Atlas must first produce measurements that are directly comparable with llama.cpp instead of relying on aggregate chat wall time or partially matched benchmarks.

This document defines the instrumentation, benchmark contract, JSON output, CLI behavior, and acceptance rules required to compare Atlas against llama.cpp using the same:

- GGUF model bytes;
- prompt token IDs;
- prompt length;
- generated-token count;
- context length;
- KV-cache format;
- prefill and decode configuration;
- warmup policy;
- repetition count;
- Apple Silicon hardware;
- power and thermal conditions.

The immediate goal is not to copy llama.cpp output formatting exactly. The goal is to expose equivalent values with enough additional Atlas-specific detail to explain the performance gap.

---

## Main requirement

Atlas must report the same high-level benchmark categories commonly used by llama.cpp:

```text
pp = prompt processing / prefill
 tg = token generation / decode
 pg = prompt processing followed by token generation
```

Atlas must also report the internal evidence required to explain why its result differs:

```text
GPU elapsed time
host wall time
CPU wait time
command buffers
dispatch count
per-kernel-family time
weight upload bytes
readback bytes
resident bytes
peak resident bytes
KV-cache bytes
allocations during measured execution
selected formats and kernels
```

---

## Terminology

### Prompt processing / prefill

The execution that consumes all prompt tokens and initializes the model state and KV cache before the first generated token.

Report:

```text
prefill_tokens
prefill_time_ms
prefill_tokens_per_second
time_to_first_token_ms
```

### Token generation / decode

The repeated one-token-at-a-time execution after prefill.

Report:

```text
decode_tokens
decode_time_ms
decode_tokens_per_second
milliseconds_per_token
```

### Combined prompt + generation

The complete matched workload.

Report:

```text
end_to_end_time_ms
end_to_end_tokens_per_second
host_wall_time_ms
```

### Warmup

Warmup work is executed before measured repetitions and excluded from the benchmark result.

Warmup must be reported explicitly.

### Measured run

One complete benchmark repetition after warmup.

---

## Strict matched-workload contract

A comparison is valid only if Atlas and llama.cpp use identical or explicitly equivalent values for all relevant settings.

### Model identity

Record:

```text
model_path
model_file_size
model_sha256
GGUF metadata hash
tensor manifest hash
model architecture
model quantization inventory
```

The path alone is not sufficient.

### Prompt identity

The strongest comparison uses exact prompt token IDs rather than only the same source text.

Record:

```text
prompt_source_text_sha256
prompt_token_ids
prompt_token_count
prompt_token_sha256
BOS policy
EOS policy
chat template identity
```

Atlas should support a benchmark mode that accepts pre-tokenized input so both engines can consume an equivalent token sequence.

### Decode identity

Record:

```text
greedy decoding enabled
temperature
top-k
top-p
seed
maximum generated tokens
stop conditions
EOS handling
```

For matched performance benchmarks use deterministic greedy decoding unless a separate sampling benchmark is explicitly requested.

### Context and cache identity

Record:

```text
maximum context
current prompt length
KV-cache format
KV-cache layout
KV-cache bytes
attention implementation
sliding-window configuration
```

A comparison with different KV-cache formats is not a strict engine comparison.

### Prefill identity

Record:

```text
prefill batch size
prefill micro-batch size
prefill chunk size
prefill chunks
prefill path
prefill token order
```

### Hardware identity

Record:

```text
Apple chip identifier
GPU device name
GPU registry ID
GPU family
GPU core count when available
unified memory bytes
recommended maximum working set
macOS version
Metal version
power source
thermal state when available
```

### Software identity

For Atlas:

```text
Atlas version
Atlas Git commit
Rust toolchain
build profile
feature flags
Metal library hash
kernel registry hash
executor configuration
```

For llama.cpp:

```text
llama.cpp Git commit
build type
Metal enabled
relevant compile options
benchmark command line
```

---

## Benchmark matrix

Atlas should implement a small standard matrix matching the typical llama.cpp split between prompt processing and generation.

### Prompt-processing workloads

Recommended initial set:

```text
pp32
pp128
pp512
pp1024
pp2048
```

Each workload processes the configured number of prompt tokens and generates no measured decode tokens.

### Token-generation workloads

Recommended initial set:

```text
tg32
tg64
tg128
tg256
```

Each workload uses a fixed prompt, performs prefill as setup, then measures only the requested decode tokens.

### Combined workloads

Recommended initial set:

```text
pp128_tg128
pp512_tg128
pp1024_tg128
pp2048_tg256
```

### Real-prompt workloads

In addition to synthetic fixed-length workloads, include real deterministic prompts used by Atlas acceptance tests:

```text
short chat prompt
medium chat prompt
long context prompt
vocabulary stress prompt
EOS-sensitive prompt
```

Synthetic and real-prompt results must be reported separately.

---

## Timing boundaries

Timing boundaries must be precise and stable.

### Model-load timing

Do not include model loading in pp, tg, or pg throughput.

Report it separately:

```text
model_open_ms
GGUF_parse_ms
plan_load_ms
weight_conversion_ms
weight_upload_ms
pipeline_creation_ms
resident_executor_build_ms
```

### Prefill timing

Start prefill timing immediately before the first command encoding or production execution step required for prompt processing.

Stop timing when:

- all prompt tokens have been processed;
- the KV cache is ready;
- the state required for the first decode token is ready.

Do not include tokenization.

### Time to first token

Measure from the start of prefill until the first generated token ID is available to the caller.

Report separately from pure prefill because it may include:

- first decode step;
- token selection;
- synchronization;
- readback.

### Decode timing

Start after prefill and any explicitly excluded warmup decode tokens.

Stop after exactly the configured number of measured decode tokens or earlier EOS only when the workload is specifically EOS-terminated.

For fixed-window performance comparisons, use prompts and policies that ensure the requested decode window is completed.

### Host wall timing

Use a monotonic high-resolution clock around the exact measured section.

### GPU timing

Where Metal provides valid GPU start/end timestamps, report command-buffer GPU elapsed time.

GPU elapsed time is diagnostic evidence and must not replace host wall time.

---

## Repetition and warmup policy

Recommended default:

```text
warmup repetitions: 1
measured repetitions: 5
```

For very short workloads, use more repetitions or aggregate multiple iterations inside one measured run.

Report every sample, not only the final mean.

Calculate:

```text
minimum
maximum
mean
median
standard deviation
coefficient of variation
```

Promotion or performance conclusions should prefer median paired results.

### Alternating engine order

When running Atlas and llama.cpp on the same machine, alternate order to reduce thermal and temporal bias:

```text
Atlas
llama.cpp
llama.cpp
Atlas
Atlas
llama.cpp
```

Or use a deterministic randomized order and store the seed.

---

## Required high-level metrics

For every benchmark repetition, record:

```text
prompt_tokens
generated_tokens
measured_generated_tokens
prefill_time_ms
prefill_tokens_per_second
decode_time_ms
decode_tokens_per_second
milliseconds_per_generated_token
time_to_first_token_ms
end_to_end_time_ms
host_wall_time_ms
```

Formulas:

```text
prefill_tokens_per_second = prompt_tokens / prefill_seconds

decode_tokens_per_second = measured_generated_tokens / decode_seconds

milliseconds_per_generated_token = decode_time_ms / measured_generated_tokens

end_to_end_tokens_per_second =
    (prompt_tokens + generated_tokens) / end_to_end_seconds
```

Do not combine prefill and decode into one tok/s number without also reporting them separately.

---

## Required Atlas execution metrics

### Command submission

Record:

```text
command_buffers_total
prefill_command_buffers
decode_command_buffers
command_buffers_per_decode_token
encoders_total
```

### Dispatches

Record:

```text
dispatches_total
prefill_dispatches
decode_dispatches
dispatches_per_decode_token
dispatches_per_layer_per_token
```

### Synchronization

Record:

```text
CPU_wait_events
CPU_wait_time_ms
GPU_idle_gap_estimate_ms
command_buffer_commit_to_complete_ms
readback_wait_time_ms
```

### Upload and readback

Record:

```text
initial_weight_upload_bytes
prefill_upload_bytes
decode_upload_bytes
decode_upload_bytes_per_token
prefill_readback_bytes
decode_readback_bytes
decode_readback_bytes_per_token
```

The expected steady-state decode values should normally be:

```text
weight upload bytes per token = 0
format conversion bytes per token = 0
```

Any nonzero value must be explained.

### Memory and residency

Record:

```text
model_file_bytes
resident_weight_bytes
converted_weight_bytes
shared_weight_bytes
KV_cache_bytes
compute_buffer_bytes
peak_resident_bytes
steady_state_resident_bytes
number_of_resident_buffers
number_of_temporary_buffers
```

### Allocations

Record allocations occurring during the measured region:

```text
Metal buffer allocations
Metal buffer reallocations
host heap allocations when measurable
pipeline-state creation
format conversions
buffer resizing
```

The ideal steady-state decode region has no allocations, pipeline creation, or weight conversion.

---

## Per-kernel-family instrumentation

Atlas should aggregate GPU and host-visible time by operation family.

Recommended families:

```text
embedding
RMS normalization
QKV projection
attention score/value path
KV append
attention output projection
FFN gate/up
activation and gate multiply
FFN down
final normalization
output projection / LM head
token selection
miscellaneous copy or conversion
```

For each family report:

```text
dispatch_count
GPU_time_ms
host_encode_time_ms
bytes_read_estimate
bytes_written_estimate
selected_kernel_names
selected_quantization_formats
```

Per-kernel data may be sampled or enabled in a diagnostic mode if full timing perturbs execution significantly.

### Instrumentation overhead

Atlas must measure and report instrumentation overhead.

Provide two modes:

```text
benchmark mode:
    minimal counters and stable timing

diagnostic mode:
    detailed per-kernel timing and execution traces
```

Performance comparison numbers should come from benchmark mode.

Diagnostic mode is used to explain the difference.

---

## llama.cpp comparison record

Atlas should support importing a llama-bench JSON result or a normalized external reference record.

Suggested structure:

```json
{
  "reference_engine": "llama.cpp",
  "reference_commit": "...",
  "model_sha256": "...",
  "workload_id": "pp512_tg128",
  "prompt_tokens": 512,
  "generated_tokens": 128,
  "prefill_tokens_per_second": 0.0,
  "decode_tokens_per_second": 0.0,
  "prefill_time_ms": 0.0,
  "decode_time_ms": 0.0,
  "samples": []
}
```

Atlas should then calculate:

```text
prefill_gap_ratio = Atlas prefill time / llama.cpp prefill time

decode_gap_ratio = Atlas decode time / llama.cpp decode time

prefill_speed_ratio = Atlas prefill tok/s / llama.cpp prefill tok/s

decode_speed_ratio = Atlas decode tok/s / llama.cpp decode tok/s
```

Interpretation:

```text
time gap ratio 2.0     Atlas takes twice as long
speed ratio 0.5        Atlas runs at half the throughput
```

---

## Proposed Atlas CLI

### Matched benchmark

```bash
cargo run --release -p atlas-cli -- benchmark matched \
  --model models/model.gguf \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --warmup-runs 1 \
  --runs 5 \
  --greedy \
  --kv-cache-type q4_0 \
  --output artifacts/benchmarks/atlas-pp512-tg128.json
```

### Pre-tokenized prompt

```bash
cargo run --release -p atlas-cli -- benchmark matched \
  --model models/model.gguf \
  --prompt-token-file artifacts/prompts/pp512.tokens.json \
  --decode-tokens 128 \
  --runs 5 \
  --output artifacts/benchmarks/atlas-pp512-tg128.json
```

### Diagnostic benchmark

```bash
cargo run --release -p atlas-cli -- benchmark matched \
  --model models/model.gguf \
  --prompt-token-file artifacts/prompts/pp512.tokens.json \
  --decode-tokens 128 \
  --runs 3 \
  --diagnostic-kernel-timing \
  --output artifacts/benchmarks/atlas-pp512-tg128-diagnostic.json
```

### Compare with llama.cpp result

```bash
cargo run --release -p atlas-cli -- benchmark compare \
  --atlas artifacts/benchmarks/atlas-pp512-tg128.json \
  --reference artifacts/benchmarks/llama-pp512-tg128.json \
  --output artifacts/benchmarks/comparison-pp512-tg128.json
```

---

## Proposed JSON schema

```json
{
  "schema_version": 1,
  "benchmark_kind": "matched_engine_comparison",
  "engine": "atlas",
  "executor": "resident_metal",
  "mode": "benchmark",

  "identity": {
    "model_sha256": "...",
    "tensor_manifest_sha256": "...",
    "atlas_git_commit": "...",
    "metal_library_sha256": "...",
    "kernel_registry_sha256": "...",
    "hardware": {
      "chip": "...",
      "device_name": "...",
      "registry_id": 0,
      "unified_memory_bytes": 0,
      "macos_version": "...",
      "metal_version": "..."
    }
  },

  "workload": {
    "id": "pp512_tg128",
    "prompt_token_count": 512,
    "prompt_token_sha256": "...",
    "decode_tokens": 128,
    "greedy": true,
    "context_limit": 4096,
    "KV_cache_type": "q4_0",
    "prefill_batch_size": 512,
    "prefill_micro_batch_size": 128,
    "prefill_chunk_size": 128,
    "warmup_runs": 1,
    "measured_runs": 5
  },

  "samples": [
    {
      "run_index": 0,
      "prefill_time_ms": 0.0,
      "prefill_tokens_per_second": 0.0,
      "decode_time_ms": 0.0,
      "decode_tokens_per_second": 0.0,
      "milliseconds_per_token": 0.0,
      "time_to_first_token_ms": 0.0,
      "end_to_end_time_ms": 0.0,
      "GPU_elapsed_ms": 0.0,
      "CPU_wait_time_ms": 0.0,
      "command_buffers": 0,
      "dispatches": 0,
      "upload_bytes": 0,
      "readback_bytes": 0,
      "resident_bytes": 0,
      "peak_resident_bytes": 0
    }
  ],

  "summary": {
    "prefill_time_ms": {
      "minimum": 0.0,
      "maximum": 0.0,
      "mean": 0.0,
      "median": 0.0,
      "standard_deviation": 0.0
    },
    "prefill_tokens_per_second": {
      "mean": 0.0,
      "median": 0.0
    },
    "decode_time_ms": {
      "minimum": 0.0,
      "maximum": 0.0,
      "mean": 0.0,
      "median": 0.0,
      "standard_deviation": 0.0
    },
    "decode_tokens_per_second": {
      "mean": 0.0,
      "median": 0.0
    }
  },

  "execution": {
    "selected_group_formats": [],
    "selected_kernels": {},
    "prefill_path": "...",
    "attention_kernel": "...",
    "command_buffers_per_decode_token": 0.0,
    "dispatches_per_decode_token": 0.0,
    "decode_upload_bytes_per_token": 0.0,
    "decode_readback_bytes_per_token": 0.0,
    "allocations_during_decode": 0
  },

  "parity": {
    "prompt_token_sha256": "...",
    "generated_token_sha256": "...",
    "measured_token_sha256": "...",
    "EOS_position": null,
    "finish_reason": "max_tokens"
  }
}
```

---

## Rust implementation outline

### Suggested modules

```text
atlas-cli/src/benchmark_matched.rs
atlas-model/src/benchmark_metrics.rs
atlas-metal/src/execution_counters.rs
atlas-metal/src/kernel_timing.rs
```

### Core structures

```rust
pub struct MatchedBenchmarkConfig {
    pub prompt_tokens: Vec<u32>,
    pub decode_tokens: usize,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub context_limit: usize,
    pub KV_cache_type: Gemma4KVCacheType,
    pub prefill_batch_size: usize,
    pub prefill_micro_batch_size: usize,
    pub diagnostic_kernel_timing: bool,
}
```

```rust
pub struct BenchmarkSample {
    pub prefill_time: Duration,
    pub decode_time: Duration,
    pub time_to_first_token: Duration,
    pub end_to_end_time: Duration,
    pub GPU_elapsed: Option<Duration>,
    pub CPU_wait_time: Duration,

    pub command_buffers: u64,
    pub dispatches: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,

    pub generated_token_ids: Vec<u32>,
    pub EOS_position: Option<usize>,
}
```

```rust
pub struct ExecutionCounters {
    pub command_buffers_total: AtomicU64,
    pub dispatches_total: AtomicU64,
    pub upload_bytes: AtomicU64,
    pub readback_bytes: AtomicU64,
    pub buffer_allocations: AtomicU64,
    pub pipeline_creations: AtomicU64,
    pub CPU_wait_nanoseconds: AtomicU64,
}
```

Counters must be reset at the start of every measured region.

### Per-operation tagging

Every Resident dispatch should carry a stable operation-family tag:

```rust
pub enum OperationFamily {
    Embedding,
    RmsNorm,
    QKVProjection,
    Attention,
    KVAppend,
    AttentionOutput,
    FfnGateUp,
    FfnActivation,
    FfnDown,
    FinalNorm,
    OutputProjection,
    TokenSelection,
    CopyOrConversion,
}
```

This allows diagnostic aggregation without depending only on kernel-name parsing.

---

## Benchmark execution workflow

```text
1. Load and validate model identity.
2. Load and validate the quantization plan if provided.
3. Build the Resident Metal executor.
4. Resolve exact prompt token IDs.
5. Run unmeasured warmup repetitions.
6. Reset counters.
7. Run one measured prefill.
8. Record prefill completion and first-decode boundary.
9. Run exactly N measured decode tokens.
10. Record host, GPU, synchronization, memory, upload, and readback metrics.
11. Store token and EOS hashes.
12. Tear down per-run state where necessary.
13. Repeat for the configured number of runs.
14. Calculate summary statistics.
15. Write JSON atomically.
```

For repeated runs, clearly choose and record whether the executor and resident weights are reused.

Recommended default:

```text
resident weights and compiled pipelines reused
KV cache and request state reset
no model reload between measured repetitions
```

Add a separate cold-start benchmark for model loading and plan application.

---

## Comparison workflow against llama.cpp

### Step 1: Produce prompt token artifact

Create a canonical token file:

```json
{
  "model_sha256": "...",
  "tokenizer_identity": "...",
  "prompt_token_ids": [1, 2, 3],
  "prompt_token_sha256": "..."
}
```

### Step 2: Run llama.cpp

Run llama-bench or an equivalent llama.cpp benchmark using matched model and workload parameters.

Preserve:

```text
full command line
llama.cpp commit
raw JSON or console output
hardware state
```

### Step 3: Run Atlas

Use the same model, token count, decode window, context, and cache policy.

### Step 4: Normalize results

Convert both results to a common comparison record.

### Step 5: Generate gap report

The report should show:

```text
prefill speed ratio
decode speed ratio
time-to-first-token ratio
end-to-end ratio
memory ratio
KV-cache ratio
command-buffer and dispatch evidence for Atlas
largest Atlas kernel-family time shares
```

---

## Gap-analysis report

Suggested output:

```json
{
  "workload_id": "pp512_tg128",
  "atlas": {
    "prefill_tokens_per_second": 0.0,
    "decode_tokens_per_second": 0.0,
    "time_to_first_token_ms": 0.0
  },
  "llama_cpp": {
    "prefill_tokens_per_second": 0.0,
    "decode_tokens_per_second": 0.0,
    "time_to_first_token_ms": 0.0
  },
  "gap": {
    "prefill_speed_ratio": 0.0,
    "decode_speed_ratio": 0.0,
    "prefill_time_ratio": 0.0,
    "decode_time_ratio": 0.0
  },
  "atlas_bottlenecks": [
    {
      "operation_family": "output_projection",
      "decode_time_share_percent": 0.0,
      "dispatches_per_token": 0.0,
      "selected_format": "q6_k",
      "selected_kernel": "..."
    }
  ]
}
```

---

## Acceptance requirements

The instrumentation is accepted when all of the following are true:

1. Atlas reports separate pp, tg, and combined metrics.
2. Prompt and generated token hashes are included.
3. Warmup is excluded from measured timing.
4. Model load is excluded from pp and tg.
5. At least five measured samples can be emitted.
6. Mean, median, standard deviation, minimum, and maximum are calculated correctly.
7. Host wall time is always present.
8. GPU elapsed time is present when supported and clearly marked when unavailable.
9. Command-buffer, dispatch, upload, readback, and residency counters are emitted.
10. Measured decode reports whether allocations or pipeline creation occurred.
11. The same benchmark can be reproduced from a saved prompt-token artifact.
12. Atlas can import or normalize a llama.cpp result.
13. Atlas produces an explicit speed and time ratio against llama.cpp.
14. Diagnostic instrumentation can be disabled for clean benchmark numbers.
15. Enabling minimal benchmark counters changes throughput by less than an accepted small threshold, initially 1%.

---

## Tests

### Unit tests

```text
throughput formulas
milliseconds-per-token formula
summary statistics
JSON serialization
prompt-token digest
model identity digest
counter reset
operation-family aggregation
```

### Integration tests

```text
fixed pp-only benchmark
fixed tg-only benchmark
combined pp+tg benchmark
warmup excluded
model loading excluded
exact generated-token window
EOS-terminated workload
fixed-window workload
cached plan versus explicit plan
no allocation during decode
no weight upload during decode
```

### Apple Silicon acceptance artifacts

For every supported Apple Silicon family, preserve:

```text
Atlas benchmark JSON
llama.cpp raw benchmark result
normalized comparison JSON
model SHA
prompt token artifact
Atlas and llama.cpp commits
hardware and OS identity
Metal System Trace or GPU capture for one representative workload
```

---

## Implementation order

### Phase 1: High-level matched metrics

Implement:

```text
pp timing
tg timing
combined timing
time to first token
five-run statistics
prompt and generated token hashes
JSON output
```

### Phase 2: Resident execution counters

Implement:

```text
command buffers
dispatches
uploads
readbacks
resident memory
KV-cache bytes
allocations
CPU wait time
```

### Phase 3: Per-operation diagnostics

Implement:

```text
operation-family tags
GPU time by family
host encoding time by family
dispatch count by family
selected kernel and format map
```

### Phase 4: llama.cpp import and comparison

Implement:

```text
llama-bench result importer
normalized common schema
speed and time ratios
gap report
```

### Phase 5: Preflight integration

Use the matched benchmark data to guide Atlas preflight:

```text
identify largest time-share groups
identify groups farthest from the external performance reference
rank kernel, fusion, layout, and quantization experiments
store benchmark evidence in the Atlas plan artifact
```

---

## Final rule

Atlas should not conclude that a quantization change, kernel, or execution plan closes the llama.cpp gap unless the result is measured with:

```text
the same model
the same prompt tokens
the same token windows
the same context and KV configuration
matched warmup and repetitions
separate prefill and decode timing
complete Resident Metal execution
exact token and EOS evidence
```

The benchmark system must answer two separate questions:

1. **How much slower or faster is Atlas than llama.cpp for the same workload?**
2. **Which Atlas operation families, synchronization points, memory behaviors, or kernel choices explain the difference?**

Only after both questions are measurable should the preflight use llama.cpp as an external performance reference for prioritizing optimizations.
