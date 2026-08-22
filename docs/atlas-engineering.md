# Atlas Engineering Reference

Consolidated on 2026-08-18 from the top-level documents in `docs/` (the
original plan, the shared phase contract, GGUF guide, quantization-preflight
status and strategy specs, benchmark instrumentation spec, profiler spec, and
llama.cpp gap analysis). Executed plans and design documents were deleted in
favor of this file; their full records remain in git history.

Per-phase status is tracked in
[atlas-metal-phases/README.md](atlas-metal-phases/README.md) (status source of
truth) and open work in
[atlas-metal-phases/next-improvements.md](atlas-metal-phases/next-improvements.md).

## 1. Overview and design principles

Atlas is a Rust-first, Apple-Silicon LLM inference engine. Its primary runtime
contract is correct, measurable Metal inference — a CPU fallback that only
appears to work is not acceptable.

- **Rust-first.** Core runtime, model execution, memory management, scheduling,
  tokenizer integration, and API serving in Rust; shaders in Metal Shading
  Language.
- **Metal-native first.** Use Metal directly rather than a cross-platform GPU
  abstraction, for control over buffer allocation, unified memory, command
  scheduling, pipelines, threadgroups, kernel specialization, synchronization,
  residency, profiling, and resource reuse.
- **Correctness before optimization.** Every Metal operation is first compared
  against a trusted CPU implementation.
- **Modular operators.** Models depend on tensor operations, not on Metal
  directly. Later backends (CPU, CUDA, Vulkan, WebGPU) must be possible.
- **Bounded active memory.** Design from the start for sliding local attention,
  paged/segmented KV storage, memory eviction, and graph-memory retrieval.

Technical choices: Rust, Metal + MSL, GGUF as the model format, native
tokenizer integration, serde-based serialization.

## 2. Architecture and correctness contract

Current workspace crates:

- `atlas-metal` — Metal device, buffers, command encoding, kernels, GPU
  telemetry.
- `atlas-ops` — neural-network operators and their numerical behavior.
- `atlas-model` — model loading, prefill/decode execution, generation, executor
  modes, residency/parity diagnostics.
- `atlas-cli` — user-facing CLI commands and runtime reporting.
- `atlas-profiler` — performance profiler (disabled by default).

### Executor modes

- **Resident** — the production default. Weights and KV stay GPU-resident and
  run through Metal. All production flows, benchmarks, and GPU validation use
  it.
- **Reference** — an oracle for parity, diagnostics, and deliberately named
  comparison tests. It is never a silent production fallback; when resident
  execution fails, surface and investigate the failure.

GPU-residency claims need evidence: resident bytes, upload/readback,
allocation, command-buffer, or timing metrics. Kernel dispatch alone is not
enough, and the selected executor mode must be observable in code,
diagnostics, or test assertions.

### Correctness

Every operator requires a CPU reference, a Metal implementation, and shape,
dtype, edge-case, and numerical-tolerance tests. Model validation levels:

1. Individual operator outputs
2. Single Transformer block
3. Complete prefill logits
4. Complete decode logits
5. Generated sequence
6. Long-running session

Tolerance contract (Path B decision, 2026-08-13): the production `Flash16`
decode attention is the merged-slice v4 kernel — kernel-level parity is
max-abs < 1e-3 rather than bitwise. Correctness therefore lives in the
kernel-level tolerance tests; the matched-workload greedy-stream hash
(`f23c2962…`) is a drift diagnostic. The exact `Flash16Exact` and
`legacy_fused` paths remain byte-identical.

## 3. Shared phase contract

A phase completes only when its acceptance test passes on Apple Silicon and
its evidence is recorded. A CPU, Candle, or Python implementation may be an
oracle; it is never the production path being accepted.

### Model fixtures

Use one Llama-compatible family to keep format and tokenizer changes from
hiding runtime regressions.

| Tier | Repository | Purpose |
| --- | --- | --- |
| Small | [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Fast correctness fixture. |
| Larger | [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | Memory, throughput, and sustained-generation gate. |

Pin the resolved model revision in `models/manifest.toml`; model files are test
fixtures and must not be committed. Dry-run before every new download:

```zsh
python3 -m pip install --upgrade huggingface_hub
mkdir -p models/hf
hf download HuggingFaceTB/SmolLM2-135M-Instruct --dry-run
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  --local-dir models/hf/SmolLM2-135M-Instruct
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct --dry-run
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --local-dir models/hf/SmolLM2-1.7B-Instruct
```

### Required evidence

Store ignored run artifacts under `artifacts/phase-XX/`: model revision and
SHA256 manifest, macOS/device information, command line, result, numerical
tolerances, seed/token IDs, and requested metrics. Commit only a compact
checksum/summary fixture.

The small-model gate is mandatory for every phase. The larger model is also
mandatory from Phase 3 onward as a memory/performance gate.

### Promotion rules

1. Never silently overwrite golden outputs or tolerances.
2. The accepted path invokes Atlas Rust and Metal components wherever the phase
   claims coverage.
3. Atlas memory phases first validate bounded memory, data flow, isolation, and
   observability. Do not claim semantic quality until separately versioned
   trained fusion/router weights and evaluation exist.

## 4. Model and GGUF workflow

Convert a manifest-backed Llama SafeTensors model into a verified GGUF artifact
for resident Metal inference. The converter writes Q4_0 or Q8_0 block-32
matrices and retains only required F32 tensors such as normalization weights.
The source must be a verified `safetensors-fp32` model in `models/manifest.toml`
with one `model.safetensors` file.

```zsh
cargo run -p atlas-cli -- model quantize --model small --id small-q4 --format q4_0 --quantizer auto --progress human
```

The artifact is written to `models/gguf/small-q4/`: source `config.json` and
`tokenizer.json` are copied, `model.gguf` is written, every file is hashed, and
the manifest record is appended atomically only after all checks pass.

- `--quantizer auto` uses Metal when a device is available and otherwise
  reports CPU conversion; `--quantizer gpu` requires Metal; `--quantizer cpu`
  disables Metal deliberately.
- Use `--progress json` (JSON Lines `conversion_progress` /
  `conversion_completed` events) when another tool consumes progress; human
  progress is written to stderr during `scan`, `quantize`, `write`, and
  `manifest`.

Import and verify a GGUF that was not produced by the converter:

```zsh
cargo run -p atlas-cli -- model import-gguf --path /path/to/model.gguf --id imported-q4 --config /path/to/config.json --tokenizer /path/to/tokenizer.json --source example/model --revision immutable-revision
cargo run -p atlas-cli -- model verify --model imported-q4
```

Import validates the Atlas GGUF header and supported tensor encodings, then
copies files into Atlas-managed storage without changing the source artifact.
GGUF loading is a Resident-only path: packed matrices are uploaded directly and
never silently dequantized or rerun through the reference executor.

## 5. Quantization preflight

When a Gemma model has no valid cached plan, Atlas performs a hardware-specific
Resident Metal format-selection preflight:

1. Loads and hashes the GGUF model.
2. Profiles the existing mixed Q4/Q6 Resident path.
3. Profiles the implemented all-Q4 candidate.
4. Compares throughput, Resident memory, KV-cache bytes, upload/readback
   accounting, generated-token hashes, and EOS positions.
5. Promotes a candidate only when it satisfies exact parity and performance
   gates.
6. Writes a validated JSON sidecar beside the GGUF model.
7. Uses the sidecar on later loads without repeating profiling.

The downloaded GGUF is already quantized. Its safe baseline is `mixed_q4_q6`:
eligible projection weights use Q4, Gemma vocabulary tensors use Q6, and
control/normalization tensors that require F16/F32 retain those formats. The
`all_q4` candidate converts the eligible Q6 vocabulary tensors to Q4 while
leaving unsupported F16/F32 tensors unchanged. The implementation is a Resident
quantization-format autotuner and cache, not a general-purpose dynamic
quantizer.

Recorded Apple-Silicon evidence (release preflight, two runs per workload):

| Workload | Mixed Q4/Q6 | All-Q4 | Candidate result |
| --- | ---: | ---: | --- |
| Long context | 44.25 tok/s | 50.03 tok/s | 13.06% faster |
| Short context | 49.41 tok/s | 56.63 tok/s | 14.60% faster |

All-Q4 is rejected because its generated-token and EOS results do not exactly
match the mixed Q4/Q6 oracle (different EOS position on the long workload;
short-window parity check fails). Atlas therefore writes and selects a ready
mixed Q4/Q6 baseline plan. State after a cached load:

```text
decision=use_cached_plan
executor=resident
weight_format=mixed_q4_q6
quantization_preflight_state=ready
quantization_rejections=[]
```

The validated sidecar is
`models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.quantization-plan.json`
(243 tensor entries): model SHA, hardware identity, tensor/group source and
selected formats, selected kernels, profiling configuration, timing,
residency/upload evidence, and parity digests.

Phase 12.2 remains open for broader per-group quantization selection: whether
preflight can safely choose the best format per executable tensor group
(group boundaries, guided versus exhaustive search, calibration windows, exact
parity against the mixed-format and later an F16 oracle, cache representation,
promotion rules). The research prompt is stored at
[PROMPT-quantization-preflight-selection.md](../PROMPT-quantization-preflight-selection.md).

### Reproduction commands

Profile and write the plan artifact:

```zsh
cargo run --release -p atlas-cli -- model quantization-plan \
  --profile --model gemma4-e2b-q4_0 \
  --output artifacts/quantization-plans/gemma4-e2b-q4_0.json
```

Verify cached-plan inference and validate the sidecar:

```zsh
ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=auto \
cargo run --release -p atlas-cli -- chat \
  --model gemma4-e2b-q4_0 \
  --prompt "Reply exactly: cached plan works" \
  --max-tokens 8
cargo run --release -p atlas-cli -- model quantization-plan \
  --model gemma4-e2b-q4_0
```

Run the Resident acceptance suite:

```zsh
cargo test -p atlas-model --test phase_12a_gemma4_resident -- --ignored
```

## 6. Performance instrumentation

> Note: the user-facing `atlas-cli` binary is intentionally minimal (chat,
> model search, model download). All benchmark/profile/instrumentation commands
> below live in the `atlas-dev` binary (a separate workspace member) so the
> main CLI stays simple for users. The commands below use `atlas-dev`.

Two complementary tools answer "where is Atlas losing time compared with
llama.cpp, and what should be optimized first?":

- **Matched benchmark** produces trustworthy comparable numbers.
- **Profiler (diagnostic)** explains the difference.
- `scripts/compare-atlas-llama.py` combines both into a prioritized
  optimization report.

### Matched-workload contract

Comparison against llama.cpp is only meaningful with the same: GGUF model
bytes, prompt token IDs, prompt length, generated-token count, context length,
KV-cache format, prefill and decode configuration, warmup policy, repetition
count, Apple Silicon hardware, and power/thermal conditions.

Report the llama.cpp categories separately:

```text
pp = prompt processing / prefill
 tg = token generation / decode
 pg = prompt processing followed by token generation
```

Do not combine prefill and decode into one tok/s number without reporting them
separately.

### Warmup and repetition policy

```text
warmup repetitions: 1
measured repetitions: 5
```

For very short workloads use more repetitions or aggregate iterations inside
one measured run. Report every sample, not only the final mean; compute
minimum, maximum, mean, median, standard deviation, and coefficient of
variation. Prefer median paired results for promotion. When running both
engines on the same machine, alternate the order (Atlas, llama.cpp, llama.cpp,
Atlas, Atlas, llama.cpp) or use a deterministic randomized order with a stored
seed.

### Required metrics

High level: `prompt_tokens`, `generated_tokens`, `prefill_time_ms`,
`prefill_tokens_per_second`, `decode_time_ms`, `decode_tokens_per_second`,
`milliseconds_per_generated_token`, `time_to_first_token_ms`,
`end_to_end_time_ms`, `host_wall_time_ms`.

Atlas execution evidence: command-buffer and encoder counts (per phase and per
decode token), dispatch counts (`dispatches_per_decode_token`,
`dispatches_per_layer_per_token`), synchronization events and waits
(`CPU_wait_time_ms`, `GPU_idle_gap_estimate_ms`, `readback_wait_time_ms`),
upload/readback bytes, resident and peak memory, KV-cache bytes, and
allocations during measured execution.

Comparison ratios: `prefill_gap_ratio`, `decode_gap_ratio`,
`prefill_speed_ratio`, `decode_speed_ratio`
(Atlas time / llama.cpp time, or Atlas tok/s / llama.cpp tok/s).

### Commands

Profile bottlenecks (writes `atlas-profile.json` and `atlas-profile.md`; schema
version 4; default recommendations scoped to `decode_measured`):

```zsh
cargo run --release -p atlas-cli -- profile bottlenecks \
  --model gemma4-e2b-q4_0 \
  --prompt 'Explain Resident inference.' \
  --warmup-decode-tokens 32 \
  --decode-tokens 128 \
  --mode diagnostic \
  --output artifacts/profiles/atlas-profile.json
```

Matched benchmark:

```zsh
cargo run --release -p atlas-cli -- benchmark matched \
  --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --warmup-runs 1 \
  --runs 5 \
  --greedy \
  --kv-cache-type q4_0 \
  --output artifacts/benchmarks/atlas-pp512-tg128.json
```

A pre-tokenized prompt can be supplied with `--prompt-token-file` so both
engines consume the identical token stream; `--diagnostic-kernel-timing` adds
per-dispatch timing (not a throughput oracle).

### Profiling rules

- Benchmark mode is low-overhead; diagnostic mode uses a production-boundary
  Resident pass for wall time plus a separate exact per-dispatch pass for
  attribution.
- Profiling must use the production Resident Metal executor — no Reference or
  CPU fallback — and benchmark versus diagnostic modes must be clearly
  separated.
- Instrumentation overhead must be measured.
- A recommendation in a profile is an investigation priority, not proof that
  an optimization is safe or faster; matched Resident workloads and exact
  token parity remain required for promotion.
- Reports include a reconciliation section: `dispatch_calls` is the number of
  Metal dispatch calls; `threadgroups_dispatched` / `threads_dispatched` are
  launch-size totals and are not interchangeable with dispatch calls.
  Profiles with incomplete coverage are marked incomplete. Kernel labels stay
  conservative (`compute_candidate`, `bandwidth_candidate`,
  `dispatch_overhead_candidate`, `synchronization_candidate`, `mixed`,
  `unknown`).

## 7. llama.cpp gap summary

Measured on the same GGUF, same `q4_0` KV cache, and same Apple M2 Max
(evidence in `artifacts/atlas-vs-llama/` and `artifacts/profiles/`).

Historical baseline (phase-13.1 root cause): Atlas was 12–45× slower at
prefill (gap grew with prompt length) and 2–5× slower at decode. Two structural
causes: prefill re-read weights per token instead of one batched GEMM, and
decode was dispatch/occupancy-bound (~550 dispatches/token, ~8.7× the
weight-read bandwidth floor) rather than bandwidth-bound (llama.cpp ~2.1×).

What landed:

- **R1 (13.1)** — batched prefill kernels; prefill ~190 tok/s flat, byte-identical stream.
- **D1 (13.2)** — q4 attention defaults to the no-value-barrier flash16 kernel; decode GPU −8.6%.
- **D2 (13.3)** — staged, chunked, exact-ordered attention KV scan with wide threadgroups; decode GPU −31.6% at pp512, byte-identical stream.
- **D3 (13.4, reverted)** — dispatch fusion measured as a wash: decode is GPU-latency-bound, not dispatch-bound.
- **Prefill attention staging (13.5, reverted)** — measured regression.
- **Batched prefill GEMM (13.6)** — weight-stationary 4-token tile
  (`matmul_q4_0_batch_32row`); prefill ~190 → ~308–322 tok/s.
- **Path B + Flash16 v4 (13.7)** — merged-slice non-exact decode attention
  (tolerance-level parity, approved tradeoff); decode 30.5 → 64.9 tok/s at
  pp512.

Current state (re-measured 2026-08-13):

| Prompt | Prefill ratio vs llama.cpp | Decode ratio vs llama.cpp |
|---:|---:|---:|
| 100 | 3.5× | 1.1× |
| 512 | 5.4× | 1.2× |
| 1024 | 5.4× | 1.3× |

Prefill is flat (~289–327 tok/s) and the prefill gap closed from 12–45× to
~4–5×; decode is within ~1.1–1.3× (the honest decode gap is smaller, since
llama-bench `tg` decodes from an empty context, and Atlas decode is
~0.8–0.9× of that). The remaining lever is a **llama-grade `mul_mm`-style q4
batched GEMM** to close the prefill gap; prefill-first is justified for
large-prompt workloads, where the prefill absolute gap dominates. Matvec
breadth tuning (D4) should not be pursued further until the GPU-latency levers
are spent.

> Note (2026-08-22): these ratios predate the phase-13.19 prefill default-flip
> (llama mul_mm GEMMs + Flash16-v7, now ~1575–1589 tok/s prefill) and the
> phase-13.21 wide qk_norm_rope (now the default; decode +12–14% to ~77 tok/s,
> prefill +3%, with a deliberate canonical-fixture re-baseline to the wide
> greedy stream — the exact kernels remain via `ATLAS_GEMMA4_EXACT_QKNORM=1`).
> The authoritative current-state index and improved numbers are in
> [atlas-metal-phases/README.md](atlas-metal-phases/README.md); the open work
> list is in
> [atlas-metal-phases/next-improvements.md](atlas-metal-phases/next-improvements.md).

## 8. Historical documents

Deleted from `docs/` on 2026-08-18; full contents remain in git history:

- `Atlas_Metal_Inference_Engine_Plan.md` — the original full-phase plan
  (phases 0–15, Metal kernel roadmap, memory strategy, milestones, MVP
  sequence, risks).
- `Atlas_Metal_Inference_Engine_Phase_Subplans.md` — replaced by section 3.
- `atlas-gguf-conversion.md` — replaced by section 4.
- `current_status.md`, `dynamic_quantization_preflight_strategy.md`,
  `quantization_preflight_artifact.md` — replaced by section 5.
- `llama_cpp_matched_benchmark_instrumentation.md`,
  `atlas_performance_profiler.md` — replaced by section 6.
- `atlas-vs-llama-gap-analysis.md` — replaced by section 7.
- `plan-close-prefill-gap.md`, `plan-close-llama-gap-path-b.md`,
  `plan-prefill-batched-gemm.md` — executed phase plans (13.x); their results
  are recorded above and in the phase records.