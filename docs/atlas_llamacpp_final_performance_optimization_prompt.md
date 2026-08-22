# Atlas Final Performance Optimization — Reach llama.cpp Decode Performance

You are acting as a senior GPU inference performance engineer specializing in Apple Silicon, Metal compute kernels, quantized LLM inference, GGUF, and low-latency autoregressive decoding.

Your task is to perform the **final performance optimization of Atlas**, with the concrete objective of bringing Atlas inference performance as close as possible to — and where specialization permits, potentially beyond — the current `llama.cpp` Metal backend.

## Primary objective

Optimize the current Atlas `main` branch for **real end-to-end inference performance**, with particular emphasis on:

- autoregressive decode at batch size 1;
- Q4_0 GGUF execution;
- Gemma 4 / current Atlas production model;
- Apple Silicon / Metal;
- token latency and tokens/second;
- minimizing GPU dispatch and synchronization overhead;
- maximizing effective memory bandwidth during GEMV;
- maintaining exact model correctness.

The existing Atlas implementation already has reasonably competitive prefill performance.

The remaining major gap is decode performance.

Do **not** redesign Atlas unnecessarily.

Treat the current architecture as the baseline and identify the smallest set of high-impact changes required to close the performance gap with `llama.cpp`.

## Reference implementation

Use the current upstream `llama.cpp` Metal backend as the primary performance reference.

Study in particular:

- `ggml/src/ggml-metal/ggml-metal.metal`
- Metal MUL_MV / GEMV implementations;
- Q4_0 dequantization;
- `mul_vec_q_n_f32_impl`;
- `kernel_mul_mv_q4_0_f32`;
- extended MUL_MV kernels;
- `NR0` multi-row processing;
- SIMD-group organization;
- Metal function constants;
- register reuse;
- `float4` / `float4x4` dequantization;
- threadgroup sizing;
- dispatch configuration;
- quantized dot-product implementations.

Do not blindly port llama.cpp.

Instead determine **why** each optimization exists and implement the equivalent strategy appropriate for Atlas.

Atlas can specialize more aggressively than llama.cpp because it does not need to preserve ggml's extremely generic execution model.

## Current Atlas characteristics

Before modifying anything, inspect the current `main` branch and verify the implementation rather than relying only on this prompt.

Atlas already contains optimizations such as:

- resident Metal execution;
- one command buffer covering the decode token where possible;
- SIMD RMSNorm;
- vectorized RMSNorm;
- fused Gemma-specific epilogues;
- fused decode attention;
- Q4/Q8 KV cache paths;
- online-softmax attention;
- specialized Gemma attention kernels;
- Flash16 experiments;
- Q4_0 blocked matvec;
- reduced dispatch counts compared with earlier implementations.

Preserve successful existing optimizations unless measurements demonstrate that replacing them improves end-to-end performance.

## Key hypothesis to investigate first

The highest-priority hypothesis is that Atlas's remaining decode gap is dominated by the Q4_0 GEMV implementation.

The current Atlas approach is conceptually close to:

```text
one SIMD group -> one output row
```

This gives good intra-row parallelism but potentially causes the same activation vector to be loaded repeatedly across output rows.

By comparison, llama.cpp's optimized Metal MUL_MV implementation processes **multiple output rows per SIMD/threadgroup**, allowing activation data loaded into registers to be reused across several rows.

Conceptually:

```text
load activation chunk once

for row in NR0:
    accumulate Q4 dot product for row

reduce accumulated rows
write NR0 outputs
```

This is especially important during batch-1 autoregressive decode because GEMV is largely memory-bandwidth constrained.

Investigate this hypothesis first.

## Phase 1 — Establish an immutable baseline

Before making performance changes:

1. Build current Atlas `main`.
2. Record:
   - model;
   - quantization;
   - machine / Apple SoC;
   - context size;
   - prompt length;
   - generated token count;
   - prefill tok/s;
   - decode tok/s;
   - time/token;
   - GPU execution time;
   - CPU submission/wait time;
   - command buffers/token;
   - dispatches/token.
3. Run enough iterations to distinguish improvements from normal measurement noise.
4. Record correctness output / greedy token sequence.

Create a reproducible benchmark command.

Every optimization must be compared against this baseline.

Do not report a performance improvement from a single noisy run.

## Phase 2 — Profile before changing architecture

Use Atlas's existing telemetry/profiling facilities and Metal profiling where useful.

Determine where decode time actually goes.

Produce a ranked table similar to:

| Rank | Kernel / operation | Dispatches/token | GPU time/token | % decode | Estimated bytes |
|---|---|---:|---:|---:|---:|

At minimum identify the contribution of:

- Q/K/V projections;
- output projection;
- FFN projections;
- Q4_0 GEMV;
- attention;
- RMSNorm;
- elementwise operations;
- KV cache operations;
- logits projection;
- dispatch/encoder overhead.

Do not optimize kernels merely because they appear theoretically inefficient.

Prioritize measured end-to-end cost.

## Phase 3 — Implement a multi-row Q4_0 GEMV

This is the first major optimization experiment.

Design a new Atlas Metal kernel inspired by llama.cpp's multi-row MUL_MV architecture.

Do not simply modify the existing production kernel initially.

Introduce an experimental kernel such as:

```text
matvec_q4_0_multirow
```

The kernel should investigate processing multiple output rows while sharing activation loads.

Initial configurations to benchmark should include approximately:

```text
NR0 = 2
NR0 = 4
NR0 = 8
```

and appropriate SIMD groups per threadgroup.

A reasonable initial experiment on Apple Silicon is:

```text
32 threads / SIMD group
4 SIMD groups
128 threads / threadgroup
NR0 = 4
```

but this is a hypothesis, not a fixed requirement.

Benchmark alternatives.

The important property is:

```text
activation values should be loaded once and reused across multiple weight rows
```

where practical.

Avoid repeatedly streaming the complete activation vector independently for every output row.

## Phase 4 — Optimize Q4_0 unpack/dequantization

Compare Atlas's Q4_0 decoding directly with llama.cpp.

Investigate:

- packed 16-bit loads;
- nibble extraction;
- vectorized conversion;
- `float4`;
- `float4x4`;
- precomputed masks;
- scale transformations;
- avoiding unnecessary integer-to-float operations;
- avoiding repeated subtraction/multiplication;
- loop unrolling;
- register pressure.

Keep weights compressed.

Do not dequantize the entire model to F16/F32 as an optimization.

The objective is efficient **on-the-fly quantized GEMV**.

## Phase 5 — Activation register reuse

Study llama.cpp's use of local activation arrays/registers during MUL_MV.

Experiment with loading an activation tile into registers and reusing it across multiple output rows.

Measure:

- register pressure;
- occupancy;
- memory transactions;
- kernel duration;
- end-to-end decode speed.

Do not assume higher occupancy is automatically better.

For bandwidth-bound GEMV, greater register reuse may justify reduced occupancy.

Measure the tradeoff.

## Phase 6 — Compile-time specialization

Atlas has an opportunity llama.cpp does not fully have: aggressive model specialization.

Identify important Gemma 4 projection shapes and create specialized fast paths where worthwhile.

Prefer compile-time knowledge for frequently repeated decode shapes.

Investigate specialization of:

- hidden dimension;
- FFN dimensions;
- Q projection;
- K projection;
- V projection;
- attention output projection;
- FFN gate/up/down projections;
- final logits projection.

Where beneficial, replace runtime dimensions with templates, generated specialized Metal entry points, Metal function constants, or compile-time constants.

Maintain a generic fallback path.

## Phase 7 — Dispatch-count reduction

Measure the actual current number of Metal compute dispatches per generated token.

Atlas has already introduced several useful fusion kernels.

Continue this work based on profiling.

Prioritize fusion when:

```text
kernel execution time ~= dispatch overhead
```

Do not create enormous kernels merely to minimize a counter.

Fusion must improve actual token latency.

## Phase 8 — Investigate projection-level fusion

After GEMV itself is optimized, investigate whether Gemma-specific execution allows multiple logically related projections to share activation reads.

Candidates include:

```text
Q + K + V
```

and potentially:

```text
FFN gate + up
```

The goal would be:

```text
load activation tile once
apply multiple quantized matrices
produce multiple projection outputs
```

Implement this only after establishing the performance of the standalone multi-row GEMV.

## Phase 9 — Re-evaluate attention

Atlas already contains sophisticated fused attention implementations.

Do not rewrite attention without evidence.

Profile global attention, sliding-window attention, Q4/Q8 KV cache and Flash16 variants.

As GEMV becomes faster, attention may become the next bottleneck. Only then optimize it further.

## Phase 10 — Command encoding and runtime overhead

Verify that the normal fast path:

- does not submit one command buffer per kernel;
- does not wait for completion between dependent kernels;
- does not perform unnecessary CPU/GPU synchronization;
- does not allocate temporary Metal buffers during each token;
- does not rebuild pipeline state;
- does not perform unnecessary readbacks.

One token should remain GPU-resident as much as possible.

Profile CPU encoding time separately from GPU execution time.

## Phase 11 — Memory layout

Analyze the memory access pattern for quantized projection weights.

First attempt to operate efficiently directly on standard GGUF storage.

If profiling demonstrates that weight layout itself limits coalescing, investigate an optional one-time load-time transformation into an Atlas-native GPU layout.

Such transformation is acceptable only if startup cost and memory overhead are measured and decode improvement is substantial.

## Phase 12 — Performance target

The primary target is:

```text
Atlas decode >= 90–95% of llama.cpp Metal decode throughput
```

for the same machine, model, GGUF, quantization, prompt, context, and generation parameters.

The stretch target is:

```text
Atlas >= llama.cpp
```

using Atlas-specific Gemma specialization.

Do not sacrifice prefill performance unless the end-to-end tradeoff is clearly favorable.

## Correctness requirements

Performance changes are invalid if model behavior is corrupted.

After every significant kernel change verify:

- numerical error against the reference implementation;
- logits where practical;
- greedy token sequence;
- generated output;
- long-generation stability;
- multiple prompts;
- different context positions.

Pay particular attention to floating-point reassociation.

Mathematically equivalent operations can alter greedy generation.

Do not use `fast-math` or reorder reductions blindly.

## Benchmark discipline

For every optimization produce results in this form:

```text
Optimization:
Baseline:
Modified:

Prefill tok/s:
Decode tok/s:
ms/token:
GPU ms/token:
CPU encode ms/token:
dispatches/token:
command buffers/token:

Speedup:
Correctness:
Decision: KEEP / REJECT
```

A rejected experiment is useful information.

Do not leave slower experimental paths enabled by default.

## Optimization priority

Unless profiling disproves it, follow this order:

1. Q4_0 multi-row GEMV
2. Q4 unpack/dequantization
3. activation register reuse
4. Gemma shape specialization
5. dispatch reduction
6. QKV / FFN projection fusion
7. attention optimization
8. runtime/encoding optimization
9. optional GPU-native weight layout

Do not begin by rewriting the execution engine.

## Important design principle

Do not attempt to turn Atlas into llama.cpp.

The desired architecture is:

```text
generic correct path
        +
optimized Metal path
        +
model-specific fast paths where justified
```

Use llama.cpp to understand the techniques required to efficiently exploit Apple Silicon, then implement those techniques in the cleanest form for Atlas.

## Final deliverable

Continue iterating until profiling shows either:

1. Atlas reaches the target performance relative to llama.cpp, or
2. the remaining gap has been quantitatively explained by measured bottlenecks.

At completion provide:

### Final benchmark

```text
                     Atlas before    Atlas optimized    llama.cpp
Prefill tok/s
Decode tok/s
ms/token
```

### Optimization contribution

Estimate the contribution of each retained optimization.

### Remaining bottlenecks

List only measured remaining bottlenecks.

### Architecture changes

Document all new specialized kernels and runtime changes.

### Rejected experiments

Document optimizations that were tested but rejected and why.

### Next theoretical limit

Estimate whether the remaining decode path is:

```text
memory-bandwidth bound
compute bound
dispatch/latency bound
attention bound
mixed
```

and identify what would be required to improve beyond llama.cpp.

## Working rule

Do not stop after implementing the first successful optimization.

This is a performance-convergence task.

Use the loop:

```text
PROFILE
   ↓
IDENTIFY BOTTLENECK
   ↓
IMPLEMENT ONE CONTROLLED CHANGE
   ↓
VERIFY CORRECTNESS
   ↓
BENCHMARK
   ↓
KEEP OR REVERT
   ↓
PROFILE AGAIN
```

Continue until Atlas's end-to-end decode performance converges toward llama.cpp.

The metric that ultimately matters is not microbenchmark GFLOPS or isolated kernel speed.

It is:

```text
correct generated tokens per second
```

on the same model, hardware, context, and generation workload.
