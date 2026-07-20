# Phase 12.1: Resident decode performance remediation

## Outcome

Atlas materially improves GPU-resident decode throughput for the larger
SmolLM2 fixture on Apple Silicon without weakening quantized quality,
residency, or normal generation semantics.

## Baseline

Phase 12 measured the 1.7B fixture on the M2 Max with the fixed 128-token
benchmark workload (`benchmark --ignore-eos`, one warm-up): Q4_0 reached 15.85
tok/s, Q8_0 reached 13.42 tok/s, and FP32 reached 3.77 tok/s. The packed paths
are resident and memory-efficient, but their throughput is not yet acceptable.

## Work

- Add per-stage resident decode profiling that separates CPU command encoding,
  Metal command-buffer scheduling, and GPU time for embedding, attention,
  packed projections, MLP, LM head, and token readback. Keep its output
  opt-in and machine-readable so the slowest stage is demonstrable.
- Remove the measured dominant host/command-buffer costs without adding a CPU
  fallback: retain resident weights, KV, activations, and token selection;
  preserve the single decode command-buffer boundary and token-only default
  readback.
- Improve the measured dominant packed-kernel path (Q4_0/Q8_0 matvec or its
  surrounding layout/dispatch) using Apple-GPU occupancy and memory-access
  evidence. Do not dequantize a full weight matrix or trade resident bytes for
  an undisclosed FP32 cache.
- Keep `benchmark --ignore-eos` benchmark-only. Normal `chat`, `generate`,
  and `diagnose` must still stop at EOS and remain on `ExecutorMode::Resident`.
- Preserve the Phase-12 manifest quality gates and add a regression assertion
  that the profiling path is observational only: it must not change generated
  token IDs, finish reason, resident bytes, or default readback behavior.

## Exit gate

On the same M2 Max, fixture revision, prompt, warm-up policy, and 128-token
workload as Phase 12, Q4_0 reaches at least 30 tok/s with
`ExecutorMode::Resident`, 128 generated tokens, and resident bytes no greater
than the Phase-12 1,366,335,800-byte baseline. Q8_0 must not regress below its
Phase-12 13.42 tok/s baseline or exceed 2,221,973,816 resident bytes.

Record the command, hardware/OS, profiler breakdown, model and resident bytes,
generated-token count, readback/command-buffer metrics, and Q4_0/Q8_0 tok/s
under `artifacts/phase-12a/`. Run the small-fixture Q4_0/Q8_0 `diagnose` gates
after the performance changes; any new drift requires an explicit, recorded
policy decision rather than a silent tolerance update.
