# Atlas Metal phase index

Status source of truth for the Atlas Metal inference phases. A phase is
complete only when its declared runnable acceptance gate passed on Apple
Silicon with the required numerical or performance evidence recorded.

## Current phase

- [phase-13.1-batched-prefill-kernels.md](phase-13.1-batched-prefill-kernels.md) —
  Token-batched prefill kernels (gap-analysis R1). Acceptance gate met: per-token
  qk/rope, V-RMS, KV-append, attention, and PLE loops collapsed into single
  dispatches; batch cap raised to 512; prefill pp512 49.6 → 199.8 tok/s,
  greedy stream hash byte-identical (artifact under `artifacts/phase-13.1/`).
- [phase-13.0-resident-decode-100-toks.md](phase-13.0-resident-decode-100-toks.md) —
  Resident decode to 100 tok/s. Active: resume, hotspot baseline, ordered
  improvement plan (RMS fusion, dispatch fusion, flash16 v3 tiling, matvec
  ILP), acceptance gates.

## Supporting analysis

- [atlas-vs-llama-gap-analysis.md](../atlas-vs-llama-gap-analysis.md) — why
  llama.cpp is 12–45× faster in prefill and 2–5× in decode on the same GGUF:
  batched GEMM vs per-token prefill loops, and dispatch/occupancy-bound decode
  vs llama.cpp's near-bandwidth-bound fused kernels. Read-only reference for
  prioritizing phase 13+ work.

## Retired phases

Phase 12.3 (resident decode optimization) reached its promotion gate with
the composed flash16_uw + mv_ext + rms-vec4 stack at 56.6/68.9 tok/s
long/short and was retired when phase 13.0 started. Earlier phase documents
were removed as part of that cleanup; their evidence artifacts remain under
`artifacts/`.
