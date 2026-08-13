# Atlas Metal phase index

Status source of truth for the Atlas Metal inference phases. A phase is
complete only when its declared runnable acceptance gate passed on Apple
Silicon with the required numerical or performance evidence recorded.

## Current phase

- [phase-13.3-flash16-staged-kv-scan.md](phase-13.3-flash16-staged-kv-scan.md) —
  Decode improvement D2 (gap analysis): staged, chunked, exact-ordered decode
  attention KV scan with wide (512 full / 256 swa) threadgroups. Acceptance
  gate met: v3 bitwise-identical to LegacyFused/`_nb` (per-token fp32 logit
  digests + exact-token stream parity), decode GPU −31.6% at matched pp512/tg128
  with a byte-identical greedy stream hash, decode 53.6→31.0→27.4 tok/s at
  pp100→512→1024 (artifact under `artifacts/phase-13.3/`).
- [phase-13.2-flash16-default-attention.md](phase-13.2-flash16-default-attention.md) —
  Decode improvement D1 (gap analysis): q4 attention defaults to the
  no-value-barrier flash16 variant. Acceptance gate met: per-token fp32
  logit-digest and exact-token parity with LegacyFused preserved, decode GPU
  −8.6% at matched pp512/tg128, greedy stream hash byte-identical (artifact
  under `artifacts/phase-13.2/`).
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
