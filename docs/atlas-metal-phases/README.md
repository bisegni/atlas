# Atlas Metal phase index

Status source of truth for the Atlas Metal inference phases. A phase is
complete only when its declared runnable acceptance gate passed on Apple
Silicon with the required numerical or performance evidence recorded.

## Current phase

- [phase-13.7-flash16-v4-decode-attention.md](phase-13.7-flash16-v4-decode-attention.md) —
  Path B merged-slice decode flash attention: `Flash16` now binds the
  non-bitwise v4 kernels (no per-key barriers; slice-merge in threadgroup
  memory), covered by the max-abs < 1e-3 tolerance contract. Decode more than
  doubled (30.5 → 64.9 tok/s at matched pp512/tg128; ~0.8× llama.cpp's
  empty-context decode), attention ~19 → ~2.5 ms/token, stream hash still
  `f23c2962…` on the matched workloads (drifts at token 50 on the chat prompt —
  the approved Path B tradeoff).
- [phase-13.6-prefill-batched-gemm.md](phase-13.6-prefill-batched-gemm.md) —
  Weight-stationary q4 batched GEMM for prefill: `matmul_q4_0_batch_32row`
  tiles 4 tokens × 32 rows per threadgroup so each weight block is read once
  and reused in registers, cutting prefill weight traffic ~4×. Prefill +60–70%
  (~190 → ~308–322 tok/s flat at pp100/512/1024), closing the llama.cpp prefill
  gap from ~8× to ~4–5× with a byte-identical greedy stream hash. Parity is
  tolerance-level (max-abs ~4.7e-10, user-approved; the bitwise token-major
  variant measured only +5%).
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
