# Next improvements

Prefill is closed as a lever (fast-path default, ~1505–1510 tok/s; the
remaining ~1.25–1.4× gap to llama.cpp is q4 dequant/bandwidth-bound, see
below).  **Decode at ~68.8 tok/s (~14.5 ms/tok) is the bottleneck**, and it
is GPU-latency-bound: dispatch fusion has been applied twice (PLE gate+gelu,
PLE epilogue) without moving throughput.  The open levers, in order:

## 1. Attention split-KV (decode flash16 scan) — High value, in scope

- **Why:** attention is the largest decode family (~3.5–4.4 ms/tok of ~18.8
  GPU: 28 swa layers ~3.0 ms + 7 full layers ~1.35 ms at pp512) and grows
  linearly with context (kv_append + attention cost).  The flash16 scan's
  reduction order is exact-locked today (byte-identical sentinel), which
  blocks the reordering the split needs.
- **Approach:** split the KV scan into chunks across parallel threadgroups
  (per-chunk partial softmax state), merge in a second pass — or relax the
  decode-attention reduction to the established tolerance contract
  (max-abs 1e-3, re-baseline the greedy hash) like the prefill attention did.
- **Expected:** attention 3.5–4.4 → ~1.5–2 ms/tok at pp512; larger wins at
  longer contexts.  Decode e2e maybe +10–15%.
- **Evidence needed:** kernel parity + hash re-baseline record.

## 2. Batch decode (N > 1) — Highest structural upside, largest effort

- **Why:** every decode matvec/attention is N=1; the vendored llama
  `mul_mm` matrix-unit machinery (phase-13.15 negative at N=1, ~2.8× slower)
  pays off only at N ≥ 16–32.  Server-style load (parallel sequences) is
  where Atlas can gain 2–5×, not single-stream.
- **Approach:** batched decode matvecs (`mul_mm` / fp16 pre-dequant weights),
  batched KV append + batched flash attention (prefill kernels already
  batch), inter-process isolation for N sequences.
- **Evidence needed:** batch-geometry kernels already exist
  (`matmul_q4_0_batch_*`); needs decoder-side commit.

## 3. Minor / recorded limits (not worth pursuing)

- **lm_head ~1.25 ms/tok** (6% of decode): 16-row band falsified on the
  huge-M q6 kernel (throughput-bound); only a structural pre-dequantized
  fp16 path could win, at tolerance cost.
- **Prefill dequant bandwidth:** flash16-v7 already halves K/V dequant
  traffic; the only further step is pre-dequantized fp16 weights
  (tolerance contract, large memory cost) — measured-wash risk on current
  tile sizes.

## Known baseline for comparisons

Recent A/B references (moved into git history with the phase-doc cleanup):

- Prefill default-flip A/B: fast 339–341 ms / ~1505–1510 tok/s vs legacy
  ~1616 ms / ~317 tok/s, hash `f23c2962…` both.
- PLE fusion A/B: PLE family 1.98 → 1.73 ms/tok, −35 dispatches/token,
  e2e flat.
- 16-row default A/B: decode −7.8% e2e (1850 vs 2007 ms, 69.2 vs 63.8 tok/s),
  hash unchanged.