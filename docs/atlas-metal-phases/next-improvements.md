# Next improvements

Prefill is closed as a lever (fast-path default, ~1505–1510 tok/s; the
remaining ~1.25–1.4× gap to llama.cpp is q4 dequant/bandwidth-bound, see
below).  **Decode at ~68.8 tok/s (~14.5 ms/tok) is the bottleneck**, and it
is GPU-latency-bound: dispatch fusion has been applied twice (PLE gate+gelu,
PLE epilogue) without moving throughput.  The open levers, in order:

## 1. Attention split-KV (decode flash16 scan) — implemented, measured negative (phase 13.20)

- **Why:** attention is the largest decode family (~3.5–4.4 ms/tok of ~18.8
  GPU: 28 swa layers ~3.0 ms + 7 full layers ~1.35 ms at pp512) and grows
  linearly with context (kv_append + attention cost).  The flash16 scan's
  reduction order is exact-locked today (byte-identical sentinel), which
  blocks the reordering the split needs.
- **Approach:** split the KV scan into chunks across parallel threadgroups
   (per-chunk partial softmax state), merge in a second pass — or relax the
  decode-attention reduction to the established tolerance contract
   (max-abs 1e-3, re-baseline the greedy hash) like the prefill attention did.
- **What landed (phase 13.20):** a two-pass `Flash16SplitKv` decode mode
   (`scan` writes per-chunk partial softmax state to resident scratch, a
   `combine` merges them into the head output), selectable via
   `--q4-attention-mode flash16_split_kv` with `ATLAS_GEMMA4_SPLIT_KV` (1–32,
   default 8).  Kernels `attention_decode_gemma4_simd_q4_0_flash16_split_[swa_]{scan,combine}`
   in `kernels.metal`; parity test `flash16_split_kv_parity.rs`
   (max-abs < 1e-3 vs the CPU Q4_0 oracle, full-512 and swa-256).  Left an
   opt-in diagnostic — the production default stays `Flash16` (v4).
- **Result (M2 Max, q4_0, pp~32/tg128):** regression at every split.  Baseline
   `flash16` 77.4 tok/s; candidate 41.6 (split 1), 62.1 (split 2), 60.4
   (split 4), 60.9 (split 8), 61.1 (split 12).  Best case −19.8%.  Residency
   identical across modes (scratch is unconditional); the greedy hash differs
   from baseline as expected under the tolerance contract.
- **Long-context result (warmup 1024 / decode 512 / ctx 2048):** also a
  regression — and *worse* as split grows.  Baseline `flash16` 62.9 tok/s;
  candidate 12.7 (split 1), 28.6 (split 2), 29.2 (split 4), 34.3 (split 8),
  35.6 (split 12).  Best case −43.4%.  The candidate token stream also diverges
  across splits (and split 4 hit a different EOS), as expected under the
  tolerance contract.
- **Why it lost:** the two-pass overhead (extra dispatch + scratch round-trip)
  outweighs the per-threadgroup parallelism gain at both short and long
  context.  Confirms decode is GPU-latency-bound, not
  attention-parallelism-bound.  The lever is closed (tested at both contexts);
  it is not a viable optimization on this engine and geometry.
- **Evidence:** `artifacts/phase-13.20-split-kv-decode/summary.json`; harness
   `scripts/run-gemma4-split-kv-ab.sh`.
- **Status:** closed (negative).  Not promoted; kept as an opt-in diagnostic so
  `flash16_split_kv` remains selectable for comparison.

## 1.1 Phase 13.21 — wide qk_norm_rope (single-stream fill+decode) — landed, positive

- **Bug found:** `gemma4_qk_norm_rope_fused_f32` / `_batch_f32` distributed the
  head-width work with `threads_per_threadgroup: 1` and `if (tid != 0) return;`,
  so one scalar thread serially reduced + rotated each entire Q/K head.  In
  decode the grid is only `q_heads + has_key` (~9), leaving the GPU almost idle
  during that dispatch (~10.8% of the per-token GPU budget, 2.31 ms/tok);
  ~3% of prefill.
- **Fix:** `gemma4_qk_norm_rope_fused_f32_wide` (decode) and
  `..._batch_f32_wide` (prefill) spread the per-head RMS sum-of-squares across
  one SIMD group (`simd_sum`) and the RoPE rotation one pair per lane, keeping
  the half-split layout and `weight[pair]` scaling.  The exact `_f32` kernels
  remain as the `ATLAS_GEMMA4_EXACT_QKNORM=1` byte-identical fallback.
- **Result (M2 Max, pp512/tg128):** decode **68.8 → ~77 tok/s (+12–14%)**,
  prefill **1537 → ~1577 tok/s (+3%)**, and the matched-workload greedy hash
  stayed **`f23c2962…`**.  The reordered reduction IS prompt-dependent, so the
  canonical greedy fixture was deliberately re-baselined to the wide-kernel
  stream (a documented correctness-contract change).
- **Parity:** `qk_norm_rope_wide_parity.rs` (max-abs < 1e-3 vs the exact
  kernels, full-512 / swa-256 / provider-K / batch-8).
- **Status:** positive and **promoted as default**.  The exact single-thread
  `_f32`/`_batch_f32` kernels remain reachable via
  `ATLAS_GEMMA4_EXACT_QKNORM=1`.
- **Next:** generalizing the same SIMD-group pattern to the other
   single-threadgroup decode norm/residual kernels (`gemma4_rms_residual_f32`,
   `gemma4_ple_rms_add_scale_f32`, ~8% combined) is the obvious follow-up.

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