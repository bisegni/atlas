# Plan — Prefill q4 batched GEMM v2 (close the 8× prefill gap)

> **Status: EXECUTED (phase-13.6).** Landed as `matmul_q4_0_batch_32row`, a
> weight-stationary tile (4 tokens × 32 rows per threadgroup) that reads each
> weight block once and reuses it in registers across four tokens. Prefill
> ~190 → ~308–322 tok/s flat at pp100/512/1024 (gap 8× → ~4–5×) with the
> greedy stream hash `f23c2962…` byte-identical. Parity is **tolerance-level**
> (max-abs ~4.7e-10, user-approved) because the interleaved accumulator chains
> shift Metal's instruction selection ~1 ulp; the bitwise-exact token-major
> variant measured only +5%. The software-prefetch idea from the original plan
> was ruled out — it also drifts ~1 ulp and is incompatible with the stream
> sentinel. See `docs/atlas-metal-phases/phase-13.6-prefill-batched-gemm.md`.

## Objective

Close the largest remaining Atlas-vs-llama.cpp gap: prefill at ~190 tok/s vs
llama.cpp's ~1150–1670 tok/s (8×). The prefill is GEMM-bound (≈570× the pure
read floor at pp512), so the fix is the q4 batched matmul's compute ILP, not
bandwidth or dispatch count.

## Context (measured)

- Prefill gap: **2.4s at pp512 / 5.0s at pp1024** — the biggest absolute gap.
- Prefill is ~200 tok/s flat (R1 batched the per-token kernels); the matmuls
  now dominate the ~2.7s pp512 prefill.
- `matmul_q4_0_batch_16row` (kernels.metal:286): 16-row tile, one `sum`
  accumulator per lane, serial block loop — weight/input load latency is never
  hidden, so per-block loads stall the FMA chain.
- Exhausted levers (measured washes/regressions, do not re-explore): D3 decode
  dispatch fusion (phase-13.4, reverted), prefill attention staging
  (phase-13.5, reverted). Decode matvecs are already near their bandwidth share.

## Hard constraint

The greedy stream hash `f23c2962…` is the drift sentinel, and the prefill is
not covered by the decode-only logit-digest gate. The new GEMM must reproduce
`matmul_q4_0_batch_16row`'s exact per-element FP32 accumulation order —
block-sequential dims, per-lane column pattern (4 dims per 32-block),
`shuffle_xor(4,2,1)` butterfly — i.e. **bitwise-identical output**.

## Design (bitwise-preserving)

1. **Software prefetch (primary win).** Prologue-load block 0's packed weight
   bytes + input values into registers; in the loop, issue block `b+1`'s loads
   while accumulating block `b`; rotate registers. Hides load latency without
   touching the accumulation order (loads are side-effect-free).
2. **Larger tile (measured knob).** 32 rows/threadgroup (8 SIMD groups × 4
   rows, 256 threads) with the identical per-lane pattern, for more in-flight
   work per threadgroup. Keep the 16-row prefetch variant as fallback if the
   wider tile regresses occupancy.

## Files

- `crates/atlas-metal/src/kernels.metal`: new `matmul_q4_0_batch_32row`
  (prefetch + tile), keep `matmul_q4_0_batch_16row` as the parity oracle.
- `crates/atlas-metal/src/lib.rs`: register the kernel (pipeline 80 → 81).
- `crates/atlas-metal/tests/batch_matmul_parity.rs`: **bitwise** parity test
  new-vs-current across geometries (incl. 2048/2304 widths) and batches.
- `crates/atlas-model/src/gemma4_executor.rs`: point
  `gemma4_q4_batch_projection_kernel()` at the new kernel (covers all prefill
  Q4 matmuls: qkv, gate-up, ffn-down).
- `crates/atlas-ops/tests/phase_02_operators.rs`: pipeline count 80 → 81.

## Validation (keep/revert gated)

1. Bitwise kernel-parity test (the stream-critical proof).
2. `cargo test --workspace`; `cargo fmt --check`.
3. Matched benchmark pp512/tg128: stream hash must stay `f23c2962…`; prefill
   tok/s target **≥250–300** (was ~190); decode unchanged.
4. Context sweep pp100/512/1024 to confirm prefill scaling.
5. If prefill regresses or the stream drifts, revert to
   `matmul_q4_0_batch_16row`.

Artifacts under `artifacts/phase-13.6/`.

## Docs

- `docs/atlas-metal-phases/phase-13.6-prefill-batched-gemm.md` + README index
  link.
- Update the prefill status in `docs/atlas-vs-llama-gap-analysis.md`.

## Expected outcome

~2–3× prefill matmul → prefill **190 → ~350–500 tok/s** (gap 8× → ~3–4×),
saving ~1.5–2s at pp512. Llama-grade parity requires their full `mul_mm`
tuning; this is the tractable first step and attacks the actual compute-ILP
bottleneck.
