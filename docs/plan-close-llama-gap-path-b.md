# Strategy — Path B: tolerance-level parity to close the llama.cpp gap

## Decision (2026-08-13)

Atlas's byte-identical greedy-stream sentinel (`f23c2962…`) forces exact FP32
accumulation order on every kernel. That mathematically caps performance below
llama.cpp:

- Decode attention (≈60% of decode GPU) uses an exact-ordered scan; llama's
  non-exact flash attention is ~5–15× faster. Decode parity is unreachable
  while the stream must stay byte-identical.
- Prefill GEMMs and decode matvecs are also constrained against llama-grade
  instruction selection (measured: even arithmetic-preserving restructures drift
  ~1 ulp from Metal codegen).

**Path B (chosen):** relax the correctness contract from *byte-identical
stream* to the *max-abs < 1e-3 kernel-level tolerance contract* (already the
phase contract for matmuls), re-baseline the greedy stream hash, and adopt
llama.cpp-style kernel algorithms (flash attention, high-occupancy q4 `mul_mm`).

## New correctness contract

- Per-kernel and end-to-end correctness asserts **max-abs < 1e-3** (and
  non-finite checks) instead of byte equality.
- The per-token fp32 logit-digest gate becomes a tolerance digest gate; the
  greedy stream hash is re-recorded and treated as a drift *diagnostic*, not a
  sentinel.
- Correctness still lives in CPU-oracle / reference-kernel tolerance tests
  (`batch_matmul_parity`, `attention_flash_correctness`, etc.).

## Work plan (priority order)

1. **Decode flash attention v4** (biggest lever, ≈60% of decode GPU). A
   block-tiled, merged-slice non-exact flash attention for q4 KV
   (`DEFINE_FLASH_ATTENTION_V2` slice-merge design, re-engineered for
   key_start/key_count, wider tiles, and better ILP). Target: decode attention
   ~19 → ~3–5 ms/token at pp512; decode ~31 → ~45–55 tok/s.

   **LANDED (phase-13.7).** `attention_decode_gemma4_simd_q4_0_flash16_v4` /
   `_swa_v4` bind to the production `Flash16` default. Decode 30.5 → 64.9
   tok/s at matched pp512/tg128 (decode GPU 4034 → 1895 ms/128; attention ~19 →
   ~2.5 ms/token). The stream still hashes `f23c2962…` on the matched
   workloads but diverges at token 50 on the chat prompt (expected Path B
   tradeoff). Model-level byte gates now exercise the `Flash16Exact` path.

2. **Relax parity gates**: digest gate → tolerance; re-baseline the stream
   hash; update the gap-analysis "drift sentinel" language to the tolerance
   contract. — In progress; the exact-path byte gates are preserved on
   `Flash16Exact`, and the v4 default is covered by the kernel-level tolerance
   test.

3. **Llama-grade q4 batched `mul_mm`** for prefill: larger tiles, input staging
   in threadgroup memory, deeper ILP. Target: prefill ~308 → ~600–1000 tok/s.

4. **Decode matvecs / flash-style mv** if the attention and prefill wins land.

## Acceptance framing

Each step must prove the new invariant (max-abs < 1e-3 vs the reference path),
retain the CPU-oracle/reference tolerance gates, and record measured
performance plus the re-baselined stream hash.
