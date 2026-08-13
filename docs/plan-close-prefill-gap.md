# Plan — Close the llama.cpp prefill gap (Path B, Phase 2)

## Goal

Prefill is ~311 tok/s vs llama.cpp's ~1150–1670 tok/s (4–5×), the only
remaining gap after decode reached ~1.1–1.3× (phase-13.7 Path B). Target:
prefill ~311 → **~600–1000+ tok/s** (gap → ~1.5–2.5×), keeping the Path B
max-abs < 1e-3 tolerance contract.

## Current bottleneck

`matmul_q4_0_batch_32row` (phase-13.6) is a weight-stationary 4-token × 32-row
tile that reads each weight block once and reuses it in registers across four
tokens (4 independent accumulator chains). It cut prefill weight re-reads 4×
and gave real ILP, but prefill is still compute/latency-bound (≈570× the pure
read floor): the K-reduction is scalar FMAs, while llama.cpp's Metal backend
uses **hardware `simdgroup_matrix_multiply`** with dequantized fp16 blocks and
fp32 accumulation — the fundamental ~4–6× GEMM throughput advantage.

## Steps

1. **Widen the weight-stationary tile (quick, measured).**
   `GEMMA4_BATCH_TILE_TOKENS` 4 → 8 (then 16): more independent accumulator
   chains and more weight reuse per thread. Measure at each size; keep the best
   non-regressing value. Low risk; registers grow ~1 float per token.

2. **Llama-grade q4 `mul_mm` with `simdgroup_matrix_multiply` (the real win).**
   New `matmul_q4_0_batch_simd` kernel:
   - dequantize each q4_0 block into 32 fp16 lane values (exact fp16
     dequantization of the fp32 scale × int8 nibble),
   - feed `simdgroup_matrix_multiply` (fp16×fp16 → fp32 accumulate) for the
     K-reduction over the token×row tile,
   - stage the shared weight block in threadgroup memory so it is read once
     per threadgroup and reused across the whole token slice,
   - fp32 accumulation inside the matrix op keeps the result inside the
     max-abs < 1e-3 contract vs the reference.
   Apply at the same prefill Q4 GEMM sites (qkv, gate-up, ffn-down, attn-out).

3. **Validate and gate.**
   - `batch_matmul_parity`: tolerance test (max-abs < 1e-3) of the new kernel
     vs `matmul_q4_0_batch_16row` across the Gemma4 geometries.
   - `cargo test --workspace`; `cargo fmt --check`.
   - Matched benchmark pp100/512/1024: record prefill tok/s, decode unchanged,
     stream hash (diagnostic), evidence under `artifacts/phase-13.8/`.
   - Keep/revert: if a step regresses prefill or exceeds the tolerance, revert
     to the previous best.

## Expected outcome

Step 1: ~311 → ~350–450 tok/s (tile widening). Step 2: → ~600–1000 tok/s
(matrix units). Together the gap closes from 4–5× to ~1.5–2.5×; the remaining
distance to llama's exact number is their further matmul tuning, which Atlas
can continue to absorb incrementally.

## Docs

`docs/atlas-metal-phases/phase-13.8-llama-grade-prefill-mul-mm.md`, README
index, gap-analysis prefill status.
