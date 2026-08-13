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

   **LANDED (phase-13.8 step 1, commit 442a74a).** Tile 8 gives ~318 tok/s at
   pp512 (+2% over tile 4); kept as non-regressing. The scalar-FMA K-reduction
   is the remaining wall.

2. **Llama-grade q4 `mul_mm` with `simdgroup_matrix_multiply` (the real win).**
   New `matmul_q4_0_batch_simd` kernel:
   - dequantize each q4_0 block into fp32 lane values,
   - feed `simdgroup_multiply_accumulate` (fp32 8×8×8) for the K-reduction,
   - stage the shared weight block in threadgroup memory (64-dim chunks).
   Apply at the same prefill Q4 GEMM sites (qkv, gate-up, ffn-down, attn-out).

   **ATTEMPTED AND REVERTED (phase-13.8 step 2).** The fp32 8×8×8 matrix-unit
   path measured SLOWER than the scalar tile (prefill 199 vs ~318 tok/s at
   pp512) and broke the tolerance contract (max-abs 7.2e-3 vs 1e-3) with a
   drifted stream. On M2 the fp32 matrix unit is not faster than the scalar
   FMAs for this q4 workload, and the per-64-dim dequant + threadgroup barrier
   overhead dominates; the fp32-unit reduction order also exceeds the 1e-3
   contract. The kernel, registration, executor binding, and parity test were
   reverted. The simdgroup path is recorded as a dead end for this contract;
   a future attempt would need a looser (≥1e-2) prefill tolerance and the fp16
   16×16 unit, which is a further contract change.

3. **Validate and gate.**
   - `batch_matmul_parity`: tolerance test (max-abs < 1e-3) of the new kernel
     vs `matmul_q4_0_batch_16row` across the Gemma4 geometries.
   - `cargo test --workspace`; `cargo fmt --check`.
   - Matched benchmark pp100/512/1024: record prefill tok/s, decode unchanged,
     stream hash (diagnostic), evidence under `artifacts/phase-13.8/`.
   - Keep/revert: if a step regresses prefill or exceeds the tolerance, revert
     to the previous best.

## Expected outcome

Step 1: ~311 → ~318 tok/s (landed, marginal). Step 2 (simdgroup matrix units):
measured as a regression on M2 within the 1e-3 contract and reverted — the
matrix-unit path is not viable here without a looser prefill tolerance. The
prefill gap remains ~4–5×; closing it further needs either (a) a scalar-GEMM
ILP/occupancy improvement that keeps fp32 accuracy, or (b) an accepted looser
prefill tolerance to unlock the fp16 matrix-unit path.

## Docs

`docs/atlas-metal-phases/phase-13.8-llama-grade-prefill-mul-mm.md`, README
index, gap-analysis prefill status.
