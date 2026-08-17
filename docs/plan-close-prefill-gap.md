# Plan — Close the llama.cpp prefill gap (Path B, Phase 2)

## Goal

Prefill is ~311 tok/s vs llama.cpp's ~1150–1670 tok/s (4–5×), the only
remaining gap after decode reached ~1.1–1.3× (phase-13.7 Path B). Target:
prefill ~311 → **~600–1000+ tok/s** (gap → ~1.5–2.5×), keeping the Path B
max-abs < 1e-3 tolerance contract.

## Current bottleneck (measured per-kernel profile, 454-token prompt)

| Prefill family | GPU ms | Share |
|---|---:|---:|
| `layer_major_batched_projection` (qkv, gate_up, attn_out) | 938 | 65.6% |
| `layer_major_batched_ffn_down_projection` | 289 | 20.2% |
| `layer_major_batched_attention` | 165 | 11.5% |
| everything else | ~1% | |

The q4 batched GEMM (`matmul_q4_0_batch_32row`) is **86% of prefill GPU**. It
runs at ~4× the weight-bandwidth floor (~370 ms: weights read once per 8-token
tile) and ~4× the compute floor (~350 ms with the dequant CSE'd across the 8
tokens). It is latency/occupancy-bound.

## Levers measured (2026-08-13)

| Approach | Prefill tok/s (pp512) | Verdict |
|---|---:|---|
| weight-stationary tile-4 (phase-13.6) | ~311 | +60% landed |
| tile-8 | ~318 | +2% landed (best scalar) |
| tile-16 | ~263 | −17% register-pressure spills |
| tile-8 + software prefetch | ~280 | −12% compiler already pipelines |
| matrix units fp32 8×8 | ~199 | slower + max-abs 7e-3 > 1e-3 |
| matrix units fp16 8×8 (Option B) | ~224 | slower + max-abs 2.6–6.5e-2, stream drifts |
| mm64 fp16 32-token tile (opt-in, 2bcb321) | ~373 / ~340 | landed opt-in (pp512/pp1024); gap 4.6× |
| mm64v2 64-token tile (2026-08-17) | ~368 / ~338 | wash (−1% vs mm64), reverted |
| mm64v2 + double-buffered staging (2026-08-17) | ~314 / ~290 | regression (32KB threadgroup → 1 tg/core), reverted |

## Conclusion

Every structural lever is exhausted within the 1e-3 contract: wider tiles and
prefetch regress (registers/pipeline), and the M2's 8×8 matrix units (the only
size this toolchain exposes) are slower than the scalar tile once q4
dequant + threadgroup staging + barriers are paid, and fp16 breaks the
tolerance. The realistic remaining options are:

1. **Accept ~4–5× prefill gap** (decode is at parity; prefill 318 tok/s is a
   defensible stopping point).
2. **fp16 pre-dequantized weights + fp16 matrix path** (llama-grade): dequantize
   q4 → fp16 once per prompt into a device buffer so the GEMM needs no
   per-chunk dequant/barrier, and accept a ~1e-2 prefill tolerance (llama.cpp's
   own accuracy level). This is a firm contract decision and the only route to
   llama-grade prefill numbers.
3. **Scalar micro-tuning** (occupancy/register reduction) — marginal, ~5–15%.

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

## simdgroup matrix-unit layout findings (2026-08-17, `rev_probe` harness)

The opt-in `matmul_q4_0_batch_mm64` kernel (commit 2bcb321) landed at
pp512 373 / pp1024 340 tok/s vs llama.cpp 1724 / 1657 — still ~4.6× off.
A probe harness (`crates/atlas-metal/examples/rev_probe.rs` + temporary
`rev_probe_*` kernels, raw output in `artifacts/mm64-layout/`) pinned down
the exact `simdgroup_half8x8` semantics on this M2 Max / toolchain so the
next kernel iteration can be written deliberately. All findings below were
cross-validated against the four transpose-flag mma combos and the shipping
parity-tested production kernel.

Semantics (element-granular; no 16-byte alignment requirement observed on
base pointers — offsets 2/4/8/64 all behaved exactly as addressed):

1. Load/store addressing: with base pointer `P`, leading stride `S`, and
   `matrix_origin = (x, y)`, the touched flat indices are
   `y*S + x + r*S + c` (r = row 0..7, c = column 0..7). `origin (0, y)` is
   equivalent to advancing the pointer by `y*S`.
2. `simdgroup_load(..., transpose = true)` reads the 8×8 block
   **column-major** into the register matrix:
   `reg[r][c] = mem[y*S + x + c*S + r]`.
3. `simdgroup_load(..., transpose = false)` reads it row-major:
   `reg[r][c] = mem[y*S + x + r*S + c]`.
4. `simdgroup_multiply_accumulate(d, a, b, c)` computes `d = a_reg · b_reg + c`
   (standard mathematical order).
5. `simdgroup_store(..., transpose = true)` writes the transposed register:
   `mem[y*S + x + r*S + c] = reg[c][r]`.

The production `mm64` combo follows directly: weights staged row-major are
loaded with `transpose = false` (register = row×k), the staged input slice
is loaded with `transpose = true` (register = k×token), the mma gives
row×token, and the transposed store lands it token-major in the output.

Harness post-mortem (why earlier bisects looked broken — no hardware
weirdness was found for in-bounds use):

- The `rev_probe_tg_*` kernels declare the output at `buffer(2)` but the
  harness bound only two buffers, so their stores hit an unbound slot and
  the printed grids were residue from the previous probe.
- The e2e "worst vs CPU dot" comparisons modeled neither the transposed
  store nor the harness's own sparse fill pattern, so their huge errors
  were expectation bugs, not GPU misbehavior (the all-simdgroup-equal
  `no_sg` cells confirm the loads/accumulation were consistent).
- Out-of-bounds load/store behavior is undefined (observed: silent drops,
  adjacent-allocation reads, apparent wraparound depending on the heap
  layout). Production kernels must stay in-bounds; `mm64` does via its
  even-geometry gate.

Design consequences for the next iteration:

- The input slice **cannot** be loaded directly from the device tensor:
  activations are fp32 and `simdgroup_load` on `simdgroup_half8x8` only
  reads fp16, so the fp32→fp16 cast must go through staged memory (or a
  separate cast pass). Weight dequant likewise still needs staging.
- The `mm64` bottleneck is therefore its serialized chunk pipeline: with a
  32-token tile each weight chunk is dequantized once per 32 tokens, and
  every 64-dim chunk pays fill → barrier → compute → barrier with no
  overlap. Two iterations were measured on 2026-08-17 (M2 Max, release,
  `benchmark matched` pp512/pp1024, 1 warmup + 5 measured runs, tg128;
  scalar baseline re-measured the same session at 322 / 299.6 tok/s):
  - 64-token tile (`matmul_q4_0_batch_mm64v2`, halves per-token dequant
    and threadgroups): pp512 ~368, pp1024 ~338 tok/s — a wash vs mm64
    (373 / 340), parity max-abs in the same 1.6e-2..6.6e-2 fp16 class.
    Widening the tile alone is not the lever.
  - Same tile with double-buffered staging (next chunk dequantized into
    the alternate buffer during compute, one barrier per chunk): pp512
    ~314, pp1024 ~290 tok/s — a regression. The double buffers take
    threadgroup memory to 32KB, dropping residency to one threadgroup per
    core, which costs more than the overlapped fill buys.
  Both were reverted; `mm64` remains the opt-in fp16 path. Conclusion: the
  staged-dequant + 8×8-fp16-unit family plateaus at ~370 tok/s pp512 on M2
  Max regardless of tile width or staging overlap. The remaining routes to
  llama-grade prefill are architectural, not tile tuning: (a) fp16
  activations/weights staged once per layer so the GEMM has no per-chunk
  dequant or cast (a contract + executor memory-layout change), or
  (b) llama.cpp-style register-level q4 dequant feeding the matrix units
  with no threadgroup staging at all.
