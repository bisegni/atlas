# Phase 13.8 — llama.cpp-style fp16 matrix-unit prefill mul_mm (Path B, opt-in)

## Problem (the remaining ~4–5× prefill gap)

After phase 13.6/13.7, prefill is ~311–318 tok/s vs llama.cpp's
~1150–1670 tok/s (gap ~4–5×), the only remaining gap. The weight-stationary
scalar tile (`matmul_q4_0_batch_32row`) is at ~4× the weight-bandwidth floor,
and the opt-in fp16 staged matrix-unit kernel (`matmul_q4_0_batch_mm64`)
plateaus at ~370 tok/s pp512 no matter the tile width or staging overlap —
every 64-dim K-chunk pays a serialized dequant → threadgroup-staging → barrier
→ compute pipeline. `docs/plan-close-prefill-gap.md` records the only two
remaining routes as architectural, and this phase takes route (b): a
llama.cpp-style GEMM that feeds the matrix units with **no threadgroup weight
staging at all**.

## Change (Path B, opt-in fp16, approved 2026-08-17)

`crates/atlas-metal/src/kernels.metal` — three new kernels:

- `gemma4_q4_0_to_f16_batch` — dequantize one resident row-major GGUF Q4_0
  tensor into a contiguous fp16 layer buffer `[out][in]` (one thread per
  output row; nibble order matches `dequantize_block` in atlas-core).
- `gemma4_cast_f32_to_f16_batch` — cast the fp32 activation slice to fp16.
- `matmul_q4_0_batch_f16` — the hot GEMM: each SIMD group computes 8 tokens ×
  8 output rows, `simdgroup_load`-ing the fp16 activation and weight fragments
  **directly from device memory** and iterating the `simdgroup_multiply_accumulate`
  over K with persistent fp32 accumulators, then one `simdgroup_store`. No
  threadgroup staging, no per-K-chunk barrier — the architectural change that
  un-sticks the `mm64` plateau. It reuses the probe-verified contraction from
  `mm64` (a-load transpose=false, b-load transpose=true, transpose=false store;
  `docs/plan-close-prefill-gap.md`).

`crates/atlas-model/src/gemma4_executor.rs` — opt-in behind
`ATLAS_GEMMA4_MUL_MM`. When set, the executor allocates two fp16 scratch
buffers (one layer-dequant weight buffer sized to the largest batched
projection, one fp16 activation slice) and `matmul_batch`/`matmul_ffn_down_batch`
route Q4_0 matmuls through the three-dispatch sequence (dequant → cast → GEMM)
behind diagnostic labels `layer_major_batched_*_mul_mm_f16`. The geometry is
recovered from the q4_0 byte length (no host scalar readback). The fp32 scalar
tile remains the default under the max-abs < 1e-3 contract.

**This is tolerance-level**: fp16 inputs give ~3–4e-4 relative error vs the
fp32 reference — llama.cpp's own accuracy level, well inside the loosened
prefill contract but outside the tight 1e-3 default, so it is gated off by
default and the existing `batch_matmul_parity` fp32 tests are untouched.

## Licensing

The fp16 mul_mm technique (register-level fragment GEMM, fp16 weights, no
threadgroup staging) is a faithful re-implementation distilled from llama.cpp's
`ggml/src/ggml-metal/ggml-metal.metal` `kernel_mul_mm`, which is **MIT
licensed**. Atlas is Apache-2.0; the two are compatible but MIT attribution
must be preserved. `kernels.metal` documents the derivation at the new kernels.

## Parity contract (new test)

`crates/atlas-metal/tests/batch_matmul_parity.rs` —
`batch_mul_mm_f16_measures_accuracy` asserts the dequant→cast→GEMM path matches
the FP32 reference within **relative < 5e-2** (measured ~3–4e-4) across the
Gemma4 geometries, plus the "final K-chunk zeroed" accumulation-regression
guard (a kernel that keeps only the last chunk fails it). The existing 16row /
32row / mm64 fp32+fp16 tests are unchanged.

## GPU acceptance evidence (M2 Max, 2026-08-17)

`cargo test -p atlas-metal --test batch_matmul_parity batch_mul_mm_f16` passes
on the local Apple M2 Max. Per geometry:

| batch | input | output | max_abs | max_ref | relative |
|---:|---:|---:|---:|---:|---:|
| 64 | 2304 | 4096 | 3.56e-2 | 8.64e1 | 4.13e-4 |
| 64 | 2048 | 2304 | 2.78e-2 | 8.72e1 | 3.19e-4 |
| 128 | 2304 | 2304 | 3.21e-2 | 1.10e2 | 2.91e-4 |

The last-chunk-zeroed guard passes at the same relative error (the kernel
accumulates across all K-chunks).

**Measured (2026-08-17, M2 Max, preliminary single-run):** the local
`models/gguf/gemma-4-E2B_q4_0-it.gguf` fixture IS present in this checkout.
`ATLAS_GEMMA4_MUL_MM=1 benchmark matched --prompt-tokens 512 --decode-tokens 64
--warmup-runs 0 --runs 1` gives **prefill ~286 tok/s (1790 ms) vs the scalar
baseline ~256 tok/s (2001 ms) = +12%**, with decode unchanged (~62 vs ~61
tok/s). This is a real but modest gain — **not** the llama-grade (~1600 tok/s)
jump. The hot GEMM no longer pays the per-chunk dequant/barrier pipeline, but
the fp16 prep passes (q4→fp16 dequant + fp32→fp16 cast) still run per layer,
and the scalar tile remains close. A full pp100/512/1024 five-run sweep under
`artifacts/phase-13.8/` is still the proper acceptance record.

## Acceptance gates (phase 13.8)

- `batch_mul_mm_f16_measures_accuracy` passes on Apple Silicon (relative < 5e-2,
  measured ~3–4e-4, last-chunk guard green);
- fp32 `batch_matmul_parity` tests unchanged and green;
- `cargo test --workspace` green (metal + model suites verified here;
  fixture-gated gemma tests need their fixture);
- `cargo fmt --check` clean;
- evidence recorded in this document;
- (pending) matched benchmark pp100/512/1024 prefill tok/s with `--model
  gemma4-e2b-q4_0` under `ATLAS_GEMMA4_MUL_MM=1`, decode unchanged, under
  `artifacts/phase-13.8/`.

## Command book

```zsh
# kernel-level GPU parity (validated here on M2 Max)
cargo test -p atlas-metal --test batch_matmul_parity batch_mul_mm_f16 -- --nocapture

# end-to-end Gemma prefill comparison (needs the local Gemma 4 E2B Q4 GGUF)
ATLAS_GEMMA4_MUL_MM=1 cargo run --release -p atlas-cli -- benchmark matched \
  --model gemma4-e2b-q4_0 --prompt-tokens 512 --decode-tokens 128 \
  --warmup-runs 1 --runs 5 --output-format json
```