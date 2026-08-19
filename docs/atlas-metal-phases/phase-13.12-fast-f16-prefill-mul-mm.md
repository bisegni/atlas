# Phase 13.12 — Fast fp16 mul_mm for the per-layer model projection

## Problem

The per-kernel prefill profile (phase-13.10/13.11 opt-in stack, pp512) showed a
single GEMM dominating prefill: `matmul_f16_batch` at **~269 ms (37.5%)**, one
dispatch. Gemma 4's `per_layer_model_proj.weight` is stored **fp16** (not
q4_0), so the phase-13.10 `llama_mul_mm_q4_0_f32` path did not cover it, and it
fell through to the naive `matmul_f16_batch` — one thread per output element
with a full `K`-length dot product (no matrix units, no tiling). The profile
also corrected an earlier phase-13.11 estimate: prefill attention is **~22%**
(159 ms), not the ~4% recorded there (the +4% figure was a delta, not the
absolute fraction).

## Change

Vendored llama.cpp's `kernel_mul_mm_f16_f32` (same simdgroup-matrix
`kernel_mul_mm` core as the q4_0 variant, MIT-attributed, but the A operand is
fp16 with no dequantization — a direct `half4x4` load) as
`llama_mul_mm_f16_f32`.

- `crates/atlas-metal/src/kernels.metal` — `llama_mul_mm_f16_f32`.
- `crates/atlas-metal/src/lib.rs` — kernel registration.
- `crates/atlas-model/src/gemma4_executor.rs` — `matmul_f16_llama_mul_mm`;
  the F16 arm of `matmul_batch` routes `per_layer_model_proj` through it when
  `ATLAS_GEMMA4_LLAMA_MUL_MM` is set (falls back to `matmul_f16_batch`
  otherwise).  Since phase-13.19 the gate defaults to on (opt-out via
  `ATLAS_GEMMA4_LLAMA_MUL_MM=0`); at the time of this phase it was opt-in.
- `crates/atlas-metal/tests/llama_mul_mm_f16_parity.rs` — CPU-oracle parity.

## Parity

`llama_mul_mm_f16_f32_matches_cpu_oracle_relative` asserts RELATIVE < 5e-2 vs
the fp32 CPU oracle across N=5..33, K=128..1536, M=64..8960 (incl. the real
8960×1536 shape and partial-tile bounds-checked stores). Measured max_rel
~9e-3. **Green.**

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+5,
## llama_mul_mm + flash prefill)

| Prefill path | Prefill tok/s | Prefill ms | F16 GEMM ms |
|---|---:|---:|---:|
| llama mul_mm + flash (phase-13.11) | ~700 | ~728 | 269.42 |
| **+ fast f16 mul_mm (this phase)** | **~1132** | **~452** | **1.96** |
| llama.cpp reference | ~1600 | ~307 | — |

The single F16 GEMM drops **269.42 → 1.96 ms** (137×), cutting prefill
**728 → 452 ms** (~700 → ~1132 tok/s, +62%), closing the llama.cpp gap from
~2.4× to **~1.4×**. The greedy stream hash stays `f23c2962…` (the fp16 path
produces the same greedy top-1 selection as before). Decode is unchanged
(~64.5 tok/s; the kernel only touches prefill).

## Remaining hotspots (per-kernel profile, 452 ms)

- q4 `llama_mul_mm` projections: **~261 ms (56%)** — 275 dispatches.
- prefill flash attention: **~163 ms (35%)** — 35 dispatches (28 sliding-window).
- everything else (norms, rope, elementwise, PLE): ~40 ms.

## Command book

```zsh
cargo test -p atlas-metal --test llama_mul_mm_f16_parity -- --nocapture
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```
