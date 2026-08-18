# Phase 13.10 — Vendored llama.cpp mul_mm closes the prefill gap to ~2.4x

## Problem

After phase-13.6/13.7/13.8, prefill was still ~316 tok/s vs llama.cpp's
~1600 tok/s (~5x). Atlas's own matrix-unit attempts (`matmul_q4_0_batch_mm64`,
`matmul_q4_0_batch_f16`) plateaued at ~286-386 tok/s because they dequantize
the WHOLE weight into a device fp16 buffer (a separate prep pass) and then
re-read it — paying memory traffic + extra dispatches that eat the
matrix-unit advantage. The repo's analysis named the missing design as
"llama.cpp-style register/tile-level dequant feeding the matrix units," but
Atlas had not reproduced it.

## Change

Vendored llama.cpp's **classic simdgroup-matrix `kernel_mul_mm`** (MIT
licensed, attribution preserved in `kernels.metal`) as
`llama_mul_mm_q4_0_f32`, specialized for q4_0 x f32, single batch, contiguous
row-major. The compute core is llama.cpp's: each threadgroup computes a 64x32
output tile, dequantizes a 64x32 weight tile + a 32x32 activation tile INTO
threadgroup memory each K-step (no whole-weight buffer), then
`simdgroup_load` + `simdgroup_multiply_accumulate` over K with an fp32
accumulator. This is the "fill the matrix" design that realizes matrix-unit
throughput.

- `crates/atlas-metal/src/kernels.metal` — `llama_block_q4_0`,
  `llama_dequantize_q4_0`, `llama_mul_mm_q4_0_f32` (MIT-attributed).
- `crates/atlas-metal/src/lib.rs` — new `dispatch_threadgroups_2d_tgm`
  (2D grid + `setThreadgroupMemoryLength`) + kernel registration.
- `crates/atlas-model/src/gemma4_executor.rs` — opt-in
  `ATLAS_GEMMA4_LLAMA_MUL_MM` routes the prefill projections
  (`matmul_batch`, `matmul_ffn_down_batch`) through it; fp32 scalar stays the
  default.

## Parity

`crates/atlas-metal/tests/llama_mul_mm_parity.rs` —
`llama_mul_mm_q4_0_matches_cpu_oracle_relative` asserts RELATIVE < 5e-2 vs the
fp32 CPU oracle (the repo's fp16 mul_mm contract; measured max_rel
~3e-3..2e-2 across N=5..33, K=128..2304, M=64..4096, incl. partial-tile
bounds-checked stores). **Green.**

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+3)

| Prefill path | Prefill tok/s | Prefill ms | Decode tok/s |
|---|---:|---:|---:|
| fp32 scalar (production default) | ~316 | ~1600 | ~64.6 |
| phase-13.8 fp16 mul_mm (opt-in) | ~286 | ~1790 | ~62 |
| **vendored llama mul_mm (opt-in)** | **~674** | **~758** | ~64.5 |
| llama.cpp reference | ~1600 | ~307 | ~80 |

The vendored kernel **more than doubles Atlas prefill (316 -> 674 tok/s,
+113%)**, closing the llama.cpp gap from ~5x to **~2.4x**. Decode is
unchanged (the kernel only touches the prefill projections). The matched
greedy stream hash stays `f23c2962...` on this workload (the fp16 prefill
perturbation does not change the greedy top-1 selection).

## Verdict / implication

The prefill gap was the GEMM kernel design, not "something else" in Atlas's
pipeline: running llama.cpp's actual fill-the-matrix kernel yields llama-class
compute throughput. The remaining ~2.4x is the next target (candidate levers:
llama.cpp's newer MetalPerformancePrimitives `mul_mm` variant, the non-GEMM
prefill fraction — attention ~11.5%, norms, rope — and tile tuning). The
vendored path is fp16 tolerance-level, so it stays opt-in; the fp32 bitwise
default is unchanged.

## Command book

```zsh
cargo test -p atlas-metal --test llama_mul_mm_parity -- --nocapture
ATLAS_GEMMA4_LLAMA_MUL_MM=1 cargo run --release -p atlas-cli -- benchmark matched \
  --model gemma4-e2b-q4_0 --prompt-tokens 512 --decode-tokens 128 \
  --warmup-runs 1 --runs 5 --output-format json
```
