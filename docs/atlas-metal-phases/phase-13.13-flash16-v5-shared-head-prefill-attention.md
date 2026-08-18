# Phase 13.13 — Flash16-v5 shared-head prefill attention

## Problem

After phase-13.12 the per-kernel profile showed prefill attention at **~163 ms
(35%)**. Gemma 4 E2B uses a single shared KV head (`kv_heads == 1`), so the
phase-13.11 `flash16_v4` prefill kernel — one threadgroup per (token, head) —
re-dequantized the SAME q4_0 K/V cache once per head (8× redundant dequant).

## Change

Added `DEFINE_FLASH_ATTENTION_V5_BATCH`: one threadgroup per token with one
SIMD group per head (8 SIMD groups, 256 threads). The K/V q4_0 dequantization
for each key is done ONCE cooperatively into threadgroup memory and shared
across all heads, then each head computes its score / online-softmax / value
accumulation from the shared dequantized K/V.

- `crates/atlas-metal/src/kernels.metal` — `attention_prefill_gemma4_simd_q4_0_flash16_[swa_]v5`.
- `crates/atlas-metal/src/lib.rs` — kernel registration.
- `crates/atlas-model/src/gemma4_executor.rs` — the flash-prefill arm now
  dispatches v5 (grid = tokens, 256 threads) instead of v4.
- `crates/atlas-metal/tests/flash16_v5_parity.rs` — v5 vs v4 tolerance parity.

## Parity

`flash16_v5_[swa|full]_matches_v4_relative` compares v5 against v4 with the
correct 9-buffer binding (output at index 2). Measured max_rel ~4e-8 (fp32
rounding from the different reduction order). End-to-end greedy hash stays
`f23c2962…`. **Green.**

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+3,
## llama_mul_mm + fp16 mul_mm + flash prefill)

| Prefill path | Prefill tok/s | Prefill ms | Attention ms |
|---|---:|---:|---:|
| phase-13.12 (flash16_v4 attention) | ~1132 | ~452 | ~163 |
| **+ flash16_v5 shared-head attention** | **~1356** | **~378** | **~80** |
| llama.cpp reference | ~1600 | ~307 | — |

Attention drops **~163 → ~80 ms** (51%), cutting prefill **~452 → ~378 ms**
(~1132 → ~1356 tok/s, +20%), closing the llama.cpp prefill gap from ~1.4× to
**~1.25×**. The remaining prefill is now GEMM-dominated (~264 ms, 68%), at
llama-grade `mul_mm` throughput.

## Command book

```zsh
cargo test -p atlas-metal --test flash16_v5_parity -- --nocapture
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```
