# Phase 13.11 — Flash16-v4 batched prefill attention

## Problem

The prefill attention used the serial per-key scan
(`attention_decode_fused_gemma4_simd_q4_0_batch`): one threadgroup per
(token, head) loops over keys with 3 threadgroup barriers per key and a
single-thread online softmax. This is the pre-flash16 decode design that
phase-13.7 replaced for decode, but it had never been replaced for prefill.

## Change

`crates/atlas-metal/src/kernels.metal` — `DEFINE_FLASH_ATTENTION_V4_BATCH`
instantiates `attention_prefill_gemma4_simd_q4_0_flash16_v4` (full, 512-wide,
12 slices, 384 threads) and `..._swa_v4` (sliding, 256-wide, 24 slices, 768
threads). Same merged-slice design as the decode v4 kernel: each simdgroup
scans a disjoint key slice with register online-softmax + value accumulators,
then merges in threadgroup memory — no per-key barriers. The grid is
batch*heads; each threadgroup reads its token's packed key_control entry
(stride `layers`) so causal / sliding ranges are honored. Tolerance-level
(slice split + merge reorder the FP32 reduction), not bitwise.

`crates/atlas-model/src/gemma4_executor.rs` — opt-in
`ATLAS_GEMMA4_FLASH_PREFILL` routes `encode_prefill_layer_major_layer` through
the flash kernel (full vs swa chosen per layer); the bitwise batched scan stays
the default.

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+3,
## with ATLAS_GEMMA4_LLAMA_MUL_MM=1)

| Config | Prefill tok/s | Prefill ms |
|---|---:|---:|
| llama mul_mm only (phase-13.10) | ~674 | ~758 |
| + flash prefill attention | **~700** | ~728 |

The gain is modest (**+4%**) because Gemma 4 E2B uses mostly sliding-window
attention, so prefill attention is only ~4% of prefill (not the ~24% estimated
before profiling). The remaining ~90% is the GEMM. Greedy stream hash stays
`f23c2962...` (the flash perturbation does not change the greedy top-1).

## Verdict / implication

Correct and a real but small win. It confirms prefill is now GEMM-dominated:
closing the remaining ~2.3x gap to llama.cpp requires the faster prefill GEMM
(see phase-13.10 "remaining ~2.4x" — the candidate is llama.cpp's
MetalPerformancePrimitives `mul_mm`).

## Command book

```zsh
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```
