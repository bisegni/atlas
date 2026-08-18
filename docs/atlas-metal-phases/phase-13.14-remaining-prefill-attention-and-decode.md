# Phase 13.14 — Remaining gap: matrix-unit prefill attention + decode

## Status: current target (not yet done)

Phases 13.10–13.13 closed the prefill gap from ~5× to ~1.2× on M2 Max. This
phase records the two remaining gaps, the evidence for why each is a genuine
(not measurement) gap, and the proposed next steps. Neither has an acceptance
gate recorded yet.

## Current measured state (M2 Max, Resident, q4_0 KV, pp512/tg128)

| Metric | Atlas | llama.cpp | Gap |
|---|---:|---:|---:|
| Prefill tok/s | ~1390 | ~1600 | 1.2× |
| Decode tok/s | ~64.5 | ~122 | 1.9× |

Prefill per-kernel profile (385 ms): q4 `llama_mul_mm` 264 ms (68%, at
llama-grade throughput), flash16-v5 attention ~80 ms (21%), everything else
~40 ms.

## Lever 1 — matrix-unit prefill attention (prefill 1.2× → ~1.07×)

**What it is.** The flash16-v5 kernel (phase-13.13) shares the K/V q4_0 dequant
across heads, but it is *latency-bound* on a serial per-key chain:
`dequant → dot-product → simd_sum → exp → accumulate`. llama.cpp's
`kernel_flash_attn_ext` instead tiles query-blocks × key-blocks and computes
`S = Q·Kᵀ` and `O += P·V` as `simdgroup_matrix` multiplies, batching the
per-key work onto the matrix units and hiding latency.

**Evidence it is the right lever.**
- Phase-13.13 measurements: v5 attention ~80 ms vs llama's ~40 ms.
- An experiment adding key-blocking (dequantizing `KEY_BLOCK` keys per barrier)
  produced **no speedup** — reducing barriers does not help because the kernel
  is thread-limited (256 threads → 4 threadgroups/core) and already hides
  barrier latency; the serial per-key compute is the ceiling.

**Scope.** This is a large (~300-line) port, not a quick edit: q4_0 K/V
dequant into f16 threadgroup memory, f16 Q staging, `simdgroup_matrix` for the
two small matmuls (M≈8 queries), online softmax with causal + sliding-window
masking via the `key_control` table, and f32 accumulation. Precision must be
re-validated against the existing `f23c2962…` hash and a tolerance parity test
(the greedy top-1 hash alone will not catch logit-level drift).

**Expected win.** ~40 ms of prefill (~1390 → ~1560 tok/s), closing the prefill
gap to ~1.07×.

## Lever 2 — decode (1.9×)

Decode is unchanged by this session's work (~64.5 tok/s vs llama ~122). The
phase-13.x history concluded decode is GPU-latency-bound rather than
dispatch-bound (see phase-13.9: dispatch fusion cut dispatch count 12.7% with
no throughput change). This is a separate, still-open gap that needs its own
occupancy/latency investigation, not more prefill work.

## Command book (for when the gates are defined)

```zsh
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```
