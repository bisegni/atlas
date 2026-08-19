# Phase 13.19 — Fast prefill by default: mul_mm + flash16 prefill are the production path

## Status: [done]

The phase-13.10/13.11/13.12 prefill stack (vendored llama.cpp `mul_mm` GEMMs +
Flash16 batched prefill attention, v7 since phase-13.14) was opt-in behind
`ATLAS_GEMMA4_LLAMA_MUL_MM=1` and `ATLAS_GEMMA4_FLASH_PREFILL=1`; a plain
benchmark therefore ran the legacy prefill path (~317 tok/s at pp512) while
the opt-in path measured ~1500 tok/s.  This phase flips both gates to
production defaults, making the fast path the out-of-the-box behavior.

## Change

`crates/atlas-model/src/gemma4_executor.rs`:

- `gemma4_llama_mul_mm_enabled()` — default **true**; opt-out via
  `ATLAS_GEMMA4_LLAMA_MUL_MM=0` (falls back to the fp32 scalar tile /
  `matmul_f16_batch`).
- `gemma4_flash_prefill_enabled()` — default **true**; opt-out via
  `ATLAS_GEMMA4_FLASH_PREFILL=0` (falls back to the bitwise batched prefill
  scan).  The flash arm still requires q4_0 KV; v7 remains the prefill
  attention kernel of record, and `ATLAS_GEMMA4_FLASH_PREFILL_V5=1` still
  falls back to v5 within the flash arm.
- The separate phase-13.8 `ATLAS_GEMMA4_MUL_MM` and `ATLAS_GEMMA4_MM64` gates
  are untouched (not part of this default; the covered fast path is exactly
  the phase-13.14 command-book configuration).

## Parity

No kernel changed; the default flip re-routes the prefill projection GEMMs
(fp16 fragments, tolerance-level ~9e-3 max_rel vs CPU per phase-13.12) and the
prefill attention (tolerance-level per phase-13.11/13.14).  The acceptance
invariant is the established one for prefill changes in this stack: the
**on-fixture greedy stream hash is byte-identical** to the previous production
path (`f23c2962…`, verified below in both directions of the flip).

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+3)

| Configuration | Prefill ms | Prefill tok/s | Decode ms | Greedy hash |
|---|---:|---:|---:|---|
| **Default (post-flip)** | 339–341 | **1505–1510** | 1850–1866 | `f23c2962…` |
| Opt-out (`*_MUL_MM=0` + `*_FLASH_PREFILL=0`) | 1615–1617 | ~317 | 1836–1853 | `f23c2962…` |
| Pre-flip default (no envs, phase-13.18 state) | ~1646 | ~311 | ~1859 | `f23c2962…` |

Decode is flat across all three (68.6–69.7 tok/s, within run noise): the flip
is prefill-only, as designed.

## Command book

```zsh
# Production default (fast prefill, no env vars needed):
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 3 \
  --output-format json

# Legacy prefill A/B (opt-out):
ATLAS_GEMMA4_LLAMA_MUL_MM=0 ATLAS_GEMMA4_FLASH_PREFILL=0 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 3 \
  --output-format json
```

Expected pass signal: default prefill ~1500 tok/s, opt-out ~317 tok/s, greedy
stream hash `f23c2962…` in both.

## Outstanding

Batch decode (N > 1) remains the open decode lever; prefill is now at the
fast-path default and the legacy scan needs no env to restore.