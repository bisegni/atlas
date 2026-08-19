# Phase 13.18 — Decode PLE input-gate fusion (dispatch reduction, measured wash for throughput)

## Problem

The per-layer decode PLE is still three small dispatches: `matvec_q4_0_16row_mv`
(`state * inp_gate` into the gate buffer), `ple_gelu_multiply_offset_f32`
(GELU * current-layer PLE slice into `activated`), and the projection matvec
into `work`. The first two are a single producer→consumer pair: the gelu pass
reads exactly the rows the matvec just wrote, through one extra gate-buffer
round trip. Phase-13.17 listed this as the remaining lever (a): 105 small
PLE dispatches/token at ~18 µs each, ~1 ms/token if the two-launcher-per-layer
minima could be fused bitwise.

## Change

`crates/atlas-metal/src/kernels.metal` — `gemma4_ple_gate_gelu_f32` replaces
the per-layer pair with one dispatch.  The matvec phase is the 16-row
kernel's per-lane block math verbatim (same `ix/il` mapping, `yl` scale
orders, `ib += 16` block stride, `sumy * -8 + acc0..acc3` accumulation, and
lane-0 row writes); the lane-0 write then applies the elementwise
GELU * PLE multiply exactly as `ple_gelu_multiply_offset_f32` does
(`0.7978845608f * (x + 0.044715f * x³)`, `atlas_tanh_f32`, the `isinf`
guard, and the scale).  Every `activated` element is therefore bitwise
identical to the two-kernel baseline, which makes the downstream projection
matvec (which depends only on `activated`) bitwise identical too.  The grid
geometry is unchanged from the matvec dispatch (`ceil(width/16)` groups ×
128 threads), so no new occupancy/scheduling structure is introduced.

`crates/atlas-model/src/gemma4_executor.rs` — the decode PLE block now issues
the fused dispatch by default and skips the gate-buffer round trip entirely
(`self.gate` is no longer written per layer).  `ATLAS_GEMMA4_PLE_SPLIT=1`
restores the two-kernel path for A/B.  The split path is unchanged, so the
phase-13.16 bitwise equivalence between bands is untouched.

## Parity (kernel-level)

`crates/atlas-metal/tests/dispatch_fusion_parity.rs` —
`ple_gate_gelu_fused_is_bitwise_identical_to_split` runs the fused kernel
against the actual production pair (`matvec_q4_0_16row_mv` →
`ple_gelu_multiply_offset_f32`) on-device at input widths 512/1024/2304 ×
widths 31/256/260/512/1280 with varying PLE offsets and asserts **bitwise
identity** (strict `to_bits` equality, not the 1e-3 tolerance).  **Green.**

## Measured evidence (M2 Max, Resident, q4_0 KV, flash16 v4, pp512/tg128)

Exact per-dispatch profile A/B (records 27/28 appended to
`artifacts/phase-12a/gemma4-resident-decode-profile.jsonl`; record 26 is the
pre-fusion 16-row baseline):

| Metric | Record 26 (split) | Record 27 (fused) | Record 28 (split re-measured) |
|---|---:|---:|---:|
| PLE-family dispatches / token | 106 | **71** (−35) | 106 |
| Total dispatches / token | 482 | **447** | 482 |
| `ple_projection` GPU / token | 1.942 ms | **1.727 ms** (−11.1%) | 1.978 ms |
| Per-token GPU total | 19.49 ms | **18.84 ms** (−3.3%) | 19.51 ms |

Matched bench A/B (fused default vs `ATLAS_GEMMA4_PLE_SPLIT=1`, 3 runs,
128 decode tokens): decode 1859.3 ± 15.5 vs 1852.9 ± 6.8 ms — **flat within
run noise** (68.8 vs 69.1 tok/s).  Greedy stream hash **`f23c2962…`
(two runs both `f23c2962…`) unchanged** in both configurations, confirming
the bitwise contract end-to-end.

## Verdict

The fusion is **bitwise-correct, strictly non-regressing, and keeps the
per-dispatch win** (−35 launches/token, −11% on the PLE family in the exact
profile), but like phase-13.9 it is **not a decode-throughput lever**: decode
e2e is GPU-latency-bound (matvec family + attention), and ~0.25 ms/token of
saved GPU time is within the 3-run bench variance.  Kept as the default
because it reduces host encode work and the gate round trip at no cost.

The remaining PLE head is the per-layer projection matvec + norm epilogue
(already fused once in phase-13.9); folding the projection matvec into the
same kernel would require cross-threadgroup ordering (projection needs all
`activated` rows, the norm needs all `work` rows) and thus a single
threadgroup — the throughput-bound trap measured in phase-13.17.  Remaining
decode levers stand: attention split-KV and batch decode.

## Command book

```zsh
cargo test -p atlas-metal --test dispatch_fusion_parity ple_gate_gelu -- --nocapture

# Fused default (production path):
cargo run --release -p atlas-cli -- profile --model gemma4-e2b-q4_0 --prompt-tokens 512
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 3 --output-format json

# Split opt-out for A/B:
ATLAS_GEMMA4_PLE_SPLIT=1 cargo run --release -p atlas-cli -- profile \
  --model gemma4-e2b-q4_0 --prompt-tokens 512
```

## Artifacts

- `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl` records 27/28 —
  fused vs split exact-per-dispatch pair at the same workload; the
  `ple_projection` family (106 → 71 dispatches, 1.978 → 1.727 ms/token) is
  the fusion evidence.
- Kernel: `gemma4_ple_gate_gelu_f32` (registered in `atlas-metal`).
- Parity: `ple_gate_gelu_fused_is_bitwise_identical_to_split`
  (bitwise contract, 15 sizes × offset variants).