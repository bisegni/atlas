# Phase 13.16 — Decode matvec: 16-row-per-threadgroup kernels (simdgroup-band shrink)

## Status: [done]

The phase-13.15 negative (mul_mm at N=1 is 2.8× slower) pointed back to the
mv_ext matvec family shape as the decode matvec floor; this phase attacks the
remaining geometry lever instead — the simdgroup band width of the mv_ext
kernels.  Shrinking the per-threadgroup row band from the llama.cpp-current
64 rows (16 lanes x 4 rows) to **16 rows (16 lanes x 1 row)** — the same band
width as the mv_ext original whose ports this family descends from — makes the
decode q4_0 matvec family measurably faster: **decode −6.1% e2e** (mean
1863.0 vs 1984.3 ms for 128 tokens, 68.7 vs 64.5 tok/s) with the greedy stream
hash `f23c2962…` preserved on all 10 runs (5 per path), and per-token GPU
**20.52 → 19.51 ms (−4.9%)** per exact per-dispatch profile.  The 16-row
kernels are bitwise identical to the 64-row family on every tested shape
(partial-width rows included), so the win is free of numerical drift.

The 16-row variants are opted in with `ATLAS_GEMMA4_DECODE_16ROW=1`
(no default flip yet — see Outstanding).

## Change

Two kernels added in `crates/atlas-metal/src/kernels.metal`:
`matvec_q4_0_16row_mv` and `matvec_q4_0_16row_mv_rms`, plus their registry
entries in `crates/atlas-metal/src/lib.rs`.  They keep the exact per-lane
block stride, y-cache scale order, and block-dot accumulation order of the
64-row kernels and differ only in the simdgroup band: 1 row per group of 16
lanes instead of 4, requiring one 16-lane simdgroup per 16-row group.

`crates/atlas-model/src/gemma4_executor.rs` routes the three decode matvec
sites that use the standalone mv_ext kernels — ffn-down, attention-output
(wo), and the PLE projection — through the 16-row bindings when
`ATLAS_GEMMA4_DECODE_16ROW` is set (`gemma4_decode_16row_enabled`,
`gemma4_decode_matvec_binding`, `gemma4_decode_rms_matvec_binding`).
The fused qkv (32-row-banded) and ffn gate/up (fused 2-row) kernels are
unchanged.

The shared-KV and per-layer KV paths, the RMS fusion order, and the dispatch
count per token are unchanged (478 dispatches/token before and after), so the
resident-stream hash is preserved by construction.

## Correctness evidence

`crates/atlas-metal/tests/matvec_16row_parity.rs` (3 tests, all pass on
Apple Silicon via Metal):

- `matvec_q4_16row_mv_matches_cpu_oracle` — tolerance parity
  (max_abs < 1e-3) against the independent CPU dequant oracle across
  input_widths 32/256/2048 and output widths 16/31/33/96/129/137 (includes
  partial-width groups).
- `matvec_q4_16row_mv_rms_matches_cpu_oracle` — same contract for the
  RMS-input kernel.
- `matvec_q4_16row_mv_is_bitwise_identical_to_64row` — **bitwise identity**
  of both kernels vs the production 64-row kernels on 31/33/96/129/137/2048
  output widths.

## Measured evidence (M2 Max, Resident, real gguf E2B fixture, pp512/tg128, warmup1+5)

| Decode path | Decode ms (5-run) | Decode tok/s ≈ | Stream hash |
|---|---:|---:|---|
| default mv_ext 64-row | 1973.7–1994.6 (mean 1984.3) | 64.5 | `f23c2962…` exact |
| **16-row (opt-in)** | **1853.2–1874.0 (mean 1863.0)** | **68.7** | `f23c2962…` exact |

- Both benches in one session, same process binary, prefill ~1625–1658 ms
  (essentially flat), so the decode delta is cross-run comparable.
- Exact per-dispatch profile A/B (fresh records, indices 21/22 in
  `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl`, appended by
  this phase's runs): per-token GPU **20.52 → 19.51 ms (−4.9%)**.
- Family deltas (16-row vs 64-row, per-token GPU):
  - ffn-down 1.920 → 1.359 ms **−29.2%** (the largest dispatch-time matvec).
  - attention-output (wo) 0.764 → 0.626 ms **−18.1%**.
  - PLE projection 2.198 → 1.956 ms **−11.0%** (106 dispatches are
    quarter-width so the band shrink helps less per dispatch).
  - untouched families shift within run noise: swa attention +1.0%,
    flash16 −0.7%, qk_norm_rope −3.1%, gate/up +3.2% — note qkv/gate_up are
    still the fused 32-row kernels, so their deltas are bandwidth-relief
    effects, not geometry changes (and not part of the claimed win).
- No parity enforcement survives into production: the hash is the acceptance
  signal (identical on all 10 runs) plus the bitwise unit contract above.

## Command book

```zsh
# Parity gates
cargo test -p atlas-metal --test matvec_16row_parity

# A/B bench (64-row vs 16-row), output as above
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
ATLAS_GEMMA4_DECODE_16ROW=1 cargo run --release -p atlas-cli -- benchmark \
  matched --model gemma4-e2b-q4_0 --prompt-tokens 512 --decode-tokens 128 \
  --warmup-runs 1 --runs 5 --output-format json

# Per-dispatch profile pair (records 21/22 in the phase-12a JSONL)
cargo run --release -p atlas-cli -- profile --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --warmup-decode-tokens 32 --decode-tokens 128
ATLAS_GEMMA4_DECODE_16ROW=1 cargo run --release -p atlas-cli -- profile \
  --model gemma4-e2b-q4_0 --prompt-tokens 512 --warmup-decode-tokens 32 \
  --decode-tokens 128
```

## Artifacts

- `crates/atlas-metal/tests/matvec_16row_parity.rs` — parity gates.
- `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl` records 21/22 —
  clean 64-row vs 16-row exact-per-dispatch pair (record 20 is the
  phase-13.15 mul_mm negative, record 19 its baseline).
- Bench outputs for this phase: `decode_ms` 1973.7–1994.6 (64-row) and
  1853.2–1874.0 (16-row), hash `f23c2962…` x5 each.

## Outstanding

The two largest remaining matvecs — fused ffn gate/up (2.784 ms/token, 35
dispatches) and fused qkv (0.943) — still run the 32-row-banded fused kernels,
which need a separate band-width variant (their 3-way/2-way fused structure
does not share the standalone kernel geometry).  Extending the 16-row band to
those sites is the natural next increment, worth roughly the remaining matvec
share if the ~6% pattern holds.  A default flip of `ATLAS_GEMMA4_DECODE_16ROW`
is also pending one more sustained-environment confirmation.