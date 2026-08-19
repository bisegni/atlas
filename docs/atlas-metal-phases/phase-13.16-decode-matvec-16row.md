# Phase 13.16 — Decode matvec: 16-row-per-threadgroup kernels (simdgroup-band shrink)

## Status: [done]

The phase-13.15 negative (mul_mm at N=1 is 2.8× slower) pointed back to the
mv_ext matvec family shape as the decode matvec floor; this phase attacks the
remaining geometry lever instead — the simdgroup band width of the mv_ext
kernels.  Shrinking the per-threadgroup row band from llama.cpp-current's 64
rows (16 lanes × 4 rows) to **16 rows (16 lanes × 1 row)** on the standalone
matvecs, and from 32 rows (4 SIMD groups × 8 rows) to **16 rows (4 SIMD groups
× 4 rows)** on the fused qkv and gate/up kernels, makes the full decode Q4
matvec family measurably faster: **decode −7.8% e2e** (mean 1850.1 vs
2006.8 ms for 128 tokens, 69.2 vs 63.8 tok/s) with the greedy stream hash
`f23c2962…` preserved on all 10 runs (5 per path), and per-token GPU
**20.89 → 19.25 ms (−7.9%)** per exact per-dispatch profile.  Every 16-row
variant is bitwise identical to its predecessor (64-row standalone, 32-row
fused) on all tested shapes, so the win is free of numerical drift.

The 16-row path is the **production default** since this phase; set
`ATLAS_GEMMA4_DECODE_16ROW=0` to opt back into the 64/32-row stack for A/B
comparisons.

## Change

Six kernels in `crates/atlas-metal/src/kernels.metal` — standalone
`matvec_q4_0_16row_mv[_rms]` and fused
`matmul_q4_0_qkv_16row_mv_rms` / `matmul_q4_0_gate_up_16row_mv_rms` — plus
their registry entries in `crates/atlas-metal/src/lib.rs`.  They keep the
exact per-lane block stride, y-cache scale order, RMS sum-of-squares order,
and block-dot accumulation order of the 64/32-row kernels and differ only in
the simdgroup band (1 row per 16-lane group for the standalone kernels, 4
rows per 32-lane group for the fusions) and the finer threadgroup grid
(`ceil(width/16)` instead of `ceil(width/64)` / `ceil(width/32)`).

`crates/atlas-model/src/gemma4_executor.rs` selects the 16-row bindings on
every decode Q4 matvec site when `gemma4_decode_16row_enabled()` (default
`true`): ffn-down, attention-output (wo), the PLE projection, the shared-KV
query, and the fused qkv and gate/up projections
(`gemma4_decode_matvec_binding`, `gemma4_q4_rms_projection_binding`,
`gemma4_q4_fused_projection_binding`).  `gemma4_kernel_family` maps the new
fused names into their `q4_qkv_projection` / `q4_ffn_gate_up_projection`
profile families.  Dispatch counts, RMS fusion order, and buffer layouts are
unchanged (478 dispatches/token before and after), so the resident-stream
hash is preserved by construction.

## Correctness evidence

`crates/atlas-metal/tests/matvec_16row_parity.rs` (6 tests, all pass on
Apple Silicon via Metal):

- `matvec_q4_16row_mv_matches_cpu_oracle` / `..._rms_...` — tolerance parity
  (max_abs < 1e-3) against the independent CPU dequant oracle across
  input_widths 32/256/2048 and output widths 16/31/33/96/129/137.
- `matvec_q4_16row_mv_is_bitwise_identical_to_64row` — bitwise identity of
  both standalone kernels vs the production 64-row kernels on
  31/33/96/129/137/2048 output widths.
- `matmul_q4_16row_qkv_fused_matches_cpu_oracle` /
  `..._gate_up_fused_...` — CPU-oracle parity for the fused kernels
  (q/k/v and gate/up, RMS fused) on partial-group widths (33/17, 129/33,
  33/97/137 rows).
- `matmul_q4_16row_fused_is_bitwise_identical_to_32row` — bitwise identity
  of both fusions vs the 32-row kernels on 31/17, 129/33, 512/256 and
  31/97/512 row sets.

## Measured evidence (M2 Max, Resident, real gguf E2B fixture, pp512/tg128, warmup1+5)

| Decode path | Decode ms (5-run) | Decode tok/s ≈ | Stream hash |
|---|---:|---:|---|
| 64/32-row stack (`ATLAS_GEMMA4_DECODE_16ROW=0`) | 1986.8–2022.9 (mean 2006.8) | 63.8 | `f23c2962…` exact |
| **16-row (production default)** | **1840.2–1874.9 (mean 1850.1)** | **69.2** | `f23c2962…` exact |

- Both benches in one session, same process binary, prefill ~1625–1665 ms
  (essentially flat), so the decode delta is cross-run comparable.
- Exact per-dispatch profile A/B (fresh records, indices 23/24 in
  `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl`): per-token GPU
  **20.89 → 19.25 ms (−7.9%)**.
- Family deltas (16-row vs 64/32-row baseline, per-token GPU):
  - ffn-down 1.957 → 1.333 ms **−31.9%** (largest dispatch-time matvec).
  - attention-output (wo) 0.782 → 0.609 ms **−22.1%**.
  - qkv projection 0.970 → 0.781 ms **−19.5%** (fused 16-row).
  - PLE projection 2.271 → 1.949 ms **−14.2%**.
  - ffn gate/up 2.834 → 2.603 ms **−8.2%** (fused 16-row).
  - untouched families shift within run noise: swa +0.1%, flash16 +0.8%,
    qk_norm_rope −2.8%, norms −0.8…−1.8%, kv_append −0.7%, argmax +1.0%.
- Without the fused extension (phase mid-point) the same two-session A/B
  measured −5…−6% with ffn-down −29%, wo −18%, PLE −11% and untouched fused
  kernels; the fused band shrink adds the qkv −19.5% / gate-up −8.2%
  increments on top.

## Command book

```zsh
# Parity gates
cargo test -p atlas-metal --test matvec_16row_parity

# A/B bench (64/32-row stack vs 16-row default), output as above
ATLAS_GEMMA4_DECODE_16ROW=0 cargo run --release -p atlas-cli -- benchmark \
  matched --model gemma4-e2b-q4_0 --prompt-tokens 512 --decode-tokens 128 \
  --warmup-runs 1 --runs 5 --output-format json
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json

# Per-dispatch profile pair (records 23/24 in the phase-12a JSONL)
ATLAS_GEMMA4_DECODE_16ROW=0 cargo run --release -p atlas-cli -- profile \
  --model gemma4-e2b-q4_0 --prompt-tokens 512 --warmup-decode-tokens 32 \
  --decode-tokens 128
cargo run --release -p atlas-cli -- profile --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --warmup-decode-tokens 32 --decode-tokens 128
```

## Artifacts

- `crates/atlas-metal/tests/matvec_16row_parity.rs` — parity gates (6 tests,
  standalone + fused).
- `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl` records 23/24 —
  clean 64/32-row vs all-16-row exact-per-dispatch pair (records 21/22 are
  the mid-point standalone-only comparison, record 20 the phase-13.15 mul_mm
  negative, record 19 its baseline).
- Bench outputs for this phase: `decode_ms` 1986.8–2022.9 (64/32-row) and
  1840.2–1874.9 (16-row), hash `f23c2962…` ×5 each.

## Outstanding

Decode is now below ~19.3 ms/token GPU (~69 tok/s) versus the phase start
(~20.9 ms, ~64 tok/s) without any numerical drift.  Remaining decode levers,
as priced in phase-13.0/13.9/13.15: (a) attention — largest family at ~17.2%
of per-token GPU (3.056 swa + 1.351 full + 2.358 qk_norm_rope_fused), with
the qk_norm_rope→attention fusion still blocked by the quantized-cache
dependency (P2d); (b) PLE projection (1.949 ms) and lm_head q6_k (1.251 ms);
(c) batch decode (N > 1) where the mul_mm machinery finally pays off.