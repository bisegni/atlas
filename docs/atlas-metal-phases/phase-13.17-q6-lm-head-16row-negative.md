# Phase 13.17 — q6_k lm_head via the 16-row band: negative result

## Status: [done]

The phase-13.16 band-shrink lever (16-row-per-threadgroup mv_ext kernels)
was applied to the last remaining 64-row matvec — the q6_k vocabulary output
(`matvec_q6_k_16row_mv[_rms]`, mirroring the q4 work with the same bitwise
per-lane order) — and **falsified**: routing the lm_head (M=262144, K=2304)
through the 16-row variants made the output projection **~14% slower**
(1.249 → 1.428 ms/token GPU, exact per-dispatch profile), so the
implementation was reverted (no code retained).  The 64-row q6 kernel stays
the lm_head path.

## Hypothesis and experiment

Phase-13.16 showed 16-row bands (4 SIMD groups × 4 rows, 128 threads,
`ceil(width/16)` grids) winning 8–30% on the decode q4 matvecs, which are all
latency/occupancy-bound at their row counts (ffn-down 1536, wo 1536, PLE
quarters, qkv 2048/256, gate/up 6144).  The q6 lm_head is the one remaining
64-row matvec, so the same band shrink was applied: new
`matvec_q6_k_16row_mv` / `..._rms` kernels preserving the exact per-lane
block stride (`ib += 2`), yl staging l0/is mapping, q6 bit extraction, and
per-simdgroup scale/d arithmetic of the 64-row family, dispatched with 128
threads and `ceil(width/16)` groups.  All decode q6 sites (the RMS-input
vocabulary output and the shared-KV non-RMS route) were routed through the
16-row binding when `gemma4_decode_16row_enabled()` (the phase-13.16 default).

Correctness was established before measurement: three new parity gates in
`matvec_mv_ext_parity.rs` (CPU-oracle tolerance for plain and RMS kernels
across input 256/2048/3584 × output 31/33/96/129/137, plus bitwise identity
vs the 64-row pair on 31/33/96/129/137/512 rows) all passed.

## Measured evidence (M2 Max, Resident, real gguf E2B fixture, pp512/tg128)

Exact per-dispatch profile A/B (records 25/26 in
`artifacts/phase-12a/gemma4-resident-decode-profile.jsonl`):

| Family | 64-row (record 25) | all-16 incl. q6 (record 26) | Δ |
|---|---:|---:|---:|
| output_projection (lm_head) | 1.249 ms/tok | **1.428 ms/tok** | **+14.4%** |
| ffn_down | 1.946 | 1.349 | −30.7% |
| attention_output | 0.838 | 0.598 | −28.7% |
| qkv | 0.983 | 0.772 | −21.5% |
| ple_projection | 2.233 | 1.965 | −12.0% |
| gate/up | 2.846 | 2.615 | −8.1% |
| per-token GPU total | 20.62 ms | 19.42 ms | −5.8% (q4 wins offset by the q6 loss) |

The 5-run bench pair matched the profile direction
(1879.8 vs 1986.5 ms mean decode, −5.4%) while the phase-13.16 all-16 pair
(which left lm_head on 64-row) measured −7.8% — the difference is the q6
regression.

## Why it fails

The band-shrink lever pays only when the 64/32-row dispatch is
occupancy/latency-bound: more, smaller threadgroups improve scheduling
granularity for small-M matvecs.  The lm_head at M=262144 already saturates
the device with 4096×256-thread groups; the 16-row variant quadruples the
group count (16384) with four times the per-group weight-base/dh loads and
smaller per-group reuse, adding scheduling and weight-read overhead without
any occupancy gain.  Design rule: **the 16-row band applies to latency-bound
decode matvecs only; throughput-bound kernels (very large M) keep their
larger bands.**

## Command book

```zsh
# The q6 16-row kernels and routing are reverted; expected lm_head behavior:
cargo test -p atlas-metal --test matvec_mv_ext_parity

# A/B bench (the q4 16-row default vs full opt-out), decode delta ≈ −5…−6%
ATLAS_GEMMA4_DECODE_16ROW=0 cargo run --release -p atlas-cli -- benchmark \
  matched --model gemma4-e2b-q4_0 --prompt-tokens 512 --decode-tokens 128 \
  --warmup-runs 1 --runs 5 --output-format json
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```

## Artifacts

- `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl` records 25/26 —
  clean 64-row vs all-16 (incl. the q6 experiment) exact-per-dispatch pair;
  the output_projection family delta is the falsification evidence.
- Reverted: `matvec_q6_k_16row_mv[_rms]` kernels, their registry entries,
  the q6 16-row routing in `matvec_labeled` / `matvec_rms_labeled`, and the
  three q6 16-row parity gates.

## Outstanding

The lm_head remains on the 64-row q6 kernel at ~1.25 ms/token (6% of decode).
With all matvec band levers now exhausted (positive for the q4 decode family,
negative for huge-M q6), the remaining decode levers are the structural ones:
(a) PLE per-layer dispatch fusion — 105 small dispatches/token (input-gate
matvec → gelu-multiply → rms → projection per layer) at ~18 µs each, worth
~1 ms/token if the two launcher-per-layer minima can be fused bitwise;
(b) attention split-KV — the flash16 decode scan (3.0 swa + 1.35 full ms)
grows linearly with context and its reduction order is currently exact-locked;
(c) batch decode (N > 1), where the per-dispatch tax and the mul_mm machinery
(phase-13.15) finally pay off.