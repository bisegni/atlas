# Phase 13.15 — Decode matvec via llama mul_mm at N=1: negative result

## Status: [done]

Lever 2 (decode 1.9×, ~64.5 vs ~122 tok/s) first hypothesis — that the decode
Q4 matvec family (qkv, ffn gate/up, ffn-down, attention-output) is
occupancy/latency-bound and would run faster through the vendored llama
matrix-unit `llama_mul_mm_q4_0_f32` — was **falsified**: routing the
single-token (N=1) matvecs through `mul_mm` made decode **2.8× slower**
(~23 tok/s vs ~64.5 tok/s), with the greedy stream hash still bitwise
preserved.  The opt-in implementation was reverted (no code retained), and the
mv_ext matvec family remains the default and correct choice for single-token
decode.

## Hypothesis and experiment

Fresh exact per-dispatch decode profile of the real `gemma-4-e2b-it-q4_0`
fixture (Resident, q4_0 KV, pp512/tg128) showed the matvec family at ~49% of
per-token GPU time (gate/up 2.884 ms, ffn-down 1.963 ms, qkv 0.953 ms, wo 0.801
ms), with the mv_ext ports running at only ~20–32% of M2 Max bandwidth — i.e.
latency/occupancy-bound, not at the weight-read floor.  `llama_mul_mm_q4_0_f32`
(NR0=64, NR1=32, NK=32, 128 threads, 8KB threadgroup memory) is
parity-tested to max_rel < 5e-2 including the small-N guarded-store path (the
`llama_mul_mm_parity.rs` contract) and already drives prefill via
`ATLAS_GEMMA4_LLAMA_MUL_MM`.  The experiment therefore routed the decode
matvecs through it at N=1, behind a temporary opt-in env.

Implementation (temporary, opt-in `ATLAS_GEMMA4_DECODE_MUL_MM`, since
removed): per token, `rms_norm_decode_f32_vec4` (32 threads, 1 threadgroup,
reusing the `attention`/`q` scratch buffers — free at those sites) followed by
`mul_mm` dispatches — qkv: q M=q_width (2048/4096), k/v M=key_length (256/512);
wo and shared-KV q-only as single `mul_mm`; gate/up: M=6144/
layer-ffn-width each; ffn-down: K=6144, M=1536.  Grid
(1, M.div_ceil(64)), 128 threads, 8192 B threadgroup memory; ne00 via the
existing `hidden`/`ffn_widths[layer]` u32 buffers, ne1 via a dedicated u32(1)
buffer.

## Measured evidence (M2 Max, Resident, real gguf E2B-it fixture, pp512/tg128, warmup1+5)

| Decode path | Decode ms (5-run) | Decode tok/s ≈ | Stream hash |
|---|---:|---:|---|
| default mv_ext matvec family | 1982.3–2008.3 (median ~1989) | 63.7–64.6 | `f23c2962…` exact |
| **mul_mm at N=1 (opt-in)** | **5575.2–5595.7 (median ~5578)** | **~22.9** | `f23c2962…` exact |

- Prefill unchanged by the decode routing (~340–343 ms, ~1492–1506 tok/s).
- Exact per-dispatch profile of the opt-in path (appended record in
  `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl`): per-token GPU
  **50.75 ms vs 20.82 ms** baseline; 617 dispatches/token vs 478.
- Per-dispatch latency, mul_mm vs mv_ext baseline:
  - ffn-down: 263–286 µs (swa layers, K=1536·smaller ffn) and
    519–642 µs (full-attention layers, K=6144) vs ~56 µs — up to 11× per
    dispatch.
  - ffn gate/up: 114–124 µs (swa) / 187–215 µs (full) vs ~41 µs.
  - qkv projections: 57–103 µs vs ~38 µs; attention-output (wo): 91–601 µs vs
    ~62 µs.
  - qkv-input/ffn-input RMS (new dispatches): 16–17 µs each.

## Conclusion

At N=1, `mul_mm` collapses its 32-row threadgroup reuse to a single output row
yet still pays the full 8 KB threadgroup-memory K-block staging, two
synchronizations per 32-column chunk, simdgroup fragment load/store latency,
and full K re-read by every threadgroup (M.div_ceil(64) threadgroups each read
the entire K weight).  The mv_ext ports — one output row per SIMD-group,
strided reads, no threadgroup staging, no matrix unit — are the right shape
for single-token decode; `mul_mm`'s advantage only materializes at real batch
(N ≥ 16–32), consistent with its +113% prefill result (phase-13.10).  The
negative also confirms decode remains latency-bound as phase-13.4/13.9
described; the observed ~20–32%-of-bandwidth matvecs are not a fixable
occupancy artifact of the matvec shape but the floor for single-token GEMMs.
(The "not a fixable occupancy artifact" claim covers the *matrix-unit* shape
specifically; the *band geometry* of the mv_ext family was still fixable — see
phase-13.16, which shrank the threadgroup band to 16 rows for ~8% decode with
bitwise parity.)

Secondary finding: the standalone-vec4-RMS + mul_mm stack preserved the exact
greedy stream hash `f23c2962…`, so the RMS-fusion reordering concern was not
an additional blocker (tolerance-level as predicted).

## Command book

```zsh
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```

Baseline expected: decode ~1990 ms ~64 tok/s, hash `f23c2962…`.  The opt-in
flag `ATLAS_GEMMA4_DECODE_MUL_MM` no longer exists (reverted); the expected
re-run signal for this comparison would be decode ~5580 ms ~23 tok/s.

## Artifacts

- `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl` — record 20
  (exact-per-dispatch profile of the mul_mm-at-N=1 path); record 19 is the
  fresh mv_ext baseline profile used for the family table above.

## Outstanding

Lever 2 (decode 1.9×) remains open.  With the matvec-occupancy hypothesis
falsified, the remaining decode levers are the ones already priced out in
phase-13.0/13.9: (a) attention — largest single family at ~21.5% of per-token
GPU (3.085 ms swa + 1.386 ms full + 2.385 ms qk_norm_rope_fused), with the
qk_norm_rope→attention fusion blocked by the quantized-cache dependency (P2d);
(b) PLE projection (2.230 ms, 106 dispatches) and lm_head q6_k (1.287 ms),
neither of which has a `mul_mm` q6 path; (c) batch decode (N > 1) where this
phase's negative shows `mul_mm` would finally pay off.