# Phase 13.3 — Staged, chunked decode attention KV scan (D2)

## Problem (from `atlas-vs-llama-gap-analysis.md` D2)

Decode attention is the largest single decode cost and is O(context): the
resident q4 attention kernels scan keys serially with a per-key
`threadgroup_barrier` pair and an online-softmax value update that round-trips
the output through device memory every key. On the matched workload decode
drops 41→19→16 tok/s as the prompt grows 100→512→1024.

Phase-13.0 P3 tried to fix this with key-range-split slices plus a dim-half
split of the value accumulator and was blocked: each slice folds only its own
key range's online-softmax weights into the V dims that slice covers, so a
dominant top-score key in one slice's range never reaches the other half of the
output (observed output collapsed to ~1e-6 uniform values). Every correct
alternative failed within the 32 KiB threadgroup budget, and the merged-slice
`flash16_uw` kernels changed FP32 rounding enough to drift the greedy stream
(they are retired for parity reasons).

## Change

Keep the exact FP32 arithmetic and eliminate the per-key barriers with a
single, chunked, three-pass kernel:

`crates/atlas-metal/src/kernels.metal` — `DEFINE_FLASH16_EXACT_V3` defines

- `attention_decode_gemma4_simd_q4_0_flash16_exact_v3` (full, 512-wide heads)
- `attention_decode_gemma4_simd_q4_0_flash16_swa_exact_v3` (swa, 256-wide)

Each threadgroup (one per head) scans keys in 128-key chunks with three
barrier-separated passes:

1. **Pass A** (all threads, no barriers): per key, compute the identical
   per-thread partial (`d = t128; d < head_dim; d += 128` within each 128-thread
   score group) and `simd_sum`, storing lane-0 results into
   `threadgroup partials[chunk][4]`. `threads / 128` keys are computed in
   parallel per iteration (the query is cached in registers), so the wide
   threadgroups keep several independent key chains in flight to hide KV memory
   latency.
2. **Pass B** (thread 0, no barriers): fold `p0 + p1 + p2 + p3`, run the exact
   key-ordered online softmax (single running maximum/denominator across the
   whole scan, no slice merging), and store `rescale`/`weight` per key.
3. **Pass C** (all threads, no barriers): apply the register-resident value
   chain `out_d = out_d * rescale + weight * value` per key, spread over the
   full threadgroup width, writing device output once at the end (divided by
   the final denominator).

The kernel is byte-identical to LegacyFused/`_nb`: same per-thread dims, same
`simd_sum`, same fold order, same key-ordered softmax exps, same per-dim value
chain. The value accumulator stays in registers instead of round-tripping
device memory per key, and the ~2 barriers per key drop to ~3 per 128-key
chunk. `head_dim` stays runtime so Metal cannot unroll or reassociate the
partial dot products (the phase-13.0 `_exact` constraint). Threads must be a
positive multiple of 128; the Resident dispatch uses 512 (full) / 256 (swa)
threads per head.

`crates/atlas-model/src/gemma4_executor.rs`:

- `Gemma4Q4AttentionMode::default()` stays `Flash16`, which now selects the v3
  binding (`gemma4_q4_flash16_v3_binding`, 512 full / 256 swa threads). The
  `_nb` no-value-barrier kernels remain selectable via the new `flash16_exact`
  diagnostic mode; `legacy_fused` remains the explicit reference path.
- `gemma4_attention_kernel` and the decode dispatch select the kernel by mode.

`crates/atlas-metal/src/lib.rs`: both v3 kernels are registered in the pipeline
list (pipeline count 78 → 80; the `atlas-ops` pipeline-count assertion updated).

## Correctness gates

- `crates/atlas-metal/tests/attention_flash_correctness.rs`
  `flash16_exact_variants_match_legacy_fused_bitwise` now includes both v3
  kernels at the 128-thread launch and the wide launches (512 full / 256 swa):
  bitwise-identical to LegacyFused for full/swa head widths across key counts
  48–2048 (full) and 48–256 (swa), rising and non-rising cache.
- `flash16_matches_legacy_resident_output_logit_digests` (ignored, release,
  M2 Max): per-token fp32 logit SHA-256 digests are byte-identical between the
  default `Flash16` (v3) path and `LegacyFused` for the canonical chat, the C++
  chat, and the 256+64 long decode window.
- `q4_kv_flash16_matches_legacy_resident_attention_across_chat_and_long_decode`
  (ignored, release): token/finish parity plus `attention_kernel` now reported
  as `attention_decode_gemma4_simd_q4_0_flash16_exact_v3`.
- `cargo test --workspace` green; `cargo fmt --check` clean.

## Performance evidence

Artifacts under `artifacts/phase-13.3/` (Resident layer-major prefill, q4_0 KV,
Flash16 (`v3`, wide) attention):

`cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0
--prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json`

| Metric | Baseline (phase-13.2, `_nb`) | After (v3, 128t) | After (v3 wide) | Delta (vs 13.2) |
|---|---:|---:|---:|---:|
| Decode GPU ms/128 tokens (median) | 5900.5 | 5150.2 | 4034.6 | −31.6% |
| Decode tok/s (median) | ~21.4 | ~24.0 | ~31.0 | +45% |
| Decode dispatches/128 tokens | 70,104 | 70,104 | 70,104 | unchanged |
| `generated_token_sha256` | `f23c2962…` | `f23c2962…` | `f23c29623e1d2980be0630e6b12db047…` | byte-identical |

Context sweep (1 warmup + 3 measured, v3 wide default):

| Prompt | Decode GPU ms/128 | Decode tok/s | phase-13.1 tok/s |
|---:|---:|---:|---:|
| 100 | 2300.3 | 53.6 | 41.3 |
| 512 | 4034.6 | 31.0 | 19.4 |
| 1024 | 4574.5 | 27.4 | 16.3 |

The wide threadgroups are the second stage of D2: the staged scan at the
original 128-thread launch already cut decode GPU 5900.5 → 5150.2 ms/128, and
widening to 512 (full) / 256 (swa) threads per head cut it again to 4034.6 ms
as the parallel key chains hide KV memory latency. Every run reports
`q4_attention_mode: flash16` and
`attention_kernel: attention_decode_gemma4_simd_q4_0_flash16_exact_v3`, the
greedy stream hash stays byte-identical to the recorded baseline at every
context, and prefill is unchanged (~194 tok/s). The stream hash is the drift
sentinel: v3 carries zero parity risk.

## Acceptance gates (phase 13.3)

All met on Apple Silicon (M2 Max, Gemma 4 E2B q4_0 fixture):

- v3 bitwise kernel parity vs LegacyFused (atlas-metal test, full + swa, 128
  and wide threadgroups);
- per-token fp32 logit-digest parity between the v3 default and LegacyFused
  (canonical, C++ chat, long-window);
- exact-token stream parity (chat + long decode);
- decode GPU −31.6% at matched pp512/tg128 with a byte-identical greedy stream
  hash; decode no longer collapses with context (53.6→31.0→27.4 tok/s at
  pp100→512→1024);
- evidence recorded under `artifacts/phase-13.3/`.

## Command book

```zsh
cargo test -p atlas-metal --test attention_flash_correctness
cargo test --release -p atlas-model --test phase_12a_gemma4_resident \
  flash16_matches_legacy_resident_output_logit_digests -- --ignored
cargo test --release -p atlas-model --test phase_12a_gemma4_resident \
  q4_kv_flash16_matches_legacy_resident_attention_across_chat_and_long_decode -- --ignored
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json
```

The `_nb` path remains reachable for A/B and diagnostics with
`--q4-attention-mode flash16_exact`.
