# Phase 13.14 — Matrix-unit prefill attention (flash16-v6): implemented, no win

## Status: [done]

Levers 1 was implemented as flash16-v6 and measured **equal to v5, not faster**;
the acceptance result is a recorded negative result with v5 retained as the
dispatched prefill attention kernel of record.  v6 stays in the tree as a
registered kernel with tolerance parity coverage for future revisits.

## Change

- `crates/atlas-metal/src/kernels.metal` — `DEFINE_FLASH_ATTENTION_V6_BATCH`
  (one threadgroup per 8-token tile, one SIMD group per head, requires
  `heads == 8 && kv_heads == 1`, Gemma 4 E2B geometry): K/V q4_0 dequant into
  f16 threadgroup tiles shared across heads, `S = Q·Kᵀ` and `O += P·V` as
  `simdgroup_matrix` multiplies, two-pass softmax (guarded per-row stats
  merge, then normalized `P = exp(S − M)/D`); per-token causal/swa masking via
  the `key_control` table, tail-tile and zero-window handling.
- `crates/atlas-metal/src/lib.rs` — registered
  `attention_prefill_gemma4_simd_q4_0_flash16_[swa_]v6`.
- `crates/atlas-metal/tests/flash16_v6_parity.rs` — v6 vs v5 tolerance parity
  (fp16 query path incl. `gemma4_cast_f32_to_f16_batch`), covering uniform,
  causal, sliding-window with `key_start > 0`, tail (batch % 8 != 0) and
  zero-count windows.
- `crates/atlas-model/src/gemma4_executor.rs` — during development the
  flash-prefill arm dispatched v6 (plus a per-layer fp16 query cast buffer);
  reverted to v5 after v6 measured no speedup.  Net diff is comment-only.

## Parity

`flash16_v6_*` (7 tests) all green: max_abs ≤ 1.4e-4, max_rel ≤ 1.4e-4 — the
fp16 matrix-unit path matches the shared-head v5 path well inside the 1e-2
tolerance.  End-to-end greedy hash stays `f23c2962…`.

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+5)

| Prefill path | Prefill tok/s (median) | Prefill ms (median) |
|---|---:|---:|
| phase-13.13 (v5) | ~1356 | ~378 |
| **+ flash16-v6 matrix-unit** | ~1359 | ~376 |
| v5 default (final state) | ~1367 | ~374.5 |

Both kernels measure the same prefill within noise and produce the identical
greedy hash `f23c2962…`.  Exact-per-dispatch diagnostic profile (v6): full-head
attention 4.6–6.6 ms/layer, swa 1.8–2.7 ms/layer, total attention ~87 ms vs
llama.cpp ~40 ms; total prefill GPU 415.7 ms in diagnostic mode (production
~376 ms).

**Conclusion.** The phase-13.14 hypothesis — v5's serial per-key chain is the
attention ceiling and matrix units will halve it — is falsified on the current
geometries.  v6 and v5 dequantize the same q4_0 K/V bytes once per key; v6's
measured time is essentially v5's, i.e. the kernel is bound by q4_0 dequant /
threadgroup-memory bandwidth and tile latency, not by accumulating the dot
product.  llama.cpp's ~2× over both kernels comes from single-pass online
softmax (no second dequant+score pass) and different threads-per-tile.

## Command book

```zsh
cargo test -p atlas-metal --test flash16_v6_parity -- --nocapture
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```

Expected pass signal: parity max_rel < 1e-2; matched benchmark prefill
~1350–1390 tok/s with stream hash `f23c2962…`.

## Artifacts

- `artifacts/phase-13.14/matched-benchmark-pp512-tg128-v5-default.jsonl` —
  final-state (v5 dispatched) 5-run matched benchmark.
- `artifacts/phase-13.14/flash16-v6-profile-pp512.jsonl` — exact-per-dispatch
  v6 profile (prefill_kernels gpu_ms per layer).

## Outstanding

Lever 2 (decode 1.9×) remains open and unchanged by this work.  Within Lever 1,
a v7 single-pass variant (online rescaling softmax, no second pass) is the
plausible next attempt against the dequant-bandwidth ceiling, plus a larger
query tile (e.g. 16 tokens/threadgroup, Q16×K16) to amortize barrier cost.