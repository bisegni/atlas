# Phase 13.14 — Matrix-unit prefill attention: v6 no-win, v7 single-pass default

## Status: [done]

Lever 1 was closed with **flash16-v7**, a single-pass matrix-unit prefill
attention kernel (online softmax rescaling, half the K/V dequant traffic of
v5/v6), measured **~7% faster than v5** and made the default dispatched prefill
attention kernel of record.  The earlier intermediate result — flash16-v6 (a
two-pass matrix-unit variant) measured equal to v5, not faster — is retained as
a recorded negative result.

## Change

- `crates/atlas-metal/src/kernels.metal` — `DEFINE_FLASH_ATTENTION_V7_BATCH`
  (one threadgroup per 8-token tile, one SIMD group per head, requires
  `heads == 8 && kv_heads == 1`, Gemma 4 E2B geometry): K/V q4_0 dequant into
  f16 threadgroup tiles shared across heads, `S = Q·Kᵀ` and `O += P·V` as
  `simdgroup_matrix` multiplies, **single-pass** online softmax: each key block
  merged into the running row max/denominator and any O-fragment rescaled when
  the row max moves, unnormalized `P = exp(S − M)` with `O += P·V`, and a single
  final denominator normalization — so each key is visited once and K/V are
  dequantized once, halving the dequant bandwidth of v5/v6.  Per-token
  causal/swa masking via the `key_control` table, tail-tile and zero-window
  handling.
- `crates/atlas-metal/src/lib.rs` — registered
  `attention_prefill_gemma4_simd_q4_0_flash16_[swa_]v7`.
- `crates/atlas-metal/tests/flash16_v7_parity.rs` — v7 vs v5 tolerance parity
  (fp16 query path incl. `gemma4_cast_f32_to_f16_batch`), covering uniform,
  causal, sliding-window with `key_start > 0`, tail (batch % 8 != 0) and
  zero-count windows.
- `crates/atlas-model/src/gemma4_executor.rs` — the flash-prefill q4_0 KV arm
  dispatches **v7 by default** (per-layer fp16 query cast buffer).  v5 remains
  available via `ATLAS_GEMMA4_FLASH_PREFILL_V5=1`.
- `crates/atlas-ops/tests/phase_02_operators.rs` — pipeline-count assertion
  96 → 98 for the two new registered kernels.

## Parity

`flash16_v7_*` (7 tests) all green: v7 matches the v5 shared-head path well
inside the 1e-2 relative tolerance across uniform, causal, swa-window, tail and
zero-count cases.  End-to-end greedy stream hash stays `f23c2962…`.

## Measured evidence (M2 Max, Resident, q4_0 KV, pp512/tg128, warmup1+5)

| Prefill path | Prefill ms (5-run) | Prefill tok/s ≈ |
|---|---:|---:|
| phase-13.13 (v5) | ~374–378 | ~1367 |
| + flash16-v6 matrix-unit (two-pass) | ~376 | ~1359 — no win, reverted |
| **+ flash16-v7 single-pass (default)** | **~345–349** | **~1475** |
| v5 opt-in (`ATLAS_GEMMA4_FLASH_PREFILL_V5=1`) | ~370–374 | ~1370 |

v7 default: prefill 344.5–349.4 ms (median ~345.8) vs v5 opt-in 369.8–374.1 ms
(median ~370.6) — a **~7% prefill improvement** — with the identical greedy
hash `f23c2962…`.  Decode is unchanged (~2000 ms) by this work.

**Conclusion.**  The phase-13.14 hypothesis that the prefill attention ceiling
was accumulating the dot product, or the matrix-unit S/O phases, was falsified
by v6 (equal to v5); the binding cost is q4_0 K/V dequant / threadgroup-memory
bandwidth, and halving that traffic (single-pass online softmax, each key/K/V
visited once) is what moves the needle (~+7%).  v7 is now the default prefill
attention kernel.

Unverified the exact hash above.  A subsequent run on the **real
`gemma-4-e2b-it-q4_0` gguf fixture** (Resident, q4_0 KV, pp512/tg128,
warmup1+5) confirmed the same result: prefill 335.0–342.2 ms (median ~337),
~1517–1528 tok/s; decode ~1990 ms (~64 tok/s); stream hash `f23c2962…`
preserved.

## Command book

```zsh
cargo test -p atlas-metal --test flash16_v7_parity -- --nocapture
ATLAS_GEMMA4_LLAMA_MUL_MM=1 ATLAS_GEMMA4_FLASH_PREFILL=1 cargo run --release \
  -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 \
  --output-format json
```

Expected pass signal: parity max_rel < 1e-2; matched benchmark prefill
~1440–1500 tok/s with stream hash `f23c2962…`.

## Artifacts

- `artifacts/phase-13.14/matched-benchmark-pp512-tg128-v5-default.jsonl` —
  v5-dispatched 5-run matched benchmark (368–377 ms).
- `artifacts/phase-13.14/matched-benchmark-pp512-tg128-v7.jsonl` —
  opt-in v7 5-run matched benchmark (341–353 ms).
- `artifacts/phase-13.14/matched-benchmark-pp512-tg128-v7-default.jsonl` —
  v7-as-default 5-run matched benchmark (344–349 ms, hash `f23c2962…`).
- `artifacts/phase-13.14/matched-benchmark-pp512-tg128-v7-gguf-e2b-it.jsonl` —
  v7-as-default 5-run matched benchmark on the real gguf E2B-it fixture
  (335–342 ms, ~1517–1528 tok/s, hash `f23c2962…`).
- `artifacts/phase-13.14/flash16-v6-profile-pp512.jsonl` — exact-per-dispatch
  v6 profile (negative-result evidence).

## Outstanding

Lever 2 (decode 1.9×) remains open and unchanged by this work.  Within Lever 1,
a larger query tile (e.g. 16 tokens/threadgroup, Q16×K16) to amortize barrier
cost is the plausible further attempt against the dequant-bandwidth ceiling.
