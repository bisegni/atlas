# Phase 13.1 — Token-batched prefill kernels (R1)

## Problem (from `atlas-vs-llama-gap-analysis.md` R1)

Gemma4 layer-major prefill matmuls were already token-batched, but four
kernels were dispatched inside `for token in 0..batch_value` loops in
`Gemma4E2bExecutor::encode_prefill_layer_major_layer` (qk-norm/rope,
provider-V RMS, KV-append, attention) plus two PLE loops (stable norm,
offset multiply). That was ~164 dispatches/token, so prefill scaled as
(prompt tokens × dispatch cost) instead of (weights read once + batched
attention), producing the 12×–45× prefill gap vs llama.cpp.

## Change

Each per-token loop became a single token-batched dispatch. New kernels in
`crates/atlas-metal/src/kernels.metal` (per-token decode kernels untouched):

- `gemma4_qk_norm_rope_fused_batch_f32` — grid `batch × (q_heads + has_key)`,
  one threadgroup per (token, head); each threadgroup derives its own buffer
  offsets (q stride `q_heads × head_dim`, K stride `head_dim`, rope stride
  `rope_pairs`).
- `kv_append_decode_{f32,q8_0,q4_0}_batch` — grid `batch × blocks`, absolute
  position read from the contiguous positions table.
- `attention_decode_fused_gemma4_simd_{f32,q8_0,q4_0}_batch` — grid
  `batch × heads`, key control read from the per-token table at the layer
  byte offset with stride `layers`; identical four-SIMD reduction and
  key-ordered online softmax.
- `vector_multiply_offset_batch_f32` — one dispatch per layer over the token
  chunk for the PLE composition.
- `rms_norm_groups_in_place_unweighted_f32` / `_stable_f32` reused unchanged,
  now dispatched once per chunk with `groups = batch` / `batch × layers`.

Prefill batch cap raised from 128 to 512 (`GEMMA4_PREFILL_BATCH_CAPACITY` and
`Gemma4PrefillPlan::new`), so a pp512 prompt is one chunk, matching llama.cpp
`n_ubatch=512`. Resident prefill scratch grows ~4× (≈ +156 MB on the E2B
geometry). Added `zero`/`rope_pairs` constant buffers; `has_key` is now the
true per-layer value (zero buffer for shared-KV layers) so the batched
`group / total` mapping is exact.

## Correctness gates

- New `crates/atlas-metal/tests/prefill_batch_parity.rs`: 10 bitwise-equality
  tests comparing each batched kernel against the per-token dispatch loop it
  replaces (qk/rope, v-norm, KV-append ×3, attention ×3, PLE stable norm,
  PLE offset multiply). Passed on Apple Silicon.
- Matched benchmark stream hash is byte-identical to the recorded baseline:
  `generated_token_sha256 f23c2962…` at both pp100 and pp512, proving the
  batched prefill preserves the greedy stream exactly.
- `phase_12a_larger_q4_128_token_resident_throughput_gate` and
  `phase_08c_resident_production_prefill_matches_reference_before_decode`
  pass. `cargo test --workspace` green; `cargo fmt --check` clean.
- Pre-existing environment note: `resident_canonical_chat_matches_pinned_tokens…`
  fails identically on clean `main` in this environment (fixture captured on
  other hardware); the batched build produces the same token stream as main,
  i.e. no regression.

## Performance evidence

Five-run gate artifact:
`artifacts/phase-13.1/matched-benchmark-pp512-tg128.jsonl` (1 warmup + 5
measured runs at pp=512, tg=128, q4_0 KV, LegacyFused attention, Resident
layer-major prefill).

`cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0
--prompt-tokens N --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json`

| Prompt | Baseline prefill tok/s | After | Ratio | prefill_chunk_size |
|---:|---:|---:|---:|---:|
| 100 | 95.1 | 199.2 | 2.1× | 100 (1 chunk) |
| 512 | 49.6 | 199.8 (five-run median 199.7–200.0) | 4.0× | 512 (1 chunk) |

The five pp512 runs are stable (199.5–200.0 tok/s) and every run's
`generated_token_sha256` is `f23c2962…` — byte-identical to the recorded
baseline, so the batched prefill preserves the greedy stream exactly. Prefill
is now roughly flat (~190–200 tok/s) across prompt length instead of dropping
95 → 50 → 36. Decode is untouched (measured decode dispatches 70,104 for 128
tokens, unchanged). Prefill dispatches/token drop from ~164 to ~1–2 (six
batched kernels per layer per chunk).

## Acceptance gates (phase 13.1)

All met on Apple Silicon (M2 Max, Gemma 4 E2B q4_0 fixture):

- 10/10 batched-kernel bitwise parity tests green;
- matched benchmark stream hash equals the recorded baseline (greedy parity);
- pp512 prefill ≥ 120 tok/s with a single 512-token chunk (measured 199.8);
- no decode regression (throughput gate green, decode kernels untouched);
- evidence recorded under `artifacts/phase-13.1/`.

## Command book

```zsh
cargo test -p atlas-metal --test prefill_batch_parity
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json
```
