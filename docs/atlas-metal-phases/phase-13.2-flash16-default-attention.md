# Phase 13.2 — Default q4 attention to the no-value-barrier flash16 variant (D1)

## Problem (from `atlas-vs-llama-gap-analysis.md` D1)

The production Resident q4 attention default (`LegacyFused`) and the
`_flash16_exact_runtime` binding both hold a `threadgroup_barrier` after every
key's value update. That barrier is redundant: each thread writes the same
`d`-strided positions of `output` every iteration and the score/softmax state
is already guarded by the two per-key barriers around the `tid == 0` reduction.
Removing it is arithmetic-invisible, and it costs ~8% of decode GPU time.

## Change

`crates/atlas-model/src/gemma4_executor.rs`:

- `Gemma4Q4AttentionMode::default()` now returns `Flash16` (was
  `LegacyFused`). `LegacyFused` remains reachable via
  `--q4-attention-mode legacy_fused` as the explicit diagnostic path.
- `gemma4_q4_flash16_binding` now points at the no-value-barrier kernels
  `attention_decode_gemma4_simd_q4_0_flash16_exact_nb` /
  `_swa_exact_nb` (was `_exact_runtime` / `_swa_exact_runtime`). The kernels
  were already defined (`crates/atlas-metal/src/kernels.metal`) and registered
  (`crates/atlas-metal/src/lib.rs`); no Metal changes.

The `_nb` variant keeps the same four-SIMD score reduction, key-ordered online
softmax, and per-token fp32 logit arithmetic as `LegacyFused`; it drops only
the redundant per-key value barrier.

## Correctness gates

- `flash16_matches_legacy_resident_output_logit_digests` (ignored, release,
  M2 Max): per-token fp32 logit SHA-256 digests are byte-identical between the
  default `Flash16` (`_nb`) path and `LegacyFused` for the canonical chat, the
  C++ chat, and the 256+64 long decode window. This test now exercises `_nb`
  end-to-end through the default binding.
- `q4_kv_flash16_matches_legacy_resident_attention_across_chat_and_long_decode`
  (ignored, release): token/finish parity plus `attention_kernel` now reported
  as `attention_decode_gemma4_simd_q4_0_flash16_exact_nb`.
- `crates/atlas-metal/tests/attention_flash_correctness.rs`
  `flash16_exact_variants_match_legacy_fused_bitwise`: `_nb` matches
  `LegacyFused` bitwise for full/swa head widths across key counts 48–2048.
- `cargo test --workspace` green; `cargo fmt --check` clean.
- Pre-existing environment note: `resident_canonical_chat_matches_pinned_tokens…`
  fails identically on clean `main` in this environment (fixture captured on
  other hardware); the `_nb` build produces the same token stream as main
  (verified via the benchmark stream hash below), i.e. no regression.

## Performance evidence

Five-run gate artifact:
`artifacts/phase-13.2/matched-benchmark-pp512-tg128.jsonl` (1 warmup + 5
measured runs at pp=512, tg=128, q4_0 KV, Flash16 (`_nb`) attention, Resident
layer-major prefill).

`cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0
--prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json`

| Metric | Baseline (phase-13.1, LegacyFused) | After (`_nb`) | Delta |
|---|---:|---:|---:|
| Decode GPU ms/128 tokens (median) | 6455.1 | 5900.5 | −8.6% |
| Decode tok/s (median) | ~19.7 | ~21.4 | +8.6% |
| Decode dispatches/128 tokens | 70,104 | 70,104 | unchanged |
| `generated_token_sha256` | `f23c2962…` | `f23c29623e1d2980be0630e6b12db047…` | byte-identical |

The `_nb` swap is dispatch-count-neutral (70,104 dispatches for 128 tokens
unchanged); the win is the removed per-key barrier inside the attention kernel.
Every run reports `q4_attention_mode: flash16` and
`attention_kernel: attention_decode_gemma4_simd_q4_0_flash16_exact_nb`, and
the greedy stream hash stays byte-identical to the recorded baseline, so D1
carries zero drift risk. Prefill (~192 tok/s) is unchanged (kernels untouched).

## Acceptance gates (phase 13.2)

All met on Apple Silicon (M2 Max, Gemma 4 E2B q4_0 fixture):

- per-token fp32 logit-digest parity between the `_nb` default and LegacyFused
  (canonical, C++ chat, long-window);
- exact-token stream parity (chat + long decode);
- `_nb` bitwise kernel parity vs LegacyFused (atlas-metal test);
- default path reports the `_nb` kernel in generation metrics;
- decode GPU improves ~8% at matched pp512/tg128 with a byte-identical greedy
  stream hash;
- evidence recorded under `artifacts/phase-13.2/`.

## Command book

```zsh
cargo test -p atlas-metal --test attention_flash_correctness
cargo test --release -p atlas-model --test phase_12a_gemma4_resident \
  flash16_matches_legacy_resident_output_logit_digests \
  q4_kv_flash16_matches_legacy_resident_attention_across_chat_and_long_decode \
  -- --ignored
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json
```
