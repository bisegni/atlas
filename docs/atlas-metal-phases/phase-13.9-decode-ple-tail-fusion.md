# Phase 13.9 — Decode PLE-tail dispatch fusion (dispatch reduction, measured wash for throughput)

## Problem

Decode launches ~547 dispatches/token (~16/layer). The per-layer PLE tail is
three elementwise dispatches over the hidden row:

```text
rms_norm_decode_f32_vec4(work, post_norm)      -> normalized
vector_add_f32(state, normalized)              -> state
scalar_multiply_f32(state, layer_output_scale) -> state
```

After phase-13.7 collapsed attention to ~2.5 ms/token, decode GPU fell to
~14.7 ms/token and host-encode came within ~1–2 ms/token of GPU, so it was
hypothesized that decode might now be host-encode-bound and that cutting
dispatches would raise throughput (re-testing the phase-13.4 D3 verdict under
the changed post-v4 cost center).

## Change

`crates/atlas-metal/src/kernels.metal` — `gemma4_ple_rms_add_scale_f32` fuses
the three PLE-tail dispatches into one. It keeps the exact
`rms_norm_decode_f32_vec4` reduction and the volatile `normalized` rounding
boundary (same pattern as `gemma4_rms_residual_f32`), then computes
`state = (state + normalized) * layer_output_scale` elementwise — bitwise
identical to the three-kernel baseline.

`crates/atlas-model/src/gemma4_executor.rs` — the decode PLE tail
(`encode_current_token`) now issues the single fused dispatch; the now-dead
`rms_norm_decode_labeled` helper is removed.

## Parity (kernel-level)

`crates/atlas-metal/tests/dispatch_fusion_parity.rs` —
`ple_rms_add_scale_matches_norm_add_then_scale` compares the fused kernel
against `rms_norm_decode_f32_vec4` → `vector_add_f32` → `scalar_multiply_f32`
at hidden 512/1280/2304 under max-abs < 1e-3. **Green.**

## Measured evidence (M2 Max, Resident, q4_0 KV, flash16 v4)

Matched `benchmark matched --model gemma4-e2b-q4_0 --prompt-tokens 512
--decode-tokens 128 --warmup-runs 1 --runs 3` (post-change) vs the pre-change
five-run baseline:

| Metric | Baseline | PLE-fused |
|---|---:|---:|
| Decode dispatches / 128 tokens | 70,104 | **61,214** |
| Dispatches / token | 547.7 | **478.2** (−69.5 = 2/layer × 35) |
| Decode tok/s | 64.9–65.8 | 64.3–64.9 (flat) |
| Decode GPU ms/128 | 1874–1884 | 1875–1881 (flat) |
| Prefill tok/s | ~318 | ~316 (unchanged, decode-only change) |
| Greedy stream hash | `f23c2962…` | `f23c2962…` (bitwise preserved) |

## Verdict

The fusion is **bitwise-correct** (greedy hash unchanged) and cuts decode
dispatch count **12.7%**, but decode throughput is **unchanged**: decode is
GPU-latency-bound (matvec family + attention), not dispatch/encode-bound. This
re-confirms the phase-13.4 D3 conclusion under the post-v4 cost center —
removing dispatches does not raise decode tok/s. The change is kept because it
is strictly correct, non-regressing, and reduces host encode work, but it is
**not** a decode-throughput lever.

## Implication for the llama.cpp gap

Decode is at parity and dispatch reduction is not its lever. The remaining gap
is **prefill** (~4–5×); the in-flight fp16 `mul_mm` (phase-13.8) measures only
~+12% prefill end-to-end, so closing the prefill gap needs the architectural
fp16 pre-dequantized-weight path (`plan-close-prefill-gap.md` option A), which
is a prefill-tolerance contract decision (~1e-2).

## Command book

```zsh
cargo test -p atlas-metal --test dispatch_fusion_parity ple_rms_add_scale -- --nocapture
cargo run --release -p atlas-cli -- benchmark matched \
  --model gemma4-e2b-q4_0 --prompt-tokens 512 --decode-tokens 128 \
  --warmup-runs 1 --runs 5 --output-format json
```
