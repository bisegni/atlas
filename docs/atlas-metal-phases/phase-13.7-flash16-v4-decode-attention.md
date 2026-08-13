# Phase 13.7 — Flash16 v4 merged-slice decode attention (Path B)

## Problem

Decode attention is ≈60% of decode GPU and is the main barrier to llama.cpp
parity. The exact-ordered v3 scan (byte-identical greedy stream) costs ~19
ms/token at pp512 because the per-dim value chain is serial and the online
softmax keeps a single running state. llama.cpp's non-exact flash attention is
~5–15× faster, but the phase's byte-identical stream sentinel
(`f23c2962…`) forbade adopting it.

## Change (Path B — tolerance-level parity, approved 2026-08-13)

`crates/atlas-metal/src/kernels.metal` — `DEFINE_FLASH_ATTENTION_V4` defines

- `attention_decode_gemma4_simd_q4_0_flash16_v4` (full, 512-wide, 12 slices)
- `attention_decode_gemma4_simd_q4_0_flash16_swa_v4` (swa, 256-wide, 24 slices)

Each head is one threadgroup of `SLICES × 32` threads. Each SIMD group scans a
disjoint slice of the key range with **register** online-softmax state and
value accumulators — no per-key threadgroup barriers — then the slice states
are merged in threadgroup memory (running max, rescale, weighted value sum) in
one pass. Unlike the retired `_uw` kernels, v4 reads the packed `key_control`
(`key_start << 16 | key_count`), so sliding-window heads scan the correct
absolute key range.

This is **not bitwise**: the slice split and merge change the FP32 reduction
order. Correctness is covered by the kernel-level max-abs < 1e-3 tolerance
contract (`flash16_uw_matches_cpu_oracle` now includes both v4 kernels); the
greedy stream hash is a drift diagnostic. In practice the matched-benchmark
stream still hashes to `f23c2962…` at pp100/512/1024, but the C++ chat prompt
diverges from LegacyFused at generated token 50 — the expected consequence of
the Path B decision.

`crates/atlas-model/src/gemma4_executor.rs` — `Flash16` (the production
default) binds the v4 kernels (384 full / 768 swa threads per head). The exact
path stays reachable as `flash16_exact` (`_nb`) and `legacy_fused`, and the
model-level byte-parity gates now exercise `Flash16Exact` vs `LegacyFused` to
preserve the exact-order proof for the exact reference path.

## Performance evidence

Artifacts under `artifacts/phase-13.7/` (Resident, q4_0 KV, v4 default,
five-run each):

| Prompt | Decode tok/s (was) | Decode GPU ms/128 (was) | Prefill tok/s | Stream hash |
|---:|---:|---:|---:|---:|
| 100 | **74.1** (52.2) | 1652 | 326.8 | `f23c2962…` |
| 512 | **64.9** (30.5) | 1895 (4034) | 311.2 | `f23c2962…` |
| 1024 | **61.3** (27.3) | 2010 | 288.5 | `f23c2962…` |

Decode GPU more than halved at pp512 (4034 → 1895 ms/128 tokens); attention
dropped from ~19 to ~2.5 ms/token (≈8×). Decode is now ~0.8–0.9× llama.cpp's
empty-context `tg` decode (the honest gap is smaller still because llama-bench
decodes from empty context). Prefill is unchanged.

## Acceptance gates (phase 13.7)

- v4 kernels match the CPU oracle within max-abs < 1e-3 (full 48–2048 keys,
  swa 48–256);
- `flash16_exact_matches_legacy…` and the exact token-parity gates pass for the
  `Flash16Exact` path (byte-identical exact reference preserved);
- `cargo test --workspace` green; `cargo fmt --check` clean;
- decode +112% at matched pp512/tg128 (30.5 → 64.9 tok/s);
- evidence recorded under `artifacts/phase-13.7/`.

## Command book

```zsh
cargo test -p atlas-metal --test attention_flash_correctness
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json
```
