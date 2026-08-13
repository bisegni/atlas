# Phase 13.6 — Weight-stationary prefill q4 batched GEMM

## Problem (the 8× prefill gap)

Prefill is ~190 tok/s vs llama.cpp's ~1150–1670 tok/s (8×), the largest
remaining gap (2.4s at pp512 / 5.0s at pp1024). Post-R1 the prefill is
GEMM-bound: `matmul_q4_0_batch_16row` (`kernels.metal`) is token-major, so each
weight row is read once **per token** — a pp512 prompt reads the whole weight
matrix 512 times, and the kernel's single-accumulator lane loop leaves load
latency un-hidden.

## Change

`crates/atlas-metal/src/kernels.metal` — `matmul_q4_0_batch_32row` is a
weight-stationary batch tile: each threadgroup computes 4 tokens × 32 output
rows (256 threads, `GEMMA4_BATCH_TILE_TOKENS = 4`), reading its shared weight
block once and reusing it in registers across the four tokens, then applying
the per-token `shuffle_xor(4,2,1)` butterfly. The weight traffic per layer
drops ~4× and the interleaved accumulator chains give the lanes real ILP. The
out-of-range token pointers are clamped so reads stay in bounds; per-token
write guards discard the clamped results.

`crates/atlas-model/src/gemma4_executor.rs` — `gemma4_q4_batch_projection_kernel`
now returns `matmul_q4_0_batch_32row`; `matmul_batch` and
`matmul_ffn_down_batch` dispatch the tiled grid
(`ceil(batch/4) × ceil(output_width/32)`, 256 threads).

## Parity contract (user-approved deviation)

Each token's dot product still accumulates block-sequentially with the identical
`in * q * scale` expression and butterfly, so the per-token FP32 order is
preserved. The interleaved four-accumulator structure changes Metal's
instruction selection by ~1 ulp (measured max-abs ≈ 4.7e-10 vs the reference),
which is **tolerance-level, not bitwise** — well inside the existing
`batch_matmul_parity` phase contract (max-abs < 1e-3). The alternative
bitwise-exact token-major tile (single-accumulator loops, weight reuse via L1)
measured only ~+5% prefill. Per the recorded decision, the fast interleaved
tile is shipped; the greedy stream hash was verified byte-identical
(`f23c2962…`) at pp100/512/1024 and the digest/exact-token gates pass, but a
~1-ulp logit drift could in principle flip a greedy token on a different prompt.
`batch_matmul_parity.rs` covers the tile at the tolerance contract
(`batch_32row_matches_batch_16row_within_tolerance`) plus the CPU-oracle test.

## Performance evidence

Artifacts under `artifacts/phase-13.6/` (Resident layer-major prefill, q4_0 KV,
Flash16 v3 decode, five-run each):

| Prompt | Prefill tok/s | Prefill ms | phase-13.3 tok/s | Decode tok/s | Stream hash |
|---:|---:|---:|---:|---:|---:|
| 100 | 321.5 | 311 | ~197 | 52.2 | `f23c2962…` byte-identical |
| 512 | 307.9 | 1663 | ~191 | 30.5 | `f23c2962…` byte-identical |
| 1024 | 293.3 | 3492 | ~182 | 27.3 | `f23c2962…` byte-identical |

Prefill is now flat at ~293–322 tok/s (+60–70%), closing the llama.cpp prefill
gap from ~8× to ~4–5×. Decode is unchanged. The matched-benchmark stream hash
stays `f23c2962…` at every context.

## Acceptance gates (phase 13.6)

- `batch_32row_matches_batch_16row_within_tolerance` (max-abs < 1e-3) and the
  existing CPU-oracle test pass;
- `cargo test --workspace` green; `cargo fmt --check` clean;
- digest + exact-token parity gates pass (decode unaffected);
- prefill +60–70% at pp100/512/1024 with the greedy stream hash byte-identical;
- evidence recorded under `artifacts/phase-13.6/`.

## Command book

```zsh
cargo test -p atlas-metal --test batch_matmul_parity
cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 5 --output-format json
```
