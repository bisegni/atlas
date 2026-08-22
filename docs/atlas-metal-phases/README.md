# Atlas Metal — current state

Status source of truth for the Atlas Metal inference engine.  Historical
per-phase records were consolidated here on 2026-08-18 (they remain in git
history; evidence artifacts stay under `artifacts/`); the open work list
lives in [next-improvements.md](next-improvements.md).

## Engine

Atlas is a Rust-first, Apple-Silicon Metal inference engine for Gemma 4
(production support is Gemma-only; the local fixture is
`gemma-4-e2b-it-q4_0`, pinned in `models/manifest.toml`).  All inference,
benchmarks, and profiles run on the **GPU-resident executor**
(`ExecutorMode::Resident`); the reference executor exists as an oracle for
parity diagnostics only.

## Correctness contract

- Greedy-stream sentinel hash **`f23c2962…`** on the matched workloads
   (pp512/tg128) — the byte-identical end-to-end gate.
- Bitwise-identical kernels: 16-row decode q4 matvec family, PLE gate+gelu
   fusion, PLE epilogue fusion.  The exact qk_norm_rope `_f32`/`_batch_f32`
   kernels remain byte-identical and are the `ATLAS_GEMMA4_EXACT_QKNORM=1`
   fallback.
- Tolerance-level (max-abs 1e-3 / max-rel 1e-2 kernel gates): decode and
   prefill Flash16 attention, mul_mm prefill GEMMs (fp16 fragments), and the
   phase-13.21 wide qk_norm_rope (default; reordered RMS reduction).  The
   canonical greedy fixture was deliberately re-baselined to the wide-kernel
   stream as a documented correctness-contract change.

## Current performance

Recorded on M2 Max, Resident, q4_0 KV, flash16, pp512/tg128, warmup 1:
`benchmark matched --model gemma4-e2b-q4_0 --prompt-tokens 512
--decode-tokens 128 --warmup-runs 1 --runs 3`.

| Metric | Value |
|---|---:|
| Prefill | **~332–336 ms, ~1575–1589 tok/s** |
| Decode (128 tokens) | ~1670–1685 ms, **~77 tok/s** (~13.0 ms/tok) |
| End-to-end host wall | ~2.05 s (prefill + 128 tokens) |
| Greedy stream hash | `f23c2962…1875` (all configs) |

The phase-13.21 wide qk_norm_rope kernels are the **default** — decode
68.8 → ~77 tok/s (+12–14%) and prefill 1537 → ~1577 tok/s (+3%) at
pp512/tg128, with the matched-workload hash still `f23c2962…`.  They reorder
the FP32 RMS reduction (tolerance-level), so the canonical greedy fixture was
re-baselined; the exact single-thread kernels remain reachable via
`ATLAS_GEMMA4_EXACT_QKNORM=1`.

Decode per-token GPU (~18.8 ms, exact per-dispatch profile,
`artifacts/phase-12a/gemma4-resident-decode-profile.jsonl`): attention
swa+full ~3.5–4.4 ms (largest family), ffn gate/up ~2.4, qk_norm_rope ~2.3,
PLE family ~1.7 (71 dispatches), ffn_down ~1.4, lm_head ~1.2, qkv ~1.2.
Decode is GPU-latency-bound; dispatch-count reduction does not raise
throughput (verified twice).  SIMD-group-parallelizing the qk_norm_rope head
reduction is the one single-stream slice that does.

## Production defaults and opt-outs

| Behavior | Default | Opt-out / fallback |
|---|---|---|
| Fast prefill: llama mul_mm GEMMs + Flash16-v7 batched attention | **on** | `ATLAS_GEMMA4_LLAMA_MUL_MM=0`, `ATLAS_GEMMA4_FLASH_PREFILL=0` (legacy prefill, ~317 tok/s A/B) |
| Prefill attention variant | v7 (single-pass) | `ATLAS_GEMMA4_FLASH_PREFILL_V5=1` → v5 |
| Decode q4 matvec band | 16-row/threadgroup | `ATLAS_GEMMA4_DECODE_16ROW=0` → 64/32-row |
| PLE input-gate fusion (gate+gelu in one dispatch) | **on** | `ATLAS_GEMMA4_PLE_SPLIT=1` → split kernels |
| lm_head | 64-row q6 (16-row falsified) | — |
| fp16 batch GEMM / mm64 | off | `ATLAS_GEMMA4_MUL_MM`, `ATLAS_GEMMA4_MM64` |
| Decode split-KV attention (2-pass scan+combine) | off (opt-in diagnostic, phase-13.20 negative) | `--q4-attention-mode flash16_split_kv`, `ATLAS_GEMMA4_SPLIT_KV` (1–32) |
| qk_norm_rope kernels | wide SIMD-group (default, phase-13.21; +12–14% decode / +3% prefill) | `ATLAS_GEMMA4_EXACT_QKNORM=1` → exact single-thread kernels |

## Validation

```zsh
cargo test --workspace                                   # Rust + Metal regression

cargo test -p atlas-metal --test phase_00_bootstrap      # Metal bootstrap
cargo run -p atlas-cli -- model verify --model gemma4-e2b-q4_0

cargo run --release -p atlas-cli -- benchmark matched --model gemma4-e2b-q4_0 \
  --prompt-tokens 512 --decode-tokens 128 --warmup-runs 1 --runs 3 \
  --output-format json                                   # performance + hash gate
```

Fixture-gated tests that need `models/hf/SmolLM2-135M-Instruct`:
`cargo test -p atlas-model --test phase_06_executors -- --ignored` (that
fixture is not present on the current machine).

## Artifacts

- `artifacts/phase-12a/gemma4-resident-decode-profile.jsonl` — exact
  per-dispatch decode profiles (records 25–28: 64-row baseline, all-16,
  fused-PLE, split-PLE A/B).
- `artifacts/phase-13.14/` — prefill attention A/B benches.
- `artifacts/phase-13.20-split-kv-decode/` — split-KV decode attention A/B
   (measured negative; left opt-in, not promoted).
- `artifacts/phase-13.21-qk-norm-rope/` — wide qk_norm_rope A/B (decode
  +11.6→13.7%, prefill +2.6→3.1%, matched hash unchanged; now default, with a
  deliberate canonical-fixture re-baseline to the wide greedy stream).
- `artifacts/atlas-vs-llama/` — llama.cpp gap measurements
  (see section 7 of [atlas-engineering.md](../atlas-engineering.md)).