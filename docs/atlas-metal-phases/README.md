# Atlas Metal phase index

Status source of truth for the Atlas Metal inference phases. A phase is
complete only when its declared runnable acceptance gate passed on Apple
Silicon with the required numerical or performance evidence recorded.

## Current phase

- [phase-13.14-remaining-prefill-attention-and-decode.md](phase-13.14-remaining-prefill-attention-and-decode.md) —
  Lever 1 implemented as flash16-v6 (matrix-unit prefill attention,
  `simdgroup_matrix` S=Q·Kᵀ/O+=P·V) and measured **equal to v5, not faster**:
  prefill ~376 vs ~374.5 ms (v5), attention ~87 vs ~80 ms, hash `f23c2962…`
  preserved; v6 kept as a registered tolerance-parity kernel, v5 remains the
  dispatched kernel of record. The attention kernel is q4_0-dequant/bandwidth-
  bound, not dot-product-latency-bound. Lever 2 (decode 1.9×) still open.
- [phase-13.13-flash16-v5-shared-head-prefill-attention.md](phase-13.13-flash16-v5-shared-head-prefill-attention.md) —
  Flash16-v5 prefill attention: one threadgroup per token with one SIMD group
  per head, sharing the K/V q4_0 dequant across heads (kv_heads == 1). Prefill
  attention ~163 → ~80 ms; prefill ~1132 → ~1356 tok/s; gap to llama.cpp ~1.4×
  → ~1.25×; hash preserved.
- [phase-13.12-fast-f16-prefill-mul-mm.md](phase-13.12-fast-f16-prefill-mul-mm.md) —
  Vendored llama.cpp's `kernel_mul_mm_f16_f32` (direct `half4x4` load, no
  dequant) for Gemma 4's fp16 `per_layer_model_proj`. The single naive
  `matmul_f16_batch` GEMM dropped 269.42 → 1.96 ms, cutting prefill ~728 → ~452
  ms (~700 → ~1132 tok/s, +62%); gap to llama.cpp 2.4× → ~1.4×; hash preserved.
- [phase-13.11-flash-prefill-attention.md](phase-13.11-flash-prefill-attention.md) —
  Flash16-v4 merged-slice batched prefill attention
  (`attention_prefill_gemma4_simd_q4_0_flash16_[swa_]v4`, opt-in
  `ATLAS_GEMMA4_FLASH_PREFILL`). Replaces the serial per-key-barrier prefill
  scan. Correct (hash `f23c2962…` preserved) but only +4% prefill because Gemma
  is mostly sliding-window; confirms prefill is now GEMM-dominated.
- [phase-13.10-vendored-llama-mul-mm-prefill.md](phase-13.10-vendored-llama-mul-mm-prefill.md) —
  Vendored llama.cpp's classic simdgroup-matrix `kernel_mul_mm` (MIT) as
  `llama_mul_mm_q4_0_f32` (opt-in `ATLAS_GEMMA4_LLAMA_MUL_MM`) + a
  2D-grid/threadgroup-memory dispatch. Prefill ~316 → ~674 tok/s (+113%),
  closing the llama.cpp gap from ~5x to ~2.4x; decode unchanged; hash preserved.
- [phase-13.9-decode-ple-tail-fusion.md](phase-13.9-decode-ple-tail-fusion.md) —
  Decode PLE-tail dispatch fusion: `gemma4_ple_rms_add_scale_f32` collapses the
  per-layer `rms_norm → vector_add → scalar_multiply` tail into one dispatch.
  Bitwise-correct (greedy hash `f23c2962…` preserved) and cuts decode dispatch
  count 547.7 → 478.2/token (−12.7%), but decode tok/s is unchanged (64.3–64.9)
  because decode is GPU-latency-bound, not dispatch-bound — re-confirming the
  phase-13.4 D3 verdict post-v4. Not a decode-throughput lever.
- [phase-13.8-llama-grade-prefill-mul-mm.md](phase-13.8-llama-grade-prefill-mul-mm.md) —
  llama.cpp-style fp16 matrix-unit prefill GEMM (opt-in `ATLAS_GEMMA4_MUL_MM`):
  `matmul_q4_0_batch_f16` feeds the matrix units from device fp16 fragments with
  NO threadgroup weight staging (dequantized once per layer via
  `gemma4_q4_0_to_f16_batch`, fp32 act cast via `gemma4_cast_f32_to_f16_batch`).
  Tolerance-level (~3–4e-4 relative, gated off by default; the fp32 scalar tile
  stays the `max-abs < 1e-3` default). Kernel parity passes on M2 Max.
  End-to-end measured on the local Gemma 4 E2B fixture: **prefill ~256 → ~286
  tok/s (+12%) at pp512** — a real but modest gain, not the llama-grade
  (~1600 tok/s) jump; see phase-13.9 "Implication for the llama.cpp gap".
- [phase-13.7-flash16-v4-decode-attention.md](phase-13.7-flash16-v4-decode-attention.md) —
  Path B merged-slice decode flash attention: `Flash16` now binds the
  non-bitwise v4 kernels (no per-key barriers; slice-merge in threadgroup
  memory), covered by the max-abs < 1e-3 tolerance contract. Decode more than
  doubled (30.5 → 64.9 tok/s at matched pp512/tg128; ~0.8× llama.cpp's
  empty-context decode), attention ~19 → ~2.5 ms/token, stream hash still
  `f23c2962…` on the matched workloads (drifts at token 50 on the chat prompt —
  the approved Path B tradeoff).
- [phase-13.6-prefill-batched-gemm.md](phase-13.6-prefill-batched-gemm.md) —
  Weight-stationary q4 batched GEMM for prefill: `matmul_q4_0_batch_32row`
  tiles 4 tokens × 32 rows per threadgroup so each weight block is read once
  and reused in registers, cutting prefill weight traffic ~4×. Prefill +60–70%
  (~190 → ~308–322 tok/s flat at pp100/512/1024), closing the llama.cpp prefill
  gap from ~8× to ~4–5× with a byte-identical greedy stream hash. Parity is
  tolerance-level (max-abs ~4.7e-10, user-approved; the bitwise token-major
  variant measured only +5%).
- [phase-13.3-flash16-staged-kv-scan.md](phase-13.3-flash16-staged-kv-scan.md) —
  Decode improvement D2 (gap analysis): staged, chunked, exact-ordered decode
  attention KV scan with wide (512 full / 256 swa) threadgroups. Acceptance
  gate met: v3 bitwise-identical to LegacyFused/`_nb` (per-token fp32 logit
  digests + exact-token stream parity), decode GPU −31.6% at matched pp512/tg128
  with a byte-identical greedy stream hash, decode 53.6→31.0→27.4 tok/s at
  pp100→512→1024 (artifact under `artifacts/phase-13.3/`).
- [phase-13.2-flash16-default-attention.md](phase-13.2-flash16-default-attention.md) —
  Decode improvement D1 (gap analysis): q4 attention defaults to the
  no-value-barrier flash16 variant. Acceptance gate met: per-token fp32
  logit-digest and exact-token parity with LegacyFused preserved, decode GPU
  −8.6% at matched pp512/tg128, greedy stream hash byte-identical (artifact
  under `artifacts/phase-13.2/`).
- [phase-13.1-batched-prefill-kernels.md](phase-13.1-batched-prefill-kernels.md) —
  Token-batched prefill kernels (gap-analysis R1). Acceptance gate met: per-token
  qk/rope, V-RMS, KV-append, attention, and PLE loops collapsed into single
  dispatches; batch cap raised to 512; prefill pp512 49.6 → 199.8 tok/s,
  greedy stream hash byte-identical (artifact under `artifacts/phase-13.1/`).
- [phase-13.0-resident-decode-100-toks.md](phase-13.0-resident-decode-100-toks.md) —
  Resident decode to 100 tok/s. Active: resume, hotspot baseline, ordered
  improvement plan (RMS fusion, dispatch fusion, flash16 v3 tiling, matvec
  ILP), acceptance gates.

## Supporting analysis

- [atlas-vs-llama-gap-analysis.md](../atlas-vs-llama-gap-analysis.md) — why
  llama.cpp is 12–45× faster in prefill and 2–5× in decode on the same GGUF:
  batched GEMM vs per-token prefill loops, and dispatch/occupancy-bound decode
  vs llama.cpp's near-bandwidth-bound fused kernels. Read-only reference for
  prioritizing phase 13+ work.

## Retired phases

Phase 12.3 (resident decode optimization) reached its promotion gate with
the composed flash16_uw + mv_ext + rms-vec4 stack at 56.6/68.9 tok/s
long/short and was retired when phase 13.0 started. Earlier phase documents
were removed as part of that cleanup; their evidence artifacts remain under
`artifacts/`.
