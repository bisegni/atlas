# Phase 12.3: Resident Decode Optimization

## Objective

Improve the matched M2 Max Resident workloads over their fresh baseline while
preserving exact greedy output, Resident execution, and accounting. There is
no fixed percentage target: a candidate must be faster in the measured
long-context workload and must not regress the short workload. Graph-cost
candidates are screened independently before any composition; long-context
attention remains a separate evidence-gated stream. No production default
changes are permitted in this phase.

## Baseline and measurement contract

Use `gemma4-e2b-q4_0`, the Resident executor, the production mixed Q4/Q6
weight plan, and the fixed benchmark prompt:

```zsh
cargo run --release -p atlas-cli -- benchmark \
  --model gemma4-e2b-q4_0 \
  --prompt 'Explain Resident inference.' \
  --warmup-decode-tokens 32 \
  --decode-tokens 128
```

The default long workload is 1024 warmup tokens and 512 measured tokens with
`--max-context 2048`. Both workloads may be overridden, but every A/B record
must retain the exact prompt, model, KV type, warmup, measured-token count,
process configuration, and executor mode.

Diagnostic profiling supplies semantic attribution and exact per-dispatch
timing. It must not be used as production throughput evidence because its
command-buffer splitting changes host synchronization cost. Normal benchmark
mode supplies throughput and production-boundary GPU telemetry.

Run diagnostic capture with `--gpu-counters auto` to record capability, the
available Metal counter names, and pipeline occupancy metadata. `required`
fails when dispatch-boundary sampling is unavailable. Counter output is
diagnostic-only and must never alter the production benchmark command-buffer
boundaries or be used as throughput evidence.

Before proposing another candidate, establish a clean baseline profile with:

```zsh
bash scripts/profile-gemma4-resident-baseline.sh
```

Then establish the context curve with the production-only baseline sweep:

```zsh
bash scripts/benchmark-gemma4-resident-context-sweep.sh
```

The sweep samples 0, 256, 512, 1024, and 1536 warmup decode tokens with 128
measured tokens and records five medians per point. A sharply rising GPU cost
with context justifies an attention/KV candidate; a flat curve means the next
candidate must reduce projection or graph cost in both workloads. This is a
measurement harness, not a candidate gate.

The script explicitly clears all active experiment selectors, pins the known
selectors to their baseline values, verifies the fixture, and stores one
artifact directory under `artifacts/phase-12.3-baseline-profile/`. It produces
five production Resident samples for each short and long workload, then one
exact per-dispatch diagnostic capture for each. The resulting
`baseline-profile-summary.md` is the decision surface: production medians are
for performance gates, while the ranked decode-measured operation, kernel, and
layer reports identify where a future hypothesis is permitted. A candidate is
not justified merely by its apparent convenience; its target must have at
least 95% timing and dispatch coverage, a stable baseline kernel selection,
and a stated dispatch, traffic, or occupancy hypothesis. The normal benchmark
median is the only speed oracle. Candidate screens have no fixed promotion
percentage: each report records short/long throughput and GPU-time deltas,
dispatches per token, estimated traffic, counter/occupancy classification,
parity, and its rollback selector.

## Work

1. Verify profiler scope boundaries, operation-family attribution, dispatch
   and GPU coverage, layer/kernel rankings, and conservative traffic estimates.
2. Rank the top three decode-measured families by total GPU time and record
   dispatches, threadgroups, estimated bytes, command buffers, Resident bytes,
   KV bytes, transfers, and selected kernels in JSON and Markdown artifacts.
3. Screen opt-in candidates independently: Gate/Up GELU epilogue, hidden-size
   RMSNorm, FFN-Down `interleaved16`, then the packed-16 FFN-Down + PLE
   projection layout (`ATLAS_GEMMA4_Q4_PACKED16_EXPERIMENT=ffn_down_ple`).
   The packed layout replaces each selected Q4 resident buffer during initial
   upload with byte-for-byte `[16-row tile][block][row]` records and uses the
   matching decode and prefill kernels. Do not combine candidates until each
   preserves the exact stream and Resident/KV accounting with no regression.
4. Run five matched baseline/candidate samples for short and long workloads
   and compare medians. Keep all selectors opt-in until result review.

`2pass_mqa_tiled` is permanently rejected. Its short decode regressed 1.3%,
long decode regressed 12.9%, and it changed the full long generated stream.
It is retained only as an explicitly rejected diagnostic selector; it must not
be added to a candidate queue or receive a five-run promotion attempt.

`ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=fused` is also rejected: the supplied
five-run M2 Max A/B preserved the full and measured token SHA/EOS and reduced
dispatches, but regressed the short median from 43.537 to 42.896 tok/s
(-1.47%) and the long median from 30.131 to 29.842 tok/s (-0.96%). Keep the
baseline composition selected in every future candidate unless a separate
evidence-backed design supersedes it.

Later graph-cost screens also found no remaining dispatch-fusion win:
`ATLAS_GEMMA4_FFN_GELU_MULTIPLY_EXPERIMENT=fused` (+1.24% long, -0.16%
short), the RMS epilogue fusion (-0.89% long, -1.28% short), and the
combined qk-norm-rope + gelu-multiply + PLE stack (+0.15% long, -1.14%
short) are all bitwise-exact but performance-neutral at
`artifacts/phase-12.3-*-ab/20260806T04*-05*Z/`. The qk-norm-rope fusion is
already the production default (`gemma4_qk_norm_rope_fused_enabled` returns
true unless explicitly disabled), so its +5.25% long / +9.35% short A/B gain
was already banked; dispatch fusion is exhausted and no further candidate
should be admitted on a pure dispatch-count hypothesis.

`ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=vec4` screens PASS on the first single-run
screen and is the strongest candidate of the phase: bitwise-exact long and
short streams, stable Resident/KV/upload accounting, and
+8.67% long (29.70 -> 32.27 tok/s) / +12.88% short (42.19 -> 47.62 tok/s) in
`artifacts/phase-12.3-rms-norm-ab/20260806T052431Z/`. The five matched runs
collected there confirm the direction with medians of +8.75% long / +12.98%
short. It remains opt-in per the no-default-change rule; it is the primary
promotion candidate for the phase that lifts that rule.

The composed screen of the two strongest opt-in candidates (RMS vec4 +
`2pass_simd_reg` attention, one matched run each) also passes exactly and
additively at `artifacts/phase-12.3-rms-simdreg-composed-ab/20260806T054938Z/`:
+14.81% long (29.83 -> 34.24 tok/s) / +14.94% short (42.17 -> 48.47 tok/s),
bitwise-exact long and short streams, stable Resident/KV/upload/readback
accounting, and both candidate kernels selected. This composed stack is the
promotion evidence for the phase that lifts the no-default-change rule.

## Speed-first attention and Q4 matvec experiments (parity gate superseded)

The bitwise-parity gate is superseded by a speed-only policy; the flash16
screen below is tolerance-checked (max-abs < 1e-3 against the production
pipeline in `crates/atlas-metal/tests/attention_flash_correctness.rs`) and
stream hashes are diagnostics only.

The single-dispatch flash16 attention kernel (llama.cpp flash-attention
structure: per-simdgroup register softmax state, per-key online rescale, one
threadgroup merge, KV read once, no partial/max/sum buffers, no combine
dispatch) PASSES on the first single-run screen:

- `attention_decode_gemma4_simd_q4_0_flash16` (head 512, 8 slices) and
  `attention_decode_gemma4_simd_q4_0_flash16_swa` (head 256, 16 slices),
  selected via `ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=flash16`;
- +19.5% long (29.56 -> 35.32 tok/s) / +6.8% short (41.77 -> 44.61 tok/s),
  GPU time -16.2% long / -5.9% short, 813 -> 778 dispatches per token
  (the per-layer combine dispatch is gone), Resident executor, stable
  accounting, at `artifacts/phase-12.3-q4-attention-flash16-ab/20260806T063412Z/`.
- Note: the model head dimension is 512 (full) / 256 (sliding window), not
  288; an earlier 288-specialized variant silently failed the support gate and
  the screen must show `candidate_kernel_selected: true`.

The Q4 FFN matvec re-tiling candidates are all rejected and the direction is
exhausted: packed-16 (-4.0% long), interleaved-16 (requires the packed-16
layout and was never screened), and `simdgroup_tiled`
(-14.8% long / -21.7% short, GPU +17.9% / +28.6%,
`artifacts/phase-12.3-q4-matvec-simdgroup-ab/20260806T060145Z/`,
verdict recorded in `simdgroup-tiled-verdict.json`). Q6 output-projection
cache-opt also showed no promotion (+0.05% long / -0.26% short).

Before writing candidate code, create `optimization-decision.json` and its
Markdown companion from a clean baseline profile plus context sweep. The
record must include fixture/token hashes, context-sweep slope, selected kernel
plan, attribution coverage, supported counter set, bottleneck classification,
an Amdahl upper bound, and an explicit approve/reject rationale. Candidate
admission requires at least 95% measured-decode attribution coverage, stable
Resident/KV/transfer accounting, and a limiter-specific design. RMSNorm
geometry changes require counter evidence of an occupancy, register, or
latency limiter. Attention is ineligible until the context sweep rises and
counters classify `attention_score` as bandwidth/cache limited.

Classify occupancy/resource pressure separately from memory/cache pressure
using the diagnostic metadata and counters before changing a workgroup layout.
Do not revisit shared-KV staging or per-key threadgroup barriers unless new
counter evidence demonstrates a specific occupancy or cache benefit. If
attention lacks that classification, route to the largest common classified
hotspot (RMS norm, Q6 output projection, or FFN projection).

Every candidate requires an environment flag, exact kernel selection, exact
prompt/generated/measured token SHA, EOS parity, Resident execution, stable
Resident/KV accounting, zero warm uploads, and no additional readback.

The packed-16 candidate is measured with:

```zsh
bash scripts/run-gemma4-q4-packed16-ab.sh
```

Use `--screen` only to reject a regression with two matched samples. The
packed-16 script retains its own configured gate and never changes a
production selector.

## llama.cpp-style 32-row matvec (`mv_ext`) screen

The Q4 matvec and Q6 LM-head kernels are re-tiled as llama.cpp `mul_mv`
ports: 32 rows per threadgroup (4 SIMD groups x 8 rows), one activation block
load per SIMD group with the nibbles pre-scaled by 1/16/1/256/1/4096 powers
of two, the -8 quant offset folded into the per-block raw activation sum, and
a full-SIMD `simd_sum` reduction. Accumulation order differs from the 16-row
kernels, so stream hashes are diagnostics only (parity gate superseded);
correctness is enforced by the kernel-level tolerance test
`crates/atlas-metal/tests/matvec_mv_ext_parity.rs` (4 tests, max-abs < 1e-3
against the production kernels, covering partial-width rows).

The first device run exposed a real layout bug that was fixed before any
screen: `matvec_q6_k_32row_mv` read ql/qh/scales at llama in-memory offsets
(+2/+130/+194) but Atlas Q6_K blocks store them at +0/+128/+192 with the f16
super-block scale at +208 (GGUF order, as read by `matvec_q6_k_8row` and
`atlas-core` dequantize). The mv kernel's scale window even clobbered the
last two scale bytes. The offsets were corrected to the Atlas layout; the
tolerance parity test for Q6 failed on the buggy kernel and passes after.

The mv_ext screen PASSES on the first screen
(`scripts/run-gemma4-mv-ext-ab.sh --screen`,
`artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260806T204506Z/`):

- selected kernels `matvec_q4_0_32row_mv`, `matmul_q4_0_qkv_32row_mv`,
  `matmul_q4_0_gate_up_32row_mv`, `matvec_q6_k_32row_mv` on top of the
  flash16 attention default, Resident executor, stable accounting, and the
  tolerance parity tests pass;
- +21.7% long (35.26 -> 42.92 tok/s, GPU 14113 -> 11541 ms) and
  +32.6% short (43.72 -> 57.99 tok/s, GPU 2716 -> 2098 ms);
- stream hashes drift (expected: accumulation-order rounding flips near-tie
  argmax), EOS and accounting stay stable.

A composed screen adding the RMS-normalization vec4 kernel
(`ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=vec4` -> `rms_norm_decode_f32_vec4`, the
third-largest decode bucket) PASSES on the first run
(`artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T004810Z/`):

- candidate = mv_ext + `rms_norm_decode_f32_vec4` on the flash16 attention
  default, Resident executor, all accounting checks pass;
- +40.6% long (35.56 -> 49.99 tok/s, GPU 14062 -> 9999 ms) and
  +57.1% short (44.09 -> 69.26 tok/s, GPU 2757 -> 1739 ms) vs the pre-mv_ext
  baseline; the vec4 kernel alone is worth roughly +7 tok/s long and
  +11 tok/s short on top of the 42.92/57.99 mv_ext screen;
- stream hashes drift as before; EOS and accounting stay stable.

## Flash16 v2 kernels

The flash16 attention kernels (top decode bucket after mv_ext + vec4) get
second-generation variants that keep the same interface and dispatch
geometry but reduce per-key work:

- `attention_decode_gemma4_simd_q4_0_flash16_u` / `..._swa_u`: the 16/8 block
  loops are force-unrolled so the per-block accumulator update chain
  (`if (b == B) accB = ...`) constant-folds instead of evaluating runtime
  branches for every value block, and the query vector is cached in registers
  for the serial key scan instead of being re-read per key;
- `..._uw` / `..._swa_uw`: same code plus wider slices (12 full-head, 24
  sliding-window SIMD groups, 384/768 threads) within the threadgroup-memory
  budget for the merge buffers;
- selected via `ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=flash16_u` /
  `flash16_uw`, labeled `gemma_attention_flash16` / `gemma_attention_flash16_swa`;
- both are semantics-preserving (same accumulation order), so the existing
  tolerance test `crates/atlas-metal/tests/attention_flash_correctness.rs`
  covers all six flash16 kernels with max-abs < 1e-3; the Metal pipeline
  count assertion in `phase_02_operators` was updated 105 -> 109.

The device threadgroup budget was confirmed by exposing the fields in
`metal-info` (DeviceInfo carries them since the flash16 work): M2 Max =
32 KiB threadgroup memory and 1024 max threads per threadgroup, so the uw
merge buffers (12x512x4 = 24 KiB full, 24x256x4 = 24 KiB sliding) fit.

The `flash16_uw` screen (`scripts/run-gemma4-attention-flash16-v2-ab.sh
--screen`, `artifacts/phase-12.3-q4-attention-flash16-v2-ab/20260809T010951Z/`)
compares flash16 vs flash16_uw on the composed mv_ext + vec4 stack:

- long +11.6% (48.31 -> 53.94 tok/s, GPU 9994 -> 8565 ms), kernels correctly
  selected and all accounting checks pass;
- short -1.78% (68.67 -> 67.45 tok/s) with overlapping GPU times
  (1686-1749 ms vs 1719-1722 ms), consistent with run-to-run noise, but the
  screen gate (no short regression) therefore does not pass;
- the five-run promotion A/B
  (`artifacts/phase-12.3-q4-attention-flash16-v2-ab/20260809T011842Z/`)
  decides the verdict: PASS with long +15.1% (47.28 -> 54.41 tok/s, GPU
  9947 -> 8565 ms) and short +1.7% (65.78 -> 66.90 tok/s), so the short
  screen dip was noise; all accounting, determinism, and residency checks
  pass and the flash16_uw kernel is correctly selected. The v2 attention
  variants are therefore promotion candidates on the composed stack.

## Composed promotion A/B (flash16_uw + mv_ext + vec4)

The five-run promotion A/B of the full composed candidate against the
production baseline PASSES
(`scripts/run-gemma4-mv-ext-ab.sh --with-rms-vec4 --with-flash16-uw`,
`artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T013211Z/`):

- candidate = `flash16_uw` attention + `matvec_q4_0_32row_mv` +
  `matmul_q4_0_qkv_32row_mv` + `matmul_q4_0_gate_up_32row_mv` +
  `matvec_q6_k_32row_mv` + `rms_norm_decode_f32_vec4`, Resident executor;
- long 35.25 -> 56.64 tok/s (+60.7%, GPU 14127 -> 8641 ms) and
  short 44.70 -> 68.94 tok/s (+54.2%, GPU 2745 -> 1737 ms);
- all accounting, residency, determinism, and kernel-selection checks pass;
  stream hashes drift as expected (accumulation-order rounding), EOS stable;
- the promotion summary initially mis-reported the attention check because
  of a script path bug (`$cl.long.checks` instead of `$cl.checks`); fixed in
  `scripts/run-gemma4-mv-ext-ab.sh` and the summary regenerated from the
  same five-run records without re-running benchmarks.

## Traffic estimate contract

Per-dispatch traffic is a conservative bound over the remaining spans of bound
Resident buffers. The estimate sums the remaining bytes of bound buffers as
reads and uses the common third binding as the output-write estimate. It is
intended for relative hotspot ranking only; it is not a Metal bandwidth result.

## Promotion gate

Promote only when all conditions pass:

- exact Atlas token and EOS parity;
- correct candidate kernel selected;
- Resident executor with no CPU fallback;
- unchanged Resident, KV, upload, readback, and allocation accounting;
- no short-workload regression at any intermediate screen;
- a measured composed result exceeds the matched short-context baseline before
  any promotion decision;
- five-run evidence recorded under `artifacts/phase-12.3/`.

If Apple-Silicon Metal or the Gemma fixture is unavailable, complete portable
tests and report the exact missing GPU evidence. The phase cannot be marked
`[done]` until the runnable Resident acceptance gate passes on Apple Silicon.
