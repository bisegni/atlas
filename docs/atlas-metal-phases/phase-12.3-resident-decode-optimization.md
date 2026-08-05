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
