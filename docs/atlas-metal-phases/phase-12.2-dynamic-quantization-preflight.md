# Phase 12.2 — Dynamic quantization preflight

Status: implemented baseline-safe preflight; per-group candidate search remains
open. The current implementation profiles the established mixed Q4/Q6 path
against the implemented all-Q4 vocabulary candidate, promotes only exact-parity
results, and caches the safe baseline when no candidate is eligible.

## Goal

On the first Gemma Resident load for a model/hardware pair, discover the fastest
quantization format that preserves generation correctness, write a validated
ready sidecar, and use that sidecar on later loads without repeating profiling.

The available model is the existing Q4_0 GGUF. Therefore the correctness oracle
for this phase is exact greedy token and EOS parity against the source-format
model. Full logit-error parity is deferred until a pinned F16 oracle is
available.

## Runtime behavior

1. Load and stream-hash the GGUF.
2. Look for the adjacent `<model>.quantization-plan.json` sidecar.
3. If the sidecar is ready, valid for the model and hardware, and has supported
   tensor groups, construct the Resident executor from it.
4. If no valid sidecar exists and preflight is enabled, profile candidates on
   Resident Metal before returning the final executor.
5. Compare every candidate with a deterministic fixed prompt and workload.
6. Require exact prompt IDs, generated IDs, measured IDs, and EOS parity.
7. Require stable KV-cache/resident accounting and a positive measurement.
8. Select only candidates that improve long-context throughput and do not
   regress short-context throughput.
9. Atomically write a ready plan. If every candidate fails, write a ready
   baseline plan so future loads do not profile repeatedly.
10. Retain only the winning resident buffers before normal generation begins.

Environment policy:

```text
ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=auto       # default
ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=disabled
ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=verify     # longer promotion-quality run
```

Explicit `ATLAS_GEMMA4_WEIGHT_FORMAT` remains higher precedence than a cached
plan or automatic discovery.

## Candidate and tensor-group support

The profiler may select only formats for which Atlas has a real Resident
kernel and conversion path:

- Q4_0 and Q6_K use the existing Gemma projection paths.
- Q8_0 is eligible only after Gemma weight dispatch is wired to the existing
  Q8 Metal kernels.
- F16 is not eligible until a matching Resident Gemma projection path exists.

Selections are made per executable tensor group, not by blindly changing one
GGUF tensor:

- QKV projection tensors share one format.
- Gate/up projection tensors share one format.
- Embedding and output vocabulary tensors are evaluated as a paired group.
- Unsupported or incomplete groups retain their source format.

The plan schema must record source format, selected format, selected kernel,
GPU timing, resident/upload bytes, parity digests, hardware identity, and the
profiler configuration for each group.

## Implementation work

### Quantization-plan model

- Extend `crates/atlas-model/src/quantization_plan.rs` with explicit plan state,
  hardware identity, profiler configuration, group membership, and benchmark
  evidence.
- Validate model SHA, hardware identity, schema, complete group coverage,
  supported formats, positive timings, parity, and format compatibility.
- Write plans through a temporary file followed by an atomic rename.
- Preserve rejection of stale or pending plans during normal Resident loading.

### Resident storage and executor

- Add format-aware resident buffer slots for candidate tensor groups.
- Add conversion helpers for every candidate format that is actually enabled.
- Add explicit executor construction with a selected format map so preflight
  candidates do not recurse into preflight themselves.
- Dispatch projection, embedding, output, and fused kernels from the selected
  group format.
- Add candidate-arena cleanup so rejected candidates do not remain resident.
- Expose selected group formats and preflight state in generation metrics.

### Profiler

- Add a `Gemma4QuantizationPreflight` controller in `atlas-model`.
- Use a deterministic chat prompt and fixed greedy decode windows.
- Default `auto` mode uses two short runs for bounded startup time.
- `verify` mode uses the longer five-run workload used by promotion A/B tests.
- Profile baseline first, then the currently implemented all-Q4 vocabulary
  candidate. General per-group candidate search remains a follow-up design
  task; see [the quantization-selection research prompt](../../PROMPT-quantization-preflight-selection.md).
- Compare exact token/EOS parity, Resident/KV accounting, and throughput.
- Cache the baseline plan when no candidate is safe or faster.
- Never silently fall back to the Reference executor or CPU execution.

### CLI and diagnostics

Add:

```zsh
atlas-cli model quantization-plan --profile \
  --model gemma4-e2b-q4_0
```

The command must use the same profiler as automatic loading, support `--output`,
print the selected format for every group, and save a complete JSON artifact.

Benchmark and generation records must include:

- `quantization_preflight_state`
- `quantization_plan_path`
- selected candidate/group formats
- baseline and candidate timings
- parity and rejection reasons
- resident/upload accounting

## Tests and acceptance

Portable tests:

- Q4/Q6/Q8 conversion round trips and finite-value checks.
- Plan round-trip, stale SHA, stale hardware, incomplete group, unsupported
  format, invalid timing, and failed-parity rejection.
- Candidate selection chooses the fastest eligible candidate.
- Baseline is selected when all candidates regress.
- Environment override precedence remains intact.
- Failed profiling leaves normal mixed Q4/Q6 behavior unchanged.
- Atomic sidecar writes reload successfully.

Apple-Silicon Resident tests:

```zsh
cargo test -p atlas-model --test phase_12a_gemma4_resident -- --ignored
cargo run --release -p atlas-cli -- model quantization-plan \
  --profile --model gemma4-e2b-q4_0
```

Acceptance requires Resident execution, exact prompt/generated/measured token
SHA and EOS parity, stable KV/resident accounting, matching selected-kernel
telemetry, at least 3% long-context improvement for a promoted candidate, and
no short-context regression. A second model load must consume the cached plan
without rerunning profiling.

Independent confirmation remains:

```zsh
bash scripts/run-gemma4-decode-attention-ab.sh \
  - ATLAS_GEMMA4_WEIGHT_FORMAT=all_q4
```

The phase may be renamed with the repository's `[done]` convention only after
the Resident artifact and all acceptance checks pass on Apple Silicon.

## Recorded acceptance evidence

The following Apple-Silicon Resident evidence is recorded for the implemented
baseline-safe path:

- Release profiling completed for two baseline and two candidate runs per
  workload. The all-Q4 candidate improved long-context throughput by 13.06%
  and short-context throughput by 14.60%, but failed exact short-window parity
  and therefore was rejected.
- A ready mixed Q4/Q6 sidecar was written and validated:
  `models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.quantization-plan.json`.
- A second model load reported `decision=use_cached_plan` and performed
  successful Resident generation with `quantization_preflight_state=ready`.
- The cached generation used `weight_format=mixed_q4_q6`, Q4 KV, resident
  layer-major prefill, and no quantization rejections.
- The ignored Resident acceptance suite completed with 4 passed, 0 failed in
  723.15 seconds.

All-Q4 is not promoted because its generated-token hashes and EOS positions do
not match the mixed Q4/Q6 oracle. Full per-tensor/group quantization search and
additional candidate formats remain open follow-up work.

## Assumptions

- Automatic first-load profiling is the intended runtime mode.
- The existing Q4_0 model is the correctness oracle for this phase.
- Exact greedy token/EOS parity replaces full logit parity until an F16 oracle
  is available.
- Unsupported formats are never selected based only on size or filename.
- If Metal is unavailable, profiling reports the blocker and leaves the
  established mixed Q4/Q6 path unchanged.
