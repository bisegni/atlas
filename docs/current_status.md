# Gemma quantization preflight status

This document summarizes the current implementation and acceptance state of
Gemma 4 quantization preflight in Atlas.

## What the implementation does

Atlas performs a hardware-specific Resident Metal format-selection preflight
when a Gemma model has no valid cached plan.

The preflight:

1. Loads and hashes the GGUF model.
2. Profiles the existing mixed Q4/Q6 Resident path.
3. Profiles the implemented all-Q4 candidate.
4. Compares throughput, Resident memory, KV-cache bytes, upload/readback
   accounting, generated-token hashes, and EOS positions.
5. Promotes a candidate only when it satisfies exact parity and performance
   gates.
6. Writes a validated JSON sidecar beside the GGUF model.
7. Uses the sidecar on later loads without repeating profiling.

The preflight is part of the inference path. Candidate profiling performs real
Resident inference, and normal chat later consumes the selected plan during
Resident inference.

## Current format terminology

The downloaded GGUF is already quantized. Its safe baseline is called
`mixed_q4_q6`:

- eligible projection weights use Q4;
- Gemma vocabulary tensors use Q6;
- control and normalization tensors that require F16/F32 retain those formats.

The `all_q4` candidate does not quantize every GGUF tensor. It converts the
eligible Q6 vocabulary tensors to Q4 while leaving unsupported F16/F32 tensors
unchanged.

The current implementation is therefore a Resident quantization-format
autotuner and cache, not a general-purpose dynamic quantizer that discovers
arbitrary per-tensor quantization levels.

## Cached plan

The generated sidecar is:

```text
models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.quantization-plan.json
```

The plan records the model SHA, hardware identity, tensor/group source and
selected formats, selected kernels, profiling configuration, timing,
residency/upload evidence, and parity digests.

The release CLI validates the sidecar with:

```zsh
cargo run --release -p atlas-cli -- model quantization-plan \
  --model gemma4-e2b-q4_0
```

The validated plan contains 243 tensor entries.

## Recorded Apple-Silicon evidence

The release preflight used two runs for each workload and completed successfully
for both baseline and candidate:

| Workload | Mixed Q4/Q6 | All-Q4 | Candidate result |
| --- | ---: | ---: | --- |
| Long context | 44.25 tok/s | 50.03 tok/s | 13.06% faster |
| Short context | 49.41 tok/s | 56.63 tok/s | 14.60% faster |

All-Q4 is rejected because its generated-token and EOS results do not exactly
match the mixed Q4/Q6 oracle. In particular, the long workload reaches EOS at a
different position, and the short-window parity check fails.

Atlas therefore writes and selects a ready mixed Q4/Q6 baseline plan instead
of promoting all-Q4.

The cached-plan inference check reports:

```text
decision=use_cached_plan
executor=resident
weight_format=mixed_q4_q6
quantization_preflight_state=ready
quantization_rejections=[]
```

The Gemma Resident acceptance suite also passes:

```text
4 passed; 0 failed; elapsed 723.15 seconds
```

The passing tests cover canonical token parity and warm behavior, fixed
benchmark windows, post-EOS continuation, and packed KV-cache determinism and
residency reduction.

## Current state

The baseline-safe preflight path is implemented and validated:

- Resident profiling runs real inference;
- failed candidates are rejected without changing the safe baseline;
- a ready plan is written atomically;
- the plan validates against the model and hardware;
- subsequent loads consume the cached plan;
- normal chat uses Resident mixed Q4/Q6 inference;
- the Resident acceptance suite passes.

The all-Q4 optimization is not promoted because exact parity fails. This is an
expected rejection under the phase gates, not a profiling failure.

Phase 12.2 remains open for broader per-group quantization selection. The
current implementation does not yet search independently over arbitrary
formats for every tensor or executable group.

## Follow-up objective

The next design task is to determine whether preflight can safely choose the
best format for each executable tensor group. The investigation must address:

- selectable group boundaries and format dependencies;
- exhaustive versus guided search strategies;
- calibration prompts and workload windows;
- exact parity against the mixed-format oracle and, later, an F16 oracle;
- interaction between local tensor speed and complete-model speed;
- cache representation for per-group decisions;
- startup cost, rejected-buffer cleanup, and promotion rules.

The research prompt is stored at
[PROMPT-quantization-preflight-selection.md](../PROMPT-quantization-preflight-selection.md).

## Reproduction commands

Profile and write the plan artifact:

```zsh
cargo run --release -p atlas-cli -- model quantization-plan \
  --profile --model gemma4-e2b-q4_0 \
  --output artifacts/quantization-plans/gemma4-e2b-q4_0.json
```

Verify cached-plan inference:

```zsh
ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT=auto \
cargo run --release -p atlas-cli -- chat \
  --model gemma4-e2b-q4_0 \
  --prompt "Reply exactly: cached plan works" \
  --max-tokens 8
```

Run the Resident acceptance suite:

```zsh
cargo test -p atlas-model --test phase_12a_gemma4_resident -- --ignored
```
