# Atlas agent instructions

Atlas is a Rust-first, Apple-Silicon inference engine. Its primary runtime
contract is correct, measurable Metal inference, not a CPU fallback that only
appears to work.

## First read and source of truth

Before changing a feature, inspect the relevant crate, its focused tests, and
the applicable phase plan under `docs/atlas-metal-phases/`. The phase index in
`docs/atlas-metal-phases/README.md` is the status source of truth.

- `crates/atlas-metal`: Metal device, buffers, command encoding, kernels, and
  GPU telemetry.
- `crates/atlas-ops`: neural-network operators and their numerical behavior.
- `crates/atlas-model`: model loading, prefill/decode execution, generation,
  executor modes, and residency/parity diagnostics.
- `crates/atlas-cli`: user-facing CLI commands and runtime reporting.
- `crates/*/tests`: focused regression and phase acceptance coverage.

Read `docs/Atlas_Metal_Inference_Engine_Phase_Subplans.md` when work involves
fixtures, acceptance evidence, phase gates, or artifacts. Preserve existing
user changes: inspect `git status` and `git diff` before editing, and do not
revert or reformat unrelated files.

## Execution policy: GPU-resident by default

Use the GPU-resident executor (`ExecutorMode::Resident`) by default for Atlas
inference, CLI flows, benchmarks, and GPU validation. New production defaults
must select the resident executor unless a documented, user-approved exception
requires another mode.

The reference executor is an oracle for parity, diagnostics, and deliberately
named comparison tests. It is not an acceptable silent production fallback.
When a resident execution fails, surface the failure and investigate it; do
not mask it by rerunning the same user-facing flow through the reference path.

For any executor-mode change, make the selected mode observable in code,
diagnostics, or test assertions. GPU-residency claims need evidence such as
resident-byte, upload/readback, allocation, command-buffer, or timing metrics;
kernel dispatch alone is not enough.

## Validation workflow

Run the narrowest meaningful checks first, then broaden validation when the
environment permits:

```zsh
# Fast Rust regression baseline
cargo test --workspace

# Metal bootstrap/device path
cargo test -p atlas-metal --test phase_00_bootstrap
cargo run -p atlas-cli -- metal-info

# Confirm the small fixture is usable before fixture-gated GPU tests
cargo run -p atlas-cli -- fixture verify --model small

# Explicit Metal + downloaded-fixture phase tests
cargo test -p atlas-model --test phase_06_executors -- --ignored
```

Use a focused crate/test command when it proves the changed behavior more
directly. Run `cargo fmt --check` for Rust edits. Treat compilation, unit
tests, fixture verification, and a real Metal execution as separate evidence
boundaries: a passing portable test suite does not prove GPU execution,
residency, correctness parity, or performance.

Do not weaken tolerances, remove parity assertions, unignore GPU gates, or
replace a real execution with a mock merely to obtain a green result. Any
intentional test/acceptance-gate change must explain the new invariant and
retain an equivalent or stronger proof.

## When automatic GPU validation is unavailable

Run GPU tests automatically whenever the current environment has Apple Silicon
with Metal access and the required local fixture. If that is unavailable—for
example because hardware, Metal permissions, model files, or a stable
performance environment are missing—state the precise blocker in the final
report and do all non-GPU validation that remains meaningful.

Then give the user:

1. The exact copy-pasteable command to run.
2. Required prerequisites, including the expected fixture path.
3. The expected pass signal and the output/metrics to report back.
4. Which claim remains unverified until the result is supplied.

For example, fixture-gated resident acceptance tests require
`models/hf/SmolLM2-135M-Instruct/` and can be run with:

```zsh
cargo test -p atlas-model --test phase_06_executors -- --ignored
```

Treat the user-provided result as the GPU acceptance evidence. Never claim a
GPU test passed, that resident execution was used, or that performance improved
without the corresponding runtime output.

## Fixtures, benchmarks, and artifacts

Model files are developer-local test fixtures and must not be committed.
Follow the shared phase contract for Hugging Face dry-run/download commands,
revision pinning, and artifact recording. Use the small SmolLM2 fixture for
correctness and parity unless the relevant phase explicitly requires another
fixture; use the larger fixture only for its stated performance/memory gate.

For performance work, warm up pipelines and resident weights before measuring.
Compare like-for-like prompt, token count, fixture, process, and executor
configuration. Report enough context to interpret the result: hardware/OS
when known, model revision, command, executor mode, prompt/token workload,
warm-up policy, elapsed time or throughput, and relevant residency metrics.

## Documentation and completion

Keep implementation and phase documentation aligned. A phase is complete only
when its declared runnable acceptance gate has passed on Apple Silicon and its
required numerical or performance evidence is recorded. When marking a phase
complete, use the repository's `[done]` filename convention and update the
link in `docs/atlas-metal-phases/README.md`; do not renumber later phases when
a fractional phase can express the work.

## Final report

State concisely:

- what changed and the affected crate(s);
- the executor actually used (`Resident`, `Reference`, or not run) and why;
- commands run and their results;
- GPU evidence collected (or the explicit missing evidence);
- any exact user-run GPU command, prerequisite, expected output, and remaining
  acceptance claim when local GPU execution was not possible.
