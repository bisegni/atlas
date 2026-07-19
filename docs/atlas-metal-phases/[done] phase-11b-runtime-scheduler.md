# Phase 11b: Runtime and scheduler

## Outcome

A bounded scheduler streams isolated sessions, supports cancellation, and
exposes per-session metrics.

## Work

- Implement one request first, then admission limits, cancellation, cleanup,
  and continuous batching.
- Preserve session-local cache and sampling; expose queue wait, TTFT, decode
  latency, token count, cache bytes, cancellation, and errors.

## Model fixture

Use the small fixture for deterministic concurrent/cancellation tests and the
larger fixture for a sustained multi-session memory/latency soak.

## Exit gate

`phase_13_runtime` runs three sessions, cancels one, preserves the other two,
proves queue bounds and cache release, and emits token-ordered events/metrics.

Run the Metal acceptance gate only after downloading the small fixture:

```zsh
scripts/download-models.sh
cargo test -p atlas-model --test phase_13_runtime -- --ignored
```

The successful run must show the three-session test passing. Its runtime
metrics must confirm `ExecutorMode::Resident`, non-zero resident bytes for the
two completed sessions, a cancelled middle session, and no active or queued
sessions after cleanup. Do not mark this phase complete until that Apple
Silicon evidence is recorded.

## Acceptance evidence

Apple-Silicon Metal acceptance completed on 2026-07-19 using the downloaded
small fixture:

```text
cargo test -p atlas-model --test phase_13_runtime -- --ignored
test phase_13_runtime_streams_three_sessions_cancels_one_and_releases_slots ... ok
```

The gate passed in 7.81 seconds. It exercised three sessions, cancelled the
middle session, and confirmed cleanup of active/queued runtime slots.
