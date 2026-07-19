# Phase 11: Runtime and scheduler

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

`phase_08_runtime` runs three sessions, cancels one, preserves the other two,
proves queue bounds and cache release, and emits token-ordered events/metrics.
