# Phase 9: Sampling engine

## Outcome

A standalone CPU sampler is deterministic under a seed and supports the
required generation policies.

## Work

- Implement greedy, temperature, top-k, top-p, repetition/frequency/presence
  penalties, stop tokens, and seeded RNG.
- Add hand-checkable logits fixtures and keep the sampler independent from the
  model/backend. GPU transforms remain a later optimization behind this API.

## Model fixture

Use the small fixture for real-logit policy tests and the larger fixture for a
256-token repeated seeded stream.

## Exit gate

`phase_07_sampling` passes exact unit cases, repeats seeded output exactly,
stops at the configured sequence, and records the sampling configuration.
