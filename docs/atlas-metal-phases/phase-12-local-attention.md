# Phase 12: Atlas local attention

## Outcome

Local attention bounds active KV storage and emits ordered expired-context
events for a future memory writer.

## Work

- Add per-layer windows, optional sink tokens, position accounting, and events
  with evicted range and source-session metadata.
- Prove standard-attention equivalence when the window covers the prompt; treat
  post-boundary behavior as intentional, not equivalent.

## Model fixture

Use the small fixture for boundary/position correctness and the larger fixture
for a long-generation bounded-memory profile.

## Exit gate

`phase_10_local_attention` shows fixed maximum KV bytes beyond the window,
lossless ordered expired chunks, and Phase-4 parity with eviction disabled.
