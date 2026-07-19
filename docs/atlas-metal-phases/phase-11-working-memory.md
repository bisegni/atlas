# Phase 11: Recurrent working memory

## Outcome

Fixed session-local slots persist across decode steps, reset/serialize safely,
and fuse through a controlled baseline.

## Work

- Implement 16/32/64 slots, importance, generation, reset, serialization, and
  deterministic read/write selection.
- Begin with gated residual fusion and an explicit zero-gate control. Test
  mechanics/isolation/bounds; do not claim untrained semantic recall.

## Model fixture

Use the small fixture for zero-gate, slot lifecycle, and serialization tests;
use the larger fixture for overhead and session-memory bounds.

## Exit gate

`phase_11_working_memory` has standard-path parity at zero gate, restores
correctly, leaks no session data, and holds stable memory for 10,000 steps.
