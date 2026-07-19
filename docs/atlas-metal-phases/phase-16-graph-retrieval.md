# Phase 16: Graph retrieval

## Outcome

A versioned query retrieves a bounded deterministic set of graph slots and
uploads them to Metal at controlled intervals.

## Work

- Implement CPU vector search, neighbor expansion, score order, query
  fingerprints, retrieval cache keys, and graph-generation invalidation.
- Add Metal query projection and compact transfers; retrieve per prompt or
  every 32–64 tokens, never every token initially.

## Model fixture

Use the small fixture for ordering/invalidation tests and the larger fixture
for long-generation latency, overlap, and retrieval-cache measurements.

## Exit gate

`phase_13_retrieval` returns expected IDs/scores, invalidates on mutation,
caps transfer bytes, and reports retrieval/upload/decode-stall time separately.
