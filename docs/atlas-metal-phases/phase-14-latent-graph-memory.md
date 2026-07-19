# Phase 14: Latent graph memory

## Outcome

A durable in-process graph accepts expired chunks, writes append-only records,
restores snapshots, and returns stable node/edge IDs.

## Work

- Implement memory-mapped node records, event log, adjacency, vector index,
  snapshots, compaction, and corruption-safe recovery; keep it CPU-resident.
- Define versioned chunk-to-node input with source, timestamp, importance,
  confidence, and embedding/model version.

## Model fixture

Use the small fixture to create real tokenized source chunks. Use the larger
fixture for multi-session graph writing and restart-recovery soak testing.

## Exit gate

`phase_12_graph_store` writes, snapshots, restarts, and replays a checksummed
graph with stable IDs/no cross-session data, reporting write/recovery metrics.
