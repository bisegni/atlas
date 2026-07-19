# Phase 9: CLI model lifecycle

## Outcome

`atlas-cli` is the supported local product surface. It selects verified local
models by manifest ID, exposes their identity and storage footprint, and
produces stable diagnostics and phase artifacts without an HTTP server.

## Work

- Add a local model manifest with ID, source revision, architecture, tokenizer
  compatibility, hashes, format, and byte totals.
- Add `atlas-cli model inspect`, `model verify`, and manifest-ID model
  selection; retain explicit paths only for fixture/developer workflows.
- Standardize JSON-lines diagnostics for chat/generate/benchmark: model ID,
  format, resident bytes, token IDs, finish reason, timing, and tok/s.
- Remove `atlas-cli serve`; server/API work is reserved for Phase 20.

## Exit gate

The CLI can inspect, verify, select, generate from, and report one pinned FP32
model without arbitrary-path ambiguity or an HTTP dependency.
