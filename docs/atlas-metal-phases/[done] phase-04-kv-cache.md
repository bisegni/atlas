# Phase 4: KV cache

## Outcome

Cached decode matches full-context decode; sliding windows evict safely across
independent sessions with bounded storage.

## Work

- Implement contiguous `[layer][K|V][head][position][dimension]` cache,
  explicit positions, append/view/reset, and per-session accounting.
- Add pages, free lists, page metadata, eviction, and optional sink tokens only
  after contiguous-cache parity passes.
- Compare no-cache, contiguous, and paged paths before the window boundary.

## Model fixture

Use the small model for exhaustive correctness and the larger model for a
1,024-token cache-growth/eviction run.

## Exit gate

`phase_04_kv_cache` proves bounded bytes in sliding-window mode, correct
positions after eviction, no cross-session data, and reports bytes/token,
fragmentation, and eviction time.
