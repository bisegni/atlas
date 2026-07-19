# Phase 8: Streaming generation

## Outcome

The executor yields generated tokens as each decode step completes, so callers
can display a response before the full completion is available.

## Work

- Add a callback or iterator-style generation API that emits token IDs, decoded
  text fragments, and a final finish event while retaining the existing
  non-streaming completion API.
- Keep greedy token order, EOS handling, KV updates, and final metrics
  identical between streaming and buffered generation.
- Report TTFT at the first token, then per-token decode latency and final
  aggregate metrics. Propagate model, tokenizer, and cancellation errors as a
  terminal stream event.
- Update `atlas-cli chat` to write fragments as they arrive; do not wait for
  the complete response before printing it.

## Exit gate

Streaming and buffered greedy generation emit identical token IDs and finish
reasons. The first visible fragment arrives before the final completion, EOS
is emitted once, and metrics show TTFT plus ordered decode timings.
