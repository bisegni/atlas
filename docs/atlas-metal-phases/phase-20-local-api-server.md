# Phase 20: Local API server and hardening

## Outcome

After the local CLI/runtime contracts are stable, `atlas-cli serve` exposes a
loopback-only OpenAI-compatible API backed by the same manifest-selected model,
sampler, scheduler, and token stream.

## Work

- Reintroduce `serve` with loopback default binding, health/models/completions/
  chat-completions endpoints, streaming SSE, structured errors, cancellation,
  admission limits, and metrics.
- Use manifest IDs only; expose the pinned revision, model format, and clear
  unavailable Metal/model status.
- Keep HTTP implementation separate from the CLI/runtime library path and
  validate CLI/library/HTTP token and finish-reason parity.

## Exit gate

HTTP and CLI emit matching greedy output for the same manifest model and
sampling configuration; SSE preserves token order, terminates with `[DONE]`,
and all endpoints reject invalid state with structured errors.
