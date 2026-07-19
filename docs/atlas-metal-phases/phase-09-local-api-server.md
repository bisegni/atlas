# Phase 9: Local API server

## Outcome

`atlas-cli serve` exposes the same greedy runtime through a loopback-only,
OpenAI-compatible HTTP subset. It consumes the Phase-8 token stream rather
than waiting for a completed response.

## Work

- Add `atlas-cli serve --model small [--host 127.0.0.1] [--port 8080]`.
- Implement `/health`, `/v1/models`, and `/v1/chat/completions`, including
  non-streaming JSON and SSE token events for `stream: true`.
- Process one request at a time. Explicit admission limits, 429 responses,
  cancellation, and concurrent sessions arrive in the runtime phase.

## Exit gate

CLI and HTTP return identical greedy tokens for the same prompt; SSE emits
ordered chunks followed by a finish chunk and `[DONE]`.
