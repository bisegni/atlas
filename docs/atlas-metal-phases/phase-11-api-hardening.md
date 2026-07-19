# Phase 11: API compatibility and hardening

## Outcome

CLI, Rust library, and local OpenAI-compatible HTTP requests share the same
runtime path and accurately identify the loaded model after the Phase-8
minimal server and Phase-10 scheduler are in place.

## Work

- Keep `atlas-server` separate; add health, models, completions, chat
  completions, streaming, structured errors, cancellation, and metrics.
- Select models by local manifest ID, not arbitrary paths; test sampling and
  stop-token semantics before exposing an endpoint.

## Model fixture

Use the small fixture for API/streaming contracts and the larger fixture for an
HTTP prefill/decode smoke test and model metadata validation.

## Exit gate

`phase_09_api` compares CLI/library/HTTP tokens and finish reasons. `/v1/models`
returns the pinned revision; `/health` clearly reports unavailable Metal/model.
