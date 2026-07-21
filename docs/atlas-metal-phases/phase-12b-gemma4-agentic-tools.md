# Phase 12b: Gemma 4 agentic tools

## Outcome

Atlas supports explicit, permission-bounded tool calls from Gemma 4 chat without
allowing model text to execute arbitrary host actions. This phase begins after
the Phase 12a-pre text-chat acceptance gate and does not block it.

## Work

- Define a versioned tool schema with stable names, typed JSON arguments,
  descriptions, result schemas, size limits, and deterministic validation.
- Add a host-owned registry. Every tool declares its filesystem, network,
  process, and user-confirmation permissions; undeclared capabilities and
  unknown tools fail closed.
- Parse tool calls from the Gemma channel protocol at token boundaries. Treat
  malformed, duplicate, oversized, or out-of-order calls as visible errors,
  never as shell text.
- Execute validated calls through a bounded dispatcher with cancellation,
  timeouts, output limits, audit events, and an explicit approval boundary for
  externally visible or destructive effects.
- Render tool results using a canonical structured tool-response turn. Preserve
  within-loop thought state only for the active tool loop; never write raw
  thoughts to standard conversation history or performance artifacts.
- Keep the production executor `Resident`. Tool dispatch must not trigger a
  Reference retry or conceal a Resident inference failure.

## Acceptance

Portable tests cover schema validation, permission denial, token-boundary call
parsing, malformed calls, cancellation, output limits, canonical tool results,
and thought-state lifetime. Apple-Silicon acceptance records a multi-call loop
with Resident metrics, an approval-required denial and approval, tool audit
events, and a final answer grounded in the returned tool results.
