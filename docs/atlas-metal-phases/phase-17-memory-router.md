# Phase 17: Learned memory router

## Outcome

Deterministic routing controls local, working, graph, and residual paths; a
learned router is only enabled with versioned weights and reproducible evals.

## Work

- Add policies for fixed layers, every-N-token retrieval, and expired-chunk
  writes; emit `RoutingDecision` traces with action/reason.
- Define feature schema, model versioning, fallbacks, budgets, limits, and an
  offline evaluation set before learned routing.

## Model fixture

Use the small fixture for deterministic trace/policy/disabled-path parity and
the larger fixture for a long session reporting actions, GPU bound, and delay.

## Exit gate

`phase_15_router` exactly reproduces its deterministic trace. Disabled routing
matches the standard engine; learned mode records version/evaluation and obeys
latency and retrieval budgets.
