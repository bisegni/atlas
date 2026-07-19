# Phase 2: Essential neural operators

## Outcome

Metal operators reproduce one decoder block's saved intermediate tensors.

## Work

- Implement embedding, add/multiply, SiLU, RMSNorm, matvec, matmul, RoPE,
  masked softmax, attention score/value aggregation, output projection, and
  logits processing in that order.
- Keep separately specialized prefill and decode paths, selected by dtype,
  dimensions, transpose, and execution mode.
- Store CPU-oracle tensors for fixed tokens, positions, and one model layer;
  record a tolerance per operator.

## Model fixture

Download the small fixture and use its actual hidden size, head layout, RoPE
settings, and first-layer weights.

## Exit gate

`cargo test -p atlas-ops --test phase_02_operators` passes both prefill and
decode shapes, reports max absolute/relative error per operator, and fails on
unclassified NaN/Inf or a tolerance regression.
