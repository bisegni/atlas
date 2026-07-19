# Phase 18: Memory fusion

## Outcome

Retrieved slots fuse into hidden state with an exact zero-effect control and
measured numerical/performance behavior.

## Work

- Implement gated residual first, then cross-attention and concatenation/
  projection through one interface; validate shape, dtype, mask, and CPU/Metal
  parity.
- Require explicit versioned fusion parameters for nonzero behavior; an
  untrained fuse path is mechanical validation, not a quality claim.

## Model fixture

Use the small fixture for hidden-state and zero-gate goldens. Use the larger
fixture for added decode latency and GPU allocation stability.

## Exit gate

`phase_18_fusion` proves zero-gate parity, records latency/bytes per slot
count, and retains a switchable normal-transformer control path.
