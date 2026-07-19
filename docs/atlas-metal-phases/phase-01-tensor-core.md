# Phase 1: Tensor core and allocator

## Outcome

FP32/FP16 tensors support shape, stride, offset, views, and Metal storage; a
decode-shaped loop performs no allocations after warm-up.

## Work

- Define backend-neutral `Tensor`, `Shape`, `Strides`, `DType`, storage, and
  explicit CPU/Metal ownership.
- Implement reshape/transpose metadata, bounds checks, CPU references, and
  FP16 conversion tests.
- Add allocation classes and pooled/arena allocation for weights, KV cache,
  session data, activations, and constants; expose telemetry.

## Model fixture

Download and verify the small fixture. Read SafeTensors headers and create
read-only descriptors for every named weight without duplicating payloads.

## Exit gate

`cargo test -p atlas-core -p atlas-metal --test phase_01_tensor_core` passes
shape/view/dtype/allocator tests. A 1,000-token simulation has zero allocations
after warm-up and reports peak/steady bytes by allocation class.
