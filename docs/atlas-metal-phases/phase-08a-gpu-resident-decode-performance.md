# Phase 8.1: GPU-resident decode performance

## Outcome

The executor keeps model weights, KV cache, and decode activations resident in
Metal buffers. A generated token uses a bounded number of command buffers and
does not copy model weights or intermediate activations through CPU `Vec`s.

## Work

- Introduce GPU-resident tensor handles for immutable model weights and
  session-local KV/activation storage. Upload each weight once when the model
  is loaded; release all session storage when its executor is dropped or reset.
- Encode dependent decode operations into one command buffer, or a documented
  small fixed number when a synchronization boundary is unavoidable. Replace
  per-operator `waitUntilCompleted()` calls with completion only at the token
  boundary.
- Keep intermediate states on the GPU across norms, projections, RoPE,
  attention, residuals, and MLP operations. Read back only the selected token,
  final user-visible metrics, and explicitly requested diagnostics.
- Replace the reference FP32 matrix-vector path, including the LM head, with
  tiled Metal kernels sized for one-token decode. Integrate the existing packed
  weight formats only when exact greedy-token parity is retained.
- Expose host wall time, GPU execution time, command-buffer count, weight-upload
  bytes, readback bytes, and post-warmup allocations in executor metrics.

## Model fixture

Use the small 30-layer fixture for per-token profiling and exact greedy parity.
Use the larger fixture for sustained decode throughput and residency checks.

## Exit gate

GPU-resident and reference greedy decode emit identical token IDs and finish
reasons. After warmup, decode performs no model-weight uploads, has no
per-token model-weight allocations, uses the documented bounded command-buffer
count, and reports GPU versus host timing. The measured token rate improves
over the Phase-8 reference baseline on the same model, prompt, and hardware.
