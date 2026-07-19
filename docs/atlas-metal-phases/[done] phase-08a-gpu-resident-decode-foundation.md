# Phase 8.1: GPU-resident decode foundation

## Outcome

Atlas exposes the Metal primitives needed for GPU-resident decode: retained
buffers, a multi-dispatch command buffer, resident model-weight handles, and
single-token decode kernels. The existing executor remains the correctness
path until Phase 8.2 wires these primitives into its layer loop.

## Work

- Retain immutable model weights in GPU-visible `GpuBuffer` handles and expose
  their upload size to the executor.
- Provide `ResidentCommand`, which encodes multiple dependent dispatches and
  completes once at the token boundary.
- Add resident decode kernels for tiled FP32 matrix-vector projection, Llama
  RoPE, KV append, decode attention, and GPU argmax.
- Expose command-buffer, GPU-time, weight-upload, and readback telemetry.

## Model fixture

Use the small fixture to compile and validate the new Metal kernel set. Phase
8.2 owns end-to-end model parity and throughput validation.

## Exit gate

The Metal runtime compiles and caches the resident kernels, model weights can
be retained in GPU buffers, and one `ResidentCommand` can encode dependent
decode dispatches without per-dispatch completion.

## Implementation status

Complete. The command/buffer and kernel foundation is available to the model
executor. It is intentionally not selected by chat generation until the
end-to-end resident executor in Phase 8.2 meets greedy-parity gates.
