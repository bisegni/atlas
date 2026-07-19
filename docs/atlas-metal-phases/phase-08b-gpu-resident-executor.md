# Phase 8.2: GPU-resident executor integration

## Outcome

The default cached-decode executor uses the Phase-8.1 resident Metal command
path. Model weights, per-session KV cache, and decode activations stay in GPU
buffers; each generated token uses one command buffer and reads back only its
selected token.

## Work

- Give each executor a fixed GPU KV arena and activation scratch arena sized
  from `ModelConfig`; reset clears logical KV positions and drop releases the
  session arena.
- Replace the CPU-`Vec` `forward_token` layer loop with a `ResidentCommand`
  sequence: embedding, norms, Q/K/V projections, Llama RoPE, KV append,
  grouped-query attention, output projection, residuals, MLP, final norm, LM
  head, and GPU argmax.
- Bind the model-owned resident weight buffers directly. Per-token CPU writes
  are limited to token/position scalar inputs; normal decode reads back only
  one `u32` token ID. Final logits remain an explicit diagnostic mode.
- Keep the current CPU-vector executor as the reference oracle. Default chat
  selects resident decode only after exact greedy-token and finish-reason
  parity succeeds for the small fixture.
- Replace global reference-path telemetry with resident token-boundary metrics:
  one command buffer per decode token, zero post-warmup model uploads and
  allocations, token-only readback, host/GPU timing, and decode tok/s.
- Keep INT8/Q4 rejected until direct packed resident projection kernels retain
  exact greedy-token parity.

## Model fixture

Use the small 30-layer fixture for exact greedy parity, command/readback
assertions, and per-token profiling. Use the larger fixture for a sustained
128-token decode run and residency stability.

## Exit gate

The resident and reference executors emit identical token IDs and finish
reasons. After warmup, resident decode reports exactly one command buffer per
generated decode token, zero weight uploads and allocations, and four bytes
of default readback per token. On the same model, prompt, and Apple-Silicon
hardware, resident decode improves tok/s over the reference baseline.
