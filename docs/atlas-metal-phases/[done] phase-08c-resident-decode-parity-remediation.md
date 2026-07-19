# Phase 8.3: Resident decode parity remediation

## Outcome

The resident decode executor matches the reference executor through every
diagnostic boundary on the pinned small fixture, emits exactly the same greedy
token IDs and finish reason, and is safe to select for `atlas-cli chat`.

This phase closes the remaining Phase-8.2 correctness gate without raising a
global numerical tolerance.

## Completion evidence

The production path now dispatches Q/K/V projections and KV append using their
logical tensor widths instead of the byte size of scalar dimension buffers.
The Apple-Silicon Phase-8.3 suite passed, and resident chat produced the same
32-token completion as reference for `The capital of France is`; the resident
run reported `13.79` decode tok/s with token-only readback.

## Delivered work

- Kept `ExecutorMode::Reference` as the chat default and added explicit
  `--executor resident` selection after parity acceptance.
- Added deterministic diagnostic boundaries and first-failure reporting for
  attention/MLP projection and residual stages.
- Validated Llama RoPE, KV `[K|V][position][kv_head][dimension]` layout,
  grouped-query attention, projection dispatch, and GPU argmax against the
  reference executor.
- Added production-prefill parity coverage so normal one-command-buffer decode
  cannot diverge while the diagnostic trace passes.
- Preserved exact greedy-token comparison, token-only default readback, one
  command buffer per token, and no post-warmup allocations.

## Model fixture

Use the pinned small 30-layer fixture for all trace and 32-token greedy-parity
gates. Use a second prompt and a prompt long enough to exercise multiple KV
positions before promoting resident decode. Keep the larger-fixture 128-token
residency/throughput run as the Phase-8.2 performance confirmation.
