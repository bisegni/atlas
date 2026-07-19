# Phase 8.3: Resident decode parity remediation

## Outcome

The resident decode executor matches the reference executor through every
diagnostic boundary on the pinned small fixture, emits exactly the same greedy
token IDs and finish reason, and is safe to select for `atlas-cli chat`.

This phase closes the remaining Phase-8.2 correctness gate. It does not accept
a global tolerance increase as a substitute for finding and bounding the
source of a difference.

## Current evidence

`atlas-cli phase_08b_decode --trace-stages` compares the tokenized prompt
before it generates a token. The current small-fixture trace reaches prompt
token 1, layer 3, `mlp_residual`, where the first observed difference is
`1.526e-5` at element 247. Earlier recorded stages for that token are within
the current `1e-5` tolerance. This identifies the MLP down-projection/residual
boundary as the next fault-isolation target; it does not establish that
loosening the threshold is correct.

## Work

- Keep `chat` on `ExecutorMode::Reference` throughout remediation. Add an
  explicit resident-chat selection only after this phase's acceptance suite
  passes; do not make resident execution the default based on a benchmark
  alone.
- Extend diagnostic-only tracing with the unobserved operations immediately
  before each failing residual boundary: attention output projection,
  MLP down projection, and the operands/result of the residual add. Complete
  and read back only these trace commands; preserve the normal one-command-
  buffer resident decode path.
- Make stage comparison deterministic for length mismatches, NaN/infinity,
  maximum absolute error, and first failing index/value. Report prompt-token
  index, layer, stage, and the numeric values from the first failure.
- Validate Llama RoPE, KV append/layout, grouped-query decode attention, GPU
  argmax, and resident FP32 matrix-vector projection against focused reference
  oracles. Cover position zero and non-zero positions, multiple KV heads, and
  more than one cached token.
- Repair the first invalid operation rather than compensating downstream.
  Check Llama half-split RoPE indexing, KV `[K|V][position][kv_head][dimension]`
  addressing, grouped-query head mapping, matrix-vector accumulation order,
  buffer reuse/aliasing, and token/position state sequencing.
- Use `abs_error <= 1e-5` for elementwise kernels. Define any wider tolerance
  separately for reduction-heavy projection or attention stages, document its
  FP32 error bound and input shapes, and keep exact greedy token-ID comparison
  regardless of diagnostic tolerance.
- Run the trace over the first token at position zero and the full tokenized
  prefix of `The capital of France is`. The prompt-prefix test returns the
  earliest failure during investigation and must return no failure at phase
  completion.

## Model fixture

Use the pinned small 30-layer fixture for all trace and 32-token greedy-parity
gates. Use a second prompt and a prompt long enough to exercise multiple KV
positions before promoting resident decode. Keep the larger-fixture 128-token
residency/throughput run as the Phase-8.2 performance confirmation.

## Exit gate

- Focused resident micro-kernel tests pass on Apple Silicon.
- The position-zero and prompt-prefix stage traces report no divergence under
  their documented per-stage FP32 tolerances.
- Resident and reference generation produce exactly the same 32 token IDs and
  finish reason for each required prompt; final logits are retained only as a
  diagnostic and do not replace token-ID parity.
- The resident run preserves one command buffer per decode token, token-only
  default readback, and zero post-warmup model uploads and allocations.
- The resident-chat selection is exercised by the same exact-parity suite;
  `atlas-cli chat` may select it only after all preceding gates pass.
