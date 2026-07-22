# Phase 12a-perf: Gemma 4 Resident performance remediation

## Outcome

Atlas materially improves Gemma 4 E2B Q4_0 prefill and decode throughput on
Apple Silicon while preserving the completed Phase 12a-pre text, streaming,
multi-turn, thought-filtering, context-compaction, and Resident-only contracts.
The production CLI remains normal `chat`; this phase does not add public
benchmark or profiling modes.

## Implementation status

The first performance-remediation pass is implemented. It preserves the
Resident-only execution contract and substantially improves the original
short-prompt result, but the phase remains open because the required long-run
throughput gates have not passed yet.

### Simple explanation

Atlas was doing too much repeated setup around every prompt and generated
token. The implementation reduces that overhead in four ways:

1. Prompt tokens are submitted together in one Resident command buffer instead
   of submitting one command buffer per prompt token.
2. Prompt tokens that are not the final token skip the final vocabulary
   projection and token selection because those results would be discarded.
3. The Q4_0 and Q6_K matrix-vector kernels process several output rows together
   with Metal SIMD groups, which makes better use of the GPU.
4. Small dimension, token, position, and RoPE buffers are reused instead of
   being allocated repeatedly during generation.

For the canonical `hi` chat, these changes preserve the exact visible response
and improve the original approximately 15 tok/s prefill and 17 tok/s decode to
approximately 39 tok/s prefill and 40 tok/s decode on the measured Apple M2 Max.
Warm runs upload no weights, use one prefill command buffer, and keep Resident
memory unchanged at 3,489,602,512 bytes.

### Detailed implementation

#### Resident prompt command batching

`Gemma4PrefillPlan` bounds prompt chunks to 128 tokens and validates that the
prompt fits the configured context. The executor uploads each chunk's token
IDs, positions, and precomputed full/sliding-window RoPE values once. It then
encodes the dependent token operations in dispatch order inside one Resident
command buffer. GPU scratch buffers and the KV cache stay Resident throughout
the chunk, and only the final selected token is read back.

This changes the canonical ten-token prompt from ten prefill command buffers to
one. The selected path and effective chunk size are observable as
`prefill_path: "resident_chunked_command"` and `prefill_chunk_size` in both the
terminal metrics and `artifacts/chat-performance.jsonl`.

This is command-buffer batching, not yet a layer-major matrix-matrix prefill
implementation. Layers still process the prompt tokens in dependency order
inside the command buffer. The longer-prompt measurements below show why a
future layer-major batched projection and attention path is still needed.

#### Avoiding discarded prompt output work

Only the last prompt token needs to select the first generated token. Earlier
prompt tokens now stop after updating their hidden state and KV entries. They
skip final normalization, the large Q6_K vocabulary projection, logit softcap,
and argmax. The final prompt token and every generated token retain the complete
output path, so greedy generation and stopping behavior remain unchanged.

#### Q4_0 projection kernel

The production Q4_0 projection uses `matvec_q4_0_16row`. One 128-thread
threadgroup contains four SIMD groups and produces sixteen output rows. Within
each SIMD group, four groups of eight lanes independently accumulate four rows;
each lane consumes four input elements from each packed Q4_0 block. XOR SIMD
reductions combine the eight partial sums without changing the accepted greedy
output.

A 32-row/256-thread variant was measured and rejected because it reduced
short-prompt prefill to approximately 17.85 tok/s. It is not part of the final
implementation.

#### Q6_K projection kernel

The production Q6_K projection uses `matvec_q6_k_8row`. Each half-SIMD lane
group owns an output row, allowing eight rows to be processed by one
128-thread threadgroup. The packed low bits, high bits, group scales, and block
scale remain in GGUF form; Atlas does not create a full dequantized weight
cache. Experimental larger Q6_K threadgroups were rejected when they changed
the output or reduced throughput.

#### Persistent control and decode buffers

Immutable GPU buffers for hidden widths, attention widths, feed-forward widths,
PLE offsets, and related dimensions are allocated when the executor is created
and reused by every projection. Decode also reuses token, position, and
full/sliding-window RoPE buffers. This removes hundreds of small Metal buffer
allocations from a generation while leaving model weights and working state
Resident.

#### Metrics and correctness boundaries

The append-only chat performance record now includes prefill/decode throughput,
host time, prompt/generated token counts, separate prefill/decode command-buffer
counts, upload/readback bytes, Resident bytes, finish reason, selected prefill
path, and chunk size. Token ID arrays remain excluded from user-facing metrics.

The implementation was validated with the full portable workspace suite, the
Gemma GGUF fixture-header test, and the ignored Metal Q6_K row test against the
llama.cpp-derived CPU oracle. The canonical short response remains exactly:

```text
Hello! How can I help you today? 😊
```

Resident errors are still surfaced directly; there is no Reference fallback.

## Current measured evidence

The following measurements were collected on Apple M2 Max with the release
build and the Phase 12a Gemma fixture:

| Workload | State | Prefill | Decode | Prompt / generated tokens | Prefill / decode command buffers |
| --- | --- | ---: | ---: | ---: | ---: |
| Canonical `hi` | cold | 37.07 tok/s | 39.87 tok/s | 10 / 11 | 1 / 10 |
| Canonical `hi`, four-run warm median | warm | about 39.10 tok/s | about 40.03 tok/s | 10 / 11 | 1 / 10 |
| Fixed longer prompt | warm | 41.84 tok/s | 27.02 tok/s | 59 / 104 | 1 / 103 |

The warm canonical runs report zero weight uploads, 44 readback bytes, and
3,489,602,512 Resident bytes. The longer run performs 103 post-prefill decode
steps and therefore exercises the required minimum-64-step workload, but its
27.02 tok/s decode result is below the 40 tok/s gate. Its 41.84 tok/s prefill is
also below the 50 tok/s gate.

Consequently, the implementation is present in the working tree as a measured
improvement, but this phase must not be renamed `[done]` yet. Completion requires
layer-major batched prefill and/or additional measured attention and decode
kernel work sufficient to pass both long-workload thresholds over five warm
runs.

## Next implementation strategies

The following work is intentionally deferred to the next Gemma performance
iteration. Each strategy must be measured on the normal release `chat` path and
must preserve exact greedy token output before promotion.

1. **Implement layer-major batched prefill.** Store prompt activations as a
   `[tokens, hidden]` matrix and process a whole chunk through each transformer
   layer. Replace repeated Q4_0 matrix-vector projections with packed Q4_0
   matrix-matrix kernels. This is the most direct route from the current
   command-buffer batching to actual GPU compute batching and is the primary
   candidate for reaching the 50 tok/s long-prompt gate.
2. **Add batched causal and sliding-window attention.** Compute queries, keys,
   and values for the prompt chunk together, apply the correct full/sliding
   causal mask across chunk boundaries, and write the resulting keys and values
   to the shared KV cache in one batched operation. Fuse score scaling, masking,
   softmax, and value accumulation only when a focused oracle test proves the
   same numerical and token-selection result.
3. **Profile long-context decode separately.** The short decode reaches about
   40 tok/s while the 103-step workload reaches only 27.02 tok/s. Capture
   per-kernel GPU time as context grows to determine whether attention scans,
   KV-cache layout, projection traffic, synchronization, or token selection is
   responsible for the decline. Optimize the measured dominant stage rather
   than assuming the Q4_0 projections remain dominant.
4. **Improve KV-cache access locality.** Evaluate a layout that lets one SIMD
   group read contiguous key/value elements for the active attention window.
   Preserve shared-KV ownership and bounded sliding-window behavior. Compare
   bytes read and GPU duration at positions 1, 32, 64, and 128.
5. **Evaluate narrow, correctness-gated fusion.** Candidate fusions include
   RMS normalization plus projection preparation, gate plus up projections,
   and attention output plus residual addition. Keep a fusion only when kernel
   timing improves and the independent exact-token oracle remains unchanged;
   the rejected dual gate/up experiment demonstrates that throughput alone is
   not sufficient.
6. **Move remaining per-token setup to reusable GPU work.** Generate or update
   RoPE values without allocating host vectors, retain all scalar/control
   buffers, and remove avoidable CPU-to-GPU writes and encoder synchronization.
   Confirm the benefit with host time and GPU time reported separately.
7. **Tune threadgroup shapes per projection size.** The accepted Q4_0 16-row
   and Q6_K 8-row kernels are better than the original kernels, but one shape
   may not be optimal for every attention, feed-forward, and vocabulary
   projection. Select shapes from measured matrix dimensions while keeping a
   small fixed pipeline set and exact accumulation tests.
8. **Create a dedicated follow-up phase after profiling.** Record the fixed
   prompt, five-run warm baseline, hardware and OS, fixture hash, external
   oracle revision, and ranked kernel costs before implementation. The new
   phase should own the deferred throughput targets while retaining this
   phase's Resident, memory, readback, and correctness invariants.

## Fixture and baseline

Use the official ignored fixture from Phase 12a-pre:

```text
models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
```

The accepted release baseline on the user-run Apple Silicon environment is:

| Metric | Baseline |
| --- | ---: |
| Prompt | `hi` |
| Prefill throughput | 15.27 tok/s |
| Decode throughput | 17.15 tok/s |
| Prefill command buffers | 10 |
| Decode command buffers | 10 |
| Resident bytes | 3,489,602,512 |
| Readback bytes | 80 |
| Executor | `Resident` |

The observed Ollama Gemma 4 E4B QAT result, 104.83 prefill tok/s and 74.99
decode tok/s, is directional context only. It is not an acceptance oracle
because it uses a different model size, runtime, prompt template, and generated
workload.

Measure optimized Atlas builds only:

```zsh
cargo run --release -p atlas-cli -- chat \
  --model gemma4-e2b-q4_0 \
  --max-tokens 128
```

For a warm, like-for-like short-prompt sample in one process, enter `hi`, then
`/reset`, then `hi` again. Treat the first turn as pipeline and weight warm-up;
use the second record from `artifacts/chat-performance.jsonl` as the measured
sample. Record at least five measured repetitions and report median and range.

Add a fixed longer prompt artifact before implementation so prefill changes are
not accepted from the noisy ten-token short prompt alone. Keep prompt text,
response budget, fixture, process lifetime, warm-up, and executor identical
across baseline and candidate runs.

## Work

1. Add Gemma-focused performance accounting and regression coverage.
   - Keep `prefill_tok_s`, `decode_tok_s`, prompt/generated token counts,
     prefill/decode command-buffer counts, upload/readback bytes, Resident
     bytes, finish reason, and host timing in the append-only performance log.
   - Keep token IDs out of user-facing metrics. Exact IDs belong in ignored
     acceptance artifacts and focused tests.
   - Separate cold upload, prefill, and post-prefill decode accounting. A warm
     turn reports zero weight-upload bytes.

2. Profile the real release `chat` path before changing execution.
   - Capture command-buffer scheduling and GPU timing for the short and longer
     workloads under `artifacts/phase-12a-perf/`.
   - Rank measured costs across PLE lookup/projection, attention, Q4_0
     projections, normalization, MLP, final Q6_K projection, token selection,
     synchronization, and readback.
   - Do not infer the dominant cost solely from command-buffer count.

3. Implement true Resident batched prefill as the first optimization target.
   - Encode the full prompt, or bounded prompt chunks, without submitting one
     command buffer per prompt token.
   - Preserve causal and sliding/full-attention semantics, shared-KV ownership,
     RoPE positions, PLE injection, and the final selected-token boundary.
   - Keep weights, KV state, activations, and token selection GPU-resident. Do
     not introduce a CPU prefill fallback or full-weight dequantization cache.
   - Make the selected prefill path and its batch/chunk size observable in
     metrics and tests.

4. Optimize decode only from measured kernel evidence.
   - Retain at most one command buffer per generated token unless a proven
     multi-token strategy preserves greedy semantics.
   - Reduce dispatch, scheduling, or memory-traffic cost through focused fusion
     or layout changes. Do not fuse unrelated stages without before/after
     kernel evidence.
   - Keep default readback limited to token selection and required validity
     checks. Diagnostic logits and stage traces remain opt-in.

5. Preserve correctness and failure behavior.
   - `chat` stops immediately on Gemma end-of-turn or EOS and never continues
     decoding hidden control tokens to the maximum-token limit.
   - Resident failures surface directly and never retry through Reference.
   - One-shot output, two-turn context recall, thought filtering, `/reset`, and
     context compaction retain their Phase 12a-pre behavior.

## Tests

Add focused portable tests for:

- batched/chunked prefill planning, position accounting, and context bounds;
- causal plus sliding/full attention masks across chunk boundaries;
- shared-KV updates and PLE indexing for multi-token input;
- batched prefill versus the existing single-token oracle on deterministic
  small tensors;
- prompt and generated token IDs, finish reason, visible text, and control-token
  stopping remaining unchanged;
- warm upload accounting, selected-token readback, command-buffer accounting,
  and Resident-only error propagation;
- performance metrics remaining observational and excluding token ID arrays.

Fixture-backed Apple-Silicon tests compare the optimized path with the retained
diagnostic single-token prefill path before removing or demoting that oracle.
Before promoting changed Gemma kernels, pin an independent external runtime
revision and require exact greedy prompt and generated token parity for a fixed
short canonical chat. Semantic similarity is not sufficient for a kernel
promotion gate.

## Acceptance

Run the narrow tests first, followed by:

```zsh
cargo fmt --all --check
cargo test --workspace

ATLAS_GEMMA4_GGUF="$PWD/models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf" \
cargo test -p atlas-core --test phase_11a_gguf \
gemma4_e2b_fixture_header_is_read_when_available -- --exact

ATLAS_GEMMA4_GGUF="$PWD/models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf" \
cargo test -p atlas-metal --test phase_00_bootstrap \
gemma4_q6_k_ple_row_matches_llama_cpp_oracle -- --ignored --exact
```

The phase passes on the same Apple Silicon host and fixture when all of the
following are true:

- median warm prefill reaches at least 50 tok/s on the fixed longer prompt;
- median warm decode reaches at least 40 tok/s on a fixed workload that
  performs at least 64 post-prefill decode steps;
- prefill no longer submits one command buffer per prompt token and the exact
  measured command-buffer reduction is recorded;
- the optimized path preserves exact accepted prompt/generated IDs, finish
  reason, visible output, multi-turn recall, and control-token stopping;
- warm turns report zero weight upload, selected-token default readback remains
  bounded, Resident bytes do not exceed 3,489,602,512 without a documented and
  approved memory-for-speed tradeoff, and no Reference fallback occurs;
- five-run raw records, median/range summary, hardware/OS, fixture SHA-256,
  commands, external-oracle revision, and kernel/profile evidence are stored
  under ignored `artifacts/phase-12a-perf/`.

Mark completion with the `[done]` filename convention and update the phase
index only after the Apple-Silicon acceptance evidence passes.

## External software

- Ollama is an optional directional comparator only; its E4B measurement does
  not gate this phase.
- A pinned independent Gemma-capable runtime is required only for the final
  exact-token kernel-promotion check. Record its source and revision with the
  acceptance artifact.
