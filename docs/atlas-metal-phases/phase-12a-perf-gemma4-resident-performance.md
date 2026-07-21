# Phase 12a-perf: Gemma 4 Resident performance remediation

## Outcome

Atlas materially improves Gemma 4 E2B Q4_0 prefill and decode throughput on
Apple Silicon while preserving the completed Phase 12a-pre text, streaming,
multi-turn, thought-filtering, context-compaction, and Resident-only contracts.
The production CLI remains normal `chat`; this phase does not add public
benchmark or profiling modes.

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
