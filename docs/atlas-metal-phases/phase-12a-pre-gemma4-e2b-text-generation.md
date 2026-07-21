# Phase 12a-pre: Gemma 4 E2B resident text-generation foundation

## Outcome

Atlas loads the downloaded Gemma 4 E2B QAT Q4_0 GGUF artifact and produces
correct text-only greedy generation through `ExecutorMode::Resident` on Apple
Silicon. Image and audio inputs, the separate multimodal projector, and other
Gemma 4 sizes remain outside this phase.

## Model fixture

The local ignored fixture is the official Google release:

```text
models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
```

It is a GGUF v3 `gemma4` model. Its release label is Q4_0, but the actual
tensor table contains Q4_0 projection weights, F32/F16 auxiliary tensors, and
Q6_K embedding tables. Atlas must inspect the tensor table and support the
formats it finds; it must not infer the complete wire contract from the
release label.

## Work

- Preserve typed GGUF metadata and the embedded tokenizer arrays needed by
  Gemma 4. Build the tokenizer and text chat template from the artifact; do
  not require an adjacent `config.json` or `tokenizer.json`.
- Add a distinct Gemma 4 E2B model family and configuration path. Continue to
  reject non-Llama architectures on the Llama path, and reject Gemma 4
  variants other than the declared E2B text-only layout with a clear error.
- Add checked Q6_K loading and resident embedding lookup alongside the
  existing Q4_0 projection path. Packed GGUF execution remains resident-only;
  a Gemma load or resident failure must not retry the user-facing command with
  `ExecutorMode::Reference`.
- Implement E2B text decoding from the upstream Gemma 4 equations: scaled
  token embeddings, per-layer embedding identity/context projection and
  injection, input gate/projection, Q/K normalization, residual norms/scales,
  per-layer attention dimensions, shared-KV state, sliding/full attention,
  their RoPE parameters, and final-logit soft-capping.
- Extend the KV cache and resident command encoding to make the selected
  attention layout observable. Keep weights, KV state, activations, and token
  selection resident after warm-up; report uploads, resident bytes, command
  buffers, readback bytes, and timings.
- Register the fixture in the local manifest only after `model inspect`,
  `model verify`, `generate`, and `chat` can resolve its existing filename.
  `chat` applies the embedded Gemma 4 template; raw `generate` retains an
  explicit prompt contract.

## Acceptance

First build a fixed text-only oracle from the upstream Gemma 4 reference
implementation. Record its revision, prompts, input IDs, generated IDs, EOS
behavior, and logits diagnostics under ignored `artifacts/phase-12a-pre/`.
Use the same Q4 GGUF baseline only after validating its greedy output against
that reference; do not treat an unverified third-party runtime as the oracle.

Portable tests cover GGUF metadata arrays, Q6_K decoding, embedded tokenizer
round trips, tensor-name/shape validation, PLE, each attention type, shared
KV state, and soft-capping. The real fixture test verifies that the local
artifact opens as `gemma4` and contains both Q4_0 and Q6_K tensors.

On Apple Silicon, run:

```zsh
ATLAS_GEMMA4_GGUF="$PWD/models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf" \
cargo test -p atlas-core --test phase_11a_gguf \
gemma4_e2b_fixture_header_is_read_when_available -- --exact

cargo run -p atlas-cli -- metal-info
cargo run -p atlas-cli -- generate --model gemma4-e2b-q4_0 \
  --prompt 'The capital of France is' --max-tokens 8 --executor resident
```

The phase passes only when the resident command reports the expected
Gemma-derived token IDs/text, non-zero resident bytes, one-time weight upload,
and no reference fallback. Record hardware/OS, artifact SHA256, command line,
actual generated-token count, resident/upload/readback/command-buffer metrics,
and timing. Until that output exists, Gemma 4 generation and GPU residency
remain unverified.
