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

Phase 12a-pre accepts the text-chat foundation through deterministic Atlas
token evidence, focused numerical primitive oracles, fixture validation, and
real multi-turn Resident execution. Exact end-to-end token parity against an
external llama.cpp build is deliberately deferred to later optimization and
parity work; it is not claimed by this phase. This keeps the foundation gate
focused on Atlas's supported production contract while preserving exact-token
comparison as a future promotion gate for kernel changes.

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
  --prompt 'The capital of France is' --max-new-tokens 8 --greedy

cargo run -p atlas-cli -- chat --model gemma4-e2b-q4_0 \
  --prompt 'Explain the history and importance of Paris.' \
  --max-tokens 128
```

`generate` selects `ExecutorMode::Resident` unconditionally and sends the
supplied text as a raw completion prompt. `chat` renders the embedded Gemma 4
turn protocol and also executes through `ExecutorMode::Resident`. With
`--prompt`, chat streams one turn; without it, chat starts a multi-turn REPL
with `/help`, `/reset`, and `/quit`. The REPL reuses one loaded executor,
replays canonical visible history, filters thought-channel text unless
`--show-thoughts` is requested, and summarizes older complete pairs before the
4,096-token context limit is exceeded. Raw thoughts are not retained in chat
history or the performance log.

The phase passes when the resident commands report deterministic Gemma-derived
token IDs and coherent text, non-zero resident bytes, one-time weight upload,
warm-turn reuse, bounded readback, and no Reference fallback.

## Accepted evidence

Fixture: `gemma-4-E2B_q4_0-it.gguf`, 3,349,516,256 bytes, SHA-256
`fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`.

| Gate | Result |
| --- | --- |
| Portable workspace tests | Passed; focused Gemma formatter, reserved-token, thought filtering, compaction, Q6_K, PLE, shared-KV, and soft-cap coverage passed. |
| Fixture header and Q6_K lookup oracle | Passed with the official local fixture; the ignored Metal Q6_K lookup matched the independent llama.cpp-layout CPU oracle. |
| Model inspect and verify | Passed for manifest ID `gemma4-e2b-q4_0` and the recorded checksum. |
| 128-token one-shot chat | `Resident`; 128 tokens; 3,489,602,512 resident bytes; 3,333,699,724 cold upload bytes; 576 readback bytes; 144 command buffers; no Reference fallback. |
| Two-turn context | `Resident`; the second response recalled `zephyr`; first turn uploaded 3,333,699,724 bytes and the warm second turn uploaded 0 bytes; stable 3,489,602,512 resident bytes; no leaked control tokens. |
| Raw eight-token generation | Prompt IDs `[669, 5279, 529, 7001, 563]`; generated IDs `[7001, 563, 7001, 563, 7001, 563, 7001, 7001]`; finite top logits; 12 command buffers; `Resident`; no fallback. |

The interactive and raw Metal results were supplied from the user-run Apple
Silicon environment. Exact external-runtime token parity is deferred and must
be reintroduced as an explicit gate before promoting future Gemma kernel
optimizations.
