# Gemma 4 model-support expansion: work todo

Status source for loading Gemma 4 models beyond the single E2B Q4_0 QAT
artifact. Findings and strategy recorded 2026-08-18; older records remain in
git history. The interactive browse/quantize feature has its own tracking
file: `docs/model-browse-quantize-todo.md`.

## Verified context

- Today Atlas accepts exactly one Gemma 4 artifact: `gguf-gemma4-q4_0`
  (`google/gemma-4-E2B-it-qat-q4_0-gguf`). Hard gates:
  - `crates/atlas-cli/src/providers.rs:409` ("Atlas accepts only Gemma 4 E2B
    Q4_0 GGUF")
  - `crates/atlas-cli/src/main.rs:125`-133 (E2B name check + format check)
  - `crates/atlas-model/src/gemma4_executor.rs` `weight()` requires exactly
    Q4_0 matmuls, Q6_K vocab, F32/F16 small tables (e.g. ~3616)
- The Gemma 4 family (Hugging Face API, 2026-08): `gemma-4-E2B`, `E4B`,
  `12B` (tag `gemma4_unified`, any-to-any), `26B-A4B` (MoE, active-4B),
  `31B` — each with `-it` instruct variants. There is no "E2C".
- Official QAT GGUF artifacts today: only `gemma-4-E2B-it-qat-q4_0-gguf` and
  `gemma-4-26B-A4B-it-qat-q4_0-gguf`. Community GGUFs exist for 12B/31B/E4B
  (unsloth etc.) and safetensors repos for all.
- The loader itself is metadata-driven: `Gemma4E2bConfig` (`crates/atlas-model/
  src/lib.rs` ~269) reads block_count, feed_forward_length array, head counts,
  rope freqs/thetas, KV lengths, sliding window/pattern, shared_kv_layers,
  PLE dims, logit softcap from GGUF metadata. Structural assumptions that may
  not hold for other family members: contiguous KV layer layout, sliding
  pattern length == block_count, per-layer input embeddings, tensor-name
  conventions for the non-QAT GGUFs.
- `gemma4_tokenizer` accepts `tokenizer.ggml.model` `"llama" | "gemma4"` and
  builds a Unigram tokenizer from tokens+scores, so tokenizer support is not
  family-dependent as long as the GGUF embeds the vocab.

## Strategy

1. Metadata-driven acceptance: accept any `general.architecture == "gemma4"`
   GGUF whose metadata parses and whose tensor inventory matches the expected
   layout; remove the E2B-name and format hardcoding from the CLI gate.
2. Streaming on-load re-quantization: accept Q8_0/F16/(Q6_K) matmul weights
   and convert block-wise to the resident format with bounded memory (never
   materialize a full f32 model). Reuses the conversion machinery being built
   for `model quantize` (`docs/model-browse-quantize-todo.md`).
3. Safetensors path: generalize the planned converter via `config.json` so
   E4B/12B/31B (no official GGUF) become loadable products of
   `model quantize`.
4. Real executor adaptation (large): `26B-A4B` is MoE (router + experts) and
   `12B` is a different unified architecture — both need new kernels, out of
   scope for load-time work.
5. Layout diagnoser: on rejection, report exactly which structural assumption
   failed (shared KV, sliding pattern, PLE dims, tensor names/types) instead
   of a generic "unsupported".

## Task list

- [ ] Generalize the CLI/provider gate: accept any gemma4 GGUF passing
      metadata parse + tensor-inventory validation (drop E2B name check;
      keep Q4_0 matmul + Q6_K vocab type requirements unless re-quantization
      exists)
- [ ] Add a gemma4 layout diagnoser: collect and report all mismatches
      (missing/unknown tensors, wrong tensor types, metadata key gaps) on
      load failure
- [ ] Streaming on-load conversion: read arbitrary GGUF tensor types ->
      dequantize -> re-quantize to resident format, block-wise, bounded
      memory; expose chosen format in diagnostics/residency metrics
- [ ] Verify non-E2B family members against the loader: probe
      `gemma-4-E4B-it` and `gemma-4-31B-it` layouts (safetensors + any
      community GGUF) and record which structural assumptions hold
- [ ] Safetensors conversion for 12B/E4B/31B via generalized `model quantize`
      (tensor mapping driven by `config.json`)
- [ ] Assess 26B-A4B MoE and 12B unified executor work; record in
      `next-improvements` if deferred
- [ ] Tests: per-family layout acceptance tests, conversion parity tests
      against the reference executor
- [ ] Docs: README.md supported-models table; `docs/atlas-engineering.md`
      model-support section update

## Open design points (confirm before finishing)

1. Acceptance strictness: should the loader accept a gemma4 GGUF that parses
   but is missing optional structural elements (e.g. no sliding window), or
   reject with a diagnoser report until the executor handles it?
2. On-load re-quantization memory/CPU budget: per-tensor streaming is bounded
   (~largest tensor a few GB) but adds load-time cost; is
   `model quantize` the preferred path for non-Q4_0 weights, with on-load
   conversion only as fallback?
3. Family support priority: E2B-family first (strategy 1-2), or target a
   specific model (31B is dense and closest to current executor)?
