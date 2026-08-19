# Model browse + quantize: work todo

Status source for the interactive model browser and the Gemma 4 safetensors
quantization path. Everything here follows the findings recorded on
2026-08-18; older records remain in git history.

## Verified context

- `model quantize` does not exist in the current CLI (removed in commit
  6a7dc68; `docs/atlas-engineering.md` section 4 still describes it and is
  stale). The old implementation was recovered from git and is available at
  `/var/folders/w7/x_1k24352_l2zy980pn3mxwm0000gp/T/opencode/old_main.rs`.
- Current CLI offers only non-interactive `model search <query>` and
  `model download huggingface:<repo> --id <id>` (GGUF-only, then manual
  manifest registration).
- The official Gemma 4 GGUF contract was probed empirically and saved at
  `/tmp/gguf-contract.json` (541 tensors, 49 KV entries, exact `gemma4.*`
  metadata key set, tokenizer.ggml.* arrays, 15-layer/20-layer ffn split,
  shared_kv_layers 20, contiguous kv tensors, `rope_freqs.weight` F32 [256]
  with learned values). `crates/atlas-model/src/lib.rs` `Gemma4E2bConfig`
  (`from_gguf` ~269) is the authoritative loader contract.
- `google/gemma-4-E2B-it` safetensors repo is public; parsed header saved at
  `/var/folders/w7/x_1k24352_l2zy980pn3mxwm0000gp/T/opencode/safetensors-contract.json`.
  - 2011 tensors, BF16, 8-byte Xet prefix before the safetensors header.
  - Text weights are under `model.language_model.` (600 tensors, 35 layers);
    vision/audio mmproj weights under `model.vision_tower.` are skipped.
  - Per-layer mapping to GGUF is direct: `input_layernorm` -> `attn_norm`,
    `q/k/v/o_proj`, `q_norm`/`k_norm`, `gate/up/down_proj`,
    `per_layer_input_gate` -> `inp_gate`, `layer_scalar` -> `layer_output_scale`,
    `post_attention_layernorm`, `post_feedforward_layernorm`,
    `post_per_layer_input_norm`, `per_layer_projection` -> `proj`.
  - Top-level: `embed_tokens.weight` -> `token_embd`, `norm.weight` ->
    `output_norm`, `embed_tokens_per_layer.weight` ->
    `per_layer_token_embd`, `per_layer_model_projection.weight` ->
    `per_layer_model_proj`, `per_layer_projection_norm.weight` ->
    `per_layer_proj_norm` (order/naming to be confirmed against the probed
    contract when writing the converter).
  - **`rope_freqs` is NOT in the safetensors repo**; the learned F32 [256]
    values must be fetched from an official GGUF artifact (a ~1 KB range
    request).
- `gemma4_tokenizer` (`lib.rs` ~126) builds a Unigram tokenizer from
  `tokenizer.ggml.tokens` + `tokenizer.ggml.scores` only (no merges/token_type
  required at load time). Tokenizer embedding therefore needs the SPM pieces
  and scores, expected inside the repo's 32 MB `tokenizer.json`
  (`sp_model` base64); verify on first download.
- atlas-core has: safetensors readers (`read_safetensors_descriptors`,
  `read_safetensors_tensor_f32`), `GgufWriter` (string metadata only),
  `GgufMetadataArray` (Strings/F32/I32/U32/U64/I64/Bool), Q4_0/Q8_0 host
  quantizers. Atlas-metal still has `MetalRuntime::quantize_gguf` (Q4_0/Q8_0).
  **No Q6_K quantizer exists** and the official GGUF stores the vocab
  embeddings (`token_embd`, `per_layer_token_embd`) as Q6_K.

## Task list

- [x] Recover old `model quantize` implementation from git history
- [x] Probe official Gemma 4 GGUF contract (metadata + tensor layout)
- [x] Probe Gemma 4 safetensors repo (header, tensor names, dtypes, Xet prefix)
- [x] Confirm `rope_freqs` is absent from safetensors and must come from GGUF
- [ ] atlas-core: extend `GgufWriter` to serialize typed/array metadata
      (Strings/F32/I32/U32/U64/I64/Bool) with round-trip unit tests
- [ ] atlas-core: add Q6_K block quantizer (host, deterministic) with parity
      test against a reference rounding
- [ ] atlas-cli: `model quantize` command (Gemma-4-E2B safetensors -> GGUF):
      - safetensors download with resume (Range), Xet-prefix handling
      - tensor mapping incl. `model.language_model.` prefix and per-layer set
      - Q6_K for `token_embd`/`per_layer_token_embd`, Q8_0/Q4_0 elsewhere
      - `rope_freqs` fetched from official GGUF (range request)
      - `gemma4.*` metadata arrays derived from `config.json`
      - tokenizer pieces+scores from `tokenizer.json` embedded `sp_model`
      - `register_gguf_manifest` registration, atomic manifest append
- [ ] atlas-cli: safetensors download support in providers (candidate
      detection: fetch `config.json`, look for a quantizable architecture)
- [ ] atlas-cli: interactive `model browse` REPL: paged candidate list
      (GGUF + quantizable safetensors), select -> download GGUF,
      download+quantize, or verify
- [ ] Tests: unit tests for writer/quantizer/converter; `cargo fmt`
- [ ] Docs: README.md model section and `docs/atlas-engineering.md` section 4
      (currently describes the non-existent quantize command)

## Open design points (confirm before finishing)

1. `rope_freqs` sourcing: automatic 1 KB range fetch from the official GGUF
   artifact during quantize, or require the user to pre-download a GGUF?
   (Fetch is the only sane self-contained option.)
2. Tokenizer vocabulary: parse the embedded `sp_model` from the 32 MB
   `tokenizer.json`, or require an official GGUF download for its
   `tokenizer.ggml.tokens`/`scores` arrays?
3. Quantize backend: Q6_K has no Metal path yet, so `model quantize` will be
   host-side for Gemma 4. Keep old `--quantizer auto|cpu|gpu` semantics with
   `gpu` rejected for Q6_K?
4. Safetensors download size: the 10.2 GB `model.safetensors` needs a
   long-running, resumable download; confirm streaming progress + resume
   in the browse flow.