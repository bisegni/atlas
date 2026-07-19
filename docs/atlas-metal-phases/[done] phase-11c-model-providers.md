# Phase 11c: Model providers and Hugging Face discovery

## Outcome

`atlas-cli` discovers and downloads Atlas-compatible models through a provider
interface. Hugging Face is the first provider: its search results contain only
Llama-compatible repositories and artifacts that match an Atlas-supported
format, and a selected revision becomes a verified local manifest record.

## Work

- Define a provider boundary that resolves a search query into stable model
  candidates and a selected candidate into a pinned download plan. Keep the
  CLI, manifest, and model loader independent of Hugging Face API details so
  later providers implement the same boundary.
- Add `atlas-cli model search --provider huggingface <query>`. Return
  structured candidates with provider ID, repository ID, immutable revision,
  architecture, compatible format, artifact size, and any gated-model/auth
  requirement.
- Add provider-scoped login state with `atlas-cli provider login|status|logout
  huggingface`. Store validated credentials outside the repository and never
  write them to manifests or diagnostics. `HF_TOKEN` overrides the stored
  credential for automation. Provider selection is optional when exactly one
  provider is registered; otherwise `atlas-cli provider default <provider>`
  persists the user-local default and `--provider` remains an explicit
  override.
- Filter Hugging Face results to artifacts Atlas can load: a Llama-compatible
  architecture, required config/tokenizer files, and either the supported
  SafeTensors layouts using Atlas-supported FP32/FP16/BF16/I8 dtypes or the
  Phase 11a Q4_0/Q8_0 GGUF layout. Exclude
  unknown architectures, unsupported quantizations, incomplete repositories,
  and mixed tensor encodings rather than presenting them as downloadable.
- Add `atlas-cli model download <provider-model-id> --id <manifest-id>`.
  Resolve the revision before transfer, download into a staging directory,
  verify the declared files and hashes, then atomically register the model in
  `models/manifest.toml`. Preserve existing manifest entries and leave no
  partially registered model after download, validation, or authentication
  failures.
- Report source, pinned revision, format, downloaded bytes, hashes, and final
  manifest ID. Require explicit Hugging Face credentials only for gated/private
  artifacts and surface provider/network failures before a model is selected
  for inference.

## Exit gate

Fixture-backed provider tests prove that Hugging Face search returns supported
and excludes unsupported candidates, and that downloading a pinned public
candidate creates a manifest record accepted by `atlas-cli model verify`.
A gated-model test proves that missing credentials fail without modifying the
manifest or leaving a destination directory behind.

## Acceptance evidence

Phase 11c completed on 2026-07-19. The focused CLI/provider regression suite
passed, including default-provider selection and Hugging Face's string-valued
`gated = "manual"` metadata:

```text
cargo test -p atlas-cli
15 passed; 0 failed
```

A pinned public Hugging Face download was registered and verified locally:

```text
cargo run -p atlas-cli -- model download \
  'huggingface:HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac:safetensors-fp32' \
  --id phase11c-public-smollm2
{"event":"model_downloaded","model_id":"phase11c-public-smollm2",...}

cargo run -p atlas-cli -- model verify --model phase11c-public-smollm2
{"bytes":271165969,"model_id":"phase11c-public-smollm2","verified":true}
```

The gated rollback path was checked using the access-enabled
`meta-llama/Llama-3.2-1B-Instruct` revision, deliberately with `--no-auth`:

```text
cargo run -p atlas-cli -- model download \
  'huggingface:meta-llama/Llama-3.2-1B-Instruct@9213176726f574b556790deb65791e0c5aa438b6:safetensors-fp32' \
  --id phase11c-gated-negative --no-auth
Error: Hugging Face credentials are required for this gated/private artifact
```

After that expected failure, the manifest contained no
`phase11c-gated-negative` record and neither its destination nor staging
directory existed.
