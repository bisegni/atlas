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
- Filter Hugging Face results to artifacts Atlas can load: a Llama-compatible
  architecture, required config/tokenizer files, and either the supported
  FP32 SafeTensors layout or the Phase 11a Q4_0/Q8_0 GGUF layout. Exclude
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
