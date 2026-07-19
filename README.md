# Atlas Metal Inference Engine

## What we are building

Atlas is a Rust-first LLM inference engine for Apple Silicon. It will run a
decoder-only Transformer through native Metal compute kernels rather than a
cross-platform GPU layer.

The first practical goal is deliberately narrow: load a real small
Llama-compatible model, run prompt prefill and token-by-token decode on Metal,
and generate correct text with measurable performance.

Once that standard inference path is correct and stable, the engine will add
bounded local attention, recurrent working memory, persistent graph memory,
retrieval, memory fusion, and routing. Those Atlas memory features are not part
of the first MVP.

## Current status

Phases 0–2 are implemented: the workspace initializes native Metal,
compiles bootstrap kernels at runtime, caches compute pipelines, validates GPU
results against CPU references, and provides Metal/model-fixture CLI checks.
Atlas also has validated tensor metadata, FP16 conversion, SafeTensors weight
descriptors, classified pooled Metal allocations with telemetry, and a
correctness-first FP32 neural operator suite with distinct prefill/decode
projection paths. Model fixtures remain ignored by Git.

## Plan structure

- [Main architecture plan](docs/Atlas_Metal_Inference_Engine_Plan.md) —
  goals, architecture, technical choices, and the overall roadmap.
- [Shared implementation contract](docs/Atlas_Metal_Inference_Engine_Phase_Subplans.md)
  — model fixture policy, Hugging Face download commands, artifact rules, and
  cross-phase exit requirements.
- [Phase-plan index](docs/atlas-metal-phases/README.md) — one executable plan
  file for each phase from Metal bootstrap through the memory router.
- [GGUF conversion guide](docs/atlas-gguf-conversion.md) — native Q4_0/Q8_0
  conversion, progress telemetry, import, and verification.

Every phase has a concrete outcome, implementation scope, model test fixture,
and acceptance gate. A phase is not complete until its runnable test passes on
Apple Silicon and records its numerical or performance evidence.

## Test models

The initial plans use one model family so model-format and tokenizer changes do
not hide runtime regressions:

- Small correctness fixture:
  [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- Larger performance and memory fixture:
  [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)

Model files are test fixtures and must not be committed. The shared contract
contains the required `hf download --dry-run` and download commands, revision
pinning, and artifact-recording requirements.

## Phase 0 helper

Use the helper to download only the required SafeTensors/tokenizer files:

```zsh
scripts/download-models.sh
```

The script requires the Hugging Face CLI. Install it once if `hf` is not
already available:

```zsh
python3 -m pip install --user --upgrade huggingface_hub
```

The model is downloaded to `models/hf/SmolLM2-135M-Instruct/` and is ignored by
Git. The script first performs a Hugging Face dry run, then downloads only the
SafeTensors and tokenizer files needed by Atlas.

## Build, test, and use the CLI

Build the complete workspace:

```zsh
cargo check --workspace
```

Run all Rust tests:

```zsh
cargo test --workspace
```

The Phase 0 GPU integration test is also available directly:

```zsh
cargo test -p atlas-metal --test phase_00_bootstrap
```

Run the CLI to confirm that Atlas can create a Metal device and compile/cache
the Phase 0 kernels:

```zsh
cargo run -p atlas-cli -- metal-info
```

After downloading the small model, validate its configuration and SafeTensors
header without loading the model weights:

```zsh
cargo run -p atlas-cli -- fixture verify --model small
```

Talk to the model directly (omit `--prompt` for the REPL):

```zsh
cargo run -p atlas-cli -- chat --model small --prompt 'The capital of France is' --max-tokens 32
```

The supported product interface is currently the local CLI. HTTP serving is
intentionally deferred until the final API phase, after sampling, quantized
model loading, scheduling, and the memory runtime have stable CLI contracts.

## Implementation order

1. Bootstrap native Metal and validate simple kernels against CPU results.
2. Build tensors, allocation pools, and essential Transformer operators.
3. Load the small model and validate complete Metal inference.
4. Complete the local CLI with sampling, GGUF Q4_0/Q8_0 model loading,
   quantized resident inference, diagnostics, and runtime scheduling.
5. Add bounded local attention, then the Atlas memory system incrementally.
6. Add the loopback OpenAI-compatible server only after the local runtime and
   CLI contracts are complete.

For the complete sequence and exact gates, begin with
[Phase 0](docs/atlas-metal-phases/phase-00-metal-runtime-bootstrap.md).
