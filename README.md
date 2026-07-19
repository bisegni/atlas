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

Planning only. No Rust crates, Metal kernels, model files, or server are in
this repository yet.

The first implementation task is Phase 0: create the Rust workspace, initialize
Metal, and prove a vector-add compute shader from Rust.

## Plan structure

- [Main architecture plan](docs/Atlas_Metal_Inference_Engine_Plan.md) —
  goals, architecture, technical choices, and the overall roadmap.
- [Shared implementation contract](docs/Atlas_Metal_Inference_Engine_Phase_Subplans.md)
  — model fixture policy, Hugging Face download commands, artifact rules, and
  cross-phase exit requirements.
- [Phase-plan index](docs/atlas-metal-phases/README.md) — one executable plan
  file for each phase from Metal bootstrap through the memory router.

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

## Implementation order

1. Bootstrap native Metal and validate simple kernels against CPU results.
2. Build tensors, allocation pools, and essential Transformer operators.
3. Load the small model and validate complete Metal inference.
4. Add KV cache, quantization, executors, sampling, runtime scheduling, and
   API serving.
5. Add bounded local attention, then the Atlas memory system incrementally.

For the complete sequence and exact gates, begin with
[Phase 0](docs/atlas-metal-phases/phase-00-metal-runtime-bootstrap.md).
