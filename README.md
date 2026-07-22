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

Gemma 4 E2B Q4_0 runs through the GPU-resident executor with interactive chat,
one-command-buffer prompt submission, packed Q4_0/Q6_K projection kernels, and
append-only performance metrics. The current implementation materially improves
the original throughput, while longer-workload prefill and decode optimization
remains open. See the
[current evidence and next implementation strategies](docs/atlas-metal-phases/phase-12a-perf-gemma4-resident-performance.md#next-implementation-strategies).

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

## Chat with Gemma 4

Gemma 4 chat requires Apple Silicon with Metal and this local GGUF fixture:

```text
models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf
```

Model fixtures are developer-local and ignored by Git. Download the pinned
3.3 GB text model non-interactively through Atlas. The optional JSON search
shows the immutable candidate ID without opening the model browser:

```zsh
cargo run --release -p atlas-cli -- model search \
  --provider huggingface --json \
  google/gemma-4-E2B-it-qat-q4_0-gguf
```

Pass that pinned candidate ID directly to the non-interactive downloader:

```zsh
cargo run --release -p atlas-cli -- model download \
  'huggingface:google/gemma-4-E2B-it-qat-q4_0-gguf@675cff42a74c774d6cb76f76d8eacb49b48c9b93:gguf-gemma4-q4_0:gemma-4-E2B_q4_0-it.gguf' \
  --id gemma4-e2b-q4_0
```

The command downloads only the text GGUF, not the separate multimodal
projector. It validates the embedded `gemma4` architecture, supported packed
tensor formats, pinned filename, byte count, and SHA-256 before registering the
fixture. The public repository does not currently require authentication. For
a gated/private Hugging Face artifact, set `HF_TOKEN` in the environment or run
`cargo run -p atlas-cli -- provider login huggingface` first.

If the registered fixture directory already exists, Atlas refuses to overwrite
it. Verify the downloaded model and confirm that Metal is available:

```zsh
cargo run --release -p atlas-cli -- model verify --model gemma4-e2b-q4_0
cargo run -p atlas-cli -- metal-info
```

The verification command should report `"verified": true`. Then start chat as
shown below.

Start an optimized interactive chat:

```zsh
cargo run --release -p atlas-cli -- chat \
  --model gemma4-e2b-q4_0 \
  --max-tokens 128
```

Wait for the `you>` prompt and type a message. The REPL supports:

- `/help` to show the available commands;
- `/reset` to clear conversation history while keeping the loaded model warm;
- `/quit` to exit.

For example:

```text
Atlas Gemma 4 chat. Commands: /reset, /help, /quit
you> hi
model> Hello! How can I help you today? 😊
```

Run a single non-interactive turn by supplying `--prompt`:

```zsh
cargo run --release -p atlas-cli -- chat \
  --model gemma4-e2b-q4_0 \
  --prompt 'Explain the history and importance of Paris.' \
  --max-tokens 128
```

Gemma 4 chat applies the instruction template embedded in the GGUF and always
uses the GPU-resident executor. Thought-channel text is filtered by default;
pass `--show-thoughts` only when it is intentionally needed. The raw
`generate --max-new-tokens N --greedy` command is for completion and parity
diagnostics and does not apply the chat template.

Each completed turn prints Resident performance metrics and appends one JSON
record to `artifacts/chat-performance.jsonl`. The first turn includes the model
weight upload. After `/reset`, later turns in the same process should report
`"weight_upload_bytes": 0`.

The current warm canonical `hi` workload reaches approximately 39 tok/s
prefill and 40 tok/s decode on the measured Apple M2 Max, with one prefill
command buffer and 3,489,602,512 Resident bytes. Longer-workload performance
remains open; see the prioritized
[Gemma 4 next implementation strategies](docs/atlas-metal-phases/phase-12a-perf-gemma4-resident-performance.md#next-implementation-strategies).

The supported product interface is currently the local CLI. HTTP serving is
intentionally deferred until the final API phase, after sampling, quantized
model loading, scheduling, and the memory runtime have stable CLI contracts.

## External Software

- [Rust](https://www.rust-lang.org/tools/install) provides Cargo and the Rust
  compiler used to build and test the workspace.
- Apple Xcode Command Line Tools provide the macOS SDK and Metal compiler used
  to build and run the native GPU kernels.
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/cli)
  provides the optional `hf` CLI used by `scripts/download-models.sh` to fetch
  local model fixtures.

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
