# Atlas Metal Inference Engine: Shared Phase Contract

This is the common execution contract for the standalone plans in
[the Atlas phase index](atlas-metal-phases/README.md). A phase completes only
when its acceptance test passes on Apple Silicon and its evidence is recorded.
A CPU, Candle, or Python implementation may be an oracle; it is never the
production path being accepted.

## Model fixtures

Use one Llama-compatible family to keep format and tokenizer changes from
hiding runtime regressions.

| Tier | Repository | Purpose |
| --- | --- | --- |
| Small | [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Fast correctness fixture. |
| Larger | [`HuggingFaceTB/SmolLM2-1.7B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | Memory, throughput, and sustained-generation gate. |

Pin the resolved model revision in `models/manifest.toml`; model files are test
fixtures and must not be committed. Before every new download, run dry-run:

```zsh
python3 -m pip install --upgrade huggingface_hub
mkdir -p models/hf
hf download HuggingFaceTB/SmolLM2-135M-Instruct --dry-run
hf download HuggingFaceTB/SmolLM2-135M-Instruct \
  --local-dir models/hf/SmolLM2-135M-Instruct
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct --dry-run
hf download HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --local-dir models/hf/SmolLM2-1.7B-Instruct
```

## Required evidence

Store ignored run artifacts under `artifacts/phase-XX/`: model revision and
SHA256 manifest, macOS/device information, command line, result, numerical
tolerances, seed/token IDs, and requested metrics. Commit only a compact
checksum/summary fixture.

The small-model gate is mandatory for every phase. The larger model is also
mandatory from Phase 3 onward as a memory/performance gate.

## Promotion rules

1. Never silently overwrite golden outputs or tolerances.
2. The accepted path invokes Atlas Rust and Metal components wherever the phase
   claims coverage.
3. Atlas memory phases first validate bounded memory, data flow, isolation, and
   observability. Do not claim semantic quality until separately versioned
   trained fusion/router weights and evaluation exist.
