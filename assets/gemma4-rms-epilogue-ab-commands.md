# Gemma Q4 Resident RMS epilogue A/B

Run these commands from the Atlas repository root on the Apple-Silicon host
that has Metal access and the verified fixture at:

`models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf`

```zsh
cargo run --release -p atlas-cli -- model verify --model gemma4-e2b-q4_0

ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT=baseline ./scripts/run-gemma4-performance-acceptance.sh

ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT=fused ./scripts/run-gemma4-performance-acceptance.sh
```

The runner writes one timestamped directory per run under
`artifacts/phase-12a-perf/`. Paste both of these files back after the commands
finish:

```text
artifacts/phase-12a-perf/<baseline-timestamp>/acceptance-summary.json
artifacts/phase-12a-perf/<fused-timestamp>/acceptance-summary.json
```

Also paste the corresponding `q4_0/benchmark-summary.json` files if either
run fails. They contain the fixed-128 SHA, EOS position, Q4 KV residency, and
per-run decode throughput needed to decide whether the fusion can be promoted.
