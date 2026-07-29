# Gemma Q4 Resident RMS rollback A/B

Run these commands from the Atlas repository root on the Apple-Silicon host
that has Metal access and the verified fixture at:

`models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf`

```zsh
cargo run --release -p atlas-cli -- model verify --model gemma4-e2b-q4_0

bash scripts/run-gemma4-rms-norm-ab.sh --screen
```

The screen writes one timestamped directory under
`artifacts/phase-12a-rms-norm-ab/`. Paste this file back after it finishes:

```text
artifacts/phase-12a-rms-norm-ab/<timestamp>/rms-norm-ab-summary.json
```

The default is the vectorized RMS kernel. This runner compares it with the
scalar diagnostic oracle selected by `ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=baseline`.
Use it only to guard or investigate a future regression; it is no longer a
promotion gate.
