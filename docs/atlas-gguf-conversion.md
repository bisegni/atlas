# Atlas GGUF conversion

Atlas converts a manifest-backed Llama SafeTensors model into a verified GGUF artifact for resident Metal inference. The converter writes Q4_0 or Q8_0 block-32 matrices and retains only required F32 tensors such as normalization weights.

## Convert a model

The source must be a verified `safetensors-fp32` model in `models/manifest.toml` with one `model.safetensors` file.

```zsh
cargo run -p atlas-cli -- model quantize --model small --id small-q4 --format q4_0 --quantizer auto --progress human
```

The artifact is written to `models/gguf/small-q4/`. Atlas copies the source `config.json` and `tokenizer.json`, writes `model.gguf`, hashes every file, and atomically appends the manifest record only after all checks pass.

`--quantizer auto` uses Metal when a device is available and otherwise reports CPU conversion. `--quantizer gpu` requires Metal; `--quantizer cpu` disables Metal deliberately.

## Progress and rate reporting

Human progress is written to stderr during `scan`, `quantize`, `write`, and `manifest`. It includes completed and total tensors, source and packed bytes, percent complete, elapsed time, MiB/s, ETA, and the current tensor.

Use JSON Lines when another tool consumes progress:

```zsh
cargo run -p atlas-cli -- model quantize --model small --id small-q8 --format q8_0 --progress json
```

Each `conversion_progress` event provides `stage`, `tensor`, tensor/byte totals, packed bytes, `percent`, `elapsed_ms`, `source_bytes_per_second`, and `eta_ms`. `eta_ms` remains null until Atlas has a useful rate. The final `conversion_completed` event reports quantizer, source/packed bytes, elapsed time, manifest ID, and output directory.

## Import and verify

```zsh
cargo run -p atlas-cli -- model import-gguf --path /path/to/model.gguf --id imported-q4 --config /path/to/config.json --tokenizer /path/to/tokenizer.json --source example/model --revision immutable-revision
cargo run -p atlas-cli -- model verify --model imported-q4
```

Import validates the Atlas GGUF header and supported tensor encodings, then copies the files into Atlas-managed storage without changing the source artifact. GGUF loading is a Resident-only path: Atlas uploads packed matrices directly and does not silently dequantize them or rerun them through the reference executor.
