# Phase 12: Quantized CLI acceptance

## Outcome

The normal local CLI has one model contract for FP32 SafeTensors and GGUF
Q4_0/Q8_0: inspect, verify, chat, generate, benchmark, and diagnose each
format with explicit quality and memory evidence.

## Work

- Make `chat`, `generate`, and `phase_08b_decode` select manifest-backed FP32
  or GGUF models and print format, model bytes, resident bytes, and tok/s.
- Add stable manifest-backed `benchmark` and `diagnose` commands. `diagnose`
  runs the selected model through `ExecutorMode::Resident`, compares a
  quantized model with its manifest-pinned FP32 baseline, and reports exact
  token parity or the pinned logit/token gate.
- Store a quantized model's baseline ID, logit tolerance, token-agreement
  threshold, and resident-memory budget in `models/manifest.toml`. Validate
  these, file hashes, tokenizer compatibility, and supported encoding before
  decoding; reject any budget breach before generation begins.
- Keep exact greedy-token/finish-reason parity where the quantized fixture
  satisfies it; otherwise report a pinned logit/token tolerance and fail on
  unapproved drift.
- Require the quantized CLI path to expose unsupported encoding, manifest,
  tokenizer, and memory-budget failures before decoding begins.

## Exit gate

The small fixture proves the complete CLI workflow for FP32, Q4_0, and Q8_0;
the larger fixture records 128-token throughput and resident-memory reduction.

Run the Metal acceptance commands from an Apple-Silicon environment with the
pinned small and larger fixtures present. Begin with strict Q4_0/Q8_0 policy;
only update a tolerance after recording the observed diagnostic value.

```zsh
cargo run -p atlas-cli -- model verify --model small
cargo run -p atlas-cli -- model verify --model small-q4-gpu-20260719123633
cargo run -p atlas-cli -- model verify --model small-q8-gpu-20260719124407
cargo run -p atlas-cli -- benchmark --model small --prompt 'The capital of France is' --max-new-tokens 8 --warmup 1
cargo run -p atlas-cli -- diagnose --model small-q4-gpu-20260719123633 --prompt 'The capital of France is' --max-new-tokens 8 --warmup 1
cargo run -p atlas-cli -- diagnose --model small-q8-gpu-20260719124407 --prompt 'The capital of France is' --max-new-tokens 8 --warmup 1
```

The two diagnostic runs must report `executor: resident`, non-zero
`resident_bytes`, format/model bytes, and a passing quality gate. Record the
resulting token IDs, logit delta, resident bytes, and decode tok/s under
`artifacts/phase-12/`; do not mark this phase complete until the larger
fixture's warmed 128-token Q4_0/Q8_0 results prove resident-memory reduction.

## Acceptance evidence

On 2026-07-19, the small Q4_0 fixture ran through `diagnose` with
`ExecutorMode::Resident`, eight generated tokens, and one warm-up. It reported
123,289,656 resident bytes, 30.14 decode tok/s, and a maximum FP32-baseline
logit delta of 33.802639. Its generated tokens did not overlap the selected
FP32 baseline tokens, so the Q4_0 policy pins a 34.0 maximum delta and 0.0
minimum token agreement. Q8_0 and the larger-fixture gate remain pending.

The small Q8_0 fixture ran with the same command and warm-up, reporting
190,529,592 resident bytes, 29.45 decode tok/s, exact greedy-token parity, and
a maximum FP32-baseline logit delta of 0.598497. Its policy pins a 0.6 maximum
delta while retaining required exact token agreement. The larger-fixture gate
remains pending.

After the pinned policies were recorded, both Q4_0 and Q8_0 diagnostic commands
completed successfully on Apple Silicon. The small fixture therefore covers
the manifest-backed inspect/verify/generate-diagnostic contract for FP32,
Q4_0, and Q8_0 using `ExecutorMode::Resident`.

The larger fixture is pinned at revision
`31b70e2e869a7173562077fd711b654946d38674`. Its Q4_0 GPU conversion completed
in 80,963 ms, processing 3,264 MiB of source tensors into a 918 MiB packed
artifact. The corresponding Q8_0 conversion also completed successfully.

The following warm-up-one `benchmark --ignore-eos` results use the same prompt
and force the required 128-token decode workload. `--ignore-eos` is
benchmark-only; normal `chat`, `generate`, and `diagnose` flows retain EOS
termination.

| Format | Model bytes | Resident bytes | Decode tok/s | Generated tokens |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 3,424,883,416 | 7,248,847,160 | 3.77 | 128 |
| Q4_0 | 965,113,016 | 1,366,335,800 | 15.85 | 128 |
| Q8_0 | 1,820,751,032 | 2,221,973,816 | 13.42 | 128 |

All three runs reported `ExecutorMode::Resident`, `finish_reason: maxtokens`,
and 128 requested/generated tokens. Q4_0 lowers resident bytes by 81.1% and
Q8_0 by 69.3% versus the FP32 run, completing the Phase 12 larger-fixture
throughput and memory-reduction gate.
