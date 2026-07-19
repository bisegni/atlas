# Phase 12: Quantized CLI acceptance

## Outcome

The normal local CLI has one model contract for FP32 SafeTensors and GGUF
Q4_0/Q8_0: inspect, verify, chat, generate, benchmark, and diagnose each
format with explicit quality and memory evidence.

## Work

- Make `chat`, `generate`, and `phase_08b_decode` select manifest-backed FP32
  or GGUF models and print format, model bytes, resident bytes, and tok/s.
- Keep exact greedy-token/finish-reason parity where the quantized fixture
  satisfies it; otherwise report a pinned logit/token tolerance and fail on
  unapproved drift.
- Require the quantized CLI path to expose unsupported encoding, manifest,
  tokenizer, and memory-budget failures before decoding begins.

## Exit gate

The small fixture proves the complete CLI workflow for FP32, Q4_0, and Q8_0;
the larger fixture records 128-token throughput and resident-memory reduction.
