# Phase 12.1: Resident decode performance remediation

## Outcome

Atlas materially improves GPU-resident decode throughput for the larger
SmolLM2 fixture on Apple Silicon without weakening quantized quality,
residency, or normal generation semantics.

## Baseline

The reproducible runtime workload is normal one-shot resident chat:

```zsh
cargo run -p atlas-cli -- chat --model larger-q8 --prompt 'Atlas resident inference performance check: explain why GPU-resident decode matters in one sentence.' --max-tokens 32
```

Every completed chat turn appends one JSON object to
`artifacts/chat-performance.jsonl`. This append-only artifact is the runtime
evidence path; it records actual generated-token count, EOS/max-token finish
reason, TTFT, prefill/decode throughput, host/GPU time, command buffers,
upload/readback bytes, and resident bytes.

## Work

- Keep performance measurement in normal resident `chat`; there are no
  benchmark, diagnose, golden-suite, or profiling CLI modes. Q8 parity and
  golden checks remain Rust fixture acceptance coverage rather than public
  runtime commands.
- Remove the measured dominant host/command-buffer costs without adding a CPU
  fallback: retain resident weights, KV, activations, and token selection;
  preserve the single decode command-buffer boundary and token-only default
  readback.
- Improve the measured dominant packed-kernel path (Q4_0/Q8_0 matvec or its
  surrounding layout/dispatch) using Apple-GPU occupancy and memory-access
  evidence. Do not dequantize a full weight matrix or trade resident bytes for
  an undisclosed FP32 cache.
- Normal `chat` and `generate` stop at EOS and remain on
  `ExecutorMode::Resident`.
- Preserve the Phase-12 manifest quality gates and add a regression assertion
  that the profiling path is observational only: it must not change generated
  token IDs, finish reason, resident bytes, or default readback behavior.
- Restore the legacy resident score/softmax/value path as the observable
  production `ExecutorMode::Resident` attention selection while Q8 parity is
  unresolved; it remains GPU-resident and never falls back to `Reference`.
  Keep fused `attention_decode_fused_f32` behind the hidden
  `ResidentAttentionPath::Fused` diagnostic selector. Promote fused only after
  the full Q8 suite passes exactly. The opt-in profile reports the selected
  attention implementation and fused dispatch count.
- Validate FP32 fused-vs-legacy, Q8 fused-vs-legacy, and Q8 legacy-vs-FP32
  against position zero, multi-position KV reads, and the configured capacity
  boundary. Keep one command buffer per token and selected-token default
  readback assertions in those tests.
- Q4 keeps its pinned Phase-12 manifest policy. Before promoting Q8 after a
  fusion change, run the resident FP32-vs-Q8 golden suite for 32 greedy tokens
  each: `The capital of France is`, `Atlas resident decode validation.`, and
  the recorded fixed 64-token KV prompt in the Phase-12 artifact. Require
  exact generated token IDs and finish reason for every case; logits are
  diagnostic evidence only. If it fails, retain the output artifact and run
  the opt-in resident stage-parity diagnostic to identify the first divergent
  FP32 stage rather than relaxing the gate.
- Before promoting a Gemma 4 kernel optimization, add a pinned external
  runtime revision and require exact greedy prompt/generated token parity for
  a fixed canonical short chat. Phase 12a-pre intentionally accepted the
  text-chat foundation without this external-runtime comparison; optimization
  work must restore it as a promotion gate rather than relying on semantic
  similarity.

## Exit gate

On the same M2 Max, fixture revision, prompt, warm-up policy, and 128-token
workload as Phase 12, Q4_0 reaches at least 30 tok/s with
`ExecutorMode::Resident`, 128 generated tokens, and resident bytes no greater
than the Phase-12 1,366,335,800-byte baseline. Q8_0 must not regress below its
Phase-12 13.42 tok/s baseline or exceed 2,221,973,816 resident bytes.

Record the chat command, hardware/OS, model and resident bytes, actual
generated-token count, readback/command-buffer metrics, and throughput from
`artifacts/chat-performance.jsonl`. The required runtime evidence is a
Resident record with non-zero resident bytes and generated tokens; no fixed
token count is forced when EOS is reached. Use the same command with
`larger-q4` for a like-for-like comparison. Q8 parity remains covered by the
fixture-backed Rust acceptance tests.
