# Phase 6: Prefill and decode executors

## Outcome

Separate prefill/decode plans reuse pipelines and buffers while generating a
cached multi-token response.

## Work

- Create immutable prefill/decode plans with cached pipeline states and
  argument data.
- Use batched RoPE/attention for prefill and narrow matvec/KV reads for decode.
- Measure CPU encoding, GPU time, queue wait, synchronization, and allocation.

## Model fixture

Use the small fixture at prompt lengths 1/8/64/256 for parity. Use the larger
fixture for 512-token prefill plus 128-token decode throughput and memory.

## Exit gate

`phase_06_executors` matches the Phase-3 path, creates no per-token pipeline
after warm-up, and records TTFT, prefill/decode tok/s, and p50/p95 latency.

## Implementation status

- `atlas_model::executor` provides immutable prefill/decode plans and an
  `AtlasExecutor` with session-local, contiguous KV caches.  Prompt processing
  appends each layer's K/V at its absolute position; subsequent tokens use the
  cached K/V view and decode-shaped matvec projections.
- `ExecutorMetrics` reports CPU tokenization, TTFT, prefill/decode duration and
  rate, p50/p95 decode latency, and pipeline warm-up counters.  The Phase-6
  integration test is intentionally ignored because it requires the local
  Metal device and downloaded fixture.
- The reference executor currently accepts the native FP32 model tensors via
  `QuantFormat::Fp16`.  INT8/Q4 plan selection remains explicitly rejected
  until direct packed Metal projection kernels are available; it must not
  silently dequantize a complete model buffer.
