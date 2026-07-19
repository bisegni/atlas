# Phase 5: Quantization

## Outcome

FP16, INT8 weight-only, and Q4 weight-only decode run without a full
dequantized-weight copy.

## Work

- Freeze FP16 logit/performance baselines.
- Implement packed blocks, scale/zero metadata, and fused in-register or
  threadgroup dequantization with FP32 accumulation.
- Add project-owned SafeTensors conversion first; add GGUF only after that
  format is validated. Measure loading independently from decode.

## Model fixture

Use the small model for deterministic logits/tokens; use the larger model for
resident-memory reduction and decode-throughput measurement.

## Exit gate

`phase_05_quant` reports bytes, logit delta, token agreement, and tok/s for
FP16/INT8/Q4; it fails if a quantized path allocates an FP16-sized full buffer.
