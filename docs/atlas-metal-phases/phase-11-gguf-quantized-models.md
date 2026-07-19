# Phase 11: GGUF Q4_0/Q8_0 quantized models

## Outcome

Atlas directly loads Llama-compatible GGUF Q4_0 and Q8_0 models and runs their
packed weights through resident projection kernels. Quantization reduces the
persisted model and resident weight footprint without materializing a full FP32
weight copy.

## Work

- Parse the GGUF header, metadata, tensor table, and alignment rules; accept
  only the Llama architecture and Q4_0/Q8_0 tensor encodings in this phase.
- Validate tokenizer/config compatibility, tensor names/shapes, source hashes,
  and manifest metadata before allocating GPU buffers.
- Implement direct packed Q4_0/Q8_0 matvec kernels with FP32 accumulation;
  retain packed model buffers and reject unsupported/mixed encodings clearly.
- Add `atlas-cli model import-gguf` and `atlas-cli quantize` to register a
  validated artifact in the local manifest, reporting original and packed
  bytes. Existing GGUF artifacts are imported, not silently rewritten.

## Exit gate

Small-fixture Q4_0 and Q8_0 models load through Atlas, generate with direct
packed resident projections, and report lower persisted/resident weight bytes
than their FP32 source without a full FP32 dequantized buffer.
