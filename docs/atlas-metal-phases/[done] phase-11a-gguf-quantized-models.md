# Phase 11a: GGUF Q4_0/Q8_0 quantized models

## Outcome

Atlas directly loads Llama-compatible GGUF Q4_0 and Q8_0 models and runs their
packed weights through resident projection kernels. Quantization reduces the
persisted model and resident weight footprint without materializing a full FP32
weight copy.

## Work

- Parse the GGUF header, metadata, tensor table, and alignment rules; accept
  only the Llama architecture, Q4_0/Q8_0 matrix encodings, and required F32
  normalization tensors in this phase.
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
than their FP32 source without a full FP32 dequantized buffer. See the
[GGUF conversion guide](../atlas-gguf-conversion.md) for the CLI workflow and
live conversion metrics.

## Acceptance evidence

Apple-Silicon Metal acceptance ran on 2026-07-19 with the pinned small
SmolLM2 fixture and eight generated tokens per format. Both runs used
`ExecutorMode::Resident` and reported `format: gguf-packed`; neither used a
reference fallback.

| Format | Packed upload bytes | Resident bytes | TTFT | Decode rate |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 75,785,472 | 123,289,656 | 355.07 ms | 27.77 tok/s |
| Q8_0 | 143,025,408 | 190,529,592 | 352.31 ms | 28.97 tok/s |

The pinned FP32 source has 269,030,016 weight bytes, so both packed paths meet
the persisted and resident-weight reduction gate without materializing a full
FP32 projection buffer.
