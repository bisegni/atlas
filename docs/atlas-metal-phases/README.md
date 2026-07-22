# Atlas implementation phases

Each file is an independently executable plan. Start with the shared
[fixture contract](../Atlas_Metal_Inference_Engine_Phase_Subplans.md) for
the Hugging Face download commands, revision pinning, artifact format, and
cross-phase exit rules.

| Phase | Plan |
| --- | --- |
| 0 | [Metal runtime bootstrap](phase-00-metal-runtime-bootstrap.md) |
| 1 | [Tensor core and allocator](phase-01-tensor-core.md) |
| 2 | [Essential neural operators](phase-02-neural-operators.md) |
| 3 | [First transformer model](phase-03-first-transformer-model.md) |
| 4 | [KV cache](phase-04-kv-cache.md) |
| 5 | [Quantization]([done]%20phase-05-quantization.md) |
| 6 | [Prefill and decode executors]([done]%20phase-06-prefill-decode.md) |
| 7 | [Interactive CLI]([done]%20phase-07-interactive-cli.md) |
| 8 | [Streaming generation]([done]%20phase-08-streaming-generation.md) |
| 8.1 | [GPU-resident decode foundation]([done]%20phase-08a-gpu-resident-decode-foundation.md) |
| 8.2 | [GPU-resident executor integration]([done]%20phase-08b-gpu-resident-executor.md) |
| 8.3 | [Resident decode parity remediation]([done]%20phase-08c-resident-decode-parity-remediation.md) |
| 9 | [CLI model lifecycle]([done]%20phase-09-cli-model-lifecycle.md) |
| 10 | [Sampling engine](phase-10-sampling.md) |
| 11a | [GGUF Q4_0/Q8_0 quantized models]([done]%20phase-11a-gguf-quantized-models.md) |
| 11b | [Runtime and scheduler]([done]%20phase-11b-runtime-scheduler.md) |
| 11c | [Model providers and Hugging Face discovery]([done]%20phase-11c-model-providers.md) |
| 12 | [Quantized CLI acceptance]([done]%20phase-12-quantized-cli-acceptance.md) |
| 12a-pre | [Gemma 4 E2B resident text-generation foundation]([done]%20phase-12a-pre-gemma4-e2b-text-generation.md) |
| 12a-perf | [Gemma 4 Resident performance remediation](phase-12a-perf-gemma4-resident-performance.md) — open; see [next implementation strategies](phase-12a-perf-gemma4-resident-performance.md#next-implementation-strategies) |
| 12.1 | [Resident decode performance remediation](phase-12a-resident-decode-performance.md) |
| 12b | [Gemma 4 agentic tools](phase-12b-gemma4-agentic-tools.md) |
| 13 | [Atlas local attention](phase-13-local-attention.md) |
| 14 | [Recurrent working memory](phase-14-working-memory.md) |
| 15 | [Latent graph memory](phase-15-latent-graph-memory.md) |
| 16 | [Graph retrieval](phase-16-graph-retrieval.md) |
| 17 | [Memory fusion](phase-17-memory-fusion.md) |
| 18 | [Learned memory router](phase-18-memory-router.md) |
| 20 | [Local API server and hardening](phase-20-local-api-server.md) |
