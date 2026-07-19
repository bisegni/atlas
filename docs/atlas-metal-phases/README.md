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
| 5 | [Quantization](phase-05-quantization.md) |
| 6 | [Prefill and decode executors]([done]%20phase-06-prefill-decode.md) |
| 7 | [Interactive CLI]([done]%20phase-07-interactive-cli.md) |
| 8 | [Streaming generation]([done]%20phase-08-streaming-generation.md) |
| 8.1 | [GPU-resident decode performance](phase-08a-gpu-resident-decode-performance.md) |
| 9 | [Local API server](phase-09-local-api-server.md) |
| 10 | [Sampling engine](phase-10-sampling.md) |
| 11 | [Runtime and scheduler](phase-11-runtime-scheduler.md) |
| 12 | [API compatibility and hardening](phase-12-api-hardening.md) |
| 13 | [Atlas local attention](phase-13-local-attention.md) |
| 14 | [Recurrent working memory](phase-14-working-memory.md) |
| 15 | [Latent graph memory](phase-15-latent-graph-memory.md) |
| 16 | [Graph retrieval](phase-16-graph-retrieval.md) |
| 17 | [Memory fusion](phase-17-memory-fusion.md) |
| 18 | [Learned memory router](phase-18-memory-router.md) |
