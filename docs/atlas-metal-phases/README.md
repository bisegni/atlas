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
| 8 | [Local API server](phase-08-local-api-server.md) |
| 9 | [Sampling engine](phase-09-sampling.md) |
| 10 | [Runtime and scheduler](phase-10-runtime-scheduler.md) |
| 11 | [API compatibility and hardening](phase-11-api-hardening.md) |
| 12 | [Atlas local attention](phase-12-local-attention.md) |
| 13 | [Recurrent working memory](phase-13-working-memory.md) |
| 14 | [Latent graph memory](phase-14-latent-graph-memory.md) |
| 15 | [Graph retrieval](phase-15-graph-retrieval.md) |
| 16 | [Memory fusion](phase-16-memory-fusion.md) |
| 17 | [Learned memory router](phase-17-memory-router.md) |
