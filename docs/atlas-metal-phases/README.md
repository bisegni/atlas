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
| 6 | [Prefill and decode executors](phase-06-prefill-decode.md) |
| 7 | [Sampling engine](phase-07-sampling.md) |
| 8 | [Runtime and scheduler](phase-08-runtime-scheduler.md) |
| 9 | [API layer](phase-09-api-layer.md) |
| 10 | [Atlas local attention](phase-10-local-attention.md) |
| 11 | [Recurrent working memory](phase-11-working-memory.md) |
| 12 | [Latent graph memory](phase-12-latent-graph-memory.md) |
| 13 | [Graph retrieval](phase-13-graph-retrieval.md) |
| 14 | [Memory fusion](phase-14-memory-fusion.md) |
| 15 | [Learned memory router](phase-15-memory-router.md) |
