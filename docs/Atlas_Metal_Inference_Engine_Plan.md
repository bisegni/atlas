# Atlas Metal Inference Engine Plan

> Implementation companion: [phase-by-phase executable subplans](Atlas_Metal_Inference_Engine_Phase_Subplans.md).
> Each phase has a concrete runnable outcome, acceptance gate, and a pinned
> Hugging Face fixture path for both a small and a larger model.

## 1. Project Goal

Build a new LLM inference engine written primarily in Rust and optimized initially for Apple Silicon using Metal.

The engine will eventually support the complete Atlas memory architecture:

- Local Transformer attention
- Bounded KV cache
- Recurrent working memory
- Episodic memory
- Semantic graph memory
- Latent graph memory
- Learned retrieval and routing
- Persistent inference sessions

The initial version should not attempt to implement the entire Atlas memory architecture.

The first goal is:

> Run a small decoder-only Transformer model entirely through a Rust runtime and native Metal compute kernels, producing correct tokens with measurable performance.

---

## 2. Design Principles

### Rust-first

Core runtime, model execution, memory management, scheduling, tokenizer integration, and API serving should be written in Rust.

Metal shader code will be written in Metal Shading Language.

### Metal-native first

The first backend should use Metal directly rather than a cross-platform GPU abstraction.

This provides control over:

- GPU buffer allocation
- Unified-memory behavior
- Command-buffer scheduling
- Compute pipelines
- Threadgroup dimensions
- Kernel specialization
- Synchronization
- Memory residency
- Profiling
- Resource reuse

### Correctness before optimization

Each Metal operation must first be compared against a trusted CPU implementation.

### Modular operators

Model implementations must depend on tensor operations, not directly on Metal.

### Backend independence

Although Metal is the first backend, the architecture should later allow:

- CPU
- CUDA
- Vulkan
- WebGPU
- Other accelerators

### Bounded active memory

The architecture should be designed from the beginning to support:

- Sliding local attention
- Paged or segmented KV storage
- Memory eviction
- Graph-memory retrieval
- CPU/GPU tiering

---

## 3. Recommended Technology

### Main language

Rust, using a current stable Rust toolchain and the 2024 edition where supported.

### GPU interface

Use:

- `objc2`
- `objc2-metal`
- `objc2-foundation`

Do not base new development on the deprecated `metal` crate.

### Shader language

Metal Shading Language (`.metal` files).

### Model formats

Begin with:

- SafeTensors for FP16/BF16 weights
- GGUF later for quantized models

### Tokenization

Use a Rust tokenizer implementation, preferably Hugging Face Tokenizers or a small model-specific tokenizer layer.

### Serialization

Use:

- `serde`
- `serde_json`
- `toml`

### Error handling

Use:

- `thiserror` for library errors
- `anyhow` in command-line applications

### Async services

Use Tokio only in the serving and orchestration layers.

Do not introduce asynchronous complexity inside basic tensor operations.

### Testing

Use:

- Rust unit tests
- Integration tests
- Property testing where useful
- Golden model outputs
- Numerical comparison against Python or Candle

### Benchmarks

Use:

- Criterion for CPU and orchestration benchmarks
- Custom GPU timing around Metal command buffers
- Apple Metal profiling tools

---

## 4. Reference Implementations

Use Candle as a correctness and architectural reference, not necessarily as the foundation of the final runtime.

Candle can help validate:

- Weight loading
- Model architecture
- Tokenization
- Tensor shapes
- Expected logits
- Basic Metal behavior

However, the final engine should own its tensor runtime, scheduler, cache model, and specialized kernels.

Apple Metal Performance Shaders and MPSGraph may be used selectively for operations where they provide useful optimized implementations. They should remain optional execution paths rather than defining the entire engine.

---

## 5. High-Level Architecture

```text
Client/API
    │
    ▼
Inference Session Manager
    │
    ├── Tokenizer
    ├── Prompt Processor
    ├── Sampling Engine
    └── Generation Scheduler
            │
            ▼
        Model Runtime
            │
    ┌───────┼────────┐
    ▼       ▼        ▼
Tensor    Model     Cache
Engine    Graph     Manager
    │       │        │
    └───────┼────────┘
            ▼
      Backend Interface
            │
            ▼
       Metal Backend
            │
    ┌───────┼────────────┐
    ▼       ▼            ▼
Buffers   Kernels    Command Scheduler
```

Later:

```text
Model Runtime
    │
    ├── Local Attention
    ├── Recurrent Memory
    ├── Graph Retrieval
    ├── Memory Fusion
    └── Atlas Memory Router
```

---

## 6. Repository Structure

```text
atlas/
├── Cargo.toml
├── README.md
├── LICENSE
├── rust-toolchain.toml
├── crates/
│   ├── atlas-core/
│   │   ├── tensor.rs
│   │   ├── dtype.rs
│   │   ├── shape.rs
│   │   ├── layout.rs
│   │   ├── device.rs
│   │   └── error.rs
│   ├── atlas-metal/
│   │   ├── device.rs
│   │   ├── buffer.rs
│   │   ├── allocator.rs
│   │   ├── pipeline.rs
│   │   ├── command.rs
│   │   ├── profiler.rs
│   │   └── kernels/
│   │       ├── elementwise.metal
│   │       ├── reduction.metal
│   │       ├── normalization.metal
│   │       ├── matmul.metal
│   │       ├── rope.metal
│   │       ├── attention.metal
│   │       ├── quantization.metal
│   │       └── sampling.metal
│   ├── atlas-ops/
│   │   ├── unary.rs
│   │   ├── binary.rs
│   │   ├── matmul.rs
│   │   ├── norm.rs
│   │   ├── softmax.rs
│   │   ├── embedding.rs
│   │   └── attention.rs
│   ├── atlas-models/
│   │   ├── config.rs
│   │   ├── loader.rs
│   │   ├── llama.rs
│   │   ├── qwen.rs
│   │   └── common/
│   ├── atlas-cache/
│   │   ├── kv_cache.rs
│   │   ├── page.rs
│   │   ├── allocator.rs
│   │   ├── sliding_window.rs
│   │   └── session.rs
│   ├── atlas-quant/
│   │   ├── format.rs
│   │   ├── dequant.rs
│   │   ├── q4.rs
│   │   ├── q8.rs
│   │   └── loader.rs
│   ├── atlas-runtime/
│   │   ├── engine.rs
│   │   ├── scheduler.rs
│   │   ├── request.rs
│   │   ├── session.rs
│   │   ├── generation.rs
│   │   └── metrics.rs
│   ├── atlas-memory/
│   │   ├── working.rs
│   │   ├── episodic.rs
│   │   ├── semantic.rs
│   │   ├── latent.rs
│   │   ├── graph.rs
│   │   └── retrieval.rs
│   ├── atlas-tokenizer/
│   ├── atlas-server/
│   └── atlas-cli/
├── models/
├── shaders/
├── tests/
├── benchmarks/
├── tools/
└── docs/
```

---

## 7. Core Interfaces

### Device abstraction

```rust
pub trait Device {
    type Buffer;
    type CommandContext;

    fn allocate(&self, bytes: usize) -> Result<Self::Buffer>;
    fn upload(&self, data: &[u8]) -> Result<Self::Buffer>;
    fn command_context(&self) -> Result<Self::CommandContext>;
    fn synchronize(&self) -> Result<()>;
}
```

### Tensor representation

```rust
pub struct Tensor {
    pub storage: Storage,
    pub shape: Shape,
    pub strides: Strides,
    pub dtype: DType,
    pub offset: usize,
}
```

Avoid encoding Metal-specific behavior directly in `Tensor`.

### Operation interface

```rust
pub trait Operator {
    fn prepare(
        &self,
        inputs: &[Tensor],
        context: &mut ExecutionContext,
    ) -> Result<PreparedOperator>;

    fn execute(
        &self,
        prepared: &PreparedOperator,
        context: &mut ExecutionContext,
    ) -> Result<Vec<Tensor>>;
}
```

### Model interface

```rust
pub trait CausalLanguageModel {
    fn forward(
        &mut self,
        input_ids: &[u32],
        positions: &[u32],
        session: &mut InferenceSession,
    ) -> Result<Tensor>;
}
```

### Cache interface

```rust
pub trait AttentionCache {
    fn append(
        &mut self,
        layer: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<()>;

    fn view(
        &self,
        layer: usize,
        start: usize,
        end: usize,
    ) -> Result<CacheView>;

    fn evict_before(&mut self, position: usize) -> Result<()>;
}
```

---

## 8. Phase 0 — Metal Runtime Bootstrap

### Objective

Prove that Rust can compile, load, dispatch, and validate Metal compute kernels.

### Implement

- Select the default Metal device.
- Create command queue.
- Create buffers.
- Compile `.metal` source or load a precompiled Metal library.
- Create compute pipeline state.
- Dispatch kernels.
- Read results.
- Capture GPU errors.
- Measure kernel execution time.

### First kernels

1. Vector addition
2. Scalar multiplication
3. Elementwise activation
4. Reduction sum
5. Matrix transpose

### Completion criteria

- Deterministic results
- CPU-reference comparison
- Repeated execution without memory growth
- Reliable pipeline caching
- Basic GPU timing

---

## 9. Phase 1 — Tensor Core

### Objective

Create a minimal tensor system suitable for LLM inference.

### Data types

Initially:

- FP32
- FP16

Then:

- BF16 where supported
- INT8
- Packed INT4

### Tensor features

- Shape
- Stride
- Offset
- Contiguous layout
- Views
- Reshape
- Transpose metadata
- Device ownership
- Read-only weight tensors
- Temporary tensors

### Memory allocator

Implement separate classes of allocation:

- Persistent model weights
- Persistent KV cache
- Session state
- Temporary activation buffers
- Small parameter buffers

Use buffer pooling to avoid allocating Metal buffers on every token.

### Completion criteria

- No allocation inside simple repeated decode loops after warm-up
- Tensor shape validation
- CPU and Metal numerical consistency
- Allocation metrics exposed

---

## 10. Phase 2 — Essential Neural Operators

Implement operations in this order:

1. Embedding lookup
2. Elementwise add and multiply
3. SiLU
4. GELU, if required
5. RMSNorm
6. LayerNorm
7. Matrix-vector multiplication
8. Matrix-matrix multiplication
9. RoPE
10. Softmax
11. Attention score calculation
12. Attention/value aggregation
13. Linear output projection
14. Logits processing
15. Sampling

Autoregressive decoding is dominated by different matrix shapes than prompt prefill.

Create separate kernel paths for:

- Prefill matrix-matrix operations
- Decode matrix-vector or narrow matrix operations

Pipeline selection should consider:

- Input dtype
- Output dtype
- Matrix dimensions
- Transposition
- Quantization format
- Apple GPU family
- Prefill or decode mode

---

## 11. Phase 3 — First Transformer Model

### Recommended initial model

Implement one small Llama-compatible model first.

Suggested size:

- 100M–500M parameters for debugging
- 1B–3B parameters for realistic tests

Avoid beginning with a large MoE or multimodal model.

### Implement

- Model configuration parser
- SafeTensors loader
- Token embedding
- RMSNorm
- RoPE
- Grouped-query attention if required
- SwiGLU MLP
- Final norm
- LM head
- Greedy decoding

### Initial inference flow

```text
Token IDs
→ Embedding
→ Transformer layers
→ Final normalization
→ LM head
→ Logits
→ Argmax
```

### Validation

Compare layer-by-layer against a trusted implementation:

- Embeddings
- Normalization output
- Q/K/V projections
- RoPE
- Attention scores
- Attention output
- MLP output
- Final logits

Use fixed seeds and fixed prompts.

### Completion criteria

- Exact tokenizer compatibility
- Stable generation
- Numerically close logits
- At least one complete prompt-to-generation test

---

## 12. Phase 4 — KV Cache

### First implementation

Use a simple contiguous cache:

```text
[layer][K or V][head][position][dimension]
```

### Second implementation

Move to segmented pages:

```text
Session
  ├── Layer 0
  │    ├── Page 0
  │    ├── Page 1
  │    └── Page 2
  └── Layer N
```

Each page should store:

- Layer identifier
- Session identifier
- Start position
- Used token count
- Capacity
- Key buffer range
- Value buffer range
- Residency state
- Reference count

### Sliding attention

Add configurable local attention:

- Keep the last `W` tokens active.
- Evict KV pages older than the window.
- Preserve special sink tokens where useful.
- Expose evicted regions to the future Atlas memory writer.

### Completion criteria

- KV growth is measurable.
- Sliding-window mode keeps GPU cache bounded.
- Cache eviction does not corrupt active positions.
- Multiple independent sessions work correctly.

---

## 13. Phase 5 — Quantization

### Initial formats

Implement:

1. FP16
2. INT8 weight-only
3. Q4 weight-only

Do not initially quantize activations or KV cache.

### Execution

Prefer fused kernels:

```text
Quantized weights
→ Load packed blocks
→ Dequantize in registers or threadgroup memory
→ Multiply
→ Accumulate in FP32
→ Store FP16/FP32
```

Avoid fully dequantizing a model into a separate GPU buffer.

### GGUF

Add GGUF loading only after the tensor runtime and FP16 model work correctly.

### Completion criteria

- Model weights remain packed in GPU-accessible buffers.
- Quantized logits remain acceptably close to FP16.
- Memory reduction is measured.
- Decode speed is benchmarked independently from loading speed.

---

## 14. Phase 6 — Prefill and Decode Executors

Create two distinct execution paths.

### Prefill executor

Optimized for:

- Many input tokens
- Larger matrix operations
- Batched RoPE
- Batched attention
- Fast cache construction

### Decode executor

Optimized for:

- One or a few new tokens
- Matrix-vector operations
- Reading existing KV pages
- Minimal command-buffer overhead
- Kernel fusion

### Decode loop

```text
Prepare token
→ Encode command buffer
→ Run transformer layers
→ Produce logits
→ Sample token
→ Append KV
→ Repeat
```

Reduce per-token CPU overhead by:

- Caching pipeline states
- Reusing argument buffers
- Reusing command encoders where feasible
- Preallocating buffers
- Avoiding tensor graph reconstruction
- Grouping compatible kernels

---

## 15. Phase 7 — Sampling Engine

Implement on CPU first:

- Greedy
- Temperature
- Top-k
- Top-p
- Repetition penalty
- Frequency penalty
- Presence penalty
- Stop tokens

Later move high-cost logits transformations to Metal.

The sampler should accept deterministic seeds for testing.

---

## 16. Phase 8 — Runtime and Scheduler

### Session structure

```rust
pub struct InferenceSession {
    pub id: SessionId,
    pub position: usize,
    pub cache: Box<dyn AttentionCache>,
    pub recurrent_state: Option<WorkingMemory>,
    pub sampling: SamplingConfig,
    pub metrics: SessionMetrics,
}
```

### Scheduler v1

Support one active request.

### Scheduler v2

Support several sessions through continuous batching.

### Scheduling responsibilities

- Admit requests
- Group compatible decode steps
- Allocate KV pages
- Release finished sessions
- Apply backpressure
- Cancel requests
- Track latency
- Isolate user memory

### Completion criteria

- Streaming token generation
- Request cancellation
- Multiple sessions
- Bounded queue
- Per-session metrics

---

## 17. Phase 9 — API Layer

Create:

- Command-line runner
- Rust library API
- OpenAI-compatible HTTP endpoint

Suggested endpoints:

```text
POST /v1/completions
POST /v1/chat/completions
GET  /v1/models
GET  /health
GET  /metrics
```

Keep the API crate separate from the inference runtime.

---

## 18. Phase 10 — Atlas Local Attention

After the standard Transformer engine works:

- Add configurable local attention windows.
- Allow different windows per layer.
- Add attention sink positions.
- Track context regions leaving the active window.
- Generate memory-write candidates from expired chunks.

The initial Atlas memory path becomes:

```text
Current tokens
→ Local attention
→ Standard Transformer layers
→ Output
```

with bounded local KV memory.

---

## 19. Phase 11 — Recurrent Working Memory

Add persistent fixed-size memory slots per session.

```rust
pub struct WorkingMemory {
    pub slots: Tensor,
    pub importance: Vec<f32>,
    pub generation: u64,
}
```

### Operations

- Read from slots
- Write selected slots
- Retain unchanged slots
- Reset session
- Serialize session state

### Integration

At selected layers:

```text
Hidden state
  ├── Local attention
  ├── Working-memory read
  └── Residual path
          │
          ▼
      Learned fusion
```

Start with:

- 16 slots
- 32 slots
- 64 slots

---

## 20. Phase 12 — Latent Graph Memory

Implement graph memory outside the model first.

### Node data

```rust
pub struct MemoryNode {
    pub id: MemoryNodeId,
    pub node_type: MemoryNodeType,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub timestamp: u64,
    pub importance: f32,
    pub confidence: f32,
    pub source: SourceReference,
}
```

### Edge data

```rust
pub struct MemoryEdge {
    pub from: MemoryNodeId,
    pub to: MemoryNodeId,
    pub relation: RelationType,
    pub weight: f32,
    pub timestamp: u64,
}
```

### First storage engine

Use a simple in-process implementation:

- Memory-mapped node records
- Append-only event log
- In-memory adjacency lists
- Vector index
- Periodic snapshots

Do not begin with a remote graph database.

### GPU interaction

The graph itself should initially remain in CPU memory.

Only retrieved memory slots should be transferred to Metal.

---

## 21. Phase 13 — Graph Retrieval

### Retrieval flow

```text
Current hidden state
→ Query projection on Metal
→ Copy compact query to CPU
→ Vector search
→ Neighbor expansion
→ Gather node values
→ Upload retrieved slots
→ Fuse into hidden state
```

Later, move vector search or selected graph operations to Metal if profiling justifies it.

### Retrieval cache

Cache by:

- Session
- Query fingerprint
- Active goal
- Recent graph generation
- Selected layer

Avoid querying the graph for every token.

Start with retrieval:

- Once per prompt
- Once per 32–64 generated tokens
- On explicit router triggers

---

## 22. Phase 14 — Memory Fusion

Support several fusion strategies.

### Gated residual

```text
output = hidden + gate × memory
```

### Cross-attention

```text
hidden queries
→ small attention over retrieved memory slots
```

### Concatenation and projection

```text
concat(hidden, pooled memory)
→ projection
```

Start with gated residual fusion because it is easier to implement and benchmark.

---

## 23. Phase 15 — Learned Memory Router

The router decides whether to use:

- Local attention
- Working memory
- Graph memory
- Direct residual path

```rust
pub struct RoutingDecision {
    pub local_weight: f32,
    pub working_weight: f32,
    pub graph_weight: f32,
    pub retrieve_graph: bool,
    pub write_memory: bool,
}
```

Initially implement deterministic routing.

Example:

- Retrieve at fixed layer numbers.
- Retrieve every 64 tokens.
- Write memory when a chunk leaves the local window.

Only introduce a learned router after the deterministic system works.

---

## 24. Metal Kernel Roadmap

### Kernel group A — Basic operations

- Copy
- Cast
- Add
- Multiply
- Activation
- Reduction

### Kernel group B — Transformer operations

- Embedding
- RMSNorm
- RoPE
- Softmax
- Matrix multiplication
- Attention
- SwiGLU

### Kernel group C — Fused decode kernels

- RMSNorm + projection
- QKV projection
- RoPE + KV append
- Attention score + softmax
- Attention aggregation
- Gate + up projection
- SwiGLU + down projection

### Kernel group D — Quantized operations

- Q8 matrix-vector
- Q4 matrix-vector
- Quantized matrix-matrix
- Quantized embedding lookup

### Kernel group E — Atlas memory

- Query projection
- Memory-slot attention
- Gated memory fusion
- Working-memory update
- Memory-slot normalization
- Similarity calculation for small GPU-resident indexes

---

## 25. Memory Management Strategy

Apple Silicon uses unified memory, but unified memory does not remove the need for careful allocation and synchronization.

Define explicit memory classes.

### Model weights

- Long lived
- Read only
- Loaded once

### KV cache

- Session scoped
- Page allocated
- Frequently appended
- Evictable

### Working memory

- Small
- Session scoped
- Persistent across tokens

### Retrieved graph slots

- Small
- Reused
- Refreshed periodically

### Activations

- Short lived
- Recycled aggressively

### Staging buffers

- Used for loading, conversion, and CPU/GPU coordination

Build a lifetime-aware arena allocator for activations.

---

## 26. Command Scheduling

Avoid submitting one Metal command buffer for every tiny operation.

### Initial implementation

One command buffer per model layer.

### Improved implementation

One command buffer for several or all layers of a decode token.

### Long-term implementation

- Pre-encoded or reusable execution plans
- Indirect command buffers where appropriate
- Asynchronous graph retrieval overlapping GPU execution
- Separate queues only when profiling proves benefit

Track:

- CPU encoding time
- GPU execution time
- Queue wait time
- Synchronization time
- Buffer allocation time

---

## 27. Kernel Compilation

Support two modes.

### Development mode

Compile Metal shader source at runtime.

Advantages:

- Faster iteration
- Easier debugging

### Release mode

Compile `.metal` files into a Metal library during the build.

Advantages:

- Faster startup
- Predictable deployment
- Better control over compilation failures

Cache pipeline states by kernel specialization.

---

## 28. Correctness Strategy

Every operator must have:

1. CPU reference
2. Metal implementation
3. Shape tests
4. Dtype tests
5. Edge-case tests
6. Numerical-tolerance tests

### Model validation levels

- Level 1: Individual operator outputs
- Level 2: Single Transformer block
- Level 3: Complete prefill logits
- Level 4: Complete decode logits
- Level 5: Generated sequence
- Level 6: Long-running session

---

## 29. Performance Benchmarks

Track separately:

### Model loading

- File-read time
- Weight-conversion time
- GPU-buffer creation time
- Peak memory

### Prefill

- Tokens per second
- Time to first token
- GPU utilization
- Memory bandwidth

### Decode

- Tokens per second
- Per-token latency
- CPU command-encoding time
- GPU execution time

### Cache

- Bytes per token
- Allocation count
- Page fragmentation
- Eviction time

### Atlas memory

- Retrieval latency
- Upload latency
- Fusion cost
- Memory hit rate
- Recall accuracy

---

## 30. Initial Milestones

### Milestone 1 — Metal hello compute

Deliverables:

- Native Metal initialization
- Buffer allocation
- Vector-add kernel
- GPU timing
- Automated test

### Milestone 2 — Tensor runtime

Deliverables:

- Tensor metadata
- Buffer pooling
- FP16 and FP32 support
- Basic operations

### Milestone 3 — Neural kernels

Deliverables:

- RMSNorm
- RoPE
- Softmax
- Matrix multiplication
- SwiGLU

### Milestone 4 — Tiny Transformer

Deliverables:

- One Transformer block
- CPU comparison
- Correct logits

### Milestone 5 — Full small model

Deliverables:

- SafeTensors loading
- Tokenizer
- Prefill
- Decode
- Greedy generation

### Milestone 6 — KV cache

Deliverables:

- Multi-token cache
- Sliding window
- Bounded memory

### Milestone 7 — Quantized inference

Deliverables:

- Q8
- Q4
- GGUF loading
- Fused dequantization kernels

### Milestone 8 — Runtime service

Deliverables:

- Streaming
- Multiple sessions
- OpenAI-compatible API

### Milestone 9 — Working memory

Deliverables:

- Fixed recurrent slots
- Metal fusion kernels
- Session persistence

### Milestone 10 — Graph memory

Deliverables:

- Persistent graph
- Retrieval
- Memory-slot upload
- Neural fusion

---

## 31. First Practical MVP

The MVP should support:

- macOS on Apple Silicon
- One Metal GPU
- One Llama-compatible dense model
- SafeTensors FP16
- Greedy and top-p sampling
- Prompt prefill
- Token-by-token decode
- Standard contiguous KV cache
- CLI generation
- Numerical comparison against a reference implementation

Do not include in MVP:

- Quantization
- Continuous batching
- MoE
- Multimodal input
- Graph memory
- Distributed execution
- Training
- Speculative decoding
- Multiple GPU backends

---

## 32. Second Practical MVP

Add:

- GGUF
- Q4/Q8
- Paged KV cache
- Sliding local attention
- HTTP serving
- Streaming output
- Basic batching
- Performance profiler

This version becomes the base inference engine on which Atlas memory can be introduced.

---

## 33. Third Practical MVP: Atlas Memory

Add:

- Fixed recurrent working-memory slots
- Expired-context chunk encoder
- Latent-memory node creation
- CPU-resident graph
- Vector retrieval
- Memory-slot transfer to Metal
- Gated latent-memory fusion
- Bounded local KV cache

### Central demonstration

1. Introduce a fact.
2. Generate enough tokens for it to leave the local KV window.
3. Convert or preserve the information as a latent memory.
4. Delete the old KV region.
5. Retrieve the latent memory later.
6. Answer correctly.
7. Show that active GPU memory remains bounded.

---

## 34. Key Risks

### Metal kernel performance

A correct generic GEMM may perform poorly for some matrix dimensions.

Mitigation:

- Autotune tile sizes
- Separate prefill and decode kernels
- Specialize by Apple GPU family
- Benchmark against established implementations
- Use MPS or MPSGraph selectively where beneficial

### Command-buffer overhead

Small decode operations can become CPU-bound.

Mitigation:

- Kernel fusion
- Pipeline caching
- Command-buffer batching
- Preallocated argument data
- Reduced synchronization

### Numerical drift

FP16, BF16, and quantization can change model output.

Mitigation:

- FP32 accumulation
- Layer-by-layer comparisons
- Golden logits
- Configurable tolerances

### Memory fragmentation

Long-running multi-session serving may fragment cache allocation.

Mitigation:

- Fixed-size pages
- Free lists
- Allocation telemetry
- Session quotas

### Graph retrieval latency

CPU graph retrieval may interrupt token generation.

Mitigation:

- Retrieve by chunk
- Prefetch
- Cache subgraphs
- Overlap retrieval with GPU work
- Use compact fixed-size memory slots

### Excessive scope

Building a complete inference engine and a new neural architecture simultaneously is risky.

Mitigation:

- Complete standard Transformer inference first
- Add Atlas memory modules incrementally
- Maintain reference implementations
- Require measurable criteria for every phase

---

## 35. Recommended First Implementation Sequence

The engineering agent should work in this exact order:

1. Create Rust workspace.
2. Initialize Metal device and command queue.
3. Execute vector-add Metal kernel.
4. Build buffer abstraction.
5. Build tensor metadata.
6. Add CPU tensor backend for testing.
7. Add elementwise Metal operators.
8. Implement RMSNorm.
9. Implement RoPE.
10. Implement basic matrix multiplication.
11. Implement softmax.
12. Implement single-head attention.
13. Implement grouped-query attention.
14. Implement one Transformer layer.
15. Load one layer of real model weights.
16. Compare against Python or Candle.
17. Load complete small model.
18. Implement prefill.
19. Implement decode.
20. Implement contiguous KV cache.
21. Implement tokenizer and sampling.
22. Generate coherent text.
23. Profile.
24. Optimize decode operations.
25. Introduce quantization.
26. Introduce paged KV cache.
27. Add local attention.
28. Add recurrent working memory.
29. Add graph-memory retrieval.
30. Add learned routing.

---

## 36. Definition of Success

The standard inference engine is successful when it can:

- Load a real model.
- Run prefill and decode on Metal.
- Generate correct text.
- Maintain stable memory usage.
- Expose reliable profiling.
- Support quantized weights.
- Handle several sessions.

The Atlas memory engine is successful when it can:

- Bound active local KV memory.
- Persist information outside the token context.
- Retrieve old information as latent memory.
- Fuse memory directly into hidden states.
- Maintain working memory across long sessions.
- Produce comparable long-context behavior without retaining the full KV history.

---

## 37. Final Architecture Target

```text
Audio / Video / Text
          │
          ▼
     Perception Encoders
          │
          ▼
      Event Stream
          │
          ▼
      Atlas Runtime
          │
    ┌─────┼─────────────┐
    ▼     ▼             ▼
 Local  Recurrent    Graph
 KV     Working      Memory
 Cache  Memory
    └─────┼─────────────┘
          ▼
     Memory Router
          │
          ▼
   Rust Transformer
          │
          ▼
      Metal Backend
          │
          ▼
   Apple Silicon GPU
```

The immediate engineering priority is to build the bottom half first:

```text
Rust Transformer
      │
Metal Backend
      │
Apple Silicon GPU
```

Only after this path is correct, measurable, and stable should the persistent Atlas memory architecture be added.
