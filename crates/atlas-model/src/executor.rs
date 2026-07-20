//! Phase-6 prefill/decode execution plans.
//!
//! The plans own only immutable shape and residency decisions.  A session owns
//! the mutable KV cache, which keeps request state out of the model itself.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use atlas_core::{GgufTensorType, QuantFormat};
use atlas_metal::GpuBuffer;

use crate::kv_cache::{ContiguousKvCache, KvCacheConfig, LayerKv, SessionId};
use crate::{AtlasModel, Generation, LayerTrace, argmax, gather_head, scatter_head};

/// A decode boundary that can be read back by the diagnostic resident trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentStage {
    Embedding,
    InputNorm,
    Q,
    K,
    V,
    RopeQ,
    RopeK,
    Attention,
    AttentionOutputProjection,
    AttentionResidualInput,
    AttentionResidual,
    PostAttentionNorm,
    Gate,
    Up,
    SiLU,
    MlpProduct,
    MlpDownProjection,
    MlpResidualInput,
    MlpResidual,
    FinalNorm,
    Logits,
}

impl std::fmt::Display for ResidentStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Embedding => "embedding",
            Self::InputNorm => "input_norm",
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::RopeQ => "rope_q",
            Self::RopeK => "rope_k",
            Self::Attention => "attention",
            Self::AttentionOutputProjection => "attention_output_projection",
            Self::AttentionResidualInput => "attention_residual_input",
            Self::AttentionResidual => "attention_residual",
            Self::PostAttentionNorm => "post_attention_norm",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::SiLU => "silu",
            Self::MlpProduct => "mlp_product",
            Self::MlpDownProjection => "mlp_down_projection",
            Self::MlpResidualInput => "mlp_residual_input",
            Self::MlpResidual => "mlp_residual",
            Self::FinalNorm => "final_norm",
            Self::Logits => "logits",
        };
        f.write_str(name)
    }
}

/// Per-stage FP32 comparison limits for the resident trace.
///
/// Phase 8.3 starts from the caller's elementwise threshold for every stage.
/// A reduction-specific allowance may only be added after its input shape and
/// error bound are established by the hardware parity suite.
pub fn resident_stage_tolerance(stage: ResidentStage, default: f32) -> f32 {
    let _ = stage;
    default
}

/// The first observed difference at a resident decode diagnostic boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct StageComparison {
    pub prompt_token_index: usize,
    pub stage: ResidentStage,
    pub layer: Option<usize>,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub first_failing_index: Option<usize>,
    pub expected: f32,
    pub actual: f32,
}

/// Compares a single stage deterministically. Non-finite values and unequal
/// lengths are failures even when their bit patterns happen to match.
pub fn compare_stage(
    prompt_token_index: usize,
    stage: ResidentStage,
    layer: Option<usize>,
    expected_values: &[f32],
    actual_values: &[f32],
    tolerance: f32,
) -> Option<StageComparison> {
    let common = expected_values.len().min(actual_values.len());
    let mut max_abs_error = 0.0f32;
    let mut first = None;
    for index in 0..common {
        let left = expected_values[index];
        let right = actual_values[index];
        let error = (left - right).abs();
        if !left.is_finite() || !right.is_finite() {
            max_abs_error = f32::INFINITY;
        } else {
            max_abs_error = max_abs_error.max(error);
        }
        if first.is_none() && (!left.is_finite() || !right.is_finite() || error > tolerance) {
            first = Some((index, left, right));
        }
    }
    if first.is_none() && expected_values.len() != actual_values.len() {
        first = Some((
            common,
            expected_values.get(common).copied().unwrap_or(f32::NAN),
            actual_values.get(common).copied().unwrap_or(f32::NAN),
        ));
        max_abs_error = f32::INFINITY;
    }
    first.map(|(first_failing_index, expected, actual)| StageComparison {
        prompt_token_index,
        stage,
        layer,
        element_count: expected_values.len().max(actual_values.len()),
        max_abs_error,
        first_failing_index: Some(first_failing_index),
        expected,
        actual,
    })
}

#[derive(Debug)]
struct StageSnapshot {
    stage: ResidentStage,
    layer: Option<usize>,
    values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorConfig {
    pub session: SessionId,
    pub max_context: usize,
    /// The requested weight format.  FP16 denotes the existing FP32 model
    /// tensors; packed formats are reserved for the Phase-5 packed kernels.
    pub quant_format: QuantFormat,
    /// Resident is the production default. Reference is restricted to explicit
    /// parity and diagnostic execution.
    pub mode: ExecutorMode,
    /// Downloading logits defeats token-only decode readback, so it is opt-in.
    pub logits_readback: LogitsReadback,
    /// Normal inference stops on EOS. Benchmark-only callers may disable this
    /// to measure a fixed decode workload without altering product behavior.
    pub stop_on_eos: bool,
    /// Collect opt-in resident decode stage timings. This must not alter the
    /// resident command boundary or normal generation behavior.
    pub resident_decode_profile: bool,
    /// Hidden resident attention selector. `LegacyThreePass` remains the
    /// production-safe path while Q8 parity is being restored; `Fused` is an
    /// explicit diagnostic/acceptance selector only.
    #[doc(hidden)]
    pub resident_attention_path: ResidentAttentionPath,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutorMode {
    #[default]
    Reference,
    Resident,
}

/// Resident attention implementation. The fused path is retained for focused
/// parity and performance acceptance. It must not become the normal Resident
/// path until the exact-token Q8 golden suite passes.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResidentAttentionPath {
    #[default]
    LegacyThreePass,
    Fused,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LogitsReadback {
    #[default]
    SelectedToken,
    FinalLogits,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            session: SessionId(0),
            max_context: 1024,
            quant_format: QuantFormat::Fp16,
            mode: ExecutorMode::Resident,
            logits_readback: LogitsReadback::SelectedToken,
            stop_on_eos: true,
            resident_decode_profile: false,
            resident_attention_path: ResidentAttentionPath::LegacyThreePass,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillPlan {
    pub max_tokens: usize,
    pub pipeline_count: usize,
    pub quant_format: QuantFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodePlan {
    pub max_context: usize,
    pub pipeline_count: usize,
    pub quant_format: QuantFormat,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutorMetrics {
    pub cpu_encode: Duration,
    pub prefill: Duration,
    pub decode: Duration,
    pub ttft: Duration,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub decode_latencies: Vec<Duration>,
    pub pipeline_count: usize,
    pub post_warmup_pipeline_count: usize,
    pub post_warmup_allocations: u64,
    /// End-to-end request wall time, including token delivery.
    pub host_wall_time: Duration,
    /// Sum of Metal command-buffer GPU intervals observed during this request.
    pub gpu_execution_time: Duration,
    /// Command buffers submitted during this request.
    pub command_buffer_count: u64,
    pub prefill_command_buffer_count: u64,
    pub decode_command_buffer_count: u64,
    /// Immutable model parameter bytes uploaded while constructing this executor.
    pub weight_upload_bytes: u64,
    pub weight_upload_elapsed: Duration,
    /// Bytes copied from Metal buffers back to CPU-visible vectors.
    pub readback_bytes: u64,
    /// GPU-visible session arenas and immutable resident model weights.
    pub resident_bytes: u64,
    pub resident_arena_allocations: u64,
    /// Present only when resident decode profiling was explicitly enabled.
    pub resident_decode_profile: Option<ResidentDecodeProfile>,
}

/// Machine-readable, opt-in timing breakdown for post-prefill resident decode.
/// Metal exposes GPU and scheduling timestamps at the one-command-buffer
/// boundary, so those totals are attributed to stages by their dispatch count;
/// CPU encoding is measured directly around each stage's encoding work.
#[derive(Debug, Clone, Default)]
pub struct ResidentDecodeProfile {
    pub tokens: usize,
    pub attention_implementation: &'static str,
    pub fused_attention_dispatches: u64,
    pub embedding: ResidentDecodeStageMetrics,
    pub attention: ResidentDecodeStageMetrics,
    pub packed_projections: ResidentDecodeStageMetrics,
    pub mlp: ResidentDecodeStageMetrics,
    pub lm_head: ResidentDecodeStageMetrics,
    pub token_readback: ResidentDecodeStageMetrics,
    pub command_buffer_schedule: Duration,
    pub gpu_execution: Duration,
    /// Ordered exact per-dispatch records. These are populated only when the
    /// profiler isolated each kernel in its own Metal command buffer.
    pub trace: Vec<ResidentKernelTrace>,
    pub command_buffer_count: u64,
}

#[derive(Debug, Clone)]
pub struct ResidentKernelTrace {
    pub token_index: usize,
    pub phase: &'static str,
    pub layer: Option<usize>,
    pub stage: &'static str,
    pub kernel: &'static str,
    pub cpu_encode: Duration,
    pub gpu_execution: Option<Duration>,
    pub command_buffer_schedule: Duration,
    pub threads: usize,
    pub threadgroups: usize,
    pub threads_per_threadgroup: usize,
    pub readback_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct ResidentDecodeStageMetrics {
    pub cpu_encode: Duration,
    pub dispatches: u64,
    pub command_buffer_schedule: Duration,
    pub gpu_execution: Duration,
}

impl ResidentDecodeProfile {
    fn record(stage: &mut ResidentDecodeStageMetrics, elapsed: Duration, dispatches: u64) {
        stage.cpu_encode += elapsed;
        stage.dispatches += dispatches;
    }

    fn attribute_command_buffer(&mut self, timing: atlas_metal::DispatchTiming) {
        self.tokens += 1;
        let total_dispatches = self.embedding.dispatches
            + self.attention.dispatches
            + self.packed_projections.dispatches
            + self.mlp.dispatches
            + self.lm_head.dispatches;
        if total_dispatches == 0 {
            return;
        }
        self.command_buffer_schedule += timing.command_buffer_schedule;
        let gpu_time = timing.gpu_time.unwrap_or_default();
        self.gpu_execution += gpu_time;
        for stage in [
            &mut self.embedding,
            &mut self.attention,
            &mut self.packed_projections,
            &mut self.mlp,
            &mut self.lm_head,
        ] {
            let ratio = stage.dispatches as f64 / total_dispatches as f64;
            stage.command_buffer_schedule += timing.command_buffer_schedule.mul_f64(ratio);
            stage.gpu_execution += gpu_time.mul_f64(ratio);
        }
    }
}

impl ExecutorMetrics {
    pub fn prefill_tokens_per_second(&self) -> f64 {
        rate(self.prefill_tokens, self.prefill)
    }
    pub fn decode_tokens_per_second(&self) -> f64 {
        rate(self.decode_tokens, self.decode)
    }
    pub fn decode_p50(&self) -> Duration {
        percentile(&self.decode_latencies, 0.50)
    }
    pub fn decode_p95(&self) -> Duration {
        percentile(&self.decode_latencies, 0.95)
    }
}

#[derive(Debug, Clone)]
pub struct ExecutorGeneration {
    pub generation: Generation,
    pub metrics: ExecutorMetrics,
    pub finish_reason: GenerationFinishReason,
}

/// Why a streaming greedy generation completed normally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    Eos,
    MaxTokens,
}

/// An item delivered while greedy generation is in progress.
///
/// A stream emits zero or more [`Token`](Self::Token) events followed by
/// exactly one terminal [`Finished`](Self::Finished) or [`Failed`](Self::Failed)
/// event, unless the callback itself returns an error.
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    Token {
        token_id: u32,
        text: String,
        /// `None` for the first token, whose delivery time is reported by TTFT.
        decode_latency: Option<Duration>,
    },
    Finished {
        reason: GenerationFinishReason,
        metrics: ExecutorMetrics,
    },
    Failed {
        message: String,
    },
}

/// Immutable plans plus session-local cache state.  The executor never
/// rebuilds a Metal runtime or pipelines during token generation.
pub struct AtlasExecutor<'a> {
    model: &'a AtlasModel,
    prefill_plan: PrefillPlan,
    decode_plan: DecodePlan,
    caches: Vec<ContiguousKvCache>,
    resident: Option<ResidentExecutor>,
    mode: ExecutorMode,
    logits_readback: LogitsReadback,
    stop_on_eos: bool,
    resident_decode_profile: bool,
    weight_upload_bytes: u64,
    weight_upload_elapsed: Duration,
}

struct ResidentExecutor {
    weights: std::collections::HashMap<String, GpuBuffer>,
    kv: Vec<GpuBuffer>,
    token: GpuBuffer,
    position: GpuBuffer,
    selected: GpuBuffer,
    state: GpuBuffer,
    work: GpuBuffer,
    residual: GpuBuffer,
    norm: GpuBuffer,
    q: GpuBuffer,
    q_rot: GpuBuffer,
    rope_input: GpuBuffer,
    rope_output: GpuBuffer,
    k: GpuBuffer,
    k_rot: GpuBuffer,
    v: GpuBuffer,
    attention: GpuBuffer,
    attention_scores: Option<GpuBuffer>,
    attention_weights: Option<GpuBuffer>,
    attention_key_count: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    activated: GpuBuffer,
    product: GpuBuffer,
    logits: GpuBuffer,
    hidden: GpuBuffer,
    intermediate: GpuBuffer,
    heads: GpuBuffer,
    kv_heads: GpuBuffer,
    head_dim: GpuBuffer,
    kv_width: GpuBuffer,
    capacity: GpuBuffer,
    vocab: GpuBuffer,
    epsilon: GpuBuffer,
    rope_cos: GpuBuffer,
    rope_sin: GpuBuffer,
    rope_cos_host: Vec<f32>,
    rope_sin_host: Vec<f32>,
    one: GpuBuffer,
    max_context: usize,
    position_index: usize,
    logits_readback: LogitsReadback,
    attention_path: ResidentAttentionPath,
}

impl ResidentExecutor {
    fn new(
        model: &AtlasModel,
        capacity: usize,
        logits_readback: LogitsReadback,
        attention_path: ResidentAttentionPath,
    ) -> Result<Self> {
        let runtime = model.ops.runtime();
        let c = &model.config;
        let h = c.hidden_size;
        let kv_width = c.num_key_value_heads * c.head_dim();
        let allocate_f32 = |count: usize| {
            runtime
                .allocate(
                    count
                        .checked_mul(4)
                        .ok_or_else(|| anyhow::anyhow!("resident arena size overflow"))?,
                )
                .map_err(Into::into)
        };
        let kv = (0..c.num_hidden_layers)
            .map(|_| allocate_f32(2 * capacity * kv_width))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights: model.resident_weights_snapshot(),
            kv,
            token: runtime.allocate(4)?,
            position: runtime.allocate(4)?,
            selected: runtime.allocate(4)?,
            state: allocate_f32(h)?,
            work: allocate_f32(h)?,
            residual: allocate_f32(h)?,
            norm: allocate_f32(h)?,
            q: allocate_f32(h)?,
            q_rot: allocate_f32(h)?,
            rope_input: allocate_f32(h)?,
            rope_output: allocate_f32(h)?,
            k: allocate_f32(kv_width)?,
            k_rot: allocate_f32(kv_width)?,
            v: allocate_f32(kv_width)?,
            attention: allocate_f32(h)?,
            attention_scores: (attention_path == ResidentAttentionPath::LegacyThreePass)
                .then(|| allocate_f32(c.num_attention_heads * capacity))
                .transpose()?,
            attention_weights: (attention_path == ResidentAttentionPath::LegacyThreePass)
                .then(|| allocate_f32(c.num_attention_heads * capacity))
                .transpose()?,
            attention_key_count: runtime.allocate(4)?,
            gate: allocate_f32(c.intermediate_size)?,
            up: allocate_f32(c.intermediate_size)?,
            activated: allocate_f32(c.intermediate_size)?,
            product: allocate_f32(c.intermediate_size)?,
            logits: allocate_f32(c.vocab_size)?,
            hidden: runtime.upload_u32(&[u32::try_from(h)?])?,
            intermediate: runtime.upload_u32(&[u32::try_from(c.intermediate_size)?])?,
            heads: runtime.upload_u32(&[u32::try_from(c.num_attention_heads)?])?,
            kv_heads: runtime.upload_u32(&[u32::try_from(c.num_key_value_heads)?])?,
            head_dim: runtime.upload_u32(&[u32::try_from(c.head_dim())?])?,
            kv_width: runtime.upload_u32(&[u32::try_from(kv_width)?])?,
            capacity: runtime.upload_u32(&[u32::try_from(capacity)?])?,
            vocab: runtime.upload_u32(&[u32::try_from(c.vocab_size)?])?,
            epsilon: runtime.upload_f32(&[c.rms_norm_eps])?,
            rope_cos: allocate_f32(c.head_dim() / 2)?,
            rope_sin: allocate_f32(c.head_dim() / 2)?,
            rope_cos_host: vec![0.0; c.head_dim() / 2],
            rope_sin_host: vec![0.0; c.head_dim() / 2],
            one: runtime.upload_u32(&[1])?,
            max_context: capacity,
            position_index: 0,
            logits_readback,
            attention_path,
        })
    }

    fn reset(&mut self) {
        self.position_index = 0;
    }
    fn allocations(&self) -> u64 {
        30 + self.kv.len() as u64
    }
    fn resident_bytes(&self) -> u64 {
        let buffers = [
            &self.token,
            &self.position,
            &self.selected,
            &self.state,
            &self.work,
            &self.residual,
            &self.norm,
            &self.q,
            &self.q_rot,
            &self.rope_input,
            &self.rope_output,
            &self.k,
            &self.k_rot,
            &self.v,
            &self.attention,
            &self.attention_key_count,
            &self.gate,
            &self.up,
            &self.activated,
            &self.product,
            &self.logits,
            &self.hidden,
            &self.intermediate,
            &self.heads,
            &self.kv_heads,
            &self.head_dim,
            &self.kv_width,
            &self.capacity,
            &self.vocab,
            &self.epsilon,
            &self.rope_cos,
            &self.rope_sin,
            &self.one,
        ];
        self.weights
            .values()
            .map(|buffer| buffer.bytes() as u64)
            .sum::<u64>()
            + self
                .kv
                .iter()
                .map(|buffer| buffer.bytes() as u64)
                .sum::<u64>()
            + buffers
                .iter()
                .map(|buffer| buffer.bytes() as u64)
                .sum::<u64>()
            + self
                .attention_scores
                .as_ref()
                .map_or(0, |buffer| buffer.bytes() as u64)
            + self
                .attention_weights
                .as_ref()
                .map_or(0, |buffer| buffer.bytes() as u64)
    }
    fn weight(&self, name: &str) -> Result<&GpuBuffer> {
        self.weights
            .get(name)
            .with_context(|| format!("resident weight missing `{name}`"))
    }

    fn trace_dispatch(
        &self,
        runtime: &atlas_metal::MetalRuntime,
        kernel: &'static str,
        buffers: &[&GpuBuffer],
        count: usize,
    ) -> Result<()> {
        let mut command = runtime.begin_resident_command()?;
        command.dispatch_1d(kernel, buffers, count)?;
        command.finish()?;
        Ok(())
    }

    /// Match the reference RoPE oracle's host `powf`/trigonometric evaluation
    /// exactly, then bind those values to the half-split resident kernel. This
    /// avoids CPU/GPU math-library drift at non-zero positions without adding a
    /// decode-time allocation or command buffer.
    fn write_rope_tables(&mut self, model: &AtlasModel) -> Result<()> {
        let head_dim = model.config.head_dim();
        for pair in 0..head_dim / 2 {
            let angle = self.position_index as f32
                / model
                    .config
                    .rope_theta
                    .powf((pair * 2) as f32 / head_dim as f32);
            self.rope_cos_host[pair] = angle.cos();
            self.rope_sin_host[pair] = angle.sin();
        }
        let runtime = model.ops.runtime();
        runtime.write_f32(&self.rope_cos, &self.rope_cos_host)?;
        runtime.write_f32(&self.rope_sin, &self.rope_sin_host)?;
        Ok(())
    }

    fn trace_rope(
        &self,
        runtime: &atlas_metal::MetalRuntime,
        input: &GpuBuffer,
        output: &GpuBuffer,
        heads: &GpuBuffer,
        count: usize,
    ) -> Result<()> {
        self.trace_dispatch(
            runtime,
            "rope_half_to_interleaved_f32",
            &[input, &self.rope_input, heads, &self.head_dim],
            count,
        )?;
        self.trace_dispatch(
            runtime,
            "rope_f32",
            &[
                &self.rope_input,
                &self.rope_cos,
                &self.rope_sin,
                &self.rope_output,
                &self.head_dim,
            ],
            count,
        )?;
        self.trace_dispatch(
            runtime,
            "rope_interleaved_to_half_f32",
            &[&self.rope_output, output, heads, &self.head_dim],
            count,
        )
    }

    fn write_attention_key_count(&self, runtime: &atlas_metal::MetalRuntime) -> Result<()> {
        runtime.write_u32(
            &self.attention_key_count,
            &[u32::try_from(self.position_index + 1)?],
        )?;
        Ok(())
    }

    fn trace_attention(
        &self,
        runtime: &atlas_metal::MetalRuntime,
        cache: &GpuBuffer,
        heads: usize,
        hidden: usize,
    ) -> Result<()> {
        let keys = self.position_index + 1;
        if self.attention_path == ResidentAttentionPath::Fused {
            let mut command = runtime.begin_resident_command()?;
            command.dispatch_threadgroups_1d(
                "attention_decode_fused_f32",
                &[
                    &self.q_rot,
                    cache,
                    &self.attention,
                    &self.heads,
                    &self.kv_heads,
                    &self.head_dim,
                    &self.capacity,
                    &self.attention_key_count,
                ],
                heads,
                128,
            )?;
            return command.finish().map(|_| ()).map_err(Into::into);
        }
        let scores = self.attention_scores.as_ref().expect("legacy score buffer");
        let weights = self
            .attention_weights
            .as_ref()
            .expect("legacy weight buffer");
        self.trace_dispatch(
            runtime,
            "attention_scores_resident_f32",
            &[
                &self.q_rot,
                cache,
                scores,
                &self.heads,
                &self.kv_heads,
                &self.head_dim,
                &self.capacity,
                &self.attention_key_count,
            ],
            heads * keys,
        )?;
        self.trace_dispatch(
            runtime,
            "masked_softmax_resident_f32",
            &[
                scores,
                weights,
                &self.heads,
                &self.capacity,
                &self.attention_key_count,
            ],
            heads,
        )?;
        self.trace_dispatch(
            runtime,
            "attention_values_resident_f32",
            &[
                weights,
                cache,
                &self.attention,
                &self.heads,
                &self.kv_heads,
                &self.head_dim,
                &self.capacity,
                &self.attention_key_count,
            ],
            hidden,
        )
    }

    fn trace_token(&mut self, model: &AtlasModel, token: u32) -> Result<Vec<StageSnapshot>> {
        ensure!(
            self.position_index < self.max_context,
            "executor context exhausted"
        );
        let runtime = model.ops.runtime();
        let c = &model.config;
        let h = c.hidden_size;
        let kv_width = c.num_key_value_heads * c.head_dim();
        runtime.write_u32(&self.token, &[token])?;
        runtime.write_u32(&self.position, &[u32::try_from(self.position_index)?])?;
        self.write_rope_tables(model)?;
        self.write_attention_key_count(runtime)?;
        let mut snapshots = Vec::new();
        let capture = |stage, layer, buffer: &GpuBuffer, count| -> Result<StageSnapshot> {
            Ok(StageSnapshot {
                stage,
                layer,
                values: runtime.read_f32(buffer, count)?,
            })
        };
        let embed = self.weight("model.embed_tokens.weight")?;
        self.trace_dispatch(
            runtime,
            "embedding_lookup_f32",
            &[
                embed,
                &self.token,
                &self.state,
                &self.vocab,
                &self.hidden,
                &self.one,
            ],
            h,
        )?;
        snapshots.push(capture(ResidentStage::Embedding, None, &self.state, h)?);
        for layer in 0..c.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            self.trace_dispatch(
                runtime,
                "rms_norm_f32",
                &[
                    &self.state,
                    self.weight(&format!("{p}.input_layernorm.weight"))?,
                    &self.norm,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
            )?;
            snapshots.push(capture(
                ResidentStage::InputNorm,
                Some(layer),
                &self.norm,
                h,
            )?);
            for (name, output, width, stage) in [
                ("q_proj", &self.q, h, ResidentStage::Q),
                ("k_proj", &self.k, kv_width, ResidentStage::K),
                ("v_proj", &self.v, kv_width, ResidentStage::V),
            ] {
                self.trace_dispatch(
                    runtime,
                    "matvec_f32",
                    &[
                        &self.norm,
                        self.weight(&format!("{p}.self_attn.{name}.weight"))?,
                        output,
                        &self.hidden,
                        if width == h {
                            &self.hidden
                        } else {
                            &self.kv_width
                        },
                    ],
                    width,
                )?;
                snapshots.push(capture(stage, Some(layer), output, width)?);
            }
            self.trace_rope(runtime, &self.q, &self.q_rot, &self.heads, h / 2)?;
            snapshots.push(capture(ResidentStage::RopeQ, Some(layer), &self.q_rot, h)?);
            self.trace_rope(runtime, &self.k, &self.k_rot, &self.kv_heads, kv_width / 2)?;
            snapshots.push(capture(
                ResidentStage::RopeK,
                Some(layer),
                &self.k_rot,
                kv_width,
            )?);
            self.trace_dispatch(
                runtime,
                "kv_append_decode_f32",
                &[
                    &self.k_rot,
                    &self.v,
                    &self.kv[layer],
                    &self.kv_width,
                    &self.capacity,
                    &self.position,
                ],
                kv_width,
            )?;
            self.trace_attention(runtime, &self.kv[layer], c.num_attention_heads, h)?;
            snapshots.push(capture(
                ResidentStage::Attention,
                Some(layer),
                &self.attention,
                h,
            )?);
            self.trace_dispatch(
                runtime,
                "matvec_f32",
                &[
                    &self.attention,
                    self.weight(&format!("{p}.self_attn.o_proj.weight"))?,
                    &self.work,
                    &self.hidden,
                    &self.hidden,
                ],
                h,
            )?;
            snapshots.push(capture(
                ResidentStage::AttentionOutputProjection,
                Some(layer),
                &self.work,
                h,
            )?);
            snapshots.push(capture(
                ResidentStage::AttentionResidualInput,
                Some(layer),
                &self.state,
                h,
            )?);
            self.trace_dispatch(
                runtime,
                "vector_add_f32",
                &[&self.state, &self.work, &self.residual, &self.hidden],
                h,
            )?;
            snapshots.push(capture(
                ResidentStage::AttentionResidual,
                Some(layer),
                &self.residual,
                h,
            )?);
            self.trace_dispatch(
                runtime,
                "rms_norm_f32",
                &[
                    &self.residual,
                    self.weight(&format!("{p}.post_attention_layernorm.weight"))?,
                    &self.norm,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
            )?;
            snapshots.push(capture(
                ResidentStage::PostAttentionNorm,
                Some(layer),
                &self.norm,
                h,
            )?);
            for (name, output, stage) in [
                ("gate_proj", &self.gate, ResidentStage::Gate),
                ("up_proj", &self.up, ResidentStage::Up),
            ] {
                self.trace_dispatch(
                    runtime,
                    "matvec_f32",
                    &[
                        &self.norm,
                        self.weight(&format!("{p}.mlp.{name}.weight"))?,
                        output,
                        &self.hidden,
                        &self.intermediate,
                    ],
                    c.intermediate_size,
                )?;
                snapshots.push(capture(stage, Some(layer), output, c.intermediate_size)?);
            }
            self.trace_dispatch(
                runtime,
                "silu_f32",
                &[&self.gate, &self.activated, &self.intermediate],
                c.intermediate_size,
            )?;
            snapshots.push(capture(
                ResidentStage::SiLU,
                Some(layer),
                &self.activated,
                c.intermediate_size,
            )?);
            self.trace_dispatch(
                runtime,
                "vector_multiply_f32",
                &[&self.activated, &self.up, &self.product, &self.intermediate],
                c.intermediate_size,
            )?;
            snapshots.push(capture(
                ResidentStage::MlpProduct,
                Some(layer),
                &self.product,
                c.intermediate_size,
            )?);
            self.trace_dispatch(
                runtime,
                "matvec_f32",
                &[
                    &self.product,
                    self.weight(&format!("{p}.mlp.down_proj.weight"))?,
                    &self.work,
                    &self.intermediate,
                    &self.hidden,
                ],
                h,
            )?;
            snapshots.push(capture(
                ResidentStage::MlpDownProjection,
                Some(layer),
                &self.work,
                h,
            )?);
            snapshots.push(capture(
                ResidentStage::MlpResidualInput,
                Some(layer),
                &self.residual,
                h,
            )?);
            self.trace_dispatch(
                runtime,
                "vector_add_f32",
                &[&self.residual, &self.work, &self.state, &self.hidden],
                h,
            )?;
            snapshots.push(capture(
                ResidentStage::MlpResidual,
                Some(layer),
                &self.state,
                h,
            )?);
        }
        self.trace_dispatch(
            runtime,
            "rms_norm_f32",
            &[
                &self.state,
                self.weight("model.norm.weight")?,
                &self.norm,
                &self.hidden,
                &self.epsilon,
            ],
            1,
        )?;
        snapshots.push(capture(ResidentStage::FinalNorm, None, &self.norm, h)?);
        let lm_head = if c.tie_word_embeddings {
            embed
        } else {
            self.weight("lm_head.weight")?
        };
        self.trace_dispatch(
            runtime,
            "matvec_f32",
            &[&self.norm, lm_head, &self.logits, &self.hidden, &self.vocab],
            c.vocab_size,
        )?;
        snapshots.push(capture(
            ResidentStage::Logits,
            None,
            &self.logits,
            c.vocab_size,
        )?);
        self.position_index += 1;
        Ok(snapshots)
    }

    fn forward_token(
        &mut self,
        model: &AtlasModel,
        token: u32,
        mut profile: Option<&mut ResidentDecodeProfile>,
    ) -> Result<TokenStep> {
        ensure!(
            self.position_index < self.max_context,
            "executor context exhausted"
        );
        let runtime = model.ops.runtime();
        let hidden_size = model.config.hidden_size;
        let kv_width = model.config.num_key_value_heads * model.config.head_dim();
        runtime.write_u32(&self.token, &[token])?;
        runtime.write_u32(&self.position, &[u32::try_from(self.position_index)?])?;
        self.write_rope_tables(model)?;
        self.write_attention_key_count(runtime)?;
        let token_index = self.position_index;
        let mut command = runtime.begin_resident_command_with_exact_timing(profile.is_some())?;
        let embed = self.weight("model.embed_tokens.weight")?;
        let stage_started = Instant::now();
        command.dispatch_1d(
            resident_embedding_kernel(model, "model.embed_tokens.weight")?,
            &[
                embed,
                &self.token,
                &self.state,
                &self.vocab,
                &self.hidden,
                &self.one,
            ],
            model.config.hidden_size,
        )?;
        if let Some(profile) = profile.as_deref_mut() {
            ResidentDecodeProfile::record(&mut profile.embedding, stage_started.elapsed(), 1);
        }
        for layer in 0..model.config.num_hidden_layers {
            let p = format!("model.layers.{layer}");
            command.dispatch_1d(
                "rms_norm_f32",
                &[
                    &self.state,
                    self.weight(&format!("{p}.input_layernorm.weight"))?,
                    &self.norm,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
            )?;
            let stage_started = Instant::now();
            for (name, output, width_buffer, output_width) in [
                ("q_proj", &self.q, &self.hidden, hidden_size),
                ("k_proj", &self.k, &self.kv_width, kv_width),
                ("v_proj", &self.v, &self.kv_width, kv_width),
            ] {
                let weight_name = format!("{p}.self_attn.{name}.weight");
                command.dispatch_1d(
                    resident_matvec_kernel(model, &weight_name)?,
                    &[
                        &self.norm,
                        self.weight(&weight_name)?,
                        output,
                        &self.hidden,
                        width_buffer,
                    ],
                    output_width,
                )?;
            }
            if let Some(profile) = profile.as_deref_mut() {
                ResidentDecodeProfile::record(
                    &mut profile.packed_projections,
                    stage_started.elapsed(),
                    3,
                );
            }
            let stage_started = Instant::now();
            command.dispatch_1d(
                "rope_half_to_interleaved_f32",
                &[&self.q, &self.rope_input, &self.heads, &self.head_dim],
                hidden_size / 2,
            )?;
            command.dispatch_1d(
                "rope_f32",
                &[
                    &self.rope_input,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.rope_output,
                    &self.head_dim,
                ],
                hidden_size / 2,
            )?;
            command.dispatch_1d(
                "rope_interleaved_to_half_f32",
                &[&self.rope_output, &self.q_rot, &self.heads, &self.head_dim],
                hidden_size / 2,
            )?;
            command.dispatch_1d(
                "rope_half_to_interleaved_f32",
                &[&self.k, &self.rope_input, &self.kv_heads, &self.head_dim],
                kv_width / 2,
            )?;
            command.dispatch_1d(
                "rope_f32",
                &[
                    &self.rope_input,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.rope_output,
                    &self.head_dim,
                ],
                kv_width / 2,
            )?;
            command.dispatch_1d(
                "rope_interleaved_to_half_f32",
                &[
                    &self.rope_output,
                    &self.k_rot,
                    &self.kv_heads,
                    &self.head_dim,
                ],
                kv_width / 2,
            )?;
            command.dispatch_1d(
                "kv_append_decode_f32",
                &[
                    &self.k_rot,
                    &self.v,
                    &self.kv[layer],
                    &self.kv_width,
                    &self.capacity,
                    &self.position,
                ],
                kv_width,
            )?;
            let attention_keys = self.position_index + 1;
            match self.attention_path {
                ResidentAttentionPath::Fused => command.dispatch_threadgroups_1d(
                    "attention_decode_fused_f32",
                    &[
                        &self.q_rot,
                        &self.kv[layer],
                        &self.attention,
                        &self.heads,
                        &self.kv_heads,
                        &self.head_dim,
                        &self.capacity,
                        &self.attention_key_count,
                    ],
                    model.config.num_attention_heads,
                    128,
                )?,
                ResidentAttentionPath::LegacyThreePass => {
                    let scores = self.attention_scores.as_ref().expect("legacy score buffer");
                    let weights = self
                        .attention_weights
                        .as_ref()
                        .expect("legacy weight buffer");
                    command.dispatch_1d(
                        "attention_scores_resident_f32",
                        &[
                            &self.q_rot,
                            &self.kv[layer],
                            scores,
                            &self.heads,
                            &self.kv_heads,
                            &self.head_dim,
                            &self.capacity,
                            &self.attention_key_count,
                        ],
                        model.config.num_attention_heads * attention_keys,
                    )?;
                    command.dispatch_1d(
                        "masked_softmax_resident_f32",
                        &[
                            scores,
                            weights,
                            &self.heads,
                            &self.capacity,
                            &self.attention_key_count,
                        ],
                        model.config.num_attention_heads,
                    )?;
                    command.dispatch_1d(
                        "attention_values_resident_f32",
                        &[
                            weights,
                            &self.kv[layer],
                            &self.attention,
                            &self.heads,
                            &self.kv_heads,
                            &self.head_dim,
                            &self.capacity,
                            &self.attention_key_count,
                        ],
                        model.config.hidden_size,
                    )?;
                }
            }
            let attention_output_name = format!("{p}.self_attn.o_proj.weight");
            command.dispatch_1d(
                resident_matvec_kernel(model, &attention_output_name)?,
                &[
                    &self.attention,
                    self.weight(&attention_output_name)?,
                    &self.work,
                    &self.hidden,
                    &self.hidden,
                ],
                model.config.hidden_size,
            )?;
            command.dispatch_1d(
                "vector_add_f32",
                &[&self.state, &self.work, &self.residual, &self.hidden],
                model.config.hidden_size,
            )?;
            if let Some(profile) = profile.as_deref_mut() {
                let dispatches = match self.attention_path {
                    ResidentAttentionPath::Fused => {
                        profile.fused_attention_dispatches += 1;
                        10
                    }
                    ResidentAttentionPath::LegacyThreePass => 12,
                };
                ResidentDecodeProfile::record(
                    &mut profile.attention,
                    stage_started.elapsed(),
                    dispatches,
                );
            }
            command.dispatch_1d(
                "rms_norm_f32",
                &[
                    &self.residual,
                    self.weight(&format!("{p}.post_attention_layernorm.weight"))?,
                    &self.norm,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
            )?;
            let gate_name = format!("{p}.mlp.gate_proj.weight");
            let stage_started = Instant::now();
            command.dispatch_1d(
                resident_matvec_kernel(model, &gate_name)?,
                &[
                    &self.norm,
                    self.weight(&gate_name)?,
                    &self.gate,
                    &self.hidden,
                    &self.intermediate,
                ],
                model.config.intermediate_size,
            )?;
            let up_name = format!("{p}.mlp.up_proj.weight");
            command.dispatch_1d(
                resident_matvec_kernel(model, &up_name)?,
                &[
                    &self.norm,
                    self.weight(&up_name)?,
                    &self.up,
                    &self.hidden,
                    &self.intermediate,
                ],
                model.config.intermediate_size,
            )?;
            command.dispatch_1d(
                "silu_f32",
                &[&self.gate, &self.activated, &self.intermediate],
                model.config.intermediate_size,
            )?;
            command.dispatch_1d(
                "vector_multiply_f32",
                &[&self.activated, &self.up, &self.product, &self.intermediate],
                model.config.intermediate_size,
            )?;
            let down_name = format!("{p}.mlp.down_proj.weight");
            command.dispatch_1d(
                resident_matvec_kernel(model, &down_name)?,
                &[
                    &self.product,
                    self.weight(&down_name)?,
                    &self.work,
                    &self.intermediate,
                    &self.hidden,
                ],
                model.config.hidden_size,
            )?;
            command.dispatch_1d(
                "vector_add_f32",
                &[&self.residual, &self.work, &self.state, &self.hidden],
                model.config.hidden_size,
            )?;
            if let Some(profile) = profile.as_deref_mut() {
                ResidentDecodeProfile::record(&mut profile.mlp, stage_started.elapsed(), 6);
            }
        }
        let stage_started = Instant::now();
        command.dispatch_1d(
            "rms_norm_f32",
            &[
                &self.state,
                self.weight("model.norm.weight")?,
                &self.norm,
                &self.hidden,
                &self.epsilon,
            ],
            1,
        )?;
        let lm_head = if model.config.tie_word_embeddings {
            embed
        } else {
            self.weight("lm_head.weight")?
        };
        let lm_head_name = if model.config.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        command.dispatch_1d(
            resident_matvec_kernel(model, lm_head_name)?,
            &[&self.norm, lm_head, &self.logits, &self.hidden, &self.vocab],
            model.config.vocab_size,
        )?;
        command.dispatch_1d(
            "argmax_f32",
            &[&self.logits, &self.selected, &self.vocab],
            1,
        )?;
        if let Some(profile) = profile.as_deref_mut() {
            ResidentDecodeProfile::record(&mut profile.lm_head, stage_started.elapsed(), 3);
        }
        let kernel_timings = command.take_kernel_timings();
        let timing = command.finish()?;
        if let Some(profile) = profile.as_deref_mut() {
            if kernel_timings.is_empty() {
                profile.attribute_command_buffer(timing);
            } else {
                record_exact_kernel_trace(profile, token_index, &kernel_timings);
            }
        }
        self.position_index += 1;
        let readback_started = Instant::now();
        let selected = runtime.read_u32(&self.selected)?;
        let logits = if self.logits_readback == LogitsReadback::FinalLogits {
            runtime.read_f32(&self.logits, model.config.vocab_size)?
        } else {
            Vec::new()
        };
        if let Some(profile) = profile.as_deref_mut() {
            ResidentDecodeProfile::record(
                &mut profile.token_readback,
                readback_started.elapsed(),
                0,
            );
            profile.trace.push(ResidentKernelTrace {
                token_index,
                phase: "token_readback",
                layer: None,
                stage: "token_readback",
                kernel: "selected_token_readback",
                cpu_encode: readback_started.elapsed(),
                gpu_execution: None,
                command_buffer_schedule: Duration::ZERO,
                threads: 0,
                threadgroups: 0,
                threads_per_threadgroup: 0,
                readback_bytes: if self.logits_readback == LogitsReadback::FinalLogits {
                    (model.config.vocab_size * std::mem::size_of::<f32>()
                        + std::mem::size_of::<u32>()) as u64
                } else {
                    std::mem::size_of::<u32>() as u64
                },
            });
        }
        Ok(TokenStep { selected, logits })
    }
}

fn record_exact_kernel_trace(
    profile: &mut ResidentDecodeProfile,
    token_index: usize,
    timings: &[atlas_metal::ResidentKernelTiming],
) {
    let mut layer = 0usize;
    for dispatch in timings {
        let (stage, phase) = match dispatch.kernel {
            "embedding_lookup_f32" | "embedding_lookup_q4_0" | "embedding_lookup_q8_0" => {
                ("embedding", "prefill_or_decode")
            }
            "attention_scores_resident_f32"
            | "masked_softmax_resident_f32"
            | "attention_values_resident_f32"
            | "attention_decode_fused_f32"
            | "kv_append_decode_f32"
            | "rope_f32"
            | "rope_half_to_interleaved_f32"
            | "rope_interleaved_to_half_f32" => ("attention", "prefill_or_decode"),
            "matvec_f32" | "matvec_q4_0" | "matvec_q8_0" => ("projection", "prefill_or_decode"),
            "silu_f32" | "vector_multiply_f32" => ("mlp", "prefill_or_decode"),
            "argmax_f32" => ("lm_head", "prefill_or_decode"),
            "rms_norm_f32" | "vector_add_f32" => ("normalization_or_residual", "prefill_or_decode"),
            _ => ("other", "prefill_or_decode"),
        };
        let layer_for_dispatch = if stage == "embedding" || stage == "lm_head" {
            None
        } else {
            Some(layer)
        };
        let timing = dispatch.timing;
        profile.command_buffer_count += 1;
        profile.command_buffer_schedule += timing.command_buffer_schedule;
        profile.gpu_execution += timing.gpu_time.unwrap_or_default();
        profile.trace.push(ResidentKernelTrace {
            token_index,
            phase,
            layer: layer_for_dispatch,
            stage,
            kernel: dispatch.kernel,
            cpu_encode: dispatch.cpu_encode,
            gpu_execution: timing.gpu_time,
            command_buffer_schedule: timing.command_buffer_schedule,
            threads: dispatch.threads,
            threadgroups: dispatch.threadgroups,
            threads_per_threadgroup: dispatch.threads_per_threadgroup,
            readback_bytes: 0,
        });
        if dispatch.kernel == "kv_append_decode_f32" {
            layer += 1;
        }
    }
    profile.tokens += 1;
}

struct TokenStep {
    selected: u32,
    logits: Vec<f32>,
}

fn resident_matvec_kernel(model: &AtlasModel, name: &str) -> Result<&'static str> {
    Ok(match model.resident_weight_format(name) {
        None => "matvec_f32",
        Some(GgufTensorType::Q4_0) => "matvec_q4_0",
        Some(GgufTensorType::Q8_0) => "matvec_q8_0",
        Some(other) => anyhow::bail!("unsupported packed resident tensor format {other:?}"),
    })
}

fn resident_embedding_kernel(model: &AtlasModel, name: &str) -> Result<&'static str> {
    Ok(match model.resident_weight_format(name) {
        None => "embedding_lookup_f32",
        Some(GgufTensorType::Q4_0) => "embedding_lookup_q4_0",
        Some(GgufTensorType::Q8_0) => "embedding_lookup_q8_0",
        Some(other) => anyhow::bail!("unsupported packed embedding format {other:?}"),
    })
}

impl<'a> AtlasExecutor<'a> {
    /// Runs the correctness-only resident trace. Each recorded boundary is
    /// completed and read back before the next boundary; normal generation
    /// remains the one-command-buffer resident path.
    pub fn trace_resident_prompt(
        model: &'a AtlasModel,
        prompt: &str,
        tolerance: f32,
    ) -> Result<Option<StageComparison>> {
        let tokens = model.tokenize(prompt)?;
        Self::trace_resident_token_ids(model, &tokens, tolerance)
    }

    /// Token-ID form of [`trace_resident_prompt`](Self::trace_resident_prompt),
    /// useful for proving the position-zero path without retokenizing text.
    pub fn trace_resident_token_ids(
        model: &'a AtlasModel,
        tokens: &[u32],
        tolerance: f32,
    ) -> Result<Option<StageComparison>> {
        ensure!(tolerance >= 0.0, "stage tolerance must be non-negative");
        ensure!(!tokens.is_empty(), "prompt tokenizes to no tokens");
        // Match normal decode capacity so resident diagnostic buffers use the
        // same KV/value and attention-stride layout as generation.
        let capacity = ExecutorConfig::default().max_context;
        let mut reference = Self::new(
            model,
            ExecutorConfig {
                max_context: capacity,
                mode: ExecutorMode::Reference,
                ..Default::default()
            },
        )?;
        let mut resident = Self::new(
            model,
            ExecutorConfig {
                max_context: capacity,
                mode: ExecutorMode::Resident,
                ..Default::default()
            },
        )?;
        for (token_index, &token) in tokens.iter().enumerate() {
            let expected = reference.trace_token_reference(token)?;
            let actual = resident
                .resident
                .as_mut()
                .expect("resident executor exists")
                .trace_token(model, token)?;
            ensure!(
                expected.len() == actual.len(),
                "trace stage count differs: reference={} resident={}",
                expected.len(),
                actual.len()
            );
            for (left, right) in expected.iter().zip(&actual) {
                ensure!(
                    left.stage == right.stage && left.layer == right.layer,
                    "trace stage order differs: reference={:?}/{:?} resident={:?}/{:?}",
                    left.stage,
                    left.layer,
                    right.stage,
                    right.layer
                );
                if let Some(comparison) = compare_stage(
                    token_index,
                    left.stage,
                    left.layer,
                    &left.values,
                    &right.values,
                    resident_stage_tolerance(left.stage, tolerance),
                ) {
                    return Ok(Some(comparison));
                }
            }
        }
        Ok(None)
    }

    pub fn new(model: &'a AtlasModel, config: ExecutorConfig) -> Result<Self> {
        ensure!(
            config.max_context > 0,
            "executor max_context must be positive"
        );
        ensure!(
            !model.is_gguf() || config.mode == ExecutorMode::Resident,
            "GGUF models require the Resident executor; Atlas will not use a reference fallback"
        );
        ensure!(
            model.is_gguf() || config.quant_format == QuantFormat::Fp16,
            "Phase-5 packed executor projections are not available; use fp16"
        );
        let cache_config = KvCacheConfig {
            layers: 1,
            kv_heads: model.config.num_key_value_heads,
            head_dim: model.config.head_dim(),
            capacity: config.max_context,
            sliding_window: None,
            sink_tokens: 0,
        };
        let caches = if config.mode == ExecutorMode::Reference {
            (0..model.config.num_hidden_layers)
                .map(|_| ContiguousKvCache::new(config.session, cache_config))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        // Do this after compatibility validation: unsupported packed plans
        // must not cause an expensive, surprising upload.
        let upload_started = Instant::now();
        let weight_upload_bytes = if config.mode == ExecutorMode::Resident {
            model.ensure_resident_weights()?
        } else {
            0
        };
        let weight_upload_elapsed = upload_started.elapsed();
        let resident = (config.mode == ExecutorMode::Resident)
            .then(|| {
                ResidentExecutor::new(
                    model,
                    config.max_context,
                    config.logits_readback,
                    config.resident_attention_path,
                )
            })
            .transpose()?;
        let pipelines = model.ops.runtime().pipeline_count();
        Ok(Self {
            model,
            prefill_plan: PrefillPlan {
                max_tokens: config.max_context,
                pipeline_count: pipelines,
                quant_format: config.quant_format,
            },
            decode_plan: DecodePlan {
                max_context: config.max_context,
                pipeline_count: pipelines,
                quant_format: config.quant_format,
            },
            caches,
            resident,
            mode: config.mode,
            logits_readback: config.logits_readback,
            stop_on_eos: config.stop_on_eos,
            resident_decode_profile: config.resident_decode_profile,
            weight_upload_bytes,
            weight_upload_elapsed,
        })
    }

    pub fn prefill_plan(&self) -> &PrefillPlan {
        &self.prefill_plan
    }
    pub fn decode_plan(&self) -> &DecodePlan {
        &self.decode_plan
    }
    /// Bytes of immutable parameters uploaded while this executor was
    /// initialized. A second executor for the same loaded model reports zero.
    pub fn weight_upload_bytes(&self) -> u64 {
        self.weight_upload_bytes
    }

    /// GPU-visible bytes reserved by this executor before decoding begins.
    /// This includes the immutable resident weights and the session arena, so
    /// callers can enforce a model memory budget before submitting a prompt.
    pub fn resident_bytes(&self) -> u64 {
        self.resident
            .as_ref()
            .map_or(0, ResidentExecutor::resident_bytes)
    }
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
        if let Some(resident) = &mut self.resident {
            resident.reset();
        }
    }

    pub fn generate_greedy(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<ExecutorGeneration> {
        let cancellation = AtomicBool::new(false);
        self.generate_greedy_stream(prompt, max_new_tokens, &cancellation, |_| Ok(()))
    }

    /// Generates greedily and delivers each decoded token as soon as it is ready.
    ///
    /// `cancellation` is checked before prefill and between decode steps.  Model,
    /// tokenizer, decode, context, and cancellation errors are delivered as a
    /// terminal [`GenerationEvent::Failed`] before being returned to the caller.
    pub fn generate_greedy_stream<F>(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancellation: &AtomicBool,
        mut callback: F,
    ) -> Result<ExecutorGeneration>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        let encoding_start = Instant::now();
        let prompt_token_ids = match self.model.tokenize(prompt) {
            Ok(token_ids) => token_ids,
            Err(error) => {
                callback(GenerationEvent::Failed {
                    message: format!("{error:#}"),
                })?;
                return Err(error);
            }
        };
        let cpu_encode = encoding_start.elapsed();
        self.generate_token_ids_stream(
            prompt_token_ids,
            max_new_tokens,
            cpu_encode,
            cancellation,
            callback,
        )
    }

    pub fn generate_token_ids(
        &mut self,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        cpu_encode: Duration,
    ) -> Result<ExecutorGeneration> {
        let cancellation = AtomicBool::new(false);
        self.generate_token_ids_stream(
            prompt_token_ids,
            max_new_tokens,
            cpu_encode,
            &cancellation,
            |_| Ok(()),
        )
    }

    /// Streaming equivalent of [`generate_token_ids`](Self::generate_token_ids).
    pub fn generate_token_ids_stream<F>(
        &mut self,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        cpu_encode: Duration,
        cancellation: &AtomicBool,
        mut callback: F,
    ) -> Result<ExecutorGeneration>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        match self.generate_token_ids_stream_inner(
            prompt_token_ids,
            max_new_tokens,
            cpu_encode,
            cancellation,
            &mut callback,
        ) {
            Ok(generation) => Ok(generation),
            Err(error) => {
                callback(GenerationEvent::Failed {
                    message: format!("{error:#}"),
                })?;
                Err(error)
            }
        }
    }

    fn generate_token_ids_stream_inner<F>(
        &mut self,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        cpu_encode: Duration,
        cancellation: &AtomicBool,
        callback: &mut F,
    ) -> Result<ExecutorGeneration>
    where
        F: FnMut(GenerationEvent) -> Result<()>,
    {
        ensure!(
            !prompt_token_ids.is_empty(),
            "prompt tokenizes to no tokens"
        );
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
        ensure!(
            prompt_token_ids.len() <= self.prefill_plan.max_tokens,
            "prompt exceeds executor context"
        );
        ensure!(
            !cancellation.load(Ordering::Acquire),
            "generation cancelled"
        );
        let prefill_token_count = prompt_token_ids.len();
        self.reset();
        let request_start = Instant::now();
        let runtime = self.model.ops.runtime();
        let command_buffers_before = runtime.command_buffer_count();
        let gpu_before = runtime.gpu_execution_time();
        let readback_before = runtime.readback_bytes();
        let mut resident_decode_profile =
            self.resident_decode_profile.then(|| ResidentDecodeProfile {
                attention_implementation: match self
                    .resident
                    .as_ref()
                    .map(|resident| resident.attention_path)
                {
                    Some(ResidentAttentionPath::Fused) => "fused_online_softmax",
                    Some(ResidentAttentionPath::LegacyThreePass) => "legacy_three_pass",
                    None => "not_applicable",
                },
                ..Default::default()
            });
        if let Some(profile) = resident_decode_profile.as_mut() {
            profile.trace.push(ResidentKernelTrace {
                token_index: 0,
                phase: "request_tokenization",
                layer: None,
                stage: "request",
                kernel: "request_tokenization",
                cpu_encode,
                gpu_execution: None,
                command_buffer_schedule: Duration::ZERO,
                threads: 0,
                threadgroups: 0,
                threads_per_threadgroup: 0,
                readback_bytes: 0,
            });
            profile.trace.push(ResidentKernelTrace {
                token_index: 0,
                phase: "resident_upload",
                layer: None,
                stage: "resident_upload",
                kernel: "resident_weight_upload",
                cpu_encode: self.weight_upload_elapsed,
                gpu_execution: None,
                command_buffer_schedule: Duration::ZERO,
                threads: 0,
                threadgroups: 0,
                threads_per_threadgroup: 0,
                readback_bytes: self.weight_upload_bytes,
            });
        }
        let prefill_start = Instant::now();
        let prefill_gpu_before = runtime.gpu_execution_time();
        let prefill_readback_before = runtime.readback_bytes();
        let mut step = TokenStep {
            selected: 0,
            logits: Vec::new(),
        };
        for &token in &prompt_token_ids {
            // Exact profiling is decode-only. Keeping normal one-command-
            // buffer prefill avoids multiplying the golden suite's 64-token
            // prefix into thousands of serial diagnostic submissions.
            step = self.forward_token(token, None)?;
        }
        let prefill = prefill_start.elapsed();
        let prefill_command_buffer_count = runtime.command_buffer_count() - command_buffers_before;
        if let Some(profile) = resident_decode_profile.as_mut() {
            profile.trace.push(ResidentKernelTrace {
                token_index: 0,
                phase: "prefill",
                layer: None,
                stage: "prefill",
                kernel: "resident_prefill",
                cpu_encode: prefill,
                gpu_execution: Some(
                    runtime
                        .gpu_execution_time()
                        .saturating_sub(prefill_gpu_before),
                ),
                command_buffer_schedule: Duration::ZERO,
                threads: 0,
                threadgroups: prefill_command_buffer_count as usize,
                threads_per_threadgroup: 0,
                readback_bytes: runtime.readback_bytes() - prefill_readback_before,
            });
        }
        let mut ids = prompt_token_ids.clone();
        let mut latencies = Vec::new();
        let decode_start = Instant::now();
        let mut finish_reason = GenerationFinishReason::MaxTokens;
        let mut ttft = Duration::ZERO;
        for token_index in 0..max_new_tokens {
            ensure!(
                !cancellation.load(Ordering::Acquire),
                "generation cancelled"
            );
            let token = step.selected;
            ids.push(token);
            let text = self.model.decode(&[token])?;
            let decode_latency = if token_index == 0 {
                None
            } else {
                latencies.last().copied()
            };
            callback(GenerationEvent::Token {
                token_id: token,
                text,
                decode_latency,
            })?;
            if token_index == 0 {
                ttft = request_start.elapsed();
            }
            if self.stop_on_eos && Some(token) == self.model.config.eos_token_id {
                finish_reason = GenerationFinishReason::Eos;
                break;
            }
            if token_index + 1 < max_new_tokens {
                let position = self.resident.as_ref().map_or_else(
                    || self.caches[0].next_position(),
                    |resident| resident.position_index,
                );
                ensure!(
                    position < self.decode_plan.max_context,
                    "executor context exhausted"
                );
                let started = Instant::now();
                let trace_start = resident_decode_profile
                    .as_ref()
                    .map_or(0, |profile| profile.trace.len());
                step = self.forward_token(token, resident_decode_profile.as_mut())?;
                if let Some(profile) = resident_decode_profile.as_mut() {
                    for entry in &mut profile.trace[trace_start..] {
                        if entry.phase == "prefill_or_decode" {
                            entry.phase = "decode";
                        }
                    }
                }
                latencies.push(started.elapsed());
            }
        }
        let decode = decode_start.elapsed();
        let generated_token_ids = ids[prompt_token_ids.len()..].to_vec();
        let pipelines = self.model.ops.runtime().pipeline_count();
        let generation = ExecutorGeneration {
            generation: Generation {
                prompt_token_ids,
                generated_token_ids,
                text: self.model.decode(&ids)?,
                trace: LayerTrace::default(),
                final_logits: step.logits,
            },
            metrics: ExecutorMetrics {
                cpu_encode,
                prefill,
                decode,
                ttft,
                prefill_tokens: prefill_token_count,
                decode_tokens: latencies.len(),
                decode_latencies: latencies,
                pipeline_count: self.prefill_plan.pipeline_count,
                post_warmup_pipeline_count: pipelines,
                post_warmup_allocations: 0,
                host_wall_time: request_start.elapsed(),
                gpu_execution_time: runtime.gpu_execution_time().saturating_sub(gpu_before),
                command_buffer_count: runtime.command_buffer_count() - command_buffers_before,
                prefill_command_buffer_count,
                decode_command_buffer_count: runtime.command_buffer_count()
                    - command_buffers_before
                    - prefill_command_buffer_count,
                weight_upload_bytes: self.weight_upload_bytes,
                weight_upload_elapsed: self.weight_upload_elapsed,
                readback_bytes: runtime.readback_bytes() - readback_before,
                resident_bytes: self
                    .resident
                    .as_ref()
                    .map_or(0, ResidentExecutor::resident_bytes),
                resident_arena_allocations: self
                    .resident
                    .as_ref()
                    .map_or(0, ResidentExecutor::allocations),
                resident_decode_profile,
            },
            finish_reason,
        };
        callback(GenerationEvent::Finished {
            reason: finish_reason,
            metrics: generation.metrics.clone(),
        })?;
        Ok(generation)
    }

    fn forward_token(
        &mut self,
        token: u32,
        profile: Option<&mut ResidentDecodeProfile>,
    ) -> Result<TokenStep> {
        if self.mode == ExecutorMode::Resident {
            return self
                .resident
                .as_mut()
                .expect("resident executor exists")
                .forward_token(self.model, token, profile);
        }
        self.forward_token_reference(token)
    }

    fn forward_token_reference(&mut self, token: u32) -> Result<TokenStep> {
        let position = self.caches[0].next_position();
        let h = self.model.config.hidden_size;
        let hd = self.model.config.head_dim();
        let embedding = self.model.weight("model.embed_tokens.weight")?;
        let mut state = self
            .model
            .ops
            .embedding(&embedding, self.model.config.vocab_size, h, &[token])?
            .0;
        for layer in 0..self.model.config.num_hidden_layers {
            state = self.layer_token(layer, position, &state, hd)?;
        }
        let norm = self.model.weight("model.norm.weight")?;
        state = self
            .model
            .ops
            .rms_norm(&state, 1, h, &norm, self.model.config.rms_norm_eps)?
            .0;
        let lm_head = if self.model.config.tie_word_embeddings {
            embedding
        } else {
            self.model.weight("lm_head.weight")?
        };
        let logits = self
            .model
            .ops
            .project(
                atlas_ops::ExecutionMode::Decode,
                &state,
                &lm_head,
                1,
                h,
                self.model.config.vocab_size,
            )?
            .0;
        Ok(TokenStep {
            selected: argmax(&logits) as u32,
            logits: if self.logits_readback == LogitsReadback::FinalLogits {
                logits
            } else {
                Vec::new()
            },
        })
    }

    fn trace_token_reference(&mut self, token: u32) -> Result<Vec<StageSnapshot>> {
        let position = self.caches[0].next_position();
        let h = self.model.config.hidden_size;
        let hd = self.model.config.head_dim();
        let embedding = self.model.weight("model.embed_tokens.weight")?;
        let mut snapshots = Vec::new();
        let mut state = self
            .model
            .ops
            .embedding(&embedding, self.model.config.vocab_size, h, &[token])?
            .0;
        snapshots.push(StageSnapshot {
            stage: ResidentStage::Embedding,
            layer: None,
            values: state.clone(),
        });
        for layer in 0..self.model.config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let normalized = self
                .model
                .ops
                .rms_norm(
                    &state,
                    1,
                    h,
                    &self
                        .model
                        .weight(&format!("{prefix}.input_layernorm.weight"))?,
                    self.model.config.rms_norm_eps,
                )?
                .0;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::InputNorm,
                layer: Some(layer),
                values: normalized.clone(),
            });
            let project = |name: &str, output: usize| {
                self.model.project_width(
                    1,
                    &normalized,
                    &format!("{prefix}.{name}.weight"),
                    h,
                    output,
                    atlas_ops::ExecutionMode::Decode,
                )
            };
            let q = project("self_attn.q_proj", h)?;
            let k = project(
                "self_attn.k_proj",
                self.model.config.num_key_value_heads * hd,
            )?;
            let v = project(
                "self_attn.v_proj",
                self.model.config.num_key_value_heads * hd,
            )?;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::Q,
                layer: Some(layer),
                values: q.clone(),
            });
            snapshots.push(StageSnapshot {
                stage: ResidentStage::K,
                layer: Some(layer),
                values: k.clone(),
            });
            snapshots.push(StageSnapshot {
                stage: ResidentStage::V,
                layer: Some(layer),
                values: v.clone(),
            });
            let q = self
                .model
                .rope_at(&q, position, self.model.config.num_attention_heads, hd)?;
            let k = self
                .model
                .rope_at(&k, position, self.model.config.num_key_value_heads, hd)?;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::RopeQ,
                layer: Some(layer),
                values: q.clone(),
            });
            snapshots.push(StageSnapshot {
                stage: ResidentStage::RopeK,
                layer: Some(layer),
                values: k.clone(),
            });
            self.caches[layer].append(
                position,
                &[LayerKv {
                    keys: &k,
                    values: &v,
                }],
            )?;
            let view = self.caches[layer].view(0)?;
            let mut attention = vec![0.0; h];
            let group =
                self.model.config.num_attention_heads / self.model.config.num_key_value_heads;
            for head in 0..self.model.config.num_attention_heads {
                let kv_head = head / group;
                let query = gather_head(&q, 1, self.model.config.num_attention_heads, hd, head);
                let count = view.positions.len();
                let keys = &view.keys[kv_head * count * hd..(kv_head + 1) * count * hd];
                let values = &view.values[kv_head * count * hd..(kv_head + 1) * count * hd];
                let scores = self
                    .model
                    .ops
                    .attention_scores(&query, keys, 1, count, hd, (hd as f32).sqrt().recip())?
                    .0;
                let weights = self
                    .model
                    .ops
                    .masked_softmax(&scores, &vec![0.0; count], 1, count)?
                    .0;
                let result = self
                    .model
                    .ops
                    .attention_values(&weights, values, 1, count, hd)?
                    .0;
                scatter_head(
                    &mut attention,
                    &result,
                    1,
                    self.model.config.num_attention_heads,
                    hd,
                    head,
                );
            }
            snapshots.push(StageSnapshot {
                stage: ResidentStage::Attention,
                layer: Some(layer),
                values: attention.clone(),
            });
            let projected = self.model.project_width(
                1,
                &attention,
                &format!("{prefix}.self_attn.o_proj.weight"),
                h,
                h,
                atlas_ops::ExecutionMode::Decode,
            )?;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::AttentionOutputProjection,
                layer: Some(layer),
                values: projected.clone(),
            });
            snapshots.push(StageSnapshot {
                stage: ResidentStage::AttentionResidualInput,
                layer: Some(layer),
                values: state.clone(),
            });
            let residual = self.model.ops.add(&state, &projected)?.0;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::AttentionResidual,
                layer: Some(layer),
                values: residual.clone(),
            });
            let post_norm = self
                .model
                .ops
                .rms_norm(
                    &residual,
                    1,
                    h,
                    &self
                        .model
                        .weight(&format!("{prefix}.post_attention_layernorm.weight"))?,
                    self.model.config.rms_norm_eps,
                )?
                .0;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::PostAttentionNorm,
                layer: Some(layer),
                values: post_norm.clone(),
            });
            let gate = self.model.project_width(
                1,
                &post_norm,
                &format!("{prefix}.mlp.gate_proj.weight"),
                h,
                self.model.config.intermediate_size,
                atlas_ops::ExecutionMode::Decode,
            )?;
            let up = self.model.project_width(
                1,
                &post_norm,
                &format!("{prefix}.mlp.up_proj.weight"),
                h,
                self.model.config.intermediate_size,
                atlas_ops::ExecutionMode::Decode,
            )?;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::Gate,
                layer: Some(layer),
                values: gate.clone(),
            });
            snapshots.push(StageSnapshot {
                stage: ResidentStage::Up,
                layer: Some(layer),
                values: up.clone(),
            });
            let activated = self.model.ops.silu(&gate)?.0;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::SiLU,
                layer: Some(layer),
                values: activated.clone(),
            });
            let product = self.model.ops.multiply(&activated, &up)?.0;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::MlpProduct,
                layer: Some(layer),
                values: product.clone(),
            });
            let mlp = self.model.project_width(
                1,
                &product,
                &format!("{prefix}.mlp.down_proj.weight"),
                self.model.config.intermediate_size,
                h,
                atlas_ops::ExecutionMode::Decode,
            )?;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::MlpDownProjection,
                layer: Some(layer),
                values: mlp.clone(),
            });
            snapshots.push(StageSnapshot {
                stage: ResidentStage::MlpResidualInput,
                layer: Some(layer),
                values: residual.clone(),
            });
            state = self.model.ops.add(&residual, &mlp)?.0;
            snapshots.push(StageSnapshot {
                stage: ResidentStage::MlpResidual,
                layer: Some(layer),
                values: state.clone(),
            });
        }
        state = self
            .model
            .ops
            .rms_norm(
                &state,
                1,
                h,
                &self.model.weight("model.norm.weight")?,
                self.model.config.rms_norm_eps,
            )?
            .0;
        snapshots.push(StageSnapshot {
            stage: ResidentStage::FinalNorm,
            layer: None,
            values: state.clone(),
        });
        let lm_head = if self.model.config.tie_word_embeddings {
            embedding
        } else {
            self.model.weight("lm_head.weight")?
        };
        let logits = self
            .model
            .ops
            .project(
                atlas_ops::ExecutionMode::Decode,
                &state,
                &lm_head,
                1,
                h,
                self.model.config.vocab_size,
            )?
            .0;
        snapshots.push(StageSnapshot {
            stage: ResidentStage::Logits,
            layer: None,
            values: logits,
        });
        Ok(snapshots)
    }

    fn layer_token(
        &mut self,
        layer: usize,
        position: usize,
        state: &[f32],
        hd: usize,
    ) -> Result<Vec<f32>> {
        let h = self.model.config.hidden_size;
        let prefix = format!("model.layers.{layer}");
        let norm = self
            .model
            .weight(&format!("{prefix}.input_layernorm.weight"))?;
        let normalized = self
            .model
            .ops
            .rms_norm(state, 1, h, &norm, self.model.config.rms_norm_eps)?
            .0;
        let q = self.model.project_width(
            1,
            &normalized,
            &format!("{prefix}.self_attn.q_proj.weight"),
            h,
            self.model.config.num_attention_heads * hd,
            atlas_ops::ExecutionMode::Decode,
        )?;
        let k = self.model.project_width(
            1,
            &normalized,
            &format!("{prefix}.self_attn.k_proj.weight"),
            h,
            self.model.config.num_key_value_heads * hd,
            atlas_ops::ExecutionMode::Decode,
        )?;
        let v = self.model.project_width(
            1,
            &normalized,
            &format!("{prefix}.self_attn.v_proj.weight"),
            h,
            self.model.config.num_key_value_heads * hd,
            atlas_ops::ExecutionMode::Decode,
        )?;
        let q = self
            .model
            .rope_at(&q, position, self.model.config.num_attention_heads, hd)?;
        let k = self
            .model
            .rope_at(&k, position, self.model.config.num_key_value_heads, hd)?;
        self.caches[layer].append(
            position,
            &[LayerKv {
                keys: &k,
                values: &v,
            }],
        )?;
        let view = self.caches[layer].view(0)?;
        let mut attention = vec![0.0; h];
        let group = self.model.config.num_attention_heads / self.model.config.num_key_value_heads;
        for head in 0..self.model.config.num_attention_heads {
            let kv_head = head / group;
            let query = gather_head(&q, 1, self.model.config.num_attention_heads, hd, head);
            let count = view.positions.len();
            let keys = &view.keys[kv_head * count * hd..(kv_head + 1) * count * hd];
            let values = &view.values[kv_head * count * hd..(kv_head + 1) * count * hd];
            let scores = self
                .model
                .ops
                .attention_scores(&query, keys, 1, count, hd, (hd as f32).sqrt().recip())?
                .0;
            let weights = self
                .model
                .ops
                .masked_softmax(&scores, &vec![0.0; count], 1, count)?
                .0;
            let values = self
                .model
                .ops
                .attention_values(&weights, values, 1, count, hd)?
                .0;
            scatter_head(
                &mut attention,
                &values,
                1,
                self.model.config.num_attention_heads,
                hd,
                head,
            );
        }
        let attention_output = self.model.project_width(
            1,
            &attention,
            &format!("{prefix}.self_attn.o_proj.weight"),
            h,
            h,
            atlas_ops::ExecutionMode::Decode,
        )?;
        let residual = self.model.ops.add(state, &attention_output)?.0;
        let post_norm = self
            .model
            .weight(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let mlp_input = self
            .model
            .ops
            .rms_norm(&residual, 1, h, &post_norm, self.model.config.rms_norm_eps)?
            .0;
        let gate = self.model.project_width(
            1,
            &mlp_input,
            &format!("{prefix}.mlp.gate_proj.weight"),
            h,
            self.model.config.intermediate_size,
            atlas_ops::ExecutionMode::Decode,
        )?;
        let up = self.model.project_width(
            1,
            &mlp_input,
            &format!("{prefix}.mlp.up_proj.weight"),
            h,
            self.model.config.intermediate_size,
            atlas_ops::ExecutionMode::Decode,
        )?;
        let activated = self.model.ops.silu(&gate)?.0;
        let product = self.model.ops.multiply(&activated, &up)?.0;
        let mlp = self.model.project_width(
            1,
            &product,
            &format!("{prefix}.mlp.down_proj.weight"),
            self.model.config.intermediate_size,
            h,
            atlas_ops::ExecutionMode::Decode,
        )?;
        Ok(self.model.ops.add(&residual, &mlp)?.0)
    }
}

fn rate(tokens: usize, duration: Duration) -> f64 {
    if duration.is_zero() {
        0.0
    } else {
        tokens as f64 / duration.as_secs_f64()
    }
}
fn percentile(samples: &[Duration], percentile: f64) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let mut values = samples.to_vec();
    values.sort_unstable();
    values[((values.len() - 1) as f64 * percentile).ceil() as usize]
}
