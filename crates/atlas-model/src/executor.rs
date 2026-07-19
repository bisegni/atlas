//! Phase-6 prefill/decode execution plans.
//!
//! The plans own only immutable shape and residency decisions.  A session owns
//! the mutable KV cache, which keeps request state out of the model itself.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use atlas_core::QuantFormat;
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
    AttentionResidual,
    PostAttentionNorm,
    Gate,
    Up,
    SiLU,
    MlpProduct,
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
            Self::AttentionResidual => "attention_residual",
            Self::PostAttentionNorm => "post_attention_norm",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::SiLU => "silu",
            Self::MlpProduct => "mlp_product",
            Self::MlpResidual => "mlp_residual",
            Self::FinalNorm => "final_norm",
            Self::Logits => "logits",
        };
        f.write_str(name)
    }
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
    /// Reference remains the default until resident decode passes its hardware
    /// parity gate. Resident is explicit for diagnostics and benchmarking.
    pub mode: ExecutorMode,
    /// Downloading logits defeats token-only decode readback, so it is opt-in.
    pub logits_readback: LogitsReadback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutorMode {
    #[default]
    Reference,
    Resident,
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
            mode: ExecutorMode::Reference,
            logits_readback: LogitsReadback::SelectedToken,
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
    /// Bytes copied from Metal buffers back to CPU-visible vectors.
    pub readback_bytes: u64,
    pub resident_arena_allocations: u64,
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
    weight_upload_bytes: u64,
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
    k: GpuBuffer,
    k_rot: GpuBuffer,
    v: GpuBuffer,
    attention: GpuBuffer,
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
    theta: GpuBuffer,
    one: GpuBuffer,
    max_context: usize,
    position_index: usize,
    logits_readback: LogitsReadback,
}

impl ResidentExecutor {
    fn new(model: &AtlasModel, capacity: usize, logits_readback: LogitsReadback) -> Result<Self> {
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
            k: allocate_f32(kv_width)?,
            k_rot: allocate_f32(kv_width)?,
            v: allocate_f32(kv_width)?,
            attention: allocate_f32(h)?,
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
            theta: runtime.upload_f32(&[c.rope_theta])?,
            one: runtime.upload_u32(&[1])?,
            max_context: capacity,
            position_index: 0,
            logits_readback,
        })
    }

    fn reset(&mut self) {
        self.position_index = 0;
    }
    fn allocations(&self) -> u64 {
        24 + self.kv.len() as u64
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
            self.trace_dispatch(
                runtime,
                "rope_llama_decode_f32",
                &[
                    &self.q,
                    &self.q_rot,
                    &self.heads,
                    &self.head_dim,
                    &self.position,
                    &self.theta,
                ],
                h / 2,
            )?;
            snapshots.push(capture(ResidentStage::RopeQ, Some(layer), &self.q_rot, h)?);
            self.trace_dispatch(
                runtime,
                "rope_llama_decode_f32",
                &[
                    &self.k,
                    &self.k_rot,
                    &self.kv_heads,
                    &self.head_dim,
                    &self.position,
                    &self.theta,
                ],
                kv_width / 2,
            )?;
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
            self.trace_dispatch(
                runtime,
                "attention_decode_f32",
                &[
                    &self.q_rot,
                    &self.kv[layer],
                    &self.attention,
                    &self.heads,
                    &self.kv_heads,
                    &self.head_dim,
                    &self.capacity,
                    &self.position,
                ],
                h,
            )?;
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

    fn forward_token(&mut self, model: &AtlasModel, token: u32) -> Result<TokenStep> {
        ensure!(
            self.position_index < self.max_context,
            "executor context exhausted"
        );
        let runtime = model.ops.runtime();
        runtime.write_u32(&self.token, &[token])?;
        runtime.write_u32(&self.position, &[u32::try_from(self.position_index)?])?;
        let mut command = runtime.begin_resident_command()?;
        let embed = self.weight("model.embed_tokens.weight")?;
        command.dispatch_1d(
            "embedding_lookup_f32",
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
            for (name, output, width) in [
                ("q_proj", &self.q, &self.hidden),
                ("k_proj", &self.k, &self.kv_width),
                ("v_proj", &self.v, &self.kv_width),
            ] {
                command.dispatch_1d(
                    "matvec_f32",
                    &[
                        &self.norm,
                        self.weight(&format!("{p}.self_attn.{name}.weight"))?,
                        output,
                        &self.hidden,
                        width,
                    ],
                    width.bytes() / 4,
                )?;
            }
            command.dispatch_1d(
                "rope_llama_decode_f32",
                &[
                    &self.q,
                    &self.q_rot,
                    &self.heads,
                    &self.head_dim,
                    &self.position,
                    &self.theta,
                ],
                model.config.hidden_size / 2,
            )?;
            command.dispatch_1d(
                "rope_llama_decode_f32",
                &[
                    &self.k,
                    &self.k_rot,
                    &self.kv_heads,
                    &self.head_dim,
                    &self.position,
                    &self.theta,
                ],
                self.k.bytes() / 8,
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
                self.kv_width.bytes() / 4,
            )?;
            command.dispatch_1d(
                "attention_decode_f32",
                &[
                    &self.q_rot,
                    &self.kv[layer],
                    &self.attention,
                    &self.heads,
                    &self.kv_heads,
                    &self.head_dim,
                    &self.capacity,
                    &self.position,
                ],
                model.config.hidden_size,
            )?;
            command.dispatch_1d(
                "matvec_f32",
                &[
                    &self.attention,
                    self.weight(&format!("{p}.self_attn.o_proj.weight"))?,
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
            command.dispatch_1d(
                "matvec_f32",
                &[
                    &self.norm,
                    self.weight(&format!("{p}.mlp.gate_proj.weight"))?,
                    &self.gate,
                    &self.hidden,
                    &self.intermediate,
                ],
                model.config.intermediate_size,
            )?;
            command.dispatch_1d(
                "matvec_f32",
                &[
                    &self.norm,
                    self.weight(&format!("{p}.mlp.up_proj.weight"))?,
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
            command.dispatch_1d(
                "matvec_f32",
                &[
                    &self.product,
                    self.weight(&format!("{p}.mlp.down_proj.weight"))?,
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
        }
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
        command.dispatch_1d(
            "matvec_f32",
            &[&self.norm, lm_head, &self.logits, &self.hidden, &self.vocab],
            model.config.vocab_size,
        )?;
        command.dispatch_1d(
            "argmax_f32",
            &[&self.logits, &self.selected, &self.vocab],
            1,
        )?;
        command.finish()?;
        self.position_index += 1;
        Ok(TokenStep {
            selected: runtime.read_u32(&self.selected)?,
            logits: if self.logits_readback == LogitsReadback::FinalLogits {
                runtime.read_f32(&self.logits, model.config.vocab_size)?
            } else {
                Vec::new()
            },
        })
    }
}

struct TokenStep {
    selected: u32,
    logits: Vec<f32>,
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
        let capacity = tokens.len();
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
                    tolerance,
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
        // The Phase-5 packing format is represented in the public plan now;
        // this correctness-first executor continues to use the model's native
        // tensors until packed Metal projection kernels replace the reference
        // kernels.
        ensure!(
            config.quant_format == QuantFormat::Fp16,
            "packed executor projections are not available yet; use fp16"
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
        let weight_upload_bytes = if config.mode == ExecutorMode::Resident {
            model.ensure_resident_weights()?
        } else {
            0
        };
        let resident = (config.mode == ExecutorMode::Resident)
            .then(|| ResidentExecutor::new(model, config.max_context, config.logits_readback))
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
            weight_upload_bytes,
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
        let prefill_start = Instant::now();
        let mut step = TokenStep {
            selected: 0,
            logits: Vec::new(),
        };
        for &token in &prompt_token_ids {
            step = self.forward_token(token)?;
        }
        let prefill = prefill_start.elapsed();
        let prefill_command_buffer_count = runtime.command_buffer_count() - command_buffers_before;
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
            if Some(token) == self.model.config.eos_token_id {
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
                step = self.forward_token(token)?;
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
                readback_bytes: runtime.readback_bytes() - readback_before,
                resident_arena_allocations: self
                    .resident
                    .as_ref()
                    .map_or(0, ResidentExecutor::allocations),
            },
        };
        callback(GenerationEvent::Finished {
            reason: finish_reason,
            metrics: generation.metrics.clone(),
        })?;
        Ok(generation)
    }

    fn forward_token(&mut self, token: u32) -> Result<TokenStep> {
        if self.mode == ExecutorMode::Resident {
            return self
                .resident
                .as_mut()
                .expect("resident executor exists")
                .forward_token(self.model, token);
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
