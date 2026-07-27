//! Resident-only Gemma 4 E2B greedy execution.
//!
//! Gemma cannot share the Llama executor: its PLE state, one-head shared KV
//! cache, mixed full/sliding attention, and final Q6_K tied projection are
//! architectural state, not optional Llama features.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use atlas_core::GgufTensorType;
use atlas_metal::GpuBuffer;

use crate::{Gemma4E2bModel, Generation, LayerTrace, gemma4_shared_kv_sources};

const GEMMA4_TRACE_STAGES_PER_LAYER: usize = 13;
const GEMMA4_TRACE_GLOBAL_STAGES: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4KvCacheType {
    F32,
    Q8_0,
    Q4_0,
}

impl Gemma4KvCacheType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "f32" => Ok(Self::F32),
            "q8_0" => Ok(Self::Q8_0),
            "q4_0" => Ok(Self::Q4_0),
            _ => anyhow::bail!(
                "unsupported Gemma KV cache type `{value}`; expected f32, q8_0, or q4_0"
            ),
        }
    }

    const fn bytes_per_block(self) -> usize {
        match self {
            Self::F32 => 32 * std::mem::size_of::<f32>(),
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
        }
    }

    fn cache_bytes(self, capacity: usize, width: usize) -> Result<usize> {
        ensure!(width % 32 == 0, "Gemma KV width must be block aligned");
        capacity
            .checked_mul(width / 32)
            .and_then(|blocks| blocks.checked_mul(2))
            .and_then(|blocks| blocks.checked_mul(self.bytes_per_block()))
            .context("Gemma KV cache allocation overflows")
    }
}

fn gemma4_trace_layer_slot(layer: usize, stage: usize) -> usize {
    GEMMA4_TRACE_GLOBAL_STAGES + layer * GEMMA4_TRACE_STAGES_PER_LAYER + stage
}

fn gemma4_rope_angle(
    position: usize,
    pair: usize,
    rotary: usize,
    theta: f32,
    frequency_factor: f32,
) -> f32 {
    if pair >= rotary / 2 {
        return 0.0;
    }
    position as f32 / theta.powf((pair * 2) as f32 / rotary as f32) / frequency_factor
}

fn gemma4_should_finish(token: u32, eos_token: u32, decoded: &str, chat: bool) -> bool {
    token == eos_token || (chat && decoded.contains("<turn|>"))
}

/// Number of keys visible to a query at `position`.  The cache owns absolute
/// positions; sliding attention only changes the visible suffix, never the
/// shared-KV source or the write position.
fn gemma4_attention_key_count(position: usize, sliding: bool, sliding_window: usize) -> usize {
    if sliding {
        position.min(sliding_window.saturating_sub(1)) + 1
    } else {
        position + 1
    }
}

/// Build immutable attention controls for one encoded command.  The table is
/// row-major `[token_in_command][layer]`, so binding an offset retains the
/// correct causal key count even though the GPU executes after all host
/// encoding has completed.
fn gemma4_attention_key_count_table(
    start_position: usize,
    tokens: usize,
    sliding_pattern: &[bool],
    sliding_window: usize,
) -> Result<Vec<u32>> {
    ensure!(tokens > 0, "Gemma attention control table requires a token");
    let mut table = Vec::with_capacity(
        tokens
            .checked_mul(sliding_pattern.len())
            .context("Gemma attention control table size overflow")?,
    );
    for token in 0..tokens {
        let position = start_position
            .checked_add(token)
            .context("Gemma attention control position overflow")?;
        for &sliding in sliding_pattern {
            table.push(u32::try_from(gemma4_attention_key_count(
                position,
                sliding,
                sliding_window,
            ))?);
        }
    }
    Ok(table)
}

#[derive(Debug, Clone)]
pub struct Gemma4Metrics {
    pub resident_bytes: u64,
    pub weight_upload_bytes: u64,
    pub readback_bytes: u64,
    pub command_buffers: u64,
    pub prefill_command_buffers: u64,
    pub decode_command_buffers: u64,
    pub prefill: Duration,
    pub decode: Duration,
    pub host_wall_time: Duration,
    pub prefill_path: &'static str,
    pub prefill_chunk_size: usize,
    pub prefill_chunks: usize,
    pub attention_kernel: &'static str,
    pub kv_cache_type: Gemma4KvCacheType,
    pub kv_cache_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gemma4PrefillPlan {
    pub prompt_tokens: usize,
    pub chunk_size: usize,
    pub chunks: usize,
}

impl Gemma4PrefillPlan {
    pub fn new(prompt_tokens: usize, max_context: usize) -> Result<Self> {
        ensure!(
            prompt_tokens > 0,
            "Gemma prefill requires at least one token"
        );
        ensure!(
            prompt_tokens <= max_context,
            "Gemma prefill exceeds context capacity"
        );
        let chunk_size = prompt_tokens.min(128);
        Ok(Self {
            prompt_tokens,
            chunk_size,
            chunks: prompt_tokens.div_ceil(chunk_size),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4Generation {
    pub generation: Generation,
    pub metrics: Gemma4Metrics,
    pub finish_reason: Gemma4FinishReason,
    /// The one-based generated-token ordinal at which the model first chose
    /// EOS. This is populated by the diagnostic fixed-workload benchmark even
    /// though that benchmark deliberately continues decoding after EOS.
    pub first_eos_position: Option<usize>,
}

/// One exact-timing observation from the diagnostic decode profiler.  The
/// normal Resident path never creates these records: each observed dispatch is
/// deliberately submitted and waited independently so Metal can report its
/// GPU duration without contaminating chat measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DecodeKernelProfile {
    pub family: &'static str,
    pub dispatches: u64,
    pub gpu_nanos: u128,
    pub cpu_encode_nanos: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DecodeProfileSample {
    /// Decode ordinal, starting at one after the prompt prefill.
    pub decode_position: usize,
    /// Absolute KV position used by this forward pass.
    pub context_position: usize,
    pub attention_key_count: usize,
    pub full_attention_layers: usize,
    pub sliding_attention_layers: usize,
    pub resident_bytes: u64,
    pub readback_bytes: u64,
    pub kernels: Vec<Gemma4DecodeKernelProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DecodeProfile {
    pub prompt_tokens: usize,
    pub requested_decode_tokens: usize,
    pub prefill_path: &'static str,
    pub attention_kernel: &'static str,
    pub kv_cache_type: Gemma4KvCacheType,
    pub samples: Vec<Gemma4DecodeProfileSample>,
}

fn gemma4_kernel_family(kernel: &str) -> &'static str {
    if kernel.starts_with("matvec_q4")
        || kernel.starts_with("matvec_q6")
        || kernel.starts_with("matmul_q4")
        || kernel.starts_with("matmul_q6")
    {
        "q4_q6_projections"
    } else if kernel.contains("attention") || kernel.contains("attn_") {
        "gemma_attention"
    } else if kernel.contains("rms_norm") || kernel.contains("rope") {
        "rms_rope"
    } else if kernel.contains("kv_") {
        "kv_append"
    } else if kernel.contains("gelu")
        || kernel.contains("vector_multiply")
        || kernel.contains("vector_add")
        || kernel.contains("scalar_multiply")
    {
        "mlp_activation_residual"
    } else if kernel.contains("softcap") {
        "softcap"
    } else if kernel.contains("argmax") {
        "argmax"
    } else {
        "other"
    }
}

fn gemma4_attention_kernel(cache_type: Gemma4KvCacheType) -> &'static str {
    if cache_type == Gemma4KvCacheType::F32
        && std::env::var_os("ATLAS_GEMMA4_ATTENTION_BASELINE").is_some()
    {
        "attention_decode_fused_gemma4_f32"
    } else {
        match cache_type {
            Gemma4KvCacheType::F32 => "attention_decode_fused_gemma4_simd_f32",
            Gemma4KvCacheType::Q8_0 => "attention_decode_fused_gemma4_simd_q8_0",
            Gemma4KvCacheType::Q4_0 => "attention_decode_fused_gemma4_simd_q4_0",
        }
    }
}

fn gemma4_kv_append_kernel(cache_type: Gemma4KvCacheType) -> &'static str {
    match cache_type {
        Gemma4KvCacheType::F32 => "kv_append_decode_f32",
        Gemma4KvCacheType::Q8_0 => "kv_append_decode_q8_0",
        Gemma4KvCacheType::Q4_0 => "kv_append_decode_q4_0",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4FinishReason {
    Eos,
    MaxTokens,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4TokenEvent {
    pub token_id: u32,
    pub text: String,
    pub latency: Duration,
}

pub struct Gemma4E2bExecutor<'a> {
    model: &'a Gemma4E2bModel,
    max_context: usize,
    position: usize,
    kv_sources: Vec<usize>,
    kv: Vec<Option<GpuBuffer>>,
    kv_cache_type: Gemma4KvCacheType,
    token: GpuBuffer,
    position_buffer: GpuBuffer,
    selected: GpuBuffer,
    state: GpuBuffer,
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
    work: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    activated: GpuBuffer,
    product: GpuBuffer,
    trace_gate: GpuBuffer,
    trace_activated: GpuBuffer,
    trace_gelu_cubic: Option<GpuBuffer>,
    trace_gelu_argument: Option<GpuBuffer>,
    trace_gelu_tanh: Option<GpuBuffer>,
    ffn_trace_width: usize,
    ple_lookup: GpuBuffer,
    ple_projected: GpuBuffer,
    ple: GpuBuffer,
    logits: GpuBuffer,
    validity: GpuBuffer,
    stage_max_abs: GpuBuffer,
    hidden: GpuBuffer,
    ple_total: GpuBuffer,
    ple_width: GpuBuffer,
    head_full: GpuBuffer,
    head_swa: GpuBuffer,
    q_width_full: GpuBuffer,
    q_width_swa: GpuBuffer,
    ffn_widths: Vec<GpuBuffer>,
    ple_offsets: Vec<GpuBuffer>,
    layers: GpuBuffer,
    heads: GpuBuffer,
    kv_heads: GpuBuffer,
    vocab: GpuBuffer,
    capacity: GpuBuffer,
    one: GpuBuffer,
    epsilon: GpuBuffer,
    embed_scale: GpuBuffer,
    ple_projection_scale: GpuBuffer,
    ple_input_scale: GpuBuffer,
    ple_embedding_scale: GpuBuffer,
    final_softcap: GpuBuffer,
    rope_full_cos: GpuBuffer,
    rope_full_sin: GpuBuffer,
    rope_swa_cos: GpuBuffer,
    rope_swa_sin: GpuBuffer,
    rope_freq_factors: Vec<f32>,
    pending_weight_upload_bytes: u64,
}

impl<'a> Gemma4E2bExecutor<'a> {
    pub fn new(model: &'a Gemma4E2bModel, max_context: usize) -> Result<Self> {
        Self::new_with_kv_cache(model, max_context, Gemma4KvCacheType::F32)
    }

    pub fn new_with_kv_cache(
        model: &'a Gemma4E2bModel,
        max_context: usize,
        kv_cache_type: Gemma4KvCacheType,
    ) -> Result<Self> {
        ensure!(
            max_context > 0,
            "Gemma executor max_context must be positive"
        );
        let c = &model.config;
        ensure!(
            c.key_length == c.value_length,
            "Gemma E2B requires equal K/V dimensions"
        );
        ensure!(
            c.key_length == c.rope_dimensions,
            "Gemma E2B only supports full RoPE over K/Q head width"
        );
        let providers = (0..c.layers)
            .map(|layer| {
                model
                    .gguf()
                    .tensors
                    .iter()
                    .any(|tensor| tensor.name == format!("blk.{layer}.attn_k.weight"))
            })
            .collect::<Vec<_>>();
        let kv_sources = gemma4_shared_kv_sources(&c.sliding_pattern, &providers)?;
        let runtime = model.runtime();
        let allocate = |count: usize| -> Result<GpuBuffer> {
            runtime
                .allocate(
                    count
                        .checked_mul(4)
                        .context("Gemma resident arena size overflow")?,
                )
                .map_err(anyhow::Error::from)
        };
        let h = c.hidden_size;
        let head = c.key_length.max(c.key_length_swa);
        let q_width = c.attention_heads * head;
        let ple_total = c.layers * c.per_layer_embedding_size;
        let max_ffn = c
            .feed_forward_sizes
            .iter()
            .copied()
            .max()
            .context("Gemma E2B has no FFN size")?;
        let trace_gelu = std::env::var_os("ATLAS_GEMMA4_TRACE_STAGES").is_some()
            && std::env::var_os("ATLAS_GEMMA4_TRACE_GELU").is_some();
        let rope_freqs = model
            .gguf()
            .tensors
            .iter()
            .find(|tensor| tensor.name == "rope_freqs.weight")
            .context("Gemma 4 GGUF is missing rope_freqs.weight")?;
        ensure!(
            rope_freqs.tensor_type == GgufTensorType::F32 && rope_freqs.dims == [c.key_length / 2],
            "Gemma 4 rope_freqs.weight must be F32 [{}]",
            c.key_length / 2
        );
        let rope_freq_factors = model
            .gguf()
            .tensor_data(rope_freqs)?
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 RoPE factor bytes")))
            .collect::<Vec<_>>();
        ensure!(
            rope_freq_factors
                .iter()
                .all(|factor| factor.is_finite() && *factor > 0.0),
            "Gemma 4 rope_freqs.weight contains a non-finite or non-positive factor"
        );
        let kv = providers
            .iter()
            .enumerate()
            .map(|(layer, provider)| {
                provider
                    .then(|| {
                        let source_head = if c.sliding_pattern[layer] {
                            c.key_length_swa
                        } else {
                            c.key_length
                        };
                        let bytes = kv_cache_type.cache_bytes(max_context, source_head)?;
                        runtime.allocate(bytes).map_err(anyhow::Error::from)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let weight_upload_bytes = model.ensure_resident_weights()?;
        Ok(Self {
            model,
            max_context,
            position: 0,
            kv_sources,
            kv,
            kv_cache_type,
            token: runtime.allocate(4)?,
            position_buffer: runtime.allocate(4)?,
            selected: runtime.allocate(4)?,
            state: allocate(h)?,
            residual: allocate(h)?,
            norm: allocate(h)?,
            q: allocate(q_width)?,
            q_rot: allocate(q_width)?,
            rope_input: allocate(q_width)?,
            rope_output: allocate(q_width)?,
            k: allocate(head)?,
            k_rot: allocate(head)?,
            v: allocate(head)?,
            attention: allocate(q_width)?,
            work: allocate(h)?,
            gate: allocate(max_ffn)?,
            up: allocate(max_ffn)?,
            activated: allocate(max_ffn)?,
            product: allocate(max_ffn)?,
            trace_gate: allocate(c.layers * max_ffn)?,
            trace_activated: allocate(c.layers * max_ffn)?,
            trace_gelu_cubic: trace_gelu
                .then(|| allocate(c.layers * max_ffn))
                .transpose()?,
            trace_gelu_argument: trace_gelu
                .then(|| allocate(c.layers * max_ffn))
                .transpose()?,
            trace_gelu_tanh: trace_gelu
                .then(|| allocate(c.layers * max_ffn))
                .transpose()?,
            ffn_trace_width: max_ffn,
            ple_lookup: allocate(ple_total)?,
            ple_projected: allocate(ple_total)?,
            ple: allocate(ple_total)?,
            logits: allocate(c.vocab_size)?,
            validity: runtime.allocate(4)?,
            stage_max_abs: runtime.allocate(
                (GEMMA4_TRACE_GLOBAL_STAGES + c.layers * GEMMA4_TRACE_STAGES_PER_LAYER) * 4,
            )?,
            hidden: runtime.upload_u32(&[u32::try_from(h)?])?,
            ple_total: runtime.upload_u32(&[u32::try_from(ple_total)?])?,
            ple_width: runtime.upload_u32(&[u32::try_from(c.per_layer_embedding_size)?])?,
            head_full: runtime.upload_u32(&[u32::try_from(c.key_length)?])?,
            head_swa: runtime.upload_u32(&[u32::try_from(c.key_length_swa)?])?,
            q_width_full: runtime
                .upload_u32(&[u32::try_from(c.attention_heads * c.key_length)?])?,
            q_width_swa: runtime
                .upload_u32(&[u32::try_from(c.attention_heads * c.key_length_swa)?])?,
            ffn_widths: c
                .feed_forward_sizes
                .iter()
                .map(|width| -> Result<GpuBuffer> {
                    Ok(runtime.upload_u32(&[u32::try_from(*width)?])?)
                })
                .collect::<Result<Vec<_>>>()?,
            ple_offsets: (0..c.layers)
                .map(|layer| -> Result<GpuBuffer> {
                    Ok(
                        runtime
                            .upload_u32(&[u32::try_from(layer * c.per_layer_embedding_size)?])?,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            layers: runtime.upload_u32(&[u32::try_from(c.layers)?])?,
            heads: runtime.upload_u32(&[u32::try_from(c.attention_heads)?])?,
            kv_heads: runtime.upload_u32(&[1])?,
            vocab: runtime.upload_u32(&[u32::try_from(c.vocab_size)?])?,
            capacity: runtime.upload_u32(&[u32::try_from(max_context)?])?,
            one: runtime.upload_u32(&[1])?,
            epsilon: runtime.upload_f32(&[c.rms_norm_eps])?,
            embed_scale: runtime.upload_f32(&[(h as f32).sqrt()])?,
            ple_projection_scale: runtime.upload_f32(&[(h as f32).sqrt().recip()])?,
            ple_input_scale: runtime.upload_f32(&[2.0f32.sqrt().recip()])?,
            ple_embedding_scale: runtime
                .upload_f32(&[(c.per_layer_embedding_size as f32).sqrt()])?,
            final_softcap: runtime.upload_f32(&[c.final_logit_softcap])?,
            rope_full_cos: allocate(head / 2)?,
            rope_full_sin: allocate(head / 2)?,
            rope_swa_cos: allocate(head / 2)?,
            rope_swa_sin: allocate(head / 2)?,
            rope_freq_factors,
            pending_weight_upload_bytes: weight_upload_bytes,
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        self.model.resident_weight_bytes()
            + self
                .kv
                .iter()
                .flatten()
                .map(|v| v.bytes() as u64)
                .sum::<u64>()
            + [
                &self.token,
                &self.position_buffer,
                &self.selected,
                &self.state,
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
                &self.work,
                &self.gate,
                &self.up,
                &self.activated,
                &self.product,
                &self.trace_gate,
                &self.trace_activated,
                &self.ple_lookup,
                &self.ple_projected,
                &self.ple,
                &self.logits,
                &self.validity,
                &self.stage_max_abs,
            ]
            .iter()
            .map(|v| v.bytes() as u64)
            .sum::<u64>()
            + [
                self.trace_gelu_cubic.as_ref(),
                self.trace_gelu_argument.as_ref(),
                self.trace_gelu_tanh.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|v| v.bytes() as u64)
            .sum::<u64>()
    }

    pub fn kv_cache_type(&self) -> Gemma4KvCacheType {
        self.kv_cache_type
    }

    pub fn kv_cache_bytes(&self) -> u64 {
        self.kv
            .iter()
            .flatten()
            .map(|buffer| buffer.bytes() as u64)
            .sum()
    }

    fn weight(&self, name: &str, expected: GgufTensorType) -> Result<GpuBuffer> {
        ensure!(
            self.model.resident_weight_format(name)? == expected,
            "Gemma tensor `{name}` has an unsupported resident format"
        );
        self.model.resident_weight(name)
    }

    fn matvec(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width_buffer: &GpuBuffer,
        output_width: usize,
        format: GgufTensorType,
    ) -> Result<()> {
        let kernel = match format {
            GgufTensorType::Q4_0 => "matvec_q4_0_16row",
            GgufTensorType::Q6K => "matvec_q6_k_8row",
            GgufTensorType::F16 => "matvec_f16",
            other => anyhow::bail!("unsupported Gemma matvec format {other:?}"),
        };
        let buffers = &[input, weight, output, input_width, output_width_buffer];
        if format == GgufTensorType::Q4_0 {
            command.dispatch_threadgroups_1d(kernel, buffers, output_width.div_ceil(16), 128)?;
        } else if format == GgufTensorType::Q6K {
            command.dispatch_threadgroups_1d(kernel, buffers, output_width.div_ceil(8), 128)?;
        } else {
            command.dispatch_1d(kernel, buffers, output_width)?;
        }
        Ok(())
    }

    pub fn generate_greedy(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<Gemma4Generation> {
        static NEVER_CANCEL: AtomicBool = AtomicBool::new(false);
        self.generate_greedy_stream(prompt, max_new_tokens, &NEVER_CANCEL, |_| Ok(()))
    }

    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Profile the requested decode ordinals with Metal's exact-per-dispatch
    /// timing path. This is intentionally a separate API: production chat and
    /// generate keep their single command-buffer decode boundary.
    pub fn profile_decode(
        &mut self,
        prompt: &str,
        decode_tokens: usize,
    ) -> Result<Gemma4DecodeProfile> {
        ensure!(
            decode_tokens >= 128,
            "Gemma decode profile needs at least 128 tokens"
        );
        let prompt_ids = self.model.tokenize(prompt)?;
        ensure!(!prompt_ids.is_empty(), "prompt tokenizes to no tokens");
        ensure!(
            prompt_ids.len() + decode_tokens <= self.max_context,
            "Gemma executor context exhausted"
        );
        self.position = 0;
        let mut selected = self
            .forward_tokens(&prompt_ids, true)?
            .context("Gemma prefill did not select a first decode token")?;
        let targets = [1usize, 32, 64, 128];
        let runtime = self.model.runtime();
        let readback_before = runtime.readback_bytes();
        let mut samples = Vec::with_capacity(targets.len());
        for decode_position in 1..=decode_tokens {
            let context_position = self.position;
            let timings = if targets.contains(&decode_position) {
                Some(self.forward_token_profiled(selected)?)
            } else {
                selected = self.forward_token(selected)?;
                None
            };
            if let Some((next, timings)) = timings {
                selected = next;
                let mut kernels: BTreeMap<&'static str, Gemma4DecodeKernelProfile> =
                    BTreeMap::new();
                for timing in timings {
                    let family = gemma4_kernel_family(timing.kernel);
                    let entry = kernels.entry(family).or_insert(Gemma4DecodeKernelProfile {
                        family,
                        dispatches: 0,
                        gpu_nanos: 0,
                        cpu_encode_nanos: 0,
                    });
                    entry.dispatches += 1;
                    entry.gpu_nanos += timing.timing.gpu_time.unwrap_or_default().as_nanos();
                    entry.cpu_encode_nanos += timing.cpu_encode.as_nanos();
                }
                samples.push(Gemma4DecodeProfileSample {
                    decode_position,
                    context_position,
                    attention_key_count: context_position + 1,
                    full_attention_layers: self
                        .model
                        .config
                        .sliding_pattern
                        .iter()
                        .filter(|&&sliding| !sliding)
                        .count(),
                    sliding_attention_layers: self
                        .model
                        .config
                        .sliding_pattern
                        .iter()
                        .filter(|&&sliding| sliding)
                        .count(),
                    resident_bytes: self.resident_bytes(),
                    readback_bytes: runtime.readback_bytes() - readback_before,
                    kernels: kernels.into_values().collect(),
                });
            }
        }
        Ok(Gemma4DecodeProfile {
            prompt_tokens: prompt_ids.len(),
            requested_decode_tokens: decode_tokens,
            prefill_path: "resident_chunked_command",
            attention_kernel: gemma4_attention_kernel(self.kv_cache_type),
            kv_cache_type: self.kv_cache_type,
            samples,
        })
    }

    pub fn generate_greedy_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        self.generate_greedy_stream_inner(prompt, max_new_tokens, cancelled, false, false, emit)
    }

    pub fn generate_greedy_chat_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        self.generate_greedy_stream_inner(prompt, max_new_tokens, cancelled, true, false, emit)
    }

    /// Generate a fixed number of selections for performance measurement.
    ///
    /// This is deliberately diagnostic-only: unlike `generate_greedy_stream`
    /// and `generate_greedy_chat_stream`, it continues after EOS and chat
    /// end-turn markers so all compared cache modes execute an identical
    /// decode workload. It must not be used by normal user-facing generation.
    pub fn generate_greedy_fixed_benchmark_stream(
        &mut self,
        prompt: &str,
        decode_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        self.generate_greedy_stream_inner(prompt, decode_tokens, cancelled, false, true, emit)
    }

    fn generate_greedy_stream_inner(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        stop_on_end_turn: bool,
        continue_after_eos: bool,
        mut emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
        let prompt_ids = self.model.tokenize(prompt)?;
        ensure!(!prompt_ids.is_empty(), "prompt tokenizes to no tokens");
        ensure!(
            prompt_ids.len() + max_new_tokens <= self.max_context,
            "Gemma executor context exhausted"
        );
        let runtime = self.model.runtime();
        let command_before = runtime.command_buffer_count();
        let readback_before = runtime.readback_bytes();
        let started = Instant::now();
        self.position = 0;
        let prefill_started = Instant::now();
        let plan = Gemma4PrefillPlan::new(prompt_ids.len(), self.max_context)?;
        let mut selected = 0;
        let chunk_count = plan.chunks;
        for (chunk_index, chunk) in prompt_ids.chunks(plan.chunk_size).enumerate() {
            if let Some(token) = self.forward_tokens(chunk, chunk_index + 1 == chunk_count)? {
                selected = token;
            }
        }
        let prefill = prefill_started.elapsed();
        let prefill_commands = runtime.command_buffer_count() - command_before;
        let decode_started = Instant::now();
        let mut generated = Vec::new();
        let mut finish_reason = Gemma4FinishReason::MaxTokens;
        let mut first_eos_position = None;
        let mut decoded = String::new();
        let mut token_latency = prefill;
        for index in 0..max_new_tokens {
            if cancelled.load(Ordering::Acquire) {
                finish_reason = Gemma4FinishReason::Cancelled;
                break;
            }
            generated.push(selected);
            let next_decoded = self.model.decode(&generated)?;
            let fragment = next_decoded
                .strip_prefix(&decoded)
                .unwrap_or(&next_decoded)
                .to_owned();
            decoded = next_decoded;
            emit(Gemma4TokenEvent {
                token_id: selected,
                text: fragment,
                latency: token_latency,
            })?;
            if selected == self.model.config.eos_token_id {
                first_eos_position.get_or_insert(index + 1);
            }
            if !continue_after_eos
                && gemma4_should_finish(
                    selected,
                    self.model.config.eos_token_id,
                    &decoded,
                    stop_on_end_turn,
                )
            {
                finish_reason = Gemma4FinishReason::Eos;
                break;
            }
            if index + 1 < max_new_tokens {
                let token_started = Instant::now();
                selected = self.forward_token(selected)?;
                token_latency = token_started.elapsed();
            }
        }
        let ids = [prompt_ids.clone(), generated.clone()].concat();
        let final_logits = if std::env::var_os("ATLAS_GEMMA4_TRACE_LOGITS").is_some() {
            runtime.read_f32(&self.logits, self.model.config.vocab_size)?
        } else {
            Vec::new()
        };
        let weight_upload_bytes = std::mem::take(&mut self.pending_weight_upload_bytes);
        Ok(Gemma4Generation {
            generation: Generation {
                prompt_token_ids: prompt_ids,
                generated_token_ids: generated,
                text: self.model.decode(&ids)?,
                trace: LayerTrace::default(),
                final_logits,
            },
            metrics: Gemma4Metrics {
                resident_bytes: self.resident_bytes(),
                weight_upload_bytes,
                readback_bytes: runtime.readback_bytes() - readback_before,
                command_buffers: runtime.command_buffer_count() - command_before,
                prefill_command_buffers: prefill_commands,
                decode_command_buffers: runtime.command_buffer_count()
                    - command_before
                    - prefill_commands,
                prefill,
                decode: decode_started.elapsed(),
                host_wall_time: started.elapsed(),
                prefill_path: "resident_chunked_command",
                prefill_chunk_size: plan.chunk_size,
                prefill_chunks: plan.chunks,
                attention_kernel: gemma4_attention_kernel(self.kv_cache_type),
                kv_cache_type: self.kv_cache_type,
                kv_cache_bytes: self.kv_cache_bytes(),
            },
            finish_reason,
            first_eos_position,
        })
    }

    fn forward_token(&mut self, token: u32) -> Result<u32> {
        Ok(self.forward_token_inner(token, false)?.0)
    }

    fn forward_token_profiled(
        &mut self,
        token: u32,
    ) -> Result<(u32, Vec<atlas_metal::ResidentKernelTiming>)> {
        self.forward_token_inner(token, true)
    }

    fn forward_token_inner(
        &mut self,
        token: u32,
        exact_profile: bool,
    ) -> Result<(u32, Vec<atlas_metal::ResidentKernelTiming>)> {
        ensure!(
            self.position < self.max_context,
            "Gemma executor context exhausted"
        );
        let runtime = self.model.runtime();
        runtime.write_u32(&self.token, &[token])?;
        runtime.write_u32(&self.position_buffer, &[u32::try_from(self.position)?])?;
        let rope_pairs = self
            .model
            .config
            .key_length
            .max(self.model.config.key_length_swa)
            / 2;
        let mut full_cos = vec![0.0; rope_pairs];
        let mut full_sin = vec![0.0; rope_pairs];
        let mut swa_cos = vec![0.0; rope_pairs];
        let mut swa_sin = vec![0.0; rope_pairs];
        for pair in 0..rope_pairs {
            let full_angle = gemma4_rope_angle(
                self.position,
                pair,
                self.model.config.rope_dimensions,
                self.model.config.rope_theta,
                self.rope_freq_factors[pair],
            );
            let swa_angle = gemma4_rope_angle(
                self.position,
                pair,
                self.model.config.rope_dimensions_swa,
                self.model.config.rope_theta_swa,
                1.0,
            );
            full_cos[pair] = full_angle.cos();
            full_sin[pair] = full_angle.sin();
            swa_cos[pair] = swa_angle.cos();
            swa_sin[pair] = swa_angle.sin();
        }
        runtime.write_f32(&self.rope_full_cos, &full_cos)?;
        runtime.write_f32(&self.rope_full_sin, &full_sin)?;
        runtime.write_f32(&self.rope_swa_cos, &swa_cos)?;
        runtime.write_f32(&self.rope_swa_sin, &swa_sin)?;
        let rope_full_cos = self.rope_full_cos.clone();
        let rope_full_sin = self.rope_full_sin.clone();
        let rope_swa_cos = self.rope_swa_cos.clone();
        let rope_swa_sin = self.rope_swa_sin.clone();
        let key_counts = runtime.upload_u32(&gemma4_attention_key_count_table(
            self.position,
            1,
            &self.model.config.sliding_pattern,
            self.model.config.sliding_window,
        )?)?;
        let trace_stages = std::env::var_os("ATLAS_GEMMA4_TRACE_STAGES").is_some();
        let mut command =
            runtime.begin_resident_command_with_exact_timing(trace_stages || exact_profile)?;
        self.encode_current_token(
            &mut command,
            0,
            true,
            rope_pairs,
            &rope_full_cos,
            &rope_full_sin,
            &rope_swa_cos,
            &rope_swa_sin,
            &key_counts,
        )?;
        let timings = command.take_kernel_timings();
        command.finish()?;
        self.position += 1;
        Ok((runtime.read_u32(&self.selected)?, timings))
    }

    fn forward_tokens(&mut self, tokens: &[u32], select_last: bool) -> Result<Option<u32>> {
        ensure!(!tokens.is_empty(), "Gemma token batch must not be empty");
        ensure!(
            self.position + tokens.len() <= self.max_context,
            "Gemma executor context exhausted"
        );
        // Prompt tokens are known before execution. Keep them and their
        // positions in GPU-visible buffers, then encode the dependent token
        // forwards into one command buffer. Scratch and KV buffers are reused
        // in dispatch order; only the final token selection is read back.
        let runtime = self.model.runtime();
        let token_batch = runtime.upload_u32(tokens)?;
        let positions = (self.position..self.position + tokens.len())
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let position_batch = runtime.upload_u32(&positions)?;
        let rope_pairs = self
            .model
            .config
            .key_length
            .max(self.model.config.key_length_swa)
            / 2;
        let mut full_cos = vec![0.0; tokens.len() * rope_pairs];
        let mut full_sin = vec![0.0; tokens.len() * rope_pairs];
        let mut swa_cos = vec![0.0; tokens.len() * rope_pairs];
        let mut swa_sin = vec![0.0; tokens.len() * rope_pairs];
        for (batch_index, position) in (self.position..self.position + tokens.len()).enumerate() {
            for pair in 0..rope_pairs {
                let full_angle = gemma4_rope_angle(
                    position,
                    pair,
                    self.model.config.rope_dimensions,
                    self.model.config.rope_theta,
                    self.rope_freq_factors[pair],
                );
                let swa_angle = gemma4_rope_angle(
                    position,
                    pair,
                    self.model.config.rope_dimensions_swa,
                    self.model.config.rope_theta_swa,
                    1.0,
                );
                full_cos[batch_index * rope_pairs + pair] = full_angle.cos();
                full_sin[batch_index * rope_pairs + pair] = full_angle.sin();
                swa_cos[batch_index * rope_pairs + pair] = swa_angle.cos();
                swa_sin[batch_index * rope_pairs + pair] = swa_angle.sin();
            }
        }
        let full_cos = runtime.upload_f32(&full_cos)?;
        let full_sin = runtime.upload_f32(&full_sin)?;
        let swa_cos = runtime.upload_f32(&swa_cos)?;
        let swa_sin = runtime.upload_f32(&swa_sin)?;
        let key_counts = runtime.upload_u32(&gemma4_attention_key_count_table(
            self.position,
            tokens.len(),
            &self.model.config.sliding_pattern,
            self.model.config.sliding_window,
        )?)?;
        let trace_stages = std::env::var_os("ATLAS_GEMMA4_TRACE_STAGES").is_some();
        let mut command = runtime.begin_resident_command_with_exact_timing(trace_stages)?;
        for index in 0..tokens.len() {
            command.dispatch_1d_at(
                "copy_u32",
                &[
                    (&token_batch, index * std::mem::size_of::<u32>()),
                    (&self.token, 0),
                    (&self.one, 0),
                ],
                1,
            )?;
            command.dispatch_1d_at(
                "copy_u32",
                &[
                    (&position_batch, index * std::mem::size_of::<u32>()),
                    (&self.position_buffer, 0),
                    (&self.one, 0),
                ],
                1,
            )?;
            self.encode_current_token(
                &mut command,
                index,
                select_last && index + 1 == tokens.len(),
                rope_pairs,
                &full_cos,
                &full_sin,
                &swa_cos,
                &swa_sin,
                &key_counts,
            )?;
            self.position += 1;
        }
        command.finish()?;
        select_last
            .then(|| runtime.read_u32(&self.selected))
            .transpose()
            .map_err(Into::into)
    }

    fn encode_current_token(
        &mut self,
        mut command: &mut atlas_metal::ResidentCommand<'_>,
        batch_index: usize,
        select_output: bool,
        rope_pairs: usize,
        full_cos: &GpuBuffer,
        full_sin: &GpuBuffer,
        swa_cos: &GpuBuffer,
        swa_sin: &GpuBuffer,
        key_counts: &GpuBuffer,
    ) -> Result<()> {
        ensure!(
            self.position < self.max_context,
            "Gemma executor context exhausted"
        );
        let c = &self.model.config;
        let runtime = self.model.runtime();
        let h = c.hidden_size;
        let ple_total = c.layers * c.per_layer_embedding_size;
        let trace_stages = std::env::var_os("ATLAS_GEMMA4_TRACE_STAGES").is_some();
        let trace_gelu = trace_stages && std::env::var_os("ATLAS_GEMMA4_TRACE_GELU").is_some();
        let trace_sync = trace_stages && std::env::var_os("ATLAS_GEMMA4_TRACE_SYNC").is_some();
        if trace_stages {
            runtime.write_u32(&self.validity, &[u32::MAX])?;
        }
        let token_embd = self.weight("token_embd.weight", GgufTensorType::Q6K)?;
        let per_layer_embd = self.weight("per_layer_token_embd.weight", GgufTensorType::Q6K)?;
        let per_layer_proj = self.weight("per_layer_model_proj.weight", GgufTensorType::F16)?;
        let per_layer_norm = self.weight("per_layer_proj_norm.weight", GgufTensorType::F32)?;
        command.dispatch_1d(
            "embedding_lookup_q6_k",
            &[
                &token_embd,
                &self.token,
                &self.state,
                &self.vocab,
                &self.hidden,
                &self.one,
            ],
            h,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[&self.state, &self.state, &self.embed_scale, &self.hidden],
            h,
        )?;
        if trace_stages {
            let slot = runtime.upload_u32(&[0])?;
            command.dispatch_1d(
                "first_nonfinite_f32",
                &[&self.state, &self.validity, &self.hidden, &slot],
                1,
            )?;
            command.dispatch_1d(
                "max_abs_f32",
                &[&self.state, &self.stage_max_abs, &self.hidden, &slot],
                1,
            )?;
        }
        let ple_total_buffer = &self.ple_total;
        command.dispatch_1d(
            "embedding_lookup_q6_k",
            &[
                &per_layer_embd,
                &self.token,
                &self.ple_lookup,
                &self.vocab,
                &ple_total_buffer,
                &self.one,
            ],
            ple_total,
        )?;
        if trace_stages {
            let slot = runtime.upload_u32(&[1])?;
            command.dispatch_1d(
                "first_nonfinite_f32",
                &[&self.ple_lookup, &self.validity, &ple_total_buffer, &slot],
                1,
            )?;
            command.dispatch_1d(
                "max_abs_f32",
                &[
                    &self.ple_lookup,
                    &self.stage_max_abs,
                    &ple_total_buffer,
                    &slot,
                ],
                1,
            )?;
        }
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.ple_lookup,
                &self.ple_lookup,
                &self.ple_embedding_scale,
                &ple_total_buffer,
            ],
            ple_total,
        )?;
        if trace_stages {
            let slot = runtime.upload_u32(&[2])?;
            command.dispatch_1d(
                "first_nonfinite_f32",
                &[&self.ple_lookup, &self.validity, &ple_total_buffer, &slot],
                1,
            )?;
            command.dispatch_1d(
                "max_abs_f32",
                &[
                    &self.ple_lookup,
                    &self.stage_max_abs,
                    &ple_total_buffer,
                    &slot,
                ],
                1,
            )?;
        }
        self.matvec(
            &mut command,
            &self.state,
            &per_layer_proj,
            &self.ple_projected,
            &self.hidden,
            &self.ple_total,
            ple_total,
            GgufTensorType::F16,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.ple_projected,
                &self.ple_projected,
                &self.ple_projection_scale,
                &ple_total_buffer,
            ],
            ple_total,
        )?;
        if trace_stages {
            let slot = runtime.upload_u32(&[3])?;
            command.dispatch_1d(
                "first_nonfinite_f32",
                &[
                    &self.ple_projected,
                    &self.validity,
                    &ple_total_buffer,
                    &slot,
                ],
                1,
            )?;
            command.dispatch_1d(
                "max_abs_f32",
                &[
                    &self.ple_projected,
                    &self.stage_max_abs,
                    &ple_total_buffer,
                    &slot,
                ],
                1,
            )?;
        }
        command.dispatch_1d(
            "rms_norm_groups_in_place_stable_f32",
            &[
                &self.ple_projected,
                &per_layer_norm,
                &self.ple_width,
                &self.layers,
                &self.epsilon,
            ],
            ple_total,
        )?;
        if trace_stages {
            let slot = runtime.upload_u32(&[4])?;
            command.dispatch_1d(
                "first_nonfinite_f32",
                &[
                    &self.ple_projected,
                    &self.validity,
                    &ple_total_buffer,
                    &slot,
                ],
                1,
            )?;
            command.dispatch_1d(
                "max_abs_f32",
                &[
                    &self.ple_projected,
                    &self.stage_max_abs,
                    &ple_total_buffer,
                    &slot,
                ],
                1,
            )?;
        }
        command.dispatch_1d(
            "vector_add_f32",
            &[
                &self.ple_lookup,
                &self.ple_projected,
                &self.ple,
                &ple_total_buffer,
            ],
            ple_total,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.ple,
                &self.ple,
                &self.ple_input_scale,
                &ple_total_buffer,
            ],
            ple_total,
        )?;
        if trace_stages {
            let slot = runtime.upload_u32(&[5])?;
            command.dispatch_1d(
                "first_nonfinite_f32",
                &[&self.ple, &self.validity, &ple_total_buffer, &slot],
                1,
            )?;
            command.dispatch_1d(
                "max_abs_f32",
                &[&self.ple, &self.stage_max_abs, &ple_total_buffer, &slot],
                1,
            )?;
        }
        for layer in 0..c.layers {
            let p = format!("blk.{layer}");
            let sliding = c.sliding_pattern[layer];
            let head = if sliding {
                c.key_length_swa
            } else {
                c.key_length
            };
            let q_width = c.attention_heads * head;
            let q_width_buffer = if sliding {
                &self.q_width_swa
            } else {
                &self.q_width_full
            };
            let rope_offset = batch_index * rope_pairs * std::mem::size_of::<f32>();
            let (cos, sin) = if sliding {
                (swa_cos, swa_sin)
            } else {
                (full_cos, full_sin)
            };
            let head_width = if sliding {
                &self.head_swa
            } else {
                &self.head_full
            };
            let attn_norm = self.weight(&format!("{p}.attn_norm.weight"), GgufTensorType::F32)?;
            let wq = self.weight(&format!("{p}.attn_q.weight"), GgufTensorType::Q4_0)?;
            let q_norm = self.weight(&format!("{p}.attn_q_norm.weight"), GgufTensorType::F32)?;
            command.dispatch_threadgroups_1d(
                "rms_norm_decode_f32",
                &[
                    &self.state,
                    &attn_norm,
                    &self.norm,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            self.matvec(
                &mut command,
                &self.norm,
                &wq,
                &self.q,
                &self.hidden,
                q_width_buffer,
                q_width,
                GgufTensorType::Q4_0,
            )?;
            command.dispatch_1d(
                "rms_norm_groups_in_place_f32",
                &[&self.q, &q_norm, head_width, &self.heads, &self.epsilon],
                q_width,
            )?;
            command.dispatch_1d(
                "rope_half_to_interleaved_f32",
                &[&self.q, &self.rope_input, &self.heads, head_width],
                q_width / 2,
            )?;
            command.dispatch_1d_at(
                "rope_f32",
                &[
                    (&self.rope_input, 0),
                    (cos, rope_offset),
                    (sin, rope_offset),
                    (&self.rope_output, 0),
                    (head_width, 0),
                ],
                q_width / 2,
            )?;
            command.dispatch_1d(
                "rope_interleaved_to_half_f32",
                &[&self.rope_output, &self.q_rot, &self.heads, head_width],
                q_width / 2,
            )?;
            let source = self.kv_sources[layer];
            if source == layer {
                let wk = self.weight(&format!("{p}.attn_k.weight"), GgufTensorType::Q4_0)?;
                let wv = self.weight(&format!("{p}.attn_v.weight"), GgufTensorType::Q4_0)?;
                let k_norm =
                    self.weight(&format!("{p}.attn_k_norm.weight"), GgufTensorType::F32)?;
                self.matvec(
                    &mut command,
                    &self.norm,
                    &wk,
                    &self.k,
                    &self.hidden,
                    head_width,
                    head,
                    GgufTensorType::Q4_0,
                )?;
                self.matvec(
                    &mut command,
                    &self.norm,
                    &wv,
                    &self.v,
                    &self.hidden,
                    head_width,
                    head,
                    GgufTensorType::Q4_0,
                )?;
                command.dispatch_1d(
                    "rms_norm_groups_in_place_f32",
                    &[&self.k, &k_norm, head_width, &self.one, &self.epsilon],
                    head,
                )?;
                command.dispatch_1d(
                    "rms_norm_groups_in_place_unweighted_f32",
                    &[&self.v, head_width, &self.one, &self.epsilon],
                    head,
                )?;
                command.dispatch_1d(
                    "rope_half_to_interleaved_f32",
                    &[&self.k, &self.rope_input, &self.one, head_width],
                    head / 2,
                )?;
                command.dispatch_1d_at(
                    "rope_f32",
                    &[
                        (&self.rope_input, 0),
                        (cos, rope_offset),
                        (sin, rope_offset),
                        (&self.rope_output, 0),
                        (head_width, 0),
                    ],
                    head / 2,
                )?;
                command.dispatch_1d(
                    "rope_interleaved_to_half_f32",
                    &[&self.rope_output, &self.k_rot, &self.one, head_width],
                    head / 2,
                )?;
                let cache = self.kv[layer].as_ref().expect("KV provider has cache");
                let append_count = if self.kv_cache_type == Gemma4KvCacheType::F32 {
                    head
                } else {
                    head / 32
                };
                command.dispatch_1d(
                    gemma4_kv_append_kernel(self.kv_cache_type),
                    &[
                        &self.k_rot,
                        &self.v,
                        cache,
                        head_width,
                        &self.capacity,
                        &self.position_buffer,
                    ],
                    append_count,
                )?;
            }
            // Gemma's cache source is explicit and observable through kv_sources; the existing resident attention kernel remains valid for one KV head.
            let cache = self.kv[source].as_ref().expect("Gemma KV source has cache");
            let key_count_offset = batch_index
                .checked_mul(c.layers)
                .and_then(|entry| entry.checked_add(layer))
                .and_then(|entry| entry.checked_mul(std::mem::size_of::<u32>()))
                .context("Gemma attention control-table offset overflows")?;
            command.dispatch_threadgroups_1d_at(
                gemma4_attention_kernel(self.kv_cache_type),
                &[
                    (&self.q_rot, 0),
                    (cache, 0),
                    (&self.attention, 0),
                    (&self.heads, 0),
                    (&self.kv_heads, 0),
                    (head_width, 0),
                    (&self.capacity, 0),
                    (key_counts, key_count_offset),
                ],
                c.attention_heads,
                128,
            )?;
            let wo = self.weight(&format!("{p}.attn_output.weight"), GgufTensorType::Q4_0)?;
            self.matvec(
                &mut command,
                &self.attention,
                &wo,
                &self.work,
                q_width_buffer,
                &self.hidden,
                h,
                GgufTensorType::Q4_0,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 1))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.work, &self.validity, &self.hidden, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.work, &self.stage_max_abs, &self.hidden, &slot],
                    1,
                )?;
            }
            let post_attn = self.weight(
                &format!("{p}.post_attention_norm.weight"),
                GgufTensorType::F32,
            )?;
            command.dispatch_threadgroups_1d(
                "rms_norm_decode_f32",
                &[
                    &self.work,
                    &post_attn,
                    &self.work,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            command.dispatch_1d(
                "vector_add_f32",
                &[&self.state, &self.work, &self.residual, &self.hidden],
                h,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 2))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.residual, &self.validity, &self.hidden, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.residual, &self.stage_max_abs, &self.hidden, &slot],
                    1,
                )?;
            }
            let ffn_norm = self.weight(&format!("{p}.ffn_norm.weight"), GgufTensorType::F32)?;
            command.dispatch_threadgroups_1d(
                "rms_norm_decode_f32",
                &[
                    &self.residual,
                    &ffn_norm,
                    &self.norm,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            let ffn = c.feed_forward_sizes[layer];
            let ffn_buffer = &self.ffn_widths[layer];
            let gate = self.weight(&format!("{p}.ffn_gate.weight"), GgufTensorType::Q4_0)?;
            let up = self.weight(&format!("{p}.ffn_up.weight"), GgufTensorType::Q4_0)?;
            let down = self.weight(&format!("{p}.ffn_down.weight"), GgufTensorType::Q4_0)?;
            self.matvec(
                &mut command,
                &self.norm,
                &gate,
                &self.gate,
                &self.hidden,
                ffn_buffer,
                ffn,
                GgufTensorType::Q4_0,
            )?;
            if trace_stages {
                let trace_offset = layer
                    .checked_mul(self.ffn_trace_width)
                    .and_then(|offset| offset.checked_mul(std::mem::size_of::<f32>()))
                    .context("Gemma FFN trace offset overflows")?;
                command.dispatch_1d_at(
                    "copy_f32",
                    &[
                        (&self.gate, 0),
                        (&self.trace_gate, trace_offset),
                        (&ffn_buffer, 0),
                    ],
                    ffn,
                )?;
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 3))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.gate, &self.validity, &ffn_buffer, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.gate, &self.stage_max_abs, &ffn_buffer, &slot],
                    1,
                )?;
            }
            self.matvec(
                &mut command,
                &self.norm,
                &up,
                &self.up,
                &self.hidden,
                ffn_buffer,
                ffn,
                GgufTensorType::Q4_0,
            )?;
            // Keep GELU out-of-place. The Metal kernel accepts distinct input and
            // output buffers, and this avoids relying on aliasing semantics before
            // the dependent gated product is encoded.
            if trace_gelu {
                let trace_offset = layer
                    .checked_mul(self.ffn_trace_width)
                    .and_then(|offset| offset.checked_mul(std::mem::size_of::<f32>()))
                    .context("Gemma FFN trace offset overflows")?;
                let cubic = self
                    .trace_gelu_cubic
                    .as_ref()
                    .context("Gemma GELU cubic trace buffer is unavailable")?;
                let argument = self
                    .trace_gelu_argument
                    .as_ref()
                    .context("Gemma GELU argument trace buffer is unavailable")?;
                let tanh = self
                    .trace_gelu_tanh
                    .as_ref()
                    .context("Gemma GELU tanh trace buffer is unavailable")?;
                command.dispatch_1d_at(
                    "gelu_trace_f32",
                    &[
                        (&self.gate, 0),
                        (&self.activated, 0),
                        (cubic, trace_offset),
                        (argument, trace_offset),
                        (tanh, trace_offset),
                        (&ffn_buffer, 0),
                    ],
                    ffn,
                )?;
            } else {
                command.dispatch_1d(
                    "gelu_f32",
                    &[&self.gate, &self.activated, &ffn_buffer],
                    ffn,
                )?;
            }
            if trace_stages {
                let trace_offset = layer
                    .checked_mul(self.ffn_trace_width)
                    .and_then(|offset| offset.checked_mul(std::mem::size_of::<f32>()))
                    .context("Gemma FFN trace offset overflows")?;
                command.dispatch_1d_at(
                    "copy_f32",
                    &[
                        (&self.activated, 0),
                        (&self.trace_activated, trace_offset),
                        (&ffn_buffer, 0),
                    ],
                    ffn,
                )?;
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 4))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.activated, &self.validity, &ffn_buffer, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.activated, &self.stage_max_abs, &ffn_buffer, &slot],
                    1,
                )?;
            }
            command.dispatch_1d(
                "vector_multiply_f32",
                &[&self.activated, &self.up, &self.product, &ffn_buffer],
                ffn,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 5))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.product, &self.validity, &ffn_buffer, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.product, &self.stage_max_abs, &ffn_buffer, &slot],
                    1,
                )?;
            }
            self.matvec(
                &mut command,
                &self.product,
                &down,
                &self.work,
                ffn_buffer,
                &self.hidden,
                h,
                GgufTensorType::Q4_0,
            )?;
            let post_ffn =
                self.weight(&format!("{p}.post_ffw_norm.weight"), GgufTensorType::F32)?;
            command.dispatch_threadgroups_1d(
                "rms_norm_decode_f32",
                &[
                    &self.work,
                    &post_ffn,
                    &self.work,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 6))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.work, &self.validity, &self.hidden, &slot],
                    1,
                )?;
            }
            command.dispatch_1d(
                "vector_add_f32",
                &[&self.residual, &self.work, &self.state, &self.hidden],
                h,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 7))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.state, &self.validity, &self.hidden, &slot],
                    1,
                )?;
            }
            let inp_gate = self.weight(&format!("{p}.inp_gate.weight"), GgufTensorType::Q4_0)?;
            let projection = self.weight(&format!("{p}.proj.weight"), GgufTensorType::Q4_0)?;
            let post_norm = self.weight(&format!("{p}.post_norm.weight"), GgufTensorType::F32)?;
            self.matvec(
                &mut command,
                &self.state,
                &inp_gate,
                &self.gate,
                &self.hidden,
                &self.ple_width,
                c.per_layer_embedding_size,
                GgufTensorType::Q4_0,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 8))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.gate, &self.validity, &self.ple_width, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.gate, &self.stage_max_abs, &self.ple_width, &slot],
                    1,
                )?;
            }
            command.dispatch_1d(
                "gelu_f32",
                &[&self.gate, &self.gate, &self.ple_width],
                c.per_layer_embedding_size,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 9))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.gate, &self.validity, &self.ple_width, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.gate, &self.stage_max_abs, &self.ple_width, &slot],
                    1,
                )?;
            }
            // Current layer PLE is a contiguous [256] slice in the resident [layer][width] table.
            let ple_offset = &self.ple_offsets[layer];
            command.dispatch_1d(
                "vector_multiply_offset_f32",
                &[
                    &self.gate,
                    &self.ple,
                    &self.activated,
                    &ple_offset,
                    &self.ple_width,
                ],
                c.per_layer_embedding_size,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 10))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.activated, &self.validity, &self.ple_width, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.activated, &self.stage_max_abs, &self.ple_width, &slot],
                    1,
                )?;
            }
            self.matvec(
                &mut command,
                &self.activated,
                &projection,
                &self.work,
                &self.ple_width,
                &self.hidden,
                h,
                GgufTensorType::Q4_0,
            )?;
            command.dispatch_threadgroups_1d(
                "rms_norm_decode_f32",
                &[
                    &self.work,
                    &post_norm,
                    &self.work,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            if trace_stages {
                let slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 11))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.work, &self.validity, &self.hidden, &slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.work, &self.stage_max_abs, &self.hidden, &slot],
                    1,
                )?;
            }
            command.dispatch_1d(
                "vector_add_f32",
                &[&self.state, &self.work, &self.state, &self.hidden],
                h,
            )?;
            let scale = self.weight(
                &format!("{p}.layer_output_scale.weight"),
                GgufTensorType::F32,
            )?;
            command.dispatch_1d(
                "scalar_multiply_f32",
                &[&self.state, &self.state, &scale, &self.hidden],
                h,
            )?;
            if trace_stages {
                let layer_slot =
                    runtime.upload_u32(&[u32::try_from(gemma4_trace_layer_slot(layer, 12))?])?;
                command.dispatch_1d(
                    "first_nonfinite_f32",
                    &[&self.state, &self.validity, &self.hidden, &layer_slot],
                    1,
                )?;
                command.dispatch_1d(
                    "max_abs_f32",
                    &[&self.state, &self.stage_max_abs, &self.hidden, &layer_slot],
                    1,
                )?;
            }
        }
        if !select_output {
            return Ok(());
        }
        let output_norm = self.weight("output_norm.weight", GgufTensorType::F32)?;
        command.dispatch_threadgroups_1d(
            "rms_norm_decode_f32",
            &[
                &self.state,
                &output_norm,
                &self.norm,
                &self.hidden,
                &self.epsilon,
            ],
            1,
            32,
        )?;
        self.matvec(
            &mut command,
            &self.norm,
            &token_embd,
            &self.logits,
            &self.hidden,
            &self.vocab,
            c.vocab_size,
            GgufTensorType::Q6K,
        )?;
        command.dispatch_1d(
            "softcap_f32",
            &[&self.logits, &self.final_softcap, &self.vocab],
            c.vocab_size,
        )?;
        command.dispatch_threadgroups_1d(
            "argmax_f32",
            &[&self.logits, &self.selected, &self.vocab],
            1,
            256,
        )?;
        let trace_dispatches = if trace_sync {
            command
                .take_kernel_timings()
                .into_iter()
                .rev()
                .take(12)
                .map(|timing| {
                    format!(
                        "{}:{:.3}ms",
                        timing.kernel,
                        timing.timing.wall_time.as_secs_f64() * 1_000.0
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        } else {
            String::new()
        };
        if trace_stages {
            let marker = runtime.read_u32(&self.validity)?;
            if marker != u32::MAX {
                let slot = marker >> 16;
                let index = marker & 0xffff;
                let total_trace_stages = GEMMA4_TRACE_GLOBAL_STAGES
                    + self.model.config.layers * GEMMA4_TRACE_STAGES_PER_LAYER;
                let ranges = runtime.read_f32(&self.stage_max_abs, total_trace_stages)?;
                if usize::try_from(slot)? < GEMMA4_TRACE_GLOBAL_STAGES {
                    let stage = match slot {
                        0 => "input_embedding",
                        1 => "ple_lookup_raw",
                        2 => "ple_lookup_scaled",
                        3 => "ple_projection_scaled",
                        4 => "ple_projection_rms",
                        _ => "ple_combined",
                    };
                    let ple_layer =
                        usize::try_from(index)? / self.model.config.per_layer_embedding_size;
                    anyhow::bail!(
                        "Gemma resident state became non-finite at {stage} index {index} (PLE layer {ple_layer}); max_abs input_embedding={} ple_lookup_raw={} ple_lookup_scaled={} ple_projection_scaled={} ple_projection_rms={} ple_combined={}; sync_dispatches=[{}]",
                        ranges[0],
                        ranges[1],
                        ranges[2],
                        ranges[3],
                        ranges[4],
                        ranges[5],
                        trace_dispatches,
                    );
                }
                let stages = u32::try_from(GEMMA4_TRACE_STAGES_PER_LAYER)?;
                let layer_slot = slot - u32::try_from(GEMMA4_TRACE_GLOBAL_STAGES)?;
                let layer = layer_slot / stages;
                let stage = match usize::try_from(layer_slot)? % GEMMA4_TRACE_STAGES_PER_LAYER {
                    0 => "input_embedding",
                    1 => "attention_projection",
                    2 => "post_attention",
                    3 => "ffn_gate",
                    4 => "ffn_gate_gelu",
                    5 => "ffn_product",
                    6 => "ffn_down_norm",
                    7 => "post_mlp",
                    8 => "ple_gate",
                    9 => "ple_gate_gelu",
                    10 => "ple_product",
                    11 => "ple_projection_norm",
                    _ => "post_ple",
                };
                let base = GEMMA4_TRACE_GLOBAL_STAGES
                    + usize::try_from(layer).expect("trace slot fits usize")
                        * GEMMA4_TRACE_STAGES_PER_LAYER;
                anyhow::bail!(
                    "Gemma resident state became non-finite at layer {layer} {stage} hidden index {index}; max_abs input_embedding={} attention_projection={} post_attention={} ffn_gate={} ffn_gate_gelu={} ffn_product={} ffn_down_norm={} post_mlp={} ple_gate={} ple_gate_gelu={} ple_product={} ple_projection_norm={} post_ple={}; sync_dispatches=[{}]",
                    ranges[base],
                    ranges[base + 1],
                    ranges[base + 2],
                    ranges[base + 3],
                    ranges[base + 4],
                    ranges[base + 5],
                    ranges[base + 6],
                    ranges[base + 7],
                    ranges[base + 8],
                    ranges[base + 9],
                    ranges[base + 10],
                    ranges[base + 11],
                    ranges[base + 12],
                    trace_dispatches,
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Gemma4KvCacheType, Gemma4PrefillPlan, gemma4_attention_key_count,
        gemma4_attention_key_count_table, gemma4_kernel_family, gemma4_rope_angle,
        gemma4_should_finish,
    };

    #[test]
    fn prefill_plan_batches_short_prompts_and_bounds_long_prompts() {
        let short = Gemma4PrefillPlan::new(10, 4096).unwrap();
        assert_eq!((short.chunk_size, short.chunks), (10, 1));
        let long = Gemma4PrefillPlan::new(300, 4096).unwrap();
        assert_eq!((long.chunk_size, long.chunks), (128, 3));
        assert!(Gemma4PrefillPlan::new(0, 4096).is_err());
        assert!(Gemma4PrefillPlan::new(4097, 4096).is_err());
    }

    #[test]
    fn prefill_chunk_boundaries_preserve_absolute_positions() {
        let plan = Gemma4PrefillPlan::new(300, 4096).unwrap();
        assert_eq!(plan.chunks, 3);
        let chunks = (0..plan.prompt_tokens)
            .collect::<Vec<_>>()
            .chunks(plan.chunk_size)
            .map(|chunk| (chunk[0], *chunk.last().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(chunks, vec![(0, 127), (128, 255), (256, 299)]);
    }

    #[test]
    fn full_and_sliding_attention_expose_only_causal_keys() {
        assert_eq!(gemma4_attention_key_count(0, false, 4), 1);
        assert_eq!(gemma4_attention_key_count(127, false, 4), 128);
        assert_eq!(gemma4_attention_key_count(0, true, 4), 1);
        assert_eq!(gemma4_attention_key_count(2, true, 4), 3);
        assert_eq!(gemma4_attention_key_count(3, true, 4), 4);
        assert_eq!(gemma4_attention_key_count(127, true, 4), 4);
    }

    #[test]
    fn attention_control_table_keeps_each_token_and_layer_immutable() {
        let controls = gemma4_attention_key_count_table(126, 3, &[false, true, false], 128)
            .expect("build controls");
        assert_eq!(controls, vec![127, 127, 127, 128, 128, 128, 129, 128, 129]);
    }

    #[test]
    fn kv_cache_types_are_explicit_and_block_aligned() {
        assert_eq!(
            Gemma4KvCacheType::parse("f32").unwrap(),
            Gemma4KvCacheType::F32
        );
        assert_eq!(
            Gemma4KvCacheType::parse("q8_0").unwrap(),
            Gemma4KvCacheType::Q8_0
        );
        assert_eq!(
            Gemma4KvCacheType::parse("q4_0").unwrap(),
            Gemma4KvCacheType::Q4_0
        );
        assert!(Gemma4KvCacheType::parse("q5_1").is_err());
        let f32_bytes = Gemma4KvCacheType::F32.cache_bytes(128, 256).unwrap();
        let q8_bytes = Gemma4KvCacheType::Q8_0.cache_bytes(128, 256).unwrap();
        let q4_bytes = Gemma4KvCacheType::Q4_0.cache_bytes(128, 256).unwrap();
        assert_eq!(q8_bytes * 4, f32_bytes * 34 / 32);
        assert!(q4_bytes < q8_bytes && q8_bytes < f32_bytes);
        assert!(Gemma4KvCacheType::Q8_0.cache_bytes(128, 255).is_err());
    }

    #[test]
    fn gemma4_rope_honors_proportional_factors_and_partial_rotary_width() {
        let normal = gemma4_rope_angle(8, 0, 256, 1_000_000.0, 1.0);
        let suppressed = gemma4_rope_angle(8, 64, 256, 1_000_000.0, 1.0e30);
        let outside_partial_width = gemma4_rope_angle(8, 64, 128, 10_000.0, 1.0);
        assert_eq!(normal, 8.0);
        assert!(suppressed.abs() < 1.0e-30);
        assert_eq!(outside_partial_width, 0.0);
    }

    #[test]
    fn decode_profile_groups_gemma_kernel_families_without_hiding_other_work() {
        assert_eq!(
            gemma4_kernel_family("matvec_q4_0_16row"),
            "q4_q6_projections"
        );
        assert_eq!(
            gemma4_kernel_family("matvec_q6_k_8row"),
            "q4_q6_projections"
        );
        assert_eq!(
            gemma4_kernel_family("matmul_q4_0_batch_16row"),
            "q4_q6_projections"
        );
        assert_eq!(gemma4_kernel_family("rms_norm_decode_f32"), "rms_rope");
        assert_eq!(
            gemma4_kernel_family("attention_decode_fused_f32"),
            "gemma_attention"
        );
        assert_eq!(gemma4_kernel_family("argmax_f32"), "argmax");
        assert_eq!(gemma4_kernel_family("embedding_lookup_q6_k"), "other");
    }

    #[test]
    fn chat_stops_on_end_turn_while_raw_generation_does_not() {
        assert!(gemma4_should_finish(106, 1, "answer<turn|>", true));
        assert!(!gemma4_should_finish(106, 1, "answer<turn|>", false));
        assert!(gemma4_should_finish(1, 1, "answer", false));
    }
}
