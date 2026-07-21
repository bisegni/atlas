//! Resident-only Gemma 4 E2B greedy execution.
//!
//! Gemma cannot share the Llama executor: its PLE state, one-head shared KV
//! cache, mixed full/sliding attention, and final Q6_K tied projection are
//! architectural state, not optional Llama features.

use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use atlas_core::GgufTensorType;
use atlas_metal::GpuBuffer;

use crate::{Gemma4E2bModel, Generation, LayerTrace, gemma4_shared_kv_sources};

const GEMMA4_TRACE_STAGES_PER_LAYER: usize = 13;
const GEMMA4_TRACE_GLOBAL_STAGES: usize = 6;

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
}

#[derive(Debug, Clone)]
pub struct Gemma4Generation {
    pub generation: Generation,
    pub metrics: Gemma4Metrics,
    pub finish_reason: Gemma4FinishReason,
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
    ple_width: GpuBuffer,
    head_full: GpuBuffer,
    head_swa: GpuBuffer,
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
    rope_cos: GpuBuffer,
    rope_sin: GpuBuffer,
    rope_cos_host: Vec<f32>,
    rope_sin_host: Vec<f32>,
    rope_freq_factors: Vec<f32>,
    pending_weight_upload_bytes: u64,
}

impl<'a> Gemma4E2bExecutor<'a> {
    pub fn new(model: &'a Gemma4E2bModel, max_context: usize) -> Result<Self> {
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
        let allocate = |count: usize| {
            runtime
                .allocate(
                    count
                        .checked_mul(4)
                        .context("Gemma resident arena size overflow")?,
                )
                .map_err(Into::into)
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
                        allocate(2 * max_context * source_head)
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
            ple_width: runtime.upload_u32(&[u32::try_from(c.per_layer_embedding_size)?])?,
            head_full: runtime.upload_u32(&[u32::try_from(c.key_length)?])?,
            head_swa: runtime.upload_u32(&[u32::try_from(c.key_length_swa)?])?,
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
            rope_cos: allocate(head / 2)?,
            rope_sin: allocate(head / 2)?,
            rope_cos_host: vec![0.0; head / 2],
            rope_sin_host: vec![0.0; head / 2],
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

    fn weight(&self, name: &str, expected: GgufTensorType) -> Result<GpuBuffer> {
        ensure!(
            self.model.resident_weight_format(name)? == expected,
            "Gemma tensor `{name}` has an unsupported resident format"
        );
        self.model.resident_weight(name)
    }

    fn write_rope(&mut self, sliding: bool) -> Result<()> {
        let c = &self.model.config;
        let theta = if sliding {
            c.rope_theta_swa
        } else {
            c.rope_theta
        };
        let head = if sliding {
            c.key_length_swa
        } else {
            c.key_length
        };
        let rotary = if sliding {
            c.rope_dimensions_swa
        } else {
            c.rope_dimensions
        };
        ensure!(
            rotary > 0 && rotary <= head && rotary.is_multiple_of(2),
            "Gemma 4 rotary width {rotary} is invalid for head width {head}"
        );
        for pair in 0..head / 2 {
            let factor = if sliding {
                1.0
            } else {
                self.rope_freq_factors[pair]
            };
            let angle = gemma4_rope_angle(self.position, pair, rotary, theta, factor);
            self.rope_cos_host[pair] = angle.cos();
            self.rope_sin_host[pair] = angle.sin();
        }
        self.model
            .runtime()
            .write_f32(&self.rope_cos, &self.rope_cos_host)?;
        self.model
            .runtime()
            .write_f32(&self.rope_sin, &self.rope_sin_host)?;
        Ok(())
    }

    fn matvec(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width: usize,
        format: GgufTensorType,
    ) -> Result<()> {
        let kernel = match format {
            GgufTensorType::Q4_0 => "matvec_q4_0_blocked",
            GgufTensorType::Q6K => "matvec_q6_k",
            GgufTensorType::F16 => "matvec_f16",
            other => anyhow::bail!("unsupported Gemma matvec format {other:?}"),
        };
        let output_width_buffer = self
            .model
            .runtime()
            .upload_u32(&[u32::try_from(output_width)?])?;
        let buffers = &[input, weight, output, input_width, &output_width_buffer];
        if format == GgufTensorType::Q4_0 {
            // matvec_q4_0_blocked assigns one SIMD group to each output row.
            // A flat dispatch makes `lane` exceed 31 and leaves most rows
            // unwritten, producing uninitialized NaNs in Gemma projections.
            command.dispatch_threadgroups_1d(kernel, buffers, output_width, 32)?;
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

    pub fn generate_greedy_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        self.generate_greedy_stream_inner(prompt, max_new_tokens, cancelled, false, emit)
    }

    pub fn generate_greedy_chat_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        self.generate_greedy_stream_inner(prompt, max_new_tokens, cancelled, true, emit)
    }

    fn generate_greedy_stream_inner(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        stop_on_end_turn: bool,
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
        let mut selected = 0;
        for token in &prompt_ids {
            selected = self.forward_token(*token)?;
        }
        let prefill = prefill_started.elapsed();
        let prefill_commands = runtime.command_buffer_count() - command_before;
        let decode_started = Instant::now();
        let mut generated = Vec::new();
        let mut finish_reason = Gemma4FinishReason::MaxTokens;
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
            if gemma4_should_finish(
                selected,
                self.model.config.eos_token_id,
                &decoded,
                stop_on_end_turn,
            ) {
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
            },
            finish_reason,
        })
    }

    fn forward_token(&mut self, token: u32) -> Result<u32> {
        ensure!(
            self.position < self.max_context,
            "Gemma executor context exhausted"
        );
        let c = &self.model.config;
        let runtime = self.model.runtime();
        let h = c.hidden_size;
        let ple_total = c.layers * c.per_layer_embedding_size;
        runtime.write_u32(&self.token, &[token])?;
        runtime.write_u32(&self.position_buffer, &[u32::try_from(self.position)?])?;
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
        let mut command = runtime.begin_resident_command_with_exact_timing(trace_sync)?;
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
        let ple_total_buffer = runtime.upload_u32(&[u32::try_from(ple_total)?])?;
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
            self.write_rope(sliding)?;
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
                q_width,
                GgufTensorType::Q4_0,
            )?;
            let q_width_buffer = runtime.upload_u32(&[u32::try_from(q_width)?])?;
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
            command.dispatch_1d(
                "rope_f32",
                &[
                    &self.rope_input,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.rope_output,
                    head_width,
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
                    head,
                    GgufTensorType::Q4_0,
                )?;
                self.matvec(
                    &mut command,
                    &self.norm,
                    &wv,
                    &self.v,
                    &self.hidden,
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
                command.dispatch_1d(
                    "rope_f32",
                    &[
                        &self.rope_input,
                        &self.rope_cos,
                        &self.rope_sin,
                        &self.rope_output,
                        head_width,
                    ],
                    head / 2,
                )?;
                command.dispatch_1d(
                    "rope_interleaved_to_half_f32",
                    &[&self.rope_output, &self.k_rot, &self.one, head_width],
                    head / 2,
                )?;
                let cache = self.kv[layer].as_ref().expect("KV provider has cache");
                command.dispatch_1d(
                    "kv_append_decode_f32",
                    &[
                        &self.k_rot,
                        &self.v,
                        cache,
                        head_width,
                        &self.capacity,
                        &self.position_buffer,
                    ],
                    head,
                )?;
            }
            // Gemma's cache source is explicit and observable through kv_sources; the existing resident attention kernel remains valid for one KV head.
            let cache = self.kv[source].as_ref().expect("Gemma KV source has cache");
            let key_count = runtime.upload_u32(&[u32::try_from(if sliding {
                self.position.min(c.sliding_window.saturating_sub(1)) + 1
            } else {
                self.position + 1
            })?])?;
            command.dispatch_threadgroups_1d(
                "attention_decode_fused_gemma4_f32",
                &[
                    &self.q_rot,
                    cache,
                    &self.attention,
                    &self.heads,
                    &self.kv_heads,
                    head_width,
                    &self.capacity,
                    &key_count,
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
                &q_width_buffer,
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
            let ffn_buffer = runtime.upload_u32(&[u32::try_from(ffn)?])?;
            let gate = self.weight(&format!("{p}.ffn_gate.weight"), GgufTensorType::Q4_0)?;
            let up = self.weight(&format!("{p}.ffn_up.weight"), GgufTensorType::Q4_0)?;
            let down = self.weight(&format!("{p}.ffn_down.weight"), GgufTensorType::Q4_0)?;
            self.matvec(
                &mut command,
                &self.norm,
                &gate,
                &self.gate,
                &self.hidden,
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
                &ffn_buffer,
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
            let ple_offset =
                runtime.upload_u32(&[u32::try_from(layer * c.per_layer_embedding_size)?])?;
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
        command.finish()?;
        self.position += 1;
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
        Ok(runtime.read_u32(&self.selected)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{gemma4_rope_angle, gemma4_should_finish};

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
    fn chat_stops_on_end_turn_while_raw_generation_does_not() {
        assert!(gemma4_should_finish(106, 1, "answer<turn|>", true));
        assert!(!gemma4_should_finish(106, 1, "answer<turn|>", false));
        assert!(gemma4_should_finish(1, 1, "answer", false));
    }
}
