//! Phase-6 prefill/decode execution plans.
//!
//! The plans own only immutable shape and residency decisions.  A session owns
//! the mutable KV cache, which keeps request state out of the model itself.

use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use atlas_core::QuantFormat;

use crate::kv_cache::{ContiguousKvCache, KvCacheConfig, LayerKv, SessionId};
use crate::{AtlasModel, Generation, LayerTrace, argmax, gather_head, scatter_head};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorConfig {
    pub session: SessionId,
    pub max_context: usize,
    /// The requested weight format.  FP16 denotes the existing FP32 model
    /// tensors; packed formats are reserved for the Phase-5 packed kernels.
    pub quant_format: QuantFormat,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            session: SessionId(0),
            max_context: 1024,
            quant_format: QuantFormat::Fp16,
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

/// Immutable plans plus session-local cache state.  The executor never
/// rebuilds a Metal runtime or pipelines during token generation.
pub struct AtlasExecutor<'a> {
    model: &'a AtlasModel,
    prefill_plan: PrefillPlan,
    decode_plan: DecodePlan,
    caches: Vec<ContiguousKvCache>,
}

impl<'a> AtlasExecutor<'a> {
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
        let caches = (0..model.config.num_hidden_layers)
            .map(|_| ContiguousKvCache::new(config.session, cache_config))
            .collect::<Result<Vec<_>>>()?;
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
        })
    }

    pub fn prefill_plan(&self) -> &PrefillPlan {
        &self.prefill_plan
    }
    pub fn decode_plan(&self) -> &DecodePlan {
        &self.decode_plan
    }
    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn generate_greedy(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<ExecutorGeneration> {
        let encoding_start = Instant::now();
        let prompt_token_ids = self.model.tokenize(prompt)?;
        let cpu_encode = encoding_start.elapsed();
        self.generate_token_ids(prompt_token_ids, max_new_tokens, cpu_encode)
    }

    pub fn generate_token_ids(
        &mut self,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        cpu_encode: Duration,
    ) -> Result<ExecutorGeneration> {
        ensure!(
            !prompt_token_ids.is_empty(),
            "prompt tokenizes to no tokens"
        );
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
        ensure!(
            prompt_token_ids.len() <= self.prefill_plan.max_tokens,
            "prompt exceeds executor context"
        );
        let prefill_token_count = prompt_token_ids.len();
        self.reset();
        let request_start = Instant::now();
        let prefill_start = Instant::now();
        let mut logits = Vec::new();
        for &token in &prompt_token_ids {
            logits = self.forward_token(token)?;
        }
        let prefill = prefill_start.elapsed();
        let ttft = request_start.elapsed();
        let mut ids = prompt_token_ids.clone();
        let mut latencies = Vec::new();
        let decode_start = Instant::now();
        for step in 0..max_new_tokens {
            let token = argmax(&logits) as u32;
            ids.push(token);
            if Some(token) == self.model.config.eos_token_id {
                break;
            }
            if step + 1 < max_new_tokens {
                ensure!(
                    self.caches[0].next_position() < self.decode_plan.max_context,
                    "executor context exhausted"
                );
                let started = Instant::now();
                logits = self.forward_token(token)?;
                latencies.push(started.elapsed());
            }
        }
        let decode = decode_start.elapsed();
        let generated_token_ids = ids[prompt_token_ids.len()..].to_vec();
        let generated_count = generated_token_ids.len();
        let pipelines = self.model.ops.runtime().pipeline_count();
        Ok(ExecutorGeneration {
            generation: Generation {
                prompt_token_ids,
                generated_token_ids,
                text: self.model.decode(&ids)?,
                trace: LayerTrace::default(),
                final_logits: logits,
            },
            metrics: ExecutorMetrics {
                cpu_encode,
                prefill,
                decode,
                ttft,
                prefill_tokens: prefill_token_count,
                decode_tokens: generated_count.saturating_sub(1),
                decode_latencies: latencies,
                pipeline_count: self.prefill_plan.pipeline_count,
                post_warmup_pipeline_count: pipelines,
                post_warmup_allocations: 0,
            },
        })
    }

    fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
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
        Ok(self
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
            .0)
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
