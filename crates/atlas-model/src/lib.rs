//! Llama-compatible, correctness-first transformer execution for Atlas Phase 3.
//!
//! This module deliberately recomputes the complete prompt for each greedy
//! token.  The Phase-4 cache types live in [`kv_cache`]; executor integration
//! is deliberately deferred to Phase 6, where prefill and decode plans are
//! introduced together.

pub mod executor;
pub mod kv_cache;
pub mod runtime;
pub mod sampling;

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{GgufModel, GgufTensorType, f16_bits_to_f32, read_safetensors_tensor_f32};
use atlas_metal::GpuBuffer;
use atlas_ops::{ExecutionMode, NeuralOps};
use serde_json::Value;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let value: Value = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("read {}", path.display()))?,
        )
        .context("parse model config")?;
        let architecture = value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("");
        ensure!(
            architecture == "llama",
            "only Llama-compatible `model_type: llama` is supported, got `{architecture}`"
        );
        let required = |key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|v| usize::try_from(v).ok())
                .with_context(|| format!("config is missing positive integer `{key}`"))
        };
        let hidden_size = required("hidden_size")?;
        let num_attention_heads = required("num_attention_heads")?;
        let num_key_value_heads = value
            .get("num_key_value_heads")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(num_attention_heads);
        ensure!(hidden_size > 0 && num_attention_heads > 0 && num_key_value_heads > 0);
        ensure!(
            hidden_size % num_attention_heads == 0,
            "hidden_size must divide num_attention_heads"
        );
        ensure!(
            num_attention_heads % num_key_value_heads == 0,
            "attention heads must divide key/value heads"
        );
        Ok(Self {
            vocab_size: required("vocab_size")?,
            hidden_size,
            intermediate_size: required("intermediate_size")?,
            num_hidden_layers: required("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps: value
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5) as f32,
            rope_theta: value
                .get("rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(10_000.0) as f32,
            bos_token_id: value
                .get("bos_token_id")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok()),
            eos_token_id: value
                .get("eos_token_id")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok()),
            tie_word_embeddings: value
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub name: String,
    pub max_abs: f32,
    pub len: usize,
}
#[derive(Debug, Clone, Default)]
pub struct LayerTrace {
    pub entries: Vec<TraceEntry>,
}
impl LayerTrace {
    fn record(&mut self, name: impl Into<String>, values: &[f32]) {
        self.entries.push(TraceEntry {
            name: name.into(),
            max_abs: values.iter().map(|v| v.abs()).fold(0.0, f32::max),
            len: values.len(),
        });
    }
}

pub struct AtlasModel {
    root: PathBuf,
    pub config: ModelConfig,
    tokenizer: Tokenizer,
    weights: HashMap<String, WeightSource>,
    weight_cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    // Immutable buffers are deliberately owned by the model, rather than an
    // individual request.  Session executors only own mutable KV/activation
    // state and therefore cannot trigger a model-weight re-upload per token.
    resident_weights: Mutex<ResidentWeights>,
    ops: NeuralOps,
}

#[derive(Default)]
struct ResidentWeights {
    buffers: HashMap<String, GpuBuffer>,
    formats: HashMap<String, Option<GgufTensorType>>,
    uploaded_bytes: u64,
}

#[derive(Clone)]
enum WeightSource {
    SafeTensor(PathBuf),
    GgufF32(Vec<u8>),
    GgufPacked {
        bytes: Vec<u8>,
        format: GgufTensorType,
    },
}

impl AtlasModel {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let config = ModelConfig::from_path(root.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("load tokenizer.json: {error}"))?;
        let weights = if root.join("model.gguf").is_file() {
            gguf_weight_map(&root)?
        } else {
            weight_map(&root)?
        };
        let ops = NeuralOps::new().context("initialize Metal execution")?;
        Ok(Self {
            root,
            config,
            tokenizer,
            weights,
            weight_cache: Mutex::new(HashMap::new()),
            resident_weights: Mutex::new(ResidentWeights::default()),
            ops,
        })
    }
    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            // Hugging Face's Llama tokenizer uses its post-processor to add
            // the configured BOS token, so special tokens must stay enabled.
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|e| anyhow::anyhow!("decode tokens: {e}"))
    }
    pub fn generate_greedy(&self, prompt: &str, max_new_tokens: usize) -> Result<Generation> {
        ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
        let mut ids = self.tokenize(prompt)?;
        ensure!(!ids.is_empty(), "prompt tokenizes to no tokens");
        let prompt_token_ids = ids.clone();
        let mut trace = LayerTrace::default();
        let mut final_logits = Vec::new();
        for step in 0..max_new_tokens {
            eprintln!(
                "atlas: generating token {}/{} (full prompt recomputation; KV cache begins in Phase 4)",
                step + 1,
                max_new_tokens
            );
            let logits = self.forward(&ids, &mut trace, self.config.num_hidden_layers)?;
            let token = argmax(&logits) as u32;
            final_logits = logits;
            ids.push(token);
            if Some(token) == self.config.eos_token_id {
                break;
            }
        }
        let generated_token_ids = ids[prompt_token_ids.len()..].to_vec();
        Ok(Generation {
            prompt_token_ids,
            generated_token_ids,
            text: self.decode(&ids)?,
            trace,
            final_logits,
        })
    }
    /// Executes the requested prefix of layers; `layers=1` is the larger-model gate.
    pub fn forward(
        &self,
        token_ids: &[u32],
        trace: &mut LayerTrace,
        layers: usize,
    ) -> Result<Vec<f32>> {
        ensure!(!token_ids.is_empty(), "forward needs at least one token");
        ensure!(
            layers <= self.config.num_hidden_layers,
            "requested layers exceeds model config"
        );
        let sequence = token_ids.len();
        let hidden = self.config.hidden_size;
        let embedding = self.weight("model.embed_tokens.weight")?;
        let mut state = self
            .ops
            .embedding(&embedding, self.config.vocab_size, hidden, token_ids)?
            .0;
        trace.record("embeddings", &state);
        for layer in 0..layers {
            state = self.layer(layer, &state, sequence, trace)?;
        }
        // A partial forward is intentionally useful for the larger fixture validation.
        if layers != self.config.num_hidden_layers {
            return Ok(state);
        }
        let norm = self.weight("model.norm.weight")?;
        state = self
            .ops
            .rms_norm(&state, sequence, hidden, &norm, self.config.rms_norm_eps)?
            .0;
        let lm_head = if self.config.tie_word_embeddings {
            embedding
        } else {
            self.weight("lm_head.weight")?
        };
        let logits = self
            .ops
            .project(
                ExecutionMode::Prefill,
                &state[(sequence - 1) * hidden..],
                &lm_head,
                1,
                hidden,
                self.config.vocab_size,
            )?
            .0;
        trace.record("final_logits", &logits);
        Ok(logits)
    }
    fn layer(
        &self,
        layer: usize,
        state: &[f32],
        sequence: usize,
        trace: &mut LayerTrace,
    ) -> Result<Vec<f32>> {
        let h = self.config.hidden_size;
        let hd = self.config.head_dim();
        let prefix = format!("model.layers.{layer}");
        let input_norm = self.weight(&format!("{prefix}.input_layernorm.weight"))?;
        let normalized = self
            .ops
            .rms_norm(state, sequence, h, &input_norm, self.config.rms_norm_eps)?
            .0;
        let q = self.project(
            sequence,
            &normalized,
            &format!("{prefix}.self_attn.q_proj.weight"),
            self.config.num_attention_heads * hd,
        )?;
        let k = self.project(
            sequence,
            &normalized,
            &format!("{prefix}.self_attn.k_proj.weight"),
            self.config.num_key_value_heads * hd,
        )?;
        let v = self.project(
            sequence,
            &normalized,
            &format!("{prefix}.self_attn.v_proj.weight"),
            self.config.num_key_value_heads * hd,
        )?;
        trace.record(format!("layer.{layer}.q"), &q);
        trace.record(format!("layer.{layer}.k"), &k);
        trace.record(format!("layer.{layer}.v"), &v);
        let q = self.rope(&q, sequence, self.config.num_attention_heads, hd)?;
        let k = self.rope(&k, sequence, self.config.num_key_value_heads, hd)?;
        let mut attention = vec![0.0; sequence * h];
        let group = self.config.num_attention_heads / self.config.num_key_value_heads;
        for head in 0..self.config.num_attention_heads {
            let kv_head = head / group;
            let queries = gather_head(&q, sequence, self.config.num_attention_heads, hd, head);
            let keys = gather_head(&k, sequence, self.config.num_key_value_heads, hd, kv_head);
            let values = gather_head(&v, sequence, self.config.num_key_value_heads, hd, kv_head);
            let scores = self
                .ops
                .attention_scores(
                    &queries,
                    &keys,
                    sequence,
                    sequence,
                    hd,
                    (hd as f32).sqrt().recip(),
                )?
                .0;
            let mask = causal_mask(sequence);
            let weights = self
                .ops
                .masked_softmax(&scores, &mask, sequence, sequence)?
                .0;
            let values = self
                .ops
                .attention_values(&weights, &values, sequence, sequence, hd)?
                .0;
            scatter_head(
                &mut attention,
                &values,
                sequence,
                self.config.num_attention_heads,
                hd,
                head,
            );
        }
        trace.record(format!("layer.{layer}.attention"), &attention);
        let attention_output = self.project(
            sequence,
            &attention,
            &format!("{prefix}.self_attn.o_proj.weight"),
            h,
        )?;
        let residual = self.ops.add(state, &attention_output)?.0;
        let post_norm = self.weight(&format!("{prefix}.post_attention_layernorm.weight"))?;
        let mlp_input = self
            .ops
            .rms_norm(&residual, sequence, h, &post_norm, self.config.rms_norm_eps)?
            .0;
        let gate = self.project(
            sequence,
            &mlp_input,
            &format!("{prefix}.mlp.gate_proj.weight"),
            self.config.intermediate_size,
        )?;
        let up = self.project(
            sequence,
            &mlp_input,
            &format!("{prefix}.mlp.up_proj.weight"),
            self.config.intermediate_size,
        )?;
        let activated = self.ops.silu(&gate)?.0;
        let mlp_product = self.ops.multiply(&activated, &up)?.0;
        let mlp = self.project_width(
            sequence,
            &mlp_product,
            &format!("{prefix}.mlp.down_proj.weight"),
            self.config.intermediate_size,
            h,
            ExecutionMode::Prefill,
        )?;
        trace.record(format!("layer.{layer}.mlp"), &mlp);
        Ok(self.ops.add(&residual, &mlp)?.0)
    }
    fn project(&self, rows: usize, input: &[f32], name: &str, output: usize) -> Result<Vec<f32>> {
        self.project_width(
            rows,
            input,
            name,
            self.config.hidden_size,
            output,
            ExecutionMode::Prefill,
        )
    }
    fn project_width(
        &self,
        rows: usize,
        input: &[f32],
        name: &str,
        input_width: usize,
        output: usize,
        mode: ExecutionMode,
    ) -> Result<Vec<f32>> {
        let weights = self.weight(name)?;
        Ok(self
            .ops
            .project(mode, input, &weights, rows, input_width, output)?
            .0)
    }
    fn rope_at(
        &self,
        input: &[f32],
        position: usize,
        heads: usize,
        dim: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            input.len() == heads * dim && dim.is_multiple_of(2),
            "invalid one-token RoPE shape"
        );
        let mut interleaved = vec![0.0; input.len()];
        let mut cos = vec![0.0; dim / 2];
        let mut sin = vec![0.0; dim / 2];
        for pair in 0..dim / 2 {
            let angle =
                position as f32 / self.config.rope_theta.powf((pair * 2) as f32 / dim as f32);
            cos[pair] = angle.cos();
            sin[pair] = angle.sin();
            for head in 0..heads {
                let base = head * dim;
                interleaved[base + pair * 2] = input[base + pair];
                interleaved[base + pair * 2 + 1] = input[base + pair + dim / 2];
            }
        }
        let rotated = self.ops.rope(&interleaved, heads, dim, &cos, &sin)?.0;
        let mut output = vec![0.0; input.len()];
        for head in 0..heads {
            for pair in 0..dim / 2 {
                let base = head * dim;
                output[base + pair] = rotated[base + pair * 2];
                output[base + pair + dim / 2] = rotated[base + pair * 2 + 1];
            }
        }
        Ok(output)
    }
    fn rope(&self, input: &[f32], sequence: usize, heads: usize, dim: usize) -> Result<Vec<f32>> {
        ensure!(dim % 2 == 0, "RoPE head dimension must be even");
        // Metal's Phase-2 RoPE primitive is interleaved; Llama rotates halves.
        let mut interleaved = vec![0.0; input.len()];
        let mut cos = vec![0.0; sequence * dim / 2];
        let mut sin = cos.clone();
        for position in 0..sequence {
            for pair in 0..dim / 2 {
                let angle =
                    position as f32 / self.config.rope_theta.powf((pair * 2) as f32 / dim as f32);
                cos[position * dim / 2 + pair] = angle.cos();
                sin[position * dim / 2 + pair] = angle.sin();
                for head in 0..heads {
                    let base = (position * heads + head) * dim;
                    let pair_base = base + pair * 2;
                    interleaved[pair_base] = input[base + pair];
                    interleaved[pair_base + 1] = input[base + pair + dim / 2];
                }
            }
        }
        let mut rotated = vec![0.0; input.len()];
        for position in 0..sequence {
            let start = position * heads * dim;
            let end = start + heads * dim;
            rotated[start..end].copy_from_slice(
                &self
                    .ops
                    .rope(
                        &interleaved[start..end],
                        heads,
                        dim,
                        &cos[position * dim / 2..(position + 1) * dim / 2],
                        &sin[position * dim / 2..(position + 1) * dim / 2],
                    )?
                    .0,
            );
        }
        let mut output = vec![0.0; input.len()];
        for position in 0..sequence {
            for head in 0..heads {
                for pair in 0..dim / 2 {
                    let base = (position * heads + head) * dim;
                    let pair_base = base + pair * 2;
                    output[base + pair] = rotated[pair_base];
                    output[base + pair + dim / 2] = rotated[pair_base + 1];
                }
            }
        }
        Ok(output)
    }
    fn weight(&self, name: &str) -> Result<Arc<Vec<f32>>> {
        if let Some(weight) = self
            .weight_cache
            .lock()
            .expect("weight cache lock")
            .get(name)
            .cloned()
        {
            return Ok(weight);
        }
        let source = self
            .weights
            .get(name)
            .with_context(|| format!("model lacks required tensor `{name}`"))?;
        let weight = Arc::new(match source {
            WeightSource::SafeTensor(path) => read_safetensors_tensor_f32(path, name)?,
            WeightSource::GgufF32(bytes) => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
                .collect(),
            WeightSource::GgufPacked { .. } => bail!(
                "packed GGUF tensor `{name}` cannot run through the reference executor; use Resident"
            ),
        });
        self.weight_cache
            .lock()
            .expect("weight cache lock")
            .insert(name.to_owned(), weight.clone());
        Ok(weight)
    }
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resident_tensor_count(&self) -> usize {
        self.weights.len()
    }

    /// Materialize every immutable parameter in a GPU-visible buffer once.
    /// Repeated calls are idempotent and return only bytes uploaded by this
    /// invocation, which makes executor warm-up telemetry unambiguous.
    pub(crate) fn ensure_resident_weights(&self) -> Result<u64> {
        let mut resident = self.resident_weights.lock().expect("resident weight lock");
        let mut uploaded = 0u64;
        for (name, source) in &self.weights {
            if resident.buffers.contains_key(name) {
                continue;
            }
            let (buffer, format) = match source {
                WeightSource::SafeTensor(_) | WeightSource::GgufF32(_) => {
                    let values = self.weight(name)?;
                    (self.ops.runtime().upload_f32(&values)?, None)
                }
                WeightSource::GgufPacked { bytes, format } => {
                    (self.ops.runtime().upload_bytes(bytes)?, Some(*format))
                }
            };
            uploaded = uploaded.saturating_add(buffer.bytes() as u64);
            resident.buffers.insert(name.clone(), buffer);
            resident.formats.insert(name.clone(), format);
        }
        resident.uploaded_bytes = resident.uploaded_bytes.saturating_add(uploaded);
        Ok(uploaded)
    }

    pub(crate) fn resident_weights_snapshot(&self) -> HashMap<String, GpuBuffer> {
        self.resident_weights
            .lock()
            .expect("resident weight lock")
            .buffers
            .clone()
    }

    pub(crate) fn resident_weight_format(&self, name: &str) -> Option<GgufTensorType> {
        self.resident_weights
            .lock()
            .expect("resident weight lock")
            .formats
            .get(name)
            .copied()
            .flatten()
    }

    pub(crate) fn is_gguf(&self) -> bool {
        self.weights
            .values()
            .any(|source| !matches!(source, WeightSource::SafeTensor(_)))
    }

    pub fn format_name(&self) -> &'static str {
        if self.is_gguf() {
            "gguf-packed"
        } else {
            "safetensors-fp32"
        }
    }
}

#[derive(Debug, Clone)]
pub struct Generation {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub trace: LayerTrace,
    pub final_logits: Vec<f32>,
}

/// Compare a generation to the pinned raw-token Phase 3 oracle JSON.
pub fn validate_generation_golden(path: impl AsRef<Path>, generation: &Generation) -> Result<()> {
    let path = path.as_ref();
    let value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read golden {}", path.display()))?,
    )?;
    let ids = value
        .get("generated_token_ids")
        .and_then(Value::as_array)
        .context("golden is missing generated_token_ids")?
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .context("golden token ID is invalid")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        ids == generation.generated_token_ids,
        "golden token sequence differs: expected {ids:?}, got {:?}",
        generation.generated_token_ids
    );
    if let Some(expected) = value.get("final_logits").and_then(Value::as_array) {
        let tolerance = value
            .get("logit_abs_tolerance")
            .and_then(Value::as_f64)
            .unwrap_or(1e-4) as f32;
        ensure!(
            expected.len() == generation.final_logits.len(),
            "golden logits length differs"
        );
        for (index, (expected, actual)) in expected.iter().zip(&generation.final_logits).enumerate()
        {
            let expected = expected.as_f64().context("golden logit is invalid")? as f32;
            ensure!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "golden logit drift at {index}: expected {expected}, got {actual}, tolerance {tolerance}"
            );
        }
    }
    Ok(())
}
fn weight_map(root: &Path) -> Result<HashMap<String, WeightSource>> {
    let index = root.join("model.safetensors.index.json");
    if index.exists() {
        let v: Value = serde_json::from_slice(&fs::read(&index)?)?;
        return v["weight_map"]
            .as_object()
            .context("SafeTensors index missing weight_map")?
            .iter()
            .map(|(name, shard)| {
                Ok((
                    name.clone(),
                    WeightSource::SafeTensor(
                        root.join(shard.as_str().context("weight_map shard is not a string")?),
                    ),
                ))
            })
            .collect();
    }
    let file = root.join("model.safetensors");
    ensure!(
        file.exists(),
        "no SafeTensors model found in {}",
        root.display()
    );
    let descriptors = atlas_core::read_safetensors_descriptors(&file)?;
    Ok(descriptors
        .into_iter()
        .map(|d| (d.name, WeightSource::SafeTensor(file.clone())))
        .collect())
}

fn gguf_name_to_atlas(name: &str) -> Result<String> {
    if name == "token_embd.weight" {
        return Ok("model.embed_tokens.weight".into());
    }
    if name == "output_norm.weight" {
        return Ok("model.norm.weight".into());
    }
    if name == "output.weight" {
        return Ok("lm_head.weight".into());
    }
    let rest = name
        .strip_prefix("blk.")
        .context("unsupported GGUF tensor name")?;
    let (layer, tail) = rest.split_once('.').context("invalid GGUF block tensor")?;
    let tail = match tail {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => bail!("unsupported GGUF tensor `{name}`"),
    };
    Ok(format!("model.layers.{layer}.{tail}"))
}

fn gguf_weight_map(root: &Path) -> Result<HashMap<String, WeightSource>> {
    let model = GgufModel::open(root.join("model.gguf"))?;
    ensure!(
        model
            .metadata
            .get("general.architecture")
            .map(String::as_str)
            == Some("llama"),
        "GGUF architecture is not Llama"
    );
    let mut weights = HashMap::new();
    for tensor in &model.tensors {
        let name = gguf_name_to_atlas(&tensor.name)?;
        let bytes = model.tensor_data(tensor)?.to_vec();
        let source = match tensor.tensor_type {
            GgufTensorType::F32 => WeightSource::GgufF32(bytes),
            GgufTensorType::Q4_0 | GgufTensorType::Q8_0 | GgufTensorType::Q6K => {
                WeightSource::GgufPacked {
                    bytes,
                    format: tensor.tensor_type,
                }
            }
            GgufTensorType::F16 => WeightSource::GgufF32(
                bytes
                    .chunks_exact(2)
                    .flat_map(|chunk| {
                        f16_bits_to_f32(u16::from_le_bytes(chunk.try_into().unwrap())).to_le_bytes()
                    })
                    .collect(),
            ),
        };
        weights.insert(name, source);
    }
    Ok(weights)
}
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
fn causal_mask(sequence: usize) -> Vec<f32> {
    (0..sequence)
        .flat_map(|q| (0..sequence).map(move |k| if k > q { -1e9 } else { 0.0 }))
        .collect()
}
fn gather_head(values: &[f32], sequence: usize, heads: usize, dim: usize, head: usize) -> Vec<f32> {
    (0..sequence)
        .flat_map(|row| {
            values[(row * heads + head) * dim..(row * heads + head + 1) * dim]
                .iter()
                .copied()
        })
        .collect()
}
fn scatter_head(
    target: &mut [f32],
    values: &[f32],
    sequence: usize,
    heads: usize,
    dim: usize,
    head: usize,
) {
    for row in 0..sequence {
        target[(row * heads + head) * dim..(row * heads + head + 1) * dim]
            .copy_from_slice(&values[row * dim..(row + 1) * dim]);
    }
}
