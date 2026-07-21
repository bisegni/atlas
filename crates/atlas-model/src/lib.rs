//! Llama-compatible, correctness-first transformer execution for Atlas Phase 3.
//!
//! This module deliberately recomputes the complete prompt for each greedy
//! token.  The Phase-4 cache types live in [`kv_cache`]; executor integration
//! is deliberately deferred to Phase 6, where prefill and decode plans are
//! introduced together.

pub mod executor;
pub mod gemma4_executor;
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
use atlas_core::{
    GgufMetadataArray, GgufModel, GgufTensorType, f16_bits_to_f32, read_safetensors_tensor_f32,
};
use atlas_metal::GpuBuffer;
use atlas_ops::{ExecutionMode, NeuralOps};
use serde_json::Value;
use tokenizers::{
    Tokenizer,
    models::unigram::Unigram,
    pre_tokenizers::metaspace::{Metaspace, PrependScheme},
};

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

/// Construct Gemma's SentencePiece-compatible Unigram tokenizer directly
/// from the GGUF metadata bundled with official QAT artifacts.
pub fn gemma4_tokenizer(model: &GgufModel) -> Result<Tokenizer> {
    ensure!(
        model
            .metadata
            .get("general.architecture")
            .map(String::as_str)
            == Some("gemma4"),
        "embedded tokenizer is only available for Gemma 4 GGUF artifacts"
    );
    let tokenizer_model = model
        .metadata
        .get("tokenizer.ggml.model")
        .map(String::as_str)
        .unwrap_or("");
    ensure!(
        matches!(tokenizer_model, "llama" | "gemma4"),
        "unsupported Gemma 4 tokenizer model `{tokenizer_model}`"
    );
    let tokens = match model.metadata_arrays.get("tokenizer.ggml.tokens") {
        Some(GgufMetadataArray::Strings(tokens)) => tokens,
        _ => bail!("Gemma 4 GGUF is missing tokenizer.ggml.tokens"),
    };
    let scores = match model.metadata_arrays.get("tokenizer.ggml.scores") {
        Some(GgufMetadataArray::F32(scores)) => scores,
        _ => bail!("Gemma 4 GGUF is missing tokenizer.ggml.scores"),
    };
    ensure!(
        tokens.len() == scores.len() && !tokens.is_empty(),
        "Gemma 4 GGUF tokenizer tokens/scores are inconsistent"
    );
    let unknown = model
        .metadata
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(|value| value.parse::<usize>().ok());
    let unigram = Unigram::from(
        tokens
            .iter()
            .cloned()
            .zip(scores.iter().copied().map(f64::from))
            .collect(),
        unknown,
        true,
    )
    .map_err(|error| anyhow::anyhow!("build Gemma 4 Unigram tokenizer: {error}"))?;
    let mut tokenizer = Tokenizer::new(unigram);
    let metaspace = Metaspace::new('▁', PrependScheme::Always, true);
    tokenizer.with_pre_tokenizer(Some(metaspace.clone()));
    tokenizer.with_decoder(Some(metaspace));
    Ok(tokenizer)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4E2bConfig {
    pub vocab_size: usize,
    pub layers: usize,
    pub hidden_size: usize,
    pub feed_forward_sizes: Vec<usize>,
    pub attention_heads: usize,
    pub key_value_heads: Vec<usize>,
    pub rope_theta: f32,
    pub rope_theta_swa: f32,
    pub rms_norm_eps: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub key_length_swa: usize,
    pub value_length_swa: usize,
    pub rope_dimensions: usize,
    pub rope_dimensions_swa: usize,
    pub sliding_window: usize,
    pub sliding_pattern: Vec<bool>,
    pub shared_kv_layers: usize,
    pub per_layer_embedding_size: usize,
    pub final_logit_softcap: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

/// Combine Gemma 4's token-identity and context-aware per-layer embedding
/// channels. Keeping this scalar operation explicit gives both executor paths
/// the same PLE boundary and prevents a silent Llama-style omission.
pub fn gemma4_combine_ple(identity: &[f32], context: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        identity.len() == context.len(),
        "Gemma 4 PLE inputs have different lengths: {} and {}",
        identity.len(),
        context.len()
    );
    const INV_SQRT_2: f32 = 0.707_106_77;
    Ok(identity
        .iter()
        .zip(context)
        .map(|(identity, context)| (identity + context) * INV_SQRT_2)
        .collect())
}

/// Apply Gemma 4's final-logit soft cap without changing the argmax ordering
/// for finite values in the unsaturated range.
pub fn gemma4_softcap_logits(logits: &mut [f32], cap: f32) -> Result<()> {
    ensure!(
        cap.is_finite() && cap > 0.0,
        "Gemma 4 logit soft cap must be positive"
    );
    for logit in logits {
        *logit = cap * (*logit / cap).tanh();
    }
    Ok(())
}

/// Resolve the source layer for each Gemma 4 KV state. A layer that omits its
/// K/V tensors reuses the most recent non-shared layer of the same attention
/// kind; sharing across sliding and full attention is invalid.
pub fn gemma4_shared_kv_sources(
    sliding_layers: &[bool],
    kv_provider_layers: &[bool],
) -> Result<Vec<usize>> {
    ensure!(
        sliding_layers.len() == kv_provider_layers.len(),
        "Gemma 4 attention and KV-provider layouts have different lengths"
    );
    let mut sources = Vec::with_capacity(sliding_layers.len());
    for (layer, (&sliding, &provides_kv)) in
        sliding_layers.iter().zip(kv_provider_layers).enumerate()
    {
        if provides_kv {
            sources.push(layer);
            continue;
        }
        let source = (0..layer)
            .rev()
            .find(|&candidate| {
                kv_provider_layers[candidate] && sliding_layers[candidate] == sliding
            })
            .with_context(|| {
                format!(
                    "Gemma 4 layer {layer} shares KV but has no earlier {}-attention KV source",
                    if sliding { "sliding" } else { "full" }
                )
            })?;
        sources.push(source);
    }
    Ok(sources)
}

impl Gemma4E2bConfig {
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        ensure!(
            model
                .metadata
                .get("general.architecture")
                .map(String::as_str)
                == Some("gemma4"),
            "GGUF architecture is not Gemma 4"
        );
        let integer = |key: &str| -> Result<usize> {
            model
                .metadata
                .get(key)
                .with_context(|| format!("Gemma 4 GGUF is missing `{key}`"))?
                .parse::<usize>()
                .with_context(|| format!("Gemma 4 GGUF `{key}` is not an integer"))
        };
        let scalar = |key: &str| -> Result<f32> {
            model
                .metadata
                .get(key)
                .with_context(|| format!("Gemma 4 GGUF is missing `{key}`"))?
                .parse::<f32>()
                .with_context(|| format!("Gemma 4 GGUF `{key}` is not a scalar"))
        };
        let layers = integer("gemma4.block_count")?;
        let key_value_head_count = integer("gemma4.attention.head_count_kv")?;
        let key_value_heads = vec![key_value_head_count; layers];
        let feed_forward_sizes = match model.metadata_arrays.get("gemma4.feed_forward_length") {
            Some(GgufMetadataArray::I32(values)) => values
                .iter()
                .map(|value| {
                    usize::try_from(*value).context("Gemma 4 feed-forward length is negative")
                })
                .collect::<Result<Vec<_>>>()?,
            Some(GgufMetadataArray::U32(values)) => values
                .iter()
                .map(|value| {
                    usize::try_from(*value).context("Gemma 4 feed-forward length overflows")
                })
                .collect::<Result<Vec<_>>>()?,
            Some(GgufMetadataArray::U64(values)) => values
                .iter()
                .map(|value| {
                    usize::try_from(*value).context("Gemma 4 feed-forward length overflows")
                })
                .collect::<Result<Vec<_>>>()?,
            Some(GgufMetadataArray::I64(values)) => values
                .iter()
                .map(|value| {
                    usize::try_from(*value).context("Gemma 4 feed-forward length is negative")
                })
                .collect::<Result<Vec<_>>>()?,
            _ => bail!(
                "Gemma 4 GGUF is missing a supported gemma4.feed_forward_length array; retained arrays: {:?}; retained scalars: {:?}",
                model.metadata_arrays.keys().collect::<Vec<_>>(),
                model
                    .metadata
                    .keys()
                    .filter(|key| key.starts_with("gemma4."))
                    .collect::<Vec<_>>(),
            ),
        };
        let sliding_pattern = match model
            .metadata_arrays
            .get("gemma4.attention.sliding_window_pattern")
        {
            Some(GgufMetadataArray::Bool(values)) => values.clone(),
            _ => bail!("Gemma 4 GGUF is missing gemma4.attention.sliding_window_pattern"),
        };
        ensure!(
            feed_forward_sizes.len() == layers && sliding_pattern.len() == layers,
            "Gemma 4 per-layer metadata does not match block count"
        );
        let vocab_size = match model.metadata_arrays.get("tokenizer.ggml.tokens") {
            Some(GgufMetadataArray::Strings(values)) => values.len(),
            _ => bail!("Gemma 4 GGUF is missing tokenizer.ggml.tokens"),
        };
        Ok(Self {
            vocab_size,
            layers,
            hidden_size: integer("gemma4.embedding_length")?,
            feed_forward_sizes,
            attention_heads: integer("gemma4.attention.head_count")?,
            key_value_heads,
            rope_theta: scalar("gemma4.rope.freq_base")?,
            rope_theta_swa: scalar("gemma4.rope.freq_base_swa")?,
            rms_norm_eps: scalar("gemma4.attention.layer_norm_rms_epsilon")?,
            key_length: integer("gemma4.attention.key_length")?,
            value_length: integer("gemma4.attention.value_length")?,
            key_length_swa: integer("gemma4.attention.key_length_swa")?,
            value_length_swa: integer("gemma4.attention.value_length_swa")?,
            rope_dimensions: integer("gemma4.rope.dimension_count")?,
            rope_dimensions_swa: integer("gemma4.rope.dimension_count_swa")?,
            sliding_window: integer("gemma4.attention.sliding_window")?,
            sliding_pattern,
            shared_kv_layers: integer("gemma4.attention.shared_kv_layers")?,
            per_layer_embedding_size: integer("gemma4.embedding_length_per_layer_input")?,
            final_logit_softcap: scalar("gemma4.final_logit_softcapping")?,
            bos_token_id: u32::try_from(integer("tokenizer.ggml.bos_token_id")?)
                .context("Gemma 4 BOS token ID overflows")?,
            eos_token_id: u32::try_from(integer("tokenizer.ggml.eos_token_id")?)
                .context("Gemma 4 EOS token ID overflows")?,
        })
    }

    /// Validate the exact text-only E2B tensor contract before allocating
    /// resident state. This deliberately rejects another Gemma 4 size or a
    /// multimodal artifact instead of letting Llama's tensor map misinterpret
    /// it later in a user-facing generation command.
    pub fn validate_e2b_text_layout(&self, model: &GgufModel) -> Result<()> {
        ensure!(
            self.vocab_size == 262_144
                && self.layers == 35
                && self.hidden_size == 1_536
                && self.per_layer_embedding_size == 256,
            "Gemma 4 GGUF is not the supported E2B text-only layout (vocab={}, layers={}, hidden={}, per-layer embedding={})",
            self.vocab_size,
            self.layers,
            self.hidden_size,
            self.per_layer_embedding_size,
        );
        let tensor = |name: &str| {
            model
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .with_context(|| format!("Gemma 4 E2B GGUF is missing tensor `{name}`"))
        };
        let token_embeddings = tensor("token_embd.weight")?;
        ensure!(
            token_embeddings.tensor_type == GgufTensorType::Q6K
                && token_embeddings.dims == [self.hidden_size, self.vocab_size],
            "Gemma 4 E2B token_embd.weight must be Q6_K [{}, {}]",
            self.hidden_size,
            self.vocab_size,
        );
        let per_layer_embeddings = tensor("per_layer_token_embd.weight")?;
        ensure!(
            per_layer_embeddings.tensor_type == GgufTensorType::Q6K
                && per_layer_embeddings.dims.len() == 2
                && per_layer_embeddings.dims[1] == self.vocab_size,
            "Gemma 4 E2B per_layer_token_embd.weight must be a Q6_K vocabulary table"
        );
        for name in [
            "output_norm.weight",
            "per_layer_model_proj.weight",
            "per_layer_proj_norm.weight",
        ] {
            let _ = tensor(name)?;
        }
        let rope_freqs = tensor("rope_freqs.weight")?;
        ensure!(
            rope_freqs.tensor_type == GgufTensorType::F32
                && rope_freqs.dims == [self.key_length / 2],
            "Gemma 4 E2B rope_freqs.weight must be F32 [{}]",
            self.key_length / 2
        );
        for layer in 0..self.layers {
            for suffix in [
                "attn_norm.weight",
                "attn_output.weight",
                "attn_q.weight",
                "attn_q_norm.weight",
                "ffn_down.weight",
                "ffn_gate.weight",
                "ffn_norm.weight",
                "ffn_up.weight",
                "inp_gate.weight",
                "layer_output_scale.weight",
                "post_attention_norm.weight",
                "post_ffw_norm.weight",
                "post_norm.weight",
                "proj.weight",
            ] {
                let _ = tensor(&format!("blk.{layer}.{suffix}"))?;
            }
            let key = model
                .tensors
                .iter()
                .any(|tensor| tensor.name == format!("blk.{layer}.attn_k.weight"));
            let key_norm = model
                .tensors
                .iter()
                .any(|tensor| tensor.name == format!("blk.{layer}.attn_k_norm.weight"));
            let value = model
                .tensors
                .iter()
                .any(|tensor| tensor.name == format!("blk.{layer}.attn_v.weight"));
            ensure!(
                key == key_norm && key == value,
                "Gemma 4 E2B layer {layer} has an incomplete shared KV tensor group"
            );
        }
        Ok(())
    }
}

/// Direct GGUF loader for the declared Gemma 4 E2B text-only family.
///
/// Unlike [`AtlasModel`], this path intentionally has no `config.json` or
/// `tokenizer.json` dependency. The loaded GGUF remains owned by the model so
/// a later resident executor can upload its packed tensors without reopening
/// or reparsing the artifact.
pub struct Gemma4E2bModel {
    pub config: Gemma4E2bConfig,
    tokenizer: Tokenizer,
    gguf: GgufModel,
    ops: NeuralOps,
    resident_weights: Mutex<Gemma4ResidentWeights>,
}

#[derive(Default)]
struct Gemma4ResidentWeights {
    buffers: HashMap<String, GpuBuffer>,
    formats: HashMap<String, GgufTensorType>,
}

impl Gemma4E2bModel {
    pub fn load_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let gguf = GgufModel::open(path)
            .map_err(|error| anyhow::anyhow!("read Gemma 4 GGUF {}: {error}", path.display()))?;
        let config = Gemma4E2bConfig::from_gguf(&gguf)
            .with_context(|| format!("parse Gemma 4 GGUF {}", path.display()))?;
        config
            .validate_e2b_text_layout(&gguf)
            .with_context(|| format!("validate Gemma 4 E2B GGUF {}", path.display()))?;
        let tokenizer = gemma4_tokenizer(&gguf)
            .with_context(|| format!("build Gemma 4 tokenizer for {}", path.display()))?;
        Ok(Self {
            config,
            tokenizer,
            gguf,
            ops: NeuralOps::new().context("initialize Metal execution for Gemma 4")?,
            resident_weights: Mutex::new(Gemma4ResidentWeights::default()),
        })
    }

    pub fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(prompt, true)
            .map_err(|error| anyhow::anyhow!("tokenize Gemma 4 prompt: {error}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(token_ids, true)
            .map_err(|error| anyhow::anyhow!("decode Gemma 4 token IDs: {error}"))
    }

    /// Text-only subset of the embedded canonical template.  The artifact
    /// owns the template contract; Atlas intentionally rejects multimodal and
    /// tool turns in this E2B text-generation phase instead of silently using
    /// Llama's `user:` prompt convention.
    pub fn render_text_chat_prompt(&self, user: &str) -> Result<String> {
        ensure!(
            self.gguf.metadata.contains_key("tokenizer.chat_template"),
            "Gemma 4 GGUF is missing tokenizer.chat_template"
        );
        let content = user.trim();
        ensure!(!content.is_empty(), "Gemma chat prompt is empty");
        Ok(format!("<|turn>user\n{content}<turn|>\n<|turn>model\n"))
    }

    pub fn gguf(&self) -> &GgufModel {
        &self.gguf
    }

    pub(crate) fn runtime(&self) -> &atlas_metal::MetalRuntime {
        self.ops.runtime()
    }

    /// Upload every checked Gemma tensor once. The returned count deliberately
    /// excludes already-resident weights so request telemetry distinguishes a
    /// cold model upload from a warm executor allocation.
    pub(crate) fn ensure_resident_weights(&self) -> Result<u64> {
        let mut resident = self
            .resident_weights
            .lock()
            .expect("Gemma resident weight lock");
        let mut uploaded = 0u64;
        for tensor in &self.gguf.tensors {
            if resident.buffers.contains_key(&tensor.name) {
                continue;
            }
            let bytes = self.gguf.tensor_data(tensor)?;
            validate_gemma_q6_k_scales(tensor, bytes)?;
            let buffer = self.runtime().upload_bytes(bytes)?;
            uploaded = uploaded.saturating_add(buffer.bytes() as u64);
            resident
                .formats
                .insert(tensor.name.clone(), tensor.tensor_type);
            resident.buffers.insert(tensor.name.clone(), buffer);
        }
        Ok(uploaded)
    }

    pub(crate) fn resident_weight(&self, name: &str) -> Result<GpuBuffer> {
        self.resident_weights
            .lock()
            .expect("Gemma resident weight lock")
            .buffers
            .get(name)
            .cloned()
            .with_context(|| format!("Gemma resident weight missing `{name}`"))
    }

    pub(crate) fn resident_weight_format(&self, name: &str) -> Result<GgufTensorType> {
        self.resident_weights
            .lock()
            .expect("Gemma resident weight lock")
            .formats
            .get(name)
            .copied()
            .with_context(|| format!("Gemma resident tensor format missing `{name}`"))
    }

    pub(crate) fn resident_weight_bytes(&self) -> u64 {
        self.resident_weights
            .lock()
            .expect("Gemma resident weight lock")
            .buffers
            .values()
            .map(|buffer| buffer.bytes() as u64)
            .sum()
    }
}

/// Reject corrupt Q6_K source blocks before their bytes reach resident GPU
/// memory.  This is intentionally a diagnostic boundary, not a clamp: Gemma
/// QAT embedding values may be large, but the f16 block scale must always be
/// finite.  The tensor-relative and GGUF-relative offsets make a bad view or
/// row-stride calculation directly actionable.
fn validate_gemma_q6_k_scales(tensor: &atlas_core::GgufTensor, bytes: &[u8]) -> Result<()> {
    if tensor.tensor_type != GgufTensorType::Q6K {
        return Ok(());
    }
    let block_bytes = GgufTensorType::Q6K.block_bytes();
    ensure!(
        bytes.len().is_multiple_of(block_bytes),
        "Gemma Q6_K tensor `{}` has {} bytes, not a whole number of {block_bytes}-byte blocks",
        tensor.name,
        bytes.len()
    );
    let row_width = tensor.dims.first().copied().unwrap_or_default();
    ensure!(
        row_width > 0 && row_width.is_multiple_of(256),
        "Gemma Q6_K tensor `{}` has unsupported row width {row_width}",
        tensor.name
    );
    let row_blocks = row_width / 256;
    for (block_index, block) in bytes.chunks_exact(block_bytes).enumerate() {
        let scale_bits = u16::from_le_bytes([block[208], block[209]]);
        let scale = atlas_core::f16_bits_to_f32(scale_bits);
        if !scale.is_finite() {
            let row = block_index / row_blocks;
            let block_in_row = block_index % row_blocks;
            let tensor_byte_offset = block_index * block_bytes + 208;
            anyhow::bail!(
                "Gemma Q6_K non-finite block scale: tensor=`{}` row={row} block={block_in_row} tensor_byte_offset={tensor_byte_offset} gguf_byte_offset={} scale_bits=0x{scale_bits:04x}",
                tensor.name,
                tensor.offset + tensor_byte_offset,
            );
        }
    }
    Ok(())
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

#[cfg(test)]
mod gemma_q6_validation_tests {
    use super::validate_gemma_q6_k_scales;
    use atlas_core::{GgufTensor, GgufTensorType};

    #[test]
    fn resident_upload_guard_reports_the_q6_k_scale_field_not_quant_payload() {
        let tensor = GgufTensor {
            name: "per_layer_token_embd.weight".into(),
            dims: vec![256, 1],
            tensor_type: GgufTensorType::Q6K,
            offset: 4_096,
            bytes: GgufTensorType::Q6K.block_bytes(),
        };
        let mut bytes = vec![0; tensor.bytes];
        bytes[208..].copy_from_slice(&0x7e00u16.to_le_bytes());
        let error = validate_gemma_q6_k_scales(&tensor, &bytes).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Gemma Q6_K non-finite block scale: tensor=`per_layer_token_embd.weight` row=0 block=0 tensor_byte_offset=208 gguf_byte_offset=4304 scale_bits=0x7e00"
        );
    }
}
