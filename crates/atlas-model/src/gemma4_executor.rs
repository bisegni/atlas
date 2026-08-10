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
use atlas_core::{GgufTensorType, dequantize_block, quantize_q4_0};
use atlas_metal::GpuBuffer;

use crate::{
    Gemma4E2bModel, Generation, LayerTrace, gemma4_shared_kv_sources,
    quantization_plan::QuantizationPlan,
};

const GEMMA4_PREFILL_BATCH_CAPACITY: usize = 128;
#[cfg(test)]
const GEMMA4_DECODE_PROFILE_TARGETS: [usize; 9] = [1, 32, 64, 128, 256, 512, 1024, 2048, 4096];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4WeightFormat {
    MixedQ4Q6,
    Q4Embeddings,
    Q4LmHead,
    AllQ4,
}

impl Gemma4WeightFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MixedQ4Q6 => "mixed_q4_q6",
            Self::Q4Embeddings => "q4_embeddings",
            Self::Q4LmHead => "q4_lm_head",
            Self::AllQ4 => "all_q4",
        }
    }

    const fn embedding_format(self) -> GgufTensorType {
        match self {
            Self::Q4Embeddings | Self::AllQ4 => GgufTensorType::Q4_0,
            Self::MixedQ4Q6 | Self::Q4LmHead => GgufTensorType::Q6K,
        }
    }

    const fn output_format(self) -> GgufTensorType {
        match self {
            Self::Q4LmHead | Self::AllQ4 => GgufTensorType::Q4_0,
            Self::MixedQ4Q6 | Self::Q4Embeddings => GgufTensorType::Q6K,
        }
    }

    fn embedding_kernel(self) -> &'static str {
        match self.embedding_format() {
            GgufTensorType::Q4_0 => "embedding_lookup_q4_0",
            GgufTensorType::Q6K => "embedding_lookup_q6_k",
            _ => unreachable!("unsupported Gemma vocabulary embedding format"),
        }
    }

    fn output_projection_kernel(self) -> &'static str {
        match self.output_format() {
            GgufTensorType::Q4_0 => gemma4_q4_projection_kernel(),
            GgufTensorType::Q6K => gemma4_q6_projection_kernel(),
            _ => unreachable!("unsupported Gemma vocabulary output format"),
        }
    }

    const fn derives_embeddings(self) -> bool {
        matches!(self, Self::Q4Embeddings | Self::AllQ4)
    }

    const fn derives_output_projection(self) -> bool {
        matches!(self, Self::Q4LmHead | Self::AllQ4)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4SelectedGroupFormat {
    pub group: &'static str,
    pub source_format: GgufTensorType,
    pub selected_format: GgufTensorType,
    pub selected_kernel: &'static str,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Gemma4SelectedFormatMap {
    formats: BTreeMap<&'static str, Gemma4SelectedGroupFormat>,
}

impl Gemma4SelectedFormatMap {
    fn insert(&mut self, format: Gemma4SelectedGroupFormat) {
        self.formats.insert(format.group, format);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Gemma4SelectedGroupFormat> {
        self.formats.values()
    }

    fn selected_format(&self, group: &str) -> Option<GgufTensorType> {
        self.formats.get(group).map(|entry| entry.selected_format)
    }

    fn rejection_reasons(&self) -> Vec<String> {
        self.formats
            .values()
            .filter_map(|entry| entry.rejection_reason.clone())
            .collect()
    }
}

const GEMMA4_SELECTED_GROUP_VOCAB_EMBEDDINGS: &str = "vocabulary_embeddings";
const GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT: &str = "vocabulary_output_projection";
const GEMMA4_SELECTED_GROUP_QKV: &str = "qkv_projection";
const GEMMA4_SELECTED_GROUP_GATE_UP: &str = "gate_up_projection";
const GEMMA4_SELECTED_GROUP_ATTN_OUTPUT: &str = "attention_output_projection";
const GEMMA4_SELECTED_GROUP_FFN_DOWN: &str = "ffn_down_projection";
const GEMMA4_SELECTED_GROUP_PLE_PROJECTION: &str = "ple_projection";
const GEMMA4_SELECTED_GROUP_PER_LAYER_PROJ: &str = "per_layer_model_projection";

fn gemma4_weight_format() -> Result<Gemma4WeightFormat> {
    match std::env::var("ATLAS_GEMMA4_WEIGHT_FORMAT").as_deref() {
        Err(_) | Ok("mixed") | Ok("mixed_q4_q6") => Ok(Gemma4WeightFormat::MixedQ4Q6),
        Ok("q4_embeddings") => Ok(Gemma4WeightFormat::Q4Embeddings),
        Ok("q4_lm_head") => Ok(Gemma4WeightFormat::Q4LmHead),
        Ok("all_q4") => Ok(Gemma4WeightFormat::AllQ4),
        Ok(value) => anyhow::bail!(
            "unsupported ATLAS_GEMMA4_WEIGHT_FORMAT `{value}`; expected mixed_q4_q6, q4_embeddings, q4_lm_head, or all_q4"
        ),
    }
}

pub(crate) fn gemma4_weight_format_with_plan(
    plan: Option<&QuantizationPlan>,
) -> Result<Gemma4WeightFormat> {
    if std::env::var_os("ATLAS_GEMMA4_WEIGHT_FORMAT").is_some() {
        return gemma4_weight_format();
    }
    let Some(plan) = plan else {
        return gemma4_weight_format();
    };
    let token = plan.selected_format("token_embd.weight");
    let per_layer = plan.selected_format("per_layer_token_embd.weight");
    match (token, per_layer) {
        (Some(GgufTensorType::Q4_0), Some(GgufTensorType::Q4_0)) => Ok(Gemma4WeightFormat::AllQ4),
        (Some(GgufTensorType::Q6K), Some(GgufTensorType::Q6K)) => Ok(Gemma4WeightFormat::MixedQ4Q6),
        (None, None) => gemma4_weight_format(),
        (Some(left), Some(right)) => anyhow::bail!(
            "quantization plan selects incompatible Gemma vocabulary formats: token_embd={left:?}, per_layer_token_embd={right:?}"
        ),
        _ => anyhow::bail!("quantization plan must select both Gemma vocabulary tensors together"),
    }
}

pub(crate) fn gemma4_selected_group_formats(
    weight_format: Gemma4WeightFormat,
) -> Gemma4SelectedFormatMap {
    let mut formats = Gemma4SelectedFormatMap::default();
    let vocabulary_embedding_format = weight_format.embedding_format();
    let vocabulary_output_format = weight_format.output_format();
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_VOCAB_EMBEDDINGS,
        source_format: GgufTensorType::Q6K,
        selected_format: vocabulary_embedding_format,
        selected_kernel: match vocabulary_embedding_format {
            GgufTensorType::Q4_0 => "embedding_lookup_q4_0",
            GgufTensorType::Q6K => "embedding_lookup_q6_k",
            _ => "embedding_lookup_q6_k",
        },
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT,
        source_format: GgufTensorType::Q6K,
        selected_format: vocabulary_output_format,
        selected_kernel: match vocabulary_output_format {
            GgufTensorType::Q4_0 => gemma4_q4_projection_kernel(),
            GgufTensorType::Q6K => gemma4_q6_projection_kernel(),
            _ => gemma4_q6_projection_kernel(),
        },
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_QKV,
        source_format: GgufTensorType::Q4_0,
        selected_format: GgufTensorType::Q4_0,
        selected_kernel: gemma4_q4_qkv_projection_kernel(),
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_GATE_UP,
        source_format: GgufTensorType::Q4_0,
        selected_format: GgufTensorType::Q4_0,
        selected_kernel: gemma4_q4_gate_up_projection_kernel(),
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_ATTN_OUTPUT,
        source_format: GgufTensorType::Q4_0,
        selected_format: GgufTensorType::Q4_0,
        selected_kernel: gemma4_q4_projection_kernel(),
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_FFN_DOWN,
        source_format: GgufTensorType::Q4_0,
        selected_format: GgufTensorType::Q4_0,
        selected_kernel: gemma4_ffn_down_projection_kernel(),
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_PLE_PROJECTION,
        source_format: GgufTensorType::Q4_0,
        selected_format: GgufTensorType::Q4_0,
        selected_kernel: gemma4_q4_projection_kernel(),
        rejection_reason: None,
    });
    formats.insert(Gemma4SelectedGroupFormat {
        group: GEMMA4_SELECTED_GROUP_PER_LAYER_PROJ,
        source_format: GgufTensorType::F16,
        selected_format: GgufTensorType::F16,
        selected_kernel: "matmul_f16_batch",
        rejection_reason: None,
    });
    formats
}

/// Convert row-major GGML Q6_K data to deterministic row-major Q4_0 blocks.
/// The Q6 source is the oracle: no model tensor is mutated or rewritten.
fn gemma4_q6_k_to_q4_0(source: &[u8], row_width: usize) -> Result<Vec<u8>> {
    ensure!(
        row_width > 0 && row_width.is_multiple_of(256),
        "Gemma Q6_K vocabulary row width must be a positive multiple of 256"
    );
    ensure!(
        source
            .len()
            .is_multiple_of(GgufTensorType::Q6K.block_bytes()),
        "Gemma Q6_K vocabulary source is not block aligned"
    );
    let source_blocks_per_row = row_width / 256;
    ensure!(
        source.len() / GgufTensorType::Q6K.block_bytes() % source_blocks_per_row == 0,
        "Gemma Q6_K vocabulary source does not contain whole rows"
    );
    let mut values = vec![0.0f32; 256];
    let mut output = Vec::with_capacity(source.len() / GgufTensorType::Q6K.block_bytes() * 8 * 18);
    for block in source.chunks_exact(GgufTensorType::Q6K.block_bytes()) {
        dequantize_block(GgufTensorType::Q6K, block, &mut values)
            .context("dequantize Gemma Q6_K vocabulary block")?;
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "Gemma Q6_K vocabulary block dequantized to a non-finite value"
        );
        output.extend_from_slice(
            &quantize_q4_0(&values).context("quantize Gemma Q4_0 vocabulary block")?,
        );
    }
    Ok(output)
}

#[cfg(test)]
fn gemma4_decode_profile_targets(decode_tokens: usize) -> Vec<usize> {
    GEMMA4_DECODE_PROFILE_TARGETS
        .into_iter()
        .filter(|target| *target <= decode_tokens)
        .collect()
}

fn gemma4_q4_flash16_supported(attention_heads: usize, head_dim: usize) -> bool {
    attention_heads == 8 && (head_dim == 512 || head_dim == 256)
}

/// The promoted production attention binding: the flash16 unweighted
/// two-tile attention kernel (with the sliding-window variant for sliding
/// layers). Sliding layers keep their own kernel and label but share the
/// same pipeline, buffers, and reduction order.
fn gemma4_q4_flash16_binding(sliding: bool) -> (&'static str, &'static str, u32) {
    if sliding {
        (
            "attention_decode_gemma4_simd_q4_0_flash16_swa_uw",
            "gemma_attention_flash16_swa",
            768,
        )
    } else {
        (
            "attention_decode_gemma4_simd_q4_0_flash16_uw",
            "gemma_attention_flash16",
            384,
        )
    }
}

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
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub command_buffers: u64,
    pub dispatches: u64,
    pub buffer_allocations: u64,
    pub peak_resident_bytes: u64,
    pub gpu_execution_time: Duration,
    pub prefill_command_buffers: u64,
    pub decode_command_buffers: u64,
    pub prefill: Duration,
    pub decode: Duration,
    pub host_wall_time: Duration,
    pub prefill_path: &'static str,
    pub prefill_chunk_size: usize,
    pub prefill_chunks: usize,
    pub quantization_preflight_state: &'static str,
    pub selected_group_formats: Vec<Gemma4SelectedGroupFormat>,
    pub quantization_rejections: Vec<String>,
    pub attention_kernel: &'static str,
    pub weight_format: Gemma4WeightFormat,
    pub embedding_kernel: &'static str,
    pub output_projection_kernel: &'static str,
    pub q6_projection_kernel: &'static str,
    pub q4_projection_kernel: &'static str,
    pub q4_qkv_projection_kernel: &'static str,
    pub q4_gate_up_projection_kernel: &'static str,
    pub ffn_gate_up_activation_kernel: &'static str,
    pub ffn_gate_up_scratch_bytes: u64,
    pub ple_composition_kernel: &'static str,
    pub rms_epilogue_kernel: &'static str,
    pub q4_batch_projection_kernel: &'static str,
    pub ffn_down_projection_kernel: &'static str,
    pub ple_projection_kernel: &'static str,
    pub rms_norm_kernel: &'static str,
    pub rms_fused_projection_kernel: &'static str,
    pub kv_append_kernel: &'static str,
    pub kv_cache_type: Gemma4KvCacheType,
    pub kv_cache_bytes: u64,
    pub quantization_plan_path: Option<String>,
    pub warmup_decode_tokens: usize,
    pub measured_decode_tokens: usize,
    pub completed_decode_tokens: usize,
    pub warmup_scope: Gemma4ScopeMetrics,
    pub measured_scope: Gemma4ScopeMetrics,
    pub complete_decode_scope: Gemma4ScopeMetrics,
    pub prefill_scope: Gemma4ScopeMetrics,
    pub physical_command_buffer_overlap: bool,
    pub physical_command_buffer_overlap_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Gemma4ScopeMetrics {
    pub wall_time: Duration,
    pub host_start_ns: u64,
    pub host_end_ns: u64,
    pub telemetry: atlas_metal::RuntimeTelemetry,
    pub completed_tokens: usize,
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

fn gemma4_prefill_path(prompt_tokens: usize) -> &'static str {
    if prompt_tokens > 1 {
        "resident_layer_major"
    } else {
        "resident_chunked_command"
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
    pub kernel_name: &'static str,
    pub layer_index: Option<u32>,
    pub command_buffer_id: Option<u64>,
    pub dispatches: u64,
    pub gpu_nanos: u128,
    pub cpu_encode_nanos: u128,
    pub threadgroups: u64,
    pub threads: u64,
    pub bytes_read_estimate: u64,
    pub bytes_written_estimate: u64,
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
    pub scope: &'static str,
    pub kernels: Vec<Gemma4DecodeKernelProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gemma4DecodeProfile {
    pub prompt_tokens: usize,
    pub requested_decode_tokens: usize,
    pub warmup_decode_tokens: usize,
    pub measured_decode_tokens: usize,
    pub completed_decode_tokens: usize,
    pub generated_token_ids: Vec<u32>,
    pub first_eos_position: Option<usize>,
    pub prefill: Duration,
    pub warmup_decode: Duration,
    pub measured_decode: Duration,
    pub decode: Duration,
    pub host_wall_time: Duration,
    pub prefill_telemetry: atlas_metal::RuntimeTelemetry,
    pub warmup_telemetry: atlas_metal::RuntimeTelemetry,
    pub measured_telemetry: atlas_metal::RuntimeTelemetry,
    pub decode_telemetry: atlas_metal::RuntimeTelemetry,
    pub complete_decode_telemetry: atlas_metal::RuntimeTelemetry,
    pub prefill_kernels: Vec<Gemma4DecodeKernelProfile>,
    pub prefill_path: &'static str,
    pub attention_kernel: &'static str,
    pub kv_cache_type: Gemma4KvCacheType,
    pub samples: Vec<Gemma4DecodeProfileSample>,
}

fn gemma4_kernel_family(kernel: &str) -> &'static str {
    if matches!(
        kernel,
        "matmul_q4_0_qkv_32row_mv" | "matmul_q4_0_qkv_32row_mv_rms"
    ) {
        "q4_qkv_projection"
    } else if matches!(
        kernel,
        "matmul_q4_0_gate_up_32row_mv" | "matmul_q4_0_gate_up_32row_mv_rms"
    ) {
        "q4_ffn_gate_up_projection"
    } else if matches!(
        kernel,
        "matvec_q6_k_32row_mv"
            | "matvec_q6_k_32row_mv_rms"
            | "matvec_q6_k_64row_mv"
            | "matvec_q6_k_64row_mv_rms"
    ) {
        "q6_lm_head_projection"
    } else if kernel.starts_with("matmul_q4_0_batch") || kernel.starts_with("matmul_q6_k_batch") {
        "batched_projection"
    } else if kernel.starts_with("matvec_q4") || kernel.starts_with("matmul_q4") {
        "q4_projection_other"
    } else if kernel.starts_with("matvec_q6") || kernel.starts_with("matmul_q6") {
        "q6_projection_other"
    } else if kernel.contains("attention") || kernel.contains("attn_") {
        "gemma_attention"
    } else if kernel == "gemma4_qk_norm_rope_fused_f32" {
        "qk_norm_rope_fused"
    } else if kernel.starts_with("rms_norm") {
        "rms_norm"
    } else if kernel == "rope_f32" || kernel == "rope_llama_decode_f32" {
        "rope_rotation"
    } else if kernel.starts_with("rope_") {
        "rope_layout"
    } else if kernel.contains("kv_") {
        "kv_append"
    } else if kernel.starts_with("embedding_lookup") {
        "embedding_lookup"
    } else if kernel.starts_with("ple_") {
        "ple_projection"
    } else if kernel.contains("gelu") || kernel.contains("vector_multiply") {
        "ffn_activation"
    } else if kernel.contains("vector_add") {
        "residual"
    } else if kernel.contains("scalar_multiply") {
        "conversion"
    } else if kernel.contains("softcap") {
        "softcap"
    } else if kernel.contains("argmax") {
        "argmax"
    } else {
        "other"
    }
}

fn gemma4_profile_family(profiling_label: Option<&'static str>, kernel: &str) -> &'static str {
    match profiling_label {
        // Keep the PLE input-gate projection in the existing aggregate while
        // distinguishing it from the packed PLE output projection at dispatch
        // selection time.
        Some("ple_input_gate") => "ple_projection",
        other => other.unwrap_or_else(|| gemma4_kernel_family(kernel)),
    }
}

fn aggregate_profile_timings(
    timings: Vec<atlas_metal::ResidentKernelTiming>,
) -> Vec<Gemma4DecodeKernelProfile> {
    let mut kernels: BTreeMap<
        (Option<u64>, Option<u32>, &'static str, &'static str),
        Gemma4DecodeKernelProfile,
    > = BTreeMap::new();
    for timing in timings {
        let family = gemma4_profile_family(timing.profiling_label, timing.kernel);
        let entry = kernels
            .entry((
                timing.command_buffer_id,
                timing.layer_index,
                family,
                timing.kernel,
            ))
            .or_insert(Gemma4DecodeKernelProfile {
                family,
                kernel_name: timing.kernel,
                layer_index: timing.layer_index,
                command_buffer_id: timing.command_buffer_id,
                dispatches: 0,
                gpu_nanos: 0,
                cpu_encode_nanos: 0,
                threadgroups: 0,
                threads: 0,
                bytes_read_estimate: 0,
                bytes_written_estimate: 0,
            });
        entry.dispatches += 1;
        entry.gpu_nanos += timing.timing.gpu_time.unwrap_or_default().as_nanos();
        entry.cpu_encode_nanos += timing.cpu_encode.as_nanos();
        entry.threadgroups += timing.threadgroups as u64;
        entry.threads += timing.threads as u64;
        entry.bytes_read_estimate += timing.bytes_read_estimate;
        entry.bytes_written_estimate += timing.bytes_written_estimate;
    }
    kernels.into_values().collect()
}

fn gemma4_attention_binding(cache_type: Gemma4KvCacheType) -> (&'static str, usize) {
    match cache_type {
        Gemma4KvCacheType::F32 => ("attention_decode_fused_gemma4_simd_f32", 128),
        Gemma4KvCacheType::Q8_0 => ("attention_decode_fused_gemma4_simd_q8_0", 128),
        Gemma4KvCacheType::Q4_0 => ("attention_decode_fused_gemma4_simd_q4_0", 128),
    }
}

fn gemma4_attention_kernel(
    cache_type: Gemma4KvCacheType,
    attention_heads: usize,
    full_head_dim: usize,
    sliding: bool,
) -> &'static str {
    if cache_type == Gemma4KvCacheType::Q4_0
        && gemma4_q4_flash16_supported(attention_heads, full_head_dim)
    {
        return gemma4_q4_flash16_binding(sliding).0;
    }
    gemma4_attention_binding(cache_type).0
}

fn gemma4_q6_projection_kernel() -> &'static str {
    "matvec_q6_k_64row_mv"
}

pub(crate) fn gemma4_q4_projection_kernel() -> &'static str {
    "matvec_q4_0_64row_mv"
}

pub(crate) fn gemma4_q4_qkv_projection_kernel() -> &'static str {
    "matmul_q4_0_qkv_32row_mv_rms"
}

pub(crate) fn gemma4_q4_gate_up_projection_kernel() -> &'static str {
    "matmul_q4_0_gate_up_32row_mv_rms"
}

fn gemma4_q4_batch_projection_kernel() -> &'static str {
    "matmul_q4_0_batch_16row"
}

pub(crate) fn gemma4_ffn_down_projection_kernel() -> &'static str {
    "matvec_q4_0_64row_mv"
}

fn gemma4_rms_norm_decode_kernel(hidden_size: usize) -> &'static str {
    if hidden_size.is_multiple_of(128) {
        "rms_norm_decode_f32_vec4"
    } else {
        "rms_norm_decode_f32"
    }
}

fn gemma4_kv_append_kernel(cache_type: Gemma4KvCacheType) -> &'static str {
    match cache_type {
        Gemma4KvCacheType::F32 => "kv_append_decode_f32",
        Gemma4KvCacheType::Q8_0 => "kv_append_decode_q8_0",
        Gemma4KvCacheType::Q4_0 => "kv_append_decode_q4_0",
    }
}

fn gemma4_kv_append_vnorm_kernel(cache_type: Gemma4KvCacheType) -> &'static str {
    match cache_type {
        Gemma4KvCacheType::F32 => "kv_append_decode_f32_vnorm",
        Gemma4KvCacheType::Q8_0 => "kv_append_decode_q8_0_vnorm",
        Gemma4KvCacheType::Q4_0 => "kv_append_decode_q4_0_vnorm",
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

/// Reusable GPU matrices for layer-major prompt prefill.  Decode deliberately
/// continues to use the narrow single-token buffers below; prompt execution
/// owns one row per token and swaps `state`/`next_state` after each layer.
struct Gemma4PrefillBuffers {
    state: GpuBuffer,
    next_state: GpuBuffer,
    residual: GpuBuffer,
    norm: GpuBuffer,
    q: GpuBuffer,
    q_rot: GpuBuffer,
    k: GpuBuffer,
    k_rot: GpuBuffer,
    v: GpuBuffer,
    attention: GpuBuffer,
    work: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    activated: GpuBuffer,
    product: GpuBuffer,
    ple_lookup: GpuBuffer,
    ple_projected: GpuBuffer,
    ple: GpuBuffer,
}

pub struct Gemma4E2bExecutor<'a> {
    model: &'a Gemma4E2bModel,
    max_context: usize,
    position: usize,
    kv_sources: Vec<usize>,
    kv: Vec<Option<GpuBuffer>>,
    kv_cache_type: Gemma4KvCacheType,
    weight_format: Gemma4WeightFormat,
    quantization_plan_path: Option<String>,
    token_embedding: GpuBuffer,
    per_layer_embedding: GpuBuffer,
    output_projection: GpuBuffer,
    derived_vocabulary_bytes: u64,
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
    ple_lookup: GpuBuffer,
    ple_projected: GpuBuffer,
    ple: GpuBuffer,
    logits: GpuBuffer,
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
    prefill: Gemma4PrefillBuffers,
    pending_weight_upload_bytes: u64,
    selected_group_formats: Gemma4SelectedFormatMap,
    quantization_preflight_state: &'static str,
    quantization_rejections: Vec<String>,
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
        Self::new_with_kv_cache_from_selection(model, max_context, kv_cache_type, None)
    }

    pub(crate) fn new_with_kv_cache_from_selection(
        model: &'a Gemma4E2bModel,
        max_context: usize,
        kv_cache_type: Gemma4KvCacheType,
        selection: Option<(Gemma4SelectedFormatMap, Option<String>, &'static str)>,
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
        let (quantization_plan_path, quantization_preflight_state, selected_group_formats) =
            if let Some((
                selected_group_formats,
                quantization_plan_path,
                quantization_preflight_state,
            )) = selection
            {
                (
                    quantization_plan_path,
                    quantization_preflight_state,
                    selected_group_formats,
                )
            } else if std::env::var_os(
                crate::gemma4_quantization_preflight::ATLAS_GEMMA4_WEIGHT_FORMAT,
            )
            .is_some()
            {
                let weight_format =
                    crate::gemma4_quantization_preflight::parse_weight_format_override(
                        std::env::var(
                            crate::gemma4_quantization_preflight::ATLAS_GEMMA4_WEIGHT_FORMAT,
                        )
                        .ok()
                        .as_deref(),
                    )?
                    .unwrap_or(Gemma4WeightFormat::MixedQ4Q6);
                (
                    None,
                    "explicit",
                    gemma4_selected_group_formats(weight_format),
                )
            } else if crate::gemma4_quantization_preflight::parse_preflight_policy(
                std::env::var(
                    crate::gemma4_quantization_preflight::ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT,
                )
                .ok()
                .as_deref(),
            )?
                == crate::gemma4_quantization_preflight::Gemma4QuantizationPreflightPolicy::Disabled
            {
                (
                    None,
                    "disabled",
                    gemma4_selected_group_formats(Gemma4WeightFormat::MixedQ4Q6),
                )
            } else {
                let hardware_identity = {
                    let info = model.runtime().device_info();
                    format!("{}#{}", info.name, info.registry_id)
                };
                let quantization_plan =
                    model.quantization_plan_with_identity(Some(&hardware_identity))?;
                let weight_format = gemma4_weight_format_with_plan(quantization_plan.as_ref())?;
                let quantization_plan_path = quantization_plan.map(|_| {
                    crate::quantization_plan::default_sidecar_path(model.model_path())
                        .display()
                        .to_string()
                });
                let quantization_preflight_state = if quantization_plan_path.is_some() {
                    "ready"
                } else {
                    "disabled"
                };
                (
                    quantization_plan_path,
                    quantization_preflight_state,
                    gemma4_selected_group_formats(weight_format),
                )
            };
        let weight_format =
            match selected_group_formats.selected_format(GEMMA4_SELECTED_GROUP_VOCAB_EMBEDDINGS) {
                Some(GgufTensorType::Q4_0) => match selected_group_formats
                    .selected_format(GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT)
                {
                    Some(GgufTensorType::Q4_0) => Gemma4WeightFormat::AllQ4,
                    Some(GgufTensorType::Q6K) => Gemma4WeightFormat::Q4Embeddings,
                    _ => Gemma4WeightFormat::AllQ4,
                },
                Some(GgufTensorType::Q6K) => match selected_group_formats
                    .selected_format(GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT)
                {
                    Some(GgufTensorType::Q4_0) => Gemma4WeightFormat::Q4LmHead,
                    Some(GgufTensorType::Q6K) => Gemma4WeightFormat::MixedQ4Q6,
                    _ => Gemma4WeightFormat::MixedQ4Q6,
                },
                _ => gemma4_weight_format()?,
            };
        let quantization_rejections = selected_group_formats.rejection_reasons();
        let mut weight_upload_bytes =
            model.ensure_resident_weights(weight_format == Gemma4WeightFormat::AllQ4)?;
        let derive = |name: &str| -> Result<GpuBuffer> {
            let tensor = model
                .gguf()
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .with_context(|| format!("Gemma Q4 vocabulary source tensor missing `{name}`"))?;
            ensure!(
                tensor.tensor_type == GgufTensorType::Q6K,
                "Gemma Q4 vocabulary source tensor `{name}` must be Q6_K"
            );
            let row_width = tensor.dims.first().copied().unwrap_or_default();
            let packed = gemma4_q6_k_to_q4_0(model.gguf().tensor_data(tensor)?, row_width)?;
            runtime.upload_bytes(&packed).map_err(Into::into)
        };
        let token_embedding = if weight_format.derives_embeddings() {
            derive("token_embd.weight")?
        } else {
            model.resident_weight("token_embd.weight")?
        };
        let per_layer_embedding = if weight_format.derives_embeddings() {
            derive("per_layer_token_embd.weight")?
        } else {
            model.resident_weight("per_layer_token_embd.weight")?
        };
        let output_projection = if weight_format.derives_output_projection() {
            if weight_format == Gemma4WeightFormat::AllQ4 {
                token_embedding.clone()
            } else {
                derive("token_embd.weight")?
            }
        } else {
            model.resident_weight("token_embd.weight")?
        };
        let derived_vocabulary_bytes = (usize::from(weight_format.derives_embeddings())
            * (token_embedding.bytes() + per_layer_embedding.bytes())
            + usize::from(weight_format == Gemma4WeightFormat::Q4LmHead)
                * output_projection.bytes()) as u64;
        weight_upload_bytes = weight_upload_bytes.saturating_add(derived_vocabulary_bytes);
        let prefill = Gemma4PrefillBuffers {
            state: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * h)?,
            next_state: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * h)?,
            residual: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * h)?,
            norm: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * h)?,
            q: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * q_width)?,
            q_rot: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * q_width)?,
            k: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * head)?,
            k_rot: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * head)?,
            v: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * head)?,
            attention: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * q_width)?,
            work: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * h)?,
            gate: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * max_ffn)?,
            up: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * max_ffn)?,
            activated: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * max_ffn)?,
            product: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * max_ffn)?,
            ple_lookup: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * ple_total)?,
            ple_projected: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * ple_total)?,
            ple: allocate(GEMMA4_PREFILL_BATCH_CAPACITY * ple_total)?,
        };
        Ok(Self {
            model,
            max_context,
            position: 0,
            kv_sources,
            kv,
            kv_cache_type,
            weight_format,
            quantization_plan_path,
            token_embedding,
            per_layer_embedding,
            output_projection,
            derived_vocabulary_bytes,
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
            ple_lookup: allocate(ple_total)?,
            ple_projected: allocate(ple_total)?,
            ple: allocate(ple_total)?,
            logits: allocate(c.vocab_size)?,
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
            prefill,
            pending_weight_upload_bytes: weight_upload_bytes,
            selected_group_formats,
            quantization_preflight_state,
            quantization_rejections,
        })
    }

    pub fn resident_bytes(&self) -> u64 {
        self.model.resident_weight_bytes()
            + self.derived_vocabulary_bytes
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
                &self.activated,
                &self.product,
                &self.ple_lookup,
                &self.ple_projected,
                &self.ple,
                &self.logits,
            ]
            .iter()
            .map(|v| v.bytes() as u64)
            .sum::<u64>()
            + self.up.bytes() as u64
            + [
                &self.prefill.state,
                &self.prefill.next_state,
                &self.prefill.residual,
                &self.prefill.norm,
                &self.prefill.q,
                &self.prefill.q_rot,
                &self.prefill.k,
                &self.prefill.k_rot,
                &self.prefill.v,
                &self.prefill.attention,
                &self.prefill.work,
                &self.prefill.gate,
                &self.prefill.up,
                &self.prefill.activated,
                &self.prefill.product,
                &self.prefill.ple_lookup,
                &self.prefill.ple_projected,
                &self.prefill.ple,
            ]
            .iter()
            .map(|buffer| buffer.bytes() as u64)
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

    pub fn runtime_telemetry(&self) -> atlas_metal::RuntimeTelemetry {
        self.model.runtime().runtime_telemetry()
    }

    pub fn diagnostic_counter_metadata(&self) -> atlas_metal::DiagnosticCounterMetadata {
        self.model.runtime().diagnostic_counter_metadata()
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
        self.matvec_labeled(
            command,
            None,
            input,
            weight,
            output,
            input_width,
            output_width_buffer,
            output_width,
            format,
        )
    }

    fn matvec_labeled(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        profiling_label: Option<&'static str>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width_buffer: &GpuBuffer,
        output_width: usize,
        format: GgufTensorType,
    ) -> Result<()> {
        let kernel = match format {
            GgufTensorType::Q4_0 => gemma4_q4_projection_kernel(),
            GgufTensorType::Q6K => gemma4_q6_projection_kernel(),
            GgufTensorType::F16 => "matvec_f16",
            other => anyhow::bail!("unsupported Gemma matvec format {other:?}"),
        };
        let buffers = &[input, weight, output, input_width, output_width_buffer];
        if format == GgufTensorType::Q4_0 {
            command.dispatch_threadgroups_1d_labeled(
                kernel,
                profiling_label,
                buffers,
                output_width.div_ceil(64),
                256,
            )?;
        } else if format == GgufTensorType::Q6K {
            command.dispatch_threadgroups_1d_labeled(
                kernel,
                profiling_label,
                buffers,
                output_width.div_ceil(64),
                256,
            )?;
        } else {
            command.dispatch_1d_labeled(kernel, profiling_label, buffers, output_width)?;
        }
        Ok(())
    }

    /// Matvec over an input that is normalized in-kernel by the RMS-input
    /// variants of the mv_ext kernels. Only the vocabulary output projection
    /// consumes this path; the attention-input and FFN-input norms ride the
    /// fused qkv/gate-up dispatches instead.
    fn matvec_rms_labeled(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        profiling_label: Option<&'static str>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width_buffer: &GpuBuffer,
        output_width: usize,
        format: GgufTensorType,
        rms_weight: &GpuBuffer,
    ) -> Result<()> {
        let (kernel, rows_per_group): (&'static str, usize) = match format {
            GgufTensorType::Q4_0 => ("matvec_q4_0_64row_mv_rms", 64),
            GgufTensorType::Q6K => ("matvec_q6_k_64row_mv_rms", 64),
            _ => anyhow::bail!(
                "RMS-input matvec requires the mv_ext kernel family for format {format:?}"
            ),
        };
        let buffers = &[
            input,
            weight,
            output,
            input_width,
            output_width_buffer,
            rms_weight,
            &self.epsilon,
        ];
        command.dispatch_threadgroups_1d_labeled(
            kernel,
            profiling_label,
            buffers,
            output_width.div_ceil(rows_per_group),
            if rows_per_group == 64 { 256 } else { 128 },
        )?;
        Ok(())
    }

    fn matvec_ffn_down(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width: &GpuBuffer,
        output_width_value: usize,
    ) -> Result<()> {
        command.dispatch_threadgroups_1d_labeled(
            gemma4_ffn_down_projection_kernel(),
            Some("ffn_down_projection"),
            &[input, weight, output, input_width, output_width],
            output_width_value.div_ceil(64),
            256,
        )?;
        Ok(())
    }

    fn rms_norm_decode_labeled(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        profiling_label: Option<&'static str>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
    ) -> Result<()> {
        let kernel = gemma4_rms_norm_decode_kernel(self.model.config.hidden_size);
        command.dispatch_threadgroups_1d_labeled(
            kernel,
            profiling_label,
            &[input, weight, output, &self.hidden, &self.epsilon],
            1,
            32,
        )?;
        Ok(())
    }

    /// Apply one resident weight matrix to every row of a prompt activation
    /// matrix.  The input/output layout is row-major `[tokens, width]`; model
    /// weights remain in their original GGUF packing.
    fn matmul_batch(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width: &GpuBuffer,
        batch: &GpuBuffer,
        output_width_value: usize,
        batch_value: usize,
        format: GgufTensorType,
    ) -> Result<()> {
        let kernel = match format {
            GgufTensorType::Q4_0 => gemma4_q4_batch_projection_kernel(),
            GgufTensorType::Q6K => "matmul_q6_k_batch_8row",
            GgufTensorType::F16 => "matmul_f16_batch",
            other => anyhow::bail!("unsupported Gemma batched matrix format {other:?}"),
        };
        let buffers = &[input, weight, output, input_width, output_width, batch];
        match format {
            GgufTensorType::Q4_0 => command.dispatch_threadgroups_1d_labeled(
                kernel,
                Some("layer_major_batched_projection"),
                buffers,
                batch_value * output_width_value.div_ceil(16),
                128,
            )?,
            GgufTensorType::Q6K => command.dispatch_threadgroups_1d_labeled(
                kernel,
                Some("layer_major_batched_projection"),
                buffers,
                batch_value * output_width_value.div_ceil(8),
                128,
            )?,
            GgufTensorType::F16 => command.dispatch_1d_labeled(
                kernel,
                Some("layer_major_batched_projection"),
                buffers,
                batch_value * output_width_value,
            )?,
            _ => unreachable!("formats above are exhaustive"),
        }
        Ok(())
    }

    fn matmul_ffn_down_batch(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        input_width: &GpuBuffer,
        output_width: &GpuBuffer,
        batch: &GpuBuffer,
        output_width_value: usize,
        batch_value: usize,
    ) -> Result<()> {
        let kernel = gemma4_q4_batch_projection_kernel();
        command.dispatch_threadgroups_1d_labeled(
            kernel,
            Some("layer_major_batched_ffn_down_projection"),
            &[input, weight, output, input_width, output_width, batch],
            batch_value * output_width_value.div_ceil(16),
            128,
        )?;
        Ok(())
    }

    fn rms_norm_batch_decode_order(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) -> Result<()> {
        command
            .dispatch_threadgroups_1d_labeled(
                "rms_norm_decode_batch_f32",
                Some("layer_major_rms_norm"),
                &[input, weight, output, &self.hidden, &self.epsilon],
                rows,
                32,
            )
            .map_err(Into::into)
    }

    fn matmul_q4_0_qkv(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        q_weight: &GpuBuffer,
        k_weight: &GpuBuffer,
        v_weight: &GpuBuffer,
        q_output: &GpuBuffer,
        k_output: &GpuBuffer,
        v_output: &GpuBuffer,
        q_width: &GpuBuffer,
        kv_width: &GpuBuffer,
        q_width_value: usize,
        kv_width_value: usize,
        rms_weight: Option<&GpuBuffer>,
    ) -> Result<()> {
        let (kernel, buffers): (&'static str, Vec<&GpuBuffer>) = match rms_weight {
            Some(rms_weight) => (
                "matmul_q4_0_qkv_32row_mv_rms",
                vec![
                    input,
                    q_weight,
                    k_weight,
                    v_weight,
                    q_output,
                    k_output,
                    v_output,
                    &self.hidden,
                    q_width,
                    kv_width,
                    rms_weight,
                    &self.epsilon,
                ],
            ),
            _ => (
                gemma4_q4_qkv_projection_kernel(),
                vec![
                    input,
                    q_weight,
                    k_weight,
                    v_weight,
                    q_output,
                    k_output,
                    v_output,
                    &self.hidden,
                    q_width,
                    kv_width,
                ],
            ),
        };
        command.dispatch_threadgroups_1d_labeled(
            kernel,
            Some("qkv_projection"),
            &buffers,
            q_width_value.div_ceil(32) + 2 * kv_width_value.div_ceil(32),
            128,
        )?;
        Ok(())
    }

    fn matmul_q4_0_gate_up(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        input: &GpuBuffer,
        gate_weight: &GpuBuffer,
        up_weight: &GpuBuffer,
        gate_output: &GpuBuffer,
        up_output: &GpuBuffer,
        output_width: &GpuBuffer,
        output_width_value: usize,
        rms_weight: Option<&GpuBuffer>,
    ) -> Result<()> {
        let (kernel, buffers): (&'static str, Vec<&GpuBuffer>) = match rms_weight {
            Some(rms_weight) => (
                "matmul_q4_0_gate_up_32row_mv_rms",
                vec![
                    input,
                    gate_weight,
                    up_weight,
                    gate_output,
                    up_output,
                    &self.hidden,
                    output_width,
                    rms_weight,
                    &self.epsilon,
                ],
            ),
            _ => (
                gemma4_q4_gate_up_projection_kernel(),
                vec![
                    input,
                    gate_weight,
                    up_weight,
                    gate_output,
                    up_output,
                    &self.hidden,
                    output_width,
                ],
            ),
        };
        command.dispatch_threadgroups_1d_labeled(
            kernel,
            Some("ffn_gate_up_projection"),
            &buffers,
            2 * output_width_value.div_ceil(32),
            128,
        )?;
        Ok(())
    }

    fn gemma4_qk_norm_rope_fused(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        q_weight: &GpuBuffer,
        k_weight: &GpuBuffer,
        cosine: &GpuBuffer,
        sine: &GpuBuffer,
        rope_offset: usize,
        head_width: &GpuBuffer,
        provider_key: bool,
    ) -> Result<()> {
        command.dispatch_threadgroups_1d_at(
            "gemma4_qk_norm_rope_fused_f32",
            &[
                (&self.q, 0),
                (&self.k, 0),
                (q_weight, 0),
                (k_weight, 0),
                (cosine, rope_offset),
                (sine, rope_offset),
                (&self.q_rot, 0),
                (&self.k_rot, 0),
                (head_width, 0),
                (&self.heads, 0),
                (&self.one, 0),
                (&self.epsilon, 0),
            ],
            self.model.config.attention_heads + usize::from(provider_key),
            1,
        )?;
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
        self.profile_decode_with_timing(prompt, 0, decode_tokens, true)
    }

    /// Profile a workload while optionally retaining the production
    /// command-buffer boundary. Exact per-dispatch timing is diagnostic-only;
    /// benchmark callers must pass `false`.
    pub fn profile_decode_with_timing(
        &mut self,
        prompt: &str,
        warmup_decode_tokens: usize,
        measured_decode_tokens: usize,
        exact_profile: bool,
    ) -> Result<Gemma4DecodeProfile> {
        ensure!(
            measured_decode_tokens > 0,
            "Gemma decode profile needs a positive measured window"
        );
        let total_decode_tokens = warmup_decode_tokens
            .checked_add(measured_decode_tokens)
            .context("Gemma decode profile window overflows")?;
        let prompt_ids = self.model.tokenize(prompt)?;
        ensure!(!prompt_ids.is_empty(), "prompt tokenizes to no tokens");
        ensure!(
            prompt_ids.len() + total_decode_tokens <= self.max_context,
            "Gemma executor context exhausted"
        );
        self.position = 0;
        let runtime = self.model.runtime();
        let prefill_before = runtime.runtime_telemetry();
        let started = Instant::now();
        let prefill_started = Instant::now();
        let (selected, prefill_timings) =
            self.forward_tokens_with_profile(&prompt_ids, true, exact_profile)?;
        let mut selected = selected.context("Gemma prefill did not select a first decode token")?;
        let prefill = prefill_started.elapsed();
        let prefill_telemetry = runtime.runtime_telemetry().delta_from(prefill_before);
        // Diagnostic mode profiles the complete requested decode window. This
        // is intentionally expensive: sampled observations cannot support a
        // high-confidence phase reconciliation.
        let readback_before = runtime.readback_bytes();
        let warmup_started = Instant::now();
        let warmup_before = runtime.runtime_telemetry();
        let mut measured_started = (warmup_decode_tokens == 0).then(Instant::now);
        let mut measured_before = (warmup_decode_tokens == 0).then_some(warmup_before);
        let mut warmup_end = (warmup_decode_tokens == 0).then_some(warmup_before);
        let complete_decode_started = Instant::now();
        let mut generated_token_ids = vec![selected];
        let mut first_eos_position = (selected == self.model.config.eos_token_id).then_some(1);
        let mut samples = Vec::with_capacity(total_decode_tokens.saturating_sub(1));
        for decode_position in 1..total_decode_tokens {
            let context_position = self.position;
            if decode_position == warmup_decode_tokens {
                let now = Instant::now();
                warmup_end = Some(runtime.runtime_telemetry());
                measured_started = Some(now);
                measured_before = warmup_end;
            }
            let (next, timings) = self.forward_token_inner(selected, exact_profile)?;
            selected = next;
            generated_token_ids.push(selected);
            if first_eos_position.is_none() && selected == self.model.config.eos_token_id {
                first_eos_position = Some(generated_token_ids.len());
            }
            {
                let mut kernels: BTreeMap<
                    (Option<u64>, Option<u32>, &'static str, &'static str),
                    Gemma4DecodeKernelProfile,
                > = BTreeMap::new();
                for timing in timings {
                    let family = gemma4_profile_family(timing.profiling_label, timing.kernel);
                    let entry = kernels
                        .entry((
                            timing.command_buffer_id,
                            timing.layer_index,
                            family,
                            timing.kernel,
                        ))
                        .or_insert(Gemma4DecodeKernelProfile {
                            family,
                            kernel_name: timing.kernel,
                            layer_index: timing.layer_index,
                            command_buffer_id: timing.command_buffer_id,
                            dispatches: 0,
                            gpu_nanos: 0,
                            cpu_encode_nanos: 0,
                            threadgroups: 0,
                            threads: 0,
                            bytes_read_estimate: 0,
                            bytes_written_estimate: 0,
                        });
                    entry.dispatches += 1;
                    entry.gpu_nanos += timing.timing.gpu_time.unwrap_or_default().as_nanos();
                    entry.cpu_encode_nanos += timing.cpu_encode.as_nanos();
                    entry.threadgroups += timing.threadgroups as u64;
                    entry.threads += timing.threads as u64;
                    entry.bytes_read_estimate += timing.bytes_read_estimate;
                    entry.bytes_written_estimate += timing.bytes_written_estimate;
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
                    scope: if decode_position < warmup_decode_tokens {
                        "decode_warmup"
                    } else {
                        "decode_measured"
                    },
                    kernels: kernels.into_values().collect(),
                });
            }
        }
        let warmup_telemetry = warmup_end
            .unwrap_or(warmup_before)
            .delta_from(warmup_before);
        let measured_telemetry = runtime
            .runtime_telemetry()
            .delta_from(measured_before.unwrap_or(warmup_before));
        let complete_decode_telemetry = runtime.runtime_telemetry().delta_from(warmup_before);
        Ok(Gemma4DecodeProfile {
            prompt_tokens: prompt_ids.len(),
            requested_decode_tokens: measured_decode_tokens,
            warmup_decode_tokens,
            measured_decode_tokens,
            completed_decode_tokens: generated_token_ids.len(),
            generated_token_ids,
            first_eos_position,
            prefill,
            warmup_decode: measured_started.map_or_else(
                || warmup_started.elapsed(),
                |started| started.duration_since(warmup_started),
            ),
            measured_decode: measured_started.map_or(Duration::ZERO, |started| started.elapsed()),
            decode: complete_decode_started.elapsed(),
            host_wall_time: started.elapsed(),
            prefill_telemetry,
            warmup_telemetry,
            measured_telemetry,
            decode_telemetry: complete_decode_telemetry,
            complete_decode_telemetry,
            prefill_kernels: aggregate_profile_timings(prefill_timings),
            prefill_path: gemma4_prefill_path(prompt_ids.len()),
            attention_kernel: gemma4_attention_kernel(
                self.kv_cache_type,
                self.model.config.attention_heads,
                self.model.config.key_length,
                false,
            ),
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
        self.generate_greedy_stream_inner(prompt, max_new_tokens, cancelled, false, false, 0, emit)
    }

    pub fn generate_greedy_chat_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        self.generate_greedy_stream_inner(prompt, max_new_tokens, cancelled, true, false, 0, emit)
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
        self.generate_greedy_fixed_benchmark_window_stream(
            prompt,
            0,
            decode_tokens,
            cancelled,
            emit,
        )
    }

    /// Run a fixed Resident decode window after deterministic warm-up tokens.
    ///
    /// The warm-up selections grow the same KV cache used by normal generation,
    /// but are excluded from `metrics.decode` and `decode_command_buffers`.
    /// This makes long-context throughput comparable without changing the
    /// user-facing chat path.
    pub fn generate_greedy_fixed_benchmark_window_stream(
        &mut self,
        prompt: &str,
        warmup_decode_tokens: usize,
        measured_decode_tokens: usize,
        cancelled: &AtomicBool,
        emit: impl FnMut(Gemma4TokenEvent) -> Result<()>,
    ) -> Result<Gemma4Generation> {
        ensure!(
            measured_decode_tokens > 0,
            "Gemma benchmark measurement window must be positive"
        );
        let total_decode_tokens = warmup_decode_tokens
            .checked_add(measured_decode_tokens)
            .context("Gemma benchmark decode window overflows")?;
        self.generate_greedy_stream_inner(
            prompt,
            total_decode_tokens,
            cancelled,
            false,
            true,
            warmup_decode_tokens,
            emit,
        )
    }

    fn generate_greedy_stream_inner(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        stop_on_end_turn: bool,
        continue_after_eos: bool,
        decode_window_start: usize,
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
        let dispatch_before = runtime.dispatch_count();
        let upload_before = runtime.uploaded_bytes();
        let allocation_before = runtime.runtime_telemetry().buffer_allocations;
        let gpu_before = runtime.gpu_execution_time();
        let readback_before = runtime.readback_bytes();
        let started = Instant::now();
        self.position = 0;
        let prefill_started = Instant::now();
        let prefill_before = runtime.runtime_telemetry();
        let plan = Gemma4PrefillPlan::new(prompt_ids.len(), self.max_context)?;
        let prefill_path = gemma4_prefill_path(prompt_ids.len());
        let mut selected = 0;
        let chunk_count = plan.chunks;
        for (chunk_index, chunk) in prompt_ids.chunks(plan.chunk_size).enumerate() {
            if let Some(token) = self.forward_tokens(chunk, chunk_index + 1 == chunk_count)? {
                selected = token;
            }
        }
        let prefill = prefill_started.elapsed();
        let prefill_end_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let prefill_scope = Gemma4ScopeMetrics {
            wall_time: prefill,
            host_start_ns: 0,
            host_end_ns: prefill_end_ns,
            telemetry: runtime.runtime_telemetry().delta_from(prefill_before),
            completed_tokens: prompt_ids.len(),
        };
        let prefill_commands = runtime.command_buffer_count() - command_before;
        let warmup_started = Instant::now();
        let mut measured_started = (decode_window_start == 0).then(Instant::now);
        let warmup_before = runtime.runtime_telemetry();
        let mut measured_before = (decode_window_start == 0).then_some(warmup_before);
        let mut warmup_end = (decode_window_start == 0).then_some(warmup_before);
        let mut measured_start_ns = (decode_window_start == 0).then_some(prefill_end_ns);
        let decode_complete_started = Instant::now();
        let mut decode_command_before = runtime.command_buffer_count();
        let complete_decode_before = runtime.runtime_telemetry();
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
                if index + 1 == decode_window_start {
                    let now = Instant::now();
                    measured_started = Some(now);
                    warmup_end = Some(runtime.runtime_telemetry());
                    measured_before = warmup_end;
                    measured_start_ns =
                        Some(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                    decode_command_before = runtime.command_buffer_count();
                }
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
        let complete_decode_scope = Gemma4ScopeMetrics {
            wall_time: decode_complete_started.elapsed(),
            host_start_ns: prefill_end_ns,
            host_end_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            telemetry: runtime
                .runtime_telemetry()
                .delta_from(complete_decode_before),
            completed_tokens: generated.len(),
        };
        let measured_scope = Gemma4ScopeMetrics {
            wall_time: measured_started.map_or(Duration::ZERO, |started| started.elapsed()),
            host_start_ns: measured_start_ns.unwrap_or(prefill_end_ns),
            host_end_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            telemetry: runtime
                .runtime_telemetry()
                .delta_from(measured_before.unwrap_or(warmup_before)),
            completed_tokens: generated.len().saturating_sub(decode_window_start),
        };
        let warmup_scope = Gemma4ScopeMetrics {
            wall_time: measured_started.map_or_else(
                || warmup_started.elapsed(),
                |started| started.duration_since(warmup_started),
            ),
            host_start_ns: prefill_end_ns,
            host_end_ns: measured_start_ns.unwrap_or(prefill_end_ns),
            telemetry: warmup_end
                .unwrap_or(warmup_before)
                .delta_from(warmup_before),
            completed_tokens: generated.len().min(decode_window_start),
        };
        let completed_decode_tokens = generated.len();
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
                upload_bytes: runtime.uploaded_bytes() - upload_before,
                readback_bytes: runtime.readback_bytes() - readback_before,
                command_buffers: runtime.command_buffer_count() - command_before,
                dispatches: runtime.dispatch_count() - dispatch_before,
                buffer_allocations: runtime.runtime_telemetry().buffer_allocations
                    - allocation_before,
                peak_resident_bytes: runtime.runtime_telemetry().peak_resident_bytes,
                gpu_execution_time: runtime.gpu_execution_time().saturating_sub(gpu_before),
                prefill_command_buffers: prefill_commands,
                decode_command_buffers: runtime.command_buffer_count() - decode_command_before,
                prefill,
                decode: measured_started
                    .expect("decode window starts before the first measured decode dispatch")
                    .elapsed(),
                host_wall_time: started.elapsed(),
                prefill_path,
                prefill_chunk_size: plan.chunk_size,
                prefill_chunks: plan.chunks,
                quantization_preflight_state: self.quantization_preflight_state,
                selected_group_formats: self.selected_group_formats.iter().cloned().collect(),
                quantization_rejections: self.quantization_rejections.clone(),
                attention_kernel: gemma4_attention_kernel(
                    self.kv_cache_type,
                    self.model.config.attention_heads,
                    self.model.config.key_length,
                    false,
                ),
                weight_format: self.weight_format,
                embedding_kernel: self.weight_format.embedding_kernel(),
                output_projection_kernel: match self.weight_format.output_format() {
                    GgufTensorType::Q6K => gemma4_q6_projection_kernel(),
                    GgufTensorType::Q4_0 => self.weight_format.output_projection_kernel(),
                    _ => unreachable!("unsupported Gemma vocabulary output format"),
                },
                q6_projection_kernel: match self.weight_format.output_format() {
                    GgufTensorType::Q6K => gemma4_q6_projection_kernel(),
                    GgufTensorType::Q4_0 => "none",
                    _ => unreachable!("unsupported Gemma vocabulary output format"),
                },
                q4_projection_kernel: gemma4_q4_projection_kernel(),
                q4_qkv_projection_kernel: gemma4_q4_qkv_projection_kernel(),
                q4_gate_up_projection_kernel: gemma4_q4_gate_up_projection_kernel(),
                ffn_gate_up_activation_kernel: "gelu_multiply_f32",
                ffn_gate_up_scratch_bytes: self.up.bytes() as u64,
                ple_composition_kernel: "ple_gelu_multiply_offset_f32",
                rms_epilogue_kernel: "gemma4_rms_residual_f32",
                q4_batch_projection_kernel: gemma4_q4_batch_projection_kernel(),
                ffn_down_projection_kernel: gemma4_ffn_down_projection_kernel(),
                ple_projection_kernel: gemma4_q4_projection_kernel(),
                rms_norm_kernel: gemma4_rms_norm_decode_kernel(self.model.config.hidden_size),
                rms_fused_projection_kernel: "rms_input_matvec_fused",
                kv_append_kernel: gemma4_kv_append_vnorm_kernel(self.kv_cache_type),
                kv_cache_type: self.kv_cache_type,
                kv_cache_bytes: self.kv_cache_bytes(),
                quantization_plan_path: self.quantization_plan_path.clone(),
                warmup_decode_tokens: decode_window_start,
                measured_decode_tokens: measured_scope.completed_tokens,
                completed_decode_tokens,
                warmup_scope,
                measured_scope,
                complete_decode_scope,
                prefill_scope,
                physical_command_buffer_overlap: decode_window_start > 0,
                physical_command_buffer_overlap_reason: (decode_window_start > 0).then(|| {
                    "first generated token is selected by the prefill command buffer".into()
                }),
            },
            finish_reason,
            first_eos_position,
        })
    }

    fn forward_token(&mut self, token: u32) -> Result<u32> {
        Ok(self.forward_token_inner(token, false)?.0)
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
        let mut command = runtime.begin_resident_command_with_exact_timing(exact_profile)?;
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
        Ok(self
            .forward_tokens_with_profile(tokens, select_last, false)?
            .0)
    }

    fn forward_tokens_with_profile(
        &mut self,
        tokens: &[u32],
        select_last: bool,
        exact_profile: bool,
    ) -> Result<(Option<u32>, Vec<atlas_metal::ResidentKernelTiming>)> {
        ensure!(!tokens.is_empty(), "Gemma token batch must not be empty");
        ensure!(
            self.position + tokens.len() <= self.max_context,
            "Gemma executor context exhausted"
        );
        if gemma4_prefill_path(tokens.len()) == "resident_layer_major" {
            return self.forward_tokens_layer_major(tokens, select_last, exact_profile);
        }
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
        let mut command = runtime.begin_resident_command_with_exact_timing(exact_profile)?;
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
        let timings = command.take_kernel_timings();
        command.finish()?;
        let selected = select_last
            .then(|| runtime.read_u32(&self.selected))
            .transpose()?;
        Ok((selected, timings))
    }

    fn forward_tokens_layer_major(
        &mut self,
        tokens: &[u32],
        select_last: bool,
        exact_profile: bool,
    ) -> Result<(Option<u32>, Vec<atlas_metal::ResidentKernelTiming>)> {
        ensure!(
            tokens.len() <= GEMMA4_PREFILL_BATCH_CAPACITY,
            "Gemma layer-major prefill batch exceeds capacity"
        );
        let runtime = self.model.runtime();
        let c = &self.model.config;
        let batch_value = tokens.len();
        let batch = runtime.upload_u32(&[u32::try_from(batch_value)?])?;
        let token_batch = runtime.upload_u32(tokens)?;
        let positions = (self.position..self.position + batch_value)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let position_batch = runtime.upload_u32(&positions)?;
        let rope_pairs = c.key_length.max(c.key_length_swa) / 2;
        let mut full_cos = vec![0.0; batch_value * rope_pairs];
        let mut full_sin = vec![0.0; batch_value * rope_pairs];
        let mut swa_cos = vec![0.0; batch_value * rope_pairs];
        let mut swa_sin = vec![0.0; batch_value * rope_pairs];
        for (token, position) in (self.position..self.position + batch_value).enumerate() {
            for pair in 0..rope_pairs {
                let full_angle = gemma4_rope_angle(
                    position,
                    pair,
                    c.rope_dimensions,
                    c.rope_theta,
                    self.rope_freq_factors[pair],
                );
                let swa_angle =
                    gemma4_rope_angle(position, pair, c.rope_dimensions_swa, c.rope_theta_swa, 1.0);
                full_cos[token * rope_pairs + pair] = full_angle.cos();
                full_sin[token * rope_pairs + pair] = full_angle.sin();
                swa_cos[token * rope_pairs + pair] = swa_angle.cos();
                swa_sin[token * rope_pairs + pair] = swa_angle.sin();
            }
        }
        let full_cos = runtime.upload_f32(&full_cos)?;
        let full_sin = runtime.upload_f32(&full_sin)?;
        let swa_cos = runtime.upload_f32(&swa_cos)?;
        let swa_sin = runtime.upload_f32(&swa_sin)?;
        let key_counts = runtime.upload_u32(&gemma4_attention_key_count_table(
            self.position,
            batch_value,
            &c.sliding_pattern,
            c.sliding_window,
        )?)?;
        let h = c.hidden_size;
        let ple_total = c.layers * c.per_layer_embedding_size;
        let h_batch = runtime.upload_u32(&[u32::try_from(batch_value * h)?])?;
        let ple_batch = runtime.upload_u32(&[u32::try_from(batch_value * ple_total)?])?;
        let mut command = runtime.begin_resident_command_with_exact_timing(exact_profile)?;
        let per_layer_proj = self.weight("per_layer_model_proj.weight", GgufTensorType::F16)?;
        let per_layer_norm = self.weight("per_layer_proj_norm.weight", GgufTensorType::F32)?;
        command.dispatch_1d(
            self.weight_format.embedding_kernel(),
            &[
                &self.output_projection,
                &token_batch,
                &self.prefill.state,
                &self.vocab,
                &self.hidden,
                &batch,
            ],
            batch_value * h,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.prefill.state,
                &self.prefill.state,
                &self.embed_scale,
                &h_batch,
            ],
            batch_value * h,
        )?;
        command.dispatch_1d(
            self.weight_format.embedding_kernel(),
            &[
                &self.per_layer_embedding,
                &token_batch,
                &self.prefill.ple_lookup,
                &self.vocab,
                &self.ple_total,
                &batch,
            ],
            batch_value * ple_total,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.prefill.ple_lookup,
                &self.prefill.ple_lookup,
                &self.ple_embedding_scale,
                &ple_batch,
            ],
            batch_value * ple_total,
        )?;
        self.matmul_batch(
            &mut command,
            &self.prefill.state,
            &per_layer_proj,
            &self.prefill.ple_projected,
            &self.hidden,
            &self.ple_total,
            &batch,
            ple_total,
            batch_value,
            GgufTensorType::F16,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.prefill.ple_projected,
                &self.prefill.ple_projected,
                &self.ple_projection_scale,
                &ple_batch,
            ],
            batch_value * ple_total,
        )?;
        for token in 0..batch_value {
            let offset = token * ple_total * std::mem::size_of::<f32>();
            command.dispatch_1d_at(
                "rms_norm_groups_in_place_stable_f32",
                &[
                    (&self.prefill.ple_projected, offset),
                    (&per_layer_norm, 0),
                    (&self.ple_width, 0),
                    (&self.layers, 0),
                    (&self.epsilon, 0),
                ],
                c.layers,
            )?;
        }
        command.dispatch_1d(
            "vector_add_f32",
            &[
                &self.prefill.ple_lookup,
                &self.prefill.ple_projected,
                &self.prefill.ple,
                &ple_batch,
            ],
            batch_value * ple_total,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[
                &self.prefill.ple,
                &self.prefill.ple,
                &self.ple_input_scale,
                &ple_batch,
            ],
            batch_value * ple_total,
        )?;

        for layer in 0..c.layers {
            self.encode_prefill_layer_major_layer(
                &mut command,
                layer,
                batch_value,
                &batch,
                &position_batch,
                &full_cos,
                &full_sin,
                &swa_cos,
                &swa_sin,
                rope_pairs,
                &key_counts,
            )?;
        }
        self.position += batch_value;
        if select_last {
            let state_offset = (batch_value - 1) * h * std::mem::size_of::<f32>();
            let output_norm = self.weight("output_norm.weight", GgufTensorType::F32)?;
            command.dispatch_threadgroups_1d_at(
                "rms_norm_decode_f32",
                &[
                    (&self.prefill.state, state_offset),
                    (&output_norm, 0),
                    (&self.norm, 0),
                    (&self.hidden, 0),
                    (&self.epsilon, 0),
                ],
                1,
                32,
            )?;
            self.matvec(
                &mut command,
                &self.norm,
                &self.token_embedding,
                &self.logits,
                &self.hidden,
                &self.vocab,
                c.vocab_size,
                self.weight_format.output_format(),
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
        }
        let timings = command.take_kernel_timings();
        command.finish()?;
        let selected = select_last
            .then(|| runtime.read_u32(&self.selected))
            .transpose()?;
        Ok((selected, timings))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_prefill_layer_major_layer(
        &self,
        command: &mut atlas_metal::ResidentCommand<'_>,
        layer: usize,
        batch_value: usize,
        batch: &GpuBuffer,
        positions: &GpuBuffer,
        full_cos: &GpuBuffer,
        full_sin: &GpuBuffer,
        swa_cos: &GpuBuffer,
        swa_sin: &GpuBuffer,
        rope_pairs: usize,
        key_counts: &GpuBuffer,
    ) -> Result<()> {
        command.set_layer_index(Some(layer as u32));
        let runtime = self.model.runtime();
        let c = &self.model.config;
        let h = c.hidden_size;
        let h_batch = runtime.upload_u32(&[u32::try_from(batch_value * h)?])?;
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
        let head_width = if sliding {
            &self.head_swa
        } else {
            &self.head_full
        };
        let source = self.kv_sources[layer];
        let attn_norm = self.weight(&format!("{p}.attn_norm.weight"), GgufTensorType::F32)?;
        self.rms_norm_batch_decode_order(
            command,
            &self.prefill.state,
            &attn_norm,
            &self.prefill.norm,
            batch_value,
        )?;
        let wq = self.weight(&format!("{p}.attn_q.weight"), GgufTensorType::Q4_0)?;
        self.matmul_batch(
            command,
            &self.prefill.norm,
            &wq,
            &self.prefill.q,
            &self.hidden,
            q_width_buffer,
            batch,
            q_width,
            batch_value,
            GgufTensorType::Q4_0,
        )?;
        let q_norm = self.weight(&format!("{p}.attn_q_norm.weight"), GgufTensorType::F32)?;
        let (wk, wv, k_norm) = if source == layer {
            (
                Some(self.weight(&format!("{p}.attn_k.weight"), GgufTensorType::Q4_0)?),
                Some(self.weight(&format!("{p}.attn_v.weight"), GgufTensorType::Q4_0)?),
                Some(self.weight(&format!("{p}.attn_k_norm.weight"), GgufTensorType::F32)?),
            )
        } else {
            (None, None, None)
        };
        if source == layer {
            self.matmul_batch(
                command,
                &self.prefill.norm,
                wk.as_ref().expect("provider K weight"),
                &self.prefill.k,
                &self.hidden,
                head_width,
                batch,
                head,
                batch_value,
                GgufTensorType::Q4_0,
            )?;
            self.matmul_batch(
                command,
                &self.prefill.norm,
                wv.as_ref().expect("provider V weight"),
                &self.prefill.v,
                &self.hidden,
                head_width,
                batch,
                head,
                batch_value,
                GgufTensorType::Q4_0,
            )?;
        }
        let (cos, sin) = if sliding {
            (swa_cos, swa_sin)
        } else {
            (full_cos, full_sin)
        };
        for token in 0..batch_value {
            let q_offset = token * q_width * std::mem::size_of::<f32>();
            let k_offset = token * head * std::mem::size_of::<f32>();
            let rope_offset = token * rope_pairs * std::mem::size_of::<f32>();
            command.dispatch_threadgroups_1d_at(
                "gemma4_qk_norm_rope_fused_f32",
                &[
                    (&self.prefill.q, q_offset),
                    (&self.prefill.k, k_offset),
                    (&q_norm, 0),
                    (k_norm.as_ref().unwrap_or(&q_norm), 0),
                    (cos, rope_offset),
                    (sin, rope_offset),
                    (&self.prefill.q_rot, q_offset),
                    (&self.prefill.k_rot, k_offset),
                    (head_width, 0),
                    (&self.heads, 0),
                    (&self.one, 0),
                    (&self.epsilon, 0),
                ],
                c.attention_heads + usize::from(source == layer),
                1,
            )?;
            if source == layer {
                command.dispatch_1d_at(
                    "rms_norm_groups_in_place_unweighted_f32",
                    &[
                        (&self.prefill.v, k_offset),
                        (head_width, 0),
                        (&self.one, 0),
                        (&self.epsilon, 0),
                    ],
                    1,
                )?;
                let cache = self.kv[layer].as_ref().expect("KV provider has cache");
                let append_count = if self.kv_cache_type == Gemma4KvCacheType::F32 {
                    head
                } else {
                    head / 32
                };
                command.dispatch_1d_at(
                    gemma4_kv_append_kernel(self.kv_cache_type),
                    &[
                        (&self.prefill.k_rot, k_offset),
                        (&self.prefill.v, k_offset),
                        (cache, 0),
                        (head_width, 0),
                        (&self.capacity, 0),
                        (positions, token * std::mem::size_of::<u32>()),
                    ],
                    append_count,
                )?;
            }
        }
        let cache = self.kv[source].as_ref().expect("Gemma KV source has cache");
        let (attention_kernel, attention_threads) = gemma4_attention_binding(self.kv_cache_type);
        for token in 0..batch_value {
            let q_offset = token * q_width * std::mem::size_of::<f32>();
            let controls_offset = (token * c.layers + layer) * std::mem::size_of::<u32>();
            command.dispatch_threadgroups_1d_at(
                attention_kernel,
                &[
                    (&self.prefill.q_rot, q_offset),
                    (cache, 0),
                    (&self.prefill.attention, q_offset),
                    (&self.heads, 0),
                    (&self.kv_heads, 0),
                    (head_width, 0),
                    (&self.capacity, 0),
                    (key_counts, controls_offset),
                ],
                c.attention_heads,
                usize::try_from(attention_threads).expect("attention threadgroup size"),
            )?;
        }
        let wo = self.weight(&format!("{p}.attn_output.weight"), GgufTensorType::Q4_0)?;
        self.matmul_batch(
            command,
            &self.prefill.attention,
            &wo,
            &self.prefill.work,
            q_width_buffer,
            &self.hidden,
            batch,
            h,
            batch_value,
            GgufTensorType::Q4_0,
        )?;
        let post_attn = self.weight(
            &format!("{p}.post_attention_norm.weight"),
            GgufTensorType::F32,
        )?;
        self.rms_norm_batch_decode_order(
            command,
            &self.prefill.work,
            &post_attn,
            &self.prefill.work,
            batch_value,
        )?;
        command.dispatch_1d(
            "vector_add_f32",
            &[
                &self.prefill.state,
                &self.prefill.work,
                &self.prefill.residual,
                &h_batch,
            ],
            batch_value * h,
        )?;
        let ffn_norm = self.weight(&format!("{p}.ffn_norm.weight"), GgufTensorType::F32)?;
        self.rms_norm_batch_decode_order(
            command,
            &self.prefill.residual,
            &ffn_norm,
            &self.prefill.norm,
            batch_value,
        )?;
        let ffn = c.feed_forward_sizes[layer];
        let ffn_buffer = &self.ffn_widths[layer];
        let ffn_batch = runtime.upload_u32(&[u32::try_from(batch_value * ffn)?])?;
        let gate = self.weight(&format!("{p}.ffn_gate.weight"), GgufTensorType::Q4_0)?;
        let up = self.weight(&format!("{p}.ffn_up.weight"), GgufTensorType::Q4_0)?;
        let down = self.weight(&format!("{p}.ffn_down.weight"), GgufTensorType::Q4_0)?;
        self.matmul_batch(
            command,
            &self.prefill.norm,
            &gate,
            &self.prefill.gate,
            &self.hidden,
            ffn_buffer,
            batch,
            ffn,
            batch_value,
            GgufTensorType::Q4_0,
        )?;
        self.matmul_batch(
            command,
            &self.prefill.norm,
            &up,
            &self.prefill.up,
            &self.hidden,
            ffn_buffer,
            batch,
            ffn,
            batch_value,
            GgufTensorType::Q4_0,
        )?;
        command.dispatch_1d(
            "gelu_f32",
            &[&self.prefill.gate, &self.prefill.activated, &ffn_batch],
            batch_value * ffn,
        )?;
        command.dispatch_1d(
            "vector_multiply_f32",
            &[
                &self.prefill.activated,
                &self.prefill.up,
                &self.prefill.product,
                &ffn_batch,
            ],
            batch_value * ffn,
        )?;
        self.matmul_ffn_down_batch(
            command,
            &self.prefill.product,
            &down,
            &self.prefill.work,
            ffn_buffer,
            &self.hidden,
            batch,
            h,
            batch_value,
        )?;
        let post_ffn = self.weight(&format!("{p}.post_ffw_norm.weight"), GgufTensorType::F32)?;
        self.rms_norm_batch_decode_order(
            command,
            &self.prefill.work,
            &post_ffn,
            &self.prefill.work,
            batch_value,
        )?;
        command.dispatch_1d(
            "vector_add_f32",
            &[
                &self.prefill.residual,
                &self.prefill.work,
                &self.prefill.state,
                &h_batch,
            ],
            batch_value * h,
        )?;
        let inp_gate = self.weight(&format!("{p}.inp_gate.weight"), GgufTensorType::Q4_0)?;
        let projection = self.weight(&format!("{p}.proj.weight"), GgufTensorType::Q4_0)?;
        let post_norm = self.weight(&format!("{p}.post_norm.weight"), GgufTensorType::F32)?;
        let ple_batch =
            runtime.upload_u32(&[u32::try_from(batch_value * c.per_layer_embedding_size)?])?;
        self.matmul_batch(
            command,
            &self.prefill.state,
            &inp_gate,
            &self.prefill.gate,
            &self.hidden,
            &self.ple_width,
            batch,
            c.per_layer_embedding_size,
            batch_value,
            GgufTensorType::Q4_0,
        )?;
        command.dispatch_1d(
            "gelu_f32",
            &[&self.prefill.gate, &self.prefill.gate, &ple_batch],
            batch_value * c.per_layer_embedding_size,
        )?;
        let ple_offset = &self.ple_offsets[layer];
        for token in 0..batch_value {
            let offset = token * c.per_layer_embedding_size * std::mem::size_of::<f32>();
            let source_offset =
                token * c.layers * c.per_layer_embedding_size * std::mem::size_of::<f32>();
            command.dispatch_1d_at(
                "vector_multiply_offset_f32",
                &[
                    (&self.prefill.gate, offset),
                    (&self.prefill.ple, source_offset),
                    (&self.prefill.activated, offset),
                    (ple_offset, 0),
                    (&self.ple_width, 0),
                ],
                c.per_layer_embedding_size,
            )?;
        }
        self.matmul_batch(
            command,
            &self.prefill.activated,
            &projection,
            &self.prefill.work,
            &self.ple_width,
            &self.hidden,
            batch,
            h,
            batch_value,
            GgufTensorType::Q4_0,
        )?;
        self.rms_norm_batch_decode_order(
            command,
            &self.prefill.work,
            &post_norm,
            &self.prefill.work,
            batch_value,
        )?;
        command.dispatch_1d(
            "vector_add_f32",
            &[
                &self.prefill.state,
                &self.prefill.work,
                &self.prefill.state,
                &h_batch,
            ],
            batch_value * h,
        )?;
        let scale = self.weight(
            &format!("{p}.layer_output_scale.weight"),
            GgufTensorType::F32,
        )?;
        command.dispatch_1d(
            "scalar_multiply_f32",
            &[&self.prefill.state, &self.prefill.state, &scale, &h_batch],
            batch_value * h,
        )?;
        Ok(())
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
        command.set_layer_index(None);
        ensure!(
            self.position < self.max_context,
            "Gemma executor context exhausted"
        );
        let c = &self.model.config;
        let h = c.hidden_size;
        let ple_total = c.layers * c.per_layer_embedding_size;
        let per_layer_proj = self.weight("per_layer_model_proj.weight", GgufTensorType::F16)?;
        let per_layer_norm = self.weight("per_layer_proj_norm.weight", GgufTensorType::F32)?;
        command.dispatch_1d(
            self.weight_format.embedding_kernel(),
            &[
                &self.token_embedding,
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
        let ple_total_buffer = &self.ple_total;
        command.dispatch_1d(
            self.weight_format.embedding_kernel(),
            &[
                &self.per_layer_embedding,
                &self.token,
                &self.ple_lookup,
                &self.vocab,
                &ple_total_buffer,
                &self.one,
            ],
            ple_total,
        )?;
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
        self.matvec_labeled(
            &mut command,
            Some("ple_projection"),
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
        command.dispatch_1d_labeled(
            "rms_norm_groups_in_place_stable_f32",
            Some("ple_norm"),
            &[
                &self.ple_projected,
                &per_layer_norm,
                &self.ple_width,
                &self.layers,
                &self.epsilon,
            ],
            ple_total,
        )?;
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
        for layer in 0..c.layers {
            command.set_layer_index(Some(layer as u32));
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
            let source = self.kv_sources[layer];
            let attn_norm = self.weight(&format!("{p}.attn_norm.weight"), GgufTensorType::F32)?;
            let wq = self.weight(&format!("{p}.attn_q.weight"), GgufTensorType::Q4_0)?;
            let q_norm = self.weight(&format!("{p}.attn_q_norm.weight"), GgufTensorType::F32)?;
            let fused_qkv = source == layer;
            let fused_qk_norm_rope = true;
            let (wk, wv, k_norm) = if source == layer {
                (
                    Some(self.weight(&format!("{p}.attn_k.weight"), GgufTensorType::Q4_0)?),
                    Some(self.weight(&format!("{p}.attn_v.weight"), GgufTensorType::Q4_0)?),
                    Some(self.weight(&format!("{p}.attn_k_norm.weight"), GgufTensorType::F32)?),
                )
            } else {
                (None, None, None)
            };
            if fused_qkv {
                self.matmul_q4_0_qkv(
                    &mut command,
                    &self.state,
                    &wq,
                    wk.as_ref().expect("provider layer has K weight"),
                    wv.as_ref().expect("provider layer has V weight"),
                    &self.q,
                    &self.k,
                    &self.v,
                    q_width_buffer,
                    head_width,
                    q_width,
                    head,
                    Some(&attn_norm),
                )?;
            }
            if fused_qk_norm_rope {
                self.gemma4_qk_norm_rope_fused(
                    &mut command,
                    &q_norm,
                    k_norm.as_ref().unwrap_or(&q_norm),
                    cos,
                    sin,
                    rope_offset,
                    head_width,
                    source == layer,
                )?;
            }
            if source == layer {
                let cache = self.kv[layer].as_ref().expect("KV provider has cache");
                let append_count = if self.kv_cache_type == Gemma4KvCacheType::F32 {
                    head
                } else {
                    head / 32
                };
                command.dispatch_1d(
                    gemma4_kv_append_vnorm_kernel(self.kv_cache_type),
                    &[
                        &self.k_rot,
                        &self.v,
                        cache,
                        head_width,
                        &self.capacity,
                        &self.position_buffer,
                        &self.epsilon,
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
            let (attention_kernel, attention_threads) =
                gemma4_attention_binding(self.kv_cache_type);
            let flash16_binding = if self.kv_cache_type == Gemma4KvCacheType::Q4_0
                && gemma4_q4_flash16_supported(c.attention_heads, head)
            {
                Some(gemma4_q4_flash16_binding(sliding))
            } else {
                None
            };
            if let Some((attention_kernel, attention_label, attention_threads)) = flash16_binding {
                command.dispatch_threadgroups_1d_at_labeled(
                    attention_kernel,
                    Some(attention_label),
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
                    usize::try_from(attention_threads).expect("attention threadgroup size"),
                )?;
            } else {
                command.dispatch_threadgroups_1d_at(
                    attention_kernel,
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
                    attention_threads,
                )?;
            }
            let wo = self.weight(&format!("{p}.attn_output.weight"), GgufTensorType::Q4_0)?;
            self.matvec_labeled(
                &mut command,
                Some("attention_output_projection"),
                &self.attention,
                &wo,
                &self.work,
                q_width_buffer,
                &self.hidden,
                h,
                GgufTensorType::Q4_0,
            )?;
            let post_attn = self.weight(
                &format!("{p}.post_attention_norm.weight"),
                GgufTensorType::F32,
            )?;
            command.dispatch_threadgroups_1d_labeled(
                "gemma4_rms_residual_f32",
                Some("post_attention_norm_residual"),
                &[
                    &self.work,
                    &post_attn,
                    &self.state,
                    &self.work,
                    &self.residual,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            let ffn_norm = self.weight(&format!("{p}.ffn_norm.weight"), GgufTensorType::F32)?;
            let ffn = c.feed_forward_sizes[layer];
            let ffn_buffer = &self.ffn_widths[layer];
            let gate = self.weight(&format!("{p}.ffn_gate.weight"), GgufTensorType::Q4_0)?;
            let up = self.weight(&format!("{p}.ffn_up.weight"), GgufTensorType::Q4_0)?;
            let down = self.weight(&format!("{p}.ffn_down.weight"), GgufTensorType::Q4_0)?;
            let up_buffer = &self.up;
            self.matmul_q4_0_gate_up(
                &mut command,
                &self.residual,
                &gate,
                &up,
                &self.gate,
                up_buffer,
                ffn_buffer,
                ffn,
                Some(&ffn_norm),
            )?;
            // Keep GELU out-of-place: the fused Gate/Up kernel has produced
            // `product` without materializing FFN intermediates.
            command.dispatch_1d(
                "gelu_multiply_f32",
                &[&self.gate, up_buffer, &self.product, ffn_buffer],
                ffn,
            )?;
            self.matvec_ffn_down(
                &mut command,
                &self.product,
                &down,
                &self.work,
                ffn_buffer,
                &self.hidden,
                h,
            )?;
            let post_ffn =
                self.weight(&format!("{p}.post_ffw_norm.weight"), GgufTensorType::F32)?;
            command.dispatch_threadgroups_1d_labeled(
                "gemma4_rms_residual_f32",
                Some("post_ffn_norm_residual"),
                &[
                    &self.work,
                    &post_ffn,
                    &self.residual,
                    &self.work,
                    &self.state,
                    &self.hidden,
                    &self.epsilon,
                ],
                1,
                32,
            )?;
            let inp_gate = self.weight(&format!("{p}.inp_gate.weight"), GgufTensorType::Q4_0)?;
            let projection = self.weight(&format!("{p}.proj.weight"), GgufTensorType::Q4_0)?;
            let post_norm = self.weight(&format!("{p}.post_norm.weight"), GgufTensorType::F32)?;
            self.matvec_labeled(
                &mut command,
                Some("ple_input_gate"),
                &self.state,
                &inp_gate,
                &self.gate,
                &self.hidden,
                &self.ple_width,
                c.per_layer_embedding_size,
                GgufTensorType::Q4_0,
            )?;
            // Current layer PLE is a contiguous [256] slice in the resident [layer][width] table.
            let ple_offset = &self.ple_offsets[layer];
            command.dispatch_1d_labeled(
                "ple_gelu_multiply_offset_f32",
                Some("ple_projection"),
                &[
                    &self.gate,
                    &self.ple,
                    &self.activated,
                    &ple_offset,
                    &self.ple_width,
                ],
                c.per_layer_embedding_size,
            )?;
            self.matvec_labeled(
                &mut command,
                Some("ple_projection"),
                &self.activated,
                &projection,
                &self.work,
                &self.ple_width,
                &self.hidden,
                h,
                GgufTensorType::Q4_0,
            )?;
            self.rms_norm_decode_labeled(
                &mut command,
                Some("ple_norm"),
                &self.work,
                &post_norm,
                &self.work,
            )?;
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
        }
        command.set_layer_index(None);
        if !select_output {
            return Ok(());
        }
        let output_norm = self.weight("output_norm.weight", GgufTensorType::F32)?;
        self.matvec_rms_labeled(
            &mut command,
            Some("output_projection"),
            &self.state,
            &self.output_projection,
            &self.logits,
            &self.hidden,
            &self.vocab,
            c.vocab_size,
            self.weight_format.output_format(),
            &output_norm,
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::{
        GEMMA4_SELECTED_GROUP_QKV, GEMMA4_SELECTED_GROUP_VOCAB_EMBEDDINGS,
        GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT, Gemma4KvCacheType, Gemma4PrefillPlan,
        Gemma4WeightFormat, QuantizationPlan, gemma4_attention_key_count,
        gemma4_attention_key_count_table, gemma4_decode_profile_targets, gemma4_kernel_family,
        gemma4_prefill_path, gemma4_profile_family, gemma4_q6_k_to_q4_0,
        gemma4_rms_norm_decode_kernel, gemma4_rope_angle, gemma4_selected_group_formats,
        gemma4_should_finish, gemma4_weight_format_with_plan,
    };
    use atlas_core::{GgufTensorType, dequantize_block};

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
    fn layer_major_prefill_is_the_normal_multi_token_resident_path() {
        assert_eq!(gemma4_prefill_path(1), "resident_chunked_command");
        assert_eq!(gemma4_prefill_path(2), "resident_layer_major");
        assert_eq!(gemma4_prefill_path(128), "resident_layer_major");
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
    fn rms_norm_kernel_uses_vec4_only_for_128_aligned_hidden_sizes() {
        assert_eq!(
            gemma4_rms_norm_decode_kernel(2304),
            "rms_norm_decode_f32_vec4"
        );
        assert_eq!(gemma4_rms_norm_decode_kernel(2305), "rms_norm_decode_f32");
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
            gemma4_kernel_family("matvec_q4_0_64row_mv"),
            "q4_projection_other"
        );
        assert_eq!(
            gemma4_kernel_family("matvec_q6_k_64row_mv"),
            "q6_lm_head_projection"
        );
        assert_eq!(
            gemma4_kernel_family("matmul_q4_0_batch_16row"),
            "batched_projection"
        );
        assert_eq!(
            gemma4_kernel_family("matmul_q4_0_qkv_32row_mv_rms"),
            "q4_qkv_projection"
        );
        assert_eq!(
            gemma4_kernel_family("matmul_q4_0_gate_up_32row_mv_rms"),
            "q4_ffn_gate_up_projection"
        );
        assert_eq!(gemma4_kernel_family("rms_norm_decode_f32"), "rms_norm");
        assert_eq!(
            gemma4_kernel_family("gemma4_qk_norm_rope_fused_f32"),
            "qk_norm_rope_fused"
        );
        assert_eq!(gemma4_kernel_family("rope_f32"), "rope_rotation");
        assert_eq!(
            gemma4_kernel_family("rope_half_to_interleaved_f32"),
            "rope_layout"
        );
        assert_eq!(
            gemma4_kernel_family("attention_decode_fused_f32"),
            "gemma_attention"
        );
        assert_eq!(gemma4_kernel_family("argmax_f32"), "argmax");
        assert_eq!(
            gemma4_kernel_family("embedding_lookup_q6_k"),
            "embedding_lookup"
        );
        assert_eq!(gemma4_kernel_family("vector_add_f32"), "residual");
        assert_eq!(gemma4_kernel_family("gelu_f32"), "ffn_activation");
    }

    #[test]
    fn ready_plan_selects_vocabulary_format_as_a_pair() {
        let mut plan = QuantizationPlan::new("gemma", "sha");
        plan.state = "ready".into();
        plan.hardware_identity = "Apple GPU 42".into();
        plan.profiler_configuration = crate::quantization_plan::QuantizationPlanProfilerConfig {
            mode: "auto".into(),
            prompt_sha256: "prompt-sha".into(),
            decode_tokens: 32,
            runs: 2,
        };
        plan.tensors.insert(
            "token_embd.weight".into(),
            crate::quantization_plan::QuantizationPlanTensor {
                group_members: vec![
                    "token_embd.weight".into(),
                    "per_layer_token_embd.weight".into(),
                ],
                source_format: GgufTensorType::Q6K,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: "embedding_lookup_q4_0".into(),
                max_abs_logit_delta: 0.0,
                median_gpu_ns: 1,
                baseline_gpu_ns: 1,
                resident_bytes: 1,
                upload_bytes: 1,
                parity_digest: "digest".into(),
                parity: true,
                rejection_reason: None,
            },
        );
        plan.tensors.insert(
            "per_layer_token_embd.weight".into(),
            crate::quantization_plan::QuantizationPlanTensor {
                group_members: vec![
                    "token_embd.weight".into(),
                    "per_layer_token_embd.weight".into(),
                ],
                source_format: GgufTensorType::Q6K,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: "embedding_lookup_q4_0".into(),
                max_abs_logit_delta: 0.0,
                median_gpu_ns: 1,
                baseline_gpu_ns: 1,
                resident_bytes: 1,
                upload_bytes: 1,
                parity_digest: "digest".into(),
                parity: true,
                rejection_reason: None,
            },
        );
        assert_eq!(
            gemma4_weight_format_with_plan(Some(&plan)).unwrap(),
            Gemma4WeightFormat::AllQ4
        );
    }

    #[test]
    fn explicit_weight_format_override_bypasses_invalid_cached_plan() {
        struct EnvVarGuard {
            key: &'static str,
            previous: Option<OsString>,
        }

        impl EnvVarGuard {
            fn set(key: &'static str, value: &str) -> Self {
                let previous = std::env::var_os(key);
                unsafe { std::env::set_var(key, value) };
                Self { key, previous }
            }
        }

        impl Drop for EnvVarGuard {
            fn drop(&mut self) {
                unsafe {
                    match &self.previous {
                        Some(value) => std::env::set_var(self.key, value),
                        None => std::env::remove_var(self.key),
                    }
                }
            }
        }

        let _guard = EnvVarGuard::set("ATLAS_GEMMA4_WEIGHT_FORMAT", "all_q4");
        let mut plan = QuantizationPlan::new("gemma", "model-sha");
        plan.state = crate::quantization_plan::QUANTIZATION_PLAN_STATE_READY.into();
        plan.tensors.insert(
            "token_embd.weight".into(),
            crate::quantization_plan::QuantizationPlanTensor {
                group_members: vec![
                    "token_embd.weight".into(),
                    "per_layer_token_embd.weight".into(),
                ],
                source_format: GgufTensorType::Q6K,
                selected_format: GgufTensorType::Q6K,
                selected_kernel: "embedding_lookup_q6_k".into(),
                max_abs_logit_delta: 0.0,
                median_gpu_ns: 1,
                baseline_gpu_ns: 1,
                resident_bytes: 1,
                upload_bytes: 1,
                parity_digest: "digest".into(),
                parity: true,
                rejection_reason: None,
            },
        );
        plan.tensors.insert(
            "per_layer_token_embd.weight".into(),
            crate::quantization_plan::QuantizationPlanTensor {
                group_members: vec![
                    "token_embd.weight".into(),
                    "per_layer_token_embd.weight".into(),
                ],
                source_format: GgufTensorType::Q6K,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: "embedding_lookup_q4_0".into(),
                max_abs_logit_delta: 0.0,
                median_gpu_ns: 1,
                baseline_gpu_ns: 1,
                resident_bytes: 1,
                upload_bytes: 1,
                parity_digest: "digest".into(),
                parity: true,
                rejection_reason: None,
            },
        );

        assert_eq!(
            gemma4_weight_format_with_plan(Some(&plan)).unwrap(),
            Gemma4WeightFormat::AllQ4
        );
    }

    #[test]
    fn all_q4_vocabulary_conversion_preserves_packed_layout_and_bounded_oracle_error() {
        // A Q6_K block with every quantized value at -32, unit group scales,
        // and a unit super-block scale is easy to audit byte-for-byte.
        let mut q6 = vec![0u8; GgufTensorType::Q6K.block_bytes()];
        q6[192..208].fill(1);
        q6[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        let q4 = gemma4_q6_k_to_q4_0(&q6, 256).expect("convert one Q6_K row");
        assert_eq!(q4.len(), 8 * GgufTensorType::Q4_0.block_bytes());
        for block in q4.chunks_exact(GgufTensorType::Q4_0.block_bytes()) {
            assert_eq!(&block[..2], &0x4400u16.to_le_bytes());
            assert!(
                atlas_core::f16_bits_to_f32(u16::from_le_bytes(block[..2].try_into().unwrap()))
                    .is_finite()
            );
            assert!(block[2..].iter().all(|byte| *byte == 0));
        }
        let mut q6_values = vec![0.0; 256];
        dequantize_block(GgufTensorType::Q6K, &q6, &mut q6_values).unwrap();
        let mut q4_values = Vec::with_capacity(256);
        for block in q4.chunks_exact(GgufTensorType::Q4_0.block_bytes()) {
            let mut values = vec![0.0; 32];
            dequantize_block(GgufTensorType::Q4_0, block, &mut values).unwrap();
            q4_values.extend(values);
        }
        let max_error = q6_values
            .iter()
            .zip(&q4_values)
            .map(|(oracle, actual)| (oracle - actual).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error <= 4.0,
            "Q4 conversion error {max_error} exceeds one scale"
        );
    }

    #[test]
    fn split_q4_routes_embedding_and_tied_projection_independently() {
        let cases = [
            (
                Gemma4WeightFormat::Q4Embeddings,
                "embedding_lookup_q4_0",
                GgufTensorType::Q6K,
            ),
            (
                Gemma4WeightFormat::Q4LmHead,
                "embedding_lookup_q6_k",
                GgufTensorType::Q4_0,
            ),
            (
                Gemma4WeightFormat::AllQ4,
                "embedding_lookup_q4_0",
                GgufTensorType::Q4_0,
            ),
        ];
        for (format, embedding_kernel, output_format) in cases {
            assert_eq!(format.embedding_kernel(), embedding_kernel);
            assert_eq!(format.output_format(), output_format);
        }
    }

    #[test]
    fn selected_group_formats_keep_unsupported_groups_on_their_source_path() {
        let mixed = gemma4_selected_group_formats(Gemma4WeightFormat::MixedQ4Q6);
        assert_eq!(
            mixed.selected_format(GEMMA4_SELECTED_GROUP_VOCAB_EMBEDDINGS),
            Some(GgufTensorType::Q6K)
        );
        assert_eq!(
            mixed.selected_format(GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT),
            Some(GgufTensorType::Q6K)
        );
        assert_eq!(
            mixed.selected_format(GEMMA4_SELECTED_GROUP_QKV),
            Some(GgufTensorType::Q4_0)
        );

        let all_q4 = gemma4_selected_group_formats(Gemma4WeightFormat::AllQ4);
        assert_eq!(
            all_q4.selected_format(GEMMA4_SELECTED_GROUP_VOCAB_EMBEDDINGS),
            Some(GgufTensorType::Q4_0)
        );
        assert_eq!(
            all_q4.selected_format(GEMMA4_SELECTED_GROUP_VOCAB_OUTPUT),
            Some(GgufTensorType::Q4_0)
        );
        assert!(all_q4.rejection_reasons().is_empty());
    }

    #[test]
    fn decode_profile_samples_long_context_horizons() {
        assert_eq!(gemma4_decode_profile_targets(128), vec![1, 32, 64, 128]);
        assert_eq!(
            gemma4_decode_profile_targets(1024),
            vec![1, 32, 64, 128, 256, 512, 1024]
        );
        assert_eq!(
            gemma4_decode_profile_targets(4096),
            vec![1, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        );
    }

    #[test]
    fn decode_profile_prefers_each_semantic_label_over_pipeline_family() {
        for label in [
            "attention_input_norm",
            "post_attention_norm",
            "ffn_input_norm",
            "post_ffn_norm",
            "ple_norm",
            "provider_kv_value_norm",
            "final_output_norm",
            "attention_output_projection",
            "ffn_down_projection",
            "qkv_projection",
            "ffn_gate_up_projection",
            "ple_projection",
            "ple_input_gate",
            "output_projection",
            "post_attention_norm_residual",
            "post_ffn_norm_residual",
            "gemma_attention_global_split_scan",
            "gemma_attention_global_split_combine",
        ] {
            assert_eq!(
                gemma4_profile_family(Some(label), "rms_norm_decode_f32"),
                if label == "ple_input_gate" {
                    "ple_projection"
                } else {
                    label
                }
            );
        }
        assert_eq!(
            gemma4_profile_family(None, "matvec_q4_0_64row_mv"),
            "q4_projection_other"
        );
    }

    #[test]
    fn rms_matvec_kernels_map_into_their_projection_families() {
        for kernel in ["matvec_q4_0_64row_mv_rms", "matvec_q4_0_64row_mv"] {
            assert_eq!(gemma4_kernel_family(kernel), "q4_projection_other");
        }
        for kernel in ["matmul_q4_0_qkv_32row_mv_rms", "matmul_q4_0_qkv_32row_mv"] {
            assert_eq!(gemma4_kernel_family(kernel), "q4_qkv_projection");
        }
        for kernel in [
            "matmul_q4_0_gate_up_32row_mv_rms",
            "matmul_q4_0_gate_up_32row_mv",
        ] {
            assert_eq!(gemma4_kernel_family(kernel), "q4_ffn_gate_up_projection");
        }
        for kernel in ["matvec_q6_k_64row_mv_rms", "matvec_q6_k_64row_mv"] {
            assert_eq!(gemma4_kernel_family(kernel), "q6_lm_head_projection");
        }
    }

    #[test]
    fn chat_stops_on_end_turn_while_raw_generation_does_not() {
        assert!(gemma4_should_finish(106, 1, "answer<turn|>", true));
        assert!(!gemma4_should_finish(106, 1, "answer<turn|>", false));
        assert!(gemma4_should_finish(1, 1, "answer", false));
    }
}
