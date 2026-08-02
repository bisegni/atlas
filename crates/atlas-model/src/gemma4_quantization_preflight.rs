//! Deterministic Gemma 4 quantization preflight.
//!
//! This controller profiles the existing Resident Gemma 4 weight-format
//! choices against a fixed prompt and workload, writes a validated ready plan
//! sidecar, and emits structured JSON evidence for CLI/artifact consumers.

use std::{
    collections::BTreeMap,
    env,
    ffi::OsString,
    fmt, fs,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, ensure, Context, Result};
use atlas_core::{GgufModel, GgufTensorType};
use serde_json::{json, Value};

use crate::{
    gemma4_executor::{
        gemma4_ffn_down_projection_kernel, gemma4_q4_gate_up_projection_kernel,
        gemma4_q4_projection_kernel, gemma4_q4_qkv_projection_kernel, Gemma4E2bExecutor,
        Gemma4FinishReason, Gemma4KvCacheType, Gemma4WeightFormat,
    },
    quantization_plan::{
        default_sidecar_path, sha256_file, QuantizationPlan, QuantizationPlanProfilerConfig,
        QuantizationPlanTensor, QUANTIZATION_PLAN_STATE_READY,
    },
    render_gemma4_chat, Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
};

pub const ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT: &str = "ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT";
pub const ATLAS_GEMMA4_WEIGHT_FORMAT: &str = "ATLAS_GEMMA4_WEIGHT_FORMAT";
pub const GEMMA4_QUANTIZATION_PREFLIGHT_PROMPT: &str = "Explain why batching prompt tokens improves transformer prefill performance on a unified-memory GPU. Compare command scheduling, matrix projection reuse, causal attention, key-value cache updates, synchronization, and readback. Keep the answer concise and use one paragraph.";
pub const GEMMA4_QUANTIZATION_PREFLIGHT_MODEL_ID: &str = "gemma4-e2b-q4_0";
pub const GEMMA4_QUANTIZATION_PREFLIGHT_LONG_CONTEXT: usize = 2048;
pub const GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_CONTEXT: usize = 4096;
pub const GEMMA4_QUANTIZATION_PREFLIGHT_KV_CACHE: Gemma4KvCacheType = Gemma4KvCacheType::Q4_0;
static QUANTIZATION_PREFLIGHT_CANCELLED: AtomicBool = AtomicBool::new(false);

const GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_MEASURE_TOKENS_AUTO: usize = 64;
const GEMMA4_QUANTIZATION_PREFLIGHT_LONG_WARMUP_TOKENS_AUTO: usize = 128;
const GEMMA4_QUANTIZATION_PREFLIGHT_LONG_MEASURE_TOKENS_AUTO: usize = 128;
const GEMMA4_QUANTIZATION_PREFLIGHT_RUNS_AUTO: usize = 2;

const GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_MEASURE_TOKENS_VERIFY: usize = 128;
const GEMMA4_QUANTIZATION_PREFLIGHT_LONG_WARMUP_TOKENS_VERIFY: usize = 1024;
const GEMMA4_QUANTIZATION_PREFLIGHT_LONG_MEASURE_TOKENS_VERIFY: usize = 512;
const GEMMA4_QUANTIZATION_PREFLIGHT_RUNS_VERIFY: usize = 5;
const GEMMA4_QUANTIZATION_PREFLIGHT_LONG_IMPROVEMENT_THRESHOLD: f64 = 3.0;
const QUANTIZATION_PREFLIGHT_LOG_PREFIX: &str = "[atlas][quantization-preflight]";

#[derive(Debug, Clone)]
struct Gemma4PlanGroupSpec {
    tensor_names: Vec<String>,
    group_members: Vec<String>,
    source_format: GgufTensorType,
    selected_format: GgufTensorType,
    selected_kernel: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4QuantizationPreflightPolicy {
    Auto,
    Disabled,
    Verify,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gemma4QuantizationPreflightInvocation {
    AutoLoad,
    CliProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Gemma4QuantizationSidecarAction {
    Skip,
    UseCachedPlan,
    ProfileAndRewrite,
}

fn log_quantization_preflight(message: impl fmt::Display) {
    eprintln!("{QUANTIZATION_PREFLIGHT_LOG_PREFIX} {message}");
}

fn format_optional_rate(rate: f64) -> String {
    if rate.is_finite() && rate > 0.0 {
        format!("{rate:.2} tok/s")
    } else {
        "n/a".to_owned()
    }
}

fn format_optional_percent(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.2}%"),
        Some(_) | None => "n/a".to_owned(),
    }
}

fn invocation_name(invocation: Gemma4QuantizationPreflightInvocation) -> &'static str {
    match invocation {
        Gemma4QuantizationPreflightInvocation::AutoLoad => "autoload",
        Gemma4QuantizationPreflightInvocation::CliProfile => "cli_profile",
    }
}

fn policy_name(policy: Gemma4QuantizationPreflightPolicy) -> &'static str {
    match policy {
        Gemma4QuantizationPreflightPolicy::Auto => "auto",
        Gemma4QuantizationPreflightPolicy::Disabled => "disabled",
        Gemma4QuantizationPreflightPolicy::Verify => "verify",
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4QuantizationPreflightSummary {
    pub runs: usize,
    pub median_decode_tok_s: f64,
    pub median_gpu_ns: u64,
    pub prompt_token_sha256: String,
    pub generated_token_sha256: String,
    pub measured_generated_token_sha256: String,
    pub first_eos_position: Option<usize>,
    pub selected_kernels: Value,
    pub resident_bytes: u64,
    pub kv_cache_bytes: u64,
    pub weight_upload_bytes: u64,
    pub readback_bytes: u64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub measured_decode_tokens: usize,
    pub finish_reason: String,
    pub weight_format: String,
    pub pass: bool,
    pub rejection_reason: Option<String>,
    pub short_median_decode_tok_s: f64,
    pub short_median_gpu_ns: u64,
    pub short_prompt_token_sha256: String,
    pub short_generated_token_sha256: String,
    pub short_measured_generated_token_sha256: String,
    pub short_first_eos_position: Option<usize>,
    pub short_selected_kernels: Value,
    pub short_resident_bytes: u64,
    pub short_kv_cache_bytes: u64,
    pub short_weight_upload_bytes: u64,
    pub short_readback_bytes: u64,
    pub short_prompt_tokens: usize,
    pub short_generated_tokens: usize,
    pub short_measured_decode_tokens: usize,
    pub short_finish_reason: String,
}

#[derive(Debug, Clone)]
pub struct Gemma4QuantizationPreflightReport {
    pub preflight_state: String,
    pub model_id: String,
    pub model_sha256: String,
    pub hardware_identity: String,
    pub quantization_plan_path: String,
    pub selected_weight_format: String,
    pub selected_group_formats: BTreeMap<String, String>,
    pub baseline: Gemma4QuantizationPreflightSummary,
    pub candidate: Option<Gemma4QuantizationPreflightSummary>,
    pub rejections: BTreeMap<String, String>,
    pub policy: String,
    pub explicit_weight_format: Option<String>,
    pub profile_runs: usize,
    pub long_context_improvement_percent: Option<f64>,
    pub short_context_regression_percent: Option<f64>,
    pub output_path: Option<PathBuf>,
}

impl Gemma4QuantizationPreflightReport {
    pub fn to_value(&self) -> Value {
        json!({
            "event": "gemma4_quantization_preflight",
            "preflight_state": self.preflight_state,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "hardware_identity": self.hardware_identity,
            "policy": self.policy,
            "explicit_weight_format": self.explicit_weight_format,
            "quantization_plan_path": self.quantization_plan_path,
            "selected_weight_format": self.selected_weight_format,
            "selected_group_formats": self.selected_group_formats,
            "baseline": summary_to_value(&self.baseline),
            "candidate": self.candidate.as_ref().map(summary_to_value),
            "rejections": self.rejections,
            "profile_runs": self.profile_runs,
            "long_context_improvement_percent": self.long_context_improvement_percent,
            "short_context_regression_percent": self.short_context_regression_percent,
            "output_path": self.output_path.as_ref().map(|path| path.display().to_string()),
        })
    }

    pub fn write_output(&self, path: impl AsRef<Path>) -> Result<()> {
        write_json_value_atomically(path.as_ref(), &self.to_value())
    }
}

pub fn parse_preflight_policy(value: Option<&str>) -> Result<Gemma4QuantizationPreflightPolicy> {
    match value.unwrap_or("auto") {
        "auto" => Ok(Gemma4QuantizationPreflightPolicy::Auto),
        "disabled" => Ok(Gemma4QuantizationPreflightPolicy::Disabled),
        "verify" => Ok(Gemma4QuantizationPreflightPolicy::Verify),
        value => bail!(
            "unsupported ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT `{value}`; expected auto, disabled, or verify"
        ),
    }
}

pub fn parse_weight_format_override(value: Option<&str>) -> Result<Option<Gemma4WeightFormat>> {
    match value {
        None | Some("mixed") | Some("mixed_q4_q6") => Ok(None),
        Some("q4_embeddings") => Ok(Some(Gemma4WeightFormat::Q4Embeddings)),
        Some("q4_lm_head") => Ok(Some(Gemma4WeightFormat::Q4LmHead)),
        Some("all_q4") => Ok(Some(Gemma4WeightFormat::AllQ4)),
        Some(value) => bail!(
            "unsupported ATLAS_GEMMA4_WEIGHT_FORMAT `{value}`; expected mixed_q4_q6, q4_embeddings, q4_lm_head, or all_q4"
        ),
    }
}

pub fn load_model_without_preflight(path: impl AsRef<Path>) -> Result<Gemma4E2bModel> {
    Gemma4E2bModel::load_gguf_without_quantization_preflight(path)
}

pub fn maybe_run_gemma4_quantization_preflight(
    model: &Gemma4E2bModel,
) -> Result<Option<Gemma4QuantizationPreflightReport>> {
    run_gemma4_quantization_preflight(model, Gemma4QuantizationPreflightInvocation::AutoLoad, None)
}

pub fn run_gemma4_quantization_preflight(
    model: &Gemma4E2bModel,
    invocation: Gemma4QuantizationPreflightInvocation,
    output_path: Option<&Path>,
) -> Result<Option<Gemma4QuantizationPreflightReport>> {
    let policy = parse_preflight_policy(
        env::var(ATLAS_GEMMA4_QUANTIZATION_PREFLIGHT)
            .ok()
            .as_deref(),
    )?;
    let explicit_weight_format =
        parse_weight_format_override(env::var(ATLAS_GEMMA4_WEIGHT_FORMAT).ok().as_deref())?;
    let hardware_identity = hardware_identity(model);
    let model_id = gemma4_plan_model_id(model);
    let model_sha256 = sha256_file(model.model_path())?;
    let sidecar_path = default_sidecar_path(model.model_path());
    log_quantization_preflight(format!(
        "invocation={} policy={} model_id={} model_path={} model_sha256={}",
        invocation_name(invocation),
        policy_name(policy),
        model_id,
        model.model_path().display(),
        model_sha256
    ));

    if let Gemma4QuantizationSidecarAction::Skip =
        resolve_sidecar_action(invocation, policy, explicit_weight_format.is_some(), false)
    {
        if explicit_weight_format.is_some() {
            log_quantization_preflight(format!(
                "decision=skip reason=explicit_weight_format_override override={}",
                explicit_weight_format
                    .as_ref()
                    .map(|format| format.as_str())
                    .unwrap_or("mixed_q4_q6")
            ));
        } else if policy == Gemma4QuantizationPreflightPolicy::Disabled {
            log_quantization_preflight("decision=skip reason=policy_disabled");
        } else {
            log_quantization_preflight("decision=skip reason=unavailable");
        }
        return Ok(None);
    }

    let cached_plan = if invocation == Gemma4QuantizationPreflightInvocation::AutoLoad
        && policy != Gemma4QuantizationPreflightPolicy::Disabled
        && explicit_weight_format.is_none()
    {
        match model.quantization_plan_with_identity(Some(&hardware_identity)) {
            Ok(plan) => plan,
            Err(_) => None,
        }
    } else {
        None
    };
    match resolve_sidecar_action(
        invocation,
        policy,
        explicit_weight_format.is_some(),
        cached_plan.is_some(),
    ) {
        Gemma4QuantizationSidecarAction::Skip => return Ok(None),
        Gemma4QuantizationSidecarAction::UseCachedPlan => {
            log_quantization_preflight(format!(
                "decision=use_cached_plan cached_plan_path={}",
                sidecar_path.display()
            ));
            return Ok(None);
        }
        Gemma4QuantizationSidecarAction::ProfileAndRewrite => {}
    }

    let config = workload_config(policy);
    log_quantization_preflight(format!(
        "profiling_configuration runs={} long_context={} short_context={} long_warmup_tokens={} long_measure_tokens={} short_measure_tokens={}",
        config.runs,
        config.long_context,
        config.short_context,
        config.long_warmup_decode_tokens,
        config.long_measure_decode_tokens,
        config.short_measure_decode_tokens
    ));
    log_quantization_preflight("baseline start format=mixed_q4_q6");
    let baseline =
        run_weight_format_workload(model.model_path(), Gemma4WeightFormat::MixedQ4Q6, &config)?;
    log_quantization_preflight("candidate start format=all_q4");
    let candidate =
        run_weight_format_workload(model.model_path(), Gemma4WeightFormat::AllQ4, &config)?;

    let (
        selected_weight_format,
        selected_summary,
        rejections,
        long_context_improvement_percent,
        short_context_regression_percent,
        state,
    ) = select_candidate(&baseline, &candidate);
    log_quantization_preflight(format!(
        "candidate selection selected_weight_format={} long_improvement={} short_regression={} status={}{}",
        selected_weight_format.as_str(),
        format_optional_percent(long_context_improvement_percent),
        format_optional_percent(short_context_regression_percent),
        if rejections.is_empty() {
            "accepted"
        } else {
            "rejected"
        },
        if rejections.is_empty() {
            String::new()
        } else {
            format!(
                " rejections={}",
                rejections
                    .iter()
                    .map(|(key, value)| format!("{key}:{value}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        }
    ));

    let plan = build_plan(
        model,
        &hardware_identity,
        &model_sha256,
        &selected_weight_format,
        &selected_summary,
        &baseline,
    )?;
    log_quantization_preflight(format!("sidecar write path={}", sidecar_path.display()));
    plan.write_path(&sidecar_path)?;
    log_quantization_preflight(format!(
        "sidecar write complete path={}",
        sidecar_path.display()
    ));

    let selected_group_formats = selected_group_formats(&plan);
    let report = Gemma4QuantizationPreflightReport {
        preflight_state: state.to_owned(),
        model_id: model
            .gguf()
            .metadata
            .get("general.name")
            .cloned()
            .unwrap_or_else(|| "gemma4-e2b-q4_0".to_owned()),
        model_sha256,
        hardware_identity,
        quantization_plan_path: sidecar_path.display().to_string(),
        selected_weight_format: selected_weight_format.as_str().to_owned(),
        selected_group_formats,
        baseline,
        candidate: Some(candidate),
        rejections,
        policy: match policy {
            Gemma4QuantizationPreflightPolicy::Auto => "auto",
            Gemma4QuantizationPreflightPolicy::Disabled => "disabled",
            Gemma4QuantizationPreflightPolicy::Verify => "verify",
        }
        .to_owned(),
        explicit_weight_format: explicit_weight_format.map(|format| format.as_str().to_owned()),
        profile_runs: config.runs,
        long_context_improvement_percent,
        short_context_regression_percent,
        output_path: output_path.map(Path::to_path_buf),
    };

    if let Some(output_path) = output_path {
        report.write_output(output_path)?;
    }
    log_quantization_preflight(format!(
        "complete state={} selected_weight_format={} sidecar_path={} output_path={}",
        state,
        selected_weight_format.as_str(),
        sidecar_path.display(),
        output_path
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_owned())
    ));
    Ok(Some(report))
}

fn resolve_sidecar_action(
    invocation: Gemma4QuantizationPreflightInvocation,
    policy: Gemma4QuantizationPreflightPolicy,
    explicit_weight_format: bool,
    cached_plan_present: bool,
) -> Gemma4QuantizationSidecarAction {
    match invocation {
        Gemma4QuantizationPreflightInvocation::CliProfile => {
            Gemma4QuantizationSidecarAction::ProfileAndRewrite
        }
        Gemma4QuantizationPreflightInvocation::AutoLoad => {
            if explicit_weight_format || policy == Gemma4QuantizationPreflightPolicy::Disabled {
                Gemma4QuantizationSidecarAction::Skip
            } else if cached_plan_present {
                Gemma4QuantizationSidecarAction::UseCachedPlan
            } else {
                Gemma4QuantizationSidecarAction::ProfileAndRewrite
            }
        }
    }
}

fn workload_config(
    policy: Gemma4QuantizationPreflightPolicy,
) -> Gemma4QuantizationPreflightWorkload {
    match policy {
        Gemma4QuantizationPreflightPolicy::Verify => Gemma4QuantizationPreflightWorkload {
            runs: GEMMA4_QUANTIZATION_PREFLIGHT_RUNS_VERIFY,
            long_warmup_decode_tokens: GEMMA4_QUANTIZATION_PREFLIGHT_LONG_WARMUP_TOKENS_VERIFY,
            long_measure_decode_tokens: GEMMA4_QUANTIZATION_PREFLIGHT_LONG_MEASURE_TOKENS_VERIFY,
            short_measure_decode_tokens: GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_MEASURE_TOKENS_VERIFY,
            long_context: GEMMA4_QUANTIZATION_PREFLIGHT_LONG_CONTEXT,
            short_context: GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_CONTEXT,
        },
        Gemma4QuantizationPreflightPolicy::Auto | Gemma4QuantizationPreflightPolicy::Disabled => {
            Gemma4QuantizationPreflightWorkload {
                runs: GEMMA4_QUANTIZATION_PREFLIGHT_RUNS_AUTO,
                long_warmup_decode_tokens: GEMMA4_QUANTIZATION_PREFLIGHT_LONG_WARMUP_TOKENS_AUTO,
                long_measure_decode_tokens: GEMMA4_QUANTIZATION_PREFLIGHT_LONG_MEASURE_TOKENS_AUTO,
                short_measure_decode_tokens:
                    GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_MEASURE_TOKENS_AUTO,
                long_context: GEMMA4_QUANTIZATION_PREFLIGHT_LONG_CONTEXT,
                short_context: GEMMA4_QUANTIZATION_PREFLIGHT_SHORT_CONTEXT,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Gemma4QuantizationPreflightWorkload {
    runs: usize,
    long_warmup_decode_tokens: usize,
    long_measure_decode_tokens: usize,
    short_measure_decode_tokens: usize,
    long_context: usize,
    short_context: usize,
}

fn run_weight_format_workload(
    model_path: &Path,
    weight_format: Gemma4WeightFormat,
    config: &Gemma4QuantizationPreflightWorkload,
) -> Result<Gemma4QuantizationPreflightSummary> {
    log_quantization_preflight(format!(
        "format={} phase=long start runs={} warmup_tokens={} measure_tokens={} context={}",
        weight_format.as_str(),
        config.runs,
        config.long_warmup_decode_tokens,
        config.long_measure_decode_tokens,
        config.long_context
    ));
    let long_runs = run_fixed_workload(
        model_path,
        weight_format,
        config.runs,
        config.long_warmup_decode_tokens,
        config.long_measure_decode_tokens,
        config.long_context,
        "long",
    )?;
    let long_summary = summarize_runs(&long_runs, config.runs, "long")?;
    log_quantization_preflight(format!(
        "format={} phase=long complete runs={} median_decode_tok_s={} median_gpu_ns={} pass={}",
        weight_format.as_str(),
        config.runs,
        format_optional_rate(long_summary.median_decode_tok_s),
        long_summary.median_gpu_ns,
        long_summary.pass
    ));
    log_quantization_preflight(format!(
        "format={} phase=short start runs={} warmup_tokens={} measure_tokens={} context={}",
        weight_format.as_str(),
        config.runs,
        0,
        config.short_measure_decode_tokens,
        config.short_context
    ));
    let short_runs = run_fixed_workload(
        model_path,
        weight_format,
        config.runs,
        0,
        config.short_measure_decode_tokens,
        config.short_context,
        "short",
    )?;
    let short_summary = summarize_runs(&short_runs, config.runs, "short")?;
    log_quantization_preflight(format!(
        "format={} phase=short complete runs={} median_decode_tok_s={} median_gpu_ns={} pass={}",
        weight_format.as_str(),
        config.runs,
        format_optional_rate(short_summary.median_decode_tok_s),
        short_summary.median_gpu_ns,
        short_summary.pass
    ));
    let pass = long_summary.pass && short_summary.pass;
    let rejection_reason =
        (!pass).then(|| "workload summary failed deterministic or accounting checks".into());
    Ok(Gemma4QuantizationPreflightSummary {
        runs: config.runs,
        median_decode_tok_s: long_summary.median_decode_tok_s,
        median_gpu_ns: long_summary.median_gpu_ns,
        prompt_token_sha256: long_summary.prompt_token_sha256,
        generated_token_sha256: long_summary.generated_token_sha256,
        measured_generated_token_sha256: long_summary.measured_generated_token_sha256,
        first_eos_position: long_summary.first_eos_position,
        selected_kernels: json!({
            "long": long_summary.selected_kernels,
            "short": short_summary.selected_kernels,
        }),
        resident_bytes: long_summary.resident_bytes,
        kv_cache_bytes: long_summary.kv_cache_bytes,
        weight_upload_bytes: long_summary.weight_upload_bytes,
        readback_bytes: long_summary.readback_bytes,
        prompt_tokens: long_summary.prompt_tokens,
        generated_tokens: long_summary.generated_tokens,
        measured_decode_tokens: long_summary.measured_decode_tokens,
        finish_reason: long_summary.finish_reason,
        weight_format: weight_format.as_str().to_owned(),
        pass,
        rejection_reason,
        short_median_decode_tok_s: short_summary.median_decode_tok_s,
        short_prompt_token_sha256: short_summary.prompt_token_sha256,
        short_generated_token_sha256: short_summary.generated_token_sha256,
        short_measured_generated_token_sha256: short_summary.measured_generated_token_sha256,
        short_first_eos_position: short_summary.first_eos_position,
        short_selected_kernels: short_summary.selected_kernels,
        short_resident_bytes: short_summary.resident_bytes,
        short_kv_cache_bytes: short_summary.kv_cache_bytes,
        short_weight_upload_bytes: short_summary.weight_upload_bytes,
        short_readback_bytes: short_summary.readback_bytes,
        short_median_gpu_ns: short_summary.median_gpu_ns,
        short_prompt_tokens: short_summary.prompt_tokens,
        short_generated_tokens: short_summary.generated_tokens,
        short_measured_decode_tokens: short_summary.measured_decode_tokens,
        short_finish_reason: short_summary.finish_reason,
    })
}

#[derive(Debug, Clone)]
struct FixedWorkloadRun {
    prompt_token_sha256: String,
    generated_token_sha256: String,
    measured_generated_token_sha256: String,
    first_eos_position: Option<usize>,
    selected_kernels: Value,
    resident_bytes: u64,
    kv_cache_bytes: u64,
    weight_upload_bytes: u64,
    readback_bytes: u64,
    decode_gpu_ns: u64,
    decode_tok_s: f64,
    prefill_tok_s: f64,
    prompt_tokens: usize,
    generated_tokens: usize,
    measured_decode_tokens: usize,
    finish_reason: String,
}

#[derive(Debug, Clone)]
struct FixedWorkloadSummary {
    median_decode_tok_s: f64,
    median_gpu_ns: u64,
    prompt_token_sha256: String,
    generated_token_sha256: String,
    measured_generated_token_sha256: String,
    first_eos_position: Option<usize>,
    selected_kernels: Value,
    resident_bytes: u64,
    kv_cache_bytes: u64,
    weight_upload_bytes: u64,
    readback_bytes: u64,
    prompt_tokens: usize,
    generated_tokens: usize,
    measured_decode_tokens: usize,
    finish_reason: String,
    pass: bool,
}

fn run_fixed_workload(
    model_path: &Path,
    weight_format: Gemma4WeightFormat,
    runs: usize,
    warmup_decode_tokens: usize,
    measured_decode_tokens: usize,
    max_context: usize,
    phase: &'static str,
) -> Result<Vec<FixedWorkloadRun>> {
    let mut records = Vec::with_capacity(runs);
    for run_index in 0..runs {
        let run_number = run_index + 1;
        log_quantization_preflight(format!(
            "format={} phase={} run={}/{} start warmup_tokens={} measure_tokens={} context={}",
            weight_format.as_str(),
            phase,
            run_number,
            runs,
            warmup_decode_tokens,
            measured_decode_tokens,
            max_context
        ));
        let started = Instant::now();
        let _weight_format_guard =
            EnvVarGuard::set(ATLAS_GEMMA4_WEIGHT_FORMAT, weight_format.as_str());
        let model = load_model_without_preflight(model_path)?;
        let mut executor = Gemma4E2bExecutor::new_with_kv_cache(
            &model,
            max_context,
            GEMMA4_QUANTIZATION_PREFLIGHT_KV_CACHE,
        )?;
        let prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
            Gemma4ChatRole::User,
            GEMMA4_QUANTIZATION_PREFLIGHT_PROMPT,
        )])?;
        let prompt_token_ids = model.tokenize(&prompt)?;
        let prompt_token_sha256 = token_ids_sha256(&prompt_token_ids);
        let generation = executor.generate_greedy_fixed_benchmark_window_stream(
            &prompt,
            warmup_decode_tokens,
            measured_decode_tokens,
            &QUANTIZATION_PREFLIGHT_CANCELLED,
            |_| Ok(()),
        )?;
        let generated_token_sha256 = token_ids_sha256(&generation.generation.generated_token_ids);
        let measured_generated_token_sha256 =
            token_ids_sha256(&generation.generation.generated_token_ids[warmup_decode_tokens..]);
        let selected_kernels = json!({
            "attention": generation.metrics.attention_kernel,
            "weight_format": generation.metrics.weight_format.as_str(),
            "embedding": generation.metrics.embedding_kernel,
            "output_projection": generation.metrics.output_projection_kernel,
            "q6_projection": generation.metrics.q6_projection_kernel,
            "q4_projection": generation.metrics.q4_projection_kernel,
            "q4_qkv_projection": generation.metrics.q4_qkv_projection_kernel,
            "q4_gate_up_projection": generation.metrics.q4_gate_up_projection_kernel,
            "q4_batch_projection": generation.metrics.q4_batch_projection_kernel,
            "ffn_down_projection": generation.metrics.ffn_down_projection_kernel,
            "rms_norm": generation.metrics.rms_norm_kernel,
        });
        records.push(FixedWorkloadRun {
            prompt_token_sha256,
            generated_token_sha256,
            measured_generated_token_sha256,
            first_eos_position: generation.first_eos_position,
            selected_kernels,
            resident_bytes: generation.metrics.resident_bytes,
            kv_cache_bytes: generation.metrics.kv_cache_bytes,
            weight_upload_bytes: generation.metrics.weight_upload_bytes,
            readback_bytes: generation.metrics.readback_bytes,
            decode_gpu_ns: generation
                .metrics
                .decode
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64,
            decode_tok_s: rate(
                generation.metrics.decode_command_buffers as usize,
                generation.metrics.decode,
            ),
            prefill_tok_s: rate(
                generation.generation.prompt_token_ids.len(),
                generation.metrics.prefill,
            ),
            prompt_tokens: generation.generation.prompt_token_ids.len(),
            generated_tokens: generation.generation.generated_token_ids.len(),
            measured_decode_tokens,
            finish_reason: match generation.finish_reason {
                Gemma4FinishReason::Eos => "eos",
                Gemma4FinishReason::MaxTokens => "max_tokens",
                Gemma4FinishReason::Cancelled => "cancelled",
            }
            .to_owned(),
        });
        let elapsed = started.elapsed();
        let decode_tok_s = records.last().map(|run| run.decode_tok_s).unwrap_or(0.0);
        let prefill_tok_s = records.last().map(|run| run.prefill_tok_s).unwrap_or(0.0);
        log_quantization_preflight(format!(
            "format={} phase={} run={}/{} complete elapsed_ms={:.3} decode_throughput={} prefill_throughput={}",
            weight_format.as_str(),
            phase,
            run_number,
            runs,
            elapsed.as_secs_f64() * 1_000.0,
            format_optional_rate(decode_tok_s),
            format_optional_rate(prefill_tok_s)
        ));
    }
    Ok(records)
}

fn summarize_runs(
    runs: &[FixedWorkloadRun],
    expected_runs: usize,
    label: &str,
) -> Result<FixedWorkloadSummary> {
    ensure!(
        runs.len() == expected_runs,
        "{label} workload expected {expected_runs} runs, got {}",
        runs.len()
    );
    let decode_tok_s = runs.iter().map(|run| run.decode_tok_s).collect::<Vec<_>>();
    let median_decode_tok_s = median(&decode_tok_s);
    let decode_gpu_ns_values = runs.iter().map(|run| run.decode_gpu_ns).collect::<Vec<_>>();
    let median_gpu_ns = median_u64(&decode_gpu_ns_values);
    let prompt_token_sha256 =
        unique_string(runs.iter().map(|run| run.prompt_token_sha256.as_str()))
            .context("prompt token digest is not deterministic")?;
    let generated_token_sha256 =
        unique_string(runs.iter().map(|run| run.generated_token_sha256.as_str()))
            .context("generated token digest is not deterministic")?;
    let measured_generated_token_sha256 = unique_string(
        runs.iter()
            .map(|run| run.measured_generated_token_sha256.as_str()),
    )
    .context("measured token digest is not deterministic")?;
    let first_eos_position = unique_option(runs.iter().map(|run| run.first_eos_position))?;
    let selected_kernels = unique_value(runs.iter().map(|run| run.selected_kernels.clone()))
        .context("selected kernels are not deterministic")?;
    let resident_bytes = unique_u64(runs.iter().map(|run| run.resident_bytes))
        .context("resident bytes are not stable")?;
    let kv_cache_bytes = unique_u64(runs.iter().map(|run| run.kv_cache_bytes))
        .context("KV cache bytes are not stable")?;
    let weight_upload_bytes = unique_u64(runs.iter().map(|run| run.weight_upload_bytes))
        .context("weight upload bytes are not stable")?;
    let readback_bytes = unique_u64(runs.iter().map(|run| run.readback_bytes))
        .context("readback bytes are not stable")?;
    let prompt_tokens = unique_usize(runs.iter().map(|run| run.prompt_tokens))
        .context("prompt tokens are not stable")?;
    let generated_tokens = unique_usize(runs.iter().map(|run| run.generated_tokens))
        .context("generated tokens are not stable")?;
    let measured_decode_tokens = unique_usize(runs.iter().map(|run| run.measured_decode_tokens))
        .context("measured decode tokens are not stable")?;
    let finish_reason = unique_string(runs.iter().map(|run| run.finish_reason.as_str()))
        .context("finish reason is not deterministic")?;
    let pass = median_decode_tok_s.is_finite()
        && median_decode_tok_s > 0.0
        && median_gpu_ns > 0
        && runs
            .iter()
            .all(|run| run.decode_tok_s.is_finite() && run.decode_tok_s > 0.0)
        && runs
            .iter()
            .all(|run| run.prefill_tok_s.is_finite() && run.prefill_tok_s > 0.0);
    Ok(FixedWorkloadSummary {
        median_decode_tok_s,
        median_gpu_ns,
        prompt_token_sha256,
        generated_token_sha256,
        measured_generated_token_sha256,
        first_eos_position,
        selected_kernels,
        resident_bytes,
        kv_cache_bytes,
        weight_upload_bytes,
        readback_bytes,
        prompt_tokens,
        generated_tokens,
        measured_decode_tokens,
        finish_reason,
        pass,
    })
}

fn select_candidate(
    baseline: &Gemma4QuantizationPreflightSummary,
    candidate: &Gemma4QuantizationPreflightSummary,
) -> (
    Gemma4WeightFormat,
    Gemma4QuantizationPreflightSummary,
    BTreeMap<String, String>,
    Option<f64>,
    Option<f64>,
    &'static str,
) {
    let mut rejections = BTreeMap::new();
    let long_selected_consistent = candidate.selected_kernels == candidate.short_selected_kernels;
    let long_parity = candidate.prompt_token_sha256 == baseline.prompt_token_sha256
        && candidate.generated_token_sha256 == baseline.generated_token_sha256
        && candidate.measured_generated_token_sha256 == baseline.measured_generated_token_sha256
        && candidate.first_eos_position == baseline.first_eos_position;
    let short_parity = candidate.short_prompt_token_sha256 == baseline.short_prompt_token_sha256
        && candidate.short_generated_token_sha256 == baseline.short_generated_token_sha256
        && candidate.short_measured_generated_token_sha256
            == baseline.short_measured_generated_token_sha256
        && candidate.short_first_eos_position == baseline.short_first_eos_position;
    // Accounting is an invariant within a workload format, not across
    // formats. A candidate is expected to change resident weight bytes, and
    // the long and short workloads intentionally use different KV horizons.
    // `summarize_runs` already rejects run-to-run changes in resident/KV,
    // upload, and readback accounting for each format and phase.
    let accounting_present = candidate.resident_bytes > 0
        && candidate.kv_cache_bytes > 0
        && candidate.weight_upload_bytes > 0
        && candidate.readback_bytes > 0
        && candidate.short_resident_bytes > 0
        && candidate.short_kv_cache_bytes > 0
        && candidate.short_weight_upload_bytes > 0
        && candidate.short_readback_bytes > 0;
    let candidate_selected_kernels_differ = candidate.selected_kernels != baseline.selected_kernels;
    let long_improvement = if baseline.median_decode_tok_s > 0.0 {
        ((candidate.median_decode_tok_s / baseline.median_decode_tok_s) - 1.0) * 100.0
    } else {
        f64::NEG_INFINITY
    };
    let short_regression = if baseline.short_median_decode_tok_s > 0.0 {
        ((candidate.short_median_decode_tok_s / baseline.short_median_decode_tok_s) - 1.0) * 100.0
    } else {
        f64::NEG_INFINITY
    };
    let pass = baseline.pass
        && candidate.pass
        && candidate_selected_kernels_differ
        && long_selected_consistent
        && long_parity
        && short_parity
        && accounting_present
        && long_improvement >= GEMMA4_QUANTIZATION_PREFLIGHT_LONG_IMPROVEMENT_THRESHOLD
        && short_regression >= 0.0;
    if !baseline.pass {
        rejections.insert(
            "baseline".into(),
            "baseline workload failed deterministic checks".into(),
        );
    }
    if !candidate.pass {
        rejections.insert(
            "candidate".into(),
            "candidate workload failed deterministic checks".into(),
        );
    }
    if !candidate_selected_kernels_differ {
        rejections.insert(
            "candidate".into(),
            "candidate selected kernels did not differ from baseline".into(),
        );
    }
    if !long_selected_consistent {
        rejections.insert(
            "candidate".into(),
            "candidate selected kernels were not consistent between long and short workloads"
                .into(),
        );
    }
    if !long_parity {
        rejections.insert(
            "candidate".into(),
            "candidate failed exact prompt/generated/measured token or EOS parity".into(),
        );
    }
    if !short_parity {
        rejections.insert(
            "candidate".into(),
            "candidate failed short-window parity".into(),
        );
    }
    if !accounting_present {
        rejections.insert(
            "candidate".into(),
            "candidate is missing positive Resident, KV, upload, or readback accounting".into(),
        );
    }
    if long_improvement < GEMMA4_QUANTIZATION_PREFLIGHT_LONG_IMPROVEMENT_THRESHOLD {
        rejections.insert(
            "candidate".into(),
            format!(
                "candidate long-context improvement {:.2}% is below the required {:.2}%",
                long_improvement, GEMMA4_QUANTIZATION_PREFLIGHT_LONG_IMPROVEMENT_THRESHOLD
            ),
        );
    }
    if short_regression < 0.0 {
        rejections.insert(
            "candidate".into(),
            format!(
                "candidate short-context regression {:.2}% is below zero",
                short_regression
            ),
        );
    }
    let selected_weight_format = if pass {
        Gemma4WeightFormat::AllQ4
    } else {
        Gemma4WeightFormat::MixedQ4Q6
    };
    let selected_summary = if pass {
        candidate.clone()
    } else {
        baseline.clone()
    };
    let state = if pass {
        QUANTIZATION_PLAN_STATE_READY
    } else {
        QUANTIZATION_PLAN_STATE_READY
    };
    (
        selected_weight_format,
        selected_summary,
        rejections,
        Some(long_improvement),
        Some(short_regression),
        state,
    )
}

fn build_plan(
    model: &Gemma4E2bModel,
    hardware_identity: &str,
    model_sha256: &str,
    selected_weight_format: &Gemma4WeightFormat,
    selected_summary: &Gemma4QuantizationPreflightSummary,
    baseline: &Gemma4QuantizationPreflightSummary,
) -> Result<QuantizationPlan> {
    let mut plan = QuantizationPlan::new(gemma4_plan_model_id(model), model_sha256);
    plan.state = QUANTIZATION_PLAN_STATE_READY.into();
    plan.oracle_sha256 = selected_summary.generated_token_sha256.clone();
    plan.hardware_identity = hardware_identity.to_owned();
    plan.profiler_configuration = QuantizationPlanProfilerConfig {
        mode: "auto".into(),
        prompt_sha256: selected_summary.prompt_token_sha256.clone(),
        decode_tokens: selected_summary.measured_decode_tokens as u32,
        runs: selected_summary.runs as u32,
    };
    plan.max_abs_logit_delta = crate::quantization_plan::DEFAULT_MAX_ABS_LOGIT_DELTA;
    for spec in gemma4_plan_group_specs(model.gguf(), model.config.layers, *selected_weight_format)?
    {
        for tensor_name in &spec.tensor_names {
            let source_tensor = model
                .gguf()
                .tensors
                .iter()
                .find(|tensor| tensor.name == *tensor_name)
                .with_context(|| format!("Gemma 4 GGUF missing tensor `{tensor_name}`"))?;
            ensure!(
                source_tensor.tensor_type == spec.source_format,
                "Gemma 4 GGUF tensor `{tensor_name}` must be {:?}",
                spec.source_format
            );
            plan.tensors.insert(
                tensor_name.to_owned(),
                QuantizationPlanTensor {
                    group_members: spec.group_members.clone(),
                    source_format: spec.source_format,
                    selected_format: spec.selected_format,
                    selected_kernel: spec.selected_kernel.to_owned(),
                    max_abs_logit_delta: crate::quantization_plan::DEFAULT_MAX_ABS_LOGIT_DELTA,
                    median_gpu_ns: plan_gpu_ns(selected_summary),
                    baseline_gpu_ns: plan_gpu_ns(baseline),
                    resident_bytes: selected_summary.resident_bytes,
                    upload_bytes: selected_summary.weight_upload_bytes,
                    parity_digest: format!(
                        "{}:{}:{}:{}",
                        selected_summary.prompt_token_sha256,
                        selected_summary.generated_token_sha256,
                        selected_summary.measured_generated_token_sha256,
                        selected_summary.first_eos_position.unwrap_or_default()
                    ),
                    parity: true,
                    rejection_reason: None,
                },
            );
        }
    }
    Ok(plan)
}

fn gemma4_plan_model_id(model: &Gemma4E2bModel) -> String {
    model
        .gguf()
        .metadata
        .get("general.name")
        .cloned()
        .unwrap_or_else(|| GEMMA4_QUANTIZATION_PREFLIGHT_MODEL_ID.to_owned())
}

fn selected_group_formats(plan: &QuantizationPlan) -> BTreeMap<String, String> {
    plan.tensors
        .iter()
        .map(|(name, tensor)| (name.clone(), format_name(tensor.selected_format).to_owned()))
        .collect()
}

fn plan_gpu_ns(summary: &Gemma4QuantizationPreflightSummary) -> u64 {
    summary.median_gpu_ns
}

fn gemma4_plan_group_specs(
    model: &GgufModel,
    layers: usize,
    selected_weight_format: Gemma4WeightFormat,
) -> Result<Vec<Gemma4PlanGroupSpec>> {
    let vocabulary_selected_format = match selected_weight_format {
        Gemma4WeightFormat::MixedQ4Q6 | Gemma4WeightFormat::Q4LmHead => GgufTensorType::Q6K,
        Gemma4WeightFormat::Q4Embeddings | Gemma4WeightFormat::AllQ4 => GgufTensorType::Q4_0,
    };
    let mut specs = vec![
        Gemma4PlanGroupSpec {
            tensor_names: vec![
                "token_embd.weight".into(),
                "per_layer_token_embd.weight".into(),
            ],
            group_members: vec![
                "token_embd.weight".into(),
                "per_layer_token_embd.weight".into(),
            ],
            source_format: GgufTensorType::Q6K,
            selected_format: vocabulary_selected_format,
            selected_kernel: match vocabulary_selected_format {
                GgufTensorType::Q4_0 => "embedding_lookup_q4_0",
                GgufTensorType::Q6K => "embedding_lookup_q6_k",
                _ => unreachable!("unsupported Gemma vocabulary format"),
            },
        },
        Gemma4PlanGroupSpec {
            tensor_names: vec!["per_layer_model_proj.weight".into()],
            group_members: vec!["per_layer_model_proj.weight".into()],
            source_format: GgufTensorType::F16,
            selected_format: GgufTensorType::F16,
            selected_kernel: "matmul_f16_batch",
        },
    ];
    for layer in 0..layers {
        let p = format!("blk.{layer}");
        let attn_q = format!("{p}.attn_q.weight");
        let attn_k = format!("{p}.attn_k.weight");
        let attn_v = format!("{p}.attn_v.weight");
        let has_attn_k = gemma4_tensor_exists(model, &attn_k);
        let has_attn_v = gemma4_tensor_exists(model, &attn_v);
        ensure!(
            has_attn_k == has_attn_v,
            "Gemma 4 layer {layer} has an incomplete shared KV tensor group"
        );
        let mut qkv_tensor_names = vec![attn_q.clone()];
        if has_attn_k {
            qkv_tensor_names.push(attn_k.clone());
            qkv_tensor_names.push(attn_v.clone());
        }
        specs.extend([
            Gemma4PlanGroupSpec {
                tensor_names: qkv_tensor_names.clone(),
                group_members: qkv_tensor_names,
                source_format: GgufTensorType::Q4_0,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: gemma4_q4_qkv_projection_kernel(),
            },
            Gemma4PlanGroupSpec {
                tensor_names: vec![format!("{p}.ffn_gate.weight"), format!("{p}.ffn_up.weight")],
                group_members: vec![format!("{p}.ffn_gate.weight"), format!("{p}.ffn_up.weight")],
                source_format: GgufTensorType::Q4_0,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: gemma4_q4_gate_up_projection_kernel(),
            },
            Gemma4PlanGroupSpec {
                tensor_names: vec![format!("{p}.attn_output.weight")],
                group_members: vec![format!("{p}.attn_output.weight")],
                source_format: GgufTensorType::Q4_0,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: gemma4_q4_projection_kernel(),
            },
            Gemma4PlanGroupSpec {
                tensor_names: vec![format!("{p}.ffn_down.weight")],
                group_members: vec![format!("{p}.ffn_down.weight")],
                source_format: GgufTensorType::Q4_0,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: gemma4_ffn_down_projection_kernel(),
            },
            Gemma4PlanGroupSpec {
                tensor_names: vec![format!("{p}.proj.weight")],
                group_members: vec![format!("{p}.proj.weight")],
                source_format: GgufTensorType::Q4_0,
                selected_format: GgufTensorType::Q4_0,
                selected_kernel: gemma4_q4_projection_kernel(),
            },
        ]);
    }
    Ok(specs)
}

pub(crate) fn validate_gemma4_quantization_plan_groups(
    plan: &QuantizationPlan,
    model: &GgufModel,
    layers: usize,
) -> Result<()> {
    let selected_weight_format = match (
        plan.selected_format("token_embd.weight"),
        plan.selected_format("per_layer_token_embd.weight"),
    ) {
        (Some(GgufTensorType::Q4_0), Some(GgufTensorType::Q4_0)) => Gemma4WeightFormat::AllQ4,
        (Some(GgufTensorType::Q6K), Some(GgufTensorType::Q6K)) => Gemma4WeightFormat::MixedQ4Q6,
        (Some(left), Some(right)) => {
            bail!(
                "quantization plan selects incompatible Gemma vocabulary formats: token_embd={left:?}, per_layer_token_embd={right:?}"
            );
        }
        _ => bail!("quantization plan must select both Gemma vocabulary tensors together"),
    };
    let expected_specs = gemma4_plan_group_specs(model, layers, selected_weight_format)?;
    let mut actual_groups = BTreeMap::<Vec<String>, Vec<(String, &QuantizationPlanTensor)>>::new();
    for (name, entry) in &plan.tensors {
        let mut members = entry.group_members.clone();
        members.sort();
        members.dedup();
        actual_groups
            .entry(members)
            .or_default()
            .push((name.clone(), entry));
    }
    let mut expected_groups = BTreeMap::<Vec<String>, &Gemma4PlanGroupSpec>::new();
    for spec in &expected_specs {
        let mut members = spec.group_members.clone();
        members.sort();
        members.dedup();
        expected_groups.insert(members, spec);
    }
    ensure!(
        actual_groups.len() == expected_groups.len(),
        "quantization plan group coverage is incomplete: expected {:?}, got {:?}",
        expected_groups.keys().collect::<Vec<_>>(),
        actual_groups.keys().collect::<Vec<_>>()
    );
    for (members, spec) in expected_groups {
        let entries = actual_groups
            .get(&members)
            .with_context(|| format!("quantization plan is missing group {:?}", members))?;
        let expected_names = spec
            .tensor_names
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let actual_names = entries
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            actual_names == expected_names,
            "quantization plan group coverage is incomplete: expected {:?}, got {:?}",
            expected_names,
            actual_names
        );
        for (name, entry) in entries {
            ensure!(
                entry.source_format == spec.source_format,
                "quantization plan source format mismatch for `{name}`: plan={:?}, expected={:?}",
                entry.source_format,
                spec.source_format
            );
            ensure!(
                entry.selected_format == spec.selected_format,
                "quantization plan selected format mismatch for `{name}`: plan={:?}, expected={:?}",
                entry.selected_format,
                spec.selected_format
            );
            ensure!(
                entry.selected_kernel == spec.selected_kernel,
                "quantization plan selected kernel mismatch for `{name}`: plan={}, expected={}",
                entry.selected_kernel,
                spec.selected_kernel
            );
        }
    }
    Ok(())
}

fn gemma4_tensor_exists(model: &GgufModel, name: &str) -> bool {
    model.tensors.iter().any(|tensor| tensor.name == name)
}

fn summary_to_value(summary: &Gemma4QuantizationPreflightSummary) -> Value {
    json!({
    "runs": summary.runs,
    "median_decode_tok_s": summary.median_decode_tok_s,
    "median_gpu_ns": summary.median_gpu_ns,
    "prompt_token_sha256": summary.prompt_token_sha256,
    "generated_token_sha256": summary.generated_token_sha256,
    "measured_generated_token_sha256": summary.measured_generated_token_sha256,
    "first_eos_position": summary.first_eos_position,
    "selected_kernels": summary.selected_kernels,
    "resident_bytes": summary.resident_bytes,
    "kv_cache_bytes": summary.kv_cache_bytes,
    "weight_upload_bytes": summary.weight_upload_bytes,
    "readback_bytes": summary.readback_bytes,
    "prompt_tokens": summary.prompt_tokens,
    "generated_tokens": summary.generated_tokens,
    "measured_decode_tokens": summary.measured_decode_tokens,
        "finish_reason": summary.finish_reason,
        "weight_format": summary.weight_format,
        "pass": summary.pass,
        "rejection_reason": summary.rejection_reason,
        "short_median_decode_tok_s": summary.short_median_decode_tok_s,
        "short_median_gpu_ns": summary.short_median_gpu_ns,
        "short_prompt_token_sha256": summary.short_prompt_token_sha256,
        "short_generated_token_sha256": summary.short_generated_token_sha256,
        "short_measured_generated_token_sha256": summary.short_measured_generated_token_sha256,
        "short_first_eos_position": summary.short_first_eos_position,
        "short_selected_kernels": summary.short_selected_kernels,
        "short_resident_bytes": summary.short_resident_bytes,
        "short_kv_cache_bytes": summary.short_kv_cache_bytes,
        "short_weight_upload_bytes": summary.short_weight_upload_bytes,
        "short_readback_bytes": summary.short_readback_bytes,
        "short_prompt_tokens": summary.short_prompt_tokens,
        "short_generated_tokens": summary.short_generated_tokens,
        "short_measured_decode_tokens": summary.short_measured_decode_tokens,
        "short_finish_reason": summary.short_finish_reason,
    })
}

fn hardware_identity(model: &Gemma4E2bModel) -> String {
    let info = model.runtime().device_info();
    format!("{}#{}", info.name, info.registry_id)
}

fn token_ids_sha256(tokens: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn median_u64(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut values = values.to_vec();
    values.sort_unstable();
    values[values.len() / 2]
}

fn unique_string<'a>(mut values: impl Iterator<Item = &'a str>) -> Result<String> {
    let first = values.next().context("missing values")?.to_owned();
    ensure!(values.all(|value| value == first), "values are not unique");
    Ok(first)
}

fn unique_option<T: Eq + Clone>(mut values: impl Iterator<Item = Option<T>>) -> Result<Option<T>> {
    let first = values.next().context("missing values")?;
    ensure!(values.all(|value| value == first), "values are not unique");
    Ok(first)
}

fn unique_value(mut values: impl Iterator<Item = Value>) -> Result<Value> {
    let first = values.next().context("missing values")?;
    ensure!(values.all(|value| value == first), "values are not unique");
    Ok(first)
}

fn unique_u64(mut values: impl Iterator<Item = u64>) -> Result<u64> {
    let first = values.next().context("missing values")?;
    ensure!(values.all(|value| value == first), "values are not unique");
    Ok(first)
}

fn unique_usize(mut values: impl Iterator<Item = usize>) -> Result<usize> {
    let first = values.next().context("missing values")?;
    ensure!(values.all(|value| value == first), "values are not unique");
    Ok(first)
}

fn format_name(format: GgufTensorType) -> &'static str {
    match format {
        GgufTensorType::F32 => "f32",
        GgufTensorType::F16 => "f16",
        GgufTensorType::Q4_0 => "q4_0",
        GgufTensorType::Q8_0 => "q8_0",
        GgufTensorType::Q6K => "q6_k",
    }
}

fn write_json_value_atomically(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value)?;
    let temp_path = temp_path(path);
    fs::write(&temp_path, &bytes)
        .with_context(|| format!("write temporary output {}", temp_path.display()))?;
    fs::rename(&temp_path, path)
        .with_context(|| format!("move {} to {}", temp_path.display(), path.display()))?;
    Ok(())
}

fn temp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("output.json");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.with_file_name(format!(".{file_name}.{stamp}.{}.tmp", std::process::id()))
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = env::var_os(key);
        unsafe { env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(value) => env::set_var(self.key, value),
                None => env::remove_var(self.key),
            }
        }
    }
}

#[cfg(test)]
fn build_selection_report(
    summary: &FixedWorkloadSummary,
    weight_format: Gemma4WeightFormat,
) -> Gemma4QuantizationPreflightSummary {
    Gemma4QuantizationPreflightSummary {
        runs: 0,
        median_decode_tok_s: summary.median_decode_tok_s,
        median_gpu_ns: summary.median_gpu_ns,
        prompt_token_sha256: summary.prompt_token_sha256.clone(),
        generated_token_sha256: summary.generated_token_sha256.clone(),
        measured_generated_token_sha256: summary.measured_generated_token_sha256.clone(),
        first_eos_position: summary.first_eos_position,
        selected_kernels: summary.selected_kernels.clone(),
        resident_bytes: summary.resident_bytes,
        kv_cache_bytes: summary.kv_cache_bytes,
        weight_upload_bytes: summary.weight_upload_bytes,
        readback_bytes: summary.readback_bytes,
        prompt_tokens: summary.prompt_tokens,
        generated_tokens: summary.generated_tokens,
        measured_decode_tokens: summary.measured_decode_tokens,
        finish_reason: summary.finish_reason.clone(),
        weight_format: weight_format.as_str().to_owned(),
        pass: summary.pass,
        rejection_reason: None,
        short_median_decode_tok_s: summary.median_decode_tok_s,
        short_median_gpu_ns: summary.median_gpu_ns,
        short_prompt_token_sha256: summary.prompt_token_sha256.clone(),
        short_generated_token_sha256: summary.generated_token_sha256.clone(),
        short_measured_generated_token_sha256: summary.measured_generated_token_sha256.clone(),
        short_first_eos_position: summary.first_eos_position,
        short_selected_kernels: summary.selected_kernels.clone(),
        short_resident_bytes: summary.resident_bytes,
        short_kv_cache_bytes: summary.kv_cache_bytes,
        short_weight_upload_bytes: summary.weight_upload_bytes,
        short_readback_bytes: summary.readback_bytes,
        short_prompt_tokens: summary.prompt_tokens,
        short_generated_tokens: summary.generated_tokens,
        short_measured_decode_tokens: summary.measured_decode_tokens,
        short_finish_reason: summary.finish_reason.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atlas_core::{GgufTensorType, GgufWriter};

    fn push_test_tensor(writer: &mut GgufWriter, name: &str, tensor_type: GgufTensorType) {
        const ELEMENTS: usize = 256;
        let bytes = match tensor_type {
            GgufTensorType::F16 => vec![0; ELEMENTS * 2],
            GgufTensorType::F32 => vec![0; ELEMENTS * 4],
            GgufTensorType::Q4_0 => vec![0; (ELEMENTS / 32) * tensor_type.block_bytes()],
            GgufTensorType::Q6K => vec![0; (ELEMENTS / 256) * tensor_type.block_bytes()],
            GgufTensorType::Q8_0 => vec![0; (ELEMENTS / 32) * tensor_type.block_bytes()],
        };
        writer
            .push_tensor(name, vec![ELEMENTS], tensor_type, bytes)
            .unwrap();
    }

    fn test_model(shared_kv_layers: &[bool]) -> atlas_core::GgufModel {
        let mut writer = GgufWriter::new();
        push_test_tensor(&mut writer, "token_embd.weight", GgufTensorType::Q6K);
        push_test_tensor(
            &mut writer,
            "per_layer_token_embd.weight",
            GgufTensorType::Q6K,
        );
        push_test_tensor(
            &mut writer,
            "per_layer_model_proj.weight",
            GgufTensorType::F16,
        );
        for (layer, has_kv) in shared_kv_layers.iter().copied().enumerate() {
            let prefix = format!("blk.{layer}");
            push_test_tensor(
                &mut writer,
                &format!("{prefix}.attn_q.weight"),
                GgufTensorType::Q4_0,
            );
            if has_kv {
                push_test_tensor(
                    &mut writer,
                    &format!("{prefix}.attn_k.weight"),
                    GgufTensorType::Q4_0,
                );
                push_test_tensor(
                    &mut writer,
                    &format!("{prefix}.attn_v.weight"),
                    GgufTensorType::Q4_0,
                );
            }
            for suffix in [
                "ffn_gate.weight",
                "ffn_up.weight",
                "attn_output.weight",
                "ffn_down.weight",
                "proj.weight",
            ] {
                push_test_tensor(
                    &mut writer,
                    &format!("{prefix}.{suffix}"),
                    GgufTensorType::Q4_0,
                );
            }
        }
        atlas_core::GgufModel::from_bytes(writer.finish().unwrap()).unwrap()
    }

    fn build_test_plan(
        model: &atlas_core::GgufModel,
        selected_weight_format: Gemma4WeightFormat,
        layers: usize,
    ) -> QuantizationPlan {
        let mut plan = QuantizationPlan::new("gemma", "model-sha");
        plan.state = QUANTIZATION_PLAN_STATE_READY.into();
        plan.oracle_sha256 = "oracle-sha".into();
        plan.hardware_identity = "Apple GPU 42".into();
        plan.profiler_configuration = QuantizationPlanProfilerConfig {
            mode: "auto".into(),
            prompt_sha256: "prompt-sha".into(),
            decode_tokens: 32,
            runs: 2,
        };
        plan.max_abs_logit_delta = crate::quantization_plan::DEFAULT_MAX_ABS_LOGIT_DELTA;
        for spec in gemma4_plan_group_specs(model, layers, selected_weight_format).unwrap() {
            for tensor_name in &spec.tensor_names {
                plan.tensors.insert(
                    tensor_name.clone(),
                    QuantizationPlanTensor {
                        group_members: spec.group_members.clone(),
                        source_format: spec.source_format,
                        selected_format: spec.selected_format,
                        selected_kernel: spec.selected_kernel.to_owned(),
                        max_abs_logit_delta: crate::quantization_plan::DEFAULT_MAX_ABS_LOGIT_DELTA,
                        median_gpu_ns: 1_000,
                        baseline_gpu_ns: 2_000,
                        resident_bytes: 1,
                        upload_bytes: 1,
                        parity_digest: "digest".into(),
                        parity: true,
                        rejection_reason: None,
                    },
                );
            }
        }
        plan
    }

    #[test]
    fn parse_policy_accepts_expected_values() {
        assert_eq!(
            parse_preflight_policy(Some("auto")).unwrap(),
            Gemma4QuantizationPreflightPolicy::Auto
        );
        assert_eq!(
            parse_preflight_policy(Some("disabled")).unwrap(),
            Gemma4QuantizationPreflightPolicy::Disabled
        );
        assert_eq!(
            parse_preflight_policy(Some("verify")).unwrap(),
            Gemma4QuantizationPreflightPolicy::Verify
        );
    }

    #[test]
    fn parse_weight_format_override_honors_mixed_default_and_explicit_values() {
        assert_eq!(parse_weight_format_override(None).unwrap(), None);
        assert_eq!(
            parse_weight_format_override(Some("mixed_q4_q6")).unwrap(),
            None
        );
        assert_eq!(
            parse_weight_format_override(Some("all_q4")).unwrap(),
            Some(Gemma4WeightFormat::AllQ4)
        );
    }

    #[test]
    fn resolve_sidecar_action_prefers_explicit_override_and_disabled_mixed_behavior() {
        assert_eq!(
            resolve_sidecar_action(
                Gemma4QuantizationPreflightInvocation::AutoLoad,
                Gemma4QuantizationPreflightPolicy::Auto,
                true,
                false
            ),
            Gemma4QuantizationSidecarAction::Skip
        );
        assert_eq!(
            resolve_sidecar_action(
                Gemma4QuantizationPreflightInvocation::AutoLoad,
                Gemma4QuantizationPreflightPolicy::Disabled,
                false,
                false
            ),
            Gemma4QuantizationSidecarAction::Skip
        );
    }

    #[test]
    fn resolve_sidecar_action_profiles_invalid_auto_sidecars_and_uses_ready_plans() {
        assert_eq!(
            resolve_sidecar_action(
                Gemma4QuantizationPreflightInvocation::AutoLoad,
                Gemma4QuantizationPreflightPolicy::Auto,
                false,
                false
            ),
            Gemma4QuantizationSidecarAction::ProfileAndRewrite
        );
        assert_eq!(
            resolve_sidecar_action(
                Gemma4QuantizationPreflightInvocation::AutoLoad,
                Gemma4QuantizationPreflightPolicy::Auto,
                false,
                true
            ),
            Gemma4QuantizationSidecarAction::UseCachedPlan
        );
        assert_eq!(
            resolve_sidecar_action(
                Gemma4QuantizationPreflightInvocation::CliProfile,
                Gemma4QuantizationPreflightPolicy::Disabled,
                false,
                true
            ),
            Gemma4QuantizationSidecarAction::ProfileAndRewrite
        );
    }

    #[test]
    fn plan_gpu_ns_helper_uses_elapsed_time_not_throughput() {
        let summary = Gemma4QuantizationPreflightSummary {
            runs: 2,
            median_decode_tok_s: 1.5,
            median_gpu_ns: 98_765,
            prompt_token_sha256: "prompt".into(),
            generated_token_sha256: "generated".into(),
            measured_generated_token_sha256: "measured".into(),
            first_eos_position: Some(7),
            selected_kernels: json!({"attention": "attention"}),
            resident_bytes: 11,
            kv_cache_bytes: 13,
            weight_upload_bytes: 17,
            readback_bytes: 19,
            prompt_tokens: 23,
            generated_tokens: 29,
            measured_decode_tokens: 31,
            finish_reason: "max_tokens".into(),
            weight_format: "all_q4".into(),
            pass: true,
            rejection_reason: None,
            short_median_decode_tok_s: 1.5,
            short_median_gpu_ns: 98_765,
            short_prompt_token_sha256: "prompt".into(),
            short_generated_token_sha256: "generated".into(),
            short_measured_generated_token_sha256: "measured".into(),
            short_first_eos_position: Some(7),
            short_selected_kernels: json!({"attention": "attention"}),
            short_resident_bytes: 11,
            short_kv_cache_bytes: 13,
            short_weight_upload_bytes: 17,
            short_readback_bytes: 19,
            short_prompt_tokens: 23,
            short_generated_tokens: 29,
            short_measured_decode_tokens: 31,
            short_finish_reason: "max_tokens".into(),
        };
        assert_eq!(plan_gpu_ns(&summary), 98_765);
    }

    #[test]
    fn gemma4_plan_group_specs_cover_current_resident_groups_and_validate() {
        let model = test_model(&[true, true]);
        let specs = gemma4_plan_group_specs(&model, 2, Gemma4WeightFormat::AllQ4).unwrap();
        assert_eq!(specs.len(), 12);
        assert!(specs
            .iter()
            .any(|spec| spec.tensor_names == vec!["per_layer_model_proj.weight"]));
        assert!(specs.iter().any(|spec| spec.tensor_names
            == vec![
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight"
            ]));
        assert!(specs
            .iter()
            .all(|spec| spec.selected_format != GgufTensorType::Q8_0));

        let plan = build_test_plan(&model, Gemma4WeightFormat::AllQ4, 2);
        validate_gemma4_quantization_plan_groups(&plan, &model, 2).unwrap();

        let mut incomplete = plan.clone();
        incomplete.tensors.remove("blk.1.ffn_down.weight");
        let error = validate_gemma4_quantization_plan_groups(&incomplete, &model, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("group coverage is incomplete"));

        let mut wrong_format = plan.clone();
        wrong_format
            .tensors
            .get_mut("blk.0.attn_q.weight")
            .unwrap()
            .selected_format = GgufTensorType::Q6K;
        let error = validate_gemma4_quantization_plan_groups(&wrong_format, &model, 2)
            .unwrap_err()
            .to_string();
        assert!(error.contains("selected format mismatch"));
    }

    #[test]
    fn gemma4_plan_group_specs_preserve_shared_kv_absence() {
        let model = test_model(&[true, false]);
        let specs = gemma4_plan_group_specs(&model, 2, Gemma4WeightFormat::MixedQ4Q6).unwrap();
        assert!(specs
            .iter()
            .any(|spec| spec.tensor_names == vec!["blk.1.attn_q.weight"]));
        assert!(specs.iter().any(|spec| spec.tensor_names
            == vec![
                "blk.0.attn_q.weight",
                "blk.0.attn_k.weight",
                "blk.0.attn_v.weight"
            ]));

        let plan = build_test_plan(&model, Gemma4WeightFormat::MixedQ4Q6, 2);
        validate_gemma4_quantization_plan_groups(&plan, &model, 2).unwrap();
    }

    #[test]
    fn selection_report_uses_median_and_preserves_weight_format() {
        let summary = FixedWorkloadSummary {
            median_decode_tok_s: 42.0,
            median_gpu_ns: 1_234,
            prompt_token_sha256: "prompt".into(),
            generated_token_sha256: "generated".into(),
            measured_generated_token_sha256: "measured".into(),
            first_eos_position: Some(7),
            selected_kernels: json!({"attention": "attention"}),
            resident_bytes: 11,
            kv_cache_bytes: 13,
            weight_upload_bytes: 17,
            readback_bytes: 19,
            prompt_tokens: 23,
            generated_tokens: 29,
            measured_decode_tokens: 31,
            finish_reason: "max_tokens".into(),
            pass: true,
        };
        let report = build_selection_report(&summary, Gemma4WeightFormat::AllQ4);
        assert_eq!(report.weight_format, "all_q4");
        assert_eq!(report.median_decode_tok_s, 42.0);
        assert_eq!(report.first_eos_position, Some(7));
    }

    #[test]
    fn candidate_accounting_may_differ_from_baseline() {
        let summary = FixedWorkloadSummary {
            median_decode_tok_s: 42.0,
            median_gpu_ns: 1_000,
            prompt_token_sha256: "prompt".into(),
            generated_token_sha256: "generated".into(),
            measured_generated_token_sha256: "measured".into(),
            first_eos_position: Some(7),
            selected_kernels: json!({"weight_format": "mixed_q4_q6"}),
            resident_bytes: 1_000,
            kv_cache_bytes: 2_000,
            weight_upload_bytes: 1_000,
            readback_bytes: 1,
            prompt_tokens: 23,
            generated_tokens: 29,
            measured_decode_tokens: 31,
            finish_reason: "max_tokens".into(),
            pass: true,
        };
        let baseline = build_selection_report(&summary, Gemma4WeightFormat::MixedQ4Q6);
        let mut candidate = baseline.clone();
        candidate.weight_format = Gemma4WeightFormat::AllQ4.as_str().into();
        candidate.selected_kernels = json!({"weight_format": "all_q4"});
        candidate.short_selected_kernels = candidate.selected_kernels.clone();
        candidate.median_decode_tok_s = 44.0;
        candidate.short_median_decode_tok_s = 44.0;
        candidate.resident_bytes = 800;
        candidate.short_resident_bytes = 800;
        candidate.weight_upload_bytes = 800;
        candidate.short_weight_upload_bytes = 800;

        let (selected, _, rejections, _, _, _) = select_candidate(&baseline, &candidate);
        assert_eq!(selected, Gemma4WeightFormat::AllQ4);
        assert!(!rejections
            .values()
            .any(|reason| reason.contains("accounting")));
    }
}
