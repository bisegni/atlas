#![recursion_limit = "256"]

use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{GgufModel, GgufTensorType};
use atlas_metal::MetalRuntime;
use atlas_model::{
    Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
    gemma4_executor::{
        Gemma4E2bExecutor, Gemma4FinishReason, Gemma4Generation, Gemma4KvCacheType,
        Gemma4Q4AttentionMode, Gemma4SelectedGroupFormat,
    },
    gemma4_quantization_preflight::{
        Gemma4QuantizationPreflightInvocation, run_gemma4_quantization_preflight,
    },
    quantization_plan::default_sidecar_path,
    render_gemma4_chat,
};
use atlas_profiler::{
    AttentionKind, AttentionScanPass, BenchmarkCompatibility, ClockDomain, DecodeScope,
    MeasuredWindow, OperationFamily, PhaseSummary, ProfileCounters, ProfileEvent, ProfileMode,
    ProfilePhase, ProfileScope, ProfileWorkload, Profiler, ScopeContract, TimingBoundary,
    TimingKind,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

mod providers;

static CHAT_INTERRUPTED: AtomicBool = AtomicBool::new(false);
const MODEL_MANIFEST: &str = "models/manifest.toml";
const CHAT_PERFORMANCE_LOG: &str = "artifacts/chat-performance.jsonl";
const GEMMA_DECODE_PROFILE_LOG: &str = "artifacts/phase-12a/gemma4-resident-decode-profile.jsonl";
const GEMMA_FIXED_BENCHMARK_LOG: &str = "artifacts/phase-12a-perf/gemma4-fixed-workload.jsonl";
const FLASH16_DIAGNOSIS_DIR: &str = "artifacts/flash16-diagnosis";

#[derive(Debug, PartialEq, Eq)]
struct GemmaProfileArgs {
    model_id: String,
    prompt: String,
    warmup_decode_tokens: usize,
    decode_tokens: usize,
    max_context: usize,
    kv_cache_type: Gemma4KvCacheType,
    q4_attention_mode: Gemma4Q4AttentionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuCountersMode {
    Auto,
    Required,
}

impl GpuCountersMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "required" => Ok(Self::Required),
            _ => bail!("unknown GPU counter mode `{value}`; expected auto or required"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct GemmaBenchmarkArgs {
    model_id: String,
    prompt: String,
    warmup_decode_tokens: usize,
    measured_decode_tokens: usize,
    max_context: usize,
    kv_cache_type: Gemma4KvCacheType,
    q4_attention_mode: Gemma4Q4AttentionMode,
}

#[derive(Debug)]
struct ModelManifest {
    models: Vec<ModelRecord>,
}
#[derive(Debug)]
struct ModelRecord {
    id: String,
    source: String,
    revision: String,
    path: PathBuf,
    architecture: String,
    tokenizer: PathBuf,
    model_file: PathBuf,
    embedded_tokenizer: bool,
    format: String,
    bytes: u64,
    files: Vec<ModelFile>,
}
#[derive(Debug)]
struct ModelFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl ModelRecord {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.architecture == "gemma4"
                && self.embedded_tokenizer
                && self.tokenizer == Path::new("embedded")
                && !self.model_file.as_os_str().is_empty(),
            "Atlas supports only Gemma 4 E2B with an embedded tokenizer"
        );
        ensure!(
            self.format == "gguf-gemma4-q4_0",
            "Atlas supports only Gemma 4 E2B Q4_0 GGUF"
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("metal-info") => metal_info(),
        Some("provider") => provider_command(&args[1..]),
        Some("model") => model_command(&args[1..]),
        Some("generate") => generate(&args[1..]),
        Some("chat") => chat(&args[1..]),
        Some("profile") => profile(&args[1..]),
        Some("benchmark") => benchmark(&args[1..]),
        _ => bail!(
            "usage: atlas-cli generate|chat|profile|benchmark --model gemma4-e2b-q4_0 ... | atlas-cli model search|download|inspect|verify --model ID"
        ),
    }
}

fn provider_command(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .context("provider command requires a subcommand")?;
    let provider = args
        .get(1)
        .map(String::as_str)
        .unwrap_or(providers::HUGGING_FACE);
    match command.as_str() {
        "status" => {
            let (source, _) = providers::token(provider)?;
            let authentication = match source {
                providers::AuthSource::Environment => "environment",
                providers::AuthSource::Keychain => "keychain",
                providers::AuthSource::Missing => "unauthenticated",
            };
            println!(
                "{}",
                json!({"provider":provider,"authentication":authentication,"default_provider":providers::load_default_provider()?})
            );
        }
        "default" => {
            let value = args
                .get(1)
                .context("provider default requires a provider ID or --clear")?;
            providers::set_default_provider((value != "--clear").then_some(value))?;
            println!(
                "{}",
                json!({"default_provider":if value == "--clear" { None } else { Some(value) }})
            );
        }
        "login" => {
            ensure!(
                provider == providers::HUGGING_FACE,
                "only Hugging Face is supported"
            );
            eprint!("Hugging Face access token: ");
            let token = rpassword::read_password()?;
            providers::validate_hugging_face_token(&token)?;
            providers::store_token(provider, &token)?;
        }
        "logout" => providers::logout(provider)?,
        _ => bail!("provider command must be login, logout, status, or default"),
    }
    Ok(())
}

fn chat(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    let (model_id, prompt, max_tokens, show_thoughts, kv_cache_type, q4_attention_mode) =
        parse_chat_args(args)?;
    let selection = resolve_model(&model_id)?;
    let model = load_verified_model(&selection)?;
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        4096,
        kv_cache_type,
        q4_attention_mode,
    )?;
    let mut messages = Vec::new();
    if let Some(prompt) = prompt {
        messages.push(Gemma4ChatMessage::new(Gemma4ChatRole::User, prompt));
        run_chat_turn(
            &model,
            &mut executor,
            &messages,
            max_tokens,
            show_thoughts,
            &selection,
        )?;
        return Ok(());
    }
    eprintln!("Atlas Gemma 4 chat. Commands: /reset, /help, /quit");
    let stdin = io::stdin();
    loop {
        print!("you> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        match line.trim() {
            "/quit" => break,
            "/help" => {
                println!("/reset clears the conversation; /quit exits");
                continue;
            }
            "/reset" => {
                messages.clear();
                executor.reset();
                println!("conversation reset");
                continue;
            }
            "" => continue,
            text => messages.push(Gemma4ChatMessage::new(Gemma4ChatRole::User, text)),
        }
        let visible = run_chat_turn(
            &model,
            &mut executor,
            &messages,
            max_tokens,
            show_thoughts,
            &selection,
        )?;
        messages.push(Gemma4ChatMessage::new(Gemma4ChatRole::Model, visible));
    }
    Ok(())
}

fn run_chat_turn(
    model: &Gemma4E2bModel,
    executor: &mut Gemma4E2bExecutor<'_>,
    messages: &[Gemma4ChatMessage],
    requested: Option<usize>,
    show_thoughts: bool,
    selection: &ModelRecord,
) -> Result<String> {
    let prompt = render_gemma4_chat(messages)?;
    let prompt_tokens = model.tokenize(&prompt)?.len();
    let max_tokens = requested.unwrap_or(4096usize.saturating_sub(prompt_tokens));
    ensure!(
        max_tokens > 0 && prompt_tokens + max_tokens <= 4096,
        "Gemma chat context exhausted"
    );
    let mut visible = String::new();
    let mut filter = ThoughtFilter::default();
    let generation =
        executor.generate_greedy_chat_stream(&prompt, max_tokens, &CHAT_INTERRUPTED, |event| {
            filter.push(&event.text);
            let visible_fragment = filter.visible_delta();
            let fragment = if show_thoughts {
                event.text.replace("<turn|>", "")
            } else {
                visible_fragment.clone()
            };
            if !fragment.is_empty() {
                print!("{fragment}");
                io::stdout().flush()?;
            }
            visible.push_str(&visible_fragment);
            Ok(())
        })?;
    let (final_visible, thoughts) = filter.finish();
    let tail = final_visible
        .strip_prefix(&visible)
        .unwrap_or(&final_visible);
    if !tail.is_empty() {
        print!("{tail}");
        visible.push_str(tail);
    }
    if show_thoughts && !thoughts.is_empty() {
        eprintln!("\nthoughts> {thoughts}");
    }
    write_flash16_diagnosis(selection, &generation)?;
    println!();
    emit_metrics(
        selection,
        &generation,
        &visible,
        max_tokens,
        requested.is_none(),
    )?;
    Ok(visible)
}

fn digest_hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_flash16_diagnosis(selection: &ModelRecord, generation: &Gemma4Generation) -> Result<()> {
    if generation.flash16_trace.is_empty() {
        return Ok(());
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_millis();
    let path = Path::new(FLASH16_DIAGNOSIS_DIR).join(format!("{stamp}-flash16.jsonl"));
    fs::create_dir_all(FLASH16_DIAGNOSIS_DIR)?;
    let mut file = fs::File::create(&path)?;
    writeln!(
        file,
        "{}",
        json!({
            "event": "flash16_diagnosis",
            "model_id": selection.id,
            "executor": "resident",
            "q4_attention_mode": generation.metrics.q4_attention_mode.as_str(),
            "attention_kernel": generation.metrics.attention_kernel,
            "records": generation.flash16_trace.len(),
        })
    )?;
    for record in &generation.flash16_trace {
        writeln!(
            file,
            "{}",
            json!({
                "event": "flash16_token",
                "token_index": record.token_index,
                "token_id": record.token_id,
                "selected_token_matches_top_logit": record.selected_token_matches_top_logit,
                "top_token_id": record.top_token_id,
                "top_logit": record.top_logit,
                "runner_up_token_id": record.runner_up_token_id,
                "runner_up_logit": record.runner_up_logit,
                "top_logit_margin": record.top_logit_margin,
                "logits_non_finite": record.logits_non_finite,
                "logits_digest": digest_hex(&record.logits_digest),
                "q": {"l2_norm": record.q_l2_norm, "max_abs": record.q_max_abs, "non_finite": record.q_non_finite, "digest": digest_hex(&record.q_digest)},
                "attention": {"l2_norm": record.attention_l2_norm, "max_abs": record.attention_max_abs, "non_finite": record.attention_non_finite, "digest": digest_hex(&record.attention_digest)},
                "state": {"l2_norm": record.state_l2_norm, "max_abs": record.state_max_abs, "non_finite": record.state_non_finite, "digest": digest_hex(&record.state_digest)},
                "layer_states": record.layer_states.iter().map(|layer| json!({
                    "layer_index": layer.layer_index,
                    "l2_norm": layer.l2_norm,
                    "max_abs": layer.max_abs,
                    "non_finite": layer.non_finite,
                    "digest": digest_hex(&layer.digest),
                })).collect::<Vec<_>>(),
                "output_projection_cpu": {"top_logit": record.output_projection_cpu_top_logit, "runner_up_logit": record.output_projection_cpu_runner_up_logit, "max_abs_delta": record.output_projection_cpu_max_abs_delta},
                "detail_triggered": record.detail_triggered,
            })
        )?;
    }
    eprintln!("[atlas][flash16-diagnosis] wrote {}", path.display());
    Ok(())
}

#[derive(Default)]
struct ThoughtFilter {
    pending: String,
    in_thought: bool,
    visible: String,
    thoughts: String,
    reported: usize,
}
impl ThoughtFilter {
    fn push(&mut self, text: &str) {
        self.pending.push_str(text);
        self.strip_end_turn_markers();
        loop {
            if self.in_thought {
                if let Some(end) = self.pending.find("</think>") {
                    self.thoughts.push_str(&self.pending[..end]);
                    self.pending.drain(..end + 8);
                    self.in_thought = false;
                } else {
                    break;
                }
            } else if let Some(start) = self.pending.find("<think>") {
                self.visible.push_str(&self.pending[..start]);
                self.pending.drain(..start + 7);
                self.in_thought = true;
            } else {
                // The end-of-turn marker is normally one token, but retain a
                // possible partial marker so it cannot leak into the terminal
                // or become stored assistant content when token text is split.
                let retained = trailing_marker_prefix_len(&self.pending, "<turn|>");
                let visible_len = self.pending.len() - retained;
                self.visible.push_str(&self.pending[..visible_len]);
                self.pending.drain(..visible_len);
                break;
            }
        }
    }

    fn strip_end_turn_markers(&mut self) {
        const END_TURN: &str = "<turn|>";
        while let Some(start) = self.pending.find(END_TURN) {
            self.pending.drain(start..start + END_TURN.len());
        }
    }
    fn visible_delta(&mut self) -> String {
        let value = self.visible[self.reported..].to_owned();
        self.reported = self.visible.len();
        value
    }
    fn finish(mut self) -> (String, String) {
        self.strip_end_turn_markers();
        if self.in_thought {
            self.thoughts.push_str(&self.pending);
        } else {
            let retained = trailing_marker_prefix_len(&self.pending, "<turn|>");
            let visible_len = self.pending.len() - retained;
            self.visible.push_str(&self.pending[..visible_len]);
        }
        (self.visible, self.thoughts)
    }
}

fn trailing_marker_prefix_len(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&len| text.ends_with(&marker[..len]))
        .unwrap_or(0)
}

fn emit_metrics(
    record: &ModelRecord,
    generation: &Gemma4Generation,
    visible: &str,
    max_tokens: usize,
    context_limit: bool,
) -> Result<()> {
    let finish_reason = match generation.finish_reason {
        Gemma4FinishReason::Eos => "eos",
        Gemma4FinishReason::MaxTokens => "max_tokens",
        Gemma4FinishReason::Cancelled => "cancelled",
    };
    let prefill = rate(
        generation.generation.prompt_token_ids.len(),
        generation.metrics.prefill,
    );
    let decode = rate(
        generation.metrics.decode_command_buffers as usize,
        generation.metrics.decode,
    );
    let record = json!({
        "event": "generation_metrics",
        "model_id": record.id,
        "executor": "resident",
        "format": "gguf-gemma4-q4_0",
        "weight_format": generation.metrics.weight_format.as_str(),
        "embedding_kernel": generation.metrics.embedding_kernel,
        "output_projection_kernel": generation.metrics.output_projection_kernel,
        "q4_projection_kernel": generation.metrics.q4_projection_kernel,
        "q4_qkv_projection_kernel": generation.metrics.q4_qkv_projection_kernel,
        "q4_gate_up_projection_kernel": generation.metrics.q4_gate_up_projection_kernel,
        "ffn_gate_up_activation_kernel": generation.metrics.ffn_gate_up_activation_kernel,
        "ffn_gate_up_scratch_bytes": generation.metrics.ffn_gate_up_scratch_bytes,
        "ple_composition_kernel": generation.metrics.ple_composition_kernel,
        "q4_batch_projection_kernel": generation.metrics.q4_batch_projection_kernel,
        "ffn_down_projection_kernel": generation.metrics.ffn_down_projection_kernel,
        "ple_projection_kernel": generation.metrics.ple_projection_kernel,
        "rms_norm_kernel": generation.metrics.rms_norm_kernel,
        "prompt_tokens": generation.generation.prompt_token_ids.len(),
        "generated_tokens": generation.generation.generated_token_ids.len(),
        "finish_reason": finish_reason,
        "max_new_tokens": max_tokens,
        "token_limit_source": if context_limit { "context" } else { "explicit" },
        "visible_chars": visible.chars().count(),
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "resident_bytes": generation.metrics.resident_bytes,
        "kv_cache_type": generation.metrics.kv_cache_type.as_str(),
        "kv_cache_bytes": generation.metrics.kv_cache_bytes,
        "weight_upload_bytes": generation.metrics.weight_upload_bytes,
        "readback_bytes": generation.metrics.readback_bytes,
        "command_buffers": generation.metrics.command_buffers,
        "prefill_command_buffers": generation.metrics.prefill_command_buffers,
        "decode_command_buffers": generation.metrics.decode_command_buffers,
        "prefill_path": generation.metrics.prefill_path,
        "prefill_chunk_size": generation.metrics.prefill_chunk_size,
        "prefill_chunks": generation.metrics.prefill_chunks,
        "quantization_preflight_state": generation.metrics.quantization_preflight_state,
        "quantization_plan": generation.metrics.quantization_plan_path,
        "selected_group_formats": selected_group_formats_json(&generation.metrics.selected_group_formats),
        "quantization_rejections": generation.metrics.quantization_rejections,
        "q4_attention_mode": generation.metrics.q4_attention_mode.as_str(),
        "attention_kernel": generation.metrics.attention_kernel,
        "timing": {
            "prefill_ms": generation.metrics.prefill.as_secs_f64() * 1000.0,
            "decode_ms": generation.metrics.decode.as_secs_f64() * 1000.0,
            "host_ms": generation.metrics.host_wall_time.as_secs_f64() * 1000.0
        }
    });
    eprintln!("{record}");
    append_jsonl(&record)?;
    eprintln!("chat performance log: {CHAT_PERFORMANCE_LOG}");
    Ok(())
}
fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

fn selected_group_formats_json(formats: &[Gemma4SelectedGroupFormat]) -> Vec<Value> {
    formats
        .iter()
        .map(|entry| {
            json!({
                "group": entry.group,
                "source_format": gguf_tensor_type_name(entry.source_format),
                "selected_format": gguf_tensor_type_name(entry.selected_format),
                "selected_kernel": entry.selected_kernel,
                "rejection_reason": entry.rejection_reason,
            })
        })
        .collect()
}

fn gguf_tensor_type_name(format: GgufTensorType) -> &'static str {
    match format {
        GgufTensorType::F32 => "f32",
        GgufTensorType::F16 => "f16",
        GgufTensorType::Q4_0 => "q4_0",
        GgufTensorType::Q8_0 => "q8_0",
        GgufTensorType::Q6K => "q6_k",
    }
}

fn generate(args: &[String]) -> Result<()> {
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = None;
    let mut greedy = false;
    let mut chat = false;
    let mut json_output = false;
    let mut kv_cache_type = Gemma4KvCacheType::Q4_0;
    // Flash16 remains an explicit Resident Q4-KV parity and performance
    // candidate until its exact-token and logit-digest gates pass on Metal.
    let mut q4_attention_mode = Gemma4Q4AttentionMode::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "--max-new-tokens" => {
                i += 1;
                max_tokens = Some(
                    args.get(i)
                        .context("--max-new-tokens needs a value")?
                        .parse()?,
                );
            }
            "--greedy" => greedy = true,
            "--chat" => chat = true,
            "--json" => json_output = true,
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type = Gemma4KvCacheType::parse(
                    args.get(i).context("--kv-cache-type needs a value")?,
                )?;
            }
            "--q4-attention-mode" => {
                i += 1;
                q4_attention_mode = Gemma4Q4AttentionMode::parse(
                    args.get(i).context("--q4-attention-mode needs a value")?,
                )?;
            }
            flag => bail!("unknown generate option: {flag}"),
        };
        i += 1;
    }
    ensure!(greedy, "generate requires --greedy");
    let selection = resolve_model(&model.context("--model is required")?)?;
    let gemma = load_verified_model(&selection)?;
    let prompt = prompt.context("--prompt is required")?;
    let prompt = if chat {
        render_gemma4_chat(&[Gemma4ChatMessage::new(Gemma4ChatRole::User, prompt)])?
    } else {
        prompt
    };
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &gemma,
        4096,
        kv_cache_type,
        q4_attention_mode,
    )?;
    let max_tokens = max_tokens.context("--max-new-tokens is required")?;
    let result = if chat {
        executor.generate_greedy_chat_stream(&prompt, max_tokens, &CHAT_INTERRUPTED, |_| Ok(()))?
    } else {
        executor.generate_greedy_stream(&prompt, max_tokens, &CHAT_INTERRUPTED, |_| Ok(()))?
    };
    write_flash16_diagnosis(&selection, &result)?;
    if json_output {
        let visible_text = if chat {
            visible_chat_completion(&result.generation.text, &prompt).to_owned()
        } else {
            result.generation.text.clone()
        };
        println!(
            "{}",
            json!({
                "event": "generation",
                "model_id": selection.id,
                "prompt": prompt,
                "prompt_token_ids": result.generation.prompt_token_ids,
                "generated_token_ids": result.generation.generated_token_ids,
                "finish_reason": match result.finish_reason {
                    Gemma4FinishReason::Eos => "eos",
                    Gemma4FinishReason::MaxTokens => "max_tokens",
                    Gemma4FinishReason::Cancelled => "cancelled",
                },
                "text": result.generation.text,
                "visible_text": visible_text,
                "executor": "resident",
                "kv_cache_type": result.metrics.kv_cache_type.as_str(),
                "q4_attention_mode": result.metrics.q4_attention_mode.as_str(),
                "attention_kernel": result.metrics.attention_kernel,
                "prefill_path": result.metrics.prefill_path,
                "resident_bytes": result.metrics.resident_bytes,
                "readback_bytes": result.metrics.readback_bytes,
            })
        );
        return Ok(());
    }
    println!(
        "prompt_token_ids: {:?}\ngenerated_token_ids: {:?}\ntext: {}",
        result.generation.prompt_token_ids,
        result.generation.generated_token_ids,
        result.generation.text
    );
    println!(
        "{}",
        json!({"event":"generation_metrics","model_id":selection.id,"executor":"resident","format":"gguf-gemma4-q4_0","weight_format":result.metrics.weight_format.as_str(),"embedding_kernel":result.metrics.embedding_kernel,"output_projection_kernel":result.metrics.output_projection_kernel,"q4_projection_kernel":result.metrics.q4_projection_kernel,"q4_qkv_projection_kernel":result.metrics.q4_qkv_projection_kernel,"q4_gate_up_projection_kernel":result.metrics.q4_gate_up_projection_kernel,"q4_batch_projection_kernel":result.metrics.q4_batch_projection_kernel,"ffn_down_projection_kernel":result.metrics.ffn_down_projection_kernel,"ple_projection_kernel":result.metrics.ple_projection_kernel,"rms_norm_kernel":result.metrics.rms_norm_kernel,"finish_reason":match result.finish_reason { Gemma4FinishReason::Eos => "eos", Gemma4FinishReason::MaxTokens => "max_tokens", Gemma4FinishReason::Cancelled => "cancelled" },"resident_bytes":result.metrics.resident_bytes,"peak_resident_bytes":result.metrics.peak_resident_bytes,"kv_cache_type":result.metrics.kv_cache_type.as_str(),"q4_attention_mode":result.metrics.q4_attention_mode.as_str(),"attention_kernel":result.metrics.attention_kernel,"kv_cache_bytes":result.metrics.kv_cache_bytes,"weight_upload_bytes":result.metrics.weight_upload_bytes,"upload_bytes":result.metrics.upload_bytes,"readback_bytes":result.metrics.readback_bytes,"dispatches":result.metrics.dispatches,"buffer_allocations":result.metrics.buffer_allocations,"gpu_execution_ms":result.metrics.gpu_execution_time.as_secs_f64()*1000.0,"command_buffers":result.metrics.command_buffers})
    );
    Ok(())
}

fn visible_chat_completion<'a>(protocol_text: &'a str, prompt: &str) -> &'a str {
    protocol_text
        .strip_prefix(prompt)
        .unwrap_or(protocol_text)
        .strip_suffix("<turn|>")
        .unwrap_or_else(|| protocol_text.strip_prefix(prompt).unwrap_or(protocol_text))
}

/// Diagnostic-only exact Metal timing for the growing-context Gemma decode
/// path. This command intentionally does not share the normal chat metrics
/// file because it submits every observed kernel separately.
fn profile(args: &[String]) -> Result<()> {
    if args.first().map(String::as_str) == Some("bottlenecks") {
        return profile_bottlenecks(&args[1..]);
    }
    let args = parse_profile_args(args)?;
    let record = resolve_model(&args.model_id)?;
    let model = load_verified_model(&record)?;
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        args.max_context,
        args.kv_cache_type,
        args.q4_attention_mode,
    )?;
    let prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
        Gemma4ChatRole::User,
        args.prompt.clone(),
    )])?;
    let profile = executor.profile_decode_with_timing(
        &prompt,
        args.warmup_decode_tokens,
        args.decode_tokens,
        true,
    )?;
    let samples = profile
        .samples
        .iter()
        .map(|sample| {
            json!({
                "decode_position": sample.decode_position,
                "scope": sample.scope,
                "context_position": sample.context_position,
                "attention_key_count": sample.attention_key_count,
                "full_attention_layers": sample.full_attention_layers,
                "sliding_attention_layers": sample.sliding_attention_layers,
                "resident_bytes": sample.resident_bytes,
                "readback_bytes": sample.readback_bytes,
                "kernels": sample.kernels.iter().map(|kernel| json!({
                    "family": kernel.family,
                    "dispatches": kernel.dispatches,
                    "gpu_ms": kernel.gpu_nanos as f64 / 1_000_000.0,
                    "cpu_encode_ms": kernel.cpu_encode_nanos as f64 / 1_000_000.0,
                })).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let output = json!({
        "event": "gemma4_resident_decode_profile",
        "model_id": record.id,
        "executor": "resident",
        "diagnostic": true,
        "exact_per_dispatch": true,
        "warmup_decode_tokens": profile.warmup_decode_tokens,
        "measured_decode_tokens": profile.measured_decode_tokens,
        "completed_decode_tokens": profile.completed_decode_tokens,
        "first_eos_position": profile.first_eos_position,
        "prompt_tokens": profile.prompt_tokens,
        "requested_decode_tokens": profile.requested_decode_tokens,
        "max_context": args.max_context,
        "prefill_path": profile.prefill_path,
        "attention_kernel": profile.attention_kernel,
        "kv_cache_type": profile.kv_cache_type.as_str(),
        "kv_cache_bytes": executor.kv_cache_bytes(),
        "samples": samples,
    });
    append_jsonl_at(GEMMA_DECODE_PROFILE_LOG, &output)?;
    println!("{output}");
    eprintln!("Gemma decode profile: {GEMMA_DECODE_PROFILE_LOG}");
    Ok(())
}

fn profile_bottlenecks(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    let mut profile_args = Vec::new();
    let mut output = PathBuf::from("artifacts/profiles/atlas-profile.json");
    let mut mode = ProfileMode::Diagnostic;
    let mut gpu_counters = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output = PathBuf::from(args.get(i).context("--output needs a value")?);
            }
            "--mode" => {
                i += 1;
                mode = match args.get(i).context("--mode needs a value")?.as_str() {
                    "benchmark" => ProfileMode::Benchmark,
                    "diagnostic" => ProfileMode::Diagnostic,
                    "disabled" => ProfileMode::Disabled,
                    value => bail!("unknown profiler mode `{value}`"),
                };
            }
            "--gpu-counters" => {
                i += 1;
                gpu_counters = Some(GpuCountersMode::parse(
                    args.get(i).context("--gpu-counters needs a value")?,
                )?);
            }
            flag => {
                profile_args.push(flag.to_owned());
                if flag == "--model"
                    || flag == "--prompt"
                    || flag == "--decode-tokens"
                    || flag == "--warmup-decode-tokens"
                    || flag == "--max-context"
                    || flag == "--kv-cache-type"
                    || flag == "--q4-attention-mode"
                {
                    i += 1;
                    profile_args.push(
                        args.get(i)
                            .context("profiler option needs a value")?
                            .clone(),
                    );
                }
            }
        }
        i += 1;
    }
    if gpu_counters.is_some() && mode != ProfileMode::Diagnostic {
        bail!("--gpu-counters is diagnostic-only; use --mode diagnostic");
    }
    let args = parse_profile_args(&profile_args)?;
    let record = resolve_model(&args.model_id)?;
    let model = load_verified_model(&record)?;
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        args.max_context,
        args.kv_cache_type,
        args.q4_attention_mode,
    )?;
    // Diagnostic mode is the only mode that may split command buffers to
    // obtain exact per-dispatch GPU timestamps. Benchmark mode must retain
    // the production command-buffer boundary so its wall-clock result is
    // comparable with the normal fixed-workload benchmark.
    let prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
        Gemma4ChatRole::User,
        args.prompt.clone(),
    )])?;
    let production = executor.generate_greedy_fixed_benchmark_window_stream(
        &prompt,
        args.warmup_decode_tokens,
        args.decode_tokens,
        &CHAT_INTERRUPTED,
        |_| Ok(()),
    )?;
    executor.reset();
    let profile = executor.profile_decode_with_timing(
        &prompt,
        args.warmup_decode_tokens,
        args.decode_tokens,
        mode == ProfileMode::Diagnostic,
    )?;
    let mut profiler = Profiler::new(mode);
    profiler.set_workload(ProfileWorkload {
        prompt_tokens: profile.prompt_tokens as u64,
        generated_tokens: production.generation.generated_token_ids.len() as u64,
        prefill_ns: atlas_profiler::duration_ns(production.metrics.prefill),
        decode_ns: atlas_profiler::duration_ns(production.metrics.measured_scope.wall_time),
        ttft_ns: atlas_profiler::duration_ns(production.metrics.prefill),
        warmup_decode_tokens: args.warmup_decode_tokens as u64,
        measured_decode_tokens: args.decode_tokens as u64,
        completed_decode_tokens: production.generation.generated_token_ids.len() as u64,
        ..Default::default()
    });
    let record_collection_started = Instant::now();
    for kernel in &profile.prefill_kernels {
        profiler.record(ProfileEvent {
            phase: ProfilePhase::Prefill,
            scope: ProfileScope::Prefill,
            operation_family: profiler_operation_family(kernel.family),
            kernel_name: Some(kernel.kernel_name.to_owned()),
            layer_index: kernel.layer_index,
            attention_kind: attention_dimensions(kernel.family).0,
            attention_scan_pass: attention_dimensions(kernel.family).1,
            command_buffer_id: kernel.command_buffer_id,
            dispatch_calls: kernel.dispatches,
            host_encode_ns: kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
            gpu_ns: Some(kernel.gpu_nanos.min(u64::MAX as u128) as u64),
            timing: profile_timing_boundaries(
                ProfileScope::Prefill,
                kernel.gpu_nanos.min(u64::MAX as u128) as u64,
                kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
            )
            .into_iter()
            .next()
            .unwrap_or_default(),
            timing_boundaries: profile_timing_boundaries(
                ProfileScope::Prefill,
                kernel.gpu_nanos.min(u64::MAX as u128) as u64,
                kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
            ),
            threadgroups: kernel.threadgroups,
            threads: kernel.threads,
            bytes_read_estimate: kernel.bytes_read_estimate,
            bytes_written_estimate: kernel.bytes_written_estimate,
            ..Default::default()
        });
    }
    for sample in &profile.samples {
        for kernel in &sample.kernels {
            let operation_family = profiler_operation_family(kernel.family);
            profiler.record(ProfileEvent {
                phase: ProfilePhase::Decode,
                scope: match sample.scope {
                    "decode_warmup" => ProfileScope::DecodeWarmup,
                    "decode_measured" => ProfileScope::DecodeMeasured,
                    _ => ProfileScope::Other,
                },
                token_position: Some(sample.decode_position as u32),
                operation_family,
                kernel_name: Some(kernel.kernel_name.to_owned()),
                layer_index: kernel.layer_index,
                attention_kind: attention_dimensions(kernel.family).0,
                attention_scan_pass: attention_dimensions(kernel.family).1,
                command_buffer_id: kernel.command_buffer_id,
                dispatch_calls: kernel.dispatches,
                host_encode_ns: kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
                gpu_ns: Some(kernel.gpu_nanos.min(u64::MAX as u128) as u64),
                timing: profile_timing_boundaries(
                    match sample.scope {
                        "decode_warmup" => ProfileScope::DecodeWarmup,
                        "decode_measured" => ProfileScope::DecodeMeasured,
                        _ => ProfileScope::Other,
                    },
                    kernel.gpu_nanos.min(u64::MAX as u128) as u64,
                    kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
                )
                .into_iter()
                .next()
                .unwrap_or_default(),
                timing_boundaries: profile_timing_boundaries(
                    match sample.scope {
                        "decode_warmup" => ProfileScope::DecodeWarmup,
                        "decode_measured" => ProfileScope::DecodeMeasured,
                        _ => ProfileScope::Other,
                    },
                    kernel.gpu_nanos.min(u64::MAX as u128) as u64,
                    kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
                ),
                threadgroups: kernel.threadgroups,
                threads: kernel.threads,
                bytes_read_estimate: kernel.bytes_read_estimate,
                bytes_written_estimate: kernel.bytes_written_estimate,
                ..Default::default()
            });
        }
    }
    let record_collection_cpu_ms = record_collection_started.elapsed().as_secs_f64() * 1000.0;
    let event_counters = profiler.counters().clone();
    let scope_events = |scope: ProfileScope| {
        profile
            .prefill_kernels
            .iter()
            .filter(|_| scope == ProfileScope::Prefill)
            .map(|kernel| {
                (
                    profiler_operation_family(kernel.family),
                    kernel.dispatches,
                    kernel.gpu_nanos.min(u64::MAX as u128) as u64,
                    kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
                )
            })
            .chain(
                profile
                    .samples
                    .iter()
                    .filter(move |sample| {
                        matches!(
                            (scope, sample.scope),
                            (ProfileScope::DecodeWarmup, "decode_warmup")
                                | (ProfileScope::DecodeMeasured, "decode_measured")
                                | (ProfileScope::DecodeComplete, "decode_warmup")
                                | (ProfileScope::DecodeComplete, "decode_measured")
                        )
                    })
                    .flat_map(|sample| sample.kernels.iter())
                    .map(|kernel| {
                        (
                            profiler_operation_family(kernel.family),
                            kernel.dispatches,
                            kernel.gpu_nanos.min(u64::MAX as u128) as u64,
                            kernel.cpu_encode_nanos.min(u64::MAX as u128) as u64,
                        )
                    }),
            )
            .collect::<Vec<_>>()
    };
    let scope_event_stats = |scope: ProfileScope| {
        let events = scope_events(scope);
        (
            events.iter().map(|x| x.1).sum::<u64>(),
            events
                .iter()
                .filter(|x| x.0 != OperationFamily::Other)
                .map(|x| x.1)
                .sum::<u64>(),
            events
                .iter()
                .filter(|x| x.0 != OperationFamily::Other)
                .map(|x| x.2)
                .sum::<u64>(),
            events.iter().map(|x| x.2).sum::<u64>(),
            events.iter().map(|x| x.3).sum::<u64>(),
        )
    };
    let production_scopes = [
        (ProfileScope::Prefill, production.metrics.prefill_scope),
        (ProfileScope::DecodeWarmup, production.metrics.warmup_scope),
        (
            ProfileScope::DecodeMeasured,
            production.metrics.measured_scope,
        ),
        (
            ProfileScope::DecodeComplete,
            production.metrics.complete_decode_scope,
        ),
    ];
    let mut scope_counters = BTreeMap::new();
    for (scope, measured) in production_scopes {
        let (_, categorized, categorized_gpu, attributed_gpu, host_encode) =
            scope_event_stats(scope);
        scope_counters.insert(
            scope,
            ProfileCounters {
                host_wall_ns: atlas_profiler::duration_ns(measured.wall_time),
                gpu_ns: measured.telemetry.gpu_execution_nanos,
                production_gpu_elapsed_ns: measured.telemetry.gpu_execution_nanos,
                attributed_gpu_duration_ns: attributed_gpu,
                gpu_duration_source: "production_boundary_plus_exact_dispatch_attribution".into(),
                categorized_gpu_ns: categorized_gpu,
                cpu_wait_ns: measured.telemetry.cpu_wait_nanos,
                host_encode_ns: host_encode,
                command_buffers: measured.telemetry.command_buffers,
                dispatches: measured.telemetry.dispatches,
                threadgroups_dispatched: measured.telemetry.threadgroups_dispatched,
                threads_dispatched: measured.telemetry.threads_dispatched,
                timed_dispatches: measured.telemetry.timed_dispatches,
                categorized_dispatches: categorized,
                upload_bytes: measured.telemetry.upload_bytes,
                readback_bytes: measured.telemetry.readback_bytes,
                allocations: measured.telemetry.buffer_allocations,
                resident_bytes: measured.telemetry.resident_bytes,
                peak_resident_bytes: measured.telemetry.peak_resident_bytes,
                kv_cache_bytes: executor.kv_cache_bytes(),
                upload_time_ns: measured.telemetry.upload_time_nanos,
                readback_time_ns: measured.telemetry.readback_time_nanos,
                memory_operation_time_ns: measured
                    .telemetry
                    .upload_time_nanos
                    .saturating_add(measured.telemetry.readback_time_nanos),
                ..Default::default()
            },
        );
    }
    profiler.set_scope_counters(scope_counters);
    if let Some(requested) = gpu_counters {
        let metadata = executor.diagnostic_counter_metadata();
        if requested == GpuCountersMode::Required && !metadata.dispatch_boundary_sampling_supported
        {
            bail!(
                "GPU counter sampling at dispatch boundaries is required but unsupported by {}",
                metadata.device_name
            );
        }
        profiler.set_gpu_counter_capture(Some(json!({
            "requested": match requested { GpuCountersMode::Auto => "auto", GpuCountersMode::Required => "required" },
            "capture_scope": "diagnostic_only",
            "device_name": metadata.device_name,
            "dispatch_boundary_sampling_supported": metadata.dispatch_boundary_sampling_supported,
            "available_counter_names": metadata.available_counter_names,
            "pipelines": metadata.pipelines,
            "samples": [],
            "status": if metadata.dispatch_boundary_sampling_supported { "capability_checked" } else { "unsupported" },
            "warning": "counter capability and pipeline metadata are separate from production timing; production Resident command buffers are unchanged"
        })));
    }
    profiler.set_counters(event_counters.clone());
    profiler.set_scope_contract(ScopeContract {
        hotspot_scope: ProfileScope::DecodeMeasured,
        includes_warmup: false,
        token_selection_included: true,
        readback_included: true,
    });
    profiler.set_decode_scope(DecodeScope {
        warmup_decode_tokens_requested: args.warmup_decode_tokens as u64,
        warmup_decode_tokens_completed: production.metrics.warmup_scope.completed_tokens as u64,
        measured_decode_tokens_requested: args.decode_tokens as u64,
        measured_decode_tokens_completed: production.metrics.measured_scope.completed_tokens as u64,
        completed_decode_tokens_total: production.metrics.completed_decode_tokens as u64,
        hotspot_scope: ProfileScope::DecodeMeasured,
        physical_command_buffer_overlap: production.metrics.physical_command_buffer_overlap,
        physical_command_buffer_overlap_reason: production
            .metrics
            .physical_command_buffer_overlap_reason
            .clone(),
    });
    let mut windows = BTreeMap::new();
    for (scope, measured, tokens) in [
        (
            ProfileScope::Prefill,
            production.metrics.prefill_scope,
            profile.prompt_tokens,
        ),
        (
            ProfileScope::DecodeWarmup,
            production.metrics.warmup_scope,
            args.warmup_decode_tokens,
        ),
        (
            ProfileScope::DecodeMeasured,
            production.metrics.measured_scope,
            args.decode_tokens,
        ),
        (
            ProfileScope::DecodeComplete,
            production.metrics.complete_decode_scope,
            production.metrics.completed_decode_tokens,
        ),
    ] {
        windows.insert(
            scope,
            MeasuredWindow {
                scope,
                host_start_ns: Some(measured.host_start_ns),
                host_end_ns: Some(measured.host_end_ns),
                wall_time_ms: Some(measured.wall_time.as_secs_f64() * 1000.0),
                tokens: tokens as u64,
                token_selection_included: true,
                readback_included: true,
                timing: TimingBoundary {
                    clock_domain: ClockDomain::HostMonotonic,
                    timing_kind: TimingKind::HostWall,
                    start_ns: Some(measured.host_start_ns),
                    end_ns: Some(measured.host_end_ns),
                    intervals_may_overlap: false,
                    status: "relative_boundary_recorded_by_resident_executor".into(),
                    ..Default::default()
                },
            },
        );
    }
    profiler.set_measured_windows(windows);
    let mut compatibility_warnings = Vec::new();
    if production.metrics.physical_command_buffer_overlap {
        compatibility_warnings.push(
            production
                .metrics
                .physical_command_buffer_overlap_reason
                .clone()
                .unwrap_or_else(|| "physical decode command-buffer overlap observed".into()),
        );
    }
    profiler.set_benchmark_compatibility(BenchmarkCompatibility {
        scope_matches_normal_benchmark: true,
        token_window_matches: production.generation.generated_token_ids.len()
            == profile.completed_decode_tokens,
        executor_matches: true,
        kernel_plan_matches: true,
        token_sha_matches: production.generation.generated_token_ids == profile.generated_token_ids,
        eos_matches: production.first_eos_position == profile.first_eos_position,
        prompt_token_sha256: Some(token_ids_sha256(&production.generation.prompt_token_ids)),
        generated_token_sha256: Some(token_ids_sha256(&production.generation.generated_token_ids)),
        measured_generated_token_sha256: Some(token_ids_sha256(
            &production.generation.generated_token_ids[args.warmup_decode_tokens..],
        )),
        first_eos_position: production
            .first_eos_position
            .map(|position| position as u64),
        executor: Some("resident".into()),
        kv_cache_type: Some(production.metrics.kv_cache_type.as_str().into()),
        quantization_plan: production.metrics.quantization_plan_path.clone(),
        selected_kernels: selected_kernels_map(&production.metrics),
        warnings: compatibility_warnings,
    });
    let phase_summary = |phase: ProfilePhase,
                         scope: ProfileScope,
                         measured: atlas_model::gemma4_executor::Gemma4ScopeMetrics,
                         tokens: u64,
                         categorized: u64,
                         categorized_gpu: u64,
                         attributed_gpu: u64,
                         host_encode: u64| PhaseSummary {
        phase,
        scope,
        wall_ns: atlas_profiler::duration_ns(measured.wall_time),
        gpu_ns: measured.telemetry.gpu_execution_nanos,
        attributed_gpu_ns: attributed_gpu,
        cpu_wait_ns: measured.telemetry.cpu_wait_nanos,
        command_buffers: measured.telemetry.command_buffers,
        dispatch_calls: measured.telemetry.dispatches,
        threadgroups_dispatched: measured.telemetry.threadgroups_dispatched,
        threads_dispatched: measured.telemetry.threads_dispatched,
        timed_dispatches: measured.telemetry.timed_dispatches,
        untimed_dispatches: measured
            .telemetry
            .dispatches
            .saturating_sub(measured.telemetry.timed_dispatches),
        categorized_dispatches: categorized,
        uncategorized_dispatches: measured.telemetry.dispatches.saturating_sub(categorized),
        categorized_gpu_ns: categorized_gpu,
        uncategorized_gpu_ns: measured
            .telemetry
            .gpu_execution_nanos
            .saturating_sub(categorized_gpu),
        host_encode_ns: host_encode,
        upload_time_ns: measured.telemetry.upload_time_nanos,
        readback_time_ns: measured.telemetry.readback_time_nanos,
        resident_bytes: measured.telemetry.resident_bytes,
        peak_resident_bytes: measured.telemetry.peak_resident_bytes,
        kv_cache_bytes: executor.kv_cache_bytes(),
        command_buffer_idle_gap_ns: Some(measured.telemetry.command_buffer_idle_gap_nanos),
        command_buffer_schedule_ns: Some(measured.telemetry.command_buffer_schedule_nanos),
        unexplained_ns: atlas_profiler::duration_ns(measured.wall_time)
            .saturating_sub(measured.telemetry.gpu_execution_nanos)
            .saturating_sub(measured.telemetry.cpu_wait_nanos)
            .saturating_sub(host_encode)
            .saturating_sub(measured.telemetry.upload_time_nanos)
            .saturating_sub(measured.telemetry.readback_time_nanos),
        upload_bytes: measured.telemetry.upload_bytes,
        readback_bytes: measured.telemetry.readback_bytes,
        tokens,
        tokens_per_second: if measured.wall_time.is_zero() {
            0.0
        } else {
            tokens as f64 / measured.wall_time.as_secs_f64()
        },
        ..Default::default()
    };
    profiler.set_phase_summaries(vec![
        phase_summary(
            ProfilePhase::Prefill,
            ProfileScope::Prefill,
            production.metrics.prefill_scope,
            profile.prompt_tokens as u64,
            scope_event_stats(ProfileScope::Prefill).1,
            scope_event_stats(ProfileScope::Prefill).2,
            scope_event_stats(ProfileScope::Prefill).3,
            scope_event_stats(ProfileScope::Prefill).4,
        ),
        phase_summary(
            ProfilePhase::Decode,
            ProfileScope::DecodeWarmup,
            production.metrics.warmup_scope,
            args.warmup_decode_tokens as u64,
            scope_event_stats(ProfileScope::DecodeWarmup).1,
            scope_event_stats(ProfileScope::DecodeWarmup).2,
            scope_event_stats(ProfileScope::DecodeWarmup).3,
            scope_event_stats(ProfileScope::DecodeWarmup).4,
        ),
        phase_summary(
            ProfilePhase::Decode,
            ProfileScope::DecodeMeasured,
            production.metrics.measured_scope,
            args.decode_tokens as u64,
            scope_event_stats(ProfileScope::DecodeMeasured).1,
            scope_event_stats(ProfileScope::DecodeMeasured).2,
            scope_event_stats(ProfileScope::DecodeMeasured).3,
            scope_event_stats(ProfileScope::DecodeMeasured).4,
        ),
        phase_summary(
            ProfilePhase::Decode,
            ProfileScope::DecodeComplete,
            production.metrics.complete_decode_scope,
            production.metrics.completed_decode_tokens as u64,
            scope_event_stats(ProfileScope::DecodeWarmup).1
                + scope_event_stats(ProfileScope::DecodeMeasured).1,
            scope_event_stats(ProfileScope::DecodeWarmup).2
                + scope_event_stats(ProfileScope::DecodeMeasured).2,
            scope_event_stats(ProfileScope::DecodeWarmup).3
                + scope_event_stats(ProfileScope::DecodeMeasured).3,
            scope_event_stats(ProfileScope::DecodeWarmup).4
                + scope_event_stats(ProfileScope::DecodeMeasured).4,
        ),
        PhaseSummary {
            phase: ProfilePhase::HostSynchronization,
            scope: ProfileScope::ProfilerOverhead,
            wall_ns: 0,
            ..Default::default()
        },
    ]);
    profiler.set_collection_complete(true);
    let aggregation_started = Instant::now();
    let mut report = profiler.report();
    let aggregation_cpu_ms = aggregation_started.elapsed().as_secs_f64() * 1000.0;
    let json_path = output;
    if let Some(parent) = json_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json_started = Instant::now();
    let _ = report.to_json()?;
    let json_serialization_ms = json_started.elapsed().as_secs_f64() * 1000.0;
    let markdown_path = json_path.with_extension("md");
    let markdown_started = Instant::now();
    let markdown = report.to_markdown();
    let markdown_generation_ms = markdown_started.elapsed().as_secs_f64() * 1000.0;
    report.profiler_overhead = atlas_profiler::ProfilerOverhead {
        record_collection_cpu_ms,
        aggregation_cpu_ms,
        json_serialization_ms,
        markdown_generation_ms,
        callback_overhead_estimate_ms: 0.0,
        callback_overhead_status: "not_applicable_no_callback".into(),
        total_profiler_overhead_ms: record_collection_cpu_ms
            + aggregation_cpu_ms
            + json_serialization_ms
            + markdown_generation_ms,
    };
    fs::write(&json_path, report.to_json()?)?;
    fs::write(&markdown_path, markdown)?;
    println!(
        "{}",
        json!({"json": json_path, "markdown": markdown_path, "mode": mode, "hotspot_scope": "decode_measured", "warmup_decode_tokens": report.decode_scope.warmup_decode_tokens_requested, "measured_decode_tokens": report.decode_scope.measured_decode_tokens_requested, "completed_decode_tokens": report.decode_scope.completed_decode_tokens_total, "measured_decode_wall_ms": report.measured_windows.get(&ProfileScope::DecodeMeasured).and_then(|x| x.wall_time_ms), "measured_decode_tok_s": report.measured_windows.get(&ProfileScope::DecodeMeasured).and_then(|x| x.wall_time_ms).map(|ms| report.decode_scope.measured_decode_tokens_completed as f64 / (ms / 1000.0)), "measured_decode_dispatch_calls": report.scope_counters.get(&ProfileScope::DecodeMeasured).map_or(0, |x| x.dispatches), "warmup_decode_dispatch_calls": report.scope_counters.get(&ProfileScope::DecodeWarmup).map_or(0, |x| x.dispatches), "total_decode_dispatch_calls": report.scope_counters.get(&ProfileScope::DecodeComplete).map_or(0, |x| x.dispatches), "json": json_path, "markdown": markdown_path, "dispatches": report.counters.dispatches, "recommendations": report.recommendations})
    );
    Ok(())
}

fn profile_timing_boundaries(
    _scope: ProfileScope,
    gpu_ns: u64,
    host_encode_ns: u64,
) -> Vec<TimingBoundary> {
    vec![
        TimingBoundary {
            clock_domain: ClockDomain::MetalGpu,
            timing_kind: TimingKind::GpuElapsed,
            start_ns: Some(0),
            end_ns: Some(gpu_ns),
            intervals_may_overlap: false,
            status: "exact_per_dispatch_diagnostic_pass".into(),
        },
        TimingBoundary {
            clock_domain: ClockDomain::HostMonotonic,
            timing_kind: TimingKind::CpuEncode,
            start_ns: Some(0),
            end_ns: Some(host_encode_ns),
            intervals_may_overlap: false,
            status: "dispatch_encode_interval_relative_to_event".into(),
        },
    ]
}

fn profiler_operation_family(family: &str) -> OperationFamily {
    match family {
        // Exact Resident semantic labels. Keep normalization and attention
        // value computation distinct, and never fold PLE into the LM head.
        "attention_input_norm"
        | "post_attention_norm"
        | "post_attention_norm_residual"
        | "ffn_input_norm"
        | "post_ffn_norm"
        | "ple_norm"
        | "provider_kv_value_norm"
        | "layer_major_rms_norm" => OperationFamily::RmsNorm,
        "final_output_norm" => OperationFamily::FinalNorm,
        "attention_output_projection" => OperationFamily::AttentionOutputProjection,
        "ffn_down_projection" | "layer_major_batched_ffn_down_projection" => {
            OperationFamily::FfnDown
        }
        "ple_projection" => OperationFamily::PleProjection,
        "qkv_projection" => OperationFamily::QkvProjection,
        "ffn_gate_up_projection" => OperationFamily::FfnGateUp,
        "output_projection" => OperationFamily::OutputProjection,
        "gemma_attention_global_split_scan" | "gemma_attention_sliding_split_scan" => {
            OperationFamily::AttentionScore
        }
        "gemma_attention_global_split_combine" | "gemma_attention_sliding_split_combine" => {
            OperationFamily::AttentionValue
        }
        "gemma_attention_flash16" | "gemma_attention_flash16_swa" => {
            OperationFamily::AttentionScore
        }

        // Exact fallback families emitted when a dispatch has no explicit
        // profiling label. Ambiguous projection families remain conservative.
        "q4_qkv_projection" => OperationFamily::QkvProjection,
        "q4_ffn_gate_up_projection" => OperationFamily::FfnGateUp,
        "q6_lm_head_projection" => OperationFamily::OutputProjection,
        "embedding_lookup" => OperationFamily::Embedding,
        "gemma_attention" => OperationFamily::AttentionValue,
        "qk_norm_rope_fused" | "rms_norm" => OperationFamily::RmsNorm,
        "kv_append" => OperationFamily::KvAppend,
        "ffn_activation" => OperationFamily::FfnActivationMultiply,
        "residual" => OperationFamily::Residual,
        "softcap" => OperationFamily::LogitSoftcap,
        "argmax" => OperationFamily::ArgmaxOrTokenSelection,
        "conversion" => OperationFamily::Conversion,
        "q4_projection_other" | "q6_projection_other" => OperationFamily::Other,
        "batched_projection" | "rope_rotation" | "rope_layout" | "other" => OperationFamily::Other,
        _ => OperationFamily::Other,
    }
}

fn attention_dimensions(family: &str) -> (Option<AttentionKind>, Option<AttentionScanPass>) {
    match family {
        "gemma_attention_global_split_scan" => {
            (Some(AttentionKind::Global), Some(AttentionScanPass::Scan))
        }
        "gemma_attention_global_split_combine" => (
            Some(AttentionKind::Global),
            Some(AttentionScanPass::Combine),
        ),
        "gemma_attention_sliding_split_scan" => {
            (Some(AttentionKind::Sliding), Some(AttentionScanPass::Scan))
        }
        "gemma_attention_sliding_split_combine" => (
            Some(AttentionKind::Sliding),
            Some(AttentionScanPass::Combine),
        ),
        _ => (None, None),
    }
}

/// Run a fixed-length Resident decode workload for Phase 12a performance
/// comparison. This command is intentionally separate from `chat` and
/// `generate`: it keeps selecting tokens after EOS so cache modes with
/// different EOS behavior still receive identical decode work.
fn benchmark(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    let args = parse_benchmark_args(args)?;
    let record = resolve_model(&args.model_id)?;
    let model = load_verified_model(&record)?;
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        args.max_context,
        args.kv_cache_type,
        args.q4_attention_mode,
    )?;
    let prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(Gemma4ChatRole::User, args.prompt)])?;
    let generation = executor.generate_greedy_fixed_benchmark_window_stream(
        &prompt,
        args.warmup_decode_tokens,
        args.measured_decode_tokens,
        &CHAT_INTERRUPTED,
        |_| Ok(()),
    )?;
    let prompt_token_digest = token_ids_sha256(&generation.generation.prompt_token_ids);
    let token_digest = token_ids_sha256(&generation.generation.generated_token_ids);
    let measured_token_digest =
        token_ids_sha256(&generation.generation.generated_token_ids[args.warmup_decode_tokens..]);
    let selected_kernels = json!({
        "attention": generation.metrics.attention_kernel,
        "q4_projection": generation.metrics.q4_projection_kernel,
        "q4_qkv_projection": generation.metrics.q4_qkv_projection_kernel,
        "q4_gate_up_projection": generation.metrics.q4_gate_up_projection_kernel,
        "ffn_gate_up_activation": generation.metrics.ffn_gate_up_activation_kernel,
        "ple_composition": generation.metrics.ple_composition_kernel,
        "q4_batch_projection": generation.metrics.q4_batch_projection_kernel,
        "quantization_plan": generation.metrics.quantization_plan_path,
        "ffn_down_projection": generation.metrics.ffn_down_projection_kernel,
        "ple_projection": generation.metrics.ple_projection_kernel,
        "q6_projection": generation.metrics.q6_projection_kernel,
        "rms_norm": generation.metrics.rms_norm_kernel,
        "rms_fused_projection": generation.metrics.rms_fused_projection_kernel,
        "rms_epilogue": generation.metrics.rms_epilogue_kernel,
        "embedding": generation.metrics.embedding_kernel,
        "output_projection": generation.metrics.output_projection_kernel,
        "kv_append": generation.metrics.kv_append_kernel,
    });
    let record = json!({
        "event": "gemma4_fixed_workload_benchmark",
        "diagnostic": false,
        "model_id": record.id,
        "executor": "resident",
        "format": "gguf-gemma4-q4_0",
        "weight_format": generation.metrics.weight_format.as_str(),
        "embedding_kernel": generation.metrics.embedding_kernel,
        "output_projection_kernel": generation.metrics.output_projection_kernel,
        "q4_projection_kernel": generation.metrics.q4_projection_kernel,
        "ffn_down_projection_kernel": generation.metrics.ffn_down_projection_kernel,
        "rms_norm_kernel": generation.metrics.rms_norm_kernel,
        "rms_fused_projection_kernel": generation.metrics.rms_fused_projection_kernel,
        "prompt_template": "gemma4_chat",
        "prompt_token_sha256": prompt_token_digest,
        "kv_cache_type": generation.metrics.kv_cache_type.as_str(),
        "max_context": args.max_context,
        "prompt_tokens": generation.generation.prompt_token_ids.len(),
        "warmup_decode_tokens": args.warmup_decode_tokens,
        "fixed_decode_tokens": args.measured_decode_tokens,
        "measured_decode_tokens": args.measured_decode_tokens,
        "completed_decode_tokens": generation.generation.generated_token_ids.len(),
        "decode_scope": {
            "warmup_decode_tokens_requested": args.warmup_decode_tokens,
            "warmup_decode_tokens_completed": generation.metrics.warmup_scope.completed_tokens,
            "measured_decode_tokens_requested": args.measured_decode_tokens,
            "measured_decode_tokens_completed": generation.metrics.measured_scope.completed_tokens,
            "completed_decode_tokens_total": generation.metrics.completed_decode_tokens,
            "hotspot_scope": "decode_measured",
            "physical_command_buffer_overlap": generation.metrics.physical_command_buffer_overlap,
            "physical_command_buffer_overlap_reason": generation.metrics.physical_command_buffer_overlap_reason,
        },
        "first_eos_position": generation.first_eos_position,
        "generated_token_sha256": token_digest,
        "measured_generated_token_sha256": measured_token_digest,
        "prefill_tok_s": rate(generation.generation.prompt_token_ids.len(), generation.metrics.prefill),
        "decode_tok_s": rate(generation.metrics.decode_command_buffers as usize, generation.metrics.decode),
        "resident_bytes": generation.metrics.resident_bytes,
        "kv_cache_bytes": generation.metrics.kv_cache_bytes,
        "weight_upload_bytes": generation.metrics.weight_upload_bytes,
        "readback_bytes": generation.metrics.readback_bytes,
        "command_buffers": generation.metrics.command_buffers,
        "prefill_command_buffers": generation.metrics.prefill_command_buffers,
        "decode_command_buffers": generation.metrics.decode_command_buffers,
        "warmup_decode_command_buffers": generation.metrics.warmup_scope.telemetry.command_buffers,
        "measured_decode_command_buffers": generation.metrics.measured_scope.telemetry.command_buffers,
        "total_decode_command_buffers": generation.metrics.complete_decode_scope.telemetry.command_buffers,
        "warmup_decode_dispatch_calls": generation.metrics.warmup_scope.telemetry.dispatches,
        "measured_decode_dispatch_calls": generation.metrics.measured_scope.telemetry.dispatches,
        "total_decode_dispatch_calls": generation.metrics.complete_decode_scope.telemetry.dispatches,
        "warmup_decode_threadgroups": generation.metrics.warmup_scope.telemetry.threadgroups_dispatched,
        "measured_decode_threadgroups": generation.metrics.measured_scope.telemetry.threadgroups_dispatched,
        "total_decode_threadgroups": generation.metrics.complete_decode_scope.telemetry.threadgroups_dispatched,
        "warmup_decode_gpu_ms": generation.metrics.warmup_scope.telemetry.gpu_execution_nanos as f64 / 1_000_000.0,
        "measured_decode_gpu_ms": generation.metrics.measured_scope.telemetry.gpu_execution_nanos as f64 / 1_000_000.0,
        "total_decode_gpu_ms": generation.metrics.complete_decode_scope.telemetry.gpu_execution_nanos as f64 / 1_000_000.0,
        "prefill_path": generation.metrics.prefill_path,
        "prefill_chunk_size": generation.metrics.prefill_chunk_size,
        "prefill_chunks": generation.metrics.prefill_chunks,
        "quantization_preflight_state": generation.metrics.quantization_preflight_state,
        "quantization_plan": generation.metrics.quantization_plan_path,
        "selected_group_formats": selected_group_formats_json(&generation.metrics.selected_group_formats),
        "quantization_rejections": generation.metrics.quantization_rejections,
        "q4_attention_mode": generation.metrics.q4_attention_mode.as_str(),
        "selected_kernels": selected_kernels,
        "timing": {
            "prefill_ms": generation.metrics.prefill.as_secs_f64() * 1000.0,
            "decode_ms": generation.metrics.decode.as_secs_f64() * 1000.0,
            "host_ms": generation.metrics.host_wall_time.as_secs_f64() * 1000.0,
        },
    });
    append_jsonl_at(GEMMA_FIXED_BENCHMARK_LOG, &record)?;
    println!("{record}");
    eprintln!("Gemma fixed-workload benchmark: {GEMMA_FIXED_BENCHMARK_LOG}");
    Ok(())
}

fn token_ids_sha256(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn selected_kernels_map(
    metrics: &atlas_model::gemma4_executor::Gemma4Metrics,
) -> BTreeMap<String, String> {
    let mut kernels = BTreeMap::from([
        ("attention".into(), metrics.attention_kernel.into()),
        ("q4_projection".into(), metrics.q4_projection_kernel.into()),
        (
            "q4_qkv_projection".into(),
            metrics.q4_qkv_projection_kernel.into(),
        ),
        (
            "q4_gate_up_projection".into(),
            metrics.q4_gate_up_projection_kernel.into(),
        ),
        (
            "ffn_gate_up_activation".into(),
            metrics.ffn_gate_up_activation_kernel.into(),
        ),
        (
            "ple_composition".into(),
            metrics.ple_composition_kernel.into(),
        ),
        ("rms_epilogue".into(), metrics.rms_epilogue_kernel.into()),
        (
            "q4_batch_projection".into(),
            metrics.q4_batch_projection_kernel.into(),
        ),
        (
            "ffn_down_projection".into(),
            metrics.ffn_down_projection_kernel.into(),
        ),
        (
            "ple_projection".into(),
            metrics.ple_projection_kernel.into(),
        ),
        ("q6_projection".into(), metrics.q6_projection_kernel.into()),
        ("rms_norm".into(), metrics.rms_norm_kernel.into()),
        (
            "rms_fused_projection".into(),
            metrics.rms_fused_projection_kernel.into(),
        ),
        ("embedding".into(), metrics.embedding_kernel.into()),
        (
            "output_projection".into(),
            metrics.output_projection_kernel.into(),
        ),
        ("kv_append".into(), metrics.kv_append_kernel.into()),
    ]);
    if let Some(plan) = &metrics.quantization_plan_path {
        kernels.insert("quantization_plan".into(), plan.clone());
    }
    kernels
}

fn parse_profile_args(args: &[String]) -> Result<GemmaProfileArgs> {
    let mut model_id = None;
    let mut prompt =
        "Atlas Resident decode profile: summarize GPU-resident inference in one sentence."
            .to_owned();
    let mut decode_tokens = 128;
    let mut warmup_decode_tokens = 32;
    let mut max_context = 4096;
    let mut kv_cache_type = Gemma4KvCacheType::Q4_0;
    let mut q4_attention_mode = Gemma4Q4AttentionMode::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_id = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).context("--prompt needs a value")?.clone();
            }
            "--decode-tokens" => {
                i += 1;
                decode_tokens = args
                    .get(i)
                    .context("--decode-tokens needs a value")?
                    .parse()?;
            }
            "--warmup-decode-tokens" => {
                i += 1;
                warmup_decode_tokens = args
                    .get(i)
                    .context("--warmup-decode-tokens needs a value")?
                    .parse()?;
            }
            "--max-context" => {
                i += 1;
                max_context = args
                    .get(i)
                    .context("--max-context needs a value")?
                    .parse()?;
            }
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type = Gemma4KvCacheType::parse(
                    args.get(i).context("--kv-cache-type needs a value")?,
                )?;
            }
            "--q4-attention-mode" => {
                i += 1;
                q4_attention_mode = Gemma4Q4AttentionMode::parse(
                    args.get(i).context("--q4-attention-mode needs a value")?,
                )?;
            }
            flag => bail!("unknown profile option: {flag}"),
        }
        i += 1;
    }
    ensure!(decode_tokens >= 128, "--decode-tokens must be at least 128");
    ensure!(max_context > 0, "--max-context must be positive");
    Ok(GemmaProfileArgs {
        model_id: model_id.context("--model is required")?,
        prompt,
        warmup_decode_tokens,
        decode_tokens,
        max_context,
        kv_cache_type,
        q4_attention_mode,
    })
}

fn parse_benchmark_args(args: &[String]) -> Result<GemmaBenchmarkArgs> {
    let mut model = None;
    let mut prompt = None;
    let mut decode_tokens = None;
    let mut warmup_decode_tokens = 32;
    let mut max_context = 4096;
    let mut kv_cache_type = Gemma4KvCacheType::Q4_0;
    let mut q4_attention_mode = Gemma4Q4AttentionMode::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "--decode-tokens" => {
                i += 1;
                let value: usize = args
                    .get(i)
                    .context("--decode-tokens needs a value")?
                    .parse()?;
                ensure!(value > 0, "--decode-tokens must be positive");
                decode_tokens = Some(value);
            }
            "--warmup-decode-tokens" => {
                i += 1;
                warmup_decode_tokens = args
                    .get(i)
                    .context("--warmup-decode-tokens needs a value")?
                    .parse()?;
            }
            "--max-context" => {
                i += 1;
                max_context = args
                    .get(i)
                    .context("--max-context needs a value")?
                    .parse()?;
            }
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type = Gemma4KvCacheType::parse(
                    args.get(i).context("--kv-cache-type needs a value")?,
                )?;
            }
            "--q4-attention-mode" => {
                i += 1;
                q4_attention_mode = Gemma4Q4AttentionMode::parse(
                    args.get(i).context("--q4-attention-mode needs a value")?,
                )?;
            }
            flag => bail!("unknown benchmark option: {flag}"),
        }
        i += 1;
    }
    ensure!(max_context > 0, "--max-context must be positive");
    Ok(GemmaBenchmarkArgs {
        model_id: model.context("--model is required")?,
        prompt: prompt.context("--prompt is required")?,
        warmup_decode_tokens,
        measured_decode_tokens: decode_tokens.context("--decode-tokens is required")?,
        max_context,
        kv_cache_type,
        q4_attention_mode,
    })
}

fn parse_chat_args(
    args: &[String],
) -> Result<(
    String,
    Option<String>,
    Option<usize>,
    bool,
    Gemma4KvCacheType,
    Gemma4Q4AttentionMode,
)> {
    let mut model = None;
    let mut prompt = None;
    let mut max = None;
    let mut thoughts = false;
    let mut kv_cache_type = Gemma4KvCacheType::Q4_0;
    let mut q4_attention_mode = Gemma4Q4AttentionMode::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "--max-tokens" => {
                i += 1;
                let value: usize = args.get(i).context("--max-tokens needs a value")?.parse()?;
                ensure!(value > 0, "--max-tokens must be positive");
                max = Some(value);
            }
            "--show-thoughts" => thoughts = true,
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type = Gemma4KvCacheType::parse(
                    args.get(i).context("--kv-cache-type needs a value")?,
                )?;
            }
            "--q4-attention-mode" => {
                i += 1;
                q4_attention_mode = Gemma4Q4AttentionMode::parse(
                    args.get(i).context("--q4-attention-mode needs a value")?,
                )?;
            }
            "--executor" => {
                i += 1;
                ensure!(
                    args.get(i).map(String::as_str) == Some("resident"),
                    "--executor must be `resident`; Atlas has no reference fallback"
                );
            }
            flag => bail!("unknown chat option: {flag}"),
        };
        i += 1;
    }
    Ok((
        model.context("--model is required")?,
        prompt,
        max,
        thoughts,
        kv_cache_type,
        q4_attention_mode,
    ))
}

fn model_command(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .context("model command requires a subcommand")?;
    match command.as_str() {
        "search" => model_search(&args[1..]),
        "download" => model_download(&args[1..]),
        "inspect" | "verify" | "quantization-plan" => {
            let id = option_value(&args[1..], "--model")?;
            let model = resolve_model(&id)?;
            let model_path = model.path.join(&model.model_file);
            if command == "quantization-plan" && args.iter().any(|arg| arg == "--profile") {
                let output_path = args
                    .iter()
                    .position(|arg| arg == "--output")
                    .and_then(|index| args.get(index + 1))
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        PathBuf::from(format!("artifacts/quantization-plans/{id}.json"))
                    });
                let model =
                    Gemma4E2bModel::load_gguf_without_quantization_preflight(&model_path)
                        .with_context(|| format!("load Gemma 4 GGUF {}", model_path.display()))?;
                let report = run_gemma4_quantization_preflight(
                    &model,
                    Gemma4QuantizationPreflightInvocation::CliProfile,
                    Some(&output_path),
                )?
                .context("Gemma quantization preflight produced no report")?;
                println!("{}", report.to_value());
                eprintln!(
                    "Gemma quantization preflight artifact: {}",
                    output_path.display()
                );
                return Ok(());
            }
            if command == "quantization-plan" {
                let gguf = GgufModel::open(&model_path)
                    .with_context(|| format!("read GGUF {}", model_path.display()))?;
                let plan_path = args
                    .iter()
                    .position(|arg| arg == "--plan")
                    .and_then(|index| args.get(index + 1))
                    .map(PathBuf::from)
                    .unwrap_or_else(|| default_sidecar_path(&model_path));
                ensure!(
                    plan_path.exists(),
                    "no quantization plan sidecar exists at {}",
                    plan_path.display()
                );
                let plan = atlas_model::quantization_plan::QuantizationPlan::from_path(&plan_path)?;
                let model_sha256 = sha256_file(&model_path)?;
                plan.validate_for_model(&gguf, &model_sha256)?;
                println!(
                    "{}",
                    json!({"model_id": id, "plan": plan_path, "validated": true, "schema_version": plan.schema_version, "tensor_count": plan.tensors.len()})
                );
                return Ok(());
            }
            if command == "verify" {
                verify_manifest_model(&model)?;
                println!(
                    "{}",
                    json!({"model_id":model.id,"verified":true,"format":model.format,"bytes":model.bytes})
                );
            } else {
                println!(
                    "{}",
                    json!({"model_id":model.id,"source":model.source,"revision":model.revision,"path":model.path,"architecture":model.architecture,"format":model.format,"bytes":model.bytes})
                );
            }
            Ok(())
        }
        _ => bail!("model command must be search, download, inspect, verify, or quantization-plan"),
    }
}
fn model_search(args: &[String]) -> Result<()> {
    let query = args
        .iter()
        .find(|arg| !arg.starts_with('-'))
        .map(String::as_str)
        .unwrap_or("");
    let provider = providers::provider(providers::selected(None)?.id())?;
    let page = provider.search(query, None)?;
    for candidate in page.candidates {
        println!("{}", candidate.json());
    }
    if page.next_cursor.is_some() {
        eprintln!("more results are available; refine the query");
    }
    Ok(())
}
fn model_download(args: &[String]) -> Result<()> {
    let candidate = args
        .first()
        .context("model download requires a provider model ID")?;
    let id = option_value(&args[1..], "--id")?;
    ensure!(
        candidate.starts_with("huggingface:"),
        "only Hugging Face downloads are supported"
    );
    let destination = Path::new("models/gguf").join(&id);
    ensure!(
        !destination.exists(),
        "model destination already exists: {}",
        destination.display()
    );
    let downloaded = providers::download_hugging_face(candidate, &destination, true)?;
    let file = downloaded
        .files
        .iter()
        .find(|file| file.ends_with(".gguf"))
        .context("download did not contain a GGUF")?;
    let gguf = GgufModel::open(destination.join(file))?;
    ensure!(
        gguf.metadata
            .get("general.architecture")
            .map(String::as_str)
            == Some("gemma4")
            && gguf
                .tensors
                .iter()
                .any(|tensor| tensor.tensor_type == GgufTensorType::Q4_0),
        "download is not a supported Gemma 4 E2B Q4_0 GGUF"
    );
    bail!(
        "downloaded {}@{} to {}; add its pinned manifest contract before inference",
        downloaded.repository,
        downloaded.revision,
        destination.display()
    )
}
fn option_value(args: &[String], option: &str) -> Result<String> {
    let index = args
        .iter()
        .position(|arg| arg == option)
        .context(format!("{option} is required"))?;
    args.get(index + 1)
        .cloned()
        .context(format!("{option} needs a value"))
}

fn load_manifest() -> Result<ModelManifest> {
    load_manifest_from(Path::new(MODEL_MANIFEST))
}
fn load_manifest_from(path: &Path) -> Result<ModelManifest> {
    let mut model = ModelRecord {
        id: String::new(),
        source: String::new(),
        revision: String::new(),
        path: PathBuf::new(),
        architecture: String::new(),
        tokenizer: PathBuf::new(),
        model_file: PathBuf::new(),
        embedded_tokenizer: false,
        format: String::new(),
        bytes: 0,
        files: Vec::new(),
    };
    let mut file = ModelFile {
        path: PathBuf::new(),
        bytes: 0,
        sha256: String::new(),
    };
    let mut in_file = false;
    for raw in fs::read_to_string(path)?.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line == "[[models]]" {
            continue;
        }
        if line == "[[models.files]]" {
            in_file = true;
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .context("manifest entry must use key = value")?;
        let key = key.trim();
        let value = value.trim();
        let text = value.trim_matches('"');
        if in_file {
            match key {
                "path" => file.path = text.into(),
                "bytes" => file.bytes = value.parse()?,
                "sha256" => file.sha256 = text.into(),
                _ => bail!("unknown model file manifest key `{key}`"),
            }
        } else {
            match key {
                "id" => model.id = text.into(),
                "source" => model.source = text.into(),
                "revision" => model.revision = text.into(),
                "path" => model.path = text.into(),
                "architecture" => model.architecture = text.into(),
                "tokenizer" => model.tokenizer = text.into(),
                "model_file" => model.model_file = text.into(),
                "embedded_tokenizer" => model.embedded_tokenizer = value.parse()?,
                "format" => model.format = text.into(),
                "bytes" => model.bytes = value.parse()?,
                _ => bail!("unknown model manifest key `{key}`"),
            }
        }
    }
    model.files.push(file);
    model.validate()?;
    Ok(ModelManifest {
        models: vec![model],
    })
}
fn resolve_model(id: &str) -> Result<ModelRecord> {
    load_manifest()?
        .models
        .into_iter()
        .find(|model| model.id == id)
        .with_context(|| format!("model ID `{id}` is not in {MODEL_MANIFEST}"))
}
fn verify_manifest_model(record: &ModelRecord) -> Result<()> {
    let mut total = 0;
    for file in &record.files {
        let path = record.path.join(&file.path);
        ensure!(
            fs::metadata(&path)?.len() == file.bytes,
            "byte size mismatch for {}",
            path.display()
        );
        ensure!(
            sha256_file(&path)? == file.sha256,
            "SHA-256 mismatch for {}",
            path.display()
        );
        total += file.bytes;
    }
    ensure!(total == record.bytes, "manifest byte total mismatch");
    let gguf = GgufModel::open(record.path.join(&record.model_file))?;
    ensure!(
        gguf.metadata
            .get("general.architecture")
            .map(String::as_str)
            == Some("gemma4"),
        "GGUF architecture mismatch"
    );
    Ok(())
}
fn load_verified_model(record: &ModelRecord) -> Result<Gemma4E2bModel> {
    verify_manifest_model(record)?;
    Gemma4E2bModel::load_gguf(record.path.join(&record.model_file))
}
fn sha256_file(path: &Path) -> Result<String> {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()?;
    ensure!(
        output.status.success(),
        "shasum failed for {}",
        path.display()
    );
    Ok(std::str::from_utf8(&output.stdout)?
        .split_whitespace()
        .next()
        .context("shasum produced no digest")?
        .into())
}
fn append_jsonl(value: &Value) -> Result<()> {
    append_jsonl_at(CHAT_PERFORMANCE_LOG, value)
}
fn append_jsonl_at(path: &str, value: &Value) -> Result<()> {
    let parent = Path::new(path)
        .parent()
        .context("JSONL artifact has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    use std::io::Write as _;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}
fn metal_info() -> Result<()> {
    let runtime = MetalRuntime::new()?;
    let info = runtime.device_info();
    println!(
        "device: {}\nregistry_id: {}\nmax_threadgroup_memory_bytes: {}\nmax_total_threads_per_threadgroup: {}",
        info.name,
        info.registry_id,
        info.max_threadgroup_memory_bytes,
        info.max_total_threads_per_threadgroup
    );
    Ok(())
}

#[cfg(test)]
mod kv_cache_cli_tests {
    use super::{
        Gemma4KvCacheType, Gemma4Q4AttentionMode, GpuCountersMode, ThoughtFilter,
        attention_dimensions, parse_benchmark_args, parse_chat_args, parse_profile_args,
        profiler_operation_family, token_ids_sha256, visible_chat_completion,
    };
    use atlas_profiler::{AttentionKind, AttentionScanPass, OperationFamily};

    #[test]
    fn resident_semantic_labels_have_exact_operation_attribution() {
        let cases = [
            ("attention_input_norm", OperationFamily::RmsNorm),
            ("post_attention_norm", OperationFamily::RmsNorm),
            ("post_attention_norm_residual", OperationFamily::RmsNorm),
            ("ffn_input_norm", OperationFamily::RmsNorm),
            ("post_ffn_norm", OperationFamily::RmsNorm),
            ("ple_norm", OperationFamily::RmsNorm),
            ("provider_kv_value_norm", OperationFamily::RmsNorm),
            ("layer_major_rms_norm", OperationFamily::RmsNorm),
            ("final_output_norm", OperationFamily::FinalNorm),
            (
                "attention_output_projection",
                OperationFamily::AttentionOutputProjection,
            ),
            ("ffn_down_projection", OperationFamily::FfnDown),
            (
                "layer_major_batched_ffn_down_projection",
                OperationFamily::FfnDown,
            ),
            ("ple_projection", OperationFamily::PleProjection),
            ("qkv_projection", OperationFamily::QkvProjection),
            ("ffn_gate_up_projection", OperationFamily::FfnGateUp),
            ("output_projection", OperationFamily::OutputProjection),
            (
                "gemma_attention_global_split_scan",
                OperationFamily::AttentionScore,
            ),
            (
                "gemma_attention_global_split_combine",
                OperationFamily::AttentionValue,
            ),
            ("layer_major_batched_projection", OperationFamily::Other),
        ];

        for (label, expected) in cases {
            assert_eq!(profiler_operation_family(label), expected, "label={label}");
        }
        assert_eq!(
            profiler_operation_family("attention_output_projection_extra"),
            OperationFamily::Other
        );
        assert_ne!(
            profiler_operation_family("ple_projection"),
            OperationFamily::OutputProjection
        );
    }

    #[test]
    fn attention_semantic_labels_preserve_kind_and_pass() {
        assert_eq!(
            attention_dimensions("gemma_attention_global_split_scan"),
            (Some(AttentionKind::Global), Some(AttentionScanPass::Scan))
        );
        assert_eq!(
            attention_dimensions("gemma_attention_sliding_split_combine"),
            (
                Some(AttentionKind::Sliding),
                Some(AttentionScanPass::Combine)
            )
        );
        assert_eq!(attention_dimensions("ffn_down_projection"), (None, None));
    }

    #[test]
    fn gpu_counter_mode_accepts_only_explicit_diagnostic_modes() {
        assert_eq!(
            GpuCountersMode::parse("auto").unwrap(),
            GpuCountersMode::Auto
        );
        assert_eq!(
            GpuCountersMode::parse("required").unwrap(),
            GpuCountersMode::Required
        );
        assert!(GpuCountersMode::parse("always").is_err());
    }

    #[test]
    fn chat_accepts_explicit_kv_cache_precision() {
        let args = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--kv-cache-type".to_owned(),
            "q8_0".to_owned(),
        ];
        let (_, _, _, _, cache_type, attention_mode) =
            parse_chat_args(&args).expect("parse chat options");
        assert_eq!(cache_type, Gemma4KvCacheType::Q8_0);
        assert_eq!(attention_mode, Gemma4Q4AttentionMode::LegacyFused);
    }

    #[test]
    fn chat_and_benchmark_accept_explicit_flash16_attention_mode() {
        let chat = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--q4-attention-mode".to_owned(),
            "flash16".to_owned(),
        ];
        assert_eq!(
            parse_chat_args(&chat).expect("parse chat options").5,
            Gemma4Q4AttentionMode::Flash16
        );
        let benchmark = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--prompt".to_owned(),
            "fixed workload".to_owned(),
            "--decode-tokens".to_owned(),
            "128".to_owned(),
            "--q4-attention-mode".to_owned(),
            "flash16".to_owned(),
        ];
        assert_eq!(
            parse_benchmark_args(&benchmark)
                .expect("parse benchmark options")
                .q4_attention_mode,
            Gemma4Q4AttentionMode::Flash16
        );
    }

    #[test]
    fn gemma_cli_defaults_to_the_promoted_q4_kv_cache() {
        let chat = vec!["--model".to_owned(), "gemma4-e2b-q4_0".to_owned()];
        assert_eq!(
            parse_chat_args(&chat).expect("parse chat options").4,
            Gemma4KvCacheType::Q4_0
        );

        let benchmark = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--prompt".to_owned(),
            "fixed workload".to_owned(),
            "--decode-tokens".to_owned(),
            "128".to_owned(),
        ];
        assert_eq!(
            parse_benchmark_args(&benchmark)
                .expect("parse benchmark options")
                .kv_cache_type,
            Gemma4KvCacheType::Q4_0
        );
        assert_eq!(
            parse_benchmark_args(&benchmark)
                .expect("parse benchmark options")
                .q4_attention_mode,
            Gemma4Q4AttentionMode::LegacyFused
        );

        let profile = vec!["--model".to_owned(), "gemma4-e2b-q4_0".to_owned()];
        assert_eq!(
            parse_profile_args(&profile)
                .expect("parse profile options")
                .kv_cache_type,
            Gemma4KvCacheType::Q4_0
        );
        assert_eq!(
            parse_profile_args(&profile)
                .expect("parse profile options")
                .warmup_decode_tokens,
            32
        );
    }

    #[test]
    fn benchmark_requires_an_explicit_fixed_decode_workload() {
        let args = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--prompt".to_owned(),
            "fixed workload".to_owned(),
            "--decode-tokens".to_owned(),
            "128".to_owned(),
            "--kv-cache-type".to_owned(),
            "q4_0".to_owned(),
        ];
        let parsed = parse_benchmark_args(&args).expect("parse");
        assert_eq!(parsed.model_id, "gemma4-e2b-q4_0");
        assert_eq!(parsed.prompt, "fixed workload");
        assert_eq!(parsed.measured_decode_tokens, 128);
        assert_eq!(parsed.warmup_decode_tokens, 32);
        assert_eq!(parsed.max_context, 4096);
        assert_eq!(parsed.kv_cache_type, Gemma4KvCacheType::Q4_0);
        assert_eq!(parsed.q4_attention_mode, Gemma4Q4AttentionMode::LegacyFused);
        assert!(parse_benchmark_args(&args[..4]).is_err());
    }

    #[test]
    fn benchmark_accepts_a_long_context_measurement_window() {
        let args = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--prompt".to_owned(),
            "fixed workload".to_owned(),
            "--warmup-decode-tokens".to_owned(),
            "1024".to_owned(),
            "--decode-tokens".to_owned(),
            "512".to_owned(),
            "--max-context".to_owned(),
            "2048".to_owned(),
        ];
        let parsed = parse_benchmark_args(&args).expect("parse");
        assert_eq!(parsed.warmup_decode_tokens, 1024);
        assert_eq!(parsed.measured_decode_tokens, 512);
        assert_eq!(parsed.max_context, 2048);
    }

    #[test]
    fn profile_accepts_a_long_context_horizon() {
        let args = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--decode-tokens".to_owned(),
            "1024".to_owned(),
            "--max-context".to_owned(),
            "2048".to_owned(),
            "--kv-cache-type".to_owned(),
            "q4_0".to_owned(),
        ];
        let parsed = parse_profile_args(&args).expect("parse");
        assert_eq!(parsed.decode_tokens, 1024);
        assert_eq!(parsed.max_context, 2048);
        assert_eq!(parsed.kv_cache_type, Gemma4KvCacheType::Q4_0);
    }

    #[test]
    fn chat_filter_hides_end_turn_markers_even_when_split_across_events() {
        let mut filter = ThoughtFilter::default();
        filter.push("Hello<turn");
        assert_eq!(filter.visible_delta(), "Hello");
        filter.push("|>");
        assert_eq!(filter.visible_delta(), "");
        let (visible, thoughts) = filter.finish();
        assert_eq!(visible, "Hello");
        assert!(thoughts.is_empty());
    }

    #[test]
    fn fixed_benchmark_prompt_identity_is_token_sensitive() {
        assert_eq!(token_ids_sha256(&[1, 2, 3]), token_ids_sha256(&[1, 2, 3]));
        assert_ne!(token_ids_sha256(&[1, 2, 3]), token_ids_sha256(&[1, 2, 4]));
    }

    #[test]
    fn generate_json_extracts_only_the_visible_chat_completion() {
        let prompt = "<|turn>user\nhi<turn|>\n<|turn>model\n";
        assert_eq!(
            visible_chat_completion(&format!("{prompt}Hello.<turn|>"), prompt),
            "Hello."
        );
    }
}
