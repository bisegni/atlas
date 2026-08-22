//! Atlas user CLI — minimal surface.
//!
//! The user-facing `atlas` binary exposes only model download and chat.
//! Benchmarking, profiling, llama.cpp comparison, the reference executor, and
//! the parity-test tooling live in the `atlas-dev` binary, which is used by
//! the development workflow and the test suite; this binary stays simple.

#![recursion_limit = "256"]

use std::{
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{GgufModel, GgufTensorType};
use atlas_model::{
    Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
    gemma4_executor::{
        Gemma4E2bExecutor, Gemma4FinishReason, Gemma4Generation, Gemma4KvCacheType,
        Gemma4Q4AttentionMode, Gemma4SelectedGroupFormat,
    },
    render_gemma4_chat,
};
use serde_json::{Value, json};

mod providers;

static CHAT_INTERRUPTED: AtomicBool = AtomicBool::new(false);
const MODEL_MANIFEST: &str = "models/manifest.toml";
const CHAT_PERFORMANCE_LOG: &str = "artifacts/chat-performance.jsonl";
const FLASH16_DIAGNOSIS_DIR: &str = "artifacts/flash16-diagnosis";

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ChatMetricsVerbosity {
    #[default]
    Silent,
    Text,
    Json,
}

impl ChatMetricsVerbosity {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "text" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            other => bail!("unknown --verbose mode `{other}`; expected `text` or `json`"),
        }
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("chat") => chat(&args[1..]),
        Some("model") => model_command(&args[1..]),
        _ => bail!(
            "usage: atlas chat --model gemma4-e2b-q4_0 [--prompt ...] [--max-tokens N] [--verbose text|json]\n       atlas model search <query>\n       atlas model download huggingface:<repo> --id <id>"
        ),
    }
}

fn chat(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    let (model_id, prompt, max_tokens, show_thoughts, kv_cache_type, q4_attention_mode, verbosity) =
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
            verbosity,
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
            verbosity,
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
    verbosity: ChatMetricsVerbosity,
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
        verbosity,
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
    verbosity: ChatMetricsVerbosity,
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
    let model_id = record.id.clone();
    let record = json!({
        "event": "generation_metrics",
        "model_id": model_id,
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
    append_jsonl(&record)?;
    match verbosity {
        ChatMetricsVerbosity::Silent => {}
        ChatMetricsVerbosity::Json => {
            eprintln!("{record}");
        }
        ChatMetricsVerbosity::Text => {
            emit_text_metrics(
                &model_id,
                generation,
                finish_reason,
                max_tokens,
                context_limit,
                prefill,
                decode,
            );
        }
    }
    Ok(())
}

fn emit_text_metrics(
    model_id: &str,
    generation: &Gemma4Generation,
    finish_reason: &str,
    max_tokens: usize,
    context_limit: bool,
    prefill_tok_s: f64,
    decode_tok_s: f64,
) {
    eprintln!("== generation metrics ==");
    eprintln!(
        "model:      {} · {} · {}",
        model_id,
        generation.metrics.weight_format.as_str(),
        "resident"
    );
    eprintln!(
        "cache:      {} KV · attention {} · {}",
        generation.metrics.kv_cache_type.as_str(),
        generation.metrics.q4_attention_mode.as_str(),
        generation.metrics.attention_kernel
    );
    eprintln!(
        "prefill:    {} tokens · {:.1} ms · {:.1} tok/s",
        generation.generation.prompt_token_ids.len(),
        generation.metrics.prefill.as_secs_f64() * 1000.0,
        prefill_tok_s,
    );
    eprintln!(
        "decode:     {} tokens · {:.1} ms · {:.1} tok/s",
        generation.generation.generated_token_ids.len(),
        generation.metrics.decode.as_secs_f64() * 1000.0,
        decode_tok_s,
    );
    eprintln!(
        "host total: {:.1} ms",
        generation.metrics.host_wall_time.as_secs_f64() * 1000.0,
    );
    eprintln!(
        "finish:     {finish_reason} · limit={} · max_new_tokens={max_tokens}",
        if context_limit { "context" } else { "explicit" },
    );
    eprintln!(
        "memory:     resident {} · kv {} · upload {} · readback {}",
        format_bytes(generation.metrics.resident_bytes),
        format_bytes(generation.metrics.kv_cache_bytes),
        format_bytes(generation.metrics.weight_upload_bytes),
        format_bytes(generation.metrics.readback_bytes),
    );
    eprintln!(
        "kernels:    embed {} · lm_head {}",
        generation.metrics.embedding_kernel, generation.metrics.output_projection_kernel,
    );
    eprintln!(
        "quant:      {} · {} rejections",
        generation.metrics.quantization_preflight_state,
        generation.metrics.quantization_rejections.len(),
    );
    if let Some(path) = generation.metrics.quantization_plan_path.as_deref() {
        eprintln!("plan:       {path}");
    }
    eprintln!("logged:     {CHAT_PERFORMANCE_LOG}");
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
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

fn gguf_tensor_type_name(kind: GgufTensorType) -> &'static str {
    match kind {
        GgufTensorType::F32 => "f32",
        GgufTensorType::F16 => "f16",
        GgufTensorType::Q4_0 => "q4_0",
        GgufTensorType::Q8_0 => "q8_0",
        GgufTensorType::Q6K => "q6_k",
    }
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
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
    ChatMetricsVerbosity,
)> {
    let mut model = None;
    let mut prompt = None;
    let mut max = None;
    let mut thoughts = false;
    let mut kv_cache_type = Gemma4KvCacheType::Q4_0;
    let mut q4_attention_mode = Gemma4Q4AttentionMode::default();
    let mut verbosity = ChatMetricsVerbosity::Silent;
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
            "--verbose" => match args.get(i + 1).map(String::as_str) {
                Some("text") | Some("json") => {
                    i += 1;
                    verbosity = ChatMetricsVerbosity::parse(
                        args.get(i).context("--verbose needs a value")?,
                    )?;
                }
                _ => verbosity = ChatMetricsVerbosity::Text,
            },
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
        verbosity,
    ))
}

fn model_command(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .context("model command requires a subcommand")?;
    match command.as_str() {
        "download" => model_download(&args[1..]),
        "search" => model_search(&args[1..]),
        _ => bail!("model command must be search or download"),
    }
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
