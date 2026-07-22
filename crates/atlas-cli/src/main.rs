use std::{
    collections::BTreeSet,
    env, fs,
    io::{self, BufRead, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{
    DType, GgufModel, GgufTensorType, GgufWriter, QuantFormat, QuantizedMatrix, quantize_q4_0,
    quantize_q8_0, read_safetensors_descriptors, read_safetensors_tensor_f32,
};
use atlas_metal::MetalRuntime;
use atlas_model::{
    AtlasModel, Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
    executor::{
        AtlasExecutor, ExecutorConfig, ExecutorGeneration, ExecutorMetrics, ExecutorMode,
        GenerationEvent, LogitsReadback, ResidentAttentionPath,
    },
    gemma4_executor::{Gemma4E2bExecutor, Gemma4FinishReason, Gemma4Generation},
    render_gemma4_chat,
    runtime::{AtlasRuntime, RuntimeConfig, RuntimeEvent, RuntimeRequest},
    sampling::SamplingConfig,
    validate_generation_golden,
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    crossterm::{
        event::{self, Event, KeyCode, KeyEventKind},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
};
use serde_json::{Value, json};

mod providers;

static CHAT_INTERRUPTED: AtomicBool = AtomicBool::new(false);

const MODEL_MANIFEST: &str = "models/manifest.toml";
const CHAT_PERFORMANCE_LOG: &str = "artifacts/chat-performance.jsonl";

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
    model_file: Option<PathBuf>,
    embedded_tokenizer: bool,
    format: String,
    bytes: u64,
    baseline_model: Option<String>,
    max_logit_abs_delta: Option<f32>,
    min_token_agreement: Option<f32>,
    max_resident_bytes: Option<u64>,
    files: Vec<ModelFile>,
}

impl ModelRecord {
    fn manifest_kind(&self) -> Result<ManifestModelKind> {
        match (
            self.architecture.as_str(),
            self.embedded_tokenizer,
            self.model_file.as_ref(),
            self.tokenizer.as_path(),
        ) {
            ("LlamaForCausalLM", false, None, _) => Ok(ManifestModelKind::Llama),
            ("gemma4", true, Some(_), tokenizer) if tokenizer == Path::new("embedded") => {
                Ok(ManifestModelKind::Gemma4E2b)
            }
            _ => bail!(
                "unsupported manifest/model combination for `{}`: architecture=`{}`, embedded_tokenizer={}, model_file={}, tokenizer={}; Atlas supports LlamaForCausalLM with an external tokenizer or gemma4 E2B with tokenizer = \"embedded\"",
                self.id,
                self.architecture,
                self.embedded_tokenizer,
                self.model_file
                    .as_ref()
                    .map_or("<none>", |path| path.to_str().unwrap_or("<non-UTF-8>")),
                self.tokenizer.display(),
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManifestModelKind {
    Llama,
    Gemma4E2b,
}

#[derive(Debug)]
struct ModelFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Debug)]
struct ModelSelection {
    id: String,
    directory: PathBuf,
    manifest: Option<ModelRecord>,
}

/// Generation-family selection stays explicit at the CLI boundary. Gemma is
/// intentionally never coerced through AtlasModel's Llama-only loader.
enum LoadedModel {
    Llama(AtlasModel),
    Gemma4E2b(Gemma4E2bModel),
}

fn gemma4_resident_executor_unavailable(_: &Gemma4E2bModel) -> anyhow::Error {
    gemma4_resident_executor_unavailable_error()
}

fn gemma4_resident_executor_unavailable_error() -> anyhow::Error {
    anyhow::anyhow!(
        "Gemma 4 resident executor is not implemented; Atlas will not fall back to Llama or Reference"
    )
}

extern "C" fn chat_sigint_handler(_: i32) {
    // Signal handlers may only perform async-signal-safe work. Atomic stores
    // are sufficient here; generation observes this flag between tokens.
    CHAT_INTERRUPTED.store(true, Ordering::Release);
}

unsafe extern "C" {
    fn signal(signal: i32, handler: extern "C" fn(i32)) -> usize;
}

fn install_chat_sigint_handler() {
    const SIGINT: i32 = 2;
    // The CLI is macOS-only through atlas-metal. Replacing the default handler
    // lets `/quit`, EOF, and Ctrl-C share the same metrics handoff.
    unsafe { signal(SIGINT, chat_sigint_handler) };
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("metal-info") => metal_info(),
        Some("fixture") if args.get(1).map(String::as_str) == Some("verify") => {
            fixture_verify(&args[2..])
        }
        Some("provider") => provider_command(&args[1..]),
        Some("model") => model_command(&args[1..]),
        Some("generate") => generate(&args[1..]),
        Some("chat") => chat(&args[1..]),
        Some("runtime") => runtime_command(&args[1..]),
        Some("phase_03_model") => phase_03_model(&args[1..]),
        Some("phase_05_quant") => phase_05_quant(&args[1..]),
        Some("phase_08b_decode") => phase_08b_decode(&args[1..]),
        _ => {
            eprintln!(
                "usage: atlas-cli generate|chat --model ID ... | atlas-cli provider login|logout|status|default [huggingface] | atlas-cli model search [--provider huggingface] [--json] QUERY | atlas-cli model download PROVIDER_MODEL_ID --id ID | atlas-cli model inspect|verify --model ID"
            );
            bail!("invalid command")
        }
    }
}

fn provider_command(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .context("provider command requires a subcommand")?;
    match command.as_str() {
        "default" => {
            let value = args
                .get(1)
                .context("provider default requires a provider ID or --clear")?;
            if value == "--clear" {
                providers::set_default_provider(None)?;
                println!("{}", json!({"default_provider":null}));
            } else {
                providers::set_default_provider(Some(value))?;
                println!("{}", json!({"default_provider":value}));
            }
        }
        "status" => {
            let provider = args
                .get(1)
                .map(String::as_str)
                .unwrap_or(providers::HUGGING_FACE);
            let (source, _) = providers::token(provider)?;
            let state = match source {
                providers::AuthSource::Environment => "environment",
                providers::AuthSource::Keychain => "keychain",
                providers::AuthSource::Missing => "unauthenticated",
            };
            println!(
                "{}",
                json!({"provider":provider,"authentication":state,"default_provider":providers::load_default_provider()?})
            );
        }
        "login" => {
            let provider = args
                .get(1)
                .map(String::as_str)
                .unwrap_or(providers::HUGGING_FACE);
            ensure!(
                provider == providers::HUGGING_FACE,
                "provider `{provider}` does not support login"
            );
            eprint!("Hugging Face access token (read or fine-grained): ");
            let value = rpassword::read_password().context("read Hugging Face access token")?;
            providers::validate_hugging_face_token(&value)?;
            providers::store_token(provider, &value)?;
            println!(
                "{}",
                json!({"provider":provider,"authenticated":true,"credential_store":"keychain"})
            );
        }
        "logout" => {
            let provider = args
                .get(1)
                .map(String::as_str)
                .unwrap_or(providers::HUGGING_FACE);
            providers::logout(provider)?;
            println!("{}", json!({"provider":provider,"authenticated":false}));
        }
        _ => bail!("provider command must be `login`, `logout`, `status`, or `default`"),
    }
    Ok(())
}

/// Run exactly one request through the reusable bounded runtime. Multi-session
/// admission is a library concern and is intentionally exercised by the
/// runtime acceptance test rather than exposed as an unstable CLI protocol.
fn runtime_command(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    install_chat_sigint_handler();
    let (model_args, prompt, token_limit, mode, show_thoughts) = parse_chat_args(args)?;
    ensure!(
        !show_thoughts,
        "--show-thoughts is only valid for Gemma 4 chat"
    );
    ensure!(
        mode == ExecutorMode::Resident,
        "atlas runtime requires --executor resident"
    );
    let prompt = prompt.context("atlas runtime requires --prompt")?;
    let selection = resolve_model(&model_args)?;
    let model = AtlasModel::load(&selection.directory)?;
    let runtime_config = RuntimeConfig::default();
    let prompt_tokens = model.tokenize(&prompt)?.len();
    let resolved_limit = token_limit.resolve(prompt_tokens, runtime_config.max_context)?;
    let mut runtime = AtlasRuntime::new(&model, runtime_config)?;
    let session = runtime.submit(RuntimeRequest {
        prompt,
        max_new_tokens: resolved_limit.max_new_tokens,
        sampling: SamplingConfig::default(),
    })?;
    let mut stdout = io::stdout();
    runtime.run_until_idle(|event| {
        match event {
            RuntimeEvent::Generation {
                session: event_session,
                event: GenerationEvent::Token { text, .. },
            } => {
                ensure!(
                    event_session == session,
                    "single-session runtime emitted unexpected session"
                );
                write!(stdout, "{text}")?;
                stdout.flush()?;
            }
            RuntimeEvent::Generation {
                session: event_session,
                event: GenerationEvent::Failed { message },
            } => {
                bail!("runtime session {} failed: {message}", event_session.0);
            }
            _ => {}
        }
        Ok(())
    })?;
    let completion = runtime
        .take_completed()
        .context("runtime completed without a session result")?;
    println!();
    println!(
        "{}",
        json!({
            "event": "runtime_metrics",
            "session": completion.session.0,
            "executor": match completion.metrics.executor_mode { ExecutorMode::Reference => "reference", ExecutorMode::Resident => "resident" },
            "queue_wait_ms": completion.metrics.queue_wait.as_millis(),
            "ttft_ms": completion.metrics.executor.ttft.as_millis(),
            "decode_tokens": completion.metrics.executor.decode_tokens,
            "max_new_tokens": resolved_limit.max_new_tokens,
            "token_limit_source": resolved_limit.source,
            "cache_resident_bytes": completion.metrics.executor.resident_bytes,
            "cancelled": completion.metrics.cancelled,
            "error": completion.metrics.error,
        })
    );
    Ok(())
}

fn chat(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    install_chat_sigint_handler();
    let (model_args, prompt, token_limit, mode, show_thoughts) = parse_chat_args(args)?;
    let selection = resolve_model(&model_args)?;
    let model = match load_verified_model(&selection)? {
        LoadedModel::Llama(model) => model,
        LoadedModel::Gemma4E2b(model) => {
            return gemma4_chat(&model, prompt, token_limit, show_thoughts, &selection);
        }
    };
    ensure!(
        !show_thoughts,
        "--show-thoughts is only valid for Gemma 4 chat"
    );
    if mode == ExecutorMode::Resident {
        let _ = resident_executor(&model, &selection, LogitsReadback::SelectedToken)?;
    }
    if let Some(prompt) = prompt {
        let mut turn_metrics = ChatTurnMetrics::new();
        let (result, resolved_limit) = print_completion(
            &model,
            &prompt,
            token_limit,
            mode,
            &selection.id,
            selection.manifest.as_ref().map_or(0, |record| record.bytes),
            &mut turn_metrics,
        )?;
        append_chat_performance_record(&selection.id, &model, mode, &result, resolved_limit)?;
        return Ok(());
    }
    eprintln!("Atlas chat. Commands: /reset, /help, /quit");
    let stdin = io::stdin();
    let mut history = String::new();
    let mut session_metrics = ChatSessionMetrics::default();
    loop {
        if CHAT_INTERRUPTED.load(Ordering::Acquire) {
            eprintln!("generation interrupted");
            break;
        }
        print!("you> ");
        io::stdout().flush()?;
        let mut line = String::new();
        let bytes = match stdin.lock().read_line(&mut line) {
            Ok(bytes) => bytes,
            Err(_) if CHAT_INTERRUPTED.load(Ordering::Acquire) => {
                eprintln!("generation interrupted");
                break;
            }
            Err(error) => return Err(error.into()),
        };
        if bytes == 0 {
            break;
        }
        match repl_command(line.trim()) {
            ReplCommand::Quit => break,
            ReplCommand::Help => {
                println!("/reset clears the conversation; /quit exits");
                continue;
            }
            ReplCommand::Reset => {
                history.clear();
                println!("conversation reset");
                continue;
            }
            ReplCommand::Ignore => continue,
            ReplCommand::Prompt(line) => {
                append_user_turn(&mut history, line);
            }
        }
        let mut turn_metrics = ChatTurnMetrics::new();
        let (result, resolved_limit) = match print_completion(
            &model,
            &history,
            token_limit,
            mode,
            &selection.id,
            selection.manifest.as_ref().map_or(0, |record| record.bytes),
            &mut turn_metrics,
        ) {
            Ok(result) => result,
            Err(_) if CHAT_INTERRUPTED.load(Ordering::Acquire) => {
                session_metrics.record(&turn_metrics);
                eprintln!("generation interrupted");
                break;
            }
            Err(error) => return Err(error),
        };
        append_chat_performance_record(&selection.id, &model, mode, &result, resolved_limit)?;
        session_metrics.record(&turn_metrics);
        history.push_str(&model.decode(&result.generation.generated_token_ids)?);
        history.push('\n');
    }
    eprintln!("{}", session_metrics_line(&session_metrics));
    Ok(())
}

#[derive(Debug, Default)]
struct ChatSessionMetrics {
    turns: usize,
    generated_tokens: usize,
    active_turn_time: std::time::Duration,
}

impl ChatSessionMetrics {
    fn record(&mut self, turn: &ChatTurnMetrics) {
        self.turns += 1;
        self.generated_tokens += turn.generated_tokens;
        self.active_turn_time += turn.started.elapsed();
    }

    fn generated_tokens_per_second(&self) -> f64 {
        if self.active_turn_time.is_zero() {
            0.0
        } else {
            self.generated_tokens as f64 / self.active_turn_time.as_secs_f64()
        }
    }
}

#[derive(Debug)]
struct ChatTurnMetrics {
    started: std::time::Instant,
    generated_tokens: usize,
}

impl ChatTurnMetrics {
    fn new() -> Self {
        Self {
            started: std::time::Instant::now(),
            generated_tokens: 0,
        }
    }

    fn record_event(&mut self, event: &GenerationEvent) {
        if matches!(event, GenerationEvent::Token { .. }) {
            self.generated_tokens += 1;
        }
    }
}

fn print_completion(
    model: &AtlasModel,
    prompt: &str,
    token_limit: ChatTokenLimit,
    mode: ExecutorMode,
    model_id: &str,
    model_bytes: u64,
    turn_metrics: &mut ChatTurnMetrics,
) -> Result<(ExecutorGeneration, ResolvedTokenLimit)> {
    let config = ExecutorConfig {
        mode,
        ..Default::default()
    };
    let prompt_tokens = model.tokenize(prompt)?.len();
    let resolved_limit = token_limit.resolve(prompt_tokens, config.max_context)?;
    let mut executor = AtlasExecutor::new(model, config)?;
    let mut stdout = io::stdout();
    let result = executor.generate_greedy_stream(
        prompt,
        resolved_limit.max_new_tokens,
        &CHAT_INTERRUPTED,
        |event| {
            turn_metrics.record_event(&event);
            write_stream_event(&mut stdout, &event)
        },
    )?;
    writeln!(stdout)?;
    stdout.flush()?;
    eprintln!(
        "{}",
        json!({
            "event": "generation_metrics",
            "model_id": model_id,
            "executor": match mode { ExecutorMode::Reference => "reference", ExecutorMode::Resident => "resident" },
            "format": model.format_name(),
            "model_bytes": model_bytes,
            "finish_reason": format!("{:?}", result.finish_reason).to_lowercase(),
            "max_new_tokens": resolved_limit.max_new_tokens,
            "token_limit_source": resolved_limit.source,
            "resident_bytes": result.metrics.resident_bytes,
            "weight_upload_bytes": result.metrics.weight_upload_bytes,
            "readback_bytes": result.metrics.readback_bytes,
            "command_buffers": result.metrics.command_buffer_count,
            "timing": { "ttft_ms": result.metrics.ttft.as_secs_f64() * 1000.0, "host_ms": result.metrics.host_wall_time.as_secs_f64() * 1000.0 },
            "decode_tok_s": result.metrics.decode_tokens_per_second(),
        })
    );
    Ok((result, resolved_limit))
}

fn print_gemma4_generation(
    model: &Gemma4E2bModel,
    prompt: &str,
    max_tokens: usize,
    selection: &ModelSelection,
) -> Result<()> {
    let mut executor = Gemma4E2bExecutor::new(model, 4096)?;
    let generation = executor.generate_greedy(prompt, max_tokens)?;
    println!(
        "prompt_token_ids: {:?}",
        generation.generation.prompt_token_ids
    );
    println!(
        "generated_token_ids: {:?}",
        generation.generation.generated_token_ids
    );
    println!("text: {}", generation.generation.text);
    if !generation.generation.final_logits.is_empty() {
        let mut top = generation
            .generation
            .final_logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        println!(
            "gemma4_top_logits: {:?}",
            top.into_iter().take(8).collect::<Vec<_>>()
        );
    }
    println!(
        "{}",
        json!({
            "event":"generation_metrics", "model_id":selection.id, "executor":"resident",
            "format":"gguf-gemma4-q4_0", "model_bytes":selection.manifest.as_ref().map_or(0, |record| record.bytes),
            "finish_reason":gemma4_finish_reason(generation.finish_reason),
            "resident_bytes":generation.metrics.resident_bytes,
            "weight_upload_bytes":generation.metrics.weight_upload_bytes,
            "readback_bytes":generation.metrics.readback_bytes,
            "command_buffers":generation.metrics.command_buffers,
            "prefill_command_buffers":generation.metrics.prefill_command_buffers,
            "decode_command_buffers":generation.metrics.decode_command_buffers,
            "prefill_tok_s":gemma4_prefill_tokens_per_second(&generation),
            "decode_tok_s":gemma4_decode_tokens_per_second(&generation),
            "timing":{"prefill_ms":generation.metrics.prefill.as_secs_f64()*1000.0,"decode_ms":generation.metrics.decode.as_secs_f64()*1000.0,"host_ms":generation.metrics.host_wall_time.as_secs_f64()*1000.0}
        })
    );
    Ok(())
}

fn gemma4_chat(
    model: &Gemma4E2bModel,
    prompt: Option<String>,
    token_limit: ChatTokenLimit,
    show_thoughts: bool,
    selection: &ModelSelection,
) -> Result<()> {
    let mut executor = Gemma4E2bExecutor::new(model, 4096)?;
    let mut messages = Vec::new();
    let mut session = ChatSessionMetrics::default();
    if let Some(prompt) = prompt {
        messages.push(Gemma4ChatMessage::new(Gemma4ChatRole::User, prompt));
        let _ = run_gemma4_turn(
            model,
            &mut executor,
            &messages,
            token_limit,
            show_thoughts,
            selection,
            &mut session,
        )?;
        return Ok(());
    }
    eprintln!("Atlas Gemma 4 chat. Commands: /reset, /help, /quit");
    let stdin = io::stdin();
    loop {
        if CHAT_INTERRUPTED.load(Ordering::Acquire) {
            break;
        }
        print!("you> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        match repl_command(line.trim()) {
            ReplCommand::Quit => break,
            ReplCommand::Help => {
                println!("/reset clears the conversation; /quit exits");
                continue;
            }
            ReplCommand::Reset => {
                messages.clear();
                executor.reset();
                println!("conversation reset");
                continue;
            }
            ReplCommand::Ignore => continue,
            ReplCommand::Prompt(text) => {
                messages.push(Gemma4ChatMessage::new(Gemma4ChatRole::User, text))
            }
        }
        compact_gemma4_history(model, &mut executor, &mut messages, token_limit, selection)?;
        match run_gemma4_turn(
            model,
            &mut executor,
            &messages,
            token_limit,
            show_thoughts,
            selection,
            &mut session,
        ) {
            Ok(visible) => messages.push(Gemma4ChatMessage::new(Gemma4ChatRole::Model, visible)),
            Err(error) => {
                messages.pop();
                return Err(error);
            }
        }
    }
    eprintln!("{}", session_metrics_line(&session));
    Ok(())
}

fn compact_gemma4_history(
    model: &Gemma4E2bModel,
    executor: &mut Gemma4E2bExecutor<'_>,
    messages: &mut Vec<Gemma4ChatMessage>,
    token_limit: ChatTokenLimit,
    selection: &ModelSelection,
) -> Result<()> {
    while model.tokenize(&render_gemma4_chat(messages)?)?.len()
        + token_limit.minimum_response_budget()
        > 4096
    {
        let offset = usize::from(
            messages
                .first()
                .is_some_and(|m| m.role == Gemma4ChatRole::System),
        );
        let complete_pairs_before_newest = messages.len().saturating_sub(offset + 1) / 2;
        ensure!(
            complete_pairs_before_newest > 0,
            "Gemma chat turn exceeds the 4096-token context limit and no older complete pair can be summarized"
        );
        let prior_summary = if offset == 1 {
            Some(messages[0].content.as_str())
        } else {
            None
        };
        let user = &messages[offset].content;
        let answer = &messages[offset + 1].content;
        let summary_request = format!(
            "Summarize the following earlier conversation faithfully and compactly for future turns. Preserve names, facts, decisions, and unresolved requests. Return only the summary.\n\nPrior summary:\n{}\n\nUser:\n{}\n\nModel:\n{}",
            prior_summary.unwrap_or("(none)"),
            user,
            answer
        );
        let summary_prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
            Gemma4ChatRole::User,
            summary_request,
        )])?;
        let summarized_input_tokens = model.tokenize(&summary_prompt)?.len();
        ensure!(
            summarized_input_tokens + 256 <= 4096,
            "conversation history cannot be summarized within the Gemma context limit"
        );
        let mut filter = ThoughtFilter::default();
        let generation = executor.generate_greedy_chat_stream(
            &summary_prompt,
            256,
            &CHAT_INTERRUPTED,
            |event| {
                filter.push(&event.text);
                Ok(())
            },
        )?;
        ensure!(
            generation.finish_reason != Gemma4FinishReason::Cancelled,
            "conversation summarization cancelled"
        );
        let (summary, _, _) = filter.finish();
        ensure!(
            !summary.trim().is_empty(),
            "Gemma conversation summarizer returned an empty summary"
        );
        replace_oldest_gemma4_pair_with_summary(messages, summary.trim())?;
        eprintln!("conversation history summarized");
        eprintln!(
            "{}",
            json!({"event":"conversation_compaction","model_id":selection.id,"executor":"resident","compaction_count":1,"summarized_input_tokens":summarized_input_tokens,"summary_tokens":generation.generation.generated_token_ids.len(),"resident_bytes":generation.metrics.resident_bytes,"weight_upload_bytes":generation.metrics.weight_upload_bytes,"readback_bytes":generation.metrics.readback_bytes,"command_buffers":generation.metrics.command_buffers})
        );
    }
    Ok(())
}

fn replace_oldest_gemma4_pair_with_summary(
    messages: &mut Vec<Gemma4ChatMessage>,
    summary: &str,
) -> Result<()> {
    ensure!(
        !summary.trim().is_empty(),
        "Gemma conversation summary is empty"
    );
    let offset = usize::from(
        messages
            .first()
            .is_some_and(|message| message.role == Gemma4ChatRole::System),
    );
    ensure!(
        messages.len() >= offset + 3,
        "no older complete Gemma pair is available to summarize"
    );
    ensure!(
        messages[offset].role == Gemma4ChatRole::User
            && messages[offset + 1].role == Gemma4ChatRole::Model,
        "oldest Gemma history entries are not a complete pair"
    );
    let replacement = Gemma4ChatMessage::new(Gemma4ChatRole::System, summary.trim());
    if offset == 1 {
        messages.splice(0..3, [replacement]);
    } else {
        messages.splice(0..2, [replacement]);
    }
    Ok(())
}

fn gemma4_finish_reason(reason: Gemma4FinishReason) -> &'static str {
    match reason {
        Gemma4FinishReason::Eos => "eos",
        Gemma4FinishReason::MaxTokens => "max_tokens",
        Gemma4FinishReason::Cancelled => "cancelled",
    }
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / elapsed.as_secs_f64()
    }
}

fn gemma4_prefill_tokens_per_second(generation: &Gemma4Generation) -> f64 {
    tokens_per_second(
        generation.generation.prompt_token_ids.len(),
        generation.metrics.prefill,
    )
}

fn gemma4_decode_tokens_per_second(generation: &Gemma4Generation) -> f64 {
    tokens_per_second(
        usize::try_from(generation.metrics.decode_command_buffers).unwrap_or(usize::MAX),
        generation.metrics.decode,
    )
}

#[derive(Default)]
struct ThoughtFilter {
    pending: String,
    in_thought: bool,
    visible: String,
    thoughts: String,
}

impl ThoughtFilter {
    fn push(&mut self, fragment: &str) -> String {
        const OPEN: &str = "<|channel>thought";
        const CLOSE: &str = "<channel|>";
        const MARKERS: &[&str] = &[
            OPEN,
            CLOSE,
            "<|turn>",
            "<turn|>",
            "<eos>",
            "<bos>",
            "<|endoftext|>",
        ];
        self.pending.push_str(fragment);
        let mut output = String::new();
        loop {
            let found = MARKERS
                .iter()
                .filter_map(|marker| self.pending.find(marker).map(|index| (index, *marker)))
                .min_by_key(|(index, _)| *index);
            if let Some((index, marker)) = found {
                let prefix: String = self.pending.drain(..index).collect();
                if self.in_thought {
                    self.thoughts.push_str(&prefix);
                } else {
                    self.visible.push_str(&prefix);
                    output.push_str(&prefix);
                }
                self.pending.drain(..marker.len());
                if marker == OPEN {
                    self.in_thought = true;
                } else if marker == CLOSE {
                    self.in_thought = false;
                }
            } else {
                let keep = self
                    .pending
                    .char_indices()
                    .map(|(index, _)| self.pending.len() - index)
                    .find(|length| {
                        let suffix = &self.pending[self.pending.len() - length..];
                        MARKERS.iter().any(|marker| marker.starts_with(suffix))
                    })
                    .unwrap_or(0);
                let emit_len = self.pending.len() - keep;
                let prefix: String = self.pending.drain(..emit_len).collect();
                if self.in_thought {
                    self.thoughts.push_str(&prefix);
                } else {
                    self.visible.push_str(&prefix);
                    output.push_str(&prefix);
                }
                break;
            }
        }
        output
    }
    fn finish(mut self) -> (String, String, String) {
        let tail = std::mem::take(&mut self.pending);
        if self.in_thought {
            self.thoughts.push_str(&tail);
        } else {
            self.visible.push_str(&tail);
        }
        (
            self.visible,
            self.thoughts,
            if self.in_thought { String::new() } else { tail },
        )
    }
}

fn run_gemma4_turn(
    model: &Gemma4E2bModel,
    executor: &mut Gemma4E2bExecutor<'_>,
    messages: &[Gemma4ChatMessage],
    token_limit: ChatTokenLimit,
    show_thoughts: bool,
    selection: &ModelSelection,
    session: &mut ChatSessionMetrics,
) -> Result<String> {
    let rendered = render_gemma4_chat(messages)?;
    let prompt_tokens = model.tokenize(&rendered)?.len();
    let resolved_limit = token_limit.resolve(prompt_tokens, 4096)?;
    print!("model> ");
    io::stdout().flush()?;
    let mut filter = ThoughtFilter::default();
    let mut count = 0usize;
    let generation = executor.generate_greedy_chat_stream(
        &rendered,
        resolved_limit.max_new_tokens,
        &CHAT_INTERRUPTED,
        |event| {
            count += 1;
            let text = filter.push(&event.text);
            print!("{text}");
            io::stdout().flush()?;
            Ok(())
        },
    )?;
    let (visible, thoughts, tail) = filter.finish();
    print!("{tail}\n");
    io::stdout().flush()?;
    if show_thoughts && !thoughts.is_empty() {
        eprintln!("thought> {thoughts}");
    }
    emit_gemma4_metrics(selection, &generation, resolved_limit)?;
    append_gemma4_performance_record(selection, &generation, &visible, resolved_limit)?;
    session.turns += 1;
    session.generated_tokens += count;
    session.active_turn_time += generation.metrics.host_wall_time;
    Ok(visible)
}

fn emit_gemma4_metrics(
    selection: &ModelSelection,
    generation: &Gemma4Generation,
    resolved_limit: ResolvedTokenLimit,
) -> Result<()> {
    eprintln!(
        "{}",
        json!({"event":"generation_metrics","model_id":selection.id,"executor":"resident","format":"gguf-gemma4-q4_0","finish_reason":gemma4_finish_reason(generation.finish_reason),"max_new_tokens":resolved_limit.max_new_tokens,"token_limit_source":resolved_limit.source,"resident_bytes":generation.metrics.resident_bytes,"weight_upload_bytes":generation.metrics.weight_upload_bytes,"readback_bytes":generation.metrics.readback_bytes,"command_buffers":generation.metrics.command_buffers,"prefill_command_buffers":generation.metrics.prefill_command_buffers,"decode_command_buffers":generation.metrics.decode_command_buffers,"prefill_path":generation.metrics.prefill_path,"prefill_chunk_size":generation.metrics.prefill_chunk_size,"prompt_tokens":generation.generation.prompt_token_ids.len(),"generated_tokens":generation.generation.generated_token_ids.len(),"prefill_tok_s":gemma4_prefill_tokens_per_second(generation),"decode_tok_s":gemma4_decode_tokens_per_second(generation),"timing":{"prefill_ms":generation.metrics.prefill.as_secs_f64()*1000.0,"decode_ms":generation.metrics.decode.as_secs_f64()*1000.0,"host_ms":generation.metrics.host_wall_time.as_secs_f64()*1000.0}})
    );
    Ok(())
}

fn append_gemma4_performance_record(
    selection: &ModelSelection,
    generation: &Gemma4Generation,
    visible: &str,
    resolved_limit: ResolvedTokenLimit,
) -> Result<()> {
    append_jsonl_record(
        Path::new(CHAT_PERFORMANCE_LOG),
        &json!({"model_id":selection.id,"executor":"resident","format":"gguf-gemma4-q4_0","prompt_tokens":generation.generation.prompt_token_ids.len(),"generated_tokens":generation.generation.generated_token_ids.len(),"finish_reason":gemma4_finish_reason(generation.finish_reason),"max_new_tokens":resolved_limit.max_new_tokens,"token_limit_source":resolved_limit.source,"visible_chars":visible.chars().count(),"prefill_tok_s":gemma4_prefill_tokens_per_second(generation),"decode_tok_s":gemma4_decode_tokens_per_second(generation),"host_ms":generation.metrics.host_wall_time.as_secs_f64()*1000.0,"resident_bytes":generation.metrics.resident_bytes,"weight_upload_bytes":generation.metrics.weight_upload_bytes,"readback_bytes":generation.metrics.readback_bytes,"command_buffers":generation.metrics.command_buffers,"prefill_command_buffers":generation.metrics.prefill_command_buffers,"decode_command_buffers":generation.metrics.decode_command_buffers,"prefill_path":generation.metrics.prefill_path,"prefill_chunk_size":generation.metrics.prefill_chunk_size}),
    )?;
    eprintln!("chat performance log: {CHAT_PERFORMANCE_LOG}");
    Ok(())
}

fn load_verified_model(selection: &ModelSelection) -> Result<LoadedModel> {
    if let Some(record) = &selection.manifest {
        verify_manifest_model(record)?;
        return match record.manifest_kind()? {
            ManifestModelKind::Llama => {
                Ok(LoadedModel::Llama(AtlasModel::load(&selection.directory)?))
            }
            ManifestModelKind::Gemma4E2b => {
                let path = safe_model_file(
                    &selection.directory,
                    record
                        .model_file
                        .as_ref()
                        .expect("Gemma manifest has model_file"),
                )?;
                Ok(LoadedModel::Gemma4E2b(Gemma4E2bModel::load_gguf(path)?))
            }
        };
    }
    // An explicit --model-dir remains the existing Llama developer-fixture
    // contract; manifest-backed selection never guesses an architecture.
    Ok(LoadedModel::Llama(AtlasModel::load(&selection.directory)?))
}

fn resident_executor<'a>(
    model: &'a AtlasModel,
    selection: &ModelSelection,
    logits_readback: LogitsReadback,
) -> Result<AtlasExecutor<'a>> {
    resident_executor_with_eos(model, selection, logits_readback, true, false)
}

fn resident_executor_with_eos<'a>(
    model: &'a AtlasModel,
    selection: &ModelSelection,
    logits_readback: LogitsReadback,
    stop_on_eos: bool,
    _resident_decode_profile: bool,
) -> Result<AtlasExecutor<'a>> {
    let executor = AtlasExecutor::new(
        model,
        ExecutorConfig {
            mode: ExecutorMode::Resident,
            logits_readback,
            stop_on_eos,
            // Q8's exact-token gate is presently protected by the legacy
            // numerical boundaries. Fused remains an explicit test selector.
            resident_attention_path: ResidentAttentionPath::LegacyThreePass,
            ..Default::default()
        },
    )?;
    if let Some(limit) = selection
        .manifest
        .as_ref()
        .and_then(|record| record.max_resident_bytes)
    {
        ensure!(
            executor.resident_bytes() <= limit,
            "resident memory budget exceeded for `{}`: {} > {} bytes",
            selection.id,
            executor.resident_bytes(),
            limit
        );
    }
    Ok(executor)
}

fn generation_metrics_json(
    model_id: &str,
    model: &AtlasModel,
    result: &ExecutorGeneration,
    model_bytes: u64,
) -> Value {
    json!({
        "event": "generation_metrics",
        "model_id": model_id,
        "executor": "resident",
        "format": model.format_name(),
        "model_bytes": model_bytes,
        "finish_reason": format!("{:?}", result.finish_reason).to_lowercase(),
        "resident_bytes": result.metrics.resident_bytes,
        "weight_upload_bytes": result.metrics.weight_upload_bytes,
        "readback_bytes": result.metrics.readback_bytes,
        "command_buffers": result.metrics.command_buffer_count,
        "timing": { "ttft_ms": result.metrics.ttft.as_secs_f64() * 1000.0, "host_ms": result.metrics.host_wall_time.as_secs_f64() * 1000.0 },
        "decode_tok_s": result.metrics.decode_tokens_per_second(),
    })
}

fn chat_performance_record(
    model_id: &str,
    model: &AtlasModel,
    mode: ExecutorMode,
    result: &ExecutorGeneration,
    resolved_limit: ResolvedTokenLimit,
) -> Result<Value> {
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock precedes Unix epoch")?
        .as_millis();
    Ok(json!({
        "timestamp_ms": timestamp_ms,
        "model_id": model_id,
        "format": model.format_name(),
        "executor": match mode { ExecutorMode::Reference => "reference", ExecutorMode::Resident => "resident" },
        "prompt_tokens": result.generation.prompt_token_ids.len(),
        "generated_tokens": result.generation.generated_token_ids.len(),
        "finish_reason": format!("{:?}", result.finish_reason).to_lowercase(),
        "max_new_tokens": resolved_limit.max_new_tokens,
        "token_limit_source": resolved_limit.source,
        "ttft_ms": result.metrics.ttft.as_secs_f64() * 1000.0,
        "prefill_tok_s": result.metrics.prefill_tokens_per_second(),
        "decode_tok_s": result.metrics.decode_tokens_per_second(),
        "host_ms": result.metrics.host_wall_time.as_secs_f64() * 1000.0,
        "gpu_ms": result.metrics.gpu_execution_time.as_secs_f64() * 1000.0,
        "command_buffers": result.metrics.command_buffer_count,
        "prefill_command_buffers": result.metrics.prefill_command_buffer_count,
        "decode_command_buffers": result.metrics.decode_command_buffer_count,
        "weight_upload_bytes": result.metrics.weight_upload_bytes,
        "readback_bytes": result.metrics.readback_bytes,
        "resident_bytes": result.metrics.resident_bytes,
    }))
}

fn append_jsonl_record(path: &Path, record: &Value) -> Result<()> {
    if let Some(directory) = path.parent() {
        fs::create_dir_all(directory).with_context(|| format!("create {}", directory.display()))?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    serde_json::to_writer(&mut file, record)?;
    writeln!(file)?;
    Ok(())
}

fn append_chat_performance_record(
    model_id: &str,
    model: &AtlasModel,
    mode: ExecutorMode,
    result: &ExecutorGeneration,
    resolved_limit: ResolvedTokenLimit,
) -> Result<()> {
    let path = Path::new(CHAT_PERFORMANCE_LOG);
    append_jsonl_record(
        path,
        &chat_performance_record(model_id, model, mode, result, resolved_limit)?,
    )?;
    eprintln!("chat performance log: {}", path.display());
    Ok(())
}

fn write_stream_event(writer: &mut impl Write, event: &GenerationEvent) -> Result<()> {
    if let GenerationEvent::Token { text, .. } = event {
        write!(writer, "{text}")?;
        writer.flush()?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatTokenLimit {
    Context,
    Explicit(usize),
}

impl ChatTokenLimit {
    fn minimum_response_budget(self) -> usize {
        match self {
            Self::Context => 1,
            Self::Explicit(max_new_tokens) => max_new_tokens,
        }
    }

    fn resolve(self, prompt_tokens: usize, max_context: usize) -> Result<ResolvedTokenLimit> {
        ensure!(
            prompt_tokens < max_context,
            "prompt uses {prompt_tokens} tokens and leaves no generation capacity in the {max_context}-token context"
        );
        let remaining = max_context - prompt_tokens;
        let (max_new_tokens, source) = match self {
            Self::Context => (remaining, "context"),
            Self::Explicit(max_new_tokens) => {
                ensure!(
                    max_new_tokens <= remaining,
                    "--max-tokens {max_new_tokens} exceeds the {remaining} tokens remaining in the {max_context}-token context"
                );
                (max_new_tokens, "explicit")
            }
        };
        Ok(ResolvedTokenLimit {
            max_new_tokens,
            source,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResolvedTokenLimit {
    max_new_tokens: usize,
    source: &'static str,
}

fn parse_chat_args(
    args: &[String],
) -> Result<(
    Vec<String>,
    Option<String>,
    ChatTokenLimit,
    ExecutorMode,
    bool,
)> {
    let mut model_args = Vec::new();
    let mut prompt = None;
    let mut token_limit = ChatTokenLimit::Context;
    let mut mode = ExecutorMode::Resident;
    let mut show_thoughts = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" | "--model-dir" => {
                model_args.push(args[index].clone());
                index += 1;
                model_args.push(
                    args.get(index)
                        .context("model option needs a value")?
                        .clone(),
                );
            }
            "--prompt" => {
                index += 1;
                prompt = Some(args.get(index).context("--prompt needs a value")?.clone());
            }
            "--max-tokens" => {
                index += 1;
                let max_tokens = args
                    .get(index)
                    .context("--max-tokens needs a value")?
                    .parse()
                    .context("parse --max-tokens")?;
                ensure!(max_tokens > 0, "--max-tokens must be positive");
                token_limit = ChatTokenLimit::Explicit(max_tokens);
            }
            "--executor" => {
                index += 1;
                mode = match args.get(index).map(String::as_str) {
                    Some("reference") => ExecutorMode::Reference,
                    Some("resident") => ExecutorMode::Resident,
                    _ => bail!("--executor must be `reference` or `resident`"),
                };
            }
            "--show-thoughts" => show_thoughts = true,
            flag => bail!("unknown chat option: {flag}"),
        };
        index += 1;
    }
    Ok((model_args, prompt, token_limit, mode, show_thoughts))
}

#[derive(Debug, PartialEq, Eq)]
enum ReplCommand<'a> {
    Quit,
    Help,
    Reset,
    Ignore,
    Prompt(&'a str),
}

fn repl_command(line: &str) -> ReplCommand<'_> {
    match line {
        "/quit" => ReplCommand::Quit,
        "/help" => ReplCommand::Help,
        "/reset" => ReplCommand::Reset,
        "" => ReplCommand::Ignore,
        line => ReplCommand::Prompt(line),
    }
}

fn append_user_turn(history: &mut String, line: &str) {
    history.push_str("user: ");
    history.push_str(line);
    history.push('\n');
    history.push_str("assistant: ");
}

fn metrics_line(metrics: &ExecutorMetrics) -> String {
    format!(
        "ttft_ms={:.2} prefill_tok_s={:.2} decode_tok_s={:.2} host_ms={:.2} gpu_ms={:.2} command_buffers={} prefill_command_buffers={} decode_command_buffers={} weight_upload_bytes={} readback_bytes={} resident_arena_allocations={} post_warmup_allocations={}",
        metrics.ttft.as_secs_f64() * 1000.0,
        metrics.prefill_tokens_per_second(),
        metrics.decode_tokens_per_second(),
        metrics.host_wall_time.as_secs_f64() * 1000.0,
        metrics.gpu_execution_time.as_secs_f64() * 1000.0,
        metrics.command_buffer_count,
        metrics.prefill_command_buffer_count,
        metrics.decode_command_buffer_count,
        metrics.weight_upload_bytes,
        metrics.readback_bytes,
        metrics.resident_arena_allocations,
        metrics.post_warmup_allocations,
    )
}

fn session_metrics_line(metrics: &ChatSessionMetrics) -> String {
    format!(
        "session_turns={} session_generated_tokens={} session_tok_s={:.2}",
        metrics.turns,
        metrics.generated_tokens,
        metrics.generated_tokens_per_second(),
    )
}

#[cfg(test)]
mod phase_07_tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn chat_arguments_parse_one_shot_and_reject_zero_max_tokens() {
        let (model, prompt, token_limit, mode, show_thoughts) = parse_chat_args(&[
            "--model".into(),
            "small".into(),
            "--prompt".into(),
            "hello".into(),
            "--max-tokens".into(),
            "7".into(),
        ])
        .unwrap();
        assert_eq!(model, ["--model", "small"]);
        assert_eq!(prompt.as_deref(), Some("hello"));
        assert_eq!(token_limit, ChatTokenLimit::Explicit(7));
        assert_eq!(mode, ExecutorMode::Resident);
        assert!(!show_thoughts);
        assert!(
            parse_chat_args(&[
                "--model".into(),
                "small".into(),
                "--max-tokens".into(),
                "0".into()
            ])
            .is_err()
        );
    }

    #[test]
    fn chat_defaults_to_remaining_context_capacity() {
        let (_, _, token_limit, _, _) =
            parse_chat_args(&["--model".into(), "small".into()]).unwrap();
        assert_eq!(token_limit, ChatTokenLimit::Context);
        assert_eq!(
            token_limit.resolve(17, 4096).unwrap(),
            ResolvedTokenLimit {
                max_new_tokens: 4079,
                source: "context",
            }
        );
    }

    #[test]
    fn explicit_chat_limit_must_fit_remaining_context() {
        assert_eq!(
            ChatTokenLimit::Explicit(32).resolve(64, 1024).unwrap(),
            ResolvedTokenLimit {
                max_new_tokens: 32,
                source: "explicit",
            }
        );
        assert!(ChatTokenLimit::Explicit(33).resolve(992, 1024).is_err());
        assert!(ChatTokenLimit::Context.resolve(1024, 1024).is_err());
    }

    #[test]
    fn chat_accepts_explicit_resident_executor_only() {
        let (_, _, _, mode, _) = parse_chat_args(&[
            "--model".into(),
            "small".into(),
            "--executor".into(),
            "resident".into(),
        ])
        .unwrap();
        assert_eq!(mode, ExecutorMode::Resident);
        assert!(
            parse_chat_args(&[
                "--model".into(),
                "small".into(),
                "--executor".into(),
                "automatic".into(),
            ])
            .is_err()
        );
    }

    #[test]
    fn thought_filter_handles_markers_split_across_fragments() {
        let mut filter = ThoughtFilter::default();
        let mut streamed = String::new();
        for fragment in [
            "Visible <|chan",
            "nel>thoughtsecret",
            " text<chan",
            "nel|> answer",
        ] {
            streamed.push_str(&filter.push(fragment));
        }
        let (visible, thoughts, tail) = filter.finish();
        streamed.push_str(&tail);
        assert_eq!(visible, "Visible  answer");
        assert_eq!(streamed, visible);
        assert_eq!(thoughts, "secret text");
    }

    #[test]
    fn gemma_protocol_tokens_are_neither_streamed_nor_saved_to_history() {
        let mut filter = ThoughtFilter::default();
        let mut streamed = String::new();
        for fragment in ["I noted **zephyr**.<tur", "n|><turn|><|tu", "rn><e", "os>"] {
            streamed.push_str(&filter.push(fragment));
        }
        let (visible, thoughts, tail) = filter.finish();
        streamed.push_str(&tail);
        assert_eq!(visible, "I noted **zephyr**.");
        assert_eq!(streamed, visible);
        assert!(thoughts.is_empty());
        assert!(
            render_gemma4_chat(&[
                Gemma4ChatMessage::new(Gemma4ChatRole::User, "What is the code word?"),
                Gemma4ChatMessage::new(Gemma4ChatRole::Model, visible),
                Gemma4ChatMessage::new(Gemma4ChatRole::User, "Repeat it"),
            ])
            .is_ok()
        );
    }

    #[test]
    fn show_thoughts_flag_is_parsed_for_family_validation() {
        let (_, _, _, _, show) = parse_chat_args(&[
            "--model".into(),
            "gemma4-e2b-q4_0".into(),
            "--show-thoughts".into(),
        ])
        .unwrap();
        assert!(show);
    }

    #[test]
    fn deterministic_compaction_replaces_oldest_pairs_atomically_and_keeps_newest() {
        let mut messages = vec![
            Gemma4ChatMessage::new(Gemma4ChatRole::User, "old user"),
            Gemma4ChatMessage::new(Gemma4ChatRole::Model, "old answer"),
            Gemma4ChatMessage::new(Gemma4ChatRole::User, "middle user"),
            Gemma4ChatMessage::new(Gemma4ChatRole::Model, "middle answer"),
            Gemma4ChatMessage::new(Gemma4ChatRole::User, "newest user"),
        ];
        replace_oldest_gemma4_pair_with_summary(&mut messages, "summary one").unwrap();
        assert_eq!(
            messages[0],
            Gemma4ChatMessage::new(Gemma4ChatRole::System, "summary one")
        );
        assert_eq!(messages.last().unwrap().content, "newest user");
        replace_oldest_gemma4_pair_with_summary(&mut messages, "summary two").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "summary two");
        assert_eq!(messages[1].content, "newest user");
        assert!(replace_oldest_gemma4_pair_with_summary(&mut messages, "impossible").is_err());
    }

    #[test]
    fn removed_diagnostic_commands_and_profile_flag_are_not_chat_options() {
        assert!(parse_chat_args(&["benchmark".into()]).is_err());
        assert!(parse_chat_args(&["diagnose".into()]).is_err());
        assert!(
            parse_chat_args(&[
                "--model".into(),
                "small".into(),
                "--profile-resident-decode".into(),
            ])
            .is_err()
        );
    }

    #[test]
    fn jsonl_append_creates_parent_and_preserves_records() {
        let directory = std::env::temp_dir().join(format!(
            "atlas-chat-performance-test-{}",
            std::process::id()
        ));
        let path = directory.join("nested/chat-performance.jsonl");
        append_jsonl_record(&path, &json!({"record": 1})).unwrap();
        append_jsonl_record(&path, &json!({"record": 2})).unwrap();
        let records = fs::read_to_string(&path).unwrap();
        let values = records
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values, vec![json!({"record": 1}), json!({"record": 2})]);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn manifest_file_paths_cannot_escape_the_model_root() {
        assert!(safe_model_file(Path::new("models/hf/small"), Path::new("config.json")).is_ok());
        assert!(
            safe_model_file(Path::new("models/hf/small"), Path::new("../config.json")).is_err()
        );
        assert!(
            safe_model_file(Path::new("models/hf/small"), Path::new("/tmp/config.json")).is_err()
        );
        assert!(safe_project_path(Path::new("models/hf/small")).is_ok());
        assert!(safe_project_path(Path::new("../models/hf/small")).is_err());
    }

    #[test]
    fn downloaded_manifest_id_cannot_escape_the_model_root() {
        assert!(valid_manifest_id("small-q4.1"));
        assert!(!valid_manifest_id(""));
        assert!(!valid_manifest_id("../escape"));
        assert!(!valid_manifest_id("line\nbreak"));
    }

    #[test]
    fn gemma4_manifest_selects_the_embedded_gguf_dispatch_branch() {
        let manifest_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/manifest.toml");
        let manifest = load_manifest_from(&manifest_path).unwrap();
        let gemma = manifest
            .models
            .iter()
            .find(|record| record.id == "gemma4-e2b-q4_0")
            .expect("Gemma 4 E2B manifest entry");

        assert_eq!(gemma.manifest_kind().unwrap(), ManifestModelKind::Gemma4E2b);
        assert_eq!(gemma.architecture, "gemma4");
        assert_eq!(gemma.tokenizer, PathBuf::from("embedded"));
        assert_eq!(
            gemma.model_file.as_deref(),
            Some(Path::new("gemma-4-E2B_q4_0-it.gguf"))
        );
        assert_eq!(gemma.files.len(), 1);
        assert_eq!(gemma.files[0].bytes, gemma.bytes);
        assert_eq!(
            gemma.files[0].sha256,
            "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634"
        );
    }

    #[test]
    fn gemma4_without_resident_executor_has_no_fallback() {
        let error = gemma4_resident_executor_unavailable_error();
        assert_eq!(
            error.to_string(),
            "Gemma 4 resident executor is not implemented; Atlas will not fall back to Llama or Reference"
        );
    }

    #[test]
    #[ignore = "requires the ignored 3.3 GiB Gemma 4 E2B GGUF fixture"]
    fn gemma4_manifest_fixture_verifies_architecture_and_checksum() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let manifest = load_manifest_from(&repository.join("models/manifest.toml")).unwrap();
        let mut gemma = manifest
            .models
            .into_iter()
            .find(|record| record.id == "gemma4-e2b-q4_0")
            .expect("Gemma 4 E2B manifest entry");
        gemma.path = repository.join(gemma.path);
        verify_manifest_model(&gemma).unwrap();
    }

    #[test]
    fn model_search_sizes_are_human_readable() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1_048_576), "1.0 MiB");
    }

    #[test]
    fn repl_commands_preserve_or_clear_history_as_required() {
        let mut history = String::new();
        assert_eq!(repl_command(""), ReplCommand::Ignore);
        assert_eq!(repl_command("/help"), ReplCommand::Help);
        append_user_turn(&mut history, "hello");
        let one_shot_prompt = history.clone();
        assert_eq!(one_shot_prompt, "user: hello\nassistant: ");
        assert_eq!(repl_command("/reset"), ReplCommand::Reset);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(repl_command("/quit"), ReplCommand::Quit);
    }

    #[test]
    fn phase_07_metrics_include_all_reported_rates() {
        let metrics = ExecutorMetrics {
            ttft: Duration::from_millis(12),
            prefill: Duration::from_millis(10),
            prefill_tokens: 5,
            decode: Duration::from_millis(20),
            decode_tokens: 4,
            ..Default::default()
        };
        let line = metrics_line(&metrics);
        assert!(line.contains("ttft_ms=12.00"));
        assert!(line.contains("prefill_tok_s=500.00"));
        assert!(line.contains("decode_tok_s=200.00"));
    }

    #[test]
    fn gemma_metrics_report_prefill_and_post_prefill_decode_rates() {
        let generation = Gemma4Generation {
            generation: atlas_model::Generation {
                prompt_token_ids: vec![1, 2, 3, 4],
                generated_token_ids: vec![5, 6, 7],
                text: String::new(),
                trace: atlas_model::LayerTrace::default(),
                final_logits: Vec::new(),
            },
            metrics: atlas_model::gemma4_executor::Gemma4Metrics {
                resident_bytes: 0,
                weight_upload_bytes: 0,
                readback_bytes: 0,
                command_buffers: 0,
                prefill_command_buffers: 4,
                decode_command_buffers: 2,
                prefill: Duration::from_millis(20),
                decode: Duration::from_millis(10),
                host_wall_time: Duration::from_millis(30),
                prefill_path: "resident_chunked_command",
                prefill_chunk_size: 4,
            },
            finish_reason: Gemma4FinishReason::Eos,
        };
        assert_eq!(gemma4_prefill_tokens_per_second(&generation), 200.0);
        assert_eq!(gemma4_decode_tokens_per_second(&generation), 200.0);
    }

    #[test]
    fn chat_exit_reports_visible_tokens_from_completed_and_interrupted_turns() {
        let mut session = ChatSessionMetrics::default();
        let mut completed = ChatTurnMetrics::new();
        completed.record_event(&GenerationEvent::Token {
            token_id: 1,
            text: "one".into(),
            decode_latency: None,
        });
        session.record(&completed);

        let mut interrupted = ChatTurnMetrics::new();
        for token_id in 2..=4 {
            interrupted.record_event(&GenerationEvent::Token {
                token_id,
                text: "partial".into(),
                decode_latency: None,
            });
        }
        // The caller records this before breaking on the terminal cancellation
        // error, preserving text that was already delivered to stdout.
        session.record(&interrupted);

        assert_eq!(session.turns, 2);
        assert_eq!(session.generated_tokens, 4);
        assert!(session.generated_tokens_per_second() > 0.0);
        assert!(session_metrics_line(&session).contains("session_tok_s="));
    }

    #[test]
    fn a_failed_stream_after_tokens_still_has_countable_turn_metrics() {
        let mut turn = ChatTurnMetrics::new();
        turn.record_event(&GenerationEvent::Token {
            token_id: 9,
            text: "visible".into(),
            decode_latency: None,
        });
        turn.record_event(&GenerationEvent::Failed {
            message: "generation cancelled".into(),
        });
        assert_eq!(turn.generated_tokens, 1);
    }

    #[test]
    fn streaming_writer_emits_and_flushes_token_fragments_only() {
        let mut output = Vec::new();
        write_stream_event(
            &mut output,
            &GenerationEvent::Token {
                token_id: 7,
                text: "hello".into(),
                decode_latency: None,
            },
        )
        .unwrap();
        write_stream_event(
            &mut output,
            &GenerationEvent::Finished {
                reason: atlas_model::executor::GenerationFinishReason::MaxTokens,
                metrics: ExecutorMetrics::default(),
            },
        )
        .unwrap();
        assert_eq!(output, b"hello");
    }
}

fn phase_08b_decode(args: &[String]) -> Result<()> {
    let mut model_args = Vec::new();
    let mut prompt = None;
    let mut warmup = 1usize;
    let mut max_new_tokens = 16usize;
    let mut trace_logits = false;
    let mut trace_stages = false;
    let mut trace_tolerance = 1e-5f32;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" | "--model-dir" => {
                model_args.push(args[index].clone());
                index += 1;
                model_args.push(
                    args.get(index)
                        .context("model option needs a value")?
                        .clone(),
                );
            }
            "--prompt" => {
                index += 1;
                prompt = Some(args.get(index).context("--prompt needs a value")?.clone());
            }
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .context("--warmup needs a value")?
                    .parse()
                    .context("parse --warmup")?;
            }
            "--max-new-tokens" => {
                index += 1;
                max_new_tokens = args
                    .get(index)
                    .context("--max-new-tokens needs a value")?
                    .parse()
                    .context("parse --max-new-tokens")?;
                ensure!(max_new_tokens > 0, "--max-new-tokens must be positive");
            }
            "--trace-logits" => trace_logits = true,
            "--trace-stages" => trace_stages = true,
            "--trace-tolerance" => {
                index += 1;
                trace_tolerance = args
                    .get(index)
                    .context("--trace-tolerance needs a value")?
                    .parse()
                    .context("parse --trace-tolerance")?;
                ensure!(
                    trace_tolerance.is_finite() && trace_tolerance >= 0.0,
                    "--trace-tolerance must be finite and non-negative"
                );
            }
            flag => bail!("unknown phase_08b_decode option: {flag}"),
        }
        index += 1;
    }
    let prompt = prompt.context("phase_08b_decode requires --prompt")?;
    let logits_readback = if trace_logits {
        LogitsReadback::FinalLogits
    } else {
        LogitsReadback::SelectedToken
    };
    let selection = resolve_model(&model_args)?;
    let model_name = selection.id.clone();
    let model_bytes = selection.manifest.as_ref().map_or(0, |record| record.bytes);
    let model = match load_verified_model(&selection)? {
        LoadedModel::Llama(model) => model,
        LoadedModel::Gemma4E2b(model) => return Err(gemma4_resident_executor_unavailable(&model)),
    };
    if trace_stages {
        ensure!(
            model.format_name() != "gguf-packed",
            "stage tracing requires an FP32 reference model; use `atlas-cli diagnose --model {model_name}` for GGUF quality diagnostics"
        );
        match AtlasExecutor::trace_resident_prompt(&model, &prompt, trace_tolerance)? {
            Some(result) => {
                println!(
                    "first_divergence prompt_token={} layer={} stage={} elements={} max_abs_error={:.8} first_index={} expected={:.8} actual={:.8}",
                    result.prompt_token_index,
                    result
                        .layer
                        .map_or_else(|| "final".to_owned(), |layer| layer.to_string()),
                    result.stage,
                    result.element_count,
                    result.max_abs_error,
                    result.first_failing_index.unwrap_or(0),
                    result.expected,
                    result.actual,
                );
                std::process::exit(1);
            }
            None => {
                println!("stage_trace: no divergence");
                return Ok(());
            }
        }
    }

    if model.format_name() == "gguf-packed" {
        for _ in 0..warmup {
            let mut executor = resident_executor(&model, &selection, logits_readback)?;
            executor.generate_greedy(&prompt, max_new_tokens)?;
        }
        let mut executor = resident_executor(&model, &selection, logits_readback)?;
        let resident = executor.generate_greedy(&prompt, max_new_tokens)?;
        println!("model: {model_name}");
        println!("format: {}", model.format_name());
        println!("model_bytes: {model_bytes}");
        println!("reference_decode_tok_s: unavailable (GGUF is resident-only)");
        println!(
            "resident_decode_tok_s: {:.2}",
            resident.metrics.decode_tokens_per_second()
        );
        println!("{}", metrics_line(&resident.metrics));
        return Ok(());
    }

    // Warm each implementation separately.  The measured runs begin only
    // after pipeline creation and resident weight materialization.
    for _ in 0..warmup {
        let mut reference = AtlasExecutor::new(
            &model,
            ExecutorConfig {
                mode: ExecutorMode::Reference,
                logits_readback,
                ..Default::default()
            },
        )?;
        let _ = reference.generate_greedy(&prompt, max_new_tokens)?;
        let mut executor = AtlasExecutor::new(
            &model,
            ExecutorConfig {
                mode: ExecutorMode::Resident,
                logits_readback,
                ..Default::default()
            },
        )?;
        let _ = executor.generate_greedy(&prompt, max_new_tokens)?;
    }

    let reference_start = Instant::now();
    let mut reference_executor = AtlasExecutor::new(
        &model,
        ExecutorConfig {
            mode: ExecutorMode::Reference,
            logits_readback,
            ..Default::default()
        },
    )?;
    let reference = reference_executor.generate_greedy(&prompt, max_new_tokens)?;
    let reference_elapsed = reference_start.elapsed();
    let mut executor = AtlasExecutor::new(
        &model,
        ExecutorConfig {
            mode: ExecutorMode::Resident,
            logits_readback,
            ..Default::default()
        },
    )?;
    let resident = executor.generate_greedy(&prompt, max_new_tokens)?;
    if trace_logits {
        let (index, delta) = reference
            .generation
            .final_logits
            .iter()
            .zip(&resident.generation.final_logits)
            .enumerate()
            .map(|(index, (reference, resident))| (index, (reference - resident).abs()))
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .context("trace logits are unexpectedly empty")?;
        println!("max_logit_abs_delta: {delta:.6} at_token_id={index}");
    }
    ensure!(
        resident.generation.generated_token_ids == reference.generation.generated_token_ids,
        "resident decode token IDs differ from reference: reference={:?} resident={:?}",
        reference.generation.generated_token_ids,
        resident.generation.generated_token_ids
    );
    let tokens = resident.generation.generated_token_ids.len();
    let reference_rate = if reference_elapsed.is_zero() {
        0.0
    } else {
        tokens as f64 / reference_elapsed.as_secs_f64()
    };
    let resident_rate = resident.metrics.decode_tokens_per_second();
    println!("model: {model_name}");
    println!("format: {}", model.format_name());
    println!("model_bytes: {model_bytes}");
    println!("token_agreement: true");
    println!("reference_decode_tok_s: {reference_rate:.2}");
    println!("resident_decode_tok_s: {resident_rate:.2}");
    println!("{}", metrics_line(&resident.metrics));
    ensure!(
        resident_rate > reference_rate,
        "GPU-resident decode did not improve over the reference baseline ({resident_rate:.2} <= {reference_rate:.2} tok/s)"
    );
    Ok(())
}

fn phase_05_quant(args: &[String]) -> Result<()> {
    let mut model_args = Vec::new();
    let mut format = None;
    let mut tensor_name = "lm_head.weight".to_owned();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" | "--model-dir" => {
                model_args.push(args[index].clone());
                index += 1;
                model_args.push(
                    args.get(index)
                        .context("model option needs a value")?
                        .clone(),
                );
            }
            "--format" => {
                index += 1;
                format = Some(QuantFormat::parse(
                    args.get(index).context("--format needs a value")?,
                )?);
            }
            "--tensor" => {
                index += 1;
                tensor_name = args.get(index).context("--tensor needs a value")?.clone();
            }
            flag => bail!("unknown phase_05_quant option: {flag}"),
        }
        index += 1;
    }
    let format = format.context("--format is required")?;
    let (_, directory) = model_dir(&model_args)?;
    let path = directory.join("model.safetensors");
    ensure!(
        path.exists(),
        "phase_05_quant currently requires an unsharded model.safetensors fixture"
    );
    let descriptor = read_safetensors_descriptors(&path)?
        .into_iter()
        .find(|item| item.name == tensor_name)
        .with_context(|| format!("tensor `{tensor_name}` is missing from {}", path.display()))?;
    let dims = descriptor.tensor.shape.dims();
    ensure!(dims.len() == 2, "phase_05_quant tensor must be rank 2");
    let values = read_safetensors_tensor_f32(&path, &tensor_name)?;
    let packed = QuantizedMatrix::quantize(&values, dims[0], dims[1], format)?;
    let input: Vec<f32> = (0..dims[1])
        .map(|index| ((index as f32) * 0.013).sin())
        .collect();
    let baseline_start = Instant::now();
    let baseline = (0..64)
        .map(|_| {
            (0..dims[0])
                .map(|row| {
                    (0..dims[1])
                        .map(|column| input[column] * values[row * dims[1] + column])
                        .sum::<f32>()
                })
                .collect::<Vec<_>>()
        })
        .last()
        .unwrap();
    let baseline_time = baseline_start.elapsed();
    let quant_start = Instant::now();
    let quantized = (0..64)
        .map(|_| packed.matvec_cpu(&input))
        .collect::<Result<Vec<_>, _>>()?
        .pop()
        .unwrap();
    let quant_time = quant_start.elapsed();
    let max_delta = baseline
        .iter()
        .zip(&quantized)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max);
    let baseline_token = baseline
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index);
    let quantized_token = quantized
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index);
    println!("format: {}", format.name());
    println!("tensor: {tensor_name} shape={:?}", dims);
    println!("source_bytes: {}", values.len() * 4);
    println!("packed_bytes: {}", packed.resident_bytes());
    println!("max_logit_delta: {:.8}", max_delta);
    println!("token_agreement: {}", baseline_token == quantized_token);
    println!("baseline_tok_s: {:.2}", 64.0 / baseline_time.as_secs_f64());
    println!("quantized_tok_s: {:.2}", 64.0 / quant_time.as_secs_f64());
    ensure!(
        baseline_token == quantized_token,
        "quantized greedy token differs from FP16 baseline"
    );
    ensure!(
        max_delta
            <= if format == QuantFormat::Q4Block32 {
                0.20
            } else {
                0.02
            },
        "logit delta exceeds Phase 5 threshold: {max_delta}"
    );
    Ok(())
}

fn load_manifest() -> Result<ModelManifest> {
    load_manifest_from(Path::new(MODEL_MANIFEST))
}

fn load_manifest_from(path: &Path) -> Result<ModelManifest> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut models: Vec<ModelRecord> = Vec::new();
    let mut model: Option<ModelRecord> = None;
    let mut file: Option<ModelFile> = None;
    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[models]]" {
            if let Some(file) = file.take() {
                model
                    .as_mut()
                    .context("model file appears before a model")?
                    .files
                    .push(file);
            }
            if let Some(model) = model.take() {
                models.push(model);
            }
            model = Some(ModelRecord {
                id: String::new(),
                source: String::new(),
                revision: String::new(),
                path: PathBuf::new(),
                architecture: String::new(),
                tokenizer: PathBuf::new(),
                model_file: None,
                embedded_tokenizer: false,
                format: String::new(),
                bytes: 0,
                baseline_model: None,
                max_logit_abs_delta: None,
                min_token_agreement: None,
                max_resident_bytes: None,
                files: Vec::new(),
            });
            continue;
        }
        if line == "[[models.files]]" {
            if let Some(file) = file.take() {
                model
                    .as_mut()
                    .context("model file appears before a model")?
                    .files
                    .push(file);
            }
            file = Some(ModelFile {
                path: PathBuf::new(),
                bytes: 0,
                sha256: String::new(),
            });
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .context("manifest entry must use key = value")?;
        let key = key.trim();
        let value = value.trim();
        let text = value
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'));
        if let Some(file) = file.as_mut() {
            match key {
                "path" => {
                    file.path = PathBuf::from(text.context("manifest file path must be quoted")?)
                }
                "bytes" => file.bytes = value.parse().context("parse manifest file bytes")?,
                "sha256" => {
                    file.sha256 = text.context("manifest SHA-256 must be quoted")?.to_owned()
                }
                _ => bail!("unknown model file manifest key `{key}`"),
            }
        } else {
            let model = model
                .as_mut()
                .context("manifest entry appears before [[models]]")?;
            match key {
                "id" => model.id = text.context("model id must be quoted")?.to_owned(),
                "source" => model.source = text.context("model source must be quoted")?.to_owned(),
                "revision" => {
                    model.revision = text.context("model revision must be quoted")?.to_owned()
                }
                "path" => model.path = PathBuf::from(text.context("model path must be quoted")?),
                "architecture" => {
                    model.architecture = text
                        .context("model architecture must be quoted")?
                        .to_owned()
                }
                "tokenizer" => {
                    model.tokenizer = PathBuf::from(text.context("model tokenizer must be quoted")?)
                }
                "model_file" => {
                    model.model_file =
                        Some(PathBuf::from(text.context("model_file must be quoted")?))
                }
                "embedded_tokenizer" => {
                    model.embedded_tokenizer =
                        value.parse::<bool>().context("parse embedded_tokenizer")?
                }
                "format" => model.format = text.context("model format must be quoted")?.to_owned(),
                "bytes" => model.bytes = value.parse().context("parse model bytes")?,
                "baseline_model" => {
                    model.baseline_model = Some(
                        text.context("model baseline_model must be quoted")?
                            .to_owned(),
                    )
                }
                "max_logit_abs_delta" => {
                    model.max_logit_abs_delta =
                        Some(value.parse().context("parse max_logit_abs_delta")?)
                }
                "min_token_agreement" => {
                    model.min_token_agreement =
                        Some(value.parse().context("parse min_token_agreement")?)
                }
                "max_resident_bytes" => {
                    model.max_resident_bytes =
                        Some(value.parse().context("parse max_resident_bytes")?)
                }
                _ => bail!("unknown model manifest key `{key}`"),
            }
        }
    }
    if let Some(file) = file {
        model
            .as_mut()
            .context("model file appears before a model")?
            .files
            .push(file);
    }
    if let Some(model) = model {
        models.push(model);
    }
    for model in &models {
        ensure!(
            !model.id.is_empty()
                && !model.source.is_empty()
                && !model.revision.is_empty()
                && !model.path.as_os_str().is_empty()
                && !model.architecture.is_empty()
                && !model.tokenizer.as_os_str().is_empty()
                && !model.format.is_empty()
                && !model.files.is_empty(),
            "manifest contains an incomplete model record"
        );
        if matches!(model.format.as_str(), "gguf-q4_0" | "gguf-q8_0") {
            ensure!(
                model.baseline_model.is_some() == model.max_logit_abs_delta.is_some()
                    && model.baseline_model.is_some() == model.min_token_agreement.is_some()
                    && model.baseline_model.is_some() == model.max_resident_bytes.is_some(),
                "quantized manifest model `{}` must set all acceptance policy fields together",
                model.id
            );
            if model.baseline_model.is_some() {
                ensure!(
                    model
                        .max_logit_abs_delta
                        .is_some_and(|value| value.is_finite() && value >= 0.0),
                    "quantized manifest model `{}` has invalid max_logit_abs_delta",
                    model.id
                );
                ensure!(
                    model
                        .min_token_agreement
                        .is_some_and(|value| (0.0..=1.0).contains(&value)),
                    "quantized manifest model `{}` has invalid min_token_agreement",
                    model.id
                );
                ensure!(
                    model.max_resident_bytes.is_some_and(|value| value > 0),
                    "quantized manifest model `{}` has invalid max_resident_bytes",
                    model.id
                );
            }
        }
    }
    let mut ids = BTreeSet::new();
    for model in &models {
        ensure!(
            ids.insert(model.id.as_str()),
            "duplicate model ID `{}`",
            model.id
        );
    }
    for model in &models {
        if let Some(baseline) = &model.baseline_model {
            let baseline = models
                .iter()
                .find(|candidate| candidate.id == *baseline)
                .with_context(|| {
                    format!("baseline model `{baseline}` is not in {MODEL_MANIFEST}")
                })?;
            ensure!(
                baseline.format == "safetensors-fp32",
                "baseline model `{}` must use safetensors-fp32",
                baseline.id
            );
            ensure!(
                baseline.architecture == model.architecture,
                "baseline model `{}` architecture differs from `{}`",
                baseline.id,
                model.id
            );
        }
    }
    let manifest = ModelManifest { models };
    ensure!(!manifest.models.is_empty(), "model manifest has no models");
    Ok(manifest)
}

fn resolve_model(args: &[String]) -> Result<ModelSelection> {
    let mut model = None;
    let mut directory = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                index += 1;
                model = args.get(index).cloned();
            }
            "--model-dir" => {
                index += 1;
                directory = args.get(index).map(PathBuf::from);
            }
            flag => bail!("unknown model option: {flag}"),
        }
        index += 1;
    }
    let model = model.context("--model is required")?;
    if let Some(directory) = directory {
        if !directory.join("config.json").exists() {
            bail!(
                "developer model fixture is missing at {}; pass a directory with config.json",
                directory.display()
            );
        }
        return Ok(ModelSelection {
            id: model,
            directory,
            manifest: None,
        });
    }
    let manifest = load_manifest()?;
    let mut record = manifest
        .models
        .into_iter()
        .find(|record| record.id == model)
        .with_context(|| format!("model ID `{model}` is not in {MODEL_MANIFEST}"))?;
    record.path = safe_project_path(&record.path)?;
    match record.manifest_kind()? {
        ManifestModelKind::Gemma4E2b => {
            let model_file = safe_model_file(
                &record.path,
                record
                    .model_file
                    .as_ref()
                    .expect("Gemma manifest has model_file"),
            )?;
            ensure!(
                model_file.is_file(),
                "manifest model `{model}` is missing embedded GGUF at {}; download its pinned revision",
                model_file.display()
            );
        }
        ManifestModelKind::Llama if !record.path.join("config.json").exists() => {
            bail!(
                "manifest model `{model}` is missing at {}; download its pinned revision or pass --model-dir for a developer fixture",
                record.path.display()
            );
        }
        ManifestModelKind::Llama => {}
    }
    Ok(ModelSelection {
        id: model,
        directory: record.path.clone(),
        manifest: Some(record),
    })
}

fn model_dir(args: &[String]) -> Result<(String, PathBuf)> {
    let selection = resolve_model(args)?;
    Ok((selection.id, selection.directory))
}

fn safe_model_file(root: &Path, relative: &Path) -> Result<PathBuf> {
    ensure!(
        !relative.is_absolute()
            && !relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
        "manifest file path must be relative and may not escape its model directory: {}",
        relative.display()
    );
    Ok(root.join(relative))
}

fn safe_project_path(relative: &Path) -> Result<PathBuf> {
    ensure!(
        !relative.is_absolute()
            && !relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir)),
        "manifest model path must be relative and may not escape the repository: {}",
        relative.display()
    );
    Ok(relative.to_path_buf())
}

fn sha256_file(path: &Path) -> Result<String> {
    let output = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .with_context(|| format!("run shasum for {}", path.display()))?;
    ensure!(
        output.status.success(),
        "shasum failed for {}",
        path.display()
    );
    let digest = std::str::from_utf8(&output.stdout)
        .context("shasum did not emit UTF-8")?
        .split_whitespace()
        .next()
        .context("shasum produced no digest")?;
    ensure!(
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid SHA-256 output"
    );
    Ok(digest.to_owned())
}

fn verify_manifest_model(record: &ModelRecord) -> Result<()> {
    let mut total = 0u64;
    for file in &record.files {
        let path = safe_model_file(&record.path, &file.path)?;
        let metadata = fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
        ensure!(
            metadata.len() == file.bytes,
            "byte size mismatch for {}",
            path.display()
        );
        ensure!(
            sha256_file(&path)? == file.sha256,
            "SHA-256 mismatch for {}",
            path.display()
        );
        total = total
            .checked_add(metadata.len())
            .context("model byte total overflow")?;
    }
    ensure!(
        total == record.bytes,
        "manifest byte total mismatch for `{}`",
        record.id
    );
    if record.manifest_kind()? == ManifestModelKind::Gemma4E2b {
        let model_file = record
            .model_file
            .as_ref()
            .expect("Gemma manifest has model_file");
        ensure!(
            record.files.iter().any(|file| file.path == *model_file),
            "embedded GGUF model file is not recorded in manifest files for `{}`",
            record.id
        );
        let gguf = GgufModel::open(safe_model_file(&record.path, model_file)?)?;
        ensure!(
            gguf.metadata
                .get("general.architecture")
                .map(String::as_str)
                == Some(record.architecture.as_str()),
            "embedded GGUF architecture mismatch for `{}`",
            record.id
        );
        return Ok(());
    }
    let tokenizer = safe_model_file(&record.path, &record.tokenizer)?;
    ensure!(
        tokenizer.is_file(),
        "tokenizer is missing: {}",
        tokenizer.display()
    );
    let config: Value = serde_json::from_slice(&fs::read(record.path.join("config.json"))?)?;
    let architecture = config["architectures"]
        .as_array()
        .and_then(|items| items.first())
        .and_then(Value::as_str);
    ensure!(
        architecture == Some(record.architecture.as_str()),
        "architecture mismatch for `{}`",
        record.id
    );
    if matches!(record.format.as_str(), "gguf-q4_0" | "gguf-q8_0") {
        let gguf = GgufModel::open(record.path.join("model.gguf"))?;
        ensure!(
            gguf.metadata
                .get("general.architecture")
                .map(String::as_str)
                == Some("llama"),
            "GGUF architecture mismatch for `{}`",
            record.id
        );
        let expected = if record.format == "gguf-q4_0" {
            GgufTensorType::Q4_0
        } else {
            GgufTensorType::Q8_0
        };
        ensure!(
            gguf.tensors
                .iter()
                .any(|tensor| tensor.tensor_type == expected),
            "GGUF format does not contain expected packed tensors"
        );
        Ok(())
    } else {
        fixture_details(&record.path).map(|_| ())
    }
}

fn model_command(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .context("model command requires a subcommand")?;
    match command.as_str() {
        "quantize" => return model_quantize(&args[1..]),
        "import-gguf" => return model_import_gguf(&args[1..]),
        "search" => return model_search(&args[1..]),
        "download" => return model_download(&args[1..]),
        _ => {}
    }
    let selection = resolve_model(&args[1..])?;
    let record = selection
        .manifest
        .context("`atlas-cli model` requires a manifest-backed model ID")?;
    match command.as_str() {
        "inspect" => println!(
            "{}",
            json!({
                "model_id": record.id, "source": record.source, "revision": record.revision,
                "path": record.path, "architecture": record.architecture, "format": record.format,
                "bytes": record.bytes, "baseline_model": record.baseline_model,
                "max_logit_abs_delta": record.max_logit_abs_delta,
                "min_token_agreement": record.min_token_agreement,
                "max_resident_bytes": record.max_resident_bytes,
            })
        ),
        "verify" => {
            verify_manifest_model(&record)?;
            println!(
                "{}",
                json!({"model_id": record.id, "verified": true, "format": record.format, "bytes": record.bytes})
            );
        }
        _ => bail!(
            "model command must be `search`, `download`, `inspect`, `verify`, `quantize`, or `import-gguf`"
        ),
    }
    Ok(())
}

fn model_search(args: &[String]) -> Result<()> {
    let mut requested = None;
    let mut query = None;
    let mut json_output = false;
    let mut no_ui = false;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--provider" {
            index += 1;
            requested = Some(
                args.get(index)
                    .context("--provider needs a value")?
                    .as_str(),
            );
        } else if args[index] == "--json" {
            json_output = true;
        } else if args[index] == "--no-ui" {
            no_ui = true;
        } else if query.is_none() {
            query = Some(args[index].as_str());
        } else {
            bail!("model search accepts one query");
        }
        index += 1;
    }
    let selection = providers::selected(requested)?;
    let provider = providers::provider(selection.id())?;
    let query = query.unwrap_or("");
    let first_page = if query.is_empty() {
        providers::SearchPage {
            candidates: Vec::new(),
            next_cursor: None,
        }
    } else {
        provider.search(query, None)?
    };
    let candidates = first_page.candidates;
    if json_output {
        for candidate in candidates {
            println!("{}", candidate.json());
        }
    } else if !no_ui && io::stdin().is_terminal() && io::stdout().is_terminal() {
        model_browser(
            query,
            candidates,
            load_manifest().map(|m| m.models).unwrap_or_default(),
            first_page.next_cursor,
        )?;
    } else if candidates.is_empty() {
        println!("Enter a query, for example: atlas-cli model search SmolLM2");
    } else {
        println!(
            "Found {} Atlas-compatible model{} from {}:\n",
            candidates.len(),
            if candidates.len() == 1 { "" } else { "s" },
            selection.id()
        );
        for (index, candidate) in candidates.iter().enumerate() {
            println!("{}. {}", index + 1, candidate.repository);
            println!("   Format: {}", candidate.format);
            println!("   Size: {}", human_bytes(candidate.bytes));
            println!("   Revision: {}", candidate.revision);
            println!(
                "   Access: {}",
                if candidate.requires_auth {
                    "login required"
                } else {
                    "public"
                }
            );
            println!(
                "   Download: atlas-cli model download '{}' --id <name>",
                candidate.id()
            );
            println!();
        }
    }
    Ok(())
}

fn model_browser(
    query: &str,
    mut candidates: Vec<providers::ModelCandidate>,
    models: Vec<ModelRecord>,
    mut next_cursor: Option<String>,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut focus = if query.is_empty() { 0usize } else { 1usize };
    let mut selected = 0usize;
    let mut local = 0usize;
    let mut input = query.to_owned();
    let mut status = String::new();
    let mut offset = 0usize;
    let mut cursors = vec![None];
    let result = loop {
        terminal.draw(|frame| {
            let outer = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(3), Constraint::Min(1)])
                .split(frame.area());
            let input_style = if focus == 0 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };
            frame.render_widget(
                Paragraph::new(format!(
                    "Search: {input}  page {}  [/ edit] [←/→ pages] [Tab switch tables] [q quit]  {status}", offset / providers::SEARCH_PAGE_SIZE + 1
                ))
                .style(input_style)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Atlas Model Explorer"),
                ),
                outer[0],
            );
            let rows = candidates.iter().map(|c| {
                Row::new(vec![
                    Cell::from(c.repository.clone()),
                    Cell::from(c.format.clone()),
                    Cell::from(human_bytes(c.bytes)),
                    Cell::from(c.reason.clone().unwrap_or_else(|| if c.requires_auth { "login".into() } else { "public".into() })),
                ]).style(if c.downloadable { Style::default() } else { Style::default().fg(Color::DarkGray) })
            });
            let mut state = TableState::default();
            state.select(Some(selected.min(candidates.len().saturating_sub(1))));
            let style = if focus == 1 {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            };
            let table = Table::new(
                rows,
                [
                    Constraint::Percentage(52),
                    Constraint::Length(20),
                    Constraint::Length(12),
                    Constraint::Length(10),
                ],
            )
            .header(
                Row::new(["Downloadable model", "Format", "Size", "Access"])
                    .style(Style::default().fg(Color::Green)),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Downloadable models"),
            )
            .row_highlight_style(style);
            if focus != 2 {
                frame.render_stateful_widget(table, outer[1], &mut state);
                return;
            }
            let rows = models.iter().map(|m| {
                Row::new(vec![
                    Cell::from(m.id.clone()),
                    Cell::from(m.format.clone()),
                    Cell::from(human_bytes(m.bytes)),
                    Cell::from(if m.path.exists() {
                        "present"
                    } else {
                        "missing"
                    }),
                ])
            });
            let mut state = TableState::default();
            state.select(Some(local.min(models.len().saturating_sub(1))));
            let style = if focus == 2 {
                Style::default().fg(Color::Cyan)
            } else {
                Style::default()
            };
            let table = Table::new(
                rows,
                [
                    Constraint::Percentage(52),
                    Constraint::Length(20),
                    Constraint::Length(12),
                    Constraint::Length(10),
                ],
            )
            .header(
                Row::new(["Manifest ID", "Format", "Size", "State"])
                    .style(Style::default().fg(Color::Green)),
            )
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Downloaded models"),
            )
            .row_highlight_style(style);
            frame.render_stateful_widget(table, outer[1], &mut state);
        })?;
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                KeyCode::Tab => focus = if focus == 2 { 1 } else { 2 },
                KeyCode::Char('/') => focus = 0,
                KeyCode::Up | KeyCode::Char('k') if focus == 1 => {
                    selected = selected.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') if focus == 1 => {
                    selected = (selected + 1).min(candidates.len().saturating_sub(1))
                }
                KeyCode::Up | KeyCode::Char('k') if focus == 2 => local = local.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') if focus == 2 => {
                    local = (local + 1).min(models.len().saturating_sub(1))
                }
                KeyCode::Backspace if focus == 0 => {
                    input.pop();
                }
                KeyCode::Char(ch) if focus == 0 => input.push(ch),
                KeyCode::Enter if focus == 0 => {
                    terminal.draw(|frame| {
                        frame.render_widget(
                            Paragraph::new(format!("Searching Hugging Face for `{input}`…"))
                                .style(Style::default().fg(Color::Yellow))
                                .block(
                                    Block::default()
                                        .borders(Borders::ALL)
                                        .title("Atlas Model Explorer"),
                                ),
                            frame.area(),
                        )
                    })?;
                    match providers::selected(None)
                        .and_then(|selection| providers::provider(selection.id()))
                        .and_then(|provider| provider.search(&input, None))
                    {
                        Ok(page) => {
                            candidates = page.candidates;
                            next_cursor = page.next_cursor;
                            cursors = vec![None];
                            selected = 0;
                            offset = 0;
                            focus = 1;
                            status = format!("{} result(s)", candidates.len());
                        }
                        Err(error) => {
                            status = format!("Search failed: {error:#}");
                        }
                    }
                }
                KeyCode::Right if focus == 1 && !input.is_empty() => {
                    let Some(cursor) = next_cursor.clone() else {
                        status = "No more results".into();
                        continue;
                    };
                    match providers::selected(None)
                        .and_then(|s| providers::provider(s.id()))
                        .and_then(|p| p.search(&input, Some(&cursor)))
                    {
                        Ok(page) if !page.candidates.is_empty() => {
                            candidates = page.candidates;
                            cursors.push(Some(cursor));
                            next_cursor = page.next_cursor;
                            offset += providers::SEARCH_PAGE_SIZE;
                            selected = 0;
                            status = format!("page {}", offset / providers::SEARCH_PAGE_SIZE + 1);
                        }
                        Ok(_) => status = "No more results".into(),
                        Err(error) => status = format!("Search failed: {error:#}"),
                    }
                }
                KeyCode::Left if focus == 1 && offset >= providers::SEARCH_PAGE_SIZE => {
                    let previous = offset - providers::SEARCH_PAGE_SIZE;
                    let previous_cursor = cursors[cursors.len() - 2].as_deref();
                    match providers::selected(None)
                        .and_then(|s| providers::provider(s.id()))
                        .and_then(|p| p.search(&input, previous_cursor))
                    {
                        Ok(page) => {
                            candidates = page.candidates;
                            next_cursor = cursors.pop().flatten();
                            offset = previous;
                            selected = 0;
                            status = format!("page {}", offset / providers::SEARCH_PAGE_SIZE + 1);
                        }
                        Err(error) => status = format!("Search failed: {error:#}"),
                    }
                }
                KeyCode::Enter
                    if focus == 1
                        && !candidates.is_empty()
                        && candidates[selected].downloadable =>
                {
                    break Ok(println!(
                        "atlas-cli model download '{}' --id <name>",
                        candidates[selected].id()
                    ));
                }
                KeyCode::Enter if focus == 1 && !candidates.is_empty() => {
                    status = candidates[selected]
                        .reason
                        .clone()
                        .unwrap_or_else(|| "This model cannot be downloaded by Atlas".into());
                }
                KeyCode::Enter if focus == 2 && !models.is_empty() => {
                    break Ok(println!(
                        "atlas-cli model verify --model {}",
                        models[local].id
                    ));
                }
                _ => {}
            }
        }
    };
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn valid_manifest_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn model_download(args: &[String]) -> Result<()> {
    let candidate = args
        .first()
        .context("model download requires a provider model ID")?;
    let mut id = None;
    let mut allow_auth = true;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--id" => {
                index += 1;
                id = args.get(index).cloned();
            }
            "--no-auth" => allow_auth = false,
            flag => bail!("unknown model download option: {flag}"),
        }
        index += 1;
    }
    let id = id.context("model download requires --id")?;
    ensure!(
        valid_manifest_id(&id),
        "--id may contain only letters, digits, '.', '_' and '-'"
    );
    ensure!(
        candidate.starts_with("huggingface:"),
        "unsupported provider model ID"
    );
    let manifest = load_manifest()?;
    let existing = manifest.models.into_iter().find(|model| model.id == id);
    let destination = existing.as_ref().map_or_else(
        || Path::new("models/hf").join(&id),
        |model| model.path.clone(),
    );
    ensure!(
        !destination.exists(),
        "model destination already exists: {}",
        destination.display()
    );
    let staging = destination
        .parent()
        .unwrap_or_else(|| Path::new("models/hf"))
        .join(format!(".{id}.staging-{}", std::process::id()));
    ensure!(
        !staging.exists(),
        "model staging directory already exists: {}",
        staging.display()
    );
    let result = (|| -> Result<()> {
        let downloaded = providers::download_hugging_face(candidate, &staging, allow_auth)?;
        if let Some(record) = &existing {
            ensure!(
                downloaded.repository == record.source && downloaded.revision == record.revision,
                "download candidate does not match pinned manifest source/revision for `{id}`"
            );
        }
        let gguf_file = downloaded
            .files
            .iter()
            .find(|file| file.ends_with(".gguf"))
            .cloned();
        let (format, kind, model_file) = if let Some(file) = gguf_file {
            let source_path = staging.join(&file);
            let gguf = GgufModel::open(&source_path)?;
            let architecture = gguf
                .metadata
                .get("general.architecture")
                .map(String::as_str);
            let has_q4 = gguf
                .tensors
                .iter()
                .any(|tensor| tensor.tensor_type == GgufTensorType::Q4_0);
            let has_q8 = gguf
                .tensors
                .iter()
                .any(|tensor| tensor.tensor_type == GgufTensorType::Q8_0);
            let has_q6 = gguf
                .tensors
                .iter()
                .any(|tensor| tensor.tensor_type == GgufTensorType::Q6K);
            let q4_only = gguf.tensors.iter().all(|tensor| {
                matches!(
                    tensor.tensor_type,
                    GgufTensorType::Q4_0 | GgufTensorType::F32
                )
            });
            let q8_only = gguf.tensors.iter().all(|tensor| {
                matches!(
                    tensor.tensor_type,
                    GgufTensorType::Q8_0 | GgufTensorType::F32
                )
            });
            if architecture == Some("gemma4") {
                ensure!(
                    has_q4
                        && has_q6
                        && gguf.tensors.iter().all(|tensor| matches!(
                            tensor.tensor_type,
                            GgufTensorType::Q4_0
                                | GgufTensorType::Q6K
                                | GgufTensorType::F16
                                | GgufTensorType::F32
                        )),
                    "Gemma 4 GGUF contains unsupported tensor encodings"
                );
                (
                    "gguf-gemma4-q4_0",
                    ManifestModelKind::Gemma4E2b,
                    Some(PathBuf::from(file)),
                )
            } else if architecture == Some("llama") && q4_only && has_q4 {
                fs::rename(source_path, staging.join("model.gguf"))?;
                ("gguf-q4_0", ManifestModelKind::Llama, None)
            } else if architecture == Some("llama") && q8_only && has_q8 {
                fs::rename(source_path, staging.join("model.gguf"))?;
                ("gguf-q8_0", ManifestModelKind::Llama, None)
            } else {
                bail!("GGUF architecture or tensor encodings are unsupported")
            }
        } else {
            fixture_details(&staging)?;
            for entry in fs::read_dir(&staging)? {
                let path = entry?.path();
                if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
                {
                    ensure!(
                        read_safetensors_descriptors(&path)?
                            .iter()
                            .all(|descriptor| matches!(
                                descriptor.tensor.dtype,
                                DType::F32 | DType::F16 | DType::BF16 | DType::I8
                            )),
                        "SafeTensors artifact contains an unsupported tensor dtype: {}",
                        path.display()
                    );
                }
            }
            ("safetensors-fp32", ManifestModelKind::Llama, None)
        };
        if let Some(record) = &existing {
            ensure!(
                record.format == format && record.manifest_kind()? == kind,
                "downloaded artifact does not match pinned manifest contract for `{id}`"
            );
            ensure!(
                kind != ManifestModelKind::Gemma4E2b || record.model_file == model_file,
                "downloaded Gemma artifact filename does not match pinned manifest contract for `{id}`"
            );
        }
        fs::rename(&staging, &destination)?;
        if existing.is_none() {
            if let Err(error) = register_download_manifest(
                &id,
                &downloaded.repository,
                &downloaded.revision,
                &destination,
                format,
                kind,
                model_file.as_deref(),
            ) {
                let _ = fs::remove_dir_all(&destination);
                return Err(error);
            }
        }
        let verification = load_manifest()?
            .models
            .into_iter()
            .find(|record| record.id == id)
            .context("registered model missing from manifest")
            .and_then(|record| verify_manifest_model(&record));
        if let Err(error) = verification {
            let _ = fs::remove_dir_all(&destination);
            return Err(error.context("downloaded model failed manifest verification"));
        }
        println!(
            "{}",
            json!({"event":"model_downloaded","provider":"huggingface","model_id":id,"source":downloaded.repository,"revision":downloaded.revision,"format":format,"path":destination})
        );
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn register_download_manifest(
    id: &str,
    source: &str,
    revision: &str,
    directory: &Path,
    format: &str,
    kind: ManifestModelKind,
    model_file: Option<&Path>,
) -> Result<()> {
    let files: Vec<String> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    if kind == ManifestModelKind::Llama {
        ensure!(
            files.contains(&"config.json".into()) && files.contains(&"tokenizer.json".into()),
            "download is missing config.json or tokenizer.json"
        );
    } else {
        let model_file = model_file.context("Gemma download is missing its GGUF filename")?;
        ensure!(
            files.iter().any(|file| Path::new(file) == model_file),
            "Gemma download is missing its selected GGUF"
        );
    }
    let mut text = fs::read_to_string(MODEL_MANIFEST)?;
    let bytes: u64 = files
        .iter()
        .map(|file| fs::metadata(directory.join(file)).map(|metadata| metadata.len()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .sum();
    let contract = match kind {
        ManifestModelKind::Llama => {
            "architecture = \"LlamaForCausalLM\"\ntokenizer = \"tokenizer.json\"\n".to_owned()
        }
        ManifestModelKind::Gemma4E2b => format!(
            "architecture = \"gemma4\"\ntokenizer = \"embedded\"\nembedded_tokenizer = true\nmodel_file = \"{}\"\n",
            model_file.expect("Gemma model file was checked").display()
        ),
    };
    text.push_str(&format!("\n[[models]]\nid = \"{id}\"\nsource = \"{source}\"\nrevision = \"{revision}\"\npath = \"{}\"\n{contract}format = \"{format}\"\nbytes = {bytes}\n", directory.display()));
    for file in files {
        let path = directory.join(&file);
        text.push_str(&format!(
            "\n[[models.files]]\npath = \"{file}\"\nbytes = {}\nsha256 = \"{}\"\n",
            fs::metadata(&path)?.len(),
            sha256_file(&path)?
        ));
    }
    let temporary = Path::new("models/manifest.toml.tmp");
    fs::write(temporary, text)?;
    fs::rename(temporary, MODEL_MANIFEST)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    Human,
    Json,
    Quiet,
}

impl ProgressMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "human" => Ok(Self::Human),
            "json" => Ok(Self::Json),
            "quiet" => Ok(Self::Quiet),
            _ => bail!("--progress must be `human`, `json`, or `quiet`"),
        }
    }
}

struct ConversionProgress {
    mode: ProgressMode,
    started: Instant,
    total_tensors: usize,
    total_source_bytes: u64,
    completed_tensors: usize,
    completed_source_bytes: u64,
    packed_bytes: u64,
}
impl ConversionProgress {
    fn new(mode: ProgressMode, total_tensors: usize, total_source_bytes: u64) -> Self {
        Self {
            mode,
            started: Instant::now(),
            total_tensors,
            total_source_bytes,
            completed_tensors: 0,
            completed_source_bytes: 0,
            packed_bytes: 0,
        }
    }
    fn event(&self, stage: &str, tensor: Option<&str>) {
        if self.mode == ProgressMode::Quiet {
            return;
        }
        let elapsed = self.started.elapsed();
        let seconds = elapsed.as_secs_f64();
        let rate = if seconds == 0.0 {
            0.0
        } else {
            self.completed_source_bytes as f64 / seconds
        };
        let percent = if self.total_source_bytes == 0 {
            0.0
        } else {
            self.completed_source_bytes as f64 * 100.0 / self.total_source_bytes as f64
        };
        let eta = (rate > 0.0).then(|| {
            Duration::from_secs_f64(
                (self
                    .total_source_bytes
                    .saturating_sub(self.completed_source_bytes)) as f64
                    / rate,
            )
        });
        match self.mode {
            ProgressMode::Json => println!(
                "{}",
                json!({"event":"conversion_progress","stage":stage,"tensor":tensor,"tensors_completed":self.completed_tensors,"tensors_total":self.total_tensors,"source_bytes_completed":self.completed_source_bytes,"source_bytes_total":self.total_source_bytes,"packed_bytes_written":self.packed_bytes,"percent":percent,"elapsed_ms":elapsed.as_millis(),"source_bytes_per_second":rate,"eta_ms":eta.map(|value| value.as_millis())})
            ),
            ProgressMode::Human => eprintln!(
                "gguf {stage}: {}/{} tensors, {:.1}% source={} MiB packed={} MiB rate={:.2} MiB/s eta={}{}",
                self.completed_tensors,
                self.total_tensors,
                percent,
                self.completed_source_bytes / 1024 / 1024,
                self.packed_bytes / 1024 / 1024,
                rate / 1024.0 / 1024.0,
                eta.map(|value| format!("{:.1}s", value.as_secs_f64()))
                    .unwrap_or_else(|| "estimating".into()),
                tensor
                    .map(|name| format!(" tensor={name}"))
                    .unwrap_or_default()
            ),
            ProgressMode::Quiet => {}
        }
    }
}

fn gguf_name(name: &str) -> Result<String> {
    if name == "model.embed_tokens.weight" {
        return Ok("token_embd.weight".into());
    }
    if name == "model.norm.weight" {
        return Ok("output_norm.weight".into());
    }
    if name == "lm_head.weight" {
        return Ok("output.weight".into());
    }
    let rest = name
        .strip_prefix("model.layers.")
        .context("unsupported non-Llama SafeTensors tensor")?;
    let (layer, tail) = rest.split_once('.').context("invalid Llama layer tensor")?;
    let mapped = match tail {
        "input_layernorm.weight" => "attn_norm",
        "post_attention_layernorm.weight" => "ffn_norm",
        "self_attn.q_proj.weight" => "attn_q",
        "self_attn.k_proj.weight" => "attn_k",
        "self_attn.v_proj.weight" => "attn_v",
        "self_attn.o_proj.weight" => "attn_output",
        "mlp.gate_proj.weight" => "ffn_gate",
        "mlp.up_proj.weight" => "ffn_up",
        "mlp.down_proj.weight" => "ffn_down",
        _ => bail!("unsupported Llama tensor `{name}`"),
    };
    Ok(format!("blk.{layer}.{mapped}.weight"))
}

fn model_quantize(args: &[String]) -> Result<()> {
    let mut model_args = Vec::new();
    let mut id = None;
    let mut format = None;
    let mut progress_mode = ProgressMode::Human;
    let mut quantizer = "auto";
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                index += 1;
                model_args.extend([
                    "--model".into(),
                    args.get(index).context("--model needs a value")?.clone(),
                ]);
            }
            "--id" => {
                index += 1;
                id = args.get(index).cloned();
            }
            "--format" => {
                index += 1;
                format = args.get(index).cloned();
            }
            "--progress" => {
                index += 1;
                progress_mode =
                    ProgressMode::parse(args.get(index).context("--progress needs a value")?)?;
            }
            "--quantizer" => {
                index += 1;
                quantizer = args.get(index).context("--quantizer needs a value")?;
                ensure!(
                    matches!(quantizer, "auto" | "cpu" | "gpu"),
                    "--quantizer must be `auto`, `cpu`, or `gpu`"
                );
            }
            flag => bail!("unknown model quantize option: {flag}"),
        };
        index += 1;
    }
    let gpu = match quantizer {
        "gpu" => Some(MetalRuntime::new().context("initialize Metal GPU quantizer")?),
        "auto" => MetalRuntime::new().ok(),
        "cpu" => None,
        _ => unreachable!(),
    };
    let selected_quantizer = if gpu.is_some() { "gpu" } else { "cpu" };
    let id = id.context("--id is required")?;
    ensure!(
        !id.is_empty() && !id.contains('/') && !id.contains(".."),
        "--id must be a safe model ID"
    );
    let kind = match format.as_deref() {
        Some("q4_0") => GgufTensorType::Q4_0,
        Some("q8_0") => GgufTensorType::Q8_0,
        _ => bail!("--format must be `q4_0` or `q8_0`"),
    };
    let selection = resolve_model(&model_args)?;
    let record = selection
        .manifest
        .context("quantize requires a manifest-backed FP32 model")?;
    ensure!(
        record.format == "safetensors-fp32",
        "quantize currently accepts `safetensors-fp32` models only"
    );
    verify_manifest_model(&record)?;
    let source = record.path.join("model.safetensors");
    ensure!(
        source.is_file(),
        "native GGUF conversion currently requires one unsharded model.safetensors"
    );
    let descriptors = read_safetensors_descriptors(&source)?;
    let total_source_bytes = descriptors
        .iter()
        .map(|d| (d.data_end - d.data_start) as u64)
        .sum();
    let mut progress =
        ConversionProgress::new(progress_mode, descriptors.len(), total_source_bytes);
    progress.event("scan", None);
    let mut writer = GgufWriter::new();
    writer.metadata("general.name", &id);
    writer.metadata(
        "general.file_type",
        if kind == GgufTensorType::Q4_0 {
            "2"
        } else {
            "7"
        },
    );
    let config: Value = serde_json::from_slice(&fs::read(record.path.join("config.json"))?)
        .context("parse source config")?;
    for (key, gguf_key) in [
        ("hidden_size", "llama.embedding_length"),
        ("intermediate_size", "llama.feed_forward_length"),
        ("num_hidden_layers", "llama.block_count"),
        ("num_attention_heads", "llama.attention.head_count"),
        ("num_key_value_heads", "llama.attention.head_count_kv"),
    ] {
        if let Some(value) = config.get(key).and_then(Value::as_u64) {
            writer.metadata(gguf_key, value.to_string());
        }
    }
    for descriptor in descriptors {
        let dims = descriptor.tensor.shape.dims();
        let values = read_safetensors_tensor_f32(&source, &descriptor.name)?;
        let name = gguf_name(&descriptor.name)?;
        let (tensor_type, encoded, gguf_dims) = if dims.len() == 2 {
            ensure!(
                dims[1].is_multiple_of(32),
                "matrix `{}` input width must be a multiple of 32",
                descriptor.name
            );
            let encoded = if let Some(runtime) = &gpu {
                runtime.quantize_gguf(&values, kind)?.0
            } else if kind == GgufTensorType::Q4_0 {
                quantize_q4_0(&values)?
            } else {
                quantize_q8_0(&values)?
            };
            (kind, encoded, vec![dims[1], dims[0]])
        } else if dims.len() == 1 {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            (GgufTensorType::F32, bytes, dims.to_vec())
        } else {
            bail!(
                "unsupported tensor rank {} for {}",
                dims.len(),
                descriptor.name
            );
        };
        progress.packed_bytes += encoded.len() as u64;
        writer.push_tensor(name, gguf_dims, tensor_type, encoded)?;
        progress.completed_tensors += 1;
        progress.completed_source_bytes += (descriptor.data_end - descriptor.data_start) as u64;
        progress.event("quantize", Some(&descriptor.name));
    }
    let output_dir = Path::new("models/gguf").join(&id);
    ensure!(
        !output_dir.exists(),
        "GGUF output already exists: {}",
        output_dir.display()
    );
    fs::create_dir_all(&output_dir)?;
    let result = (|| -> Result<()> {
        fs::write(output_dir.join("model.gguf"), writer.finish()?)?;
        fs::copy(
            record.path.join("config.json"),
            output_dir.join("config.json"),
        )?;
        fs::copy(
            record.path.join(&record.tokenizer),
            output_dir.join("tokenizer.json"),
        )?;
        progress.event("write", None);
        register_gguf_manifest(&id, &record.source, &record.revision, &output_dir, kind)?;
        progress.event("manifest", None);
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&output_dir);
    }
    result?;
    println!(
        "{}",
        json!({"event":"conversion_completed","model_id":id,"format": if kind == GgufTensorType::Q4_0 { "gguf-q4_0" } else { "gguf-q8_0" },"quantizer":selected_quantizer,"elapsed_ms":progress.started.elapsed().as_millis(),"source_bytes":progress.total_source_bytes,"packed_bytes":progress.packed_bytes,"output":output_dir})
    );
    Ok(())
}

fn model_import_gguf(args: &[String]) -> Result<()> {
    let mut path = None;
    let mut id = None;
    let mut config = None;
    let mut tokenizer = None;
    let mut source = None;
    let mut revision = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        index += 1;
        let value = args
            .get(index)
            .context(format!("{flag} needs a value"))?
            .clone();
        match flag.as_str() {
            "--path" => path = Some(PathBuf::from(value)),
            "--id" => id = Some(value),
            "--config" => config = Some(PathBuf::from(value)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value)),
            "--source" => source = Some(value),
            "--revision" => revision = Some(value),
            _ => bail!("unknown model import-gguf option: {flag}"),
        };
        index += 1;
    }
    let path = path.context("--path is required")?;
    let id = id.context("--id is required")?;
    let config = config.context("--config is required")?;
    let tokenizer = tokenizer.context("--tokenizer is required")?;
    let source = source.context("--source is required")?;
    let revision = revision.context("--revision is required")?;
    let model = GgufModel::open(&path)?;
    ensure!(
        model
            .metadata
            .get("general.architecture")
            .map(String::as_str)
            == Some("llama"),
        "GGUF is not a Llama artifact"
    );
    let kind = model
        .tensors
        .iter()
        .find_map(|tensor| {
            matches!(
                tensor.tensor_type,
                GgufTensorType::Q4_0 | GgufTensorType::Q8_0
            )
            .then_some(tensor.tensor_type)
        })
        .context("GGUF has no Q4_0/Q8_0 tensors")?;
    ensure!(
        model.tensors.iter().all(|tensor| matches!(
            tensor.tensor_type,
            GgufTensorType::Q4_0 | GgufTensorType::Q8_0 | GgufTensorType::F32 | GgufTensorType::F16
        )),
        "GGUF has unsupported tensor encodings"
    );
    let output_dir = Path::new("models/gguf").join(&id);
    ensure!(
        !output_dir.exists(),
        "GGUF output already exists: {}",
        output_dir.display()
    );
    fs::create_dir_all(&output_dir)?;
    let result = (|| -> Result<()> {
        fs::copy(&path, output_dir.join("model.gguf"))?;
        fs::copy(config, output_dir.join("config.json"))?;
        fs::copy(tokenizer, output_dir.join("tokenizer.json"))?;
        register_gguf_manifest(&id, &source, &revision, &output_dir, kind)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&output_dir);
    }
    result?;
    println!(
        "{}",
        json!({"event":"gguf_imported","model_id":id,"path":output_dir})
    );
    Ok(())
}

fn register_gguf_manifest(
    id: &str,
    source: &str,
    revision: &str,
    directory: &Path,
    kind: GgufTensorType,
) -> Result<()> {
    let manifest = load_manifest()?;
    ensure!(
        !manifest.models.iter().any(|model| model.id == id),
        "model ID `{id}` already exists"
    );
    let format = if kind == GgufTensorType::Q4_0 {
        "gguf-q4_0"
    } else {
        "gguf-q8_0"
    };
    let files = ["config.json", "tokenizer.json", "model.gguf"];
    let mut text = fs::read_to_string(MODEL_MANIFEST)?;
    text.push_str(&format!("\n[[models]]\nid = \"{id}\"\nsource = \"{source}\"\nrevision = \"{revision}\"\npath = \"{}\"\narchitecture = \"LlamaForCausalLM\"\ntokenizer = \"tokenizer.json\"\nformat = \"{format}\"\n", directory.display()));
    let bytes: u64 = files
        .iter()
        .map(|file| fs::metadata(directory.join(file)).map(|metadata| metadata.len()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .sum();
    text.push_str(&format!("bytes = {bytes}\n"));
    for file in files {
        let path = directory.join(file);
        text.push_str(&format!(
            "\n[[models.files]]\npath = \"{file}\"\nbytes = {}\nsha256 = \"{}\"\n",
            fs::metadata(&path)?.len(),
            sha256_file(&path)?
        ));
    }
    let temp = Path::new("models/manifest.toml.tmp");
    fs::write(temp, text)?;
    fs::rename(temp, MODEL_MANIFEST)?;
    Ok(())
}

fn generate(args: &[String]) -> Result<()> {
    let mut model_args = Vec::new();
    let mut prompt = None;
    let mut max_new_tokens = None;
    let mut greedy = false;
    let mut golden = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--prompt" => {
                index += 1;
                prompt = args.get(index).cloned();
            }
            "--max-new-tokens" => {
                index += 1;
                max_new_tokens = args
                    .get(index)
                    .context("--max-new-tokens needs a value")?
                    .parse::<usize>()
                    .context("parse --max-new-tokens")
                    .map(Some)?;
            }
            "--greedy" => greedy = true,
            "--golden" => {
                index += 1;
                golden = Some(PathBuf::from(
                    args.get(index).context("--golden needs a value")?,
                ));
            }
            "--model" | "--model-dir" => {
                model_args.push(args[index].clone());
                index += 1;
                model_args.push(
                    args.get(index)
                        .context("model option needs a value")?
                        .clone(),
                );
            }
            flag => bail!("unknown generate option: {flag}"),
        }
        index += 1;
    }
    if !greedy {
        bail!("Phase 3 supports only --greedy");
    }
    let selection = resolve_model(&model_args)?;
    eprintln!(
        "atlas: loading model fixture from {}",
        selection.directory.display()
    );
    let prompt = prompt.context("--prompt is required")?;
    let max_new_tokens = max_new_tokens.context("--max-new-tokens is required")?;
    match load_verified_model(&selection)? {
        LoadedModel::Llama(model) => {
            let mut executor =
                resident_executor(&model, &selection, LogitsReadback::SelectedToken)?;
            let generation = executor.generate_greedy(&prompt, max_new_tokens)?;
            if let Some(golden) = golden {
                validate_generation_golden(golden, &generation.generation)?;
            }
            println!(
                "prompt_token_ids: {:?}",
                generation.generation.prompt_token_ids
            );
            println!(
                "generated_token_ids: {:?}",
                generation.generation.generated_token_ids
            );
            println!("text: {}", generation.generation.text);
            println!(
                "{}",
                generation_metrics_json(
                    &selection.id,
                    &model,
                    &generation,
                    selection.manifest.as_ref().map_or(0, |record| record.bytes)
                )
            );
            return Ok(());
        }
        LoadedModel::Gemma4E2b(model) => {
            if golden.is_some() {
                bail!(
                    "--golden is currently a Llama-only JSON contract; Gemma uses artifacts/phase-12a-pre/oracle.json"
                );
            }
            return print_gemma4_generation(&model, &prompt, max_new_tokens, &selection);
        }
    }
}

fn phase_03_model(args: &[String]) -> Result<()> {
    let mut model_args = Vec::new();
    let mut requested_layers = None;
    let mut prompt = "The capital of France is".to_owned();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" | "--model-dir" => {
                model_args.push(args[index].clone());
                index += 1;
                model_args.push(
                    args.get(index)
                        .context("model option needs a value")?
                        .clone(),
                );
            }
            "--layers" => {
                index += 1;
                requested_layers = Some(
                    args.get(index)
                        .context("--layers needs a value")?
                        .parse::<usize>()
                        .context("parse --layers")?,
                );
            }
            "--prompt" => {
                index += 1;
                prompt = args.get(index).context("--prompt needs a value")?.clone();
            }
            flag => bail!("unknown phase_03_model option: {flag}"),
        }
        index += 1;
    }
    let (model, directory) = model_dir(&model_args)?;
    eprintln!(
        "atlas: loading {model} model fixture from {}",
        directory.display()
    );
    let engine = AtlasModel::load(&directory)?;
    let tokens = engine.tokenize(&prompt)?;
    let mut trace = atlas_model::LayerTrace::default();
    let layers = requested_layers.unwrap_or(if model == "larger" {
        1
    } else {
        engine.config.num_hidden_layers
    });
    let output = engine.forward(&tokens, &mut trace, layers)?;
    println!("fixture: {}", engine.root().display());
    println!("layers_executed: {layers}");
    println!("output_elements: {}", output.len());
    for entry in trace.entries {
        println!(
            "layer_trace {} len={} max_abs={:.7}",
            entry.name, entry.len, entry.max_abs
        );
    }
    Ok(())
}

fn metal_info() -> Result<()> {
    let runtime = MetalRuntime::new()?;
    let info = runtime.device_info();
    println!("device: {}", info.name);
    println!("registry_id: {}", info.registry_id);
    println!("cached_pipelines: {}", runtime.pipeline_count());
    Ok(())
}

fn fixture_verify(args: &[String]) -> Result<()> {
    let (_, directory) = model_dir(args)?;
    verify_fixture(&directory)
}

fn fixture_details(model_dir: &Path) -> Result<(String, usize)> {
    let config_path = model_dir.join("config.json");
    let config: Value = serde_json::from_slice(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )
    .context("parse config.json")?;
    let architecture = config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    let index_path = model_dir.join("model.safetensors.index.json");
    let shard_names = if index_path.exists() {
        let index: Value = serde_json::from_slice(&fs::read(&index_path)?)?;
        index
            .get("weight_map")
            .and_then(Value::as_object)
            .context("model.safetensors.index.json is missing weight_map")?
            .values()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
    } else {
        let single = model_dir.join("model.safetensors");
        if !single.exists() {
            bail!(
                "no SafeTensors index or model.safetensors in {}",
                model_dir.display()
            );
        }
        BTreeSet::from(["model.safetensors".to_owned()])
    };
    for shard in &shard_names {
        let path = model_dir.join(shard);
        let mut file = fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut length_bytes = [0; 8];
        file.read_exact(&mut length_bytes)
            .with_context(|| format!("read SafeTensors header length from {}", path.display()))?;
        let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
            .context("SafeTensors header length does not fit this platform")?;
        if header_len > 64 * 1024 * 1024 {
            bail!("SafeTensors header exceeds 64 MiB: {}", path.display());
        }
        let mut header_bytes = vec![0; header_len];
        file.read_exact(&mut header_bytes)
            .with_context(|| format!("read SafeTensors header from {}", path.display()))?;
        let header: Value = serde_json::from_slice(&header_bytes)
            .with_context(|| format!("parse SafeTensors header in {}", path.display()))?;
        if !header.is_object() {
            bail!("SafeTensors header is not an object: {}", path.display());
        }
    }
    Ok((architecture.to_owned(), shard_names.len()))
}

fn verify_fixture(model_dir: &Path) -> Result<()> {
    let (architecture, shards) = fixture_details(model_dir)?;
    println!("fixture: {}", model_dir.display());
    println!("architecture: {architecture}");
    println!("safetensors_shards: {shards}");
    Ok(())
}
