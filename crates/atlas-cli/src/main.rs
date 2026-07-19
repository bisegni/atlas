use std::{
    collections::BTreeSet,
    env, fs,
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{
    GgufModel, GgufTensorType, GgufWriter, QuantFormat, QuantizedMatrix, quantize_q4_0,
    quantize_q8_0, read_safetensors_descriptors, read_safetensors_tensor_f32,
};
use atlas_metal::MetalRuntime;
use atlas_model::{
    AtlasModel,
    executor::{
        AtlasExecutor, ExecutorConfig, ExecutorGeneration, ExecutorMetrics, ExecutorMode,
        GenerationEvent, LogitsReadback,
    },
    runtime::{AtlasRuntime, RuntimeConfig, RuntimeEvent, RuntimeRequest},
    sampling::SamplingConfig,
    validate_generation_golden,
};
use serde_json::{Value, json};

static CHAT_INTERRUPTED: AtomicBool = AtomicBool::new(false);

const MODEL_MANIFEST: &str = "models/manifest.toml";

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

#[derive(Debug)]
struct ModelSelection {
    id: String,
    directory: PathBuf,
    manifest: Option<ModelRecord>,
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
        Some("model") => model_command(&args[1..]),
        Some("generate") => generate(&args[1..]),
        Some("chat") => chat(&args[1..]),
        Some("runtime") => runtime_command(&args[1..]),
        Some("phase_03_model") => phase_03_model(&args[1..]),
        Some("phase_05_quant") => phase_05_quant(&args[1..]),
        Some("phase_08b_decode") => phase_08b_decode(&args[1..]),
        _ => {
            eprintln!(
                "usage: atlas-cli model inspect|verify --model ID | atlas-cli model quantize --model FP32_ID --id ID --format q4_0|q8_0 [--progress human|json|quiet] | atlas-cli model import-gguf --path FILE --id ID --config FILE --tokenizer FILE --source SOURCE --revision REVISION"
            );
            bail!("invalid command")
        }
    }
}

/// Run exactly one request through the reusable bounded runtime. Multi-session
/// admission is a library concern and is intentionally exercised by the
/// runtime acceptance test rather than exposed as an unstable CLI protocol.
fn runtime_command(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    install_chat_sigint_handler();
    let (model_args, prompt, max_tokens, mode) = parse_chat_args(args)?;
    ensure!(
        mode == ExecutorMode::Resident,
        "atlas runtime requires --executor resident"
    );
    let prompt = prompt.context("atlas runtime requires --prompt")?;
    let selection = resolve_model(&model_args)?;
    let model = AtlasModel::load(&selection.directory)?;
    let mut runtime = AtlasRuntime::new(&model, RuntimeConfig::default())?;
    let session = runtime.submit(RuntimeRequest {
        prompt,
        max_new_tokens: max_tokens,
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
    let (model_args, prompt, max_tokens, mode) = parse_chat_args(args)?;
    let selection = resolve_model(&model_args)?;
    let directory = &selection.directory;
    let model = AtlasModel::load(directory)?;
    if let Some(prompt) = prompt {
        let mut turn_metrics = ChatTurnMetrics::new();
        print_completion(
            &model,
            &prompt,
            max_tokens,
            mode,
            &selection.id,
            &mut turn_metrics,
        )?;
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
        let result = match print_completion(
            &model,
            &history,
            max_tokens,
            mode,
            &selection.id,
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
    max_tokens: usize,
    mode: ExecutorMode,
    model_id: &str,
    turn_metrics: &mut ChatTurnMetrics,
) -> Result<ExecutorGeneration> {
    let mut executor = AtlasExecutor::new(
        model,
        ExecutorConfig {
            mode,
            ..Default::default()
        },
    )?;
    let mut stdout = io::stdout();
    let result =
        executor.generate_greedy_stream(prompt, max_tokens, &CHAT_INTERRUPTED, |event| {
            turn_metrics.record_event(&event);
            write_stream_event(&mut stdout, &event)
        })?;
    writeln!(stdout)?;
    stdout.flush()?;
    eprintln!(
        "{}",
        json!({
            "event": "generation_metrics",
            "model_id": model_id,
            "executor": match mode { ExecutorMode::Reference => "reference", ExecutorMode::Resident => "resident" },
            "format": model.format_name(),
            "token_ids": result.generation.generated_token_ids,
            "finish_reason": format!("{:?}", result.finish_reason).to_lowercase(),
            "resident_bytes": result.metrics.resident_bytes,
            "weight_upload_bytes": result.metrics.weight_upload_bytes,
            "readback_bytes": result.metrics.readback_bytes,
            "command_buffers": result.metrics.command_buffer_count,
            "timing": { "ttft_ms": result.metrics.ttft.as_secs_f64() * 1000.0, "host_ms": result.metrics.host_wall_time.as_secs_f64() * 1000.0 },
            "decode_tok_s": result.metrics.decode_tokens_per_second(),
        })
    );
    Ok(result)
}

fn write_stream_event(writer: &mut impl Write, event: &GenerationEvent) -> Result<()> {
    if let GenerationEvent::Token { text, .. } = event {
        write!(writer, "{text}")?;
        writer.flush()?;
    }
    Ok(())
}

fn parse_chat_args(args: &[String]) -> Result<(Vec<String>, Option<String>, usize, ExecutorMode)> {
    let mut model_args = Vec::new();
    let mut prompt = None;
    let mut max_tokens = 64;
    let mut mode = ExecutorMode::Resident;
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
                max_tokens = args
                    .get(index)
                    .context("--max-tokens needs a value")?
                    .parse()
                    .context("parse --max-tokens")?;
                ensure!(max_tokens > 0, "--max-tokens must be positive");
            }
            "--executor" => {
                index += 1;
                mode = match args.get(index).map(String::as_str) {
                    Some("reference") => ExecutorMode::Reference,
                    Some("resident") => ExecutorMode::Resident,
                    _ => bail!("--executor must be `reference` or `resident`"),
                };
            }
            flag => bail!("unknown chat option: {flag}"),
        };
        index += 1;
    }
    Ok((model_args, prompt, max_tokens, mode))
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
        let (model, prompt, max, mode) = parse_chat_args(&[
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
        assert_eq!(max, 7);
        assert_eq!(mode, ExecutorMode::Resident);
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
    fn chat_accepts_explicit_resident_executor_only() {
        let (_, _, _, mode) = parse_chat_args(&[
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
    let (model_name, directory) = model_dir(&model_args)?;
    let model = AtlasModel::load(directory)?;
    if trace_stages {
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
    let path = Path::new(MODEL_MANIFEST);
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
                format: String::new(),
                bytes: 0,
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
                "format" => model.format = text.context("model format must be quoted")?.to_owned(),
                "bytes" => model.bytes = value.parse().context("parse model bytes")?,
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
    }
    let mut ids = BTreeSet::new();
    for model in &models {
        ensure!(
            ids.insert(model.id.as_str()),
            "duplicate model ID `{}`",
            model.id
        );
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
    if !record.path.join("config.json").exists() {
        bail!(
            "manifest model `{model}` is missing at {}; download its pinned revision or pass --model-dir for a developer fixture",
            record.path.display()
        );
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
                "bytes": record.bytes,
            })
        ),
        "verify" => {
            verify_manifest_model(&record)?;
            println!(
                "{}",
                json!({"model_id": record.id, "verified": true, "bytes": record.bytes})
            );
        }
        _ => bail!("model command must be `inspect`, `verify`, `quantize`, or `import-gguf`"),
    }
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
    let (_, directory) = model_dir(&model_args)?;
    eprintln!("atlas: loading model fixture from {}", directory.display());
    let generation = AtlasModel::load(&directory)?.generate_greedy(
        &prompt.context("--prompt is required")?,
        max_new_tokens.context("--max-new-tokens is required")?,
    )?;
    if let Some(golden) = golden {
        validate_generation_golden(golden, &generation)?;
    }
    println!("prompt_token_ids: {:?}", generation.prompt_token_ids);
    println!("generated_token_ids: {:?}", generation.generated_token_ids);
    println!("text: {}", generation.text);
    for entry in generation.trace.entries {
        println!(
            "trace {} len={} max_abs={:.7}",
            entry.name, entry.len, entry.max_abs
        );
    }
    Ok(())
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
