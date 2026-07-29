use std::{
    env, fs,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{GgufModel, GgufTensorType};
use atlas_metal::MetalRuntime;
use atlas_model::{
    Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
    gemma4_executor::{Gemma4E2bExecutor, Gemma4FinishReason, Gemma4Generation, Gemma4KvCacheType},
    render_gemma4_chat,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

mod providers;

static CHAT_INTERRUPTED: AtomicBool = AtomicBool::new(false);
const MODEL_MANIFEST: &str = "models/manifest.toml";
const CHAT_PERFORMANCE_LOG: &str = "artifacts/chat-performance.jsonl";
const GEMMA_DECODE_PROFILE_LOG: &str = "artifacts/phase-12a/gemma4-resident-decode-profile.jsonl";
const GEMMA_FIXED_BENCHMARK_LOG: &str = "artifacts/phase-12a-perf/gemma4-fixed-workload.jsonl";

#[derive(Debug, PartialEq, Eq)]
struct GemmaProfileArgs {
    model_id: String,
    prompt: String,
    decode_tokens: usize,
    max_context: usize,
    kv_cache_type: Gemma4KvCacheType,
}

#[derive(Debug, PartialEq, Eq)]
struct GemmaBenchmarkArgs {
    model_id: String,
    prompt: String,
    warmup_decode_tokens: usize,
    measured_decode_tokens: usize,
    max_context: usize,
    kv_cache_type: Gemma4KvCacheType,
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
    let (model_id, prompt, max_tokens, show_thoughts, kv_cache_type) = parse_chat_args(args)?;
    let selection = resolve_model(&model_id)?;
    let model = load_verified_model(&selection)?;
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache(&model, 4096, kv_cache_type)?;
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
    let record = json!({"event":"generation_metrics","model_id":record.id,"executor":"resident","format":"gguf-gemma4-q4_0","weight_format":generation.metrics.weight_format.as_str(),"embedding_kernel":generation.metrics.embedding_kernel,"output_projection_kernel":generation.metrics.output_projection_kernel,"prompt_tokens":generation.generation.prompt_token_ids.len(),"generated_tokens":generation.generation.generated_token_ids.len(),"finish_reason":finish_reason,"max_new_tokens":max_tokens,"token_limit_source":if context_limit { "context" } else { "explicit" },"visible_chars":visible.chars().count(),"prefill_tok_s":prefill,"decode_tok_s":decode,"resident_bytes":generation.metrics.resident_bytes,"kv_cache_type":generation.metrics.kv_cache_type.as_str(),"kv_cache_bytes":generation.metrics.kv_cache_bytes,"weight_upload_bytes":generation.metrics.weight_upload_bytes,"readback_bytes":generation.metrics.readback_bytes,"command_buffers":generation.metrics.command_buffers,"prefill_command_buffers":generation.metrics.prefill_command_buffers,"decode_command_buffers":generation.metrics.decode_command_buffers,"prefill_path":generation.metrics.prefill_path,"prefill_chunk_size":generation.metrics.prefill_chunk_size,"prefill_chunks":generation.metrics.prefill_chunks,"attention_kernel":generation.metrics.attention_kernel,"timing":{"prefill_ms":generation.metrics.prefill.as_secs_f64()*1000.0,"decode_ms":generation.metrics.decode.as_secs_f64()*1000.0,"host_ms":generation.metrics.host_wall_time.as_secs_f64()*1000.0}});
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

fn generate(args: &[String]) -> Result<()> {
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = None;
    let mut greedy = false;
    let mut chat = false;
    let mut kv_cache_type = Gemma4KvCacheType::F32;
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
            "--kv-cache-type" => {
                i += 1;
                kv_cache_type = Gemma4KvCacheType::parse(
                    args.get(i).context("--kv-cache-type needs a value")?,
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
    let mut executor = Gemma4E2bExecutor::new_with_kv_cache(&gemma, 4096, kv_cache_type)?;
    let max_tokens = max_tokens.context("--max-new-tokens is required")?;
    let result = if chat {
        executor.generate_greedy_chat_stream(&prompt, max_tokens, &CHAT_INTERRUPTED, |_| Ok(()))?
    } else {
        executor.generate_greedy_stream(&prompt, max_tokens, &CHAT_INTERRUPTED, |_| Ok(()))?
    };
    println!(
        "prompt_token_ids: {:?}\ngenerated_token_ids: {:?}\ntext: {}",
        result.generation.prompt_token_ids,
        result.generation.generated_token_ids,
        result.generation.text
    );
    println!(
        "{}",
        json!({"event":"generation_metrics","model_id":selection.id,"executor":"resident","format":"gguf-gemma4-q4_0","weight_format":result.metrics.weight_format.as_str(),"embedding_kernel":result.metrics.embedding_kernel,"output_projection_kernel":result.metrics.output_projection_kernel,"finish_reason":match result.finish_reason { Gemma4FinishReason::Eos => "eos", Gemma4FinishReason::MaxTokens => "max_tokens", Gemma4FinishReason::Cancelled => "cancelled" },"resident_bytes":result.metrics.resident_bytes,"kv_cache_type":result.metrics.kv_cache_type.as_str(),"kv_cache_bytes":result.metrics.kv_cache_bytes,"weight_upload_bytes":result.metrics.weight_upload_bytes,"readback_bytes":result.metrics.readback_bytes,"command_buffers":result.metrics.command_buffers})
    );
    Ok(())
}

/// Diagnostic-only exact Metal timing for the growing-context Gemma decode
/// path. This command intentionally does not share the normal chat metrics
/// file because it submits every observed kernel separately.
fn profile(args: &[String]) -> Result<()> {
    let args = parse_profile_args(args)?;
    let record = resolve_model(&args.model_id)?;
    let model = load_verified_model(&record)?;
    let mut executor =
        Gemma4E2bExecutor::new_with_kv_cache(&model, args.max_context, args.kv_cache_type)?;
    let profile = executor.profile_decode(&args.prompt, args.decode_tokens)?;
    let samples = profile
        .samples
        .iter()
        .map(|sample| {
            json!({
                "decode_position": sample.decode_position,
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

/// Run a fixed-length Resident decode workload for Phase 12a performance
/// comparison. This command is intentionally separate from `chat` and
/// `generate`: it keeps selecting tokens after EOS so cache modes with
/// different EOS behavior still receive identical decode work.
fn benchmark(args: &[String]) -> Result<()> {
    CHAT_INTERRUPTED.store(false, Ordering::Release);
    let args = parse_benchmark_args(args)?;
    let record = resolve_model(&args.model_id)?;
    let model = load_verified_model(&record)?;
    let mut executor =
        Gemma4E2bExecutor::new_with_kv_cache(&model, args.max_context, args.kv_cache_type)?;
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
    let record = json!({
        "event": "gemma4_fixed_workload_benchmark",
        "diagnostic": true,
        "model_id": record.id,
        "executor": "resident",
        "format": "gguf-gemma4-q4_0",
        "weight_format": generation.metrics.weight_format.as_str(),
        "embedding_kernel": generation.metrics.embedding_kernel,
        "output_projection_kernel": generation.metrics.output_projection_kernel,
        "prompt_template": "gemma4_chat",
        "prompt_token_sha256": prompt_token_digest,
        "kv_cache_type": generation.metrics.kv_cache_type.as_str(),
        "max_context": args.max_context,
        "prompt_tokens": generation.generation.prompt_token_ids.len(),
        "warmup_decode_tokens": args.warmup_decode_tokens,
        "fixed_decode_tokens": args.measured_decode_tokens,
        "measured_decode_tokens": args.measured_decode_tokens,
        "completed_decode_tokens": generation.generation.generated_token_ids.len(),
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
        "prefill_path": generation.metrics.prefill_path,
        "prefill_chunk_size": generation.metrics.prefill_chunk_size,
        "prefill_chunks": generation.metrics.prefill_chunks,
        "selected_kernels": {
            "attention": generation.metrics.attention_kernel,
            "q4_projection": "matvec_q4_0_16row",
            "q6_projection": generation.metrics.q6_projection_kernel,
            "embedding": generation.metrics.embedding_kernel,
            "output_projection": generation.metrics.output_projection_kernel,
            "kv_append": match generation.metrics.kv_cache_type {
                Gemma4KvCacheType::F32 => "kv_append_decode_f32",
                Gemma4KvCacheType::Q8_0 => "kv_append_decode_q8_0",
                Gemma4KvCacheType::Q4_0 => "kv_append_decode_q4_0",
            },
        },
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

fn parse_profile_args(args: &[String]) -> Result<GemmaProfileArgs> {
    let mut model_id = None;
    let mut prompt =
        "Atlas Resident decode profile: summarize GPU-resident inference in one sentence."
            .to_owned();
    let mut decode_tokens = 128;
    let mut max_context = 4096;
    let mut kv_cache_type = Gemma4KvCacheType::F32;
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
            flag => bail!("unknown profile option: {flag}"),
        }
        i += 1;
    }
    ensure!(decode_tokens >= 128, "--decode-tokens must be at least 128");
    ensure!(max_context > 0, "--max-context must be positive");
    Ok(GemmaProfileArgs {
        model_id: model_id.context("--model is required")?,
        prompt,
        decode_tokens,
        max_context,
        kv_cache_type,
    })
}

fn parse_benchmark_args(args: &[String]) -> Result<GemmaBenchmarkArgs> {
    let mut model = None;
    let mut prompt = None;
    let mut decode_tokens = None;
    let mut warmup_decode_tokens = 0;
    let mut max_context = 4096;
    let mut kv_cache_type = Gemma4KvCacheType::F32;
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
)> {
    let mut model = None;
    let mut prompt = None;
    let mut max = None;
    let mut thoughts = false;
    let mut kv_cache_type = Gemma4KvCacheType::F32;
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
    ))
}

fn model_command(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .context("model command requires a subcommand")?;
    match command.as_str() {
        "search" => model_search(&args[1..]),
        "download" => model_download(&args[1..]),
        "inspect" | "verify" => {
            let id = option_value(&args[1..], "--model")?;
            let model = resolve_model(&id)?;
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
        _ => bail!("model command must be search, download, inspect, or verify"),
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
    println!("device: {}\nregistry_id: {}", info.name, info.registry_id);
    Ok(())
}

#[cfg(test)]
mod kv_cache_cli_tests {
    use super::{
        Gemma4KvCacheType, ThoughtFilter, parse_benchmark_args, parse_chat_args,
        parse_profile_args, token_ids_sha256,
    };

    #[test]
    fn chat_accepts_explicit_kv_cache_precision() {
        let args = vec![
            "--model".to_owned(),
            "gemma4-e2b-q4_0".to_owned(),
            "--kv-cache-type".to_owned(),
            "q8_0".to_owned(),
        ];
        let (_, _, _, _, cache_type) = parse_chat_args(&args).expect("parse chat options");
        assert_eq!(cache_type, Gemma4KvCacheType::Q8_0);
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
        assert_eq!(parsed.warmup_decode_tokens, 0);
        assert_eq!(parsed.max_context, 4096);
        assert_eq!(parsed.kv_cache_type, Gemma4KvCacheType::Q4_0);
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
}
