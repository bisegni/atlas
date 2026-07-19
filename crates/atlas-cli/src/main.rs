use std::{
    collections::BTreeSet,
    env, fs,
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{
    QuantFormat, QuantizedMatrix, read_safetensors_descriptors, read_safetensors_tensor_f32,
};
use atlas_metal::MetalRuntime;
use atlas_model::{
    AtlasModel,
    executor::{AtlasExecutor, ExecutorConfig, ExecutorGeneration, ExecutorMetrics},
    validate_generation_golden,
};
use serde_json::Value;

mod server;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("metal-info") => metal_info(),
        Some("fixture") if args.get(1).map(String::as_str) == Some("verify") => {
            fixture_verify(&args[2..])
        }
        Some("generate") => generate(&args[1..]),
        Some("chat") => chat(&args[1..]),
        Some("serve") => serve(&args[1..]),
        Some("phase_03_model") => phase_03_model(&args[1..]),
        Some("phase_05_quant") => phase_05_quant(&args[1..]),
        _ => {
            eprintln!(
                "usage: atlas-cli chat --model small [--prompt TEXT] [--max-tokens N] | atlas-cli serve --model small [--host 127.0.0.1] [--port 8080] | atlas-cli metal-info | atlas-cli fixture verify --model small [--model-dir PATH] | atlas-cli generate --model small --prompt TEXT --max-new-tokens N --greedy [--golden PATH] | atlas-cli phase_03_model --model larger [--model-dir PATH] | atlas-cli phase_05_quant --model small --format fp16|int8|q4 [--tensor NAME]"
            );
            bail!("invalid command")
        }
    }
}

pub(crate) fn generate_completion(
    model: &AtlasModel,
    prompt: &str,
    max_tokens: usize,
) -> Result<ExecutorGeneration> {
    AtlasExecutor::new(model, ExecutorConfig::default())?.generate_greedy(prompt, max_tokens)
}

fn chat(args: &[String]) -> Result<()> {
    let (model_args, prompt, max_tokens) = parse_chat_args(args)?;
    let (_, directory) = model_dir(&model_args)?;
    let model = AtlasModel::load(directory)?;
    if let Some(prompt) = prompt {
        print_completion(&model, &prompt, max_tokens)?;
        return Ok(());
    }
    eprintln!("Atlas chat. Commands: /reset, /help, /quit");
    let stdin = io::stdin();
    let mut history = String::new();
    loop {
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
                history.clear();
                println!("conversation reset");
                continue;
            }
            ReplCommand::Ignore => continue,
            ReplCommand::Prompt(line) => {
                append_user_turn(&mut history, line);
            }
        }
        let result = print_completion(&model, &history, max_tokens)?;
        history.push_str(&model.decode(&result.generation.generated_token_ids)?);
        history.push('\n');
    }
    Ok(())
}

fn print_completion(
    model: &AtlasModel,
    prompt: &str,
    max_tokens: usize,
) -> Result<ExecutorGeneration> {
    let result = generate_completion(model, prompt, max_tokens)?;
    println!("{}", model.decode(&result.generation.generated_token_ids)?);
    eprintln!("{}", metrics_line(&result.metrics));
    Ok(result)
}

fn parse_chat_args(args: &[String]) -> Result<(Vec<String>, Option<String>, usize)> {
    let mut model_args = Vec::new();
    let mut prompt = None;
    let mut max_tokens = 64;
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
            flag => bail!("unknown chat option: {flag}"),
        };
        index += 1;
    }
    Ok((model_args, prompt, max_tokens))
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
        "ttft_ms={:.2} prefill_tok_s={:.2} decode_tok_s={:.2}",
        metrics.ttft.as_secs_f64() * 1000.0,
        metrics.prefill_tokens_per_second(),
        metrics.decode_tokens_per_second()
    )
}

#[cfg(test)]
mod phase_07_tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn chat_arguments_parse_one_shot_and_reject_zero_max_tokens() {
        let (model, prompt, max) = parse_chat_args(&[
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
}

fn serve(args: &[String]) -> Result<()> {
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8080u16;
    let mut filtered = Vec::new();
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--host" => {
                index += 1;
                host = args.get(index).context("--host needs a value")?.clone();
            }
            "--port" => {
                index += 1;
                port = args
                    .get(index)
                    .context("--port needs a value")?
                    .parse()
                    .context("parse --port")?;
            }
            "--model" | "--model-dir" => {
                filtered.push(args[index].clone());
                index += 1;
                filtered.push(
                    args.get(index)
                        .context("model option needs a value")?
                        .clone(),
                );
            }
            flag => bail!("unknown serve option: {flag}"),
        };
        index += 1;
    }
    let (model, directory) = model_dir(&filtered)?;
    server::serve(&model, &directory, &host, port)
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

fn model_dir(args: &[String]) -> Result<(String, PathBuf)> {
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
    let default = match model.as_str() {
        "small" => "models/hf/SmolLM2-135M-Instruct",
        "larger" => "models/hf/SmolLM2-1.7B-Instruct",
        _ => bail!("model must be `small` or `larger`"),
    };
    let directory = directory.unwrap_or_else(|| PathBuf::from(default));
    if !directory.join("config.json").exists() {
        bail!(
            "model fixture is missing at {}; run `scripts/download-models.sh {model}` or pass --model-dir PATH",
            directory.display()
        );
    }
    Ok((model, directory))
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
    let mut model = None;
    let mut model_dir = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--model" => {
                index += 1;
                model = args.get(index).cloned();
            }
            "--model-dir" => {
                index += 1;
                model_dir = args.get(index).map(PathBuf::from);
            }
            flag => bail!("unknown fixture option: {flag}"),
        }
        index += 1;
    }
    let model = model.context("--model is required")?;
    let default_dir = match model.as_str() {
        "small" => "models/hf/SmolLM2-135M-Instruct",
        "larger" => "models/hf/SmolLM2-1.7B-Instruct",
        _ => bail!("model must be `small` or `larger`"),
    };
    let model_dir = model_dir.unwrap_or_else(|| PathBuf::from(default_dir));
    verify_fixture(&model_dir)
}

fn verify_fixture(model_dir: &Path) -> Result<()> {
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
    println!("fixture: {}", model_dir.display());
    println!("architecture: {architecture}");
    println!("safetensors_shards: {}", shard_names.len());
    Ok(())
}
