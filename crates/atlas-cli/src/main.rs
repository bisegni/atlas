use std::{
    collections::BTreeSet,
    env, fs,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail, ensure};
use atlas_core::{
    QuantFormat, QuantizedMatrix, read_safetensors_descriptors, read_safetensors_tensor_f32,
};
use atlas_metal::MetalRuntime;
use atlas_model::{AtlasModel, validate_generation_golden};
use serde_json::Value;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("metal-info") => metal_info(),
        Some("fixture") if args.get(1).map(String::as_str) == Some("verify") => {
            fixture_verify(&args[2..])
        }
        Some("generate") => generate(&args[1..]),
        Some("phase_03_model") => phase_03_model(&args[1..]),
        Some("phase_05_quant") => phase_05_quant(&args[1..]),
        _ => {
            eprintln!(
                "usage: atlas-cli metal-info | atlas-cli fixture verify --model small [--model-dir PATH] | atlas-cli generate --model small --prompt TEXT --max-new-tokens N --greedy [--golden PATH] | atlas-cli phase_03_model --model larger [--model-dir PATH] | atlas-cli phase_05_quant --model small --format fp16|int8|q4 [--tensor NAME]"
            );
            bail!("invalid command")
        }
    }
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
