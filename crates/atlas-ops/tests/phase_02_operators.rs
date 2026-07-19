use std::{fs, path::Path};

use atlas_core::{read_safetensors_descriptors, read_safetensors_tensor_f32};
use atlas_metal::MetalError;
use atlas_ops::{ExecutionMode, NeuralOps, OperatorError};
use serde_json::Value;

fn require_ops() -> Option<NeuralOps> {
    match NeuralOps::new() {
        Ok(ops) => Some(ops),
        Err(OperatorError::Metal(MetalError::NoDevice)) => {
            eprintln!(
                "skipping GPU operator assertions: no Metal device is available to this process"
            );
            None
        }
        Err(error) => panic!("Metal operators should initialize: {error}"),
    }
}

fn assert_close(operator: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{operator} output length");
    let mut max_absolute = 0.0f32;
    let mut max_relative = 0.0f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite(), "{operator} produced non-finite output");
        let absolute = (actual - expected).abs();
        let relative = absolute / expected.abs().max(1e-6);
        max_absolute = max_absolute.max(absolute);
        max_relative = max_relative.max(relative);
        assert!(
            absolute <= tolerance,
            "{operator}: {actual} differs from {expected} by {absolute}"
        );
    }
    eprintln!(
        "{operator}: max_abs={max_absolute:.7}, max_rel={max_relative:.7}, tolerance={tolerance:.7}"
    );
}

#[test]
fn fp32_neural_operators_match_cpu_oracles_for_prefill_and_decode() {
    let Some(ops) = require_ops() else { return };
    assert_eq!(ops.runtime().pipeline_count(), 25);

    let table = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3];
    let (embedded, _) = ops.embedding(&table, 3, 4, &[2, 0]).unwrap();
    assert_close(
        "embedding",
        &embedded,
        &[2.0, 2.1, 2.2, 2.3, 0.0, 0.1, 0.2, 0.3],
        1e-6,
    );

    let lhs = [1.0, -2.0, 3.0, -4.0];
    let rhs = [0.5, 0.25, -1.0, -2.0];
    assert_close(
        "add",
        &ops.add(&lhs, &rhs).unwrap().0,
        &[1.5, -1.75, 2.0, -6.0],
        1e-6,
    );
    assert_close(
        "multiply",
        &ops.multiply(&lhs, &rhs).unwrap().0,
        &[0.5, -0.5, -3.0, 8.0],
        1e-6,
    );
    assert_close(
        "silu",
        &ops.silu(&lhs).unwrap().0,
        &lhs.map(|value| value / (1.0 + (-value).exp())),
        1e-6,
    );

    let norm_input = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -2.5];
    let norm_weight = [1.0, 0.5, 1.5, 2.0];
    let expected_norm = cpu_rms_norm(&norm_input, 2, 4, &norm_weight, 1e-5);
    assert_close(
        "rms_norm",
        &ops.rms_norm(&norm_input, 2, 4, &norm_weight, 1e-5)
            .unwrap()
            .0,
        &expected_norm,
        1e-5,
    );

    let input = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0];
    let weights = [0.5, 1.0, -0.5, 2.0, -1.0, 0.25];
    let expected_prefill = cpu_matmul(&input, &weights, 2, 3, 2);
    assert_close(
        "matmul_prefill",
        &ops.project(ExecutionMode::Prefill, &input, &weights, 2, 3, 2)
            .unwrap()
            .0,
        &expected_prefill,
        1e-6,
    );
    let expected_decode = cpu_matmul(&input[..3], &weights, 1, 3, 2);
    assert_close(
        "matvec_decode",
        &ops.project(ExecutionMode::Decode, &input[..3], &weights, 1, 3, 2)
            .unwrap()
            .0,
        &expected_decode,
        1e-6,
    );

    let cosine = [0.5, 0.25];
    let sine = [0.25, 0.5];
    let expected_rope = cpu_rope(&norm_input, 2, 4, &cosine, &sine);
    assert_close(
        "rope",
        &ops.rope(&norm_input, 2, 4, &cosine, &sine).unwrap().0,
        &expected_rope,
        1e-6,
    );

    let scores = [1.0, 2.0, 3.0, 4.0, 0.0, -1.0];
    let mask = [0.0, 0.0, -1e9, 0.0, -1e9, -1e9];
    let expected_softmax = cpu_masked_softmax(&scores, &mask, 2, 3);
    let (softmax, _) = ops.masked_softmax(&scores, &mask, 2, 3).unwrap();
    assert_close("masked_softmax", &softmax, &expected_softmax, 1e-6);

    let queries = [1.0, 2.0, 0.5, -1.0];
    let keys = [1.0, 0.0, 0.0, 1.0, 2.0, -1.0];
    let expected_scores = cpu_attention_scores(&queries, &keys, 2, 3, 2, 0.5);
    let (attention_scores, _) = ops.attention_scores(&queries, &keys, 2, 3, 2, 0.5).unwrap();
    assert_close(
        "attention_scores",
        &attention_scores,
        &expected_scores,
        1e-6,
    );
    let attention_weights = cpu_masked_softmax(&attention_scores, &[0.0; 6], 2, 3);
    let values = [1.0, 0.0, 0.0, 2.0, 3.0, -1.0];
    let expected_values = cpu_attention_values(&attention_weights, &values, 2, 3, 2);
    assert_close(
        "attention_values",
        &ops.attention_values(&attention_weights, &values, 2, 3, 2)
            .unwrap()
            .0,
        &expected_values,
        1e-6,
    );

    assert_close(
        "logits",
        &ops.process_logits(&[1.0, -2.0, 3.0], &[0.5, 0.0, -1.0], 0.5)
            .unwrap()
            .0,
        &[3.0, -4.0, 4.0],
        1e-6,
    );
}

#[test]
fn downloaded_fixture_exposes_llama_layout_for_operator_dimensions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config_path = root.join("models/hf/SmolLM2-135M-Instruct/config.json");
    let weights_path = root.join("models/hf/SmolLM2-135M-Instruct/model.safetensors");
    if !config_path.exists() || !weights_path.exists() {
        eprintln!("skipping model-layout check: run scripts/download-models.sh first");
        return;
    }
    let config: Value = serde_json::from_slice(&fs::read(config_path).unwrap()).unwrap();
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let heads = config["num_attention_heads"].as_u64().unwrap() as usize;
    let key_value_heads = config["num_key_value_heads"].as_u64().unwrap() as usize;
    assert!(hidden > 0 && heads > 0 && hidden % heads == 0);
    assert!(key_value_heads > 0 && key_value_heads <= heads);
    let descriptors = read_safetensors_descriptors(weights_path).unwrap();
    let q_proj = descriptors
        .iter()
        .find(|item| item.name == "model.layers.0.self_attn.q_proj.weight")
        .expect("first-layer q_proj weight");
    assert_eq!(q_proj.tensor.shape.dims()[1], hidden);
    eprintln!(
        "fixture layout: hidden={hidden}, attention_heads={heads}, key_value_heads={key_value_heads}, head_dim={}",
        hidden / heads
    );
}

#[test]
fn first_layer_rms_norm_weights_match_the_cpu_oracle() {
    let Some(ops) = require_ops() else { return };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = root.join("models/hf/SmolLM2-135M-Instruct/model.safetensors");
    if !model.exists() {
        eprintln!("skipping first-layer weight test: run scripts/download-models.sh first");
        return;
    }
    let weights =
        read_safetensors_tensor_f32(&model, "model.layers.0.input_layernorm.weight").unwrap();
    let input: Vec<f32> = (0..weights.len())
        .map(|index| (index as f32 * 0.013).sin())
        .collect();
    let expected = cpu_rms_norm(&input, 1, weights.len(), &weights, 1e-5);
    let (actual, _) = ops
        .rms_norm(&input, 1, weights.len(), &weights, 1e-5)
        .unwrap();
    assert_close("first_layer_rms_norm", &actual, &expected, 1e-5);
}

fn cpu_rms_norm(
    input: &[f32],
    rows: usize,
    hidden: usize,
    weight: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            let source = &input[row * hidden..(row + 1) * hidden];
            let inverse = (source.iter().map(|x| x * x).sum::<f32>() / hidden as f32 + epsilon)
                .sqrt()
                .recip();
            source
                .iter()
                .zip(weight)
                .map(move |(&x, &w)| x * inverse * w)
        })
        .collect()
}
fn cpu_matmul(
    input: &[f32],
    weights: &[f32],
    rows: usize,
    input_width: usize,
    output_width: usize,
) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..output_width).map(move |output| {
                (0..input_width)
                    .map(|column| {
                        input[row * input_width + column] * weights[output * input_width + column]
                    })
                    .sum()
            })
        })
        .collect()
}
fn cpu_rope(input: &[f32], rows: usize, hidden: usize, cosine: &[f32], sine: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for pair in 0..hidden / 2 {
            let base = row * hidden + pair * 2;
            output[base] = input[base] * cosine[pair] - input[base + 1] * sine[pair];
            output[base + 1] = input[base] * sine[pair] + input[base + 1] * cosine[pair];
        }
    }
    output
}
fn cpu_masked_softmax(input: &[f32], mask: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let source = &input[row * columns..(row + 1) * columns];
        let row_mask = &mask[row * columns..(row + 1) * columns];
        let maximum = source
            .iter()
            .zip(row_mask)
            .map(|(x, m)| x + m)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum = source
            .iter()
            .zip(row_mask)
            .map(|(x, m)| (x + m - maximum).exp())
            .sum::<f32>();
        for column in 0..columns {
            output[row * columns + column] =
                (source[column] + row_mask[column] - maximum).exp() / sum;
        }
    }
    output
}
fn cpu_attention_scores(
    queries: &[f32],
    keys: &[f32],
    query_count: usize,
    key_count: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    (0..query_count)
        .flat_map(|query| {
            (0..key_count).map(move |key| {
                (0..head_dim)
                    .map(|dimension| {
                        queries[query * head_dim + dimension] * keys[key * head_dim + dimension]
                    })
                    .sum::<f32>()
                    * scale
            })
        })
        .collect()
}
fn cpu_attention_values(
    weights: &[f32],
    values: &[f32],
    query_count: usize,
    key_count: usize,
    head_dim: usize,
) -> Vec<f32> {
    (0..query_count)
        .flat_map(|query| {
            (0..head_dim).map(move |dimension| {
                (0..key_count)
                    .map(|key| {
                        weights[query * key_count + key] * values[key * head_dim + dimension]
                    })
                    .sum()
            })
        })
        .collect()
}
