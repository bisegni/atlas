use std::path::Path;

use atlas_core::{GgufModel, GgufTensorType, dequantize_block, quantize_q8_0};
use atlas_metal::{MetalError, MetalRuntime};

fn packed_q8_matvec_oracle(input: &[f32], weights: &[u8], rows: usize) -> Vec<f32> {
    let blocks_per_row = input.len() / 32;
    (0..rows)
        .map(|row| {
            (0..blocks_per_row)
                .map(|block| {
                    let start = (row * blocks_per_row + block) * GgufTensorType::Q8_0.block_bytes();
                    let mut values = [0.0; 32];
                    dequantize_block(
                        GgufTensorType::Q8_0,
                        &weights[start..start + GgufTensorType::Q8_0.block_bytes()],
                        &mut values,
                    )
                    .unwrap();
                    values
                        .iter()
                        .enumerate()
                        .map(|(index, weight)| input[block * 32 + index] * weight)
                        .sum::<f32>()
                })
                .sum()
        })
        .collect()
}

#[test]
fn bootstrap_kernels_match_cpu_references_and_pipelines_are_cached() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping GPU assertions: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let cached_pipeline_count = runtime.pipeline_count();
    assert!(
        cached_pipeline_count >= 5,
        "the five Phase 0 bootstrap kernels must be cached"
    );

    let lhs: Vec<f32> = (0..1024).map(|value| value as f32 * 0.25).collect();
    let rhs: Vec<f32> = (0..1024).map(|value| -(value as f32) * 0.125).collect();
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect();

    let (result, first_timing) = runtime.vector_add(&lhs, &rhs).unwrap();
    assert!(first_timing.wall_time.as_nanos() > 0);
    assert!(first_timing.gpu_time.is_some());
    for (actual, expected) in result.iter().zip(expected) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    let (scaled, timing) = runtime.scalar_multiply(&lhs, -2.0).unwrap();
    assert!(timing.gpu_time.is_some());
    assert_eq!(
        scaled,
        lhs.iter().map(|value| value * -2.0).collect::<Vec<_>>()
    );

    let silu_input = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let (silu, timing) = runtime.silu(&silu_input).unwrap();
    assert!(timing.gpu_time.is_some());
    for (actual, input) in silu.iter().zip(silu_input) {
        let expected = input / (1.0 + (-input).exp());
        assert!((actual - expected).abs() < 1e-6);
    }

    let (sum, timing) = runtime.sum(&lhs).unwrap();
    assert!(timing.gpu_time.is_some());
    assert!((sum - lhs.iter().sum::<f32>()).abs() < 1e-3);

    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (transposed, timing) = runtime.transpose(&matrix, 2, 3).unwrap();
    assert!(timing.gpu_time.is_some());
    assert_eq!(transposed, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    for _ in 0..99 {
        let (result, _) = runtime.vector_add(&lhs, &rhs).unwrap();
        assert_eq!(
            result,
            lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect::<Vec<_>>()
        );
    }
    assert_eq!(runtime.pipeline_count(), cached_pipeline_count);
}

#[test]
fn q8_packed_matvec_matches_the_cpu_packed_block_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping GPU assertions: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let input: Vec<f32> = (0..64).map(|index| (index as f32 - 19.0) / 11.0).collect();
    let source: Vec<f32> = (0..(3 * 64))
        .map(|index| ((index as f32 * 0.17).sin() - 0.3) * 2.0)
        .collect();
    let packed = quantize_q8_0(&source).unwrap();
    let expected = packed_q8_matvec_oracle(&input, &packed, 3);
    let (actual, timing) = runtime
        .matvec_gguf_packed(&input, &packed, GgufTensorType::Q8_0, 64, 3)
        .unwrap();
    assert!(timing.gpu_time.is_some());
    for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "Q8 packed matvec mismatch at row {row}: GPU={actual}, CPU={expected}"
        );
    }
}

#[test]
#[ignore = "requires local Metal and models/gguf/small-q8-gpu-20260719124407/model.gguf"]
fn q8_fixture_projection_samples_match_the_cpu_packed_block_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping GPU assertions: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("initialize Metal runtime: {error}"),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let gguf = GgufModel::open(root.join("models/gguf/small-q8-gpu-20260719124407/model.gguf"))
        .expect("open Q8 fixture");
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        // SmolLM2 ties the LM head to token embeddings, so this fixture does
        // not contain a separate `output.weight` tensor.
        "token_embd.weight",
    ] {
        let tensor = gguf
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap_or_else(|| panic!("missing Q8 fixture tensor {name}"));
        assert_eq!(tensor.tensor_type, GgufTensorType::Q8_0, "{name}");
        assert_eq!(tensor.dims.len(), 2, "{name}");
        let input_width = tensor.dims[0];
        let output_width = tensor.dims[1];
        let input: Vec<f32> = (0..input_width)
            .map(|index| ((index as f32 * 0.013).sin() - 0.2) * 1.7)
            .collect();
        let weights = gguf.tensor_data(tensor).expect("read packed tensor");
        let expected = packed_q8_matvec_oracle(&input, weights, output_width);
        let (actual, timing) = runtime
            .matvec_gguf_packed(
                &input,
                weights,
                GgufTensorType::Q8_0,
                input_width,
                output_width,
            )
            .unwrap_or_else(|error| panic!("dispatch {name}: {error}"));
        assert!(timing.gpu_time.is_some(), "{name}");
        for (row, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-4,
                "Q8 fixture projection {name} row {row}: GPU={actual}, CPU={expected}"
            );
        }
    }
}
