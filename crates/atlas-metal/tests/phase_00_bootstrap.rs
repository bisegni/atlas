use std::path::Path;

use atlas_core::{GgufModel, GgufTensorType, dequantize_block, f16_bits_to_f32, quantize_q8_0};
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

fn llama_cpp_q4_0_block(scale_bits: u16, low: &[u8; 16], high: &[u8; 16]) -> Vec<u8> {
    let mut block = Vec::with_capacity(GgufTensorType::Q4_0.block_bytes());
    block.extend_from_slice(&scale_bits.to_le_bytes());
    block.extend(low.iter().zip(high).map(|(&lo, &hi)| lo | (hi << 4)));
    block
}

/// Test-only implementation of llama.cpp's Q6_K row decode.  Do not replace
/// this with Atlas's production decoder: this test is the independent oracle
/// for the Metal Q6_K lookup kernel.
fn llama_cpp_q6_k_row(blocks: &[u8], width: usize) -> Vec<f32> {
    assert_eq!(width % 256, 0);
    assert_eq!(
        blocks.len(),
        width / 256 * GgufTensorType::Q6K.block_bytes()
    );
    let mut values = Vec::with_capacity(width);
    for block in blocks.chunks_exact(GgufTensorType::Q6K.block_bytes()) {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for half in 0..2usize {
            for stream in 0..4usize {
                for lane in 0..32usize {
                    let ql = block[half * 64 + lane + if stream % 2 == 1 { 32 } else { 0 }];
                    let low = if stream < 2 { ql & 0x0f } else { ql >> 4 };
                    let high = (block[128 + half * 32 + lane] >> (stream * 2)) & 0x03;
                    let scale = block[192 + half * 8 + stream * 2 + lane / 16] as i8 as f32;
                    values.push((((high << 4) | low) as i32 - 32) as f32 * scale * d);
                }
            }
        }
    }
    values
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
fn gelu_kernel_is_finite_for_captured_and_extreme_finite_inputs() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => return,
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let input = [-100.0, -17.29969, -1.0, 0.0, 1.0, 17.29969, 100.0];
    let (actual, timing) = runtime.gelu(&input).expect("run GELU");
    assert!(timing.gpu_time.is_some());
    for (index, (&actual, &input)) in actual.iter().zip(&input).enumerate() {
        let argument = 0.797_884_6 * (input + 0.044_715 * input * input * input);
        let expected = 0.5 * input * (1.0 + argument.tanh());
        assert!(actual.is_finite(), "GELU output {index} is non-finite");
        assert!(
            (actual - expected).abs() <= 1e-5,
            "GELU mismatch at {index}: GPU={actual}, CPU={expected}"
        );
    }
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
fn q4_packed_matvec_uses_llama_cpp_half_block_nibble_order() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => return,
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let input: Vec<f32> = (1..=32).map(|value| value as f32).collect();
    let low = std::array::from_fn(|index| index as u8);
    let high = std::array::from_fn(|index| (15 - index) as u8);
    let packed = llama_cpp_q4_0_block(0x3c00, &low, &high); // f16 scale 1.0
    let expected = (0..16)
        .map(|index| input[index] * (index as f32 - 8.0))
        .sum::<f32>()
        + (0..16)
            .map(|index| input[index + 16] * (7.0 - index as f32))
            .sum::<f32>();
    let (actual, timing) = runtime
        .matvec_gguf_packed(&input, &packed, GgufTensorType::Q4_0, 32, 1)
        .expect("run canonical Q4_0 matvec");
    assert!(timing.gpu_time.is_some());
    assert!((actual[0] - expected).abs() <= 1e-4);
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

#[test]
#[ignore = "requires local Metal and the Gemma 4 E2B Q4 GGUF fixture"]
fn gemma4_q6_k_ple_row_matches_llama_cpp_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => return,
        Err(error) => panic!("initialize Metal runtime: {error}"),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let gguf =
        GgufModel::open(root.join("models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf"))
            .expect("open Gemma Q4 fixture");
    let tensor = gguf
        .tensors
        .iter()
        .find(|tensor| tensor.name == "per_layer_token_embd.weight")
        .expect("Gemma per-layer token embedding tensor");
    assert_eq!(tensor.tensor_type, GgufTensorType::Q6K);
    let hidden = tensor.dims[0];
    let vocabulary = tensor.dims[1];
    let token = 669usize;
    let bytes = gguf
        .tensor_data(tensor)
        .expect("read Gemma embedding bytes");
    let row_bytes = hidden / 256 * GgufTensorType::Q6K.block_bytes();
    let expected = llama_cpp_q6_k_row(&bytes[token * row_bytes..][..row_bytes], hidden);
    assert!(expected.iter().all(|value| value.is_finite()));
    let (actual, timing) = runtime
        .embedding_lookup_q6_k(bytes, vocabulary, hidden, &[token as u32])
        .expect("run Q6_K embedding lookup");
    assert!(timing.gpu_time.is_some());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = expected.abs() * 1e-6 + 1e-3;
        assert!(
            (actual - expected).abs() <= tolerance,
            "Q6_K PLE lookup mismatch at {index}: GPU={actual}, llama.cpp-oracle={expected}"
        );
    }
}
