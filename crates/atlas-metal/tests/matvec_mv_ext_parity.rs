//! Tolerance-based correctness check for the production 64-row-per-threadgroup
//! matvec kernels against an independent CPU oracle: `dequantize_block` from
//! atlas-core plus a plain dot product (and the same RMS normalization the
//! `_rms` variants fold into the dispatch). The kernels use per-lane
//! accumulation with simd-group reductions, so this asserts a small
//! max-absolute-difference (same contract as `attention_flash_correctness.rs`)
//! instead of byte equality. Covers the Gemma4 E2B projection geometries plus
//! partial-width rows (non-multiples of 32) that exercise the active-row and
//! partial-threadgroup paths.

use atlas_core::{GgufTensorType, dequantize_block};
use atlas_metal::{MetalError, MetalRuntime};

const THREADS: usize = 256;
const EPSILON: f32 = 1e-6;

fn next_u32(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn next_f32(state: &mut u32) -> f32 {
    (next_u32(state) >> 8) as f32 / (1u32 << 24) as f32
}

fn fill_f32(values: &mut [f32], state: &mut u32) {
    for value in values.iter_mut() {
        *value = (next_f32(state) - 0.5) * 2.0;
    }
}

fn fill_rms_weight(values: &mut [f32], state: &mut u32) {
    for value in values.iter_mut() {
        *value = 0.5 + next_f32(state);
    }
}

fn build_q4_weights(rows: usize, blocks: usize, state: &mut u32) -> Vec<u8> {
    let mut weights = vec![0u8; rows * blocks * 18];
    for (i, chunk) in weights.chunks_mut(18).enumerate() {
        let scale = (0.005 + (i as f32 % 101.0) * 0.0003) as f32;
        let half = (scale * 32768.0).round() as u16;
        chunk[..2].copy_from_slice(&half.to_le_bytes());
        for (j, byte) in chunk[2..].iter_mut().enumerate() {
            let nibble = ((i * 7 + j * 13) % 16) as u8;
            *byte = nibble | (nibble << 4);
        }
    }
    let _ = state;
    weights
}

fn build_q6_weights(rows: usize, blocks: usize, state: &mut u32) -> Vec<u8> {
    // Atlas Q6_K (GGUF) layout: ql[128], qh[64], scales[16], f16 d at the end.
    let mut weights = vec![0u8; rows * blocks * 210];
    for (i, chunk) in weights.chunks_mut(210).enumerate() {
        let half = (0.02 + (i as f32 % 51.0) * 0.002) as f32;
        let bits = (half * 32768.0).round() as u16;
        chunk[208..210].copy_from_slice(&bits.to_le_bytes());
        for byte in chunk[..192].iter_mut() {
            *byte = next_u32(state) as u8;
        }
        for (j, byte) in chunk[192..208].iter_mut().enumerate() {
            *byte = (((j * 3 + i % 7) % 9) as i8 - 4) as u8;
        }
    }
    weights
}

fn cpu_q4_matvec(input: &[f32], weights: &[u8], rows: usize) -> Vec<f32> {
    let blocks = input.len() / 32;
    let mut block = vec![0.0f32; 32];
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let chunk = &weights[(row * blocks + b) * 18..(row * blocks + b + 1) * 18];
            dequantize_block(GgufTensorType::Q4_0, chunk, &mut block).unwrap();
            for lane in 0..32 {
                acc += input[b * 32 + lane] * block[lane];
            }
        }
        out[row] = acc;
    }
    out
}

fn cpu_q6_matvec(input: &[f32], weights: &[u8], rows: usize) -> Vec<f32> {
    let blocks = input.len() / 256;
    let mut block = vec![0.0f32; 256];
    let mut out = vec![0.0f32; rows];
    for row in 0..rows {
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let chunk = &weights[(row * blocks + b) * 210..(row * blocks + b + 1) * 210];
            dequantize_block(GgufTensorType::Q6K, chunk, &mut block).unwrap();
            for lane in 0..256 {
                acc += input[b * 256 + lane] * block[lane];
            }
        }
        out[row] = acc;
    }
    out
}

fn cpu_rms_normalize(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inverse_rms = 1.0 / (mean_sq + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * w * inverse_rms)
        .collect()
}

fn compare(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f32;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
        mean_abs += diff;
    }
    mean_abs /= reference.len() as f32;
    eprintln!("{label}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
    assert!(
        max_abs < 1e-3,
        "{label}: mv output diverges from the CPU oracle (max_abs={max_abs:.3e})"
    );
}

fn dispatch_64(
    runtime: &MetalRuntime,
    kernel: &'static str,
    buffers: &[(&atlas_metal::GpuBuffer, usize)],
    groups: usize,
) {
    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(kernel, buffers, groups, THREADS)
        .unwrap();
    command.finish().unwrap();
}

fn run_plain(
    runtime: &MetalRuntime,
    kernel: &'static str,
    input: &[f32],
    weights: &[u8],
    rows: usize,
) -> Vec<f32> {
    let input_buf = runtime.upload_f32(input).unwrap();
    let weights_buf = runtime.upload_bytes(weights).unwrap();
    let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[rows as u32]).unwrap();
    let groups = (rows as u32 + 63) / 64;
    dispatch_64(
        runtime,
        kernel,
        &[
            (&input_buf, 0),
            (&weights_buf, 0),
            (&out_buf, 0),
            (&input_width_buf, 0),
            (&output_width_buf, 0),
        ],
        groups as usize,
    );
    runtime.read_f32(&out_buf, rows).unwrap()
}

fn run_rms(
    runtime: &MetalRuntime,
    kernel: &'static str,
    input: &[f32],
    weights: &[u8],
    rms_weight: &[f32],
    rows: usize,
) -> Vec<f32> {
    let input_buf = runtime.upload_f32(input).unwrap();
    let weights_buf = runtime.upload_bytes(weights).unwrap();
    let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[rows as u32]).unwrap();
    let rms_buf = runtime.upload_f32(rms_weight).unwrap();
    let epsilon_buf = runtime.upload_f32(&[EPSILON]).unwrap();
    let groups = (rows as u32 + 63) / 64;
    dispatch_64(
        runtime,
        kernel,
        &[
            (&input_buf, 0),
            (&weights_buf, 0),
            (&out_buf, 0),
            (&input_width_buf, 0),
            (&output_width_buf, 0),
            (&rms_buf, 0),
            (&epsilon_buf, 0),
        ],
        groups as usize,
    );
    runtime.read_f32(&out_buf, rows).unwrap()
}

#[test]
fn matvec_q4_64row_mv_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [32u32, 256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        for output_width in [31u32, 33u32, 96u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let reference = cpu_q4_matvec(&input, &weights, rows);
            let candidate = run_plain(&runtime, "matvec_q4_0_64row_mv", &input, &weights, rows);
            compare(
                &format!("matvec_q4_64row_mv input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matvec_q4_64row_mv_rms_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [32u32, 256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        for output_width in [31u32, 33u32, 96u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
            let reference = cpu_q4_matvec(&normalized, &weights, rows);
            let candidate = run_rms(
                &runtime,
                "matvec_q4_0_64row_mv_rms",
                &input,
                &weights,
                &rms_weight,
                rows,
            );
            compare(
                &format!("matvec_q4_64row_mv_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matvec_q6_64row_mv_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [256u32, 2048u32, 3584u32] {
        let blocks = (input_width / 256) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        for output_width in [31u32, 33u32, 96u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q6_weights(rows, blocks, &mut state);
            let reference = cpu_q6_matvec(&input, &weights, rows);
            let candidate = run_plain(&runtime, "matvec_q6_k_64row_mv", &input, &weights, rows);
            compare(
                &format!("matvec_q6_64row_mv input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matvec_q6_64row_mv_rms_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [256u32, 2048u32, 3584u32] {
        let blocks = (input_width / 256) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        for output_width in [31u32, 33u32, 96u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q6_weights(rows, blocks, &mut state);
            let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
            let reference = cpu_q6_matvec(&normalized, &weights, rows);
            let candidate = run_rms(
                &runtime,
                "matvec_q6_k_64row_mv_rms",
                &input,
                &weights,
                &rms_weight,
                rows,
            );
            compare(
                &format!("matvec_q6_64row_mv_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}
