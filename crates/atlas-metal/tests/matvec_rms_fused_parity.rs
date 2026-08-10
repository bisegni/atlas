//! Tolerance-based correctness check for the RMS-input `_rms` fused
//! multi-projection kernels (`matmul_q4_0_qkv_32row_mv_rms`,
//! `matmul_q4_0_gate_up_32row_mv_rms`) against an independent CPU oracle:
//! the same RMS normalization the fusion folds into the dispatch, followed by
//! `dequantize_block`-based dot products for each output projection.
//! Asserting max-abs < 1e-3 (same contract as `matvec_mv_ext_parity.rs`)
//! proves the fusion is numerically equivalent to normalize-then-matvec
//! without requiring bitwise stream parity.

use atlas_core::{GgufTensorType, dequantize_block};
use atlas_metal::{MetalError, MetalRuntime};

const THREADS: usize = 128;
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

fn cpu_rms_normalize(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
    let mean_sq = input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32;
    let inverse_rms = 1.0 / (mean_sq + epsilon).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * w * inverse_rms)
        .collect()
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
        "{label}: fused output diverges from the CPU oracle (max_abs={max_abs:.3e})"
    );
}

fn dispatch(
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

#[test]
fn qkv_32row_mv_rms_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [128u32, 256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
        for (q_width, kv_width) in [
            (96u32, 96u32),
            (128u32, 128u32),
            (512u32, 128u32),
            (513u32, 129u32),
        ] {
            let q_weights = build_q4_weights(q_width as usize, blocks, &mut state);
            let k_weights = build_q4_weights(kv_width as usize, blocks, &mut state);
            let v_weights = build_q4_weights(kv_width as usize, blocks, &mut state);
            let q_ref = cpu_q4_matvec(&normalized, &q_weights, q_width as usize);
            let k_ref = cpu_q4_matvec(&normalized, &k_weights, kv_width as usize);
            let v_ref = cpu_q4_matvec(&normalized, &v_weights, kv_width as usize);

            let input_buf = runtime.upload_f32(&input).unwrap();
            let q_buf = runtime.upload_bytes(&q_weights).unwrap();
            let k_buf = runtime.upload_bytes(&k_weights).unwrap();
            let v_buf = runtime.upload_bytes(&v_weights).unwrap();
            let q_out_buf = runtime.upload_f32(&vec![0.0f32; q_width as usize]).unwrap();
            let k_out_buf = runtime
                .upload_f32(&vec![0.0f32; kv_width as usize])
                .unwrap();
            let v_out_buf = runtime
                .upload_f32(&vec![0.0f32; kv_width as usize])
                .unwrap();
            let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
            let q_width_buf = runtime.upload_u32(&[q_width]).unwrap();
            let kv_width_buf = runtime.upload_u32(&[kv_width]).unwrap();
            let rms_buf = runtime.upload_f32(&rms_weight).unwrap();
            let epsilon_buf = runtime.upload_f32(&[EPSILON]).unwrap();
            let groups = ((q_width + 31) / 32 + 2 * ((kv_width + 31) / 32)) as usize;
            dispatch(
                &runtime,
                "matmul_q4_0_qkv_32row_mv_rms",
                &[
                    (&input_buf, 0),
                    (&q_buf, 0),
                    (&k_buf, 0),
                    (&v_buf, 0),
                    (&q_out_buf, 0),
                    (&k_out_buf, 0),
                    (&v_out_buf, 0),
                    (&input_width_buf, 0),
                    (&q_width_buf, 0),
                    (&kv_width_buf, 0),
                    (&rms_buf, 0),
                    (&epsilon_buf, 0),
                ],
                groups,
            );
            let q_got = runtime.read_f32(&q_out_buf, q_width as usize).unwrap();
            let k_got = runtime.read_f32(&k_out_buf, kv_width as usize).unwrap();
            let v_got = runtime.read_f32(&v_out_buf, kv_width as usize).unwrap();
            compare(
                &format!("qkv q input={input_width} q={q_width} kv={kv_width}"),
                &q_ref,
                &q_got,
            );
            compare(
                &format!("qkv k input={input_width} q={q_width} kv={kv_width}"),
                &k_ref,
                &k_got,
            );
            compare(
                &format!("qkv v input={input_width} q={q_width} kv={kv_width}"),
                &v_ref,
                &v_got,
            );
        }
    }
}

#[test]
fn gate_up_32row_mv_rms_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [128u32, 256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
        for output_width in [96u32, 512u32, 513u32, 129u32] {
            let gate_weights = build_q4_weights(output_width as usize, blocks, &mut state);
            let up_weights = build_q4_weights(output_width as usize, blocks, &mut state);
            let gate_ref = cpu_q4_matvec(&normalized, &gate_weights, output_width as usize);
            let up_ref = cpu_q4_matvec(&normalized, &up_weights, output_width as usize);

            let input_buf = runtime.upload_f32(&input).unwrap();
            let gate_buf = runtime.upload_bytes(&gate_weights).unwrap();
            let up_buf = runtime.upload_bytes(&up_weights).unwrap();
            let gate_out_buf = runtime
                .upload_f32(&vec![0.0f32; output_width as usize])
                .unwrap();
            let up_out_buf = runtime
                .upload_f32(&vec![0.0f32; output_width as usize])
                .unwrap();
            let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
            let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();
            let rms_buf = runtime.upload_f32(&rms_weight).unwrap();
            let epsilon_buf = runtime.upload_f32(&[EPSILON]).unwrap();
            let groups = 2 * ((output_width + 31) / 32) as usize;
            dispatch(
                &runtime,
                "matmul_q4_0_gate_up_32row_mv_rms",
                &[
                    (&input_buf, 0),
                    (&gate_buf, 0),
                    (&up_buf, 0),
                    (&gate_out_buf, 0),
                    (&up_out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&rms_buf, 0),
                    (&epsilon_buf, 0),
                ],
                groups,
            );
            let gate_got = runtime
                .read_f32(&gate_out_buf, output_width as usize)
                .unwrap();
            let up_got = runtime
                .read_f32(&up_out_buf, output_width as usize)
                .unwrap();
            compare(
                &format!("gate_up gate input={input_width} output={output_width}"),
                &gate_ref,
                &gate_got,
            );
            compare(
                &format!("gate_up up input={input_width} output={output_width}"),
                &up_ref,
                &up_got,
            );
        }
    }
}
