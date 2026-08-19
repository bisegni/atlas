//! Correctness for the llama.cpp-current 16-row-per-threadgroup q4_0 matvec
//! kernels (`matvec_q4_0_16row_mv[_rms]`, phase-13.15 follow-up).  Two
//! contracts:
//!
//! 1. tolerance parity vs an independent CPU oracle (same contract as
//!    `matvec_mv_ext_parity.rs`), including partial-width rows that exercise
//!    the active-row paths;
//! 2. bitwise identity against the production 64-row kernels on identical
//!    inputs: the 16-row kernels keep the exact per-lane block stride,
//!    y-cache scale order, and block-dot accumulation order of the 64-row
//!    family, differing only in the 4-row simdgroup band size, so every row's
//!    value must match exactly.

use atlas_core::{GgufTensorType, dequantize_block};
use atlas_metal::{MetalError, MetalRuntime};

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

fn build_q4_weights(rows: usize, blocks: usize, _state: &mut u32) -> Vec<u8> {
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

fn assert_bitwise(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    let mut diffs = 0usize;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        if r.to_bits() != c.to_bits() {
            diffs += 1;
        }
    }
    assert_eq!(
        diffs, 0,
        "{label}: 16-row and 64-row kernels diverge bitwise in {diffs} rows"
    );
}

fn dispatch(
    runtime: &MetalRuntime,
    kernel: &'static str,
    buffers: &[(&atlas_metal::GpuBuffer, usize)],
    groups: usize,
    threads: usize,
) {
    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(kernel, buffers, groups, threads)
        .unwrap();
    command.finish().unwrap();
}

fn run_plain(
    runtime: &MetalRuntime,
    kernel: &'static str,
    sixteen_row: bool,
    input: &[f32],
    weights: &[u8],
    rows: usize,
) -> Vec<f32> {
    let input_buf = runtime.upload_f32(input).unwrap();
    let weights_buf = runtime.upload_bytes(weights).unwrap();
    let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[rows as u32]).unwrap();
    let (groups, threads) = if sixteen_row {
        ((rows + 15) / 16, 128)
    } else {
        ((rows + 63) / 64, 256)
    };
    dispatch(
        runtime,
        kernel,
        &[
            (&input_buf, 0),
            (&weights_buf, 0),
            (&out_buf, 0),
            (&input_width_buf, 0),
            (&output_width_buf, 0),
        ],
        groups,
        threads,
    );
    runtime.read_f32(&out_buf, rows).unwrap()
}

fn run_rms(
    runtime: &MetalRuntime,
    kernel: &'static str,
    sixteen_row: bool,
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
    let (groups, threads) = if sixteen_row {
        ((rows + 15) / 16, 128)
    } else {
        ((rows + 63) / 64, 256)
    };
    dispatch(
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
        groups,
        threads,
    );
    runtime.read_f32(&out_buf, rows).unwrap()
}

fn run_qkv_fused(
    runtime: &MetalRuntime,
    kernel: &'static str,
    sixteen_row: bool,
    input: &[f32],
    q_weights: &[u8],
    k_weights: &[u8],
    v_weights: &[u8],
    rms_weight: &[f32],
    q_rows: usize,
    kv_rows: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let input_buf = runtime.upload_f32(input).unwrap();
    let q_w = runtime.upload_bytes(q_weights).unwrap();
    let k_w = runtime.upload_bytes(k_weights).unwrap();
    let v_w = runtime.upload_bytes(v_weights).unwrap();
    let q_out = runtime.upload_f32(&vec![0.0f32; q_rows]).unwrap();
    let k_out = runtime.upload_f32(&vec![0.0f32; kv_rows]).unwrap();
    let v_out = runtime.upload_f32(&vec![0.0f32; kv_rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let q_width_buf = runtime.upload_u32(&[q_rows as u32]).unwrap();
    let kv_width_buf = runtime.upload_u32(&[kv_rows as u32]).unwrap();
    let rms_buf = runtime.upload_f32(rms_weight).unwrap();
    let epsilon_buf = runtime.upload_f32(&[EPSILON]).unwrap();
    let per_group = if sixteen_row { 16 } else { 32 };
    let groups = q_rows.div_ceil(per_group) + 2 * kv_rows.div_ceil(per_group);
    dispatch(
        runtime,
        kernel,
        &[
            (&input_buf, 0),
            (&q_w, 0),
            (&k_w, 0),
            (&v_w, 0),
            (&q_out, 0),
            (&k_out, 0),
            (&v_out, 0),
            (&input_width_buf, 0),
            (&q_width_buf, 0),
            (&kv_width_buf, 0),
            (&rms_buf, 0),
            (&epsilon_buf, 0),
        ],
        groups,
        128,
    );
    (
        runtime.read_f32(&q_out, q_rows).unwrap(),
        runtime.read_f32(&k_out, kv_rows).unwrap(),
        runtime.read_f32(&v_out, kv_rows).unwrap(),
    )
}

fn run_gate_up_fused(
    runtime: &MetalRuntime,
    kernel: &'static str,
    sixteen_row: bool,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    rms_weight: &[f32],
    rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    let input_buf = runtime.upload_f32(input).unwrap();
    let gate_w = runtime.upload_bytes(gate_weights).unwrap();
    let up_w = runtime.upload_bytes(up_weights).unwrap();
    let gate_out = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let up_out = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[rows as u32]).unwrap();
    let rms_buf = runtime.upload_f32(rms_weight).unwrap();
    let epsilon_buf = runtime.upload_f32(&[EPSILON]).unwrap();
    let per_group = if sixteen_row { 16 } else { 32 };
    let groups = 2 * rows.div_ceil(per_group);
    dispatch(
        runtime,
        kernel,
        &[
            (&input_buf, 0),
            (&gate_w, 0),
            (&up_w, 0),
            (&gate_out, 0),
            (&up_out, 0),
            (&input_width_buf, 0),
            (&output_width_buf, 0),
            (&rms_buf, 0),
            (&epsilon_buf, 0),
        ],
        groups,
        128,
    );
    (
        runtime.read_f32(&gate_out, rows).unwrap(),
        runtime.read_f32(&up_out, rows).unwrap(),
    )
}

fn runtime_or_skip() -> Option<MetalRuntime> {
    match MetalRuntime::new() {
        Ok(runtime) => Some(runtime),
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            None
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    }
}

#[test]
fn matvec_q4_16row_mv_matches_cpu_oracle() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [32u32, 256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        for output_width in [16u32, 31u32, 33u32, 96u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let reference = cpu_q4_matvec(&input, &weights, rows);
            let candidate = run_plain(
                &runtime,
                "matvec_q4_0_16row_mv",
                true,
                &input,
                &weights,
                rows,
            );
            compare(
                &format!("matvec_q4_16row_mv input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matvec_q4_16row_mv_rms_matches_cpu_oracle() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [32u32, 256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        for output_width in [16u32, 31u32, 33u32, 96u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
            let reference = cpu_q4_matvec(&normalized, &weights, rows);
            let candidate = run_rms(
                &runtime,
                "matvec_q4_0_16row_mv_rms",
                true,
                &input,
                &weights,
                &rms_weight,
                rows,
            );
            compare(
                &format!("matvec_q4_16row_mv_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matmul_q4_16row_qkv_fused_matches_cpu_oracle() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [32u32, 256u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        for (q_rows, kv_rows) in [(33usize, 17usize), (129usize, 33usize)] {
            let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
            let q = build_q4_weights(q_rows, blocks, &mut state);
            let k = build_q4_weights(kv_rows, blocks, &mut state);
            let v = build_q4_weights(kv_rows, blocks, &mut state);
            let (gpu_q, gpu_k, gpu_v) = run_qkv_fused(
                &runtime,
                "matmul_q4_0_qkv_16row_mv_rms",
                true,
                &input,
                &q,
                &k,
                &v,
                &rms_weight,
                q_rows,
                kv_rows,
            );
            compare(
                &format!("qkv16 q={q_rows} kv={kv_rows} in={input_width}"),
                &cpu_q4_matvec(&normalized, &q, q_rows),
                &gpu_q,
            );
            compare(
                &format!("qkv16 k={kv_rows} in={input_width}"),
                &cpu_q4_matvec(&normalized, &k, kv_rows),
                &gpu_k,
            );
            compare(
                &format!("qkv16 v={kv_rows} in={input_width}"),
                &cpu_q4_matvec(&normalized, &v, kv_rows),
                &gpu_v,
            );
        }
    }
}

#[test]
fn matmul_q4_16row_gate_up_fused_matches_cpu_oracle() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [32u32, 256u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        for rows in [33usize, 97usize, 137usize] {
            let normalized = cpu_rms_normalize(&input, &rms_weight, EPSILON);
            let gate = build_q4_weights(rows, blocks, &mut state);
            let up = build_q4_weights(rows, blocks, &mut state);
            let (gpu_gate, gpu_up) = run_gate_up_fused(
                &runtime,
                "matmul_q4_0_gate_up_16row_mv_rms",
                true,
                &input,
                &gate,
                &up,
                &rms_weight,
                rows,
            );
            compare(
                &format!("gate_up16 gate rows={rows} in={input_width}"),
                &cpu_q4_matvec(&normalized, &gate, rows),
                &gpu_gate,
            );
            compare(
                &format!("gate_up16 up rows={rows} in={input_width}"),
                &cpu_q4_matvec(&normalized, &up, rows),
                &gpu_up,
            );
        }
    }
}

#[test]
fn matmul_q4_16row_fused_is_bitwise_identical_to_32row() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };
    let mut state = 0x9e37_79b9u32;
    let input_width = 256u32;
    let blocks = (input_width / 32) as usize;
    let mut input = vec![0.0f32; input_width as usize];
    fill_f32(&mut input, &mut state);
    let mut rms_weight = vec![0.0f32; input_width as usize];
    fill_rms_weight(&mut rms_weight, &mut state);
    for (q_rows, kv_rows) in [
        (31usize, 17usize),
        (129usize, 33usize),
        (512usize, 256usize),
    ] {
        let q = build_q4_weights(q_rows, blocks, &mut state);
        let k = build_q4_weights(kv_rows, blocks, &mut state);
        let v = build_q4_weights(kv_rows, blocks, &mut state);
        let (q32, k32, v32) = run_qkv_fused(
            &runtime,
            "matmul_q4_0_qkv_32row_mv_rms",
            false,
            &input,
            &q,
            &k,
            &v,
            &rms_weight,
            q_rows,
            kv_rows,
        );
        let (q16, k16, v16) = run_qkv_fused(
            &runtime,
            "matmul_q4_0_qkv_16row_mv_rms",
            true,
            &input,
            &q,
            &k,
            &v,
            &rms_weight,
            q_rows,
            kv_rows,
        );
        assert_bitwise(&format!("qkv16_vs_32 q={q_rows} kv={kv_rows}"), &q32, &q16);
        assert_bitwise(&format!("qkv16_vs_32 k={kv_rows}"), &k32, &k16);
        assert_bitwise(&format!("qkv16_vs_32 v={kv_rows}"), &v32, &v16);
    }
    for rows in [31usize, 97usize, 512usize] {
        let gate = build_q4_weights(rows, blocks, &mut state);
        let up = build_q4_weights(rows, blocks, &mut state);
        let (g32, u32) = run_gate_up_fused(
            &runtime,
            "matmul_q4_0_gate_up_32row_mv_rms",
            false,
            &input,
            &gate,
            &up,
            &rms_weight,
            rows,
        );
        let (g16, u16) = run_gate_up_fused(
            &runtime,
            "matmul_q4_0_gate_up_16row_mv_rms",
            true,
            &input,
            &gate,
            &up,
            &rms_weight,
            rows,
        );
        assert_bitwise(&format!("gate_up16_vs_32 gate rows={rows}"), &g32, &g16);
        assert_bitwise(&format!("gate_up16_vs_32 up rows={rows}"), &u32, &u16);
    }
}

#[test]
fn matvec_q4_16row_mv_is_bitwise_identical_to_64row() {
    let Some(runtime) = runtime_or_skip() else {
        return;
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [256u32, 2048u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        for output_width in [31u32, 33u32, 96u32, 129u32, 137u32, 2048u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let row64 = run_plain(
                &runtime,
                "matvec_q4_0_64row_mv",
                false,
                &input,
                &weights,
                rows,
            );
            let row16 = run_plain(
                &runtime,
                "matvec_q4_0_16row_mv",
                true,
                &input,
                &weights,
                rows,
            );
            assert_bitwise(
                &format!("matvec_q4_16row_vs_64row input={input_width} output={output_width}"),
                &row64,
                &row16,
            );
            let rms64 = run_rms(
                &runtime,
                "matvec_q4_0_64row_mv_rms",
                false,
                &input,
                &weights,
                &rms_weight,
                rows,
            );
            let rms16 = run_rms(
                &runtime,
                "matvec_q4_0_16row_mv_rms",
                true,
                &input,
                &weights,
                &rms_weight,
                rows,
            );
            assert_bitwise(
                &format!(
                    "matvec_q4_16row_rms_vs_64row_rms input={input_width} output={output_width}"
                ),
                &rms64,
                &rms16,
            );
        }
    }
}
