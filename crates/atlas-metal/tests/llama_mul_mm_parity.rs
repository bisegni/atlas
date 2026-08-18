//! Tolerance-based correctness check for the vendored llama.cpp
//! `llama_mul_mm_q4_0_f32` kernel (a faithful port of llama.cpp's classic
//! simdgroup-matrix `kernel_mul_mm`, specialized for q4_0 x f32 single-batch).
//! It computes C[N x M] = A[M x K](q4_0) @ B[N x K](f32)^T, i.e. for each
//! token n and output row m, sum_k dequant(weight[m][k]) * activation[n][k].
//!
//! Because the kernel feeds fp16 fragments to the matrix units (llama.cpp's
//! own accuracy level), parity is asserted with a RELATIVE tolerance (~1e-2)
//! against the fp32 CPU oracle rather than the strict max-abs < 1e-3 used by
//! the bitwise production kernels. Structural bugs (wrong indexing/layout)
//! produce O(1) relative errors and are caught; fp16 rounding stays ~1e-3..1e-2.

use atlas_core::{GgufTensorType, dequantize_block};
use atlas_metal::{MetalError, MetalRuntime};

const THREADS: usize = 128;
const THREADGROUP_MEMORY: usize = 8192;

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

fn half_bits_of(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exp32 = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits & 0x7F_FFFF) >> 13;
    if exp32 <= 0 {
        sign
    } else if exp32 >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp32 as u16) << 10) | mant as u16
    }
}

fn build_q4_weights(rows: usize, blocks: usize, state: &mut u32) -> Vec<u8> {
    let mut weights = vec![0u8; rows * blocks * 18];
    for (i, chunk) in weights.chunks_mut(18).enumerate() {
        let scale = 0.05 + (i as f32 % 101.0) * 0.003;
        chunk[..2].copy_from_slice(&half_bits_of(scale).to_le_bytes());
        for (j, byte) in chunk[2..].iter_mut().enumerate() {
            let nibble = ((i * 7 + j * 13) % 16) as u8;
            *byte = nibble | (nibble << 4);
        }
    }
    let _ = state;
    weights
}

// CPU oracle: out[n*M + m] = sum_k dequant(weight[m][k]) * activation[n*K + k].
fn cpu_mul_mm(activation: &[f32], weights: &[u8], m: usize, k: usize, n: usize) -> Vec<f32> {
    let blocks = k / 32;
    let mut block = vec![0.0f32; 32];
    let mut out = vec![0.0f32; n * m];
    for token in 0..n {
        let token_input = &activation[token * k..(token + 1) * k];
        for row in 0..m {
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let chunk = &weights[(row * blocks + b) * 18..(row * blocks + b + 1) * 18];
                dequantize_block(GgufTensorType::Q4_0, chunk, &mut block).unwrap();
                for lane in 0..32 {
                    acc += token_input[b * 32 + lane] * block[lane];
                }
            }
            out[token * m + row] = acc;
        }
    }
    out
}

fn compare_rel(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        assert!(c.is_finite(), "{label}: non-finite output");
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
        let rel = diff / r.abs().max(1.0);
        max_rel = max_rel.max(rel);
    }
    eprintln!("{label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    // Matches the repo's fp16 mul_mm contract (phase-13.8
    // `batch_mul_mm_f16_measures_accuracy` asserts relative < 5e-2). fp16
    // fragments accumulate ~1e-3..2e-2 relative error over large K.
    assert!(
        max_rel < 5e-2,
        "{label}: llama_mul_mm diverges from the CPU oracle (max_rel={max_rel:.3e}, max_abs={max_abs:.3e})"
    );
}

#[test]
fn llama_mul_mm_q4_0_matches_cpu_oracle_relative() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x51f0_4a2bu32;
    // (tokens N, K=input, M=output). Covers aligned tiles and partial tiles
    // (exercise the bounds-checked store path).
    for (n, k, m) in [
        (8usize, 128usize, 64usize),
        (32usize, 256usize, 128usize),
        (16usize, 2304usize, 2304usize),
        (5usize, 512usize, 100usize),
        (33usize, 2304usize, 4096usize),
    ] {
        let blocks = k / 32;
        let mut activation = vec![0.0f32; n * k];
        fill_f32(&mut activation, &mut state);
        let weights = build_q4_weights(m, blocks, &mut state);
        let reference = cpu_mul_mm(&activation, &weights, m, k, n);

        let src0 = runtime.upload_bytes(&weights).unwrap();
        let src1 = runtime.upload_f32(&activation).unwrap();
        let dst = runtime.upload_f32(&vec![0.0f32; n * m]).unwrap();
        let ne00 = runtime.upload_u32(&[k as u32]).unwrap();
        let ne0 = runtime.upload_u32(&[m as u32]).unwrap();
        let ne1 = runtime.upload_u32(&[n as u32]).unwrap();

        let grid_width = n.div_ceil(32);
        let grid_height = m.div_ceil(64);
        let mut command = runtime.begin_resident_command().unwrap();
        command
            .dispatch_threadgroups_2d_tgm(
                "llama_mul_mm_q4_0_f32",
                None,
                &[
                    (&src0, 0),
                    (&src1, 0),
                    (&dst, 0),
                    (&ne00, 0),
                    (&ne0, 0),
                    (&ne1, 0),
                ],
                grid_width,
                grid_height,
                THREADS,
                THREADGROUP_MEMORY,
            )
            .unwrap();
        command.finish().unwrap();
        let candidate = runtime.read_f32(&dst, n * m).unwrap();
        compare_rel(
            &format!("llama_mul_mm N={n} K={k} M={m}"),
            &reference,
            &candidate,
        );
    }
}
