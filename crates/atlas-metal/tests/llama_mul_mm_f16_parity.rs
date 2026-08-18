//! Tolerance-based correctness check for the vendored llama.cpp
//! `llama_mul_mm_f16_f32` kernel (llama.cpp's `kernel_mul_mm_f16_f32`: the
//! same simdgroup-matrix `kernel_mul_mm` core as the q4_0 variant, but the
//! A operand is fp16 with no dequantization — a direct `half4x4` load). It
//! computes C[N x M] = A[M x K](f16) @ B[N x K](f32)^T, i.e. for each token n
//! and output row m, sum_k weight[m][k] * activation[n][k].
//!
//! Because the kernel feeds fp16 fragments to the matrix units (llama.cpp's
//! own accuracy level), parity is asserted with a RELATIVE tolerance (~5e-2)
//! against the fp32 CPU oracle — the repo's fp16 mul_mm contract.

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

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 112u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// CPU oracle: out[n*M + m] = sum_k f16_weight[m*K + k] * activation[n*K + k].
fn cpu_mul_mm(activation: &[f32], weights_f16: &[u16], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * m];
    for token in 0..n {
        let token_input = &activation[token * k..(token + 1) * k];
        for row in 0..m {
            let wrow = &weights_f16[row * k..(row + 1) * k];
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += token_input[col] * half_to_f32(wrow[col]);
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
    assert!(
        max_rel < 5e-2,
        "{label}: llama_mul_mm_f16 diverges from the CPU oracle (max_rel={max_rel:.3e}, max_abs={max_abs:.3e})"
    );
}

#[test]
fn llama_mul_mm_f16_f32_matches_cpu_oracle_relative() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x51f0_4a2bu32;
    // K must be a multiple of 32 (the kernel's NK tiling and the aligned
    // half4x4 weight load require it). Aligned and partial (bounds-checked)
    // M/N tiles are both exercised.
    for (n, k, m) in [
        (8usize, 128usize, 64usize),
        (32usize, 512usize, 128usize),
        (5usize, 512usize, 100usize),
        (33usize, 1536usize, 256usize),
        (16usize, 1536usize, 8960usize),
    ] {
        let mut weight_f32 = vec![0.0f32; m * k];
        fill_f32(&mut weight_f32, &mut state);
        let weights_f16: Vec<u16> = weight_f32.iter().map(|&v| half_bits_of(v)).collect();
        let weight_bytes: Vec<u8> = weights_f16.iter().flat_map(|&h| h.to_le_bytes()).collect();
        let mut activation = vec![0.0f32; n * k];
        fill_f32(&mut activation, &mut state);
        let reference = cpu_mul_mm(&activation, &weights_f16, m, k, n);

        let src0 = runtime.upload_bytes(&weight_bytes).unwrap();
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
                "llama_mul_mm_f16_f32",
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
            &format!("llama_mul_mm_f16 N={n} K={k} M={m}"),
            &reference,
            &candidate,
        );
    }
}
