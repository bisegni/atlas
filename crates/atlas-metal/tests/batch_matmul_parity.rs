//! Tolerance-based correctness check for the token-tiled batched Q4_0
//! projection kernel (`matmul_q4_0_batch_16row_token_tiled`) against the
//! proven per-token 16-row batch kernel (`matmul_q4_0_batch_16row`). The
//! tiled kernel stages the input tile and a 16-row weight block in
//! threadgroup memory and reuses them across eight prompt tokens, changing
//! the accumulation order (per-token accumulators instead of one sequential
//! pass), so this asserts a small max-absolute-difference (same contract as
//! `matvec_mv_ext_parity.rs`). Covers the Gemma4 E2B geometries (q/k/v,
//! gate/up, ffn-down, PLE) plus partial batches and partial-width rows.

use atlas_metal::{MetalError, MetalRuntime};

const THREADS: usize = 128;

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
        "{label}: token-tiled batch output diverges from the production batch kernel (max_abs={max_abs:.3e})"
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
fn token_tiled_batch_matches_16row() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x243f_6a88u32;
    for (input_width, output_width) in [
        (2048u32, 2304u32),
        (2048u32, 14336u32),
        (2048u32, 2048u32),
        (256u32, 137u32),
    ] {
        let blocks = (input_width / 32) as usize;
        let rows = output_width as usize;
        let weights = build_q4_weights(rows, blocks, &mut state);
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();
        for batch in [1usize, 8, 17, 25, 59] {
            let mut input = vec![0.0f32; batch * input_width as usize];
            fill_f32(&mut input, &mut state);
            let input_buf = runtime.upload_f32(&input).unwrap();
            let weights_buf = runtime.upload_bytes(&weights).unwrap();
            let batch_buf = runtime.upload_u32(&[batch as u32]).unwrap();
            let output = vec![0.0f32; batch * rows];

            let out_buf = runtime.upload_f32(&output).unwrap();
            let groups = batch * (rows + 15) / 16;
            dispatch(
                &runtime,
                "matmul_q4_0_batch_16row",
                &[
                    (&input_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&batch_buf, 0),
                ],
                groups,
            );
            let reference = runtime.read_f32(&out_buf, batch * rows).unwrap();

            let out_buf = runtime.upload_f32(&output).unwrap();
            let groups = (batch + 7) / 8 * (rows + 15) / 16;
            dispatch(
                &runtime,
                "matmul_q4_0_batch_16row_token_tiled",
                &[
                    (&input_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&batch_buf, 0),
                ],
                groups,
            );
            let candidate = runtime.read_f32(&out_buf, batch * rows).unwrap();
            compare(
                &format!("batch_tiled input={input_width} output={output_width} batch={batch}"),
                &reference,
                &candidate,
            );
        }
    }
}
