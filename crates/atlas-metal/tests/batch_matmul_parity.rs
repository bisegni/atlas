//! Tolerance-based correctness check for the production batched Q4_0
//! prefill projection kernel (`matmul_q4_0_batch_16row`) against an
//! independent CPU oracle: `dequantize_block` from atlas-core plus a plain
//! per-token dot product. The kernel reduces each output row across a
//! simdgroup with shuffle reductions, so this asserts a small
//! max-absolute-difference (same contract as `matvec_mv_ext_parity.rs`).
//! Covers the Gemma4 E2B geometries (q/k/v, gate/up, ffn-down, PLE) plus
//! partial batches and partial-width rows.

use atlas_core::{GgufTensorType, dequantize_block};
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

fn cpu_batch_matmul(input: &[f32], weights: &[u8], rows: usize, batch: usize) -> Vec<f32> {
    let blocks = (input.len() / batch) / 32;
    let mut block = vec![0.0f32; 32];
    let mut out = vec![0.0f32; batch * rows];
    for token in 0..batch {
        let token_input = &input[token * blocks * 32..(token + 1) * blocks * 32];
        for row in 0..rows {
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let chunk = &weights[(row * blocks + b) * 18..(row * blocks + b + 1) * 18];
                dequantize_block(GgufTensorType::Q4_0, chunk, &mut block).unwrap();
                for lane in 0..32 {
                    acc += token_input[b * 32 + lane] * block[lane];
                }
            }
            out[token * rows + row] = acc;
        }
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
        "{label}: batch output diverges from the CPU oracle (max_abs={max_abs:.3e})"
    );
}

#[test]
fn batch_16row_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    // (batch, input_width, output_width) covering the Gemma4 E2B
    // projections: q/k/v, gate/up, ffn-down, PLE, plus partial batches and
    // partial-width rows.
    for (batch, input_width, output_width) in [
        (8u32, 128u32, 128u32),
        (8u32, 256u32, 512u32),
        (8u32, 2048u32, 128u32),
        (16u32, 2048u32, 2304u32),
        (2u32, 2304u32, 2048u32),
        (3u32, 2048u32, 137u32),
        (1u32, 512u32, 96u32),
        (8u32, 256u32, 129u32),
    ] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; (batch * input_width) as usize];
        fill_f32(&mut input, &mut state);
        let weights = build_q4_weights(output_width as usize, blocks, &mut state);
        let reference = cpu_batch_matmul(&input, &weights, output_width as usize, batch as usize);

        let input_buf = runtime.upload_f32(&input).unwrap();
        let weights_buf = runtime.upload_bytes(&weights).unwrap();
        let out_buf = runtime
            .upload_f32(&vec![0.0f32; (batch * output_width) as usize])
            .unwrap();
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();
        let batch_buf = runtime.upload_u32(&[batch]).unwrap();
        let rows_per_batch = ((output_width + 15) / 16) as usize;
        let groups = batch as usize * rows_per_batch;
        let mut command = runtime.begin_resident_command().unwrap();
        command
            .dispatch_threadgroups_1d_at(
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
                THREADS,
            )
            .unwrap();
        command.finish().unwrap();
        let candidate = runtime
            .read_f32(&out_buf, (batch * output_width) as usize)
            .unwrap();
        compare(
            &format!("batch_16row batch={batch} input={input_width} output={output_width}"),
            &reference,
            &candidate,
        );
    }
}

#[test]
fn batch_32row_matches_batch_16row_within_tolerance() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x2f6b_a1c9u32;
    // Same geometries as the CPU-oracle test plus the real 2304/16384 shapes.
    // The weight-stationary tile interleaves four per-token accumulator chains,
    // which changes Metal's instruction selection by ~1 ulp (measured max-abs
    // ~4.7e-10) while keeping each token's block-sequential accumulation order.
    // This is well inside the phase contract (max-abs < 1e-3) shared with the
    // CPU-oracle test; the greedy stream hash on the matched pp512 workload was
    // verified byte-identical (`f23c2962…`).
    for (batch, input_width, output_width) in [
        (8u32, 128u32, 128u32),
        (8u32, 256u32, 512u32),
        (8u32, 2048u32, 128u32),
        (16u32, 2048u32, 2304u32),
        (4u32, 2304u32, 4096u32),
        (2u32, 16384u32, 2304u32),
        (3u32, 2048u32, 137u32),
        (1u32, 512u32, 96u32),
    ] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; (batch * input_width) as usize];
        fill_f32(&mut input, &mut state);
        let weights = build_q4_weights(output_width as usize, blocks, &mut state);
        let input_buf = runtime.upload_f32(&input).unwrap();
        let weights_buf = runtime.upload_bytes(&weights).unwrap();
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();
        let batch_buf = runtime.upload_u32(&[batch]).unwrap();
        let output_len = (batch * output_width) as usize;

        let run =
            |kernel: &'static str, tile_tokens: usize, rows_per_group: usize, threads: usize| {
                let out_buf = runtime.upload_f32(&vec![0.0f32; output_len]).unwrap();
                let mut command = runtime.begin_resident_command().unwrap();
                command
                    .dispatch_threadgroups_1d_at(
                        kernel,
                        &[
                            (&input_buf, 0),
                            (&weights_buf, 0),
                            (&out_buf, 0),
                            (&input_width_buf, 0),
                            (&output_width_buf, 0),
                            (&batch_buf, 0),
                        ],
                        (batch as usize).div_ceil(tile_tokens)
                            * (output_width as usize).div_ceil(rows_per_group),
                        threads,
                    )
                    .unwrap();
                command.finish().unwrap();
                runtime.read_f32(&out_buf, output_len).unwrap()
            };

        let reference = run("matmul_q4_0_batch_16row", 1, 16, 128);
        let candidate = run("matmul_q4_0_batch_32row", 4, 32, 256);
        compare(
            &format!("batch_32row batch={batch} input={input_width} output={output_width}"),
            &reference,
            &candidate,
        );
    }
}
