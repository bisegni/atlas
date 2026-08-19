//! Tolerance-based correctness check for the phase-13.0 P2a dispatch-fusion
//! kernels that already exist in production but were never given dedicated
//! parity coverage: the FFN `gelu_multiply_f32`, the PLE
//! `ple_gelu_multiply_offset_f32`, and the Gemma decode epilogue
//! `gemma4_rms_residual_f32`. Each candidate is compared against the
//! exact unfused reference path the executor uses when the fusion is off
//! (gelu_f32 + vector_multiply_f32, gelu_f32 + vector_multiply_offset_f32,
//! rms_norm_decode_f32 + vector_add_f32) under the max-abs < 1e-3 phase
//! contract, and the achieved deltas are printed for the record.

use atlas_metal::{MetalError, MetalRuntime};

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

fn compare(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f32;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        assert!(
            c.is_finite(),
            "{label}: fused kernel produced non-finite output"
        );
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
        mean_abs += diff;
    }
    mean_abs /= reference.len() as f32;
    eprintln!("{label}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
    assert!(
        max_abs < 1e-3,
        "{label}: fused kernel diverges from the unfused reference (max_abs={max_abs:.3e})"
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

fn dispatch_1d(
    runtime: &MetalRuntime,
    kernel: &'static str,
    buffers: &[&atlas_metal::GpuBuffer],
    count: usize,
) {
    let mut command = runtime.begin_resident_command().unwrap();
    command.dispatch_1d(kernel, buffers, count).unwrap();
    command.finish().unwrap();
}

#[test]
fn gelu_multiply_matches_gelu_then_multiply() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x853c_49e2u32;
    for count in [256u32, 2048u32, 8192u32] {
        let len = count as usize;
        let mut gate = vec![0.0f32; len];
        fill_f32(&mut gate, &mut state);
        let mut up = vec![0.0f32; len];
        fill_f32(&mut up, &mut state);
        let gate_buf = runtime.upload_f32(&gate).unwrap();
        let up_buf = runtime.upload_f32(&up).unwrap();
        let count_buf = runtime.upload_u32(&[count]).unwrap();
        let zero = vec![0.0f32; len];

        let activated = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "gelu_f32",
            &[&gate_buf, &activated, &count_buf],
            count as usize,
        );
        let product = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "vector_multiply_f32",
            &[&activated, &up_buf, &product, &count_buf],
            count as usize,
        );
        let reference = runtime.read_f32(&product, len).unwrap();

        let product = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "gelu_multiply_f32",
            &[&gate_buf, &up_buf, &product, &count_buf],
            count as usize,
        );
        let candidate = runtime.read_f32(&product, len).unwrap();
        compare(
            &format!("gelu_multiply count={count}"),
            &reference,
            &candidate,
        );
    }
}

#[test]
fn ple_gelu_multiply_offset_matches_gelu_then_offset_multiply() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x510e_527fu32;
    for count in [128u32, 256u32, 512u32] {
        let len = count as usize;
        let mut gate = vec![0.0f32; len];
        fill_f32(&mut gate, &mut state);
        let mut ple = vec![0.0f32; 4 * len];
        fill_f32(&mut ple, &mut state);
        let ple_offset = (next_u32(&mut state) % 4) * count;
        let gate_buf = runtime.upload_f32(&gate).unwrap();
        let ple_buf = runtime.upload_f32(&ple).unwrap();
        let count_buf = runtime.upload_u32(&[count]).unwrap();
        let offset_buf = runtime.upload_u32(&[ple_offset]).unwrap();
        let zero = vec![0.0f32; len];

        let activated = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "gelu_f32",
            &[&gate_buf, &activated, &count_buf],
            count as usize,
        );
        let out = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "vector_multiply_offset_f32",
            &[&activated, &ple_buf, &out, &offset_buf, &count_buf],
            count as usize,
        );
        let reference = runtime.read_f32(&out, len).unwrap();

        let out = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "ple_gelu_multiply_offset_f32",
            &[&gate_buf, &ple_buf, &out, &offset_buf, &count_buf],
            count as usize,
        );
        let candidate = runtime.read_f32(&out, len).unwrap();
        compare(
            &format!("ple_gelu_multiply_offset count={count} offset={ple_offset}"),
            &reference,
            &candidate,
        );
    }
}

#[test]
fn rms_residual_matches_rms_norm_then_add() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    let epsilon = 1e-6f32;
    for hidden in [512u32, 2048u32, 2304u32] {
        let len = hidden as usize;
        let mut input = vec![0.0f32; len];
        fill_f32(&mut input, &mut state);
        let mut weight = vec![0.0f32; len];
        fill_f32(&mut weight, &mut state);
        let mut residual = vec![0.0f32; len];
        fill_f32(&mut residual, &mut state);
        let input_buf = runtime.upload_f32(&input).unwrap();
        let weight_buf = runtime.upload_f32(&weight).unwrap();
        let residual_buf = runtime.upload_f32(&residual).unwrap();
        let hidden_buf = runtime.upload_u32(&[hidden]).unwrap();
        let eps_buf = runtime.upload_f32(&[epsilon]).unwrap();
        let zero = vec![0.0f32; len];

        let normalized = runtime.upload_f32(&zero).unwrap();
        dispatch(
            &runtime,
            "rms_norm_decode_f32_vec4",
            &[
                (&input_buf, 0),
                (&weight_buf, 0),
                (&normalized, 0),
                (&hidden_buf, 0),
                (&eps_buf, 0),
            ],
            1,
            32,
        );
        let output = runtime.upload_f32(&zero).unwrap();
        dispatch_1d(
            &runtime,
            "vector_add_f32",
            &[&residual_buf, &normalized, &output, &hidden_buf],
            hidden as usize,
        );
        let reference = runtime.read_f32(&output, len).unwrap();

        let normalized = runtime.upload_f32(&zero).unwrap();
        let output = runtime.upload_f32(&zero).unwrap();
        dispatch(
            &runtime,
            "gemma4_rms_residual_f32",
            &[
                (&input_buf, 0),
                (&weight_buf, 0),
                (&residual_buf, 0),
                (&normalized, 0),
                (&output, 0),
                (&hidden_buf, 0),
                (&eps_buf, 0),
            ],
            1,
            32,
        );
        let candidate = runtime.read_f32(&output, len).unwrap();
        compare(
            &format!("rms_residual hidden={hidden}"),
            &reference,
            &candidate,
        );
    }
}

#[test]
fn ple_rms_add_scale_matches_norm_add_then_scale() {
    // The decode PLE-per-layer epilogue is normally three dispatches:
    //   rms_norm_decode_f32_vec4(work, post_norm) -> normalized
    //   vector_add_f32(state, normalized)         -> state
    //   scalar_multiply_f32(state, layer_output_scale) -> state
    // gemma4_ple_rms_add_scale_f32 fuses all three (one dispatch). It must
    // match that exact reference path under the max-abs < 1e-3 phase contract.
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x3a5c_9491u32;
    let epsilon = 1e-6f32;
    for hidden in [512u32, 1280u32, 2304u32] {
        let len = hidden as usize;
        let mut input = vec![0.0f32; len];
        fill_f32(&mut input, &mut state);
        let mut weight = vec![0.0f32; len];
        fill_f32(&mut weight, &mut state);

        // reverse(0.25, 2.0) so the scale is not trivially 1.0
        let scale = (next_f32(&mut state) - 0.5) * 4.0 + 1.0;
        let input_buf = runtime.upload_f32(&input).unwrap();
        let weight_buf = runtime.upload_f32(&weight).unwrap();
        let hidden_buf = runtime.upload_u32(&[hidden]).unwrap();
        let eps_buf = runtime.upload_f32(&[epsilon]).unwrap();
        let scale_buf = runtime.upload_f32(&[scale]).unwrap();

        let zero = vec![0.0f32; len];

        // Reference: norm -> add -> scale
        let mut state_ref = vec![0.0f32; len];
        fill_f32(&mut state_ref, &mut state);
        let state_ref_buf = runtime.upload_f32(&state_ref).unwrap();
        let normalized = runtime.upload_f32(&zero).unwrap();
        dispatch(
            &runtime,
            "rms_norm_decode_f32_vec4",
            &[
                (&input_buf, 0),
                (&weight_buf, 0),
                (&normalized, 0),
                (&hidden_buf, 0),
                (&eps_buf, 0),
            ],
            1,
            32,
        );
        dispatch_1d(
            &runtime,
            "vector_add_f32",
            &[&state_ref_buf, &normalized, &state_ref_buf, &hidden_buf],
            hidden as usize,
        );
        dispatch_1d(
            &runtime,
            "scalar_multiply_f32",
            &[&state_ref_buf, &state_ref_buf, &scale_buf, &hidden_buf],
            hidden as usize,
        );
        let reference = runtime.read_f32(&state_ref_buf, len).unwrap();

        // Candidate: fused single dispatch (must start from the same state)
        let state_cand_buf = runtime.upload_f32(&state_ref).unwrap();
        let normalized = runtime.upload_f32(&zero).unwrap();
        dispatch(
            &runtime,
            "gemma4_ple_rms_add_scale_f32",
            &[
                (&input_buf, 0),
                (&weight_buf, 0),
                (&state_cand_buf, 0),
                (&normalized, 0),
                (&hidden_buf, 0),
                (&eps_buf, 0),
                (&scale_buf, 0),
            ],
            1,
            32,
        );
        let candidate = runtime.read_f32(&state_cand_buf, len).unwrap();
        compare(
            &format!("ple_rms_add_scale hidden={hidden} scale={scale}"),
            &reference,
            &candidate,
        );
    }
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
        "{label}: fused and split kernels diverge bitwise in {diffs} rows"
    );
}

fn build_q4_weights(rows: usize, blocks: usize, state: &mut u32) -> Vec<u8> {
    let mut weights = vec![0u8; rows * blocks * 18];
    for (i, chunk) in weights.chunks_mut(18).enumerate() {
        let scale = (0.005 + (i as f32 % 101.0) * 0.0003) as f32;
        let half = (scale * 32768.0).round() as u16;
        chunk[..2].copy_from_slice(&half.to_le_bytes());
        let mut nibble = next_u32(state) as u8;
        for byte in chunk[2..].iter_mut() {
            nibble = nibble.wrapping_mul(7).wrapping_add(13) % 16;
            *byte = nibble | (nibble << 4);
        }
    }
    weights
}

fn run_ple_gate_gelu_split(
    runtime: &MetalRuntime,
    input: &[f32],
    weights: &[u8],
    ple: &[f32],
    ple_offset: u32,
    rows: usize,
) -> Vec<f32> {
    let input_buf = runtime.upload_f32(input).unwrap();
    let weights_buf = runtime.upload_bytes(weights).unwrap();
    let ple_buf = runtime.upload_f32(ple).unwrap();
    let gate_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[rows as u32]).unwrap();
    let offset_buf = runtime.upload_u32(&[ple_offset]).unwrap();
    dispatch(
        runtime,
        "matvec_q4_0_16row_mv",
        &[
            (&input_buf, 0),
            (&weights_buf, 0),
            (&gate_buf, 0),
            (&input_width_buf, 0),
            (&output_width_buf, 0),
        ],
        (rows + 15) / 16,
        128,
    );
    dispatch_1d(
        runtime,
        "ple_gelu_multiply_offset_f32",
        &[
            &gate_buf,
            &ple_buf,
            &out_buf,
            &offset_buf,
            &output_width_buf,
        ],
        rows,
    );
    runtime.read_f32(&out_buf, rows).unwrap()
}

fn run_ple_gate_gelu_fused(
    runtime: &MetalRuntime,
    input: &[f32],
    weights: &[u8],
    ple: &[f32],
    ple_offset: u32,
    rows: usize,
) -> Vec<f32> {
    let input_buf = runtime.upload_f32(input).unwrap();
    let weights_buf = runtime.upload_bytes(weights).unwrap();
    let ple_buf = runtime.upload_f32(ple).unwrap();
    let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input.len() as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[rows as u32]).unwrap();
    let offset_buf = runtime.upload_u32(&[ple_offset]).unwrap();
    dispatch(
        runtime,
        "gemma4_ple_gate_gelu_f32",
        &[
            (&input_buf, 0),
            (&weights_buf, 0),
            (&ple_buf, 0),
            (&offset_buf, 0),
            (&out_buf, 0),
            (&input_width_buf, 0),
            (&output_width_buf, 0),
        ],
        (rows + 15) / 16,
        128,
    );
    runtime.read_f32(&out_buf, rows).unwrap()
}

#[test]
fn ple_gate_gelu_fused_is_bitwise_identical_to_split() {
    // The phase-13.19 PLE input-gate fusion: gemma4_ple_gate_gelu_f32 replaces
    // the per-layer pair of matvec_q4_0_16row_mv (state * inp_gate) followed
    // by ple_gelu_multiply_offset_f32 (GELU * PLE slice) with one dispatch.
    // The fused kernel keeps the 16-row per-lane block math and the elementwise
    // GELU * PLE multiply verbatim, so every output element must match the
    // two-kernel baseline bitwise (not merely within 1e-3).
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x5f4b_7a3cu32;
    for input_width in [512u32, 1024u32, 2304u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        for rows in [31usize, 256usize, 260usize, 512usize, 1280usize] {
            let weights = build_q4_weights(rows, blocks, &mut state);
            let mut ple = vec![0.0f32; 4 * rows];
            fill_f32(&mut ple, &mut state);
            let ple_offset = (next_u32(&mut state) % 4) * rows as u32;
            let split = run_ple_gate_gelu_split(&runtime, &input, &weights, &ple, ple_offset, rows);
            let fused = run_ple_gate_gelu_fused(&runtime, &input, &weights, &ple, ple_offset, rows);
            assert_bitwise(
                &format!(
                    "ple_gate_gelu_fused_vs_split in={input_width} rows={rows} off={ple_offset}"
                ),
                &split,
                &fused,
            );
        }
    }
}
