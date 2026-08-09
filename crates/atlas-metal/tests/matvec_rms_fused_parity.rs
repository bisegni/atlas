//! Tolerance-based correctness check for the RMS-input `_rms` matvec variants
//! against the standalone `rms_norm_decode_f32_vec4` + unfused mv_ext path.
//! The fused kernels fold the RMS reduction into the matvec dispatch, so the
//! reference is the production pipeline: normalize the raw input with the
//! vec4 kernel, then run the plain 32-row mv_ext kernel on the normalized
//! buffer. Asserting max-abs < 1e-3 (same contract as
//! `matvec_mv_ext_parity.rs` and `attention_flash_correctness.rs`) proves the
//! fusion is numerically equivalent without requiring bitwise stream parity.

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
        "{label}: fused RMS matvec diverges from vec4+unfused (max_abs={max_abs:.3e})"
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

/// Normalize `input` with rms_weight + epsilon into `norm` using the
/// production vec4 kernel (hidden multiple of 128) or the scalar fallback.
fn normalize_reference(
    runtime: &MetalRuntime,
    input: &atlas_metal::GpuBuffer,
    rms_weight: &atlas_metal::GpuBuffer,
    norm: &atlas_metal::GpuBuffer,
    hidden: u32,
) {
    let hidden_buf = runtime.upload_u32(&[hidden]).unwrap();
    let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
    let kernel = if hidden % 128 == 0 {
        "rms_norm_decode_f32_vec4"
    } else {
        "rms_norm_decode_f32"
    };
    dispatch(
        runtime,
        kernel,
        &[
            (input, 0),
            (rms_weight, 0),
            (norm, 0),
            (&hidden_buf, 0),
            (&eps_buf, 0),
        ],
        1,
        32,
    );
}

#[test]
fn matvec_q4_32row_mv_rms_matches_vec4_plus_unfused() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e37_79b9u32;
    for input_width in [256u32, 2048u32, 2304u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        let input_buf = runtime.upload_f32(&input).unwrap();
        let rms_weight_buf = runtime.upload_f32(&rms_weight).unwrap();
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
        for output_width in [31u32, 129u32, 137u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let weights_buf = runtime.upload_bytes(&weights).unwrap();
            let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();

            let norm_buf = runtime
                .upload_f32(&vec![0.0f32; input_width as usize])
                .unwrap();
            normalize_reference(
                &runtime,
                &input_buf,
                &rms_weight_buf,
                &norm_buf,
                input_width,
            );
            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            let groups = ((output_width + 31) / 32) as usize;
            dispatch(
                &runtime,
                "matvec_q4_0_32row_mv",
                &[
                    (&norm_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                ],
                groups,
                THREADS,
            );
            let reference = runtime.read_f32(&out_buf, rows).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            dispatch(
                &runtime,
                "matvec_q4_0_32row_mv_rms",
                &[
                    (&input_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&rms_weight_buf, 0),
                    (&eps_buf, 0),
                ],
                groups,
                THREADS,
            );
            let candidate = runtime.read_f32(&out_buf, rows).unwrap();
            compare(
                &format!("matvec_q4_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn qkv_32row_mv_rms_matches_vec4_plus_unfused() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x3243_f6a8u32;
    let input_width = 2048u32;
    let blocks = (input_width / 32) as usize;
    let mut input = vec![0.0f32; input_width as usize];
    fill_f32(&mut input, &mut state);
    let mut rms_weight = vec![0.0f32; input_width as usize];
    fill_rms_weight(&mut rms_weight, &mut state);
    let input_buf = runtime.upload_f32(&input).unwrap();
    let rms_weight_buf = runtime.upload_f32(&rms_weight).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
    let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
    let norm_buf = runtime
        .upload_f32(&vec![0.0f32; input_width as usize])
        .unwrap();
    normalize_reference(
        &runtime,
        &input_buf,
        &rms_weight_buf,
        &norm_buf,
        input_width,
    );
    for (q_width, kv_width) in [(129u32, 33u32), (1024u32, 256u32), (2048u32, 512u32)] {
        let q_rows = q_width as usize;
        let kv_rows = kv_width as usize;
        let q_weights = build_q4_weights(q_rows, blocks, &mut state);
        let k_weights = build_q4_weights(kv_rows, blocks, &mut state);
        let v_weights = build_q4_weights(kv_rows, blocks, &mut state);
        let q_weights_buf = runtime.upload_bytes(&q_weights).unwrap();
        let k_weights_buf = runtime.upload_bytes(&k_weights).unwrap();
        let v_weights_buf = runtime.upload_bytes(&v_weights).unwrap();
        let q_width_buf = runtime.upload_u32(&[q_width]).unwrap();
        let kv_width_buf = runtime.upload_u32(&[kv_width]).unwrap();
        let zero = vec![0.0f32; q_rows.max(kv_rows)];

        let q_out = runtime.upload_f32(&zero).unwrap();
        let k_out = runtime.upload_f32(&zero).unwrap();
        let v_out = runtime.upload_f32(&zero).unwrap();
        let groups = ((q_width + 31) / 32 + 2 * ((kv_width + 31) / 32)) as usize;
        dispatch(
            &runtime,
            "matmul_q4_0_qkv_32row_mv",
            &[
                (&norm_buf, 0),
                (&q_weights_buf, 0),
                (&k_weights_buf, 0),
                (&v_weights_buf, 0),
                (&q_out, 0),
                (&k_out, 0),
                (&v_out, 0),
                (&input_width_buf, 0),
                (&q_width_buf, 0),
                (&kv_width_buf, 0),
            ],
            groups,
            THREADS,
        );
        let ref_q = runtime.read_f32(&q_out, q_rows).unwrap();
        let ref_k = runtime.read_f32(&k_out, kv_rows).unwrap();
        let ref_v = runtime.read_f32(&v_out, kv_rows).unwrap();

        let q_out = runtime.upload_f32(&zero).unwrap();
        let k_out = runtime.upload_f32(&zero).unwrap();
        let v_out = runtime.upload_f32(&zero).unwrap();
        dispatch(
            &runtime,
            "matmul_q4_0_qkv_32row_mv_rms",
            &[
                (&input_buf, 0),
                (&q_weights_buf, 0),
                (&k_weights_buf, 0),
                (&v_weights_buf, 0),
                (&q_out, 0),
                (&k_out, 0),
                (&v_out, 0),
                (&input_width_buf, 0),
                (&q_width_buf, 0),
                (&kv_width_buf, 0),
                (&rms_weight_buf, 0),
                (&eps_buf, 0),
            ],
            groups,
            THREADS,
        );
        let cand_q = runtime.read_f32(&q_out, q_rows).unwrap();
        let cand_k = runtime.read_f32(&k_out, kv_rows).unwrap();
        let cand_v = runtime.read_f32(&v_out, kv_rows).unwrap();
        let label = format!("qkv_rms q={q_width} kv={kv_width}");
        compare(&format!("{label} q"), &ref_q, &cand_q);
        compare(&format!("{label} k"), &ref_k, &cand_k);
        compare(&format!("{label} v"), &ref_v, &cand_v);
    }
}

#[test]
fn gate_up_32row_mv_rms_matches_vec4_plus_unfused() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0xbb67_ae85u32;
    let input_width = 2048u32;
    let blocks = (input_width / 32) as usize;
    let mut input = vec![0.0f32; input_width as usize];
    fill_f32(&mut input, &mut state);
    let mut rms_weight = vec![0.0f32; input_width as usize];
    fill_rms_weight(&mut rms_weight, &mut state);
    let input_buf = runtime.upload_f32(&input).unwrap();
    let rms_weight_buf = runtime.upload_f32(&rms_weight).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
    let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
    let norm_buf = runtime
        .upload_f32(&vec![0.0f32; input_width as usize])
        .unwrap();
    normalize_reference(
        &runtime,
        &input_buf,
        &rms_weight_buf,
        &norm_buf,
        input_width,
    );
    for output_width in [31u32, 129u32, 2048u32] {
        let rows = output_width as usize;
        let gate_weights = build_q4_weights(rows, blocks, &mut state);
        let up_weights = build_q4_weights(rows, blocks, &mut state);
        let gate_buf = runtime.upload_bytes(&gate_weights).unwrap();
        let up_buf = runtime.upload_bytes(&up_weights).unwrap();
        let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();
        let zero = vec![0.0f32; rows];

        let gate_out = runtime.upload_f32(&zero).unwrap();
        let up_out = runtime.upload_f32(&zero).unwrap();
        let groups = 2 * ((output_width + 31) / 32) as usize;
        dispatch(
            &runtime,
            "matmul_q4_0_gate_up_32row_mv",
            &[
                (&norm_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&gate_out, 0),
                (&up_out, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
            ],
            groups,
            THREADS,
        );
        let ref_gate = runtime.read_f32(&gate_out, rows).unwrap();
        let ref_up = runtime.read_f32(&up_out, rows).unwrap();

        let gate_out = runtime.upload_f32(&zero).unwrap();
        let up_out = runtime.upload_f32(&zero).unwrap();
        dispatch(
            &runtime,
            "matmul_q4_0_gate_up_32row_mv_rms",
            &[
                (&input_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&gate_out, 0),
                (&up_out, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
                (&rms_weight_buf, 0),
                (&eps_buf, 0),
            ],
            groups,
            THREADS,
        );
        let cand_gate = runtime.read_f32(&gate_out, rows).unwrap();
        let cand_up = runtime.read_f32(&up_out, rows).unwrap();
        let label = format!("gate_up_rms output={output_width}");
        compare(&format!("{label} gate"), &ref_gate, &cand_gate);
        compare(&format!("{label} up"), &ref_up, &cand_up);
    }
}

#[test]
fn matvec_q6_32row_mv_rms_matches_vec4_plus_unfused() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x3c6e_f372u32;
    for input_width in [256u32, 512u32] {
        let blocks = (input_width / 256) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        let input_buf = runtime.upload_f32(&input).unwrap();
        let rms_weight_buf = runtime.upload_f32(&rms_weight).unwrap();
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
        let norm_buf = runtime
            .upload_f32(&vec![0.0f32; input_width as usize])
            .unwrap();
        normalize_reference(
            &runtime,
            &input_buf,
            &rms_weight_buf,
            &norm_buf,
            input_width,
        );
        for output_width in [33u32, 129u32, 2048u32] {
            let rows = output_width as usize;
            let weights = build_q6_weights(rows, blocks, &mut state);
            let weights_buf = runtime.upload_bytes(&weights).unwrap();
            let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            let groups = ((output_width + 31) / 32) as usize;
            dispatch(
                &runtime,
                "matvec_q6_k_32row_mv",
                &[
                    (&norm_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                ],
                groups,
                THREADS,
            );
            let reference = runtime.read_f32(&out_buf, rows).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            dispatch(
                &runtime,
                "matvec_q6_k_32row_mv_rms",
                &[
                    (&input_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&rms_weight_buf, 0),
                    (&eps_buf, 0),
                ],
                groups,
                THREADS,
            );
            let candidate = runtime.read_f32(&out_buf, rows).unwrap();
            compare(
                &format!("matvec_q6_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matvec_q4_64row_mv_rms_matches_vec4_plus_unfused() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x2718_2818u32;
    for input_width in [256u32, 2048u32, 2304u32] {
        let blocks = (input_width / 32) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        let input_buf = runtime.upload_f32(&input).unwrap();
        let rms_weight_buf = runtime.upload_f32(&rms_weight).unwrap();
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
        let norm_buf = runtime
            .upload_f32(&vec![0.0f32; input_width as usize])
            .unwrap();
        normalize_reference(
            &runtime,
            &input_buf,
            &rms_weight_buf,
            &norm_buf,
            input_width,
        );
        for output_width in [31u32, 129u32, 137u32, 2048u32] {
            let rows = output_width as usize;
            let weights = build_q4_weights(rows, blocks, &mut state);
            let weights_buf = runtime.upload_bytes(&weights).unwrap();
            let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            let groups = ((output_width + 63) / 64) as usize;
            dispatch(
                &runtime,
                "matvec_q4_0_64row_mv",
                &[
                    (&norm_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                ],
                groups,
                256,
            );
            let reference = runtime.read_f32(&out_buf, rows).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            dispatch(
                &runtime,
                "matvec_q4_0_64row_mv_rms",
                &[
                    (&input_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&rms_weight_buf, 0),
                    (&eps_buf, 0),
                ],
                groups,
                256,
            );
            let candidate = runtime.read_f32(&out_buf, rows).unwrap();
            compare(
                &format!("matvec_q4_64row_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}

#[test]
fn matvec_q6_64row_mv_rms_matches_vec4_plus_unfused() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x3141_5926u32;
    for input_width in [256u32, 512u32, 2304u32] {
        let blocks = (input_width / 256) as usize;
        let mut input = vec![0.0f32; input_width as usize];
        fill_f32(&mut input, &mut state);
        let mut rms_weight = vec![0.0f32; input_width as usize];
        fill_rms_weight(&mut rms_weight, &mut state);
        let input_buf = runtime.upload_f32(&input).unwrap();
        let rms_weight_buf = runtime.upload_f32(&rms_weight).unwrap();
        let input_width_buf = runtime.upload_u32(&[input_width]).unwrap();
        let eps_buf = runtime.upload_f32(&[EPSILON]).unwrap();
        let norm_buf = runtime
            .upload_f32(&vec![0.0f32; input_width as usize])
            .unwrap();
        normalize_reference(
            &runtime,
            &input_buf,
            &rms_weight_buf,
            &norm_buf,
            input_width,
        );
        for output_width in [17u32, 33u32, 129u32, 2048u32] {
            let rows = output_width as usize;
            let weights = build_q6_weights(rows, blocks, &mut state);
            let weights_buf = runtime.upload_bytes(&weights).unwrap();
            let output_width_buf = runtime.upload_u32(&[output_width]).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            let groups = ((output_width + 63) / 64) as usize;
            dispatch(
                &runtime,
                "matvec_q6_k_64row_mv",
                &[
                    (&norm_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                ],
                groups,
                256,
            );
            let reference = runtime.read_f32(&out_buf, rows).unwrap();

            let out_buf = runtime.upload_f32(&vec![0.0f32; rows]).unwrap();
            dispatch(
                &runtime,
                "matvec_q6_k_64row_mv_rms",
                &[
                    (&input_buf, 0),
                    (&weights_buf, 0),
                    (&out_buf, 0),
                    (&input_width_buf, 0),
                    (&output_width_buf, 0),
                    (&rms_weight_buf, 0),
                    (&eps_buf, 0),
                ],
                groups,
                256,
            );
            let candidate = runtime.read_f32(&out_buf, rows).unwrap();
            compare(
                &format!("matvec_q6_64row_rms input={input_width} output={output_width}"),
                &reference,
                &candidate,
            );
        }
    }
}
