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
