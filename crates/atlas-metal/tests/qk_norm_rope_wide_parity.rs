//! Tolerance parity for the phase-13.21 wide (SIMD-group-parallel) qk_norm_rope
//! kernels.  The exact `_f32` / `_batch_f32` kernels reduce each head's RMS
//! sum-of-squares with one scalar thread, leaving the GPU nearly idle in
//! decode; the `_wide` variants spread that reduction across one Apple SIMD
//! group (32 lanes via `simd_sum`) and the RoPE rotation one pair per lane.
//! The parallel reduction reorders FP32 arithmetic, so this asserts the
//! established tolerance contract (max-abs < 1e-3) against the exact kernel,
//! which remains the byte-identical oracle.  Covers the Gemma4 E2B head
//! geometries (512-wide full, 256-wide sliding) and the provider/independent
//! K-head case (`has_key` = 1), plus the batched prefill variant at batch = 1.

use atlas_metal::{MetalError, MetalRuntime};

const Q_HEADS: u32 = 8;
const FULL_DIM: u32 = 512;
const SWA_DIM: u32 = 256;

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

fn fill_cos_sin(values: &mut [f32], state: &mut u32) {
    for value in values.iter_mut() {
        *value = (next_f32(state) - 0.5) * 0.5 + 0.9;
    }
}

fn compare(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len(), "{label}: length mismatch");
    let mut max_abs = 0.0f32;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
    }
    eprintln!("{label}: max_abs={max_abs:.3e}");
    assert!(
        max_abs < 1e-3,
        "{label}: wide qk_norm_rope diverges from the exact kernel (max_abs={max_abs:.3e})"
    );
}

/// Run a single qk_norm_rope kernel over the given geometry.
/// `kernel`/`threads` select exact vs wide; `batch` > 1 uses the `_batch_*`
/// kernels (grid = batch * (q_heads + has_key)), batch == 1 the per-token ones.
fn run_kernel(
    runtime: &MetalRuntime,
    kernel: &'static str,
    threads: usize,
    head_dim: u32,
    has_key: bool,
    batch: u32,
) -> (Vec<f32>, Vec<f32>) {
    let mut state = 0x7711_cc33u32;
    let q_width = (Q_HEADS * head_dim) as usize;
    // Buffers are always sized for `batch` tokens (batch == 1 for the
    // per-token kernels), because the batch kernels index per-token with
    // stride q_heads*head_dim / head_dim.
    let q_len = (batch as usize) * q_width;
    let k_len = (batch as usize) * (head_dim as usize);
    let mut q = vec![0.0f32; q_len];
    let mut k = vec![0.0f32; k_len];
    fill_f32(&mut q, &mut state);
    fill_f32(&mut k, &mut state);
    let mut q_weight = vec![0.0f32; head_dim as usize];
    let mut k_weight = vec![0.0f32; head_dim as usize];
    fill_f32(&mut q_weight, &mut state);
    fill_f32(&mut k_weight, &mut state);
    // RoPE tables sized to the full context (batch * pairs) for the batch path.
    let pairs = head_dim / 2;
    let rope_len = (batch as usize) * (pairs as usize);
    let mut cos = vec![0.0f32; rope_len];
    let mut sin = vec![0.0f32; rope_len];
    fill_cos_sin(&mut cos, &mut state);
    fill_cos_sin(&mut sin, &mut state);

    let q_buf = runtime.upload_f32(&q).unwrap();
    let k_buf = runtime.upload_f32(&k).unwrap();
    let qw_buf = runtime.upload_f32(&q_weight).unwrap();
    let kw_buf = runtime.upload_f32(&k_weight).unwrap();
    let cos_buf = runtime.upload_f32(&cos).unwrap();
    let sin_buf = runtime.upload_f32(&sin).unwrap();
    let q_rot = runtime.upload_f32(&vec![0.0f32; q_len]).unwrap();
    let k_rot = runtime.upload_f32(&vec![0.0f32; k_len]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim]).unwrap();
    let q_heads_buf = runtime.upload_u32(&[Q_HEADS]).unwrap();
    let has_key_buf = runtime.upload_u32(&[u32::from(has_key)]).unwrap();
    let epsilon_buf = runtime.upload_f32(&[1e-6]).unwrap();
    let batch_buf = runtime.upload_u32(&[batch]).unwrap();
    let rope_pairs_buf = runtime.upload_u32(&[pairs]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    if batch > 1 {
        command
            .dispatch_threadgroups_1d_at(
                kernel,
                &[
                    (&q_buf, 0),
                    (&k_buf, 0),
                    (&qw_buf, 0),
                    (&kw_buf, 0),
                    (&cos_buf, 0),
                    (&sin_buf, 0),
                    (&q_rot, 0),
                    (&k_rot, 0),
                    (&head_dim_buf, 0),
                    (&q_heads_buf, 0),
                    (&has_key_buf, 0),
                    (&epsilon_buf, 0),
                    (&batch_buf, 0),
                    (&rope_pairs_buf, 0),
                ],
                (batch as usize) * (Q_HEADS as usize + usize::from(has_key)),
                threads,
            )
            .unwrap();
    } else {
        command
            .dispatch_threadgroups_1d_at(
                kernel,
                &[
                    (&q_buf, 0),
                    (&k_buf, 0),
                    (&qw_buf, 0),
                    (&kw_buf, 0),
                    (&cos_buf, 0),
                    (&sin_buf, 0),
                    (&q_rot, 0),
                    (&k_rot, 0),
                    (&head_dim_buf, 0),
                    (&q_heads_buf, 0),
                    (&has_key_buf, 0),
                    (&epsilon_buf, 0),
                ],
                Q_HEADS as usize + usize::from(has_key),
                threads,
            )
            .unwrap();
    }
    command.finish().unwrap();

    (
        runtime.read_f32(&q_rot, q_len).unwrap(),
        runtime.read_f32(&k_rot, k_len).unwrap(),
    )
}

#[test]
fn qk_norm_rope_wide_matches_exact_full_provider() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    let (exact_q, exact_k) = run_kernel(
        &runtime,
        "gemma4_qk_norm_rope_fused_f32",
        1,
        FULL_DIM,
        true,
        1,
    );
    let (wide_q, wide_k) = run_kernel(
        &runtime,
        "gemma4_qk_norm_rope_fused_f32_wide",
        32,
        FULL_DIM,
        true,
        1,
    );
    compare("full-512-provider-q", &exact_q, &wide_q);
    compare("full-512-provider-k", &exact_k, &wide_k);
}

#[test]
fn qk_norm_rope_wide_matches_exact_swa_independent_key() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    let (exact_q, _) = run_kernel(
        &runtime,
        "gemma4_qk_norm_rope_fused_f32",
        1,
        SWA_DIM,
        false,
        1,
    );
    let (wide_q, _) = run_kernel(
        &runtime,
        "gemma4_qk_norm_rope_fused_f32_wide",
        32,
        SWA_DIM,
        false,
        1,
    );
    compare("swa-256-independent-q", &exact_q, &wide_q);
}

#[test]
fn qk_norm_rope_batch_wide_matches_exact() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    let (batch_exact_q, _) = run_kernel(
        &runtime,
        "gemma4_qk_norm_rope_fused_batch_f32",
        1,
        FULL_DIM,
        true,
        8,
    );
    let (batch_wide_q, _) = run_kernel(
        &runtime,
        "gemma4_qk_norm_rope_fused_batch_f32_wide",
        32,
        FULL_DIM,
        true,
        8,
    );
    compare("full-512-batch8-q", &batch_exact_q, &batch_wide_q);
}
