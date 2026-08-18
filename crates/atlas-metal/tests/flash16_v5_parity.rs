//! Tolerance parity for the Flash16-v5 batched prefill attention kernel. The
//! v5 variant processes one token per threadgroup with one SIMD group per
//! head, sharing the K/V q4_0 dequant across heads (Gemma 4 E2B uses
//! kv_heads == 1). It computes the same attention as the v4 merged-slice
//! kernel but with a different fp32 reduction order, so parity is asserted
//! with a relative tolerance against v4.

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

fn compare_rel(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len(), "{label}: length mismatch");
    let mut max_rel = 0.0f32;
    let mut max_abs = 0.0f32;
    for (r, c) in reference.iter().zip(candidate.iter()) {
        assert!(c.is_finite(), "{label}: non-finite output");
        let diff = (r - c).abs();
        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(diff / r.abs().max(1.0));
    }
    eprintln!("{label}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
    assert!(
        max_rel < 1e-2,
        "{label}: flash16-v5 diverges from v4 (max_rel={max_rel:.3e}, max_abs={max_abs:.3e})"
    );
}

fn run(head_dim: usize, batch: usize, heads: usize, kv_heads: usize, key_count: u32) {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x7711_cc33u32;
    let q_width = heads * head_dim;
    let capacity = 256usize;
    let layers = 3usize;
    let layer = 1usize;

    let mut query = vec![0.0f32; batch * q_width];
    fill_f32(&mut query, &mut state);
    let blocks_per_position = kv_heads * head_dim / 32;
    let cache_bytes = 2 * capacity * blocks_per_position * 18;
    let mut cache = vec![0u8; cache_bytes];
    for (block_index, block) in cache.chunks_mut(18).enumerate() {
        let scale = 0.05 + (block_index as f32 % 101.0) * 0.003;
        block[..2].copy_from_slice(&((scale * 32768.0).round() as u16).to_le_bytes());
        for (offset, byte) in block.iter_mut().enumerate().skip(2) {
            *byte = ((block_index * 7 + offset * 13) % 256) as u8;
        }
    }
    let mut key_controls = vec![0u32; batch * layers];
    for token in 0..batch {
        for l in 0..layers {
            key_controls[token * layers + l] = key_count;
        }
    }

    let query_buf = runtime.upload_f32(&query).unwrap();
    let cache_buf = runtime.upload_bytes(&cache).unwrap();
    let key_control_buf = runtime.upload_u32(&key_controls).unwrap();
    let heads_buf = runtime.upload_u32(&[heads as u32]).unwrap();
    let kv_heads_buf = runtime.upload_u32(&[kv_heads as u32]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim as u32]).unwrap();
    let capacity_buf = runtime.upload_u32(&[capacity as u32]).unwrap();
    let layers_buf = runtime.upload_u32(&[layers as u32]).unwrap();

    let controls_offset = layer * std::mem::size_of::<u32>();

    let (v4_kernel, v4_threads) = if head_dim == 256 {
        ("attention_prefill_gemma4_simd_q4_0_flash16_swa_v4", 768)
    } else {
        ("attention_prefill_gemma4_simd_q4_0_flash16_v4", 384)
    };
    let v5_kernel = if head_dim == 256 {
        "attention_prefill_gemma4_simd_q4_0_flash16_swa_v5"
    } else {
        "attention_prefill_gemma4_simd_q4_0_flash16_v5"
    };

    let out_v4 = runtime.upload_f32(&vec![0.0f32; batch * q_width]).unwrap();
    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at_labeled(
            v4_kernel,
            None,
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&out_v4, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_control_buf, controls_offset),
                (&layers_buf, 0),
            ],
            batch * heads,
            v4_threads,
        )
        .unwrap();
    command.finish().unwrap();

    let out_v5 = runtime.upload_f32(&vec![0.0f32; batch * q_width]).unwrap();
    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at_labeled(
            v5_kernel,
            None,
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&out_v5, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_control_buf, controls_offset),
                (&layers_buf, 0),
            ],
            batch,
            256,
        )
        .unwrap();
    command.finish().unwrap();

    compare_rel(
        &format!("flash16_v5 head_dim={head_dim}"),
        &runtime.read_f32(&out_v4, batch * q_width).unwrap(),
        &runtime.read_f32(&out_v5, batch * q_width).unwrap(),
    );
}

#[test]
fn flash16_v5_swa_matches_v4_relative() {
    run(256, 3, 8, 1, 64);
}

#[test]
fn flash16_v5_full_matches_v4_relative() {
    run(512, 3, 8, 1, 64);
}
