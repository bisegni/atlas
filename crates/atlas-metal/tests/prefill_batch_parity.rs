//! Bitwise parity for the phase-13.1 token-batched prefill kernels (R1).
//! Each batched kernel must reproduce, bit for bit, the output of the
//! per-token dispatch loop it replaces in `encode_prefill_layer_major_layer`
//! and `forward_tokens_layer_major`: one batched dispatch covers the whole
//! chunk while every (token, head) threadgroup performs the identical
//! per-token math.  Decode continues to use the original per-token kernels,
//! so they are the exact reference path here.

use atlas_metal::{GpuBuffer, MetalError, MetalRuntime};

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

fn read_bytes(runtime: &MetalRuntime, buffer: &GpuBuffer, byte_count: usize) -> Vec<u8> {
    assert_eq!(byte_count % 4, 0, "byte readback must be u32 aligned");
    runtime
        .read_f32(buffer, byte_count / 4)
        .unwrap()
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn compare_f32(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len(), "{label}: length mismatch");
    for (index, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
        assert_eq!(
            reference.to_bits(),
            candidate.to_bits(),
            "{label}: bitwise divergence at index {index}"
        );
    }
}

fn compare_bytes(label: &str, reference: &[u8], candidate: &[u8]) {
    assert_eq!(reference.len(), candidate.len(), "{label}: length mismatch");
    for (index, (reference, candidate)) in reference.iter().zip(candidate).enumerate() {
        assert_eq!(
            *reference, *candidate,
            "{label}: byte divergence at index {index}"
        );
    }
}

fn metal_runtime() -> MetalRuntime {
    match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            std::process::exit(0);
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    }
}

#[test]
fn qk_norm_rope_batch_matches_per_token_loop() {
    let runtime = metal_runtime();
    let mut state = 0x9e37_79b9u32;
    let batch = 3usize;
    let q_heads = 8usize;
    let head_dim = 512usize;
    let has_key = 1usize;
    let rope_pairs = head_dim / 2;
    let q_width = q_heads * head_dim;

    let mut q = vec![0.0f32; batch * q_width];
    let mut k = vec![0.0f32; batch * head_dim];
    let mut q_norm = vec![0.0f32; head_dim];
    let mut k_norm = vec![0.0f32; head_dim];
    let mut cos = vec![0.0f32; batch * rope_pairs];
    let mut sin = vec![0.0f32; batch * rope_pairs];
    fill_f32(&mut q, &mut state);
    fill_f32(&mut k, &mut state);
    fill_f32(&mut q_norm, &mut state);
    fill_f32(&mut k_norm, &mut state);
    fill_f32(&mut cos, &mut state);
    fill_f32(&mut sin, &mut state);

    let q_buf = runtime.upload_f32(&q).unwrap();
    let k_buf = runtime.upload_f32(&k).unwrap();
    let q_norm_buf = runtime.upload_f32(&q_norm).unwrap();
    let k_norm_buf = runtime.upload_f32(&k_norm).unwrap();
    let cos_buf = runtime.upload_f32(&cos).unwrap();
    let sin_buf = runtime.upload_f32(&sin).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim as u32]).unwrap();
    let heads_buf = runtime.upload_u32(&[q_heads as u32]).unwrap();
    let one_buf = runtime.upload_u32(&[1]).unwrap();
    let epsilon_buf = runtime.upload_f32(&[1e-6]).unwrap();
    let batch_buf = runtime.upload_u32(&[batch as u32]).unwrap();
    let rope_pairs_buf = runtime.upload_u32(&[rope_pairs as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let q_rot_ref = runtime.upload_f32(&vec![0.0f32; batch * q_width]).unwrap();
    let k_rot_ref = runtime.upload_f32(&vec![0.0f32; batch * head_dim]).unwrap();
    for token in 0..batch {
        command
            .dispatch_threadgroups_1d_at(
                "gemma4_qk_norm_rope_fused_f32",
                &[
                    (&q_buf, token * q_width * 4),
                    (&k_buf, token * head_dim * 4),
                    (&q_norm_buf, 0),
                    (&k_norm_buf, 0),
                    (&cos_buf, token * rope_pairs * 4),
                    (&sin_buf, token * rope_pairs * 4),
                    (&q_rot_ref, token * q_width * 4),
                    (&k_rot_ref, token * head_dim * 4),
                    (&head_dim_buf, 0),
                    (&heads_buf, 0),
                    (&one_buf, 0),
                    (&epsilon_buf, 0),
                ],
                q_heads + has_key,
                1,
            )
            .unwrap();
    }
    command.finish().unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let q_rot_batch = runtime.upload_f32(&vec![0.0f32; batch * q_width]).unwrap();
    let k_rot_batch = runtime.upload_f32(&vec![0.0f32; batch * head_dim]).unwrap();
    command
        .dispatch_threadgroups_1d(
            "gemma4_qk_norm_rope_fused_batch_f32",
            &[
                &q_buf,
                &k_buf,
                &q_norm_buf,
                &k_norm_buf,
                &cos_buf,
                &sin_buf,
                &q_rot_batch,
                &k_rot_batch,
                &head_dim_buf,
                &heads_buf,
                &one_buf,
                &epsilon_buf,
                &batch_buf,
                &rope_pairs_buf,
            ],
            batch * (q_heads + has_key),
            1,
        )
        .unwrap();
    command.finish().unwrap();

    compare_f32(
        "qk_norm_rope q_rot",
        &runtime.read_f32(&q_rot_ref, batch * q_width).unwrap(),
        &runtime.read_f32(&q_rot_batch, batch * q_width).unwrap(),
    );
    compare_f32(
        "qk_norm_rope k_rot",
        &runtime.read_f32(&k_rot_ref, batch * head_dim).unwrap(),
        &runtime.read_f32(&k_rot_batch, batch * head_dim).unwrap(),
    );
}

#[test]
fn v_norm_batch_matches_per_token_loop() {
    let runtime = metal_runtime();
    let mut state = 0x5a3c_12d7u32;
    let batch = 5usize;
    let kv_width = 512usize;
    let mut v = vec![0.0f32; batch * kv_width];
    fill_f32(&mut v, &mut state);
    let _v_buf = runtime.upload_f32(&v).unwrap();
    let v_ref = runtime.upload_f32(&vec![0.0f32; batch * kv_width]).unwrap();
    let v_batch = runtime.upload_f32(&vec![0.0f32; batch * kv_width]).unwrap();
    let width_buf = runtime.upload_u32(&[kv_width as u32]).unwrap();
    let one_buf = runtime.upload_u32(&[1]).unwrap();
    let batch_buf = runtime.upload_u32(&[batch as u32]).unwrap();
    let epsilon_buf = runtime.upload_f32(&[1e-6]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    for token in 0..batch {
        command
            .dispatch_1d_at(
                "rms_norm_groups_in_place_unweighted_f32",
                &[
                    (&v_ref, token * kv_width * 4),
                    (&width_buf, 0),
                    (&one_buf, 0),
                    (&epsilon_buf, 0),
                ],
                1,
            )
            .unwrap();
    }
    command.finish().unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_1d(
            "rms_norm_groups_in_place_unweighted_f32",
            &[&v_batch, &width_buf, &batch_buf, &epsilon_buf],
            batch,
        )
        .unwrap();
    command.finish().unwrap();

    compare_f32(
        "v_norm",
        &runtime.read_f32(&v_ref, batch * kv_width).unwrap(),
        &runtime.read_f32(&v_batch, batch * kv_width).unwrap(),
    );
}

fn kv_append_batch_matches_per_token_loop(cache_type: &str) {
    let runtime = metal_runtime();
    let mut state = 0x7f4a_2b1eu32;
    let batch = 4usize;
    let kv_width = 512usize;
    let capacity = 32usize;
    let mut key = vec![0.0f32; batch * kv_width];
    let mut value = vec![0.0f32; batch * kv_width];
    fill_f32(&mut key, &mut state);
    fill_f32(&mut value, &mut state);
    let packed = cache_type != "f32";
    let block_bytes = if cache_type == "q8_0" { 34 } else { 18 };
    let cache_bytes = if packed {
        2 * capacity * (kv_width / 32) * block_bytes
    } else {
        2 * capacity * kv_width * 4
    };
    let cache_floats = cache_bytes / 4;
    let positions: Vec<u32> = (10..10 + batch).map(|position| position as u32).collect();

    let key_buf = runtime.upload_f32(&key).unwrap();
    let value_buf = runtime.upload_f32(&value).unwrap();
    let width_buf = runtime.upload_u32(&[kv_width as u32]).unwrap();
    let capacity_buf = runtime.upload_u32(&[capacity as u32]).unwrap();
    let positions_buf = runtime.upload_u32(&positions).unwrap();
    let zero_cache = vec![0.0f32; cache_floats];
    let cache_ref = runtime.upload_f32(&zero_cache).unwrap();
    let cache_batch = runtime.upload_f32(&zero_cache).unwrap();

    let per_token_kernel = match cache_type {
        "f32" => "kv_append_decode_f32",
        "q8_0" => "kv_append_decode_q8_0",
        "q4_0" => "kv_append_decode_q4_0",
        _ => unreachable!(),
    };
    let batch_kernel = match cache_type {
        "f32" => "kv_append_decode_f32_batch",
        "q8_0" => "kv_append_decode_q8_0_batch",
        "q4_0" => "kv_append_decode_q4_0_batch",
        _ => unreachable!(),
    };
    let append_count = if packed { kv_width / 32 } else { kv_width };

    let mut command = runtime.begin_resident_command().unwrap();
    for token in 0..batch {
        command
            .dispatch_1d_at(
                per_token_kernel,
                &[
                    (&key_buf, token * kv_width * 4),
                    (&value_buf, token * kv_width * 4),
                    (&cache_ref, 0),
                    (&width_buf, 0),
                    (&capacity_buf, 0),
                    (&positions_buf, token * 4),
                ],
                append_count,
            )
            .unwrap();
    }
    command.finish().unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_1d(
            batch_kernel,
            &[
                &key_buf,
                &value_buf,
                &cache_batch,
                &width_buf,
                &capacity_buf,
                &positions_buf,
            ],
            batch * append_count,
        )
        .unwrap();
    command.finish().unwrap();

    compare_bytes(
        &format!("kv_append {cache_type}"),
        &read_bytes(&runtime, &cache_ref, cache_bytes),
        &read_bytes(&runtime, &cache_batch, cache_bytes),
    );
}

#[test]
fn kv_append_f32_batch_matches_per_token_loop() {
    kv_append_batch_matches_per_token_loop("f32");
}

#[test]
fn kv_append_q8_0_batch_matches_per_token_loop() {
    kv_append_batch_matches_per_token_loop("q8_0");
}

#[test]
fn kv_append_q4_0_batch_matches_per_token_loop() {
    kv_append_batch_matches_per_token_loop("q4_0");
}

fn attention_batch_matches_per_token_loop(cache_type: &str) {
    let runtime = metal_runtime();
    let mut state = 0x44aa_9901u32;
    let batch = 3usize;
    let heads = 8usize;
    let kv_heads = 1usize;
    let head_dim = 512usize;
    let capacity = 16usize;
    let layers = 3usize;
    let layer = 1usize;
    let q_width = heads * head_dim;
    let mut query = vec![0.0f32; batch * q_width];
    fill_f32(&mut query, &mut state);

    let block_bytes = match cache_type {
        "f32" => 4,
        "q8_0" => 34,
        "q4_0" => 18,
        _ => unreachable!(),
    };
    let blocks_per_position = kv_heads * head_dim / 32;
    let cache_bytes = if cache_type == "f32" {
        2 * capacity * head_dim * 4
    } else {
        2 * capacity * blocks_per_position * block_bytes
    };
    let cache_buf = if cache_type == "f32" {
        let mut cache_floats = vec![0.0f32; cache_bytes / 4];
        fill_f32(&mut cache_floats, &mut state);
        runtime.upload_f32(&cache_floats).unwrap()
    } else {
        let mut cache = vec![0u8; cache_bytes];
        for (block_index, block) in cache.chunks_mut(block_bytes).enumerate() {
            let scale = 0.5f32;
            block[..2].copy_from_slice(&((scale * 32768.0).round() as u16).to_le_bytes());
            for (offset, byte) in block.iter_mut().enumerate().skip(2) {
                *byte = ((block_index * 7 + offset * 13) % 256) as u8;
            }
        }
        runtime.upload_bytes(&cache).unwrap()
    };
    let mut key_counts = vec![0u32; batch * layers];
    for token in 0..batch {
        for l in 0..layers {
            let control = (((token * 2) as u32) << 16) | 2u32;
            key_counts[token * layers + l] = control;
        }
    }
    let key_counts_buf = runtime.upload_u32(&key_counts).unwrap();
    let heads_buf = runtime.upload_u32(&[heads as u32]).unwrap();
    let kv_heads_buf = runtime.upload_u32(&[kv_heads as u32]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim as u32]).unwrap();
    let capacity_buf = runtime.upload_u32(&[capacity as u32]).unwrap();
    let layers_buf = runtime.upload_u32(&[layers as u32]).unwrap();
    let query_buf = runtime.upload_f32(&query).unwrap();

    let per_token_kernel = match cache_type {
        "f32" => "attention_decode_fused_gemma4_simd_f32",
        "q8_0" => "attention_decode_fused_gemma4_simd_q8_0",
        "q4_0" => "attention_decode_fused_gemma4_simd_q4_0",
        _ => unreachable!(),
    };
    let batch_kernel = match cache_type {
        "f32" => "attention_decode_fused_gemma4_simd_f32_batch",
        "q8_0" => "attention_decode_fused_gemma4_simd_q8_0_batch",
        "q4_0" => "attention_decode_fused_gemma4_simd_q4_0_batch",
        _ => unreachable!(),
    };

    let mut command = runtime.begin_resident_command().unwrap();
    let out_ref = runtime.upload_f32(&vec![0.0f32; batch * q_width]).unwrap();
    for token in 0..batch {
        command
            .dispatch_threadgroups_1d_at(
                per_token_kernel,
                &[
                    (&query_buf, token * q_width * 4),
                    (&cache_buf, 0),
                    (&out_ref, token * q_width * 4),
                    (&heads_buf, 0),
                    (&kv_heads_buf, 0),
                    (&head_dim_buf, 0),
                    (&capacity_buf, 0),
                    (&key_counts_buf, (token * layers + layer) * 4),
                ],
                heads,
                128,
            )
            .unwrap();
    }
    command.finish().unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let out_batch = runtime.upload_f32(&vec![0.0f32; batch * q_width]).unwrap();
    command
        .dispatch_threadgroups_1d_at(
            batch_kernel,
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&out_batch, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_counts_buf, layer * 4),
                (&layers_buf, 0),
            ],
            batch * heads,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    compare_f32(
        &format!("attention {cache_type}"),
        &runtime.read_f32(&out_ref, batch * q_width).unwrap(),
        &runtime.read_f32(&out_batch, batch * q_width).unwrap(),
    );
}

#[test]
fn attention_f32_batch_matches_per_token_loop() {
    attention_batch_matches_per_token_loop("f32");
}

#[test]
fn attention_q8_0_batch_matches_per_token_loop() {
    attention_batch_matches_per_token_loop("q8_0");
}

#[test]
fn attention_q4_0_batch_matches_per_token_loop() {
    attention_batch_matches_per_token_loop("q4_0");
}

#[test]
fn ple_stable_norm_batch_matches_per_token_loop() {
    let runtime = metal_runtime();
    let mut state = 0x1c2d_3e4fu32;
    let batch = 4usize;
    let layers = 3usize;
    let ple_width = 256usize;
    let ple_total = layers * ple_width;
    let mut projected = vec![0.0f32; batch * ple_total];
    let mut weight = vec![0.0f32; ple_width];
    fill_f32(&mut projected, &mut state);
    fill_f32(&mut weight, &mut state);
    let _projected_buf = runtime.upload_f32(&projected).unwrap();
    let weight_buf = runtime.upload_f32(&weight).unwrap();
    let width_buf = runtime.upload_u32(&[ple_width as u32]).unwrap();
    let layers_buf = runtime.upload_u32(&[layers as u32]).unwrap();
    let batch_layers_buf = runtime.upload_u32(&[(batch * layers) as u32]).unwrap();
    let epsilon_buf = runtime.upload_f32(&[1e-6]).unwrap();
    let out_ref = runtime.upload_f32(&projected).unwrap();
    let out_batch = runtime.upload_f32(&projected).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    for token in 0..batch {
        command
            .dispatch_1d_at(
                "rms_norm_groups_in_place_stable_f32",
                &[
                    (&out_ref, token * ple_total * 4),
                    (&weight_buf, 0),
                    (&width_buf, 0),
                    (&layers_buf, 0),
                    (&epsilon_buf, 0),
                ],
                layers,
            )
            .unwrap();
    }
    command.finish().unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_1d(
            "rms_norm_groups_in_place_stable_f32",
            &[
                &out_batch,
                &weight_buf,
                &width_buf,
                &batch_layers_buf,
                &epsilon_buf,
            ],
            batch * layers,
        )
        .unwrap();
    command.finish().unwrap();

    compare_f32(
        "ple stable norm",
        &runtime.read_f32(&out_ref, batch * ple_total).unwrap(),
        &runtime.read_f32(&out_batch, batch * ple_total).unwrap(),
    );
}

#[test]
fn ple_offset_multiply_batch_matches_per_token_loop() {
    let runtime = metal_runtime();
    let mut state = 0x0a1b_2c3du32;
    let batch = 3usize;
    let layers = 4usize;
    let ple_width = 256usize;
    let ple_total = layers * ple_width;
    let mut gate = vec![0.0f32; batch * ple_total];
    let mut ple = vec![0.0f32; batch * ple_total];
    fill_f32(&mut gate, &mut state);
    fill_f32(&mut ple, &mut state);
    let layer = 2usize;
    let gate_buf = runtime.upload_f32(&gate).unwrap();
    let ple_buf = runtime.upload_f32(&ple).unwrap();
    let layers_buf = runtime.upload_u32(&[layers as u32]).unwrap();
    let width_buf = runtime.upload_u32(&[ple_width as u32]).unwrap();
    let batch_buf = runtime.upload_u32(&[batch as u32]).unwrap();
    let offset_buf = runtime.upload_u32(&[(layer * ple_width) as u32]).unwrap();
    let out_ref = runtime
        .upload_f32(&vec![0.0f32; batch * ple_total])
        .unwrap();
    let out_batch = runtime
        .upload_f32(&vec![0.0f32; batch * ple_total])
        .unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    for token in 0..batch {
        command
            .dispatch_1d_at(
                "vector_multiply_offset_f32",
                &[
                    (&gate_buf, token * ple_width * 4),
                    (&ple_buf, token * ple_total * 4),
                    (&out_ref, token * ple_width * 4),
                    (&offset_buf, 0),
                    (&width_buf, 0),
                ],
                ple_width,
            )
            .unwrap();
    }
    command.finish().unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_1d(
            "vector_multiply_offset_batch_f32",
            &[
                &gate_buf,
                &ple_buf,
                &out_batch,
                &offset_buf,
                &layers_buf,
                &width_buf,
                &batch_buf,
            ],
            batch * ple_width,
        )
        .unwrap();
    command.finish().unwrap();

    compare_f32(
        "ple offset multiply",
        &runtime.read_f32(&out_ref, batch * ple_total).unwrap(),
        &runtime.read_f32(&out_batch, batch * ple_total).unwrap(),
    );
}
