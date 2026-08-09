//! Buffer-level parity diagnostic: run the production Q4 two-pass first-pass
//! attention kernels on identical inputs and compare their partials, maxima,
//! and sums buffers bitwise. This is a diagnostic gate for kernel
//! experiments; it does not measure throughput.

use atlas_metal::{MetalError, MetalRuntime};

fn run_first_pass(
    runtime: &MetalRuntime,
    kernel: &'static str,
    query: &[f32],
    cache: &[u8],
    partials: &mut [f32],
    maxima: &mut [f32],
    sums: &mut [f32],
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    capacity: u32,
    key_count: u32,
    blocks: u32,
) {
    let query_buf = runtime.upload_f32(query).unwrap();
    let cache_buf = runtime.upload_bytes(cache).unwrap();
    let partials_buf = runtime.upload_f32(partials).unwrap();
    let maxima_buf = runtime.upload_f32(maxima).unwrap();
    let sums_buf = runtime.upload_f32(sums).unwrap();
    let heads_buf = runtime.upload_u32(&[heads]).unwrap();
    let kv_heads_buf = runtime.upload_u32(&[kv_heads]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim]).unwrap();
    let capacity_buf = runtime.upload_u32(&[capacity]).unwrap();
    let key_count_buf = runtime.upload_u32(&[key_count]).unwrap();
    let blocks_buf = runtime.upload_u32(&[blocks]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(
            kernel,
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&partials_buf, 0),
                (&maxima_buf, 0),
                (&sums_buf, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_count_buf, 0),
                (&blocks_buf, 0),
            ],
            (heads * blocks) as usize,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    let partial_count = (blocks * heads * head_dim) as usize;
    let slot_count = (blocks * heads) as usize;
    let got_partials = runtime.read_f32(&partials_buf, partial_count).unwrap();
    let got_maxima = runtime.read_f32(&maxima_buf, slot_count).unwrap();
    let got_sums = runtime.read_f32(&sums_buf, slot_count).unwrap();
    partials.copy_from_slice(&got_partials);
    maxima.copy_from_slice(&got_maxima);
    sums.copy_from_slice(&got_sums);
}

#[test]
fn simd_reg_matches_baseline_first_pass_buffers() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    for key_count in [48u32, 1024u32] {
        for scenario in 0..2 {
            let label = format!("keys={key_count} scenario={scenario}");
            compare_scenario(&runtime, key_count, scenario == 1, &label);
        }
    }

    for key_count in [256u32, 1024u32] {
        let label = format!("sliding keys={key_count}");
        compare_scenario_sliding(&runtime, key_count, &label);
    }
}

fn compare_scenario_sliding(runtime: &MetalRuntime, key_count: u32, label: &str) {
    let heads: u32 = 8;
    let kv_heads: u32 = 1;
    let head_dim: u32 = 256;
    let capacity: u32 = 2048;
    let blocks: u32 = 4;

    let blocks_per_position = (kv_heads * head_dim) / 32;
    let key_blocks = key_count * blocks_per_position;
    let value_blocks = capacity * blocks_per_position;

    let mut cache = vec![0u8; ((key_blocks + value_blocks) * 18) as usize];
    for (i, chunk) in cache.chunks_mut(18).enumerate() {
        let scale = 0.3 + ((i as f32) * 0.11).fract() * 0.4;
        let half = half_f32(scale);
        chunk[..2].copy_from_slice(&half.to_le_bytes());
        for (j, byte) in chunk[2..].iter_mut().enumerate() {
            let nibble = ((i * 5 + j * 11) % 256) as u8;
            *byte = nibble | (nibble << 4);
        }
    }

    let mut query = vec![0.0f32; (heads * head_dim) as usize];
    for (h, head) in query.chunks_mut(head_dim as usize).enumerate() {
        for (d, value) in head.iter_mut().enumerate() {
            *value =
                ((d as f32 * 0.17 + h as f32 * 0.41).cos() * 0.6) * (1.0 + 0.08 * (d % 13) as f32);
        }
    }

    let mut baseline_partials = vec![0.0f32; (blocks * heads * head_dim) as usize];
    let mut baseline_maxima = vec![0.0f32; (blocks * heads) as usize];
    let mut baseline_sums = vec![0.0f32; (blocks * heads) as usize];
    run_first_pass(
        runtime,
        "attention_decode_fused_gemma4_simd_q4_0_2pass_1_no_value_barrier",
        &query,
        &cache,
        &mut baseline_partials,
        &mut baseline_maxima,
        &mut baseline_sums,
        heads,
        kv_heads,
        head_dim,
        capacity,
        key_count,
        blocks,
    );

    let mut candidate_partials = vec![0.0f32; (blocks * heads * head_dim) as usize];
    let mut candidate_maxima = vec![0.0f32; (blocks * heads) as usize];
    let mut candidate_sums = vec![0.0f32; (blocks * heads) as usize];
    run_first_pass(
        runtime,
        "attention_decode_fused_gemma4_simd_q4_0_2pass_1_simd_reg",
        &query,
        &cache,
        &mut candidate_partials,
        &mut candidate_maxima,
        &mut candidate_sums,
        heads,
        kv_heads,
        head_dim,
        capacity,
        key_count,
        blocks,
    );

    let mut differing_partials = 0usize;
    for (index, (base, cand)) in baseline_partials
        .iter()
        .zip(candidate_partials.iter())
        .enumerate()
    {
        if base.to_bits() != cand.to_bits() {
            differing_partials += 1;
        }
    }
    let maxima_match = baseline_maxima
        .iter()
        .zip(candidate_maxima.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let sums_match = baseline_sums
        .iter()
        .zip(candidate_sums.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!(
        "{label}: differing partials: {differing_partials}/{}; maxima match: {maxima_match}; sums match: {sums_match}",
        baseline_partials.len()
    );
    assert_eq!(
        differing_partials, 0,
        "{label}: simd_reg sliding first-pass partials differ from the production baseline"
    );
    assert!(
        maxima_match,
        "{label}: simd_reg sliding maxima differ from the production baseline"
    );
    assert!(
        sums_match,
        "{label}: simd_reg sliding sums differ from the production baseline"
    );
}

fn compare_scenario(runtime: &MetalRuntime, key_count: u32, rising_scores: bool, label: &str) {
    let heads: u32 = 8;
    let kv_heads: u32 = 1;
    let head_dim: u32 = 512;
    let capacity: u32 = 2048;
    let blocks: u32 = 4;

    let blocks_per_position = (kv_heads * head_dim) / 32;
    let key_blocks = key_count * blocks_per_position;
    let value_blocks = capacity * blocks_per_position;

    let mut cache = vec![0u8; ((key_blocks + value_blocks) * 18) as usize];
    for (i, chunk) in cache.chunks_mut(18).enumerate() {
        let block_index = i % (key_blocks + value_blocks) as usize;
        let is_key = block_index < key_blocks as usize;
        let key_index = (block_index / blocks_per_position as usize) as u32;
        let scale = if rising_scores {
            0.05 + (key_index as f32) * 0.001
        } else {
            ((block_index as f32) * 0.05).fract() * 0.5 + 0.25
        };
        let half = half_f32(scale);
        chunk[..2].copy_from_slice(&half.to_le_bytes());
        for (j, byte) in chunk[2..].iter_mut().enumerate() {
            let nibble = if is_key && rising_scores {
                ((key_index + j as u32) % 15) as u8
            } else {
                ((i * 7 + j * 3) % 256) as u8
            };
            *byte = nibble | (nibble << 4);
        }
    }

    let mut query = vec![0.0f32; (heads * head_dim) as usize];
    for (h, head) in query.chunks_mut(head_dim as usize).enumerate() {
        for (d, value) in head.iter_mut().enumerate() {
            *value = if rising_scores {
                1.0
            } else {
                ((d as f32 * 0.13 + h as f32 * 0.37).sin() * 0.5) * (1.0 + 0.1 * (d % 17) as f32)
            };
        }
    }

    let mut baseline_partials = vec![0.0f32; (blocks * heads * head_dim) as usize];
    let mut baseline_maxima = vec![0.0f32; (blocks * heads) as usize];
    let mut baseline_sums = vec![0.0f32; (blocks * heads) as usize];
    run_first_pass(
        runtime,
        "attention_decode_fused_gemma4_simd_q4_0_2pass_1_no_value_barrier",
        &query,
        &cache,
        &mut baseline_partials,
        &mut baseline_maxima,
        &mut baseline_sums,
        heads,
        kv_heads,
        head_dim,
        capacity,
        key_count,
        blocks,
    );

    let mut candidate_partials = vec![0.0f32; (blocks * heads * head_dim) as usize];
    let mut candidate_maxima = vec![0.0f32; (blocks * heads) as usize];
    let mut candidate_sums = vec![0.0f32; (blocks * heads) as usize];
    run_first_pass(
        runtime,
        "attention_decode_fused_gemma4_simd_q4_0_2pass_1_simd_reg",
        &query,
        &cache,
        &mut candidate_partials,
        &mut candidate_maxima,
        &mut candidate_sums,
        heads,
        kv_heads,
        head_dim,
        capacity,
        key_count,
        blocks,
    );

    let mut differing_partials = 0usize;
    let mut first_diffs = Vec::new();
    for (index, (base, cand)) in baseline_partials
        .iter()
        .zip(candidate_partials.iter())
        .enumerate()
    {
        if base.to_bits() != cand.to_bits() {
            differing_partials += 1;
            if first_diffs.len() < 8 {
                first_diffs.push((index, base, cand));
            }
        }
    }
    let maxima_match = baseline_maxima
        .iter()
        .zip(candidate_maxima.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let sums_match = baseline_sums
        .iter()
        .zip(candidate_sums.iter())
        .all(|(a, b)| a.to_bits() == b.to_bits());

    eprintln!(
        "{label}: differing partials: {differing_partials}/{}; maxima match: {maxima_match}; sums match: {sums_match}",
        baseline_partials.len()
    );
    for (index, base, cand) in &first_diffs {
        eprintln!("  slot {index}: baseline={base:.9e} candidate={cand:.9e}");
    }

    assert_eq!(
        differing_partials, 0,
        "{label}: simd_reg first-pass partials differ from the production baseline"
    );
    assert!(
        maxima_match,
        "{label}: simd_reg maxima differ from the production baseline"
    );
    assert!(
        sums_match,
        "{label}: simd_reg sums differ from the production baseline"
    );
}

// Minimal half-precision round trip without pulling in a half crate: bits are
// computed directly from the f32 value using the standard half representation.
fn half_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits & 0x7f_ffff) >> 13;
    if exponent >= 31 {
        sign | 0x7c00
    } else if exponent <= 0 {
        sign
    } else {
        sign | ((exponent as u16) << 10) | (mantissa as u16)
    }
}
