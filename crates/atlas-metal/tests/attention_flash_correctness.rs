//! Tolerance-based correctness check for the single-dispatch flash16
//! attention kernels against the production two-pass pipeline. The speed-first
//! policy replaced the bitwise parity gate, so this asserts a small
//! max-absolute-difference instead of byte equality. Covers the Gemma4 E2B
//! geometries: 512-wide full-context heads and 256-wide sliding-window heads.

use atlas_metal::{MetalError, MetalRuntime};

const HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const CAPACITY: u32 = 2048;
const BLOCKS: u32 = 4;

fn half_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    if exp >= 31 {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | mantissa as u16
}

fn build_cache(key_count: u32, head_dim: u32, rising_scores: bool) -> Vec<u8> {
    let blocks_per_position = (KV_HEADS * head_dim) / 32;
    let key_blocks = key_count * blocks_per_position;
    let value_blocks = CAPACITY * blocks_per_position;
    let mut cache = vec![0u8; ((key_blocks + value_blocks) * 18) as usize];
    for (i, chunk) in cache.chunks_mut(18).enumerate() {
        let scale = if rising_scores {
            0.2 + (i as f32 % 37.0) * 0.021
        } else {
            0.3 + ((i as f32) * 0.11).fract() * 0.4
        };
        let half = half_f32(scale);
        chunk[..2].copy_from_slice(&half.to_le_bytes());
        for (j, byte) in chunk[2..].iter_mut().enumerate() {
            let nibble = if rising_scores {
                (i as u8) % 9
            } else {
                ((i * 5 + j * 11) % 256) as u8
            };
            *byte = nibble | (nibble << 4);
        }
    }
    cache
}

fn build_query(head_dim: u32) -> Vec<f32> {
    let mut query = vec![0.0f32; (HEADS * head_dim) as usize];
    for (h, head) in query.chunks_mut(head_dim as usize).enumerate() {
        for (d, value) in head.iter_mut().enumerate() {
            *value =
                ((d as f32 * 0.17 + h as f32 * 0.41).cos() * 0.6) * (1.0 + 0.08 * (d % 13) as f32);
        }
    }
    query
}

fn run_production(
    runtime: &MetalRuntime,
    query: &[f32],
    cache: &[u8],
    head_dim: u32,
    key_count: u32,
) -> Vec<f32> {
    let partial_count = (BLOCKS * HEADS * head_dim) as usize;
    let slot_count = (BLOCKS * HEADS) as usize;
    let partials = vec![0.0f32; partial_count];
    let maxima = vec![0.0f32; slot_count];
    let sums = vec![0.0f32; slot_count];
    let mut output = vec![0.0f32; (HEADS * head_dim) as usize];

    let query_buf = runtime.upload_f32(query).unwrap();
    let cache_buf = runtime.upload_bytes(cache).unwrap();
    let partials_buf = runtime.upload_f32(&partials).unwrap();
    let maxima_buf = runtime.upload_f32(&maxima).unwrap();
    let sums_buf = runtime.upload_f32(&sums).unwrap();
    let heads_buf = runtime.upload_u32(&[HEADS]).unwrap();
    let kv_heads_buf = runtime.upload_u32(&[KV_HEADS]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim]).unwrap();
    let capacity_buf = runtime.upload_u32(&[CAPACITY]).unwrap();
    let key_count_buf = runtime.upload_u32(&[key_count]).unwrap();
    let blocks_buf = runtime.upload_u32(&[BLOCKS]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "attention_decode_fused_gemma4_simd_q4_0_2pass_1_no_value_barrier",
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
            (HEADS * BLOCKS) as usize,
            128,
        )
        .unwrap();
    let output_buf = runtime.upload_f32(&output).unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "attention_decode_fused_gemma4_simd_q4_0_2pass_2",
            &[
                (&partials_buf, 0),
                (&maxima_buf, 0),
                (&sums_buf, 0),
                (&output_buf, 0),
                (&heads_buf, 0),
                (&head_dim_buf, 0),
                (&blocks_buf, 0),
            ],
            HEADS as usize,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    let got = runtime.read_f32(&output_buf, output.len()).unwrap();
    output.copy_from_slice(&got);
    output
}

fn run_flash16(
    runtime: &MetalRuntime,
    kernel: &'static str,
    threads: usize,
    query: &[f32],
    cache: &[u8],
    head_dim: u32,
    key_count: u32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; (HEADS * head_dim) as usize];
    let query_buf = runtime.upload_f32(query).unwrap();
    let cache_buf = runtime.upload_bytes(cache).unwrap();
    let output_buf = runtime.upload_f32(&output).unwrap();
    let heads_buf = runtime.upload_u32(&[HEADS]).unwrap();
    let kv_heads_buf = runtime.upload_u32(&[KV_HEADS]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim]).unwrap();
    let capacity_buf = runtime.upload_u32(&[CAPACITY]).unwrap();
    let key_count_buf = runtime.upload_u32(&[key_count]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(
            kernel,
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&output_buf, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_count_buf, 0),
            ],
            HEADS as usize,
            threads,
        )
        .unwrap();
    command.finish().unwrap();

    let got = runtime.read_f32(&output_buf, output.len()).unwrap();
    output.copy_from_slice(&got);
    output
}

fn compare(label: &str, reference: &[f32], candidate: &[f32]) {
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
        "{label}: flash16 output diverges from the production pipeline (max_abs={max_abs:.3e})"
    );
}

#[test]
fn flash16_matches_production_pipeline() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    let scenarios = [
        (
            "full-512",
            512u32,
            256usize,
            "attention_decode_gemma4_simd_q4_0_flash16",
        ),
        (
            "swa-256",
            256u32,
            512usize,
            "attention_decode_gemma4_simd_q4_0_flash16_swa",
        ),
        (
            "full-512-u",
            512u32,
            256usize,
            "attention_decode_gemma4_simd_q4_0_flash16_u",
        ),
        (
            "swa-256-u",
            256u32,
            512usize,
            "attention_decode_gemma4_simd_q4_0_flash16_swa_u",
        ),
        (
            "full-512-uw",
            512u32,
            384usize,
            "attention_decode_gemma4_simd_q4_0_flash16_uw",
        ),
        (
            "swa-256-uw",
            256u32,
            768usize,
            "attention_decode_gemma4_simd_q4_0_flash16_swa_uw",
        ),
    ];
    for (label, head_dim, threads, kernel) in scenarios {
        let query = build_query(head_dim);
        for key_count in [48u32, 256u32, 1024u32, 2048u32] {
            for rising in [false, true] {
                let cache = build_cache(key_count, head_dim, rising);
                let reference = run_production(&runtime, &query, &cache, head_dim, key_count);
                let candidate = run_flash16(
                    &runtime, kernel, threads, &query, &cache, head_dim, key_count,
                );
                compare(
                    &format!("{label} keys={key_count} rising={rising}"),
                    &reference,
                    &candidate,
                );
            }
        }
    }
}
