//! Correctness checks for Gemma Q4 attention kernels. The historical
//! slice-merge Flash16 kernels (`flash16_uw`, `flash16_swa_uw`) are checked
//! against an independent
//! CPU oracle: Q4_0-dequantized dot-product scores, exact softmax, weighted
//! sum of dequantized values. The kernels use flash-style per-slice rescaling
//! that changes the accumulation order, so this asserts a small
//! max-absolute-difference instead of byte equality. Covers the Gemma4 E2B
//! geometries: 512-wide full-context heads and 256-wide sliding-window heads.

use atlas_core::{GgufTensorType, dequantize_block};
use atlas_metal::{MetalError, MetalRuntime};

const HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const CAPACITY: u32 = 2048;

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

fn cpu_attention(query: &[f32], cache: &[u8], head_dim: u32, key_count: u32) -> Vec<f32> {
    let blocks_per_position = (KV_HEADS * head_dim) / 32;
    let value_base = (CAPACITY * blocks_per_position * 18) as usize;
    let mut block = vec![0.0f32; 32];
    let mut out = vec![0.0f32; (HEADS * head_dim) as usize];
    for h in 0..HEADS {
        let kv_head = h / (HEADS / KV_HEADS);
        let mut scores = vec![0.0f32; key_count as usize];
        for t in 0..key_count {
            let mut score = 0.0f32;
            for b in 0..blocks_per_position {
                let block_off =
                    (((t * KV_HEADS + kv_head) * blocks_per_position + b) * 18) as usize;
                dequantize_block(
                    GgufTensorType::Q4_0,
                    &cache[block_off..block_off + 18],
                    &mut block,
                )
                .unwrap();
                for lane in 0..32 {
                    score += query[(h * head_dim + b * 32 + lane) as usize] * block[lane as usize];
                }
            }
            scores[t as usize] = score;
        }
        let maximum = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut weights = vec![0.0f32; key_count as usize];
        let mut denominator = 0.0f32;
        for t in 0..key_count as usize {
            weights[t] = (scores[t] - maximum).exp();
            denominator += weights[t];
        }
        for b in 0..blocks_per_position {
            for lane in 0..32 {
                let mut acc = 0.0f32;
                for t in 0..key_count {
                    let block_off = value_base
                        + ((t * KV_HEADS + kv_head) * blocks_per_position + b) as usize * 18;
                    dequantize_block(
                        GgufTensorType::Q4_0,
                        &cache[block_off..block_off + 18],
                        &mut block,
                    )
                    .unwrap();
                    acc += weights[t as usize] * block[lane as usize];
                }
                out[(h * head_dim + b * 32 + lane) as usize] = acc / denominator;
            }
        }
    }
    out
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
        "{label}: flash16 output diverges from the CPU oracle (max_abs={max_abs:.3e})"
    );
}

#[test]
fn flash16_uw_matches_cpu_oracle() {
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
        // The swa kernel is window-agnostic: the executor clamps key_count
        // for sliding-window heads, so exercise the same key counts here.
        let key_counts: &[u32] = if label.starts_with("swa") {
            &[48, 128, 256]
        } else {
            &[48, 256, 1024, 2048]
        };
        for &key_count in key_counts {
            for rising in [false, true] {
                let cache = build_cache(key_count, head_dim, rising);
                let reference = cpu_attention(&query, &cache, head_dim, key_count);
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

#[test]
fn flash16_exact_variants_match_legacy_fused_bitwise() {
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
            "full-512-exact",
            512u32,
            "attention_decode_gemma4_simd_q4_0_flash16_exact_runtime",
            &[48, 256, 1024, 2048][..],
        ),
        (
            "full-512-exact-no-value-barrier",
            512u32,
            "attention_decode_gemma4_simd_q4_0_flash16_exact_nb",
            &[48, 256, 1024, 2048][..],
        ),
        (
            "full-512-exact-v3",
            512u32,
            "attention_decode_gemma4_simd_q4_0_flash16_exact_v3",
            &[48, 256, 1024, 2048][..],
        ),
        (
            "swa-256-exact",
            256u32,
            "attention_decode_gemma4_simd_q4_0_flash16_swa_exact_runtime",
            &[48, 128, 256][..],
        ),
        (
            "swa-256-exact-no-value-barrier",
            256u32,
            "attention_decode_gemma4_simd_q4_0_flash16_swa_exact_nb",
            &[48, 128, 256][..],
        ),
        (
            "swa-256-exact-v3",
            256u32,
            "attention_decode_gemma4_simd_q4_0_flash16_swa_exact_v3",
            &[48, 128, 256][..],
        ),
    ];
    for (label, head_dim, flash_kernel, key_counts) in scenarios {
        let query = build_query(head_dim);
        for key_count in key_counts {
            for rising in [false, true] {
                let cache = build_cache(*key_count, head_dim, rising);
                let legacy = run_flash16(
                    &runtime,
                    "attention_decode_fused_gemma4_simd_q4_0",
                    128,
                    &query,
                    &cache,
                    head_dim,
                    *key_count,
                );
                let exact = run_flash16(
                    &runtime,
                    flash_kernel,
                    128,
                    &query,
                    &cache,
                    head_dim,
                    *key_count,
                );
                assert_eq!(
                    exact, legacy,
                    "{label}: exact Flash16 must preserve LegacyFused FP32 output for keys={key_count} rising={rising}"
                );
            }
        }
    }
}
