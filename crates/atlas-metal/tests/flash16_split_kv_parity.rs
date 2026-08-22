//! Tolerance parity for the split-KV ("flash-decoding") decode Flash16
//! attention path.  Unlike the single-dispatch v4 kernel (which splits the
//! key range across SIMD groups within one threadgroup), the split-KV path
//! scans the per-head KV cache across `split` threadgroups, each emitting a
//! per-chunk partial softmax state to device scratch, then merges them in a
//! second dispatch.  The cross-threadgroup split and merge change the FP32
//! reduction order (like v4), so it is gated by the established tolerance
//! contract rather than byte equality: max-abs < 1e-3 against the independent
//! CPU Q4_0 oracle.  This test exercises the two-pass scan+combine
//! orchestration and the empty/degenerate-slice guards (zero key_count, and
//! key_count < split which forces some threadgroups to write degenerate
//! partials).

use atlas_core::{GgufTensorType, dequantize_block};
use atlas_metal::{MetalError, MetalRuntime};

const HEADS: u32 = 8;
const KV_HEADS: u32 = 1;
const CAPACITY: u32 = 2048;
const MAX_SPLIT: u32 = 32;
const COMBINE_THREADS: usize = 128;

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
    sign | ((exp as u16) << 10) | mantissa as u32 as u16
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

fn run_split(
    runtime: &MetalRuntime,
    sliding: bool,
    split: u32,
    query: &[f32],
    cache: &[u8],
    head_dim: u32,
    key_count: u32,
) -> Vec<f32> {
    let output_len = (HEADS * head_dim) as usize;
    let output = vec![0.0f32; output_len];
    let query_buf = runtime.upload_f32(query).unwrap();
    let cache_buf = runtime.upload_bytes(cache).unwrap();
    let output_buf = runtime.upload_f32(&output).unwrap();
    let heads_buf = runtime.upload_u32(&[HEADS]).unwrap();
    let kv_heads_buf = runtime.upload_u32(&[KV_HEADS]).unwrap();
    let head_dim_buf = runtime.upload_u32(&[head_dim]).unwrap();
    let capacity_buf = runtime.upload_u32(&[CAPACITY]).unwrap();
    let key_control = key_count & 0xffff;
    let key_controls = runtime.upload_u32(&[key_control]).unwrap();
    let split_buf = runtime.upload_u32(&[split]).unwrap();

    let stride = head_dim + 2;
    let partials_len = (HEADS as usize) * (MAX_SPLIT as usize) * (stride as usize);
    let partials_buf = runtime
        .allocate(partials_len * std::mem::size_of::<f32>())
        .unwrap();
    let partial_stride_buf = runtime.upload_u32(&[stride]).unwrap();

    let (scan_kernel, scan_threads) = if sliding {
        (
            "attention_decode_gemma4_simd_q4_0_flash16_split_swa_scan",
            24 * 32u32,
        )
    } else {
        (
            "attention_decode_gemma4_simd_q4_0_flash16_split_full_scan",
            12 * 32u32,
        )
    };
    let combine_kernel = if sliding {
        "attention_decode_gemma4_simd_q4_0_flash16_split_swa_combine"
    } else {
        "attention_decode_gemma4_simd_q4_0_flash16_split_full_combine"
    };

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(
            scan_kernel,
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&partials_buf, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_controls, 0),
                (&split_buf, 0),
                (&partial_stride_buf, 0),
            ],
            (HEADS as usize) * (split as usize),
            usize::try_from(scan_threads).unwrap(),
        )
        .unwrap();
    command
        .dispatch_threadgroups_1d_at(
            combine_kernel,
            &[
                (&partials_buf, 0),
                (&output_buf, 0),
                (&heads_buf, 0),
                (&partial_stride_buf, 0),
                (&split_buf, 0),
            ],
            HEADS as usize,
            COMBINE_THREADS,
        )
        .unwrap();
    command.finish().unwrap();

    runtime.read_f32(&output_buf, output_len).unwrap()
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
        "{label}: split-KV attention diverges from the CPU oracle (max_abs={max_abs:.3e})"
    );
}

#[test]
fn flash16_split_kv_matches_cpu_oracle() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    // (sliding, head_dim, split, key_counts)  splits of 1, 4, 8, 16, 24, 32
    // sweep the scan grain; key_count == 0 exercises the empty-slice guard,
    // and a small key_count with a large split forces many degenerate
    // partials.  key_count stays within CAPACITY.
    let scenarios: [(bool, u32, u32, &[u32]); 6] = [
        (false, 512, 1, &[48, 2048, 64, 0]),
        (false, 512, 8, &[4, 128, 1024, 2048, 0]),
        (false, 512, 32, &[128, 1024, 2048, 0]),
        (true, 256, 1, &[128, 256, 0]),
        (true, 256, 24, &[256, 1024, 2048, 0]),
        (true, 256, 32, &[64, 1024, 2048, 0]),
    ];
    for scenario in scenarios.iter() {
        let (sliding, head_dim, split, key_counts) = *scenario;
        let query = build_query(head_dim);
        for &key_count in key_counts {
            for rising in [false, true] {
                let cache = build_cache(key_count, head_dim, rising);
                let reference = cpu_attention(&query, &cache, head_dim, key_count);
                let candidate = run_split(
                    &runtime, sliding, split, &query, &cache, head_dim, key_count,
                );
                compare(
                    &format!(
                        "{}/{}/keys={}/split={}/rising={}",
                        if sliding { "swa" } else { "full" },
                        head_dim,
                        key_count,
                        split,
                        rising
                    ),
                    &reference,
                    &candidate,
                );
            }
        }
    }
}
