use atlas_metal::{MetalError, MetalRuntime};

#[test]
#[ignore = "requires local Metal"]
fn fused_gemma4_simd_f32_attention_matches_cpu_oracle() -> Result<(), Box<dyn std::error::Error>> {
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 64usize;
    let capacity = 8usize;
    let key_count = 6usize;

    let mut cache = vec![0.0f32; 2 * capacity * kv_heads * head_dim];
    let mut query = vec![0.0f32; heads * head_dim];
    for position in 0..key_count {
        for d in 0..head_dim {
            let v = (position as f32 * 0.37 + d as f32 * 0.11).sin();
            cache[position * kv_heads * head_dim + d] = v;
            cache[capacity * kv_heads * head_dim + position * kv_heads * head_dim + d] =
                (position as f32 * 0.23 - d as f32 * 0.07).cos();
        }
    }
    for head in 0..heads {
        for d in 0..head_dim {
            query[head * head_dim + d] = (head as f32 * 1.7 + d as f32 * 0.05).cos();
        }
    }

    let mut expected = vec![0.0f32; heads * head_dim];
    for head in 0..heads {
        let kv_head = head / (heads / kv_heads);
        let mut maximum = f32::NEG_INFINITY;
        let mut denominator = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];
        for key in 0..key_count {
            let key_base = key * kv_heads * head_dim + kv_head * head_dim;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += query[head * head_dim + d] * cache[key_base + d];
            }
            let value_base = capacity * kv_heads * head_dim;
            if score > maximum {
                let rescale = (maximum - score).exp();
                maximum = score;
                denominator = denominator * rescale + 1.0;
                for d in 0..head_dim {
                    acc[d] *= rescale;
                    acc[d] += cache[value_base + key_base + d];
                }
            } else {
                let weight = (score - maximum).exp();
                denominator += weight;
                for d in 0..head_dim {
                    acc[d] += weight * cache[value_base + key_base + d];
                }
            }
        }
        for d in 0..head_dim {
            expected[head * head_dim + d] = acc[d] / denominator;
        }
    }

    let runtime = MetalRuntime::new().expect("Metal runtime");
    let allocate = |count: usize| -> Result<atlas_metal::GpuBuffer, MetalError> {
        runtime.allocate(count * 4)
    };
    let query_buf = runtime.upload_f32(&query).expect("upload query");
    let cache_buf = runtime.upload_f32(&cache).expect("upload cache");
    let output = allocate(heads * head_dim)?;
    let heads_buf = runtime.upload_u32(&[heads as u32])?;
    let kv_heads_buf = runtime.upload_u32(&[kv_heads as u32])?;
    let head_dim_buf = runtime.upload_u32(&[head_dim as u32])?;
    let capacity_buf = runtime.upload_u32(&[capacity as u32])?;
    let key_count_buf = runtime.upload_u32(&[key_count as u32])?;

    let mut command = runtime.begin_resident_command().expect("begin command");
    command
        .dispatch_threadgroups_1d_at(
            "attention_decode_fused_gemma4_simd_f32",
            &[
                (&query_buf, 0),
                (&cache_buf, 0),
                (&output, 0),
                (&heads_buf, 0),
                (&kv_heads_buf, 0),
                (&head_dim_buf, 0),
                (&capacity_buf, 0),
                (&key_count_buf, 0),
            ],
            heads,
            128,
        )
        .expect("dispatch fused f32 attention");
    command.finish().expect("finish command");
    let got = runtime
        .read_f32(&output, heads * head_dim)
        .expect("read output");

    let mut max_abs = 0.0f32;
    for (index, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if diff > 1e-3 {
            panic!("attention mismatch at {index}: got {a} expected {b} (max_abs {max_abs})");
        }
    }
    println!("fused f32 attention matches CPU oracle: max_abs={max_abs}");
    Ok(())
}
