//! Bitwise parity for the phase-13.0 P2b KV-append + provider-V-norm
//! fusion.  The unfused reference path normalizes the provider V vector in
//! place (rms_norm_groups_in_place_unweighted_f32, single group) and then
//! appends K/V to the cache (kv_append_decode_*).  The fused kernels fold
//! the group RMS into the append dispatch, recomputing the same sequential
//! sumsq per block thread, so the quantized cache bytes must be identical
//! and the raw V buffer must remain untouched.

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

/// Runs the unfused reference path: in-place V group norm, then the append
/// kernel, and returns the full cache bytes as read back through f32 slices.
fn reference_append(
    runtime: &MetalRuntime,
    key: &atlas_metal::GpuBuffer,
    value: &atlas_metal::GpuBuffer,
    cache: &atlas_metal::GpuBuffer,
    width_buf: &atlas_metal::GpuBuffer,
    one_buf: &atlas_metal::GpuBuffer,
    epsilon_buf: &atlas_metal::GpuBuffer,
    capacity_buf: &atlas_metal::GpuBuffer,
    position_buf: &atlas_metal::GpuBuffer,
    append_kernel: &'static str,
    append_threads: usize,
) {
    dispatch_1d(
        runtime,
        "rms_norm_groups_in_place_unweighted_f32",
        &[value, width_buf, one_buf, epsilon_buf],
        append_threads,
    );
    dispatch_1d(
        runtime,
        append_kernel,
        &[key, value, cache, width_buf, capacity_buf, position_buf],
        append_threads,
    );
}

fn fused_append(
    runtime: &MetalRuntime,
    key: &atlas_metal::GpuBuffer,
    value: &atlas_metal::GpuBuffer,
    cache: &atlas_metal::GpuBuffer,
    width_buf: &atlas_metal::GpuBuffer,
    capacity_buf: &atlas_metal::GpuBuffer,
    position_buf: &atlas_metal::GpuBuffer,
    epsilon_buf: &atlas_metal::GpuBuffer,
    fused_kernel: &'static str,
    append_threads: usize,
) {
    dispatch_1d(
        runtime,
        fused_kernel,
        &[
            key,
            value,
            cache,
            width_buf,
            capacity_buf,
            position_buf,
            epsilon_buf,
        ],
        append_threads,
    );
}

fn compare_caches(label: &str, reference: &[f32], candidate: &[f32]) {
    assert_eq!(reference.len(), candidate.len());
    for (index, (r, c)) in reference.iter().zip(candidate.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            c.to_bits(),
            "{label}: fused cache diverges from the unfused reference at element {index}"
        );
    }
    eprintln!(
        "{label}: {len} cache elements bitwise identical",
        len = reference.len()
    );
}

fn run_round(
    runtime: &MetalRuntime,
    state: &mut u32,
    kv_width: u32,
    capacity: u32,
    position: u32,
    epsilon: f32,
    append_kernel: &'static str,
    fused_kernel: &'static str,
) {
    let width = kv_width as usize;
    let blocks = width / 32;
    let packed = !append_kernel.ends_with("f32");
    let block_bytes = if append_kernel.ends_with("q8_0") {
        34
    } else {
        18
    };
    let cache_floats = if packed {
        (2 * blocks * block_bytes * capacity as usize) / 4
    } else {
        2 * width * capacity as usize
    };

    let mut key = vec![0.0f32; width];
    fill_f32(&mut key, state);
    let mut value = vec![0.0f32; width];
    fill_f32(&mut value, state);
    let key_buf = runtime.upload_f32(&key).unwrap();
    let width_buf = runtime.upload_u32(&[kv_width]).unwrap();
    let one_buf = runtime.upload_u32(&[1]).unwrap();
    let epsilon_buf = runtime.upload_f32(&[epsilon]).unwrap();
    let capacity_buf = runtime.upload_u32(&[capacity]).unwrap();
    let position_buf = runtime.upload_u32(&[position]).unwrap();
    let zero_cache = vec![0.0f32; cache_floats];

    let value_ref_buf = runtime.upload_f32(&value).unwrap();
    let cache_ref = runtime.upload_f32(&zero_cache).unwrap();
    reference_append(
        runtime,
        &key_buf,
        &value_ref_buf,
        &cache_ref,
        &width_buf,
        &one_buf,
        &epsilon_buf,
        &capacity_buf,
        &position_buf,
        append_kernel,
        width,
    );
    let reference = runtime.read_f32(&cache_ref, cache_floats).unwrap();
    let normalized_value = runtime.read_f32(&value_ref_buf, width).unwrap();

    let value_fused_buf = runtime.upload_f32(&value).unwrap();
    let cache_fused = runtime.upload_f32(&zero_cache).unwrap();
    let fused_threads = if append_kernel.ends_with("f32") {
        width
    } else {
        width / 32
    };
    fused_append(
        runtime,
        &key_buf,
        &value_fused_buf,
        &cache_fused,
        &width_buf,
        &capacity_buf,
        &position_buf,
        &epsilon_buf,
        fused_kernel,
        fused_threads,
    );
    let candidate = runtime.read_f32(&cache_fused, cache_floats).unwrap();
    compare_caches(
        &format!("{fused_kernel} width={kv_width} capacity={capacity} position={position}"),
        &reference,
        &candidate,
    );
    let untouched = runtime.read_f32(&value_fused_buf, width).unwrap();
    assert_eq!(
        value, untouched,
        "fused kernel must not mutate the raw V buffer"
    );
    assert_ne!(
        value, normalized_value,
        "reference in-place norm must have modified V (sanity)"
    );
}

#[test]
fn fused_append_f32_matches_value_norm_then_append() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x1f2a_3b4cu32;
    for kv_width in [64u32, 128u32, 256u32] {
        for position in [0u32, 2u32] {
            run_round(
                &runtime,
                &mut state,
                kv_width,
                8,
                position,
                1e-6,
                "kv_append_decode_f32",
                "kv_append_decode_f32_vnorm",
            );
        }
    }
}

#[test]
fn fused_append_q8_0_matches_value_norm_then_append() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x5a6b_7c8du32;
    for kv_width in [64u32, 128u32, 256u32] {
        for position in [0u32, 2u32] {
            run_round(
                &runtime,
                &mut state,
                kv_width,
                8,
                position,
                1e-6,
                "kv_append_decode_q8_0",
                "kv_append_decode_q8_0_vnorm",
            );
        }
    }
}

#[test]
fn fused_append_q4_0_matches_value_norm_then_append() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut state = 0x9e10_a1b2u32;
    for kv_width in [64u32, 128u32, 256u32] {
        for position in [0u32, 2u32] {
            run_round(
                &runtime,
                &mut state,
                kv_width,
                8,
                position,
                1e-6,
                "kv_append_decode_q4_0",
                "kv_append_decode_q4_0_vnorm",
            );
        }
    }
}
