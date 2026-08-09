//! Buffer-level parity diagnostics for the decode Q4 FFN Gate/Up fusion.
//! The fused candidate `matmul_q4_0_gate_up_gelu_16row` must reproduce the
//! reference sequence (matmul_q4_0_gate_up_16row + gelu_f32 +
//! vector_multiply_f32) bitwise for every output element.

use atlas_metal::{MetalError, MetalRuntime};

fn half_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mantissa = ((bits >> 13) & 0x03ff) as u16;
    if exponent >= 31 {
        sign | 0x7c00
    } else if exponent <= 0 {
        sign
    } else {
        sign | ((exponent as u16) << 10) | mantissa
    }
}

fn build_q4_rows(row_count: usize, input_width: usize, seed: u32) -> Vec<u8> {
    let blocks = input_width / 32;
    let mut bytes = vec![0u8; row_count * blocks * 18];
    for (i, chunk) in bytes.chunks_mut(18).enumerate() {
        let scale = 0.1 + ((i as f32 * 0.07 + seed as f32 * 0.13) % 1.0) * 0.8;
        let half = half_f32(scale);
        chunk[..2].copy_from_slice(&half.to_le_bytes());
        for (j, byte) in chunk[2..].iter_mut().enumerate() {
            *byte = ((i * 13 + j * 7 + seed as usize * 31) % 256) as u8;
        }
    }
    bytes
}

fn gate_up_sums(
    runtime: &MetalRuntime,
    kernel: &'static str,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    output_width: usize,
) -> (Vec<f32>, Vec<f32>) {
    let input_width = input.len();
    let input_buf = runtime.upload_f32(input).unwrap();
    let gate_buf = runtime.upload_bytes(gate_weights).unwrap();
    let up_buf = runtime.upload_bytes(up_weights).unwrap();
    let gate_out = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let up_out = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let activated = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let product = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[output_width as u32]).unwrap();
    let count_buf = runtime.upload_u32(&[output_width as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let groups = output_width.div_ceil(16);
    command
        .dispatch_threadgroups_1d_at(
            kernel,
            &[
                (&input_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&gate_out, 0),
                (&up_out, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
            ],
            2 * groups,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    (
        runtime.read_f32(&gate_out, output_width).unwrap(),
        runtime.read_f32(&up_out, output_width).unwrap(),
    )
}

fn reference_sequence(
    runtime: &MetalRuntime,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    output_width: usize,
) -> Vec<f32> {
    let input_width = input.len();
    let input_buf = runtime.upload_f32(input).unwrap();
    let gate_buf = runtime.upload_bytes(gate_weights).unwrap();
    let up_buf = runtime.upload_bytes(up_weights).unwrap();
    let gate_out = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let up_out = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let activated = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let product = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[output_width as u32]).unwrap();
    let count_buf = runtime.upload_u32(&[output_width as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let groups = output_width.div_ceil(16);
    command
        .dispatch_threadgroups_1d_at(
            "matmul_q4_0_gate_up_16row",
            &[
                (&input_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&gate_out, 0),
                (&up_out, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
            ],
            2 * groups,
            128,
        )
        .unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "gelu_f32",
            &[(&gate_out, 0), (&activated, 0), (&count_buf, 0)],
            output_width,
            128,
        )
        .unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "vector_multiply_f32",
            &[
                (&activated, 0),
                (&up_out, 0),
                (&product, 0),
                (&count_buf, 0),
            ],
            output_width,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    runtime.read_f32(&product, output_width).unwrap()
}

fn fused_output(
    runtime: &MetalRuntime,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    output_width: usize,
) -> Vec<f32> {
    let input_width = input.len();
    let input_buf = runtime.upload_f32(input).unwrap();
    let gate_buf = runtime.upload_bytes(gate_weights).unwrap();
    let up_buf = runtime.upload_bytes(up_weights).unwrap();
    let product = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[output_width as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let groups = output_width.div_ceil(16);
    command
        .dispatch_threadgroups_1d_at(
            "matmul_q4_0_gate_up_gelu_16row",
            &[
                (&input_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&product, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
            ],
            groups,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    runtime.read_f32(&product, output_width).unwrap()
}

fn fused_raw_sums(
    runtime: &MetalRuntime,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    output_width: usize,
) -> (Vec<f32>, Vec<f32>) {
    fused_raw_sums_for(
        runtime,
        "matmul_q4_0_gate_up_gelu_16row_dump_sums",
        input,
        gate_weights,
        up_weights,
        output_width,
    )
}

fn fused_raw_sums_for(
    runtime: &MetalRuntime,
    kernel: &'static str,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    output_width: usize,
) -> (Vec<f32>, Vec<f32>) {
    let input_width = input.len();
    let input_buf = runtime.upload_f32(input).unwrap();
    let gate_buf = runtime.upload_bytes(gate_weights).unwrap();
    let up_buf = runtime.upload_bytes(up_weights).unwrap();
    let product = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let gate_dump = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let up_dump = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[output_width as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let groups = output_width.div_ceil(16);
    command
        .dispatch_threadgroups_1d_at(
            kernel,
            &[
                (&input_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&product, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
                (&gate_dump, 0),
                (&up_dump, 0),
            ],
            groups,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    (
        runtime.read_f32(&gate_dump, output_width).unwrap(),
        runtime.read_f32(&up_dump, output_width).unwrap(),
    )
}

fn fused_product(
    runtime: &MetalRuntime,
    kernel: &'static str,
    input: &[f32],
    gate_weights: &[u8],
    up_weights: &[u8],
    output_width: usize,
) -> Vec<f32> {
    let input_width = input.len();
    let input_buf = runtime.upload_f32(input).unwrap();
    let gate_buf = runtime.upload_bytes(gate_weights).unwrap();
    let up_buf = runtime.upload_bytes(up_weights).unwrap();
    let product = runtime.upload_f32(&vec![0.0; output_width]).unwrap();
    let input_width_buf = runtime.upload_u32(&[input_width as u32]).unwrap();
    let output_width_buf = runtime.upload_u32(&[output_width as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    let groups = output_width.div_ceil(16);
    command
        .dispatch_threadgroups_1d_at(
            kernel,
            &[
                (&input_buf, 0),
                (&gate_buf, 0),
                (&up_buf, 0),
                (&product, 0),
                (&input_width_buf, 0),
                (&output_width_buf, 0),
            ],
            groups,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    runtime.read_f32(&product, output_width).unwrap()
}

#[test]
fn gelu_multiply_fused_matches_reference_pair_bitwise() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    let count = 8192usize;
    let mut gate: Vec<f32> = (0..count)
        .map(|i| {
            let t = i as f32 / 64.0;
            (t.sin() * 8.0 + t * 0.7) * 0.9
        })
        .collect();
    gate[100] = 10.0;
    gate[101] = -10.0;
    gate[102] = 10.00001;
    gate[103] = -10.00001;
    gate[104] = 1.0e20;
    gate[105] = -1.0e20;
    gate[106] = 3.0e19;
    gate[107] = -3.0e19;
    gate[108] = 0.0;
    gate[109] = -0.0;
    let up: Vec<f32> = (0..count)
        .map(|i| {
            let t = i as f32 / 97.0;
            (t.cos() * 3.0 - t * 0.3) * 1.1
        })
        .collect();

    let gate_buf = runtime.upload_f32(&gate).unwrap();
    let up_buf = runtime.upload_f32(&up).unwrap();
    let activated = runtime.upload_f32(&vec![0.0; count]).unwrap();
    let reference = runtime.upload_f32(&vec![0.0; count]).unwrap();
    let fused = runtime.upload_f32(&vec![0.0; count]).unwrap();
    let count_buf = runtime.upload_u32(&[count as u32]).unwrap();

    let mut command = runtime.begin_resident_command().unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "gelu_f32",
            &[(&gate_buf, 0), (&activated, 0), (&count_buf, 0)],
            count,
            128,
        )
        .unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "vector_multiply_f32",
            &[
                (&activated, 0),
                (&up_buf, 0),
                (&reference, 0),
                (&count_buf, 0),
            ],
            count,
            128,
        )
        .unwrap();
    command
        .dispatch_threadgroups_1d_at(
            "gelu_multiply_f32",
            &[(&gate_buf, 0), (&up_buf, 0), (&fused, 0), (&count_buf, 0)],
            count,
            128,
        )
        .unwrap();
    command.finish().unwrap();

    let reference = runtime.read_f32(&reference, count).unwrap();
    let fused = runtime.read_f32(&fused, count).unwrap();

    let differing = reference
        .iter()
        .zip(fused.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let mut shown = 0;
    for (i, (a, b)) in reference.iter().zip(fused.iter()).enumerate() {
        if a.to_bits() != b.to_bits() && shown < 6 {
            eprintln!("  index {i}: reference={a:.9e} fused={b:.9e}");
            shown += 1;
        }
    }
    eprintln!("gelu_multiply fused: differing {differing}/{count}");
    assert_eq!(
        differing, 0,
        "gelu_multiply_f32 differs from the gelu_f32 + vector_multiply_f32 reference pair"
    );
}

#[test]
fn gate_up_gelu_fused_matches_reference_bitwise() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };

    let input_width = 1536usize;
    let output_width = 7168usize;
    let input: Vec<f32> = (0..input_width)
        .map(|d| (d as f32 * 0.11).sin() * 0.9)
        .collect();
    let gate_weights = build_q4_rows(output_width, input_width, 8);
    let up_weights = build_q4_rows(output_width, input_width, 9);

    let (gate_a, up_a) = gate_up_sums(
        &runtime,
        "matmul_q4_0_gate_up_16row",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    let (gate_b, up_b) = gate_up_sums(
        &runtime,
        "matmul_q4_0_gate_up_16row",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    let (gate_t, up_t) = gate_up_sums(
        &runtime,
        "matmul_q4_0_gate_up_16row_simdgroup_tiled",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    let (gate_f, up_f) = fused_raw_sums(&runtime, &input, &gate_weights, &up_weights, output_width);

    let compare = |name: &str, a: &[f32], b: &[f32]| {
        let differing = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        eprintln!("{name}: differing {differing}/{}", a.len());
        let mut shown = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x.to_bits() != y.to_bits() && shown < 4 {
                eprintln!(
                    "  row {i}: plain={x:.9e} fused={y:.9e} ulp_diff={}",
                    (x.to_bits() as i64 - y.to_bits() as i64).abs()
                );
                shown += 1;
            }
        }
        differing
    };

    assert_eq!(
        compare("gate run1 vs run2", &gate_a, &gate_b),
        0,
        "reference matmul is not deterministic"
    );
    assert_eq!(
        compare("up run1 vs run2", &up_a, &up_b),
        0,
        "reference matmul is not deterministic"
    );
    compare("gate plain vs tiled", &gate_a, &gate_t);
    compare("up plain vs tiled", &up_a, &up_t);
    compare("gate plain vs fused raw", &gate_a, &gate_f);
    compare("up plain vs fused raw", &up_a, &up_f);

    let (gate_i, up_i) = fused_raw_sums_for(
        &runtime,
        "matmul_q4_0_gate_up_gelu_16row_inline_loads",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    compare("gate plain vs inline-loads raw", &gate_a, &gate_i);
    compare("up plain vs inline-loads raw", &up_a, &up_i);

    let reference_product =
        reference_sequence(&runtime, &input, &gate_weights, &up_weights, output_width);
    let inline_product = fused_product(
        &runtime,
        "matmul_q4_0_gate_up_gelu_16row_inline_loads",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    compare(
        "reference product vs inline-loads product",
        &reference_product,
        &inline_product,
    );
    let split_product = fused_product(
        &runtime,
        "matmul_q4_0_gate_up_gelu_16row_split_loops",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    compare(
        "reference product vs split-loops product",
        &reference_product,
        &split_product,
    );

    let nogelu_product = fused_product(
        &runtime,
        "matmul_q4_0_gate_up_gelu_16row_exp_nogelu",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    compare(
        "reference product vs nogelu product",
        &reference_product,
        &nogelu_product,
    );

    let (nogelu_gate, nogelu_up) = fused_raw_sums_for(
        &runtime,
        "matmul_q4_0_gate_up_gelu_16row_exp_nogelu",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    compare("gate plain vs nogelu raw", &gate_a, &nogelu_gate);
    compare("up plain vs nogelu raw", &up_a, &nogelu_up);

    let fma_product = fused_product(
        &runtime,
        "matmul_q4_0_gate_up_gelu_16row_exp_fma",
        &input,
        &gate_weights,
        &up_weights,
        output_width,
    );
    compare(
        "reference product vs exp-fma product",
        &reference_product,
        &fma_product,
    );
}
