use atlas_core::{QuantFormat, QuantizedMatrix};

fn source(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|index| ((index as f32 * 0.37).sin() * 2.0) - 0.4)
        .collect()
}

#[test]
fn block32_formats_preserve_matvec_and_matmul_oracles() {
    let weights = source(5, 65);
    let input = source(3, 65);
    for format in [
        QuantFormat::Fp16,
        QuantFormat::Int8Block32,
        QuantFormat::Q4Block32,
    ] {
        let packed = QuantizedMatrix::quantize(&weights, 5, 65, format).unwrap();
        let vector = packed.matvec_cpu(&input[..65]).unwrap();
        let matrix = packed.matmul_cpu(&input, 3).unwrap();
        assert_eq!(vector.len(), 5);
        assert_eq!(matrix.len(), 15);
        for output in 0..5 {
            assert!((matrix[output] - vector[output]).abs() < 1e-5);
        }
        assert!(packed.resident_bytes() < weights.len() * 4);
    }
}

#[test]
fn q4_is_packed_and_rejects_invalid_values() {
    let packed = QuantizedMatrix::quantize(&source(2, 65), 2, 65, QuantFormat::Q4Block32).unwrap();
    assert_eq!(packed.data.len(), (2usize * 65).div_ceil(2));
    assert!(QuantizedMatrix::quantize(&[f32::NAN], 1, 1, QuantFormat::Q4Block32).is_err());
}
