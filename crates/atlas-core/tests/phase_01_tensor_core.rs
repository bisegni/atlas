use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use atlas_core::{
    DType, Shape, Storage, Strides, Tensor, f16_bits_to_f32, f32_to_f16_bits,
    read_safetensors_descriptors,
};

#[test]
fn tensor_views_validate_shape_stride_and_storage_bounds() {
    let storage = Storage::cpu(6 * DType::F32.byte_width(), false);
    let tensor = Tensor::contiguous(storage, Shape::new(vec![2, 3]).unwrap(), DType::F32).unwrap();
    assert!(tensor.is_contiguous());
    assert_eq!(tensor.strides.values(), &[3, 1]);

    let reshaped = tensor.reshape(Shape::new(vec![3, 2]).unwrap()).unwrap();
    assert_eq!(reshaped.strides.values(), &[2, 1]);
    let transposed = tensor.transpose(&[1, 0]).unwrap();
    assert_eq!(transposed.shape.dims(), &[3, 2]);
    assert_eq!(transposed.strides.values(), &[1, 3]);
    assert!(!transposed.is_contiguous());
    assert!(transposed.reshape(Shape::new(vec![6]).unwrap()).is_err());

    let view = tensor
        .view(
            Shape::new(vec![2]).unwrap(),
            Strides::new(vec![1], &Shape::new(vec![2]).unwrap()).unwrap(),
            4,
        )
        .unwrap();
    assert_eq!(view.offset_elements, 4);
    assert!(
        tensor
            .view(
                Shape::new(vec![2]).unwrap(),
                Strides::new(vec![1], &Shape::new(vec![2]).unwrap()).unwrap(),
                5
            )
            .is_err()
    );
}

#[test]
fn fp16_conversion_round_trips_representative_values() {
    for value in [0.0, -0.0, 1.0, -2.0, 0.33325, 12.5, 65504.0] {
        let decoded = f16_bits_to_f32(f32_to_f16_bits(value));
        assert!((decoded - value).abs() <= 0.0005_f32.max(value.abs() * 0.001));
    }
}

#[test]
fn safetensors_header_creates_read_only_descriptors_without_reading_payload() {
    let header = br#"{"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}}"#;
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("atlas-phase-01-{unique}.safetensors"));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&[0; 8]);
    fs::write(&path, bytes).unwrap();

    let descriptors = read_safetensors_descriptors(&path).unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(descriptors.len(), 1);
    assert_eq!(descriptors[0].name, "weight");
    assert_eq!(descriptors[0].tensor.dtype, DType::F16);
    assert!(descriptors[0].tensor.storage.read_only);
    assert_eq!(descriptors[0].data_end - descriptors[0].data_start, 8);
}
