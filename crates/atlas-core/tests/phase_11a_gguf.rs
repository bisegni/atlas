use atlas_core::{
    GgufModel, GgufTensorType, GgufWriter, dequantize_block, quantize_q4_0, quantize_q8_0,
};

#[test]
fn q4_and_q8_blocks_round_trip_without_full_dequantization() {
    let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 7.0).collect();
    for (kind, bytes) in [
        (GgufTensorType::Q4_0, quantize_q4_0(&values).unwrap()),
        (GgufTensorType::Q8_0, quantize_q8_0(&values).unwrap()),
    ] {
        assert_eq!(bytes.len(), kind.block_bytes());
        let mut decoded = [0.0; 32];
        dequantize_block(kind, &bytes, &mut decoded).unwrap();
        let limit = if kind == GgufTensorType::Q4_0 {
            0.35
        } else {
            0.03
        };
        assert!(
            values
                .iter()
                .zip(decoded)
                .all(|(a, b)| (a - b).abs() <= limit)
        );
    }
}

#[test]
fn q6_k_block_dequantizes_its_packed_low_and_high_bits() {
    let mut bytes = vec![0u8; GgufTensorType::Q6K.block_bytes()];
    bytes[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5 in f16
    for index in 0..256usize {
        let value = (index % 64) as u8;
        if index.is_multiple_of(2) {
            bytes[2 + index / 2] |= value & 0x0f;
        } else {
            bytes[2 + index / 2] |= (value & 0x0f) << 4;
        }
        bytes[130 + index / 4] |= ((value >> 4) & 3) << ((index % 4) * 2);
    }
    bytes[194..210].fill(1);
    let mut decoded = [0.0; 256];
    dequantize_block(GgufTensorType::Q6K, &bytes, &mut decoded).unwrap();
    assert_eq!(decoded[0], -16.0);
    assert_eq!(decoded[32], 0.0);
    assert_eq!(decoded[63], 15.5);
}

#[test]
fn writer_and_reader_preserve_aligned_packed_tensors() {
    let mut writer = GgufWriter::new();
    writer.metadata("llama.block_count", "1");
    writer
        .push_tensor(
            "blk.0.attn_q.weight",
            vec![32, 32],
            GgufTensorType::Q4_0,
            quantize_q4_0(&vec![0.25; 1024]).unwrap(),
        )
        .unwrap();
    let model = GgufModel::from_bytes(writer.finish().unwrap()).unwrap();
    assert_eq!(model.metadata["general.architecture"], "llama");
    assert_eq!(model.tensors[0].tensor_type, GgufTensorType::Q4_0);
    assert_eq!(model.tensor_data(&model.tensors[0]).unwrap().len(), 32 * 18);
}

#[test]
fn malformed_gguf_is_rejected_before_tensor_access() {
    assert!(GgufModel::from_bytes(b"GGUF\x03\0\0\0".to_vec()).is_err());
}

#[test]
fn gemma4_e2b_fixture_header_is_read_when_available() {
    let Ok(path) = std::env::var("ATLAS_GEMMA4_GGUF") else {
        return;
    };
    let model = GgufModel::open(path).expect("read Gemma 4 E2B GGUF fixture");
    assert_eq!(
        model
            .metadata
            .get("general.architecture")
            .map(String::as_str),
        Some("gemma4")
    );
    assert!(
        model
            .tensors
            .iter()
            .any(|tensor| tensor.tensor_type == GgufTensorType::Q4_0)
    );
    assert!(
        model
            .tensors
            .iter()
            .any(|tensor| tensor.tensor_type == GgufTensorType::Q6K)
    );
}
