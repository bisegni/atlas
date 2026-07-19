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
