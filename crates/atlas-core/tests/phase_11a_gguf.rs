use atlas_core::{
    GgufMetadataArray, GgufModel, GgufTensorType, GgufWriter, dequantize_block, f16_bits_to_f32,
    quantize_q4_0, quantize_q8_0,
};

/// Independent test oracle for llama.cpp's `dequantize_row_q6_K` layout.
///
/// Keep this deliberately separate from Atlas's production decoder: a fixture
/// regression must be able to diagnose a shared indexing mistake rather than
/// merely restating it.
fn llama_cpp_q6_k_decode(block: &[u8]) -> ([f32; 256], u16) {
    assert_eq!(block.len(), GgufTensorType::Q6K.block_bytes());
    let scale_bits = u16::from_le_bytes([block[208], block[209]]);
    let d = f16_bits_to_f32(scale_bits);
    let mut values = [0.0; 256];
    for half in 0..2usize {
        for lane in 0..32usize {
            let qh = block[128 + half * 32 + lane];
            for stream in 0..4usize {
                let ql_offset = half * 64 + lane + if stream % 2 == 1 { 32 } else { 0 };
                let ql = block[ql_offset];
                let low = if stream < 2 { ql & 0x0f } else { ql >> 4 };
                let high = (qh >> (stream * 2)) & 0x03;
                let scale = block[192 + half * 8 + stream * 2 + lane / 16] as i8 as f32;
                let index = half * 128 + stream * 32 + lane;
                values[index] = (((high << 4) | low) as i32 - 32) as f32 * scale * d;
            }
        }
    }
    (values, scale_bits)
}

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
fn q4_0_uses_llama_cpp_half_block_nibble_order() {
    let mut block = vec![0u8; GgufTensorType::Q4_0.block_bytes()];
    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // 1.0 in f16
    for i in 0..16usize {
        block[2 + i] = (i as u8 & 0x0f) | (((15 - i) as u8) << 4);
    }

    let mut decoded = [0.0; 32];
    dequantize_block(GgufTensorType::Q4_0, &block, &mut decoded).unwrap();
    for i in 0..16usize {
        assert_eq!(decoded[i], i as f32 - 8.0);
        assert_eq!(decoded[i + 16], 7.0 - i as f32);
    }
}

#[test]
fn q6_k_block_dequantizes_its_packed_low_and_high_bits() {
    let mut bytes = vec![0u8; GgufTensorType::Q6K.block_bytes()];
    bytes[208..].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5 in f16
    for index in 0..256usize {
        let value = (index % 64) as u8;
        let half = index / 128;
        let within = index % 128;
        let stream = within / 32;
        let lane = within % 32;
        let ql = half * 64 + lane + if stream % 2 == 1 { 32 } else { 0 };
        if stream >= 2 {
            bytes[ql] |= (value & 0x0f) << 4;
        } else {
            bytes[ql] |= value & 0x0f;
        }
        bytes[128 + half * 32 + lane] |= ((value >> 4) & 3) << (stream * 2);
    }
    bytes[192..208].fill(1);
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
    assert!(matches!(
        model.metadata_arrays.get("tokenizer.ggml.tokens"),
        Some(GgufMetadataArray::Strings(tokens)) if tokens.len() == 262_144
    ));
    assert!(matches!(
        model.metadata_arrays.get("tokenizer.ggml.scores"),
        Some(GgufMetadataArray::F32(scores)) if scores.len() == 262_144
    ));
}

#[test]
fn gemma4_q6_k_token_row_decodes_to_finite_bounded_values_when_fixture_is_available() {
    let Ok(path) = std::env::var("ATLAS_GEMMA4_GGUF") else {
        return;
    };
    let model = GgufModel::open(path).expect("read Gemma 4 E2B GGUF fixture");
    let tensor = model
        .tensors
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
        .expect("token embedding tensor");
    assert_eq!(tensor.tensor_type, GgufTensorType::Q6K);
    assert_eq!(tensor.dims, [1536, 262_144]);
    let row = 669usize;
    let row_blocks = tensor.dims[0] / 256;
    let block_bytes = GgufTensorType::Q6K.block_bytes();
    let data = model.tensor_data(tensor).expect("token embedding bytes");
    let mut values = Vec::with_capacity(tensor.dims[0]);
    for block in
        data[row * row_blocks * block_bytes..][..row_blocks * block_bytes].chunks_exact(block_bytes)
    {
        let mut decoded = [0.0; 256];
        dequantize_block(GgufTensorType::Q6K, block, &mut decoded).expect("decode Q6_K block");
        values.extend(decoded);
    }
    let (max_index, max_abs) = values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index, value.abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("decoded embedding values");
    assert!(values.iter().all(|value| value.is_finite()));
    // This QAT fixture intentionally contains a high-scale token row. Keep a
    // regression bound around the known finite row instead of assuming normal
    // embedding magnitudes.
    assert!(
        max_abs < 2_000_000.0,
        "Gemma token embedding row has unexpected Q6_K magnitude at index {max_index}: {max_abs}"
    );
}

#[test]
fn gemma4_q6_k_ple_row_669_has_finite_canonical_scales_and_values_when_fixture_is_available() {
    let Ok(path) = std::env::var("ATLAS_GEMMA4_GGUF") else {
        return;
    };
    let model = GgufModel::open(path).expect("read Gemma 4 E2B GGUF fixture");
    let tensor = model
        .tensors
        .iter()
        .find(|tensor| tensor.name == "per_layer_token_embd.weight")
        .expect("per-layer token embedding tensor");
    assert_eq!(tensor.tensor_type, GgufTensorType::Q6K);
    assert_eq!(tensor.dims, [8_960, 262_144]);

    let token = 669usize;
    let row_width = tensor.dims[0];
    let row_blocks = row_width / 256;
    let block_bytes = GgufTensorType::Q6K.block_bytes();
    let data = model
        .tensor_data(tensor)
        .expect("per-layer token embedding bytes");
    let row_start = token * row_blocks * block_bytes;
    let mut values = Vec::with_capacity(row_width);
    for block_index in 0..row_blocks {
        let byte_offset = row_start + block_index * block_bytes;
        let scale_byte_offset = byte_offset + 208;
        let block = &data[byte_offset..byte_offset + block_bytes];
        let (decoded, scale_bits) = llama_cpp_q6_k_decode(block);
        let scale = f16_bits_to_f32(scale_bits);
        assert!(
            scale.is_finite(),
            "per_layer_token_embd.weight row={token} block={block_index} scale_byte_offset={scale_byte_offset} scale_bits=0x{scale_bits:04x} has non-finite Q6_K scale {scale}"
        );
        assert!(
            decoded.iter().all(|value| value.is_finite()),
            "per_layer_token_embd.weight row={token} block={block_index} scale_byte_offset={scale_byte_offset} scale_bits=0x{scale_bits:04x} has non-finite decoded Q6_K value"
        );
        values.extend(decoded);
    }
    assert_eq!(values.len(), 8_960);
    assert!(
        values[3_328].is_finite(),
        "per_layer_token_embd.weight row={token} block=13 output_index=3328 scale_byte_offset={} scale_bits=0x{:04x} decoded a non-finite value",
        row_start + 13 * block_bytes + 208,
        u16::from_le_bytes([
            data[row_start + 13 * block_bytes + 208],
            data[row_start + 13 * block_bytes + 209],
        ])
    );
}
