use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use serde_json::Value;

use crate::{CoreError, DType, Shape, Storage, Tensor};

/// Metadata for one tensor declared in a SafeTensors header.
#[derive(Debug, Clone)]
pub struct SafeTensorDescriptor {
    pub name: String,
    pub data_start: usize,
    pub data_end: usize,
    pub tensor: Tensor,
}

/// Read only the SafeTensors header and create metadata descriptors.
pub fn read_safetensors_descriptors(
    path: impl AsRef<Path>,
) -> Result<Vec<SafeTensorDescriptor>, CoreError> {
    let path = path.as_ref();
    let mut file = File::open(path)
        .map_err(|error| CoreError::InvalidInput(format!("open {}: {error}", path.display())))?;
    let mut length = [0; 8];
    file.read_exact(&mut length).map_err(|error| {
        CoreError::InvalidInput(format!(
            "read header length from {}: {error}",
            path.display()
        ))
    })?;
    let header_len = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| CoreError::InvalidInput("SafeTensors header does not fit usize".into()))?;
    if header_len > 64 * 1024 * 1024 {
        return Err(CoreError::InvalidInput(
            "SafeTensors header exceeds 64 MiB".into(),
        ));
    }
    let mut header = vec![0; header_len];
    file.read_exact(&mut header).map_err(|error| {
        CoreError::InvalidInput(format!("read header from {}: {error}", path.display()))
    })?;
    let values: serde_json::Map<String, Value> = serde_json::from_slice(&header)
        .map_err(|error| CoreError::InvalidInput(format!("parse SafeTensors header: {error}")))?;

    let mut descriptors = Vec::new();
    for (name, value) in values {
        if name == "__metadata__" {
            continue;
        }
        let entry = value.as_object().ok_or_else(|| {
            CoreError::InvalidInput(format!("tensor {name} header is not an object"))
        })?;
        let dtype = parse_dtype(
            entry
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| CoreError::InvalidInput(format!("tensor {name} has no dtype")))?,
        )?;
        let dims = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| CoreError::InvalidInput(format!("tensor {name} has no shape")))?
            .iter()
            .map(|item| {
                item.as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .ok_or_else(|| {
                        CoreError::InvalidInput(format!("tensor {name} has invalid shape"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shape = Shape::new(dims)?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| CoreError::InvalidInput(format!("tensor {name} has no data offsets")))?;
        if offsets.len() != 2 {
            return Err(CoreError::InvalidInput(format!(
                "tensor {name} requires two data offsets"
            )));
        }
        let data_start = offsets[0]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                CoreError::InvalidInput(format!("tensor {name} has invalid start offset"))
            })?;
        let data_end = offsets[1]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                CoreError::InvalidInput(format!("tensor {name} has invalid end offset"))
            })?;
        if data_end < data_start {
            return Err(CoreError::InvalidInput(format!(
                "tensor {name} data offsets are reversed"
            )));
        }
        let expected = shape
            .element_count()
            .checked_mul(dtype.byte_width())
            .ok_or_else(|| {
                CoreError::InvalidShape(format!("tensor {name} byte length overflows usize"))
            })?;
        if data_end - data_start != expected {
            return Err(CoreError::InvalidInput(format!(
                "tensor {name} data length does not match dtype and shape"
            )));
        }
        let tensor = Tensor::contiguous(Storage::cpu(expected, true), shape, dtype)?;
        descriptors.push(SafeTensorDescriptor {
            name,
            data_start,
            data_end,
            tensor,
        });
    }
    descriptors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(descriptors)
}

/// Read one FP32/FP16/BF16/I8 tensor as FP32 values without loading other
/// model weights. The complete model loader remains a Phase 3 concern.
pub fn read_safetensors_tensor_f32(
    path: impl AsRef<Path>,
    name: &str,
) -> Result<Vec<f32>, CoreError> {
    let path = path.as_ref();
    let descriptor = read_safetensors_descriptors(path)?
        .into_iter()
        .find(|descriptor| descriptor.name == name)
        .ok_or_else(|| {
            CoreError::InvalidInput(format!("SafeTensors tensor `{name}` is missing"))
        })?;
    let mut file = File::open(path)
        .map_err(|error| CoreError::InvalidInput(format!("open {}: {error}", path.display())))?;
    let mut header_length = [0; 8];
    file.read_exact(&mut header_length).map_err(|error| {
        CoreError::InvalidInput(format!(
            "read header length from {}: {error}",
            path.display()
        ))
    })?;
    let payload_offset = 8u64
        .checked_add(u64::from_le_bytes(header_length))
        .and_then(|offset| offset.checked_add(descriptor.data_start as u64))
        .ok_or_else(|| {
            CoreError::InvalidInput("SafeTensors payload offset overflows u64".into())
        })?;
    file.seek(SeekFrom::Start(payload_offset))
        .map_err(|error| CoreError::InvalidInput(format!("seek {}: {error}", path.display())))?;
    let mut payload = vec![0; descriptor.data_end - descriptor.data_start];
    file.read_exact(&mut payload)
        .map_err(|error| CoreError::InvalidInput(format!("read tensor `{name}`: {error}")))?;
    Ok(match descriptor.tensor.dtype {
        DType::F32 => payload
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        DType::F16 => payload
            .chunks_exact(2)
            .map(|bytes| crate::f16_bits_to_f32(u16::from_le_bytes(bytes.try_into().unwrap())))
            .collect(),
        DType::BF16 => payload
            .chunks_exact(2)
            .map(|bytes| {
                f32::from_bits((u16::from_le_bytes(bytes.try_into().unwrap()) as u32) << 16)
            })
            .collect(),
        DType::I8 => payload.iter().map(|&value| value as i8 as f32).collect(),
    })
}

fn parse_dtype(value: &str) -> Result<DType, CoreError> {
    match value {
        "F32" => Ok(DType::F32),
        "F16" => Ok(DType::F16),
        "BF16" => Ok(DType::BF16),
        "I8" => Ok(DType::I8),
        _ => Err(CoreError::UnsupportedDType(value.into())),
    }
}
