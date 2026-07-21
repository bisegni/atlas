//! Minimal, checked GGUF v3 support for Atlas Llama weight artifacts.
//!
//! Atlas intentionally supports the small subset it can execute: F32/F16
//! vectors plus Q4_0/Q8_0 block-32 matrices.  The reader rejects every other
//! tensor encoding before a caller can allocate GPU buffers.

use std::{collections::BTreeMap, fs, path::Path};

use crate::{CoreError, f16_bits_to_f32, f32_to_f16_bits};

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_ALIGNMENT: usize = 32;
pub const GGML_QK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Q6K,
}

impl GgufTensorType {
    fn raw(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Q6K => 14,
        }
    }
    fn from_raw(raw: u32) -> Result<Self, CoreError> {
        match raw {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            8 => Ok(Self::Q8_0),
            14 => Ok(Self::Q6K),
            _ => Err(CoreError::InvalidInput(format!(
                "unsupported GGUF tensor type {raw}"
            ))),
        }
    }
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,
            Self::Q8_0 => 34,
            Self::Q6K => 210,
        }
    }
    pub fn encoded_bytes(self, elements: usize) -> Result<usize, CoreError> {
        match self {
            Self::F32 | Self::F16 => elements
                .checked_mul(self.block_bytes())
                .ok_or_else(|| CoreError::InvalidInput("GGUF tensor byte size overflows".into())),
            Self::Q4_0 | Self::Q8_0 | Self::Q6K => {
                let block = if self == Self::Q6K { 256 } else { GGML_QK };
                if !elements.is_multiple_of(block) {
                    return Err(CoreError::InvalidInput(format!(
                        "packed GGUF tensor has {elements} elements, not a multiple of {block}"
                    )));
                }
                elements
                    .checked_div(block)
                    .and_then(|n| n.checked_mul(self.block_bytes()))
                    .ok_or_else(|| {
                        CoreError::InvalidInput("GGUF packed tensor byte size overflows".into())
                    })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<usize>,
    pub tensor_type: GgufTensorType,
    pub offset: usize,
    pub bytes: usize,
}

/// Array-valued GGUF metadata retained for tokenizer and architecture-specific
/// configuration. Scalar metadata remains in [`GgufModel::metadata`] for the
/// existing Llama callers.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataArray {
    Strings(Vec<String>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

#[derive(Debug, Clone)]
pub struct GgufModel {
    pub metadata: BTreeMap<String, String>,
    pub metadata_arrays: BTreeMap<String, GgufMetadataArray>,
    pub tensors: Vec<GgufTensor>,
    data_start: usize,
    data: Vec<u8>,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CoreError> {
        Self::from_bytes(
            fs::read(path).map_err(|e| CoreError::InvalidInput(format!("read GGUF: {e}")))?,
        )
    }
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, CoreError> {
        let mut r = Reader { data: &data, at: 0 };
        if r.take(4)? != GGUF_MAGIC {
            return Err(CoreError::InvalidInput("GGUF magic is missing".into()));
        }
        if r.u32()? != GGUF_VERSION {
            return Err(CoreError::InvalidInput("only GGUF v3 is supported".into()));
        }
        let tensor_count = r.usize_u64("tensor count")?;
        let metadata_count = r.usize_u64("metadata count")?;
        if tensor_count > 1_000_000 || metadata_count > 1_000_000 {
            return Err(CoreError::InvalidInput("GGUF count is unreasonable".into()));
        }
        let mut metadata = BTreeMap::new();
        let mut metadata_arrays = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = r.string()?;
            let value_type = r.u32()?;
            if value_type == 9 {
                if let Some(value) = r.metadata_array()? {
                    metadata_arrays.insert(key, value);
                }
            } else if let Some(value) = r.metadata_value(value_type)? {
                metadata.insert(key, value);
            }
        }
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let dims_len = r.u32()? as usize;
            if dims_len == 0 || dims_len > 4 {
                return Err(CoreError::InvalidInput(format!(
                    "GGUF tensor {name} has invalid rank {dims_len}"
                )));
            }
            let mut dims = Vec::with_capacity(dims_len);
            let mut elements = 1usize;
            for _ in 0..dims_len {
                let dim = r.usize_u64("tensor dimension")?;
                if dim == 0 {
                    return Err(CoreError::InvalidInput(format!(
                        "GGUF tensor {name} has zero dimension"
                    )));
                }
                elements = elements.checked_mul(dim).ok_or_else(|| {
                    CoreError::InvalidInput("GGUF element count overflows".into())
                })?;
                dims.push(dim);
            }
            let tensor_type = GgufTensorType::from_raw(r.u32()?)?;
            let offset = r.usize_u64("tensor offset")?;
            if !offset.is_multiple_of(GGUF_ALIGNMENT) {
                return Err(CoreError::InvalidInput(format!(
                    "GGUF tensor {name} offset is not aligned"
                )));
            }
            tensors.push(GgufTensor {
                name,
                dims,
                tensor_type,
                offset,
                bytes: tensor_type.encoded_bytes(elements)?,
            });
        }
        let alignment = metadata
            .get("general.alignment")
            .map(|v| v.parse())
            .transpose()
            .map_err(|_| CoreError::InvalidInput("invalid general.alignment".into()))?
            .unwrap_or(GGUF_ALIGNMENT);
        if alignment != GGUF_ALIGNMENT {
            return Err(CoreError::InvalidInput(format!(
                "Atlas requires GGUF alignment {GGUF_ALIGNMENT}, got {alignment}"
            )));
        }
        let data_start = align(r.at, alignment)?;
        let mut ranges = Vec::new();
        for tensor in &tensors {
            let start = data_start
                .checked_add(tensor.offset)
                .ok_or_else(|| CoreError::InvalidInput("GGUF tensor offset overflows".into()))?;
            let end = start
                .checked_add(tensor.bytes)
                .ok_or_else(|| CoreError::InvalidInput("GGUF tensor range overflows".into()))?;
            if end > data.len() {
                return Err(CoreError::InvalidInput(format!(
                    "GGUF tensor {} is outside the file",
                    tensor.name
                )));
            }
            ranges.push((start, end, &tensor.name));
        }
        ranges.sort_by_key(|range| range.0);
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(CoreError::InvalidInput(format!(
                    "GGUF tensors {} and {} overlap",
                    pair[0].2, pair[1].2
                )));
            }
        }
        Ok(Self {
            metadata,
            metadata_arrays,
            tensors,
            data_start,
            data,
        })
    }
    pub fn tensor_data(&self, tensor: &GgufTensor) -> Result<&[u8], CoreError> {
        let start = self
            .data_start
            .checked_add(tensor.offset)
            .ok_or_else(|| CoreError::InvalidInput("GGUF offset overflows".into()))?;
        self.data
            .get(start..start + tensor.bytes)
            .ok_or_else(|| CoreError::InvalidInput("GGUF tensor data is outside file".into()))
    }
}

pub struct GgufWriter {
    metadata: BTreeMap<String, String>,
    tensors: Vec<(String, Vec<usize>, GgufTensorType, Vec<u8>)>,
}
impl GgufWriter {
    pub fn new() -> Self {
        let mut metadata = BTreeMap::new();
        metadata.insert("general.alignment".into(), GGUF_ALIGNMENT.to_string());
        metadata.insert("general.architecture".into(), "llama".into());
        Self {
            metadata,
            tensors: Vec::new(),
        }
    }
    pub fn metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
    pub fn push_tensor(
        &mut self,
        name: impl Into<String>,
        dims: Vec<usize>,
        tensor_type: GgufTensorType,
        data: Vec<u8>,
    ) -> Result<(), CoreError> {
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| CoreError::InvalidInput("GGUF tensor dimensions overflow".into()))?;
        if data.len() != tensor_type.encoded_bytes(elements)? {
            return Err(CoreError::InvalidInput(
                "GGUF tensor data length differs from dimensions".into(),
            ));
        }
        self.tensors.push((name.into(), dims, tensor_type, data));
        Ok(())
    }
    pub fn finish(self) -> Result<Vec<u8>, CoreError> {
        let mut out = Vec::new();
        out.extend_from_slice(GGUF_MAGIC);
        put_u32(&mut out, GGUF_VERSION);
        put_u64(&mut out, self.tensors.len() as u64);
        put_u64(&mut out, self.metadata.len() as u64);
        for (key, value) in &self.metadata {
            put_string(&mut out, key);
            put_u32(&mut out, 8);
            put_string(&mut out, value);
        }
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut offset = 0usize;
        for (_, _, _, data) in &self.tensors {
            offset = align(offset, GGUF_ALIGNMENT)?;
            offsets.push(offset);
            offset = offset.checked_add(data.len()).ok_or_else(|| {
                CoreError::InvalidInput("GGUF output exceeds address space".into())
            })?;
        }
        for ((name, dims, kind, data), offset) in self.tensors.iter().zip(&offsets) {
            put_string(&mut out, name);
            put_u32(&mut out, dims.len() as u32);
            for &dim in dims {
                put_u64(&mut out, dim as u64);
            }
            put_u32(&mut out, kind.raw());
            put_u64(&mut out, *offset as u64);
            let _ = data;
        }
        out.resize(align(out.len(), GGUF_ALIGNMENT)?, 0);
        for ((_, _, _, data), _offset) in self.tensors.iter().zip(&offsets) {
            out.resize(align(out.len(), GGUF_ALIGNMENT)?, 0);
            out.extend_from_slice(data);
        }
        Ok(out)
    }
}
impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

pub fn quantize_q4_0(values: &[f32]) -> Result<Vec<u8>, CoreError> {
    quantize(values, GgufTensorType::Q4_0)
}
pub fn quantize_q8_0(values: &[f32]) -> Result<Vec<u8>, CoreError> {
    quantize(values, GgufTensorType::Q8_0)
}
fn quantize(values: &[f32], kind: GgufTensorType) -> Result<Vec<u8>, CoreError> {
    if !values.len().is_multiple_of(GGML_QK) || values.iter().any(|v| !v.is_finite()) {
        return Err(CoreError::InvalidInput(
            "GGUF quantization requires finite block-32 values".into(),
        ));
    }
    let mut out = Vec::with_capacity(kind.encoded_bytes(values.len())?);
    for block in values.chunks_exact(GGML_QK) {
        let (max, _) = block
            .iter()
            .copied()
            .fold((0.0f32, 0.0f32), |(max, amax), v| {
                if v.abs() > amax {
                    (v, v.abs())
                } else {
                    (max, amax)
                }
            });
        let scale = if max == 0.0 {
            0.0
        } else {
            max / if kind == GgufTensorType::Q4_0 {
                -8.0
            } else {
                max.signum() * 127.0
            }
        };
        out.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        match kind {
            GgufTensorType::Q4_0 => {
                for i in 0..GGML_QK / 2 {
                    let a = if scale == 0.0 {
                        0
                    } else {
                        (block[i] / scale).round().clamp(-8.0, 7.0) as i8 + 8
                    } as u8;
                    let b = if scale == 0.0 {
                        0
                    } else {
                        (block[i + GGML_QK / 2] / scale).round().clamp(-8.0, 7.0) as i8 + 8
                    } as u8;
                    out.push(a | (b << 4));
                }
            }
            GgufTensorType::Q8_0 => out.extend(block.iter().map(|v| {
                if scale == 0.0 {
                    0
                } else {
                    (v / scale).round().clamp(-128.0, 127.0) as i8 as u8
                }
            })),
            _ => unreachable!(),
        }
    }
    Ok(out)
}
pub fn dequantize_block(
    kind: GgufTensorType,
    block: &[u8],
    output: &mut [f32],
) -> Result<(), CoreError> {
    if kind == GgufTensorType::Q6K {
        if output.len() != 256 || block.len() != kind.block_bytes() {
            return Err(CoreError::InvalidInput("invalid Q6_K block".into()));
        }
        // llama.cpp block_q6_K is ql[128], qh[64], scales[16], then the
        // f16 super-block scale. The scale is at the end of the block.
        let delta = f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        for index in 0usize..256 {
            // GGML Q6_K encodes each 128-value half as four interleaved
            // 32-element streams. The two high-bit planes select both the
            // low/high ql nibble and the per-16-value scale; they are not a
            // sequential two-bit value for every four elements.
            let half = index / 128;
            let within = index % 128;
            let stream = within / 32;
            let lane = within % 32;
            let packed = ql[half * 64 + lane + if stream % 2 == 1 { 32 } else { 0 }];
            let low = if stream >= 2 {
                packed >> 4
            } else {
                packed & 0x0f
            };
            let high = (qh[half * 32 + lane] >> (stream * 2)) & 0x03;
            let scale = scales[half * 8 + lane / 16 + stream * 2] as i8 as f32;
            output[index] = (((high << 4) | low) as i8 - 32) as f32 * scale * delta;
        }
        return Ok(());
    }
    if output.len() != GGML_QK || block.len() != kind.block_bytes() {
        return Err(CoreError::InvalidInput("invalid GGUF block".into()));
    }
    let scale = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
    match kind {
        GgufTensorType::Q4_0 => {
            for i in 0..GGML_QK {
                let nibble = if i < GGML_QK / 2 {
                    block[2 + i] & 15
                } else {
                    block[2 + i - GGML_QK / 2] >> 4
                };
                output[i] = (nibble as i8 - 8) as f32 * scale;
            }
        }
        GgufTensorType::Q8_0 => {
            for i in 0..GGML_QK {
                output[i] = block[2 + i] as i8 as f32 * scale;
            }
        }
        _ => return Err(CoreError::InvalidInput("not a packed GGUF block".into())),
    };
    Ok(())
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CoreError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| CoreError::InvalidInput("GGUF cursor overflows".into()))?;
        let bytes = self
            .data
            .get(self.at..end)
            .ok_or_else(|| CoreError::InvalidInput("truncated GGUF".into()))?;
        self.at = end;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, CoreError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, CoreError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn usize_u64(&mut self, what: &str) -> Result<usize, CoreError> {
        usize::try_from(self.u64()?)
            .map_err(|_| CoreError::InvalidInput(format!("GGUF {what} does not fit usize")))
    }
    fn string(&mut self) -> Result<String, CoreError> {
        let len = self.usize_u64("string length")?;
        std::str::from_utf8(self.take(len)?)
            .map(str::to_owned)
            .map_err(|_| CoreError::InvalidInput("GGUF string is not UTF-8".into()))
    }
    /// Read a GGUF metadata value, retaining scalar values Atlas consumes and
    /// safely skipping descriptive values such as `general.tags` arrays.
    fn metadata_value(&mut self, value_type: u32) -> Result<Option<String>, CoreError> {
        match value_type {
            0 => Ok(Some(self.take(1)?[0].to_string())),
            1 => Ok(Some((self.take(1)?[0] as i8).to_string())),
            2 => Ok(Some(
                u16::from_le_bytes(self.take(2)?.try_into().unwrap()).to_string(),
            )),
            3 => Ok(Some(
                i16::from_le_bytes(self.take(2)?.try_into().unwrap()).to_string(),
            )),
            4 => Ok(Some(self.u32()?.to_string())),
            5 => Ok(Some(
                i32::from_le_bytes(self.take(4)?.try_into().unwrap()).to_string(),
            )),
            6 => Ok(Some(
                f32::from_le_bytes(self.take(4)?.try_into().unwrap()).to_string(),
            )),
            7 => Ok(Some((self.take(1)?[0] != 0).to_string())),
            8 => Ok(Some(self.string()?)),
            9 => Err(CoreError::InvalidInput(
                "GGUF array must be read through metadata_array".into(),
            )),
            10 => Ok(Some(self.u64()?.to_string())),
            11 => Ok(Some(
                i64::from_le_bytes(self.take(8)?.try_into().unwrap()).to_string(),
            )),
            12 => Ok(Some(
                f64::from_le_bytes(self.take(8)?.try_into().unwrap()).to_string(),
            )),
            _ => Err(CoreError::InvalidInput(format!(
                "unsupported GGUF metadata type {value_type}"
            ))),
        }
    }
    fn metadata_array(&mut self) -> Result<Option<GgufMetadataArray>, CoreError> {
        let element_type = self.u32()?;
        let count = self.usize_u64("metadata array length")?;
        match element_type {
            8 => Ok(Some(GgufMetadataArray::Strings(
                (0..count)
                    .map(|_| self.string())
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            6 => Ok(Some(GgufMetadataArray::F32(
                (0..count)
                    .map(|_| {
                        Ok(f32::from_le_bytes(
                            self.take(4)?.try_into().expect("f32 bytes"),
                        ))
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?,
            ))),
            5 => Ok(Some(GgufMetadataArray::I32(
                (0..count)
                    .map(|_| {
                        Ok(i32::from_le_bytes(
                            self.take(4)?.try_into().expect("i32 bytes"),
                        ))
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?,
            ))),
            4 => Ok(Some(GgufMetadataArray::U32(
                (0..count)
                    .map(|_| self.u32())
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            10 => Ok(Some(GgufMetadataArray::U64(
                (0..count)
                    .map(|_| self.u64())
                    .collect::<Result<Vec<_>, _>>()?,
            ))),
            11 => Ok(Some(GgufMetadataArray::I64(
                (0..count)
                    .map(|_| {
                        Ok(i64::from_le_bytes(
                            self.take(8)?.try_into().expect("i64 bytes"),
                        ))
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?,
            ))),
            7 => Ok(Some(GgufMetadataArray::Bool(
                (0..count)
                    .map(|_| Ok(self.take(1)?[0] != 0))
                    .collect::<Result<Vec<_>, CoreError>>()?,
            ))),
            other => {
                for _ in 0..count {
                    let _ = self.metadata_value(other)?;
                }
                Ok(None)
            }
        }
    }
}
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_string(out: &mut Vec<u8>, value: &str) {
    put_u64(out, value.len() as u64);
    out.extend_from_slice(value.as_bytes());
}
fn align(value: usize, alignment: usize) -> Result<usize, CoreError> {
    value
        .checked_add(alignment - 1)
        .map(|v| v / alignment * alignment)
        .ok_or_else(|| CoreError::InvalidInput("GGUF alignment overflows".into()))
}
