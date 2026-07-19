//! Block-32 affine weight-only quantization.

use crate::{CoreError, f16_bits_to_f32, f32_to_f16_bits};

pub const QUANT_BLOCK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    Fp16,
    Int8Block32,
    Q4Block32,
}

impl QuantFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8Block32 => "int8",
            Self::Q4Block32 => "q4",
        }
    }

    pub fn parse(value: &str) -> Result<Self, CoreError> {
        match value {
            "fp16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8Block32),
            "q4" => Ok(Self::Q4Block32),
            _ => Err(CoreError::InvalidInput(format!(
                "unsupported quantization format `{value}`"
            ))),
        }
    }
}

/// Packed row-major matrix. Each non-FP16 block has an FP16 scale and affine
/// zero point; no API creates a full dequantized backing allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedMatrix {
    pub format: QuantFormat,
    pub rows: usize,
    pub cols: usize,
    pub block_size: usize,
    pub data: Vec<u8>,
    pub scales_f16: Vec<u16>,
    pub zero_points: Vec<i8>,
}

impl QuantizedMatrix {
    pub fn quantize(
        values: &[f32],
        rows: usize,
        cols: usize,
        format: QuantFormat,
    ) -> Result<Self, CoreError> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(values.len()) {
            return Err(CoreError::InvalidInput(
                "quantized matrix dimensions do not match values".into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CoreError::InvalidInput(
                "quantized matrix values must be finite".into(),
            ));
        }
        if format == QuantFormat::Fp16 {
            return Ok(Self {
                format,
                rows,
                cols,
                block_size: QUANT_BLOCK,
                data: values
                    .iter()
                    .flat_map(|&value| f32_to_f16_bits(value).to_le_bytes())
                    .collect(),
                scales_f16: vec![],
                zero_points: vec![],
            });
        }

        let blocks_per_row = cols.div_ceil(QUANT_BLOCK);
        let mut scales_f16 = Vec::with_capacity(rows * blocks_per_row);
        let mut zero_points = Vec::with_capacity(rows * blocks_per_row);
        let mut quantized = vec![0u8; rows * cols];
        let (qmin, qmax) = if format == QuantFormat::Int8Block32 {
            (-128.0, 127.0)
        } else {
            (0.0, 15.0)
        };
        for row in 0..rows {
            for block in 0..blocks_per_row {
                let start = row * cols + block * QUANT_BLOCK;
                let end = (start + QUANT_BLOCK).min((row + 1) * cols);
                let (min, max) = values[start..end]
                    .iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &value| {
                        (lo.min(value), hi.max(value))
                    });
                let (scale, zero) = if max == min {
                    // A short tail block may contain one value.  Give it a
                    // representable scale instead of collapsing it through a
                    // zero-width affine range.
                    ((min.abs() / qmax).max(f32::MIN_POSITIVE), 0)
                } else {
                    let scale = (max - min) / (qmax - qmin);
                    (scale, (qmin - min / scale).round().clamp(qmin, qmax) as i8)
                };
                scales_f16.push(f32_to_f16_bits(scale));
                zero_points.push(zero);
                for (offset, &value) in values[start..end].iter().enumerate() {
                    let quantized_value = (value / scale + zero as f32).round().clamp(qmin, qmax);
                    quantized[start + offset] = if format == QuantFormat::Int8Block32 {
                        (quantized_value as i8) as u8
                    } else {
                        quantized_value as u8
                    };
                }
            }
        }
        let data = if format == QuantFormat::Int8Block32 {
            quantized
        } else {
            quantized
                .chunks(2)
                .map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(0) << 4))
                .collect()
        };
        Ok(Self {
            format,
            rows,
            cols,
            block_size: QUANT_BLOCK,
            data,
            scales_f16,
            zero_points,
        })
    }

    pub fn blocks_per_row(&self) -> usize {
        self.cols.div_ceil(self.block_size)
    }

    pub fn resident_bytes(&self) -> usize {
        self.data.len() + self.scales_f16.len() * 2 + self.zero_points.len()
    }

    pub fn dequantize_at(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows && col < self.cols);
        match self.format {
            QuantFormat::Fp16 => {
                let byte = (row * self.cols + col) * 2;
                f16_bits_to_f32(u16::from_le_bytes([self.data[byte], self.data[byte + 1]]))
            }
            format => {
                let index = row * self.cols + col;
                let value = if format == QuantFormat::Int8Block32 {
                    self.data[index] as i8 as f32
                } else {
                    let byte = self.data[index / 2];
                    if index.is_multiple_of(2) {
                        (byte & 0x0f) as f32
                    } else {
                        (byte >> 4) as f32
                    }
                };
                let block = row * self.blocks_per_row() + col / self.block_size;
                (value - self.zero_points[block] as f32) * f16_bits_to_f32(self.scales_f16[block])
            }
        }
    }

    pub fn matvec_cpu(&self, input: &[f32]) -> Result<Vec<f32>, CoreError> {
        if input.len() != self.cols {
            return Err(CoreError::InvalidInput(
                "quantized matvec input width differs".into(),
            ));
        }
        Ok((0..self.rows)
            .map(|row| {
                (0..self.cols)
                    .map(|col| input[col] * self.dequantize_at(row, col))
                    .sum()
            })
            .collect())
    }

    /// CPU oracle for prefill-shaped projection. It intentionally walks the
    /// packed representation directly and never constructs a dequantized
    /// matrix, matching the residency invariant of the Metal kernels.
    pub fn matmul_cpu(&self, input: &[f32], rows: usize) -> Result<Vec<f32>, CoreError> {
        if rows == 0 || input.len() != rows * self.cols {
            return Err(CoreError::InvalidInput(
                "quantized matmul dimensions differ".into(),
            ));
        }
        let mut output = vec![0.0; rows * self.rows];
        for row in 0..rows {
            for output_column in 0..self.rows {
                output[row * self.rows + output_column] = (0..self.cols)
                    .map(|column| {
                        input[row * self.cols + column] * self.dequantize_at(output_column, column)
                    })
                    .sum();
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block32_round_trip_and_storage_are_bounded() {
        let values: Vec<f32> = (0..65).map(|i| (i as f32 - 31.0) / 7.0).collect();
        for format in [
            QuantFormat::Fp16,
            QuantFormat::Int8Block32,
            QuantFormat::Q4Block32,
        ] {
            let packed = QuantizedMatrix::quantize(&values, 1, 65, format).unwrap();
            let max_error = values
                .iter()
                .enumerate()
                .map(|(index, &value)| (value - packed.dequantize_at(0, index)).abs())
                .fold(0.0, f32::max);
            assert!(
                max_error
                    <= if format == QuantFormat::Q4Block32 {
                        0.5
                    } else {
                        0.2
                    },
                "format={} max_error={max_error}",
                format.name()
            );
        }
    }
}
