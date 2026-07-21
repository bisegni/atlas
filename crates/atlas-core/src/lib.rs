//! Backend-neutral tensor metadata and SafeTensors descriptors for Atlas.

mod gguf;
mod quant;
mod safetensors;
mod tensor;

pub use gguf::{
    GGML_QK, GGUF_ALIGNMENT, GGUF_VERSION, GgufMetadataArray, GgufModel, GgufTensor,
    GgufTensorType, GgufWriter, dequantize_block, quantize_q4_0, quantize_q8_0,
};
pub use quant::{QuantFormat, QuantizedMatrix};
pub use safetensors::{
    SafeTensorDescriptor, read_safetensors_descriptors, read_safetensors_tensor_f32,
};
pub use tensor::{
    DType, Device, Shape, Storage, Strides, Tensor, f16_bits_to_f32, f32_to_f16_bits,
};

use thiserror::Error;

/// Errors that are independent of a concrete accelerator backend.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),
    #[error("invalid tensor layout: {0}")]
    InvalidLayout(String),
    #[error("unsupported tensor dtype: {0}")]
    UnsupportedDType(String),
}
