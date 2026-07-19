use crate::CoreError;

/// Element representation recorded in tensor metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
}

impl DType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
        }
    }
}

/// Logical device ownership, kept independent from a backend implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal { registry_id: u64 },
}

/// Backing allocation metadata. The actual allocation stays owned by a backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Storage {
    pub device: Device,
    pub byte_len: usize,
    pub read_only: bool,
    pub allocation_id: Option<u64>,
}

impl Storage {
    pub const fn cpu(byte_len: usize, read_only: bool) -> Self {
        Self {
            device: Device::Cpu,
            byte_len,
            read_only,
            allocation_id: None,
        }
    }

    pub const fn metal(
        registry_id: u64,
        allocation_id: u64,
        byte_len: usize,
        read_only: bool,
    ) -> Self {
        Self {
            device: Device::Metal { registry_id },
            byte_len,
            read_only,
            allocation_id: Some(allocation_id),
        }
    }
}

/// Validated tensor dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, CoreError> {
        let dims = dims.into();
        if dims.is_empty() {
            return Err(CoreError::InvalidShape(
                "a tensor must have at least one dimension".into(),
            ));
        }
        dims.iter()
            .try_fold(1usize, |elements, &dimension| {
                elements.checked_mul(dimension)
            })
            .ok_or_else(|| CoreError::InvalidShape("element count overflows usize".into()))?;
        Ok(Self(dims))
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn element_count(&self) -> usize {
        self.0.iter().product()
    }
}

/// Element strides for a tensor view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides(Vec<usize>);

impl Strides {
    pub fn contiguous(shape: &Shape) -> Self {
        let mut strides = vec![0; shape.rank()];
        let mut current: usize = 1;
        for (index, dimension) in shape.dims().iter().enumerate().rev() {
            strides[index] = current;
            current = current.saturating_mul(*dimension);
        }
        Self(strides)
    }

    pub fn new(values: impl Into<Vec<usize>>, shape: &Shape) -> Result<Self, CoreError> {
        let values = values.into();
        if values.len() != shape.rank() {
            return Err(CoreError::InvalidLayout(
                "stride rank does not match shape rank".into(),
            ));
        }
        Ok(Self(values))
    }

    pub fn values(&self) -> &[usize] {
        &self.0
    }
}

/// Metadata-only tensor view. `offset_elements` is measured in elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub storage: Storage,
    pub shape: Shape,
    pub strides: Strides,
    pub dtype: DType,
    pub offset_elements: usize,
}

impl Tensor {
    pub fn new(
        storage: Storage,
        shape: Shape,
        strides: Strides,
        dtype: DType,
        offset_elements: usize,
    ) -> Result<Self, CoreError> {
        let tensor = Self {
            storage,
            shape,
            strides,
            dtype,
            offset_elements,
        };
        tensor.validate_bounds()?;
        Ok(tensor)
    }

    pub fn contiguous(storage: Storage, shape: Shape, dtype: DType) -> Result<Self, CoreError> {
        let strides = Strides::contiguous(&shape);
        Self::new(storage, shape, strides, dtype, 0)
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == Strides::contiguous(&self.shape)
    }

    pub fn reshape(&self, shape: Shape) -> Result<Self, CoreError> {
        if !self.is_contiguous() {
            return Err(CoreError::InvalidLayout(
                "reshape requires a contiguous tensor".into(),
            ));
        }
        if shape.element_count() != self.shape.element_count() {
            return Err(CoreError::InvalidShape(
                "reshape changes element count".into(),
            ));
        }
        Self::new(
            self.storage.clone(),
            shape.clone(),
            Strides::contiguous(&shape),
            self.dtype,
            self.offset_elements,
        )
    }

    pub fn transpose(&self, permutation: &[usize]) -> Result<Self, CoreError> {
        if permutation.len() != self.shape.rank() {
            return Err(CoreError::InvalidLayout(
                "permutation rank does not match tensor rank".into(),
            ));
        }
        let mut seen = vec![false; self.shape.rank()];
        let mut dims = Vec::with_capacity(permutation.len());
        let mut strides = Vec::with_capacity(permutation.len());
        for &axis in permutation {
            if axis >= self.shape.rank() || std::mem::replace(&mut seen[axis], true) {
                return Err(CoreError::InvalidLayout(
                    "permutation must contain each axis once".into(),
                ));
            }
            dims.push(self.shape.dims()[axis]);
            strides.push(self.strides.values()[axis]);
        }
        Self::new(
            self.storage.clone(),
            Shape::new(dims)?,
            Strides::new(strides, &self.shape)?,
            self.dtype,
            self.offset_elements,
        )
    }

    pub fn view(
        &self,
        shape: Shape,
        strides: Strides,
        offset_elements: usize,
    ) -> Result<Self, CoreError> {
        Self::new(
            self.storage.clone(),
            shape,
            strides,
            self.dtype,
            offset_elements,
        )
    }

    pub fn byte_len(&self) -> Result<usize, CoreError> {
        self.shape
            .element_count()
            .checked_mul(self.dtype.byte_width())
            .ok_or_else(|| CoreError::InvalidShape("tensor byte length overflows usize".into()))
    }

    fn validate_bounds(&self) -> Result<(), CoreError> {
        let max_element = self.max_element_index()?;
        let required = max_element
            .checked_add(1)
            .and_then(|items| items.checked_mul(self.dtype.byte_width()))
            .ok_or_else(|| CoreError::InvalidLayout("tensor byte range overflows usize".into()))?;
        if required > self.storage.byte_len {
            return Err(CoreError::InvalidLayout(format!(
                "tensor requires {required} bytes but storage contains {}",
                self.storage.byte_len
            )));
        }
        Ok(())
    }

    fn max_element_index(&self) -> Result<usize, CoreError> {
        self.shape
            .dims()
            .iter()
            .zip(self.strides.values())
            .try_fold(self.offset_elements, |index, (&dimension, &stride)| {
                let span = dimension
                    .saturating_sub(1)
                    .checked_mul(stride)
                    .ok_or_else(|| {
                        CoreError::InvalidLayout("tensor stride range overflows usize".into())
                    })?;
                index.checked_add(span).ok_or_else(|| {
                    CoreError::InvalidLayout("tensor offset range overflows usize".into())
                })
            })
    }
}

/// Convert an IEEE-754 `f32` into binary16 bits using round-to-nearest-even.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = 14 - exponent;
        return sign | ((mantissa + (1 << (shift - 1))) >> shift) as u16;
    }
    if exponent >= 31 {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    sign | ((exponent as u16) << 10) | ((mantissa + 0x1000) >> 13) as u16
}

/// Convert IEEE-754 binary16 bits into `f32`.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32 & 0x8000) << 16) as u32;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut mantissa = mantissa;
            let mut exponent = -14i32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent as u32 + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}
