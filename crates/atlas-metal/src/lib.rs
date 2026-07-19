//! Native Metal bootstrap for Atlas.
//!
//! The API intentionally owns command submission and shared buffers in one
//! small place. Higher-level tensor code is introduced in Phase 1.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

#[cfg(not(target_os = "macos"))]
compile_error!("atlas-metal currently supports macOS only");

#[cfg(target_os = "macos")]
mod macos {
    use std::{
        collections::HashMap,
        mem::size_of,
        ptr,
        time::{Duration, Instant},
    };

    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };
    use thiserror::Error;

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    const KERNEL_SOURCE: &str = include_str!("kernels.metal");

    #[derive(Debug, Error)]
    pub enum MetalError {
        #[error("no Metal device is available")]
        NoDevice,
        #[error("Metal failed to compile Atlas kernels: {0}")]
        ShaderCompile(String),
        #[error("Metal kernel `{0}` was not found")]
        MissingKernel(String),
        #[error("Metal failed to create a compute pipeline for `{0}`: {1}")]
        Pipeline(String, String),
        #[error("Metal failed to create a command buffer or encoder")]
        CommandCreation,
        #[error("Metal command buffer failed: {0}")]
        CommandFailed(String),
        #[error("invalid buffer/input: {0}")]
        InvalidInput(String),
    }

    #[derive(Debug, Clone)]
    pub struct DeviceInfo {
        pub name: String,
        pub registry_id: u64,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct DispatchTiming {
        pub wall_time: Duration,
        pub gpu_time: Option<Duration>,
    }

    pub struct MetalRuntime {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    }

    impl MetalRuntime {
        pub fn new() -> Result<Self, MetalError> {
            let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
            let queue = device
                .newCommandQueue()
                .ok_or(MetalError::CommandCreation)?;
            let source = NSString::from_str(KERNEL_SOURCE);
            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|error| MetalError::ShaderCompile(error.to_string()))?;

            let mut pipelines = HashMap::new();
            for kernel in [
                "vector_add_f32",
                "scalar_multiply_f32",
                "silu_f32",
                "reduction_sum_f32",
                "transpose_f32",
            ] {
                let function_name = NSString::from_str(kernel);
                let function = library
                    .newFunctionWithName(&function_name)
                    .ok_or_else(|| MetalError::MissingKernel(kernel.to_owned()))?;
                let pipeline = device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|error| MetalError::Pipeline(kernel.to_owned(), error.to_string()))?;
                pipelines.insert(kernel, pipeline);
            }

            Ok(Self {
                device,
                queue,
                pipelines,
            })
        }

        pub fn device_info(&self) -> DeviceInfo {
            DeviceInfo {
                name: self.device.name().to_string(),
                registry_id: self.device.registryID(),
            }
        }

        pub fn pipeline_count(&self) -> usize {
            self.pipelines.len()
        }

        pub fn vector_add(
            &self,
            lhs: &[f32],
            rhs: &[f32],
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            if lhs.len() != rhs.len() {
                return Err(MetalError::InvalidInput("vector lengths differ".into()));
            }
            let count = u32::try_from(lhs.len())
                .map_err(|_| MetalError::InvalidInput("vector is too large".into()))?;
            let mut output = vec![0.0; lhs.len()];
            let lhs_buffer = self.buffer_from_slice(lhs)?;
            let rhs_buffer = self.buffer_from_slice(rhs)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let count_buffer = self.buffer_from_slice(&[count])?;

            let timing = self.dispatch_1d(
                "vector_add_f32",
                &[&lhs_buffer, &rhs_buffer, &output_buffer, &count_buffer],
                lhs.len(),
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn scalar_multiply(
            &self,
            input: &[f32],
            scalar: f32,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            let count = count_u32(input.len())?;
            let mut output = vec![0.0; input.len()];
            let input_buffer = self.buffer_from_slice(input)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let scalar_buffer = self.buffer_from_slice(&[scalar])?;
            let count_buffer = self.buffer_from_slice(&[count])?;
            let timing = self.dispatch_1d(
                "scalar_multiply_f32",
                &[&input_buffer, &output_buffer, &scalar_buffer, &count_buffer],
                input.len(),
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn silu(&self, input: &[f32]) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            let count = count_u32(input.len())?;
            let mut output = vec![0.0; input.len()];
            let input_buffer = self.buffer_from_slice(input)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let count_buffer = self.buffer_from_slice(&[count])?;
            let timing = self.dispatch_1d(
                "silu_f32",
                &[&input_buffer, &output_buffer, &count_buffer],
                input.len(),
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn sum(&self, input: &[f32]) -> Result<(f32, DispatchTiming), MetalError> {
            let count = count_u32(input.len())?;
            let mut output = [0.0];
            let input_buffer = self.buffer_from_slice(input)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let count_buffer = self.buffer_from_slice(&[count])?;
            let timing = self.dispatch_1d(
                "reduction_sum_f32",
                &[&input_buffer, &output_buffer, &count_buffer],
                input.len().max(1),
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output[0], timing))
        }

        pub fn transpose(
            &self,
            input: &[f32],
            rows: usize,
            cols: usize,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            if input.len()
                != rows
                    .checked_mul(cols)
                    .ok_or_else(|| MetalError::InvalidInput("matrix dimensions overflow".into()))?
            {
                return Err(MetalError::InvalidInput(
                    "matrix length does not match rows * cols".into(),
                ));
            }
            let rows_u32 = count_u32(rows)?;
            let cols_u32 = count_u32(cols)?;
            let mut output = vec![0.0; input.len()];
            let input_buffer = self.buffer_from_slice(input)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let rows_buffer = self.buffer_from_slice(&[rows_u32])?;
            let cols_buffer = self.buffer_from_slice(&[cols_u32])?;
            let timing = self.dispatch_2d(
                "transpose_f32",
                &[&input_buffer, &output_buffer, &rows_buffer, &cols_buffer],
                cols,
                rows,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        fn buffer_from_slice<T: Copy>(
            &self,
            values: &[T],
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
            let bytes = values
                .len()
                .checked_mul(size_of::<T>())
                .ok_or_else(|| MetalError::InvalidInput("buffer size overflow".into()))?;
            let buffer = self
                .device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    MetalError::InvalidInput("Metal could not allocate a shared buffer".into())
                })?;
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr().cast::<u8>(),
                    buffer.contents().as_ptr().cast::<u8>(),
                    bytes,
                );
            }
            Ok(buffer)
        }

        fn copy_buffer_to_slice<T: Copy>(
            &self,
            buffer: &ProtocolObject<dyn MTLBuffer>,
            values: &mut [T],
        ) -> Result<(), MetalError> {
            let bytes = values
                .len()
                .checked_mul(size_of::<T>())
                .ok_or_else(|| MetalError::InvalidInput("buffer size overflow".into()))?;
            unsafe {
                ptr::copy_nonoverlapping(
                    buffer.contents().as_ptr().cast::<u8>(),
                    values.as_mut_ptr().cast::<u8>(),
                    bytes,
                );
            }
            Ok(())
        }

        fn dispatch_1d(
            &self,
            kernel: &'static str,
            buffers: &[&ProtocolObject<dyn MTLBuffer>],
            count: usize,
        ) -> Result<DispatchTiming, MetalError> {
            self.dispatch(
                kernel,
                buffers,
                MTLSize {
                    width: count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.pipeline_thread_width(kernel),
                    height: 1,
                    depth: 1,
                },
            )
        }

        fn dispatch_2d(
            &self,
            kernel: &'static str,
            buffers: &[&ProtocolObject<dyn MTLBuffer>],
            width: usize,
            height: usize,
        ) -> Result<DispatchTiming, MetalError> {
            self.dispatch(
                kernel,
                buffers,
                MTLSize {
                    width,
                    height,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 16,
                    depth: 1,
                },
            )
        }

        fn pipeline_thread_width(&self, kernel: &'static str) -> usize {
            self.pipelines
                .get(kernel)
                .expect("all kernels are cached during construction")
                .maxTotalThreadsPerThreadgroup()
                .min(256)
                .max(1)
        }

        fn dispatch(
            &self,
            kernel: &'static str,
            buffers: &[&ProtocolObject<dyn MTLBuffer>],
            threads: MTLSize,
            threadgroup: MTLSize,
        ) -> Result<DispatchTiming, MetalError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandCreation)?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or(MetalError::CommandCreation)?;
            let pipeline = self
                .pipelines
                .get(kernel)
                .expect("all kernels are cached during construction");
            encoder.setComputePipelineState(&**pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
            }
            encoder.dispatchThreads_threadsPerThreadgroup(threads, threadgroup);
            encoder.endEncoding();

            let start = Instant::now();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            let wall_time = start.elapsed();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err(MetalError::CommandFailed(
                    command_buffer
                        .error()
                        .map(|error| error.to_string())
                        .unwrap_or_else(|| "unknown command-buffer error".into()),
                ));
            }
            let gpu_start = command_buffer.GPUStartTime();
            let gpu_end = command_buffer.GPUEndTime();
            let gpu_time = (gpu_end > gpu_start && gpu_start > 0.0)
                .then(|| Duration::from_secs_f64(gpu_end - gpu_start));
            Ok(DispatchTiming {
                wall_time,
                gpu_time,
            })
        }
    }

    fn count_u32(count: usize) -> Result<u32, MetalError> {
        u32::try_from(count).map_err(|_| MetalError::InvalidInput("input is too large".into()))
    }
}

#[cfg(target_os = "macos")]
pub use macos::*;
