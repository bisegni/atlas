//! Native Metal bootstrap for Atlas.
//!
//! The API intentionally owns command submission and shared buffers in one
//! small place. Phase 1 adds classified pooled buffers for tensor storage.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

#[cfg(not(target_os = "macos"))]
compile_error!("atlas-metal currently supports macOS only");

#[cfg(target_os = "macos")]
mod macos {
    use std::{
        collections::{BTreeMap, HashMap},
        mem::size_of,
        ptr,
        sync::atomic::{AtomicU64, Ordering},
        time::{Duration, Instant},
    };

    use atlas_core::{GgufTensorType, Storage};
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
        /// CPU time spent submitting the completed command buffer to Metal.
        pub command_buffer_schedule: Duration,
        pub gpu_time: Option<Duration>,
    }

    /// Exact timing for one resident kernel dispatch. Present only for the
    /// opt-in diagnostic path, where each dispatch is isolated in its own
    /// command buffer.
    #[derive(Debug, Clone)]
    pub struct ResidentKernelTiming {
        pub kernel: &'static str,
        pub threads: usize,
        pub threadgroups: usize,
        pub threads_per_threadgroup: usize,
        pub cpu_encode: Duration,
        pub timing: DispatchTiming,
    }

    /// An owned, GPU-visible buffer used by the resident decode path.
    ///
    /// Atlas currently uses shared storage on Apple Silicon: this is still a
    /// Metal allocation (and is bound directly to compute encoders), while
    /// making the one-token result cheap to inspect at the token boundary.
    #[derive(Clone)]
    pub struct GpuBuffer {
        _buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        bytes: usize,
    }

    impl GpuBuffer {
        pub fn bytes(&self) -> usize {
            self.bytes
        }

        fn native(&self) -> &ProtocolObject<dyn MTLBuffer> {
            &self._buffer
        }
    }

    /// A single compute encoder that can contain all dependent dispatches for
    /// one decode token. It is completed only by [`Self::finish`].
    pub struct ResidentCommand<'a> {
        runtime: &'a MetalRuntime,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        // A token still uses one command buffer.  Each dependent dispatch gets
        // its own compute encoder so producer writes are an explicit pass
        // boundary before the next kernel consumes them.
        encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
        exact_per_dispatch: bool,
        kernel_timings: Vec<ResidentKernelTiming>,
    }

    impl<'a> ResidentCommand<'a> {
        /// Dispatch a fixed number of workgroups.  Resident fused kernels use
        /// this when their synchronization scope is one logical output unit
        /// (for example, one attention head) rather than a flat element range.
        pub fn dispatch_threadgroups_1d(
            &mut self,
            kernel: &'static str,
            buffers: &[&GpuBuffer],
            threadgroups: usize,
            threads_per_threadgroup: usize,
        ) -> Result<(), MetalError> {
            let encode_started = Instant::now();
            if self.encoder.is_none() {
                self.encoder = Some(
                    self.command_buffer
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandCreation)?,
                );
            }
            let encoder = self.encoder.as_ref().expect("compute encoder exists");
            let pipeline = self
                .runtime
                .pipelines
                .get(kernel)
                .ok_or_else(|| MetalError::MissingKernel(kernel.into()))?;
            encoder.setComputePipelineState(&**pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer.native()), 0, index) };
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threadgroups.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_threadgroup,
                    height: 1,
                    depth: 1,
                },
            );
            self.complete_profiled_dispatch(
                kernel,
                0,
                threadgroups,
                threads_per_threadgroup,
                encode_started.elapsed(),
            )?;
            Ok(())
        }

        pub fn dispatch_1d(
            &mut self,
            kernel: &'static str,
            buffers: &[&GpuBuffer],
            count: usize,
        ) -> Result<(), MetalError> {
            let encode_started = Instant::now();
            if self.encoder.is_none() {
                self.encoder = Some(
                    self.command_buffer
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandCreation)?,
                );
            }
            let encoder = self.encoder.as_ref().expect("compute encoder exists");
            let pipeline = self
                .runtime
                .pipelines
                .get(kernel)
                .ok_or_else(|| MetalError::MissingKernel(kernel.into()))?;
            encoder.setComputePipelineState(&**pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer.native()), 0, index);
                }
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: count.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.runtime.pipeline_thread_width(kernel),
                    height: 1,
                    depth: 1,
                },
            );
            self.complete_profiled_dispatch(
                kernel,
                count,
                0,
                self.runtime.pipeline_thread_width(kernel),
                encode_started.elapsed(),
            )?;
            Ok(())
        }

        /// Dispatch with byte offsets into resident buffers.  Decode keeps
        /// scalar constants in a small set of persistent buffers, so offsets
        /// avoid allocating a buffer for every kernel argument.
        pub fn dispatch_1d_at(
            &mut self,
            kernel: &'static str,
            buffers: &[(&GpuBuffer, usize)],
            count: usize,
        ) -> Result<(), MetalError> {
            let encode_started = Instant::now();
            if self.encoder.is_none() {
                self.encoder = Some(
                    self.command_buffer
                        .computeCommandEncoder()
                        .ok_or(MetalError::CommandCreation)?,
                );
            }
            let encoder = self.encoder.as_ref().expect("compute encoder exists");
            let pipeline = self
                .runtime
                .pipelines
                .get(kernel)
                .ok_or_else(|| MetalError::MissingKernel(kernel.into()))?;
            encoder.setComputePipelineState(&**pipeline);
            for (index, (buffer, offset)) in buffers.iter().enumerate() {
                if *offset > buffer.bytes {
                    return Err(MetalError::InvalidInput(
                        "resident buffer offset is out of range".into(),
                    ));
                }
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer.native()), *offset, index);
                }
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: count.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: self.runtime.pipeline_thread_width(kernel),
                    height: 1,
                    depth: 1,
                },
            );
            self.complete_profiled_dispatch(
                kernel,
                count,
                0,
                self.runtime.pipeline_thread_width(kernel),
                encode_started.elapsed(),
            )?;
            Ok(())
        }

        fn complete_profiled_dispatch(
            &mut self,
            kernel: &'static str,
            threads: usize,
            threadgroups: usize,
            threads_per_threadgroup: usize,
            cpu_encode: Duration,
        ) -> Result<(), MetalError> {
            if !self.exact_per_dispatch {
                return Ok(());
            }
            let timing = self.submit_current()?;
            self.kernel_timings.push(ResidentKernelTiming {
                kernel,
                threads,
                threadgroups,
                threads_per_threadgroup,
                cpu_encode,
                timing,
            });
            self.command_buffer = self
                .runtime
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandCreation)?;
            Ok(())
        }

        fn submit_current(&mut self) -> Result<DispatchTiming, MetalError> {
            if let Some(encoder) = self.encoder.take() {
                encoder.endEncoding();
            }
            let started = Instant::now();
            self.runtime
                .command_buffer_count
                .fetch_add(1, Ordering::Relaxed);
            let schedule_started = Instant::now();
            self.command_buffer.commit();
            let command_buffer_schedule = schedule_started.elapsed();
            self.command_buffer.waitUntilCompleted();
            let wall_time = started.elapsed();
            if self.command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err(MetalError::CommandFailed(
                    self.command_buffer
                        .error()
                        .map(|error| error.to_string())
                        .unwrap_or_else(|| "unknown command-buffer error".into()),
                ));
            }
            let gpu_start = self.command_buffer.GPUStartTime();
            let gpu_end = self.command_buffer.GPUEndTime();
            let gpu_time = (gpu_end > gpu_start && gpu_start > 0.0)
                .then(|| Duration::from_secs_f64(gpu_end - gpu_start));
            if let Some(gpu_time) = gpu_time {
                self.runtime.gpu_execution_nanos.fetch_add(
                    u64::try_from(gpu_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
            }
            Ok(DispatchTiming {
                wall_time,
                command_buffer_schedule,
                gpu_time,
            })
        }

        pub fn take_kernel_timings(&mut self) -> Vec<ResidentKernelTiming> {
            std::mem::take(&mut self.kernel_timings)
        }

        pub fn finish(mut self) -> Result<DispatchTiming, MetalError> {
            // The profiled path submits after every dispatch. Do not add an
            // empty command buffer at the end of the token.
            if self.exact_per_dispatch {
                return Ok(DispatchTiming {
                    wall_time: Duration::ZERO,
                    command_buffer_schedule: Duration::ZERO,
                    gpu_time: Some(Duration::ZERO),
                });
            }
            self.submit_current()
        }
    }

    /// Lifetime class used by the Phase 1 Metal buffer pool.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum AllocationClass {
        ModelWeights,
        KvCache,
        SessionState,
        Activations,
        Constants,
    }

    /// Per-class residency and allocation counters.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct ClassAllocationMetrics {
        pub new_buffer_allocations: u64,
        pub reused_buffer_leases: u64,
        pub resident_bytes: usize,
        pub active_bytes: usize,
        pub peak_active_bytes: usize,
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct AllocationTelemetry {
        pub by_class: BTreeMap<AllocationClass, ClassAllocationMetrics>,
    }

    impl AllocationTelemetry {
        pub fn class(&self, class: AllocationClass) -> ClassAllocationMetrics {
            self.by_class.get(&class).copied().unwrap_or_default()
        }
    }

    /// A checked-out shared Metal buffer. Return it with [`MetalBufferPool::release`].
    pub struct PooledBuffer {
        // Keeps the native Metal allocation alive while the lease is checked out
        // or retained by the free list.
        _buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        class: AllocationClass,
        capacity: usize,
        allocation_id: u64,
    }

    impl PooledBuffer {
        pub fn class(&self) -> AllocationClass {
            self.class
        }
        pub fn capacity(&self) -> usize {
            self.capacity
        }
        pub fn storage(&self, registry_id: u64, read_only: bool) -> Storage {
            Storage::metal(registry_id, self.allocation_id, self.capacity, read_only)
        }
    }

    /// Reuses shared-memory buffers while keeping model, cache, and activation
    /// lifetimes in separate pools.
    pub struct MetalBufferPool {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        registry_id: u64,
        next_allocation_id: u64,
        free: HashMap<(AllocationClass, usize), Vec<PooledBuffer>>,
        telemetry: AllocationTelemetry,
    }

    impl MetalBufferPool {
        fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
            let registry_id = device.registryID();
            Self {
                device,
                registry_id,
                next_allocation_id: 1,
                free: HashMap::new(),
                telemetry: AllocationTelemetry::default(),
            }
        }

        pub fn checkout(
            &mut self,
            class: AllocationClass,
            requested_bytes: usize,
        ) -> Result<PooledBuffer, MetalError> {
            let capacity = requested_bytes
                .max(1)
                .checked_next_power_of_two()
                .ok_or_else(|| MetalError::InvalidInput("buffer capacity overflow".into()))?;
            let metrics = self.telemetry.by_class.entry(class).or_default();
            if let Some(buffer) = self.free.get_mut(&(class, capacity)).and_then(Vec::pop) {
                metrics.reused_buffer_leases += 1;
                metrics.active_bytes += capacity;
                metrics.peak_active_bytes = metrics.peak_active_bytes.max(metrics.active_bytes);
                return Ok(buffer);
            }
            let buffer = self
                .device
                .newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    MetalError::InvalidInput("Metal could not allocate a shared buffer".into())
                })?;
            let allocation_id = self.next_allocation_id;
            self.next_allocation_id += 1;
            metrics.new_buffer_allocations += 1;
            metrics.resident_bytes += capacity;
            metrics.active_bytes += capacity;
            metrics.peak_active_bytes = metrics.peak_active_bytes.max(metrics.active_bytes);
            Ok(PooledBuffer {
                _buffer: buffer,
                class,
                capacity,
                allocation_id,
            })
        }

        pub fn release(&mut self, buffer: PooledBuffer) {
            let metrics = self.telemetry.by_class.entry(buffer.class).or_default();
            metrics.active_bytes = metrics.active_bytes.saturating_sub(buffer.capacity);
            self.free
                .entry((buffer.class, buffer.capacity))
                .or_default()
                .push(buffer);
        }

        pub fn telemetry(&self) -> AllocationTelemetry {
            self.telemetry.clone()
        }
        pub fn registry_id(&self) -> u64 {
            self.registry_id
        }
    }

    pub struct MetalRuntime {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        command_buffer_count: AtomicU64,
        gpu_execution_nanos: AtomicU64,
        readback_bytes: AtomicU64,
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
                "vector_multiply_f32",
                "embedding_lookup_f32",
                "rms_norm_f32",
                "matvec_f32",
                "matvec_q4_0",
                "matvec_q8_0",
                "embedding_lookup_q4_0",
                "embedding_lookup_q8_0",
                "quantize_q4_0",
                "quantize_q8_0",
                "matvec_tiled_f32",
                "matmul_f32",
                "rope_f32",
                "masked_softmax_f32",
                "attention_scores_f32",
                "attention_values_f32",
                "logits_process_f32",
                "rope_llama_decode_f32",
                "rope_half_to_interleaved_f32",
                "rope_interleaved_to_half_f32",
                "kv_append_decode_f32",
                "attention_decode_f32",
                "attention_decode_fused_f32",
                "attention_scores_resident_f32",
                "masked_softmax_resident_f32",
                "attention_values_resident_f32",
                "argmax_f32",
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
                command_buffer_count: AtomicU64::new(0),
                gpu_execution_nanos: AtomicU64::new(0),
                readback_bytes: AtomicU64::new(0),
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

        pub fn buffer_pool(&self) -> MetalBufferPool {
            MetalBufferPool::new(self.device.clone())
        }

        /// Upload immutable model data once and retain the resulting Metal
        /// buffer for the lifetime of the owning model/executor.
        pub fn upload_f32(&self, values: &[f32]) -> Result<GpuBuffer, MetalError> {
            let buffer = self.buffer_from_slice(values)?;
            Ok(GpuBuffer {
                _buffer: buffer,
                bytes: values.len() * size_of::<f32>(),
            })
        }

        pub fn upload_bytes(&self, values: &[u8]) -> Result<GpuBuffer, MetalError> {
            self.buffer_from_slice(values).map(|buffer| GpuBuffer {
                _buffer: buffer,
                bytes: values.len(),
            })
        }

        /// Quantize a complete block-32 matrix into the GGUF wire layout.
        /// The result is read back only because GGUF is a disk artifact; Atlas
        /// inference retains the same packed bytes in a resident GPU buffer.
        pub fn quantize_gguf(
            &self,
            values: &[f32],
            format: GgufTensorType,
        ) -> Result<(Vec<u8>, DispatchTiming), MetalError> {
            if !values.len().is_multiple_of(32)
                || !matches!(format, GgufTensorType::Q4_0 | GgufTensorType::Q8_0)
            {
                return Err(MetalError::InvalidInput(
                    "GGUF GPU quantization requires Q4_0/Q8_0 block-32 values".into(),
                ));
            }
            let input = self.buffer_from_slice(values)?;
            let mut output = vec![0u8; format.block_bytes() * (values.len() / 32)];
            let output_buffer = self.buffer_from_slice(&output)?;
            let blocks = u32::try_from(values.len() / 32)
                .map_err(|_| MetalError::InvalidInput("GGUF tensor is too large".into()))?;
            let blocks_buffer = self.buffer_from_slice(&[blocks])?;
            let kernel = if format == GgufTensorType::Q4_0 {
                "quantize_q4_0"
            } else {
                "quantize_q8_0"
            };
            let timing = self.dispatch_1d(
                kernel,
                &[&input, &output_buffer, &blocks_buffer],
                values.len() / 32,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn upload_u32(&self, values: &[u32]) -> Result<GpuBuffer, MetalError> {
            self.buffer_from_slice(values).map(|buffer| GpuBuffer {
                _buffer: buffer,
                bytes: values.len() * size_of::<u32>(),
            })
        }

        pub fn allocate(&self, bytes: usize) -> Result<GpuBuffer, MetalError> {
            let bytes = bytes.max(1);
            let buffer = self
                .device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    MetalError::InvalidInput("Metal could not allocate a shared buffer".into())
                })?;
            Ok(GpuBuffer {
                _buffer: buffer,
                bytes,
            })
        }

        pub fn write_f32(&self, buffer: &GpuBuffer, values: &[f32]) -> Result<(), MetalError> {
            self.write_buffer(buffer, values)
        }

        pub fn write_u32(&self, buffer: &GpuBuffer, values: &[u32]) -> Result<(), MetalError> {
            self.write_buffer(buffer, values)
        }

        pub fn read_u32(&self, buffer: &GpuBuffer) -> Result<u32, MetalError> {
            if buffer.bytes < size_of::<u32>() {
                return Err(MetalError::InvalidInput(
                    "u32 readback buffer is too small".into(),
                ));
            }
            let value =
                unsafe { ptr::read_unaligned(buffer.native().contents().as_ptr().cast::<u32>()) };
            self.readback_bytes
                .fetch_add(size_of::<u32>() as u64, Ordering::Relaxed);
            Ok(value)
        }

        pub fn read_f32(&self, buffer: &GpuBuffer, count: usize) -> Result<Vec<f32>, MetalError> {
            let bytes = count
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| MetalError::InvalidInput("f32 readback size overflow".into()))?;
            if bytes > buffer.bytes {
                return Err(MetalError::InvalidInput(
                    "f32 readback buffer is too small".into(),
                ));
            }
            let mut values = vec![0.0; count];
            self.copy_buffer_to_slice(buffer.native(), &mut values)?;
            Ok(values)
        }

        pub fn begin_resident_command(&self) -> Result<ResidentCommand<'_>, MetalError> {
            self.begin_resident_command_with_exact_timing(false)
        }

        pub fn begin_resident_command_with_exact_timing(
            &self,
            exact_per_dispatch: bool,
        ) -> Result<ResidentCommand<'_>, MetalError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandCreation)?;
            Ok(ResidentCommand {
                runtime: self,
                command_buffer,
                encoder: None,
                exact_per_dispatch,
                kernel_timings: Vec::new(),
            })
        }

        /// Number of submitted command buffers since runtime creation.  This
        /// is intentionally cumulative so callers can take token boundaries
        /// by subtraction without perturbing the runtime.
        pub fn command_buffer_count(&self) -> u64 {
            self.command_buffer_count.load(Ordering::Relaxed)
        }

        pub fn gpu_execution_time(&self) -> Duration {
            Duration::from_nanos(self.gpu_execution_nanos.load(Ordering::Relaxed))
        }

        pub fn readback_bytes(&self) -> u64 {
            self.readback_bytes.load(Ordering::Relaxed)
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

        pub fn vector_multiply(
            &self,
            lhs: &[f32],
            rhs: &[f32],
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            if lhs.len() != rhs.len() || lhs.is_empty() {
                return Err(MetalError::InvalidInput(
                    "multiply requires equally sized non-empty vectors".into(),
                ));
            }
            let count = count_u32(lhs.len())?;
            let mut output = vec![0.0; lhs.len()];
            let lhs_buffer = self.buffer_from_slice(lhs)?;
            let rhs_buffer = self.buffer_from_slice(rhs)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let count_buffer = self.buffer_from_slice(&[count])?;
            let timing = self.dispatch_1d(
                "vector_multiply_f32",
                &[&lhs_buffer, &rhs_buffer, &output_buffer, &count_buffer],
                lhs.len(),
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn embedding_lookup(
            &self,
            table: &[f32],
            vocabulary: usize,
            hidden: usize,
            token_ids: &[u32],
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            require_len(table, vocabulary.checked_mul(hidden), "embedding table")?;
            if hidden == 0
                || token_ids.is_empty()
                || token_ids.iter().any(|&token| token as usize >= vocabulary)
            {
                return Err(MetalError::InvalidInput(
                    "embedding dimensions or token IDs are invalid".into(),
                ));
            }
            let total = token_ids.len().checked_mul(hidden).ok_or_else(|| {
                MetalError::InvalidInput("embedding output length overflow".into())
            })?;
            let mut output = vec![0.0; total];
            let table_buffer = self.buffer_from_slice(table)?;
            let ids_buffer = self.buffer_from_slice(token_ids)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let vocabulary_buffer = self.buffer_from_slice(&[count_u32(vocabulary)?])?;
            let hidden_buffer = self.buffer_from_slice(&[count_u32(hidden)?])?;
            let tokens_buffer = self.buffer_from_slice(&[count_u32(token_ids.len())?])?;
            let timing = self.dispatch_1d(
                "embedding_lookup_f32",
                &[
                    &table_buffer,
                    &ids_buffer,
                    &output_buffer,
                    &vocabulary_buffer,
                    &hidden_buffer,
                    &tokens_buffer,
                ],
                total,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn rms_norm(
            &self,
            input: &[f32],
            rows: usize,
            hidden: usize,
            weight: &[f32],
            epsilon: f32,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            let elements = checked_product(rows, hidden, "RMSNorm dimensions")?;
            require_len(input, Some(elements), "RMSNorm input")?;
            require_len(weight, Some(hidden), "RMSNorm weight")?;
            if rows == 0 || hidden == 0 || !epsilon.is_finite() || epsilon <= 0.0 {
                return Err(MetalError::InvalidInput(
                    "RMSNorm dimensions or epsilon are invalid".into(),
                ));
            }
            let mut output = vec![0.0; elements];
            let input_buffer = self.buffer_from_slice(input)?;
            let weight_buffer = self.buffer_from_slice(weight)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let hidden_buffer = self.buffer_from_slice(&[count_u32(hidden)?])?;
            let epsilon_buffer = self.buffer_from_slice(&[epsilon])?;
            let timing = self.dispatch_1d(
                "rms_norm_f32",
                &[
                    &input_buffer,
                    &weight_buffer,
                    &output_buffer,
                    &hidden_buffer,
                    &epsilon_buffer,
                ],
                rows,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn matvec(
            &self,
            input: &[f32],
            weights: &[f32],
            input_width: usize,
            output_width: usize,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            require_len(input, Some(input_width), "matvec input")?;
            require_len(
                weights,
                Some(checked_product(
                    output_width,
                    input_width,
                    "matvec weights",
                )?),
                "matvec weights",
            )?;
            if input_width == 0 || output_width == 0 {
                return Err(MetalError::InvalidInput(
                    "matvec dimensions must be non-zero".into(),
                ));
            }
            let mut output = vec![0.0; output_width];
            let input_buffer = self.buffer_from_slice(input)?;
            let weights_buffer = self.buffer_from_slice(weights)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let input_width_buffer = self.buffer_from_slice(&[count_u32(input_width)?])?;
            let output_width_buffer = self.buffer_from_slice(&[count_u32(output_width)?])?;
            let timing = self.dispatch_1d(
                "matvec_f32",
                &[
                    &input_buffer,
                    &weights_buffer,
                    &output_buffer,
                    &input_width_buffer,
                    &output_width_buffer,
                ],
                output_width,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        /// Execute a GGUF block-32 projection without materializing an FP32
        /// weight matrix. This is deliberately exposed for packed-kernel
        /// parity diagnostics; normal resident execution binds the same
        /// buffers directly inside its one-token command buffer.
        pub fn matvec_gguf_packed(
            &self,
            input: &[f32],
            weights: &[u8],
            format: GgufTensorType,
            input_width: usize,
            output_width: usize,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            if !matches!(format, GgufTensorType::Q4_0 | GgufTensorType::Q8_0)
                || input_width == 0
                || output_width == 0
                || !input_width.is_multiple_of(32)
            {
                return Err(MetalError::InvalidInput(
                    "packed matvec requires Q4_0/Q8_0 and a non-zero block-32 width".into(),
                ));
            }
            require_len(input, Some(input_width), "packed matvec input")?;
            require_len(
                weights,
                Some(checked_product(
                    output_width,
                    input_width / 32 * format.block_bytes(),
                    "packed matvec weights",
                )?),
                "packed matvec weights",
            )?;
            let mut output = vec![0.0; output_width];
            let input_buffer = self.buffer_from_slice(input)?;
            let weights_buffer = self.buffer_from_slice(weights)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let input_width_buffer = self.buffer_from_slice(&[count_u32(input_width)?])?;
            let output_width_buffer = self.buffer_from_slice(&[count_u32(output_width)?])?;
            let kernel = if format == GgufTensorType::Q4_0 {
                "matvec_q4_0"
            } else {
                "matvec_q8_0"
            };
            let timing = self.dispatch_1d(
                kernel,
                &[
                    &input_buffer,
                    &weights_buffer,
                    &output_buffer,
                    &input_width_buffer,
                    &output_width_buffer,
                ],
                output_width,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn matmul(
            &self,
            input: &[f32],
            weights: &[f32],
            rows: usize,
            input_width: usize,
            output_width: usize,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            let input_elements = checked_product(rows, input_width, "matmul input")?;
            let output_elements = checked_product(rows, output_width, "matmul output")?;
            require_len(input, Some(input_elements), "matmul input")?;
            require_len(
                weights,
                Some(checked_product(
                    output_width,
                    input_width,
                    "matmul weights",
                )?),
                "matmul weights",
            )?;
            if rows == 0 || input_width == 0 || output_width == 0 {
                return Err(MetalError::InvalidInput(
                    "matmul dimensions must be non-zero".into(),
                ));
            }
            let mut output = vec![0.0; output_elements];
            let input_buffer = self.buffer_from_slice(input)?;
            let weights_buffer = self.buffer_from_slice(weights)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let rows_buffer = self.buffer_from_slice(&[count_u32(rows)?])?;
            let input_width_buffer = self.buffer_from_slice(&[count_u32(input_width)?])?;
            let output_width_buffer = self.buffer_from_slice(&[count_u32(output_width)?])?;
            let timing = self.dispatch_1d(
                "matmul_f32",
                &[
                    &input_buffer,
                    &weights_buffer,
                    &output_buffer,
                    &rows_buffer,
                    &input_width_buffer,
                    &output_width_buffer,
                ],
                output_elements,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn rope(
            &self,
            input: &[f32],
            rows: usize,
            hidden: usize,
            cosine: &[f32],
            sine: &[f32],
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            let elements = checked_product(rows, hidden, "RoPE input")?;
            require_len(input, Some(elements), "RoPE input")?;
            require_len(cosine, Some(hidden / 2), "RoPE cosine")?;
            require_len(sine, Some(hidden / 2), "RoPE sine")?;
            if rows == 0 || hidden == 0 || hidden % 2 != 0 {
                return Err(MetalError::InvalidInput(
                    "RoPE hidden width must be a non-zero even number".into(),
                ));
            }
            let mut output = vec![0.0; elements];
            let input_buffer = self.buffer_from_slice(input)?;
            let cosine_buffer = self.buffer_from_slice(cosine)?;
            let sine_buffer = self.buffer_from_slice(sine)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let hidden_buffer = self.buffer_from_slice(&[count_u32(hidden)?])?;
            let timing = self.dispatch_1d(
                "rope_f32",
                &[
                    &input_buffer,
                    &cosine_buffer,
                    &sine_buffer,
                    &output_buffer,
                    &hidden_buffer,
                ],
                rows * (hidden / 2),
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn masked_softmax(
            &self,
            input: &[f32],
            mask: &[f32],
            rows: usize,
            columns: usize,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            let elements = checked_product(rows, columns, "softmax input")?;
            require_len(input, Some(elements), "softmax input")?;
            require_len(mask, Some(elements), "softmax mask")?;
            if rows == 0 || columns == 0 {
                return Err(MetalError::InvalidInput(
                    "softmax dimensions must be non-zero".into(),
                ));
            }
            let mut output = vec![0.0; elements];
            let input_buffer = self.buffer_from_slice(input)?;
            let mask_buffer = self.buffer_from_slice(mask)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let columns_buffer = self.buffer_from_slice(&[count_u32(columns)?])?;
            let timing = self.dispatch_1d(
                "masked_softmax_f32",
                &[&input_buffer, &mask_buffer, &output_buffer, &columns_buffer],
                rows,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn attention_scores(
            &self,
            queries: &[f32],
            keys: &[f32],
            query_count: usize,
            key_count: usize,
            head_dim: usize,
            scale: f32,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            require_len(
                queries,
                Some(checked_product(query_count, head_dim, "query dimensions")?),
                "queries",
            )?;
            require_len(
                keys,
                Some(checked_product(key_count, head_dim, "key dimensions")?),
                "keys",
            )?;
            if query_count == 0 || key_count == 0 || head_dim == 0 || !scale.is_finite() {
                return Err(MetalError::InvalidInput(
                    "attention score dimensions or scale are invalid".into(),
                ));
            }
            let outputs = checked_product(query_count, key_count, "attention score output")?;
            let mut output = vec![0.0; outputs];
            let queries_buffer = self.buffer_from_slice(queries)?;
            let keys_buffer = self.buffer_from_slice(keys)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let key_count_buffer = self.buffer_from_slice(&[count_u32(key_count)?])?;
            let head_dim_buffer = self.buffer_from_slice(&[count_u32(head_dim)?])?;
            let scale_buffer = self.buffer_from_slice(&[scale])?;
            let timing = self.dispatch_1d(
                "attention_scores_f32",
                &[
                    &queries_buffer,
                    &keys_buffer,
                    &output_buffer,
                    &key_count_buffer,
                    &head_dim_buffer,
                    &scale_buffer,
                ],
                outputs,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn attention_values(
            &self,
            weights: &[f32],
            values: &[f32],
            query_count: usize,
            key_count: usize,
            head_dim: usize,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            require_len(
                weights,
                Some(checked_product(
                    query_count,
                    key_count,
                    "attention weights",
                )?),
                "attention weights",
            )?;
            require_len(
                values,
                Some(checked_product(key_count, head_dim, "attention values")?),
                "attention values",
            )?;
            if query_count == 0 || key_count == 0 || head_dim == 0 {
                return Err(MetalError::InvalidInput(
                    "attention value dimensions must be non-zero".into(),
                ));
            }
            let outputs = checked_product(query_count, head_dim, "attention output")?;
            let mut output = vec![0.0; outputs];
            let weights_buffer = self.buffer_from_slice(weights)?;
            let values_buffer = self.buffer_from_slice(values)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let key_count_buffer = self.buffer_from_slice(&[count_u32(key_count)?])?;
            let head_dim_buffer = self.buffer_from_slice(&[count_u32(head_dim)?])?;
            let timing = self.dispatch_1d(
                "attention_values_f32",
                &[
                    &weights_buffer,
                    &values_buffer,
                    &output_buffer,
                    &key_count_buffer,
                    &head_dim_buffer,
                ],
                outputs,
            )?;
            self.copy_buffer_to_slice(&output_buffer, &mut output)?;
            Ok((output, timing))
        }

        pub fn process_logits(
            &self,
            logits: &[f32],
            bias: &[f32],
            temperature: f32,
        ) -> Result<(Vec<f32>, DispatchTiming), MetalError> {
            if logits.is_empty()
                || logits.len() != bias.len()
                || !temperature.is_finite()
                || temperature <= 0.0
            {
                return Err(MetalError::InvalidInput(
                    "logits, bias, or temperature are invalid".into(),
                ));
            }
            let mut output = vec![0.0; logits.len()];
            let logits_buffer = self.buffer_from_slice(logits)?;
            let bias_buffer = self.buffer_from_slice(bias)?;
            let output_buffer = self.buffer_from_slice(&output)?;
            let temperature_buffer = self.buffer_from_slice(&[temperature])?;
            let count_buffer = self.buffer_from_slice(&[count_u32(logits.len())?])?;
            let timing = self.dispatch_1d(
                "logits_process_f32",
                &[
                    &logits_buffer,
                    &bias_buffer,
                    &output_buffer,
                    &temperature_buffer,
                    &count_buffer,
                ],
                logits.len(),
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

        fn write_buffer<T: Copy>(
            &self,
            buffer: &GpuBuffer,
            values: &[T],
        ) -> Result<(), MetalError> {
            let bytes = values
                .len()
                .checked_mul(size_of::<T>())
                .ok_or_else(|| MetalError::InvalidInput("buffer size overflow".into()))?;
            if bytes > buffer.bytes {
                return Err(MetalError::InvalidInput(
                    "resident buffer is too small for write".into(),
                ));
            }
            unsafe {
                ptr::copy_nonoverlapping(
                    values.as_ptr().cast::<u8>(),
                    buffer.native().contents().as_ptr().cast::<u8>(),
                    bytes,
                );
            }
            Ok(())
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
            self.readback_bytes
                .fetch_add(u64::try_from(bytes).unwrap_or(u64::MAX), Ordering::Relaxed);
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
            self.command_buffer_count.fetch_add(1, Ordering::Relaxed);
            let schedule_started = Instant::now();
            command_buffer.commit();
            let command_buffer_schedule = schedule_started.elapsed();
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
            if let Some(gpu_time) = gpu_time {
                self.gpu_execution_nanos.fetch_add(
                    u64::try_from(gpu_time.as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
            }
            Ok(DispatchTiming {
                wall_time,
                command_buffer_schedule,
                gpu_time,
            })
        }
    }

    fn count_u32(count: usize) -> Result<u32, MetalError> {
        u32::try_from(count).map_err(|_| MetalError::InvalidInput("input is too large".into()))
    }

    fn checked_product(left: usize, right: usize, label: &str) -> Result<usize, MetalError> {
        left.checked_mul(right)
            .ok_or_else(|| MetalError::InvalidInput(format!("{label} overflow")))
    }

    fn require_len<T>(
        values: &[T],
        expected: Option<usize>,
        label: &str,
    ) -> Result<(), MetalError> {
        let expected =
            expected.ok_or_else(|| MetalError::InvalidInput(format!("{label} length overflow")))?;
        if values.len() != expected {
            return Err(MetalError::InvalidInput(format!(
                "{label} has length {}, expected {expected}",
                values.len()
            )));
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
pub use macos::*;
