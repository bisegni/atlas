//! Correctness-first neural operators backed by native Metal kernels.

use atlas_metal::{DispatchTiming, MetalError, MetalRuntime};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Prompt processing with matrix-matrix projection kernels.
    Prefill,
    /// One-token processing with matrix-vector projection kernels.
    Decode,
}

#[derive(Debug, Error)]
pub enum OperatorError {
    #[error(transparent)]
    Metal(#[from] MetalError),
}

pub struct NeuralOps {
    runtime: MetalRuntime,
}

impl NeuralOps {
    pub fn new() -> Result<Self, OperatorError> {
        Ok(Self {
            runtime: MetalRuntime::new()?,
        })
    }
    pub fn runtime(&self) -> &MetalRuntime {
        &self.runtime
    }

    pub fn embedding(
        &self,
        table: &[f32],
        vocabulary: usize,
        hidden: usize,
        token_ids: &[u32],
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self
            .runtime
            .embedding_lookup(table, vocabulary, hidden, token_ids)?)
    }
    pub fn add(
        &self,
        lhs: &[f32],
        rhs: &[f32],
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self.runtime.vector_add(lhs, rhs)?)
    }
    pub fn multiply(
        &self,
        lhs: &[f32],
        rhs: &[f32],
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self.runtime.vector_multiply(lhs, rhs)?)
    }
    pub fn silu(&self, input: &[f32]) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self.runtime.silu(input)?)
    }
    pub fn rms_norm(
        &self,
        input: &[f32],
        rows: usize,
        hidden: usize,
        weight: &[f32],
        epsilon: f32,
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self
            .runtime
            .rms_norm(input, rows, hidden, weight, epsilon)?)
    }
    pub fn rope(
        &self,
        input: &[f32],
        rows: usize,
        hidden: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self.runtime.rope(input, rows, hidden, cos, sin)?)
    }
    pub fn masked_softmax(
        &self,
        input: &[f32],
        mask: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self.runtime.masked_softmax(input, mask, rows, cols)?)
    }
    pub fn attention_scores(
        &self,
        queries: &[f32],
        keys: &[f32],
        query_count: usize,
        key_count: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self
            .runtime
            .attention_scores(queries, keys, query_count, key_count, head_dim, scale)?)
    }
    pub fn attention_values(
        &self,
        weights: &[f32],
        values: &[f32],
        query_count: usize,
        key_count: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self
            .runtime
            .attention_values(weights, values, query_count, key_count, head_dim)?)
    }
    pub fn process_logits(
        &self,
        logits: &[f32],
        bias: &[f32],
        temperature: f32,
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(self.runtime.process_logits(logits, bias, temperature)?)
    }

    pub fn project(
        &self,
        mode: ExecutionMode,
        input: &[f32],
        weights: &[f32],
        rows: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<(Vec<f32>, DispatchTiming), OperatorError> {
        Ok(match mode {
            ExecutionMode::Prefill => {
                self.runtime
                    .matmul(input, weights, rows, input_width, output_width)?
            }
            ExecutionMode::Decode => {
                self.runtime
                    .matvec(input, weights, input_width, output_width)?
            }
        })
    }
}
