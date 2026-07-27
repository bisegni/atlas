//! Model-family boundary for Atlas's resident text inference.
//!
//! The CLI talks to this small object-safe surface only.  Family adapters keep
//! their tensor layouts and hot token-forward loops concrete; dynamic dispatch
//! happens once when a session is constructed, never inside generation.

use std::{sync::atomic::AtomicBool, time::Duration};

use anyhow::Result;

use crate::{
    Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
    gemma4_executor::{Gemma4E2bExecutor, Gemma4FinishReason, Gemma4KvCacheType},
    render_gemma4_chat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelIdentity {
    pub family: &'static str,
    pub format: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Model,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Eos,
    MaxTokens,
    Cancelled,
}

#[derive(Debug, Clone)]
pub struct ResidentMetrics {
    pub resident_bytes: u64,
    pub weight_upload_bytes: u64,
    pub readback_bytes: u64,
    pub command_buffers: u64,
    pub prefill_command_buffers: u64,
    pub decode_command_buffers: u64,
    pub prefill: Duration,
    pub decode: Duration,
    pub host_wall_time: Duration,
    pub prefill_path: &'static str,
    pub prefill_chunk_size: usize,
    pub prefill_chunks: usize,
    pub attention_kernel: &'static str,
    pub kv_cache_type: Gemma4KvCacheType,
    pub kv_cache_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct InferenceGeneration {
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    pub metrics: ResidentMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TokenEvent {
    pub token_id: u32,
    pub text: String,
    pub latency: Duration,
}

/// Family-neutral model capabilities needed by normal CLI inference.
pub trait InferenceModel {
    fn identity(&self) -> ModelIdentity;
    fn max_context(&self) -> usize;
    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String>;
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>>;
    fn start_resident_session(
        &self,
        max_context: usize,
    ) -> Result<Box<dyn ResidentInferenceSession + '_>>;
}

/// A stateful, GPU-resident request session.  This deliberately has no
/// reference-mode entry point: production failures must be surfaced.
pub trait ResidentInferenceSession {
    fn reset(&mut self);
    fn generate_greedy_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<InferenceGeneration>;
}

/// The first adapter.  It owns the GGUF-backed Gemma model while each session
/// borrows it, leaving Gemma's Metal implementation monomorphized.
pub struct Gemma4E2bInferenceModel {
    model: Gemma4E2bModel,
}

impl Gemma4E2bInferenceModel {
    pub fn new(model: Gemma4E2bModel) -> Self {
        Self { model }
    }
}

impl InferenceModel for Gemma4E2bInferenceModel {
    fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            family: "gemma4-e2b",
            format: "gguf-gemma4-q4_0",
        }
    }

    fn max_context(&self) -> usize {
        4096
    }

    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String> {
        let messages = messages
            .iter()
            .map(|message| {
                Gemma4ChatMessage::new(
                    match message.role {
                        ChatRole::System => Gemma4ChatRole::System,
                        ChatRole::User => Gemma4ChatRole::User,
                        ChatRole::Model => Gemma4ChatRole::Model,
                    },
                    message.content.clone(),
                )
            })
            .collect::<Vec<_>>();
        render_gemma4_chat(&messages)
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>> {
        self.model.tokenize(prompt)
    }

    fn start_resident_session(
        &self,
        max_context: usize,
    ) -> Result<Box<dyn ResidentInferenceSession + '_>> {
        Ok(Box::new(Gemma4E2bInferenceSession {
            executor: Gemma4E2bExecutor::new(&self.model, max_context)?,
        }))
    }
}

struct Gemma4E2bInferenceSession<'a> {
    executor: Gemma4E2bExecutor<'a>,
}

impl ResidentInferenceSession for Gemma4E2bInferenceSession<'_> {
    fn reset(&mut self) {
        self.executor.reset();
    }

    fn generate_greedy_stream(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
        cancelled: &AtomicBool,
        emit: &mut dyn FnMut(TokenEvent) -> Result<()>,
    ) -> Result<InferenceGeneration> {
        let result = self.executor.generate_greedy_chat_stream(
            prompt,
            max_new_tokens,
            cancelled,
            |event| {
                emit(TokenEvent {
                    token_id: event.token_id,
                    text: event.text,
                    latency: event.latency,
                })
            },
        )?;
        let finish_reason = match result.finish_reason {
            Gemma4FinishReason::Eos => FinishReason::Eos,
            Gemma4FinishReason::MaxTokens => FinishReason::MaxTokens,
            Gemma4FinishReason::Cancelled => FinishReason::Cancelled,
        };
        Ok(InferenceGeneration {
            prompt_token_ids: result.generation.prompt_token_ids,
            generated_token_ids: result.generation.generated_token_ids,
            text: result.generation.text,
            finish_reason,
            metrics: ResidentMetrics {
                resident_bytes: result.metrics.resident_bytes,
                weight_upload_bytes: result.metrics.weight_upload_bytes,
                readback_bytes: result.metrics.readback_bytes,
                command_buffers: result.metrics.command_buffers,
                prefill_command_buffers: result.metrics.prefill_command_buffers,
                decode_command_buffers: result.metrics.decode_command_buffers,
                prefill: result.metrics.prefill,
                decode: result.metrics.decode,
                host_wall_time: result.metrics.host_wall_time,
                prefill_path: result.metrics.prefill_path,
                prefill_chunk_size: result.metrics.prefill_chunk_size,
                prefill_chunks: result.metrics.prefill_chunks,
                attention_kernel: result.metrics.attention_kernel,
                kv_cache_type: result.metrics.kv_cache_type,
                kv_cache_bytes: result.metrics.kv_cache_bytes,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_contract_preserves_roles_and_text() {
        let message = ChatMessage::new(ChatRole::User, "hello");
        assert_eq!(message.role, ChatRole::User);
        assert_eq!(message.content, "hello");
    }

    #[test]
    fn finish_reasons_are_family_neutral() {
        assert_ne!(FinishReason::Eos, FinishReason::MaxTokens);
        assert_ne!(FinishReason::Cancelled, FinishReason::Eos);
    }
}
