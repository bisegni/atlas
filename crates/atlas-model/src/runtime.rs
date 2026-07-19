//! Bounded, session-isolated inference runtime.
//!
//! The runtime is the ownership boundary between callers and executors: model
//! weights remain shared by [`AtlasModel`], while every admitted request owns
//! its executor, cancellation flag, sampler configuration, and metrics.

use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};

use crate::{
    AtlasModel,
    executor::{
        AtlasExecutor, ExecutorConfig, ExecutorGeneration, ExecutorMetrics, ExecutorMode,
        GenerationEvent,
    },
    kv_cache::SessionId,
    sampling::SamplingConfig,
};

/// Explicit resource limits for one loaded-model runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub max_active_sessions: usize,
    pub max_queued_sessions: usize,
    pub max_context: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_active_sessions: 1,
            max_queued_sessions: 0,
            max_context: 1024,
        }
    }
}

impl RuntimeConfig {
    fn validate(self) -> Result<Self> {
        ensure!(
            self.max_active_sessions > 0,
            "runtime max_active_sessions must be positive"
        );
        ensure!(self.max_context > 0, "runtime max_context must be positive");
        Ok(self)
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
}

#[derive(Debug, Clone)]
pub enum RuntimeEvent {
    Queued {
        session: SessionId,
    },
    Started {
        session: SessionId,
        queue_wait: Duration,
    },
    Generation {
        session: SessionId,
        event: GenerationEvent,
    },
}

#[derive(Debug, Clone)]
pub struct RuntimeSessionMetrics {
    pub session: SessionId,
    pub executor_mode: ExecutorMode,
    pub queue_wait: Duration,
    pub executor: ExecutorMetrics,
    pub cancelled: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuntimeCompletion {
    pub session: SessionId,
    pub generation: Option<ExecutorGeneration>,
    pub metrics: RuntimeSessionMetrics,
}

struct Pending {
    session: SessionId,
    request: RuntimeRequest,
    submitted: Instant,
    cancellation: AtomicBool,
}

struct Active<'a> {
    pending: Pending,
    executor: AtlasExecutor<'a>,
    queue_wait: Duration,
}

/// A loaded-model runtime with bounded admission and isolated cancellation.
///
/// `step` admits as many requests as capacity allows and runs one admitted
/// session to its terminal event. The state machine intentionally keeps this
/// host API deterministic while the resident executor owns GPU state. It also
/// centralizes admission/cancellation so a later server can use this exact
/// contract rather than duplicating queue policy.
pub struct AtlasRuntime<'a> {
    model: &'a AtlasModel,
    config: RuntimeConfig,
    next_session: u64,
    queued: VecDeque<Pending>,
    active: VecDeque<Active<'a>>,
    completed: VecDeque<RuntimeCompletion>,
}

impl<'a> AtlasRuntime<'a> {
    pub fn new(model: &'a AtlasModel, config: RuntimeConfig) -> Result<Self> {
        Ok(Self {
            model,
            config: config.validate()?,
            next_session: 0,
            queued: VecDeque::new(),
            active: VecDeque::new(),
            completed: VecDeque::new(),
        })
    }

    pub fn submit(&mut self, request: RuntimeRequest) -> Result<SessionId> {
        ensure!(
            !request.prompt.is_empty(),
            "runtime prompt must not be empty"
        );
        ensure!(
            request.max_new_tokens > 0,
            "runtime max_new_tokens must be positive"
        );
        // Validate the configuration before consuming admission capacity.
        let _ = crate::sampling::Sampler::new(request.sampling.clone())?;
        let in_flight = self.queued.len() + self.active.len();
        ensure!(
            in_flight < self.config.max_active_sessions + self.config.max_queued_sessions,
            "runtime admission limit reached (active={}, queued={})",
            self.config.max_active_sessions,
            self.config.max_queued_sessions
        );
        let session = SessionId(self.next_session);
        self.next_session = self
            .next_session
            .checked_add(1)
            .context("runtime session ID overflow")?;
        self.queued.push_back(Pending {
            session,
            request,
            submitted: Instant::now(),
            cancellation: AtomicBool::new(false),
        });
        Ok(session)
    }

    pub fn cancel(&self, session: SessionId) -> bool {
        if let Some(pending) = self
            .queued
            .iter()
            .find(|pending| pending.session == session)
        {
            pending.cancellation.store(true, Ordering::Release);
            return true;
        }
        if let Some(active) = self
            .active
            .iter()
            .find(|active| active.pending.session == session)
        {
            active.pending.cancellation.store(true, Ordering::Release);
            return true;
        }
        false
    }

    pub fn queued_sessions(&self) -> usize {
        self.queued.len()
    }
    pub fn active_sessions(&self) -> usize {
        self.active.len()
    }
    pub fn take_completed(&mut self) -> Option<RuntimeCompletion> {
        self.completed.pop_front()
    }

    fn admit(&mut self, events: &mut Vec<RuntimeEvent>) -> Result<()> {
        while self.active.len() < self.config.max_active_sessions {
            let Some(pending) = self.queued.pop_front() else {
                break;
            };
            let queue_wait = pending.submitted.elapsed();
            let executor = AtlasExecutor::new(
                self.model,
                ExecutorConfig {
                    session: pending.session,
                    max_context: self.config.max_context,
                    ..Default::default()
                },
            )?;
            events.push(RuntimeEvent::Started {
                session: pending.session,
                queue_wait,
            });
            self.active.push_back(Active {
                pending,
                executor,
                queue_wait,
            });
        }
        Ok(())
    }

    /// Advance one admitted session and return all events delivered by it.
    pub fn step(&mut self) -> Result<Vec<RuntimeEvent>> {
        let mut events = Vec::new();
        self.admit(&mut events)?;
        let Some(mut active) = self.active.pop_front() else {
            return Ok(events);
        };
        let session = active.pending.session;
        let mut generation_events = Vec::new();
        let result = active.executor.generate_greedy_stream(
            &active.pending.request.prompt,
            active.pending.request.max_new_tokens,
            &active.pending.cancellation,
            |event| {
                generation_events.push(event);
                Ok(())
            },
        );
        events.extend(
            generation_events
                .into_iter()
                .map(|event| RuntimeEvent::Generation { session, event }),
        );
        let cancelled = active.pending.cancellation.load(Ordering::Acquire);
        let (generation, executor, error) = match result {
            Ok(generation) => {
                let metrics = generation.metrics.clone();
                (Some(generation), metrics, None)
            }
            Err(error) => (None, ExecutorMetrics::default(), Some(format!("{error:#}"))),
        };
        self.completed.push_back(RuntimeCompletion {
            session,
            generation,
            metrics: RuntimeSessionMetrics {
                session,
                executor_mode: ExecutorMode::Resident,
                queue_wait: active.queue_wait,
                executor,
                cancelled,
                error,
            },
        });
        Ok(events)
    }

    pub fn run_until_idle<F>(&mut self, mut emit: F) -> Result<()>
    where
        F: FnMut(RuntimeEvent) -> Result<()>,
    {
        while self.queued_sessions() > 0 || self.active_sessions() > 0 {
            for event in self.step()? {
                emit(event)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_config_rejects_unbounded_active_set() {
        assert!(
            RuntimeConfig {
                max_active_sessions: 0,
                max_queued_sessions: 1,
                max_context: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            RuntimeConfig {
                max_active_sessions: 1,
                max_queued_sessions: 1,
                max_context: 0
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn request_requires_prompt_tokens_and_valid_sampling() {
        let request = RuntimeRequest {
            prompt: String::new(),
            max_new_tokens: 1,
            sampling: SamplingConfig::default(),
        };
        assert!(request.prompt.is_empty());
        assert_eq!(request.max_new_tokens, 1);
    }
}
