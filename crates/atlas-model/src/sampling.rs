//! Backend-independent CPU token sampling.
//!
//! The sampler accepts only logits and token IDs.  It deliberately has no
//! dependency on AtlasModel, Metal, or the CLI so a later GPU implementation
//! can preserve this public configuration and result contract.

use anyhow::{Context, Result, ensure};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Selects either the highest-scoring candidate or a seeded random candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    Greedy,
    Temperature { temperature: f32, seed: u64 },
}

/// Policy applied to a logits vector before selecting its next token.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    pub strategy: SamplingStrategy,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Multiplicative penalty applied once to every token present in history.
    pub repetition_penalty: f32,
    /// Value subtracted once for each occurrence in history.
    pub frequency_penalty: f32,
    /// Value subtracted once when a token occurs at least once in history.
    pub presence_penalty: f32,
    /// Generated-token suffixes that terminate sampling once selected.
    pub stop_sequences: Vec<Vec<u32>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
        }
    }
}

impl SamplingConfig {
    /// A stable, compact description suitable for an artifact or caller log.
    pub fn summary(&self) -> String {
        let strategy = match self.strategy {
            SamplingStrategy::Greedy => "greedy".to_owned(),
            SamplingStrategy::Temperature { temperature, seed } => {
                format!("temperature={temperature},seed={seed}")
            }
        };
        format!(
            "strategy={strategy},top_k={:?},top_p={:?},repetition_penalty={},frequency_penalty={},presence_penalty={},stop_sequences={:?}",
            self.top_k,
            self.top_p,
            self.repetition_penalty,
            self.frequency_penalty,
            self.presence_penalty,
            self.stop_sequences,
        )
    }

    fn validate(&self) -> Result<()> {
        if let SamplingStrategy::Temperature { temperature, .. } = self.strategy {
            ensure!(
                temperature.is_finite() && temperature > 0.0,
                "sampling temperature must be finite and positive"
            );
        }
        if let Some(top_k) = self.top_k {
            ensure!(top_k > 0, "top_k must be positive");
        }
        if let Some(top_p) = self.top_p {
            ensure!(
                top_p.is_finite() && (0.0..=1.0).contains(&top_p),
                "top_p must be finite and in (0, 1]"
            );
            ensure!(top_p > 0.0, "top_p must be finite and in (0, 1]");
        }
        ensure!(
            self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0,
            "repetition_penalty must be finite and positive"
        );
        ensure!(
            self.frequency_penalty.is_finite(),
            "frequency_penalty must be finite"
        );
        ensure!(
            self.presence_penalty.is_finite(),
            "presence_penalty must be finite"
        );
        ensure!(
            self.stop_sequences
                .iter()
                .all(|sequence| !sequence.is_empty()),
            "stop sequences must not be empty"
        );
        Ok(())
    }
}

/// Result of selecting one token from a logits vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub token_id: u32,
    pub probability: f32,
    pub stopped: bool,
}

/// Stateful CPU sampler. The seeded RNG state advances once for each sampled
/// token, so identical configurations and inputs reproduce identical streams.
#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplingConfig,
    rng: Option<StdRng>,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Result<Self> {
        config.validate()?;
        let rng = match config.strategy {
            SamplingStrategy::Greedy => None,
            SamplingStrategy::Temperature { seed, .. } => Some(StdRng::seed_from_u64(seed)),
        };
        Ok(Self { config, rng })
    }

    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Select a token using `history` as the generated-token history for
    /// penalties and stop-sequence detection.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> Result<Sample> {
        ensure!(!logits.is_empty(), "sampling logits must not be empty");
        ensure!(
            logits.iter().all(|logit| logit.is_finite()),
            "sampling logits must all be finite"
        );
        let mut counts = vec![0usize; logits.len()];
        for &token in history {
            let index = usize::try_from(token).context("history token ID does not fit usize")?;
            ensure!(
                index < logits.len(),
                "history token ID {token} exceeds logits vocabulary"
            );
            counts[index] += 1;
        }

        let temperature = match self.config.strategy {
            SamplingStrategy::Greedy => 1.0,
            SamplingStrategy::Temperature { temperature, .. } => temperature,
        };
        let mut candidates = logits
            .iter()
            .enumerate()
            .map(|(token_id, &logit)| {
                let count = counts[token_id];
                let mut adjusted = logit;
                if count > 0 {
                    adjusted = if adjusted >= 0.0 {
                        adjusted / self.config.repetition_penalty
                    } else {
                        adjusted * self.config.repetition_penalty
                    };
                    adjusted -= self.config.frequency_penalty * count as f32;
                    adjusted -= self.config.presence_penalty;
                }
                (token_id, adjusted / temperature)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_id, left), (right_id, right)| {
            right.total_cmp(left).then_with(|| left_id.cmp(right_id))
        });
        if let Some(top_k) = self.config.top_k {
            candidates.truncate(top_k.min(candidates.len()));
        }

        let mut probabilities = softmax(&candidates)?;
        if let Some(top_p) = self.config.top_p {
            let mut retained = 0usize;
            let mut cumulative = 0.0;
            for probability in &probabilities {
                retained += 1;
                cumulative += *probability;
                if cumulative >= top_p {
                    break;
                }
            }
            candidates.truncate(retained);
            probabilities.truncate(retained);
            normalize(&mut probabilities)?;
        }

        let selected_index = match self.config.strategy {
            SamplingStrategy::Greedy => 0,
            SamplingStrategy::Temperature { .. } => categorical(
                self.rng.as_mut().expect("temperature sampler owns an RNG"),
                &probabilities,
            ),
        };
        let token_id = u32::try_from(candidates[selected_index].0)
            .context("sampled token ID does not fit u32")?;
        let stopped = self.config.stop_sequences.iter().any(|sequence| {
            sequence.last() == Some(&token_id)
                && history.ends_with(&sequence[..sequence.len().saturating_sub(1)])
        });
        Ok(Sample {
            token_id,
            probability: probabilities[selected_index],
            stopped,
        })
    }
}

fn softmax(candidates: &[(usize, f32)]) -> Result<Vec<f32>> {
    let max = candidates[0].1;
    let mut probabilities = candidates
        .iter()
        .map(|(_, logit)| (logit - max).exp())
        .collect::<Vec<_>>();
    normalize(&mut probabilities)?;
    Ok(probabilities)
}

fn normalize(probabilities: &mut [f32]) -> Result<()> {
    let total: f32 = probabilities.iter().sum();
    ensure!(
        total.is_finite() && total > 0.0,
        "sampling probabilities are invalid"
    );
    for probability in probabilities {
        *probability /= total;
    }
    Ok(())
}

fn categorical(rng: &mut StdRng, probabilities: &[f32]) -> usize {
    let draw = rng.r#gen::<f32>();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative || index + 1 == probabilities.len() {
            return index;
        }
    }
    unreachable!("a normalized categorical distribution always selects a candidate")
}
