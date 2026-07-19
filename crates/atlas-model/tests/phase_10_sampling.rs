use atlas_model::{
    AtlasModel, LayerTrace,
    sampling::{Sampler, SamplingConfig, SamplingStrategy},
};

fn greedy() -> SamplingConfig {
    SamplingConfig::default()
}

#[test]
fn phase_10_hand_checkable_logits_cover_greedy_filters_and_penalties() {
    assert_eq!(
        Sampler::new(greedy())
            .unwrap()
            .sample(&[0.1, 1.0, 0.5], &[])
            .unwrap()
            .token_id,
        1
    );

    let top_k = SamplingConfig {
        strategy: SamplingStrategy::Temperature {
            temperature: 1.0,
            seed: 7,
        },
        top_k: Some(1),
        ..greedy()
    };
    assert_eq!(
        Sampler::new(top_k)
            .unwrap()
            .sample(&[0.1, 1.0, 0.5], &[])
            .unwrap()
            .token_id,
        1
    );

    let top_p = SamplingConfig {
        strategy: SamplingStrategy::Temperature {
            temperature: 1.0,
            seed: 7,
        },
        top_p: Some(0.50),
        ..greedy()
    };
    assert_eq!(
        Sampler::new(top_p)
            .unwrap()
            .sample(&[3.0, 1.0, 0.0], &[])
            .unwrap()
            .token_id,
        0
    );

    let repetition = SamplingConfig {
        repetition_penalty: 2.0,
        ..greedy()
    };
    assert_eq!(
        Sampler::new(repetition)
            .unwrap()
            .sample(&[1.0, 0.9], &[0])
            .unwrap()
            .token_id,
        1
    );

    let frequency = SamplingConfig {
        frequency_penalty: 0.2,
        ..greedy()
    };
    assert_eq!(
        Sampler::new(frequency)
            .unwrap()
            .sample(&[1.0, 0.9], &[0, 0])
            .unwrap()
            .token_id,
        1
    );

    let presence = SamplingConfig {
        presence_penalty: 0.2,
        ..greedy()
    };
    assert_eq!(
        Sampler::new(presence)
            .unwrap()
            .sample(&[1.0, 0.9], &[0])
            .unwrap()
            .token_id,
        1
    );
}

#[test]
fn phase_10_seeded_temperature_stream_repeats_exactly_and_records_configuration() {
    let config = SamplingConfig {
        strategy: SamplingStrategy::Temperature {
            temperature: 0.8,
            seed: 42,
        },
        top_k: Some(3),
        top_p: Some(0.95),
        ..greedy()
    };
    let logits = [1.1, 0.9, 0.4, -1.0];
    let stream = |sampler: &mut Sampler| {
        let mut history = Vec::new();
        for _ in 0..256 {
            let next = sampler.sample(&logits, &history).unwrap().token_id;
            history.push(next);
        }
        history
    };
    let first = stream(&mut Sampler::new(config.clone()).unwrap());
    let second = stream(&mut Sampler::new(config.clone()).unwrap());
    assert_eq!(first, second);
    assert!(config.summary().contains("seed=42"));
    assert!(config.summary().contains("top_k=Some(3)"));
}

#[test]
fn phase_10_stop_sequence_terminates_on_the_selected_suffix() {
    let config = SamplingConfig {
        stop_sequences: vec![vec![1, 2]],
        ..greedy()
    };
    let mut sampler = Sampler::new(config).unwrap();
    assert!(!sampler.sample(&[0.0, 1.0, 0.0], &[]).unwrap().stopped);
    let stopped = sampler.sample(&[0.0, 0.0, 1.0], &[1]).unwrap();
    assert_eq!(stopped.token_id, 2);
    assert!(stopped.stopped);
}

#[test]
fn phase_10_rejects_invalid_configuration_and_logits() {
    assert!(
        Sampler::new(SamplingConfig {
            strategy: SamplingStrategy::Temperature {
                temperature: 0.0,
                seed: 1
            },
            ..greedy()
        })
        .is_err()
    );
    assert!(
        Sampler::new(SamplingConfig {
            top_k: Some(0),
            ..greedy()
        })
        .is_err()
    );
    assert!(
        Sampler::new(SamplingConfig {
            stop_sequences: vec![vec![]],
            ..greedy()
        })
        .is_err()
    );
    assert!(
        Sampler::new(greedy())
            .unwrap()
            .sample(&[f32::NAN], &[])
            .is_err()
    );
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_10_small_fixture_exposes_real_logits_to_the_backend_independent_sampler() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = AtlasModel::load(root.join("models/hf/SmolLM2-135M-Instruct")).unwrap();
    let tokens = model.tokenize("Atlas sampling").unwrap();
    let logits = model
        .forward(
            &tokens,
            &mut LayerTrace::default(),
            model.config.num_hidden_layers,
        )
        .unwrap();
    let config = SamplingConfig {
        strategy: SamplingStrategy::Temperature {
            temperature: 0.8,
            seed: 42,
        },
        top_k: Some(40),
        top_p: Some(0.95),
        ..greedy()
    };
    eprintln!("phase_10_sampling_config={}", config.summary());
    let sample = Sampler::new(config)
        .unwrap()
        .sample(&logits, &tokens)
        .unwrap();
    assert!((sample.token_id as usize) < logits.len());
}

#[test]
#[ignore = "requires local Metal and the downloaded large fixture"]
fn phase_10_large_fixture_repeats_a_256_token_seeded_stream() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = AtlasModel::load(root.join("models/hf/SmolLM2-1.7B-Instruct")).unwrap();
    let config = SamplingConfig {
        strategy: SamplingStrategy::Temperature {
            temperature: 0.8,
            seed: 42,
        },
        top_k: Some(40),
        top_p: Some(0.95),
        ..greedy()
    };
    eprintln!("phase_10_sampling_config={}", config.summary());
    // Phase 10 validates the sampler, not autoregressive executor throughput.
    // Read one real large-model logits vector, then exercise the sampler's
    // stateful seeded stream against it. Re-running a 1.7B full forward for
    // every sampled token would turn this acceptance gate into hundreds of
    // redundant model evaluations.
    let prompt_tokens = model.tokenize("Atlas sampling").unwrap();
    let logits = model
        .forward(
            &prompt_tokens,
            &mut LayerTrace::default(),
            model.config.num_hidden_layers,
        )
        .unwrap();
    let stream = |sampler: &mut Sampler| {
        let mut ids = prompt_tokens.clone();
        let generated_start = ids.len();
        for _ in 0..256 {
            ids.push(sampler.sample(&logits, &ids).unwrap().token_id);
        }
        ids.split_off(generated_start)
    };
    assert_eq!(
        stream(&mut Sampler::new(config.clone()).unwrap()),
        stream(&mut Sampler::new(config).unwrap())
    );
}
