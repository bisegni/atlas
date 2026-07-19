use std::{path::Path, sync::atomic::AtomicBool, time::Duration};

use atlas_core::QuantFormat;
use atlas_model::{
    AtlasModel,
    executor::{
        AtlasExecutor, ExecutorConfig, ExecutorMetrics, ExecutorMode, GenerationEvent,
        GenerationFinishReason, LogitsReadback, ResidentStage, compare_stage,
        resident_stage_tolerance,
    },
};

#[test]
fn phase_06_metrics_report_rates_and_latency_percentiles() {
    let metrics = ExecutorMetrics {
        prefill: Duration::from_millis(20),
        prefill_tokens: 10,
        decode: Duration::from_millis(40),
        decode_tokens: 4,
        decode_latencies: vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(10),
        ],
        ..Default::default()
    };
    assert_eq!(metrics.prefill_tokens_per_second(), 500.0);
    assert_eq!(metrics.decode_tokens_per_second(), 100.0);
    assert_eq!(metrics.decode_p50(), Duration::from_millis(3));
    assert_eq!(metrics.decode_p95(), Duration::from_millis(10));
}

#[test]
fn resident_stage_comparison_is_strict_and_reports_the_first_failure() {
    assert!(
        compare_stage(
            0,
            ResidentStage::Q,
            Some(1),
            &[1.0, -2.0],
            &[1.0, -2.0],
            1e-5
        )
        .is_none()
    );
    assert!(compare_stage(0, ResidentStage::Q, Some(1), &[1.0], &[1.0 + 5e-6], 1e-5).is_none());

    let mismatch = compare_stage(
        3,
        ResidentStage::Attention,
        Some(2),
        &[1.0, 2.0, 3.0],
        &[1.0, 2.1, 3.2],
        0.05,
    )
    .unwrap();
    assert_eq!(mismatch.prompt_token_index, 3);
    assert_eq!(mismatch.first_failing_index, Some(1));
    assert_eq!(mismatch.expected, 2.0);
    assert_eq!(mismatch.actual, 2.1);
    assert!((mismatch.max_abs_error - 0.2).abs() < 1e-5);

    let non_finite = compare_stage(
        0,
        ResidentStage::Logits,
        None,
        &[f32::NAN],
        &[f32::NAN],
        1e-5,
    )
    .unwrap();
    assert_eq!(non_finite.first_failing_index, Some(0));
    assert!(non_finite.max_abs_error.is_infinite());

    let length = compare_stage(0, ResidentStage::V, Some(0), &[1.0, 2.0], &[1.0], 1e-5).unwrap();
    assert_eq!(length.element_count, 2);
    assert_eq!(length.first_failing_index, Some(1));
    assert_eq!(length.expected, 2.0);
    assert!(length.actual.is_nan());
}

#[test]
fn resident_stage_tolerance_preserves_the_caller_gate_until_a_bound_is_proven() {
    assert_eq!(resident_stage_tolerance(ResidentStage::Q, 1e-5), 1e-5);
    assert_eq!(
        resident_stage_tolerance(ResidentStage::MlpDownProjection, 1e-5),
        1e-5
    );
    assert_eq!(
        resident_stage_tolerance(ResidentStage::MlpResidual, 3e-5),
        3e-5
    );
}

#[test]
fn phase_08a_metrics_expose_gpu_residency_observability() {
    let metrics = ExecutorMetrics {
        host_wall_time: Duration::from_millis(12),
        gpu_execution_time: Duration::from_millis(9),
        command_buffer_count: 1,
        weight_upload_bytes: 4096,
        readback_bytes: 4,
        resident_bytes: 4096,
        post_warmup_allocations: 0,
        ..Default::default()
    };
    assert!(metrics.host_wall_time >= metrics.gpu_execution_time);
    assert_eq!(metrics.command_buffer_count, 1);
    assert_eq!(metrics.weight_upload_bytes, 4096);
    assert_eq!(metrics.readback_bytes, 4);
    assert_eq!(metrics.resident_bytes, 4096);
    assert_eq!(metrics.post_warmup_allocations, 0);
}

#[test]
fn phase_11a_defaults_to_resident_for_production_inference() {
    assert_eq!(ExecutorConfig::default().mode, ExecutorMode::Resident);
    assert_eq!(
        ExecutorConfig::default().logits_readback,
        LogitsReadback::SelectedToken
    );
}

#[test]
fn phase_06_plan_rejects_packed_weights_until_packed_metal_kernels_exist() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    if !fixture.join("model.safetensors").exists() {
        return;
    }
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => {
            return;
        }
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    assert!(
        AtlasExecutor::new(
            &model,
            ExecutorConfig {
                quant_format: QuantFormat::Int8Block32,
                ..Default::default()
            }
        )
        .is_err()
    );
}

/// Run explicitly when validating a local Metal setup.  It deliberately uses
/// a short prompt because the fixture/model files are developer-local.
#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_06_cached_decode_matches_phase_3_reference_and_keeps_pipelines_warm() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    let reference = model.generate_greedy("Atlas", 2).unwrap();
    let mut executor = AtlasExecutor::new(
        &model,
        ExecutorConfig {
            max_context: 64,
            ..Default::default()
        },
    )
    .unwrap();
    let actual = executor.generate_greedy("Atlas", 2).unwrap();
    assert_eq!(
        actual.generation.generated_token_ids,
        reference.generated_token_ids
    );
    assert_eq!(
        actual.metrics.pipeline_count,
        actual.metrics.post_warmup_pipeline_count
    );
    assert_eq!(actual.metrics.post_warmup_allocations, 0);
    assert!(actual.metrics.ttft > Duration::ZERO);
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08a_model_weight_upload_is_once_per_loaded_model() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    let resident_config = ExecutorConfig {
        mode: ExecutorMode::Resident,
        ..Default::default()
    };
    let first = AtlasExecutor::new(&model, resident_config).unwrap();
    assert!(first.weight_upload_bytes() > 0);
    let second = AtlasExecutor::new(&model, resident_config).unwrap();
    assert_eq!(second.weight_upload_bytes(), 0);
}

/// Performance is hardware-sensitive, so this is an explicit Apple-Silicon
/// gate rather than part of the portable test suite. It keeps the reference
/// and cached-decode paths on the same fixture, prompt, and process.
#[test]
#[ignore = "requires local Metal, the downloaded small fixture, and a stable performance environment"]
fn phase_08a_cached_decode_is_faster_than_reference_and_keeps_greedy_parity() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    // Materialize resident weights and warm pipelines before measuring either
    // decode path, so this is not a model-load benchmark.
    let resident_config = ExecutorConfig {
        mode: ExecutorMode::Resident,
        ..Default::default()
    };
    let mut warmup = AtlasExecutor::new(&model, resident_config).unwrap();
    warmup.generate_greedy("Atlas", 3).unwrap();

    let reference_started = std::time::Instant::now();
    let reference = model.generate_greedy("Atlas", 3).unwrap();
    let reference_elapsed = reference_started.elapsed();
    let mut executor = AtlasExecutor::new(&model, resident_config).unwrap();
    let actual = executor.generate_greedy("Atlas", 3).unwrap();

    assert_eq!(
        actual.generation.generated_token_ids,
        reference.generated_token_ids
    );
    let tokens = actual.generation.generated_token_ids.len();
    let reference_rate = tokens as f64 / reference_elapsed.as_secs_f64();
    let executor_rate = actual.metrics.decode_tokens_per_second();
    eprintln!("phase_08a reference_tok_s={reference_rate:.2} executor_tok_s={executor_rate:.2}");
    assert!(
        executor_rate > 0.0,
        "executor must report a positive decode rate"
    );
    assert!(
        executor_rate > reference_rate,
        "cached decode regression: executor {executor_rate:.2} tok/s <= reference {reference_rate:.2} tok/s"
    );
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08c_resident_production_prefill_matches_reference_before_decode() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    let prompt = "The capital of France is";
    let mut reference = AtlasExecutor::new(
        &model,
        ExecutorConfig {
            mode: ExecutorMode::Reference,
            ..Default::default()
        },
    )
    .unwrap();
    let expected = reference.generate_greedy(prompt, 1).unwrap();
    let mut resident = AtlasExecutor::new(
        &model,
        ExecutorConfig {
            mode: ExecutorMode::Resident,
            ..Default::default()
        },
    )
    .unwrap();
    let actual = resident.generate_greedy(prompt, 1).unwrap();
    assert_eq!(
        actual.generation.generated_token_ids,
        expected.generation.generated_token_ids
    );
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert_eq!(
        actual.metrics.prefill_command_buffer_count,
        actual.metrics.prefill_tokens as u64
    );
    assert_eq!(
        actual.metrics.readback_bytes,
        4 * (actual.metrics.prefill_tokens + actual.metrics.decode_tokens) as u64
    );
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08c_resident_matches_reference_for_32_tokens_and_keeps_the_token_boundary_resident() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    for prompt in [
        "The capital of France is",
        "Atlas is a local inference engine.",
    ] {
        let mut reference = AtlasExecutor::new(
            &model,
            ExecutorConfig {
                mode: ExecutorMode::Reference,
                ..Default::default()
            },
        )
        .unwrap();
        let expected = reference.generate_greedy(prompt, 32).unwrap();
        let mut resident = AtlasExecutor::new(
            &model,
            ExecutorConfig {
                mode: ExecutorMode::Resident,
                ..Default::default()
            },
        )
        .unwrap();
        let actual = resident.generate_greedy(prompt, 32).unwrap();
        assert_eq!(
            actual.generation.generated_token_ids,
            expected.generation.generated_token_ids
        );
        assert_eq!(actual.finish_reason, expected.finish_reason);
        assert!(actual.generation.final_logits.is_empty());
        assert_eq!(
            actual.metrics.prefill_command_buffer_count,
            actual.metrics.prefill_tokens as u64
        );
        assert_eq!(
            actual.metrics.decode_command_buffer_count,
            actual.metrics.decode_tokens as u64
        );
        assert_eq!(
            actual.metrics.command_buffer_count,
            actual.metrics.prefill_command_buffer_count
                + actual.metrics.decode_command_buffer_count
        );
        assert_eq!(
            actual.metrics.readback_bytes,
            4 * (actual.metrics.prefill_tokens + actual.metrics.decode_tokens) as u64
        );
        assert_eq!(actual.metrics.post_warmup_allocations, 0);
        assert!(actual.metrics.resident_arena_allocations > 0);
    }
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08b_final_logits_are_explicit_diagnostics() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    let mut executor = AtlasExecutor::new(
        &model,
        ExecutorConfig {
            mode: ExecutorMode::Resident,
            logits_readback: LogitsReadback::FinalLogits,
            ..Default::default()
        },
    )
    .unwrap();
    let result = executor.generate_greedy("Atlas", 1).unwrap();
    assert_eq!(
        result.generation.final_logits.len(),
        model.config.vocab_size
    );
    assert!(result.metrics.readback_bytes > 4);
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08c_single_token_stage_trace_is_exact_at_position_zero() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    let token = model.tokenize("The").unwrap().into_iter().next().unwrap();
    assert!(
        AtlasExecutor::trace_resident_token_ids(&model, &[token], 1e-5)
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08c_prompt_prefix_stage_trace_has_no_divergence() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = match AtlasModel::load(&fixture) {
        Ok(model) => model,
        Err(error) if format!("{error:#}").contains("no Metal device is available") => return,
        Err(error) => panic!("load small fixture: {error:#}"),
    };
    assert!(
        AtlasExecutor::trace_resident_prompt(&model, "The capital of France is", 1e-5)
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08_streaming_matches_buffered_and_reports_terminal_metrics() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = AtlasModel::load(&fixture).unwrap();
    let mut buffered_executor = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
    let buffered = buffered_executor.generate_greedy("Atlas", 3).unwrap();

    let cancellation = AtomicBool::new(false);
    let mut events = Vec::new();
    let mut streamed_executor = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
    let streamed = streamed_executor
        .generate_greedy_stream("Atlas", 3, &cancellation, |event| {
            events.push(event);
            Ok(())
        })
        .unwrap();

    let streamed_ids = events
        .iter()
        .filter_map(|event| match event {
            GenerationEvent::Token { token_id, .. } => Some(*token_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(streamed_ids, buffered.generation.generated_token_ids);
    assert_eq!(
        streamed.generation.generated_token_ids,
        buffered.generation.generated_token_ids
    );
    assert!(matches!(
        events.last(),
        Some(GenerationEvent::Finished { .. })
    ));
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, GenerationEvent::Finished { .. }))
            .count(),
        1
    );
    assert!(streamed.metrics.ttft > Duration::ZERO);
    assert_eq!(
        streamed.metrics.decode_latencies.len(),
        streamed.metrics.decode_tokens
    );
    assert!(events.iter().any(|event| matches!(
        event,
        GenerationEvent::Finished {
            reason: GenerationFinishReason::Eos | GenerationFinishReason::MaxTokens,
            ..
        }
    )));
}

#[test]
#[ignore = "requires local Metal and the downloaded small fixture"]
fn phase_08_cancellation_is_a_terminal_failure_event() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture = root.join("models/hf/SmolLM2-135M-Instruct");
    let model = AtlasModel::load(&fixture).unwrap();
    let cancellation = AtomicBool::new(true);
    let mut events = Vec::new();
    let mut executor = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
    assert!(
        executor
            .generate_greedy_stream("Atlas", 3, &cancellation, |event| {
                events.push(event);
                Ok(())
            })
            .is_err()
    );
    assert!(matches!(
        events.as_slice(),
        [GenerationEvent::Failed { .. }]
    ));
}
