use std::{path::Path, sync::atomic::AtomicBool, time::Duration};

use atlas_core::QuantFormat;
use atlas_model::{
    AtlasModel,
    executor::{
        AtlasExecutor, ExecutorConfig, ExecutorMetrics, GenerationEvent, GenerationFinishReason,
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
fn phase_08a_metrics_expose_gpu_residency_observability() {
    let metrics = ExecutorMetrics {
        host_wall_time: Duration::from_millis(12),
        gpu_execution_time: Duration::from_millis(9),
        command_buffer_count: 1,
        weight_upload_bytes: 4096,
        readback_bytes: 4,
        post_warmup_allocations: 0,
        ..Default::default()
    };
    assert!(metrics.host_wall_time >= metrics.gpu_execution_time);
    assert_eq!(metrics.command_buffer_count, 1);
    assert_eq!(metrics.weight_upload_bytes, 4096);
    assert_eq!(metrics.readback_bytes, 4);
    assert_eq!(metrics.post_warmup_allocations, 0);
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
    let first = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
    assert!(first.weight_upload_bytes() > 0);
    let second = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
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
    let mut warmup = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
    warmup.generate_greedy("Atlas", 3).unwrap();

    let reference_started = std::time::Instant::now();
    let reference = model.generate_greedy("Atlas", 3).unwrap();
    let reference_elapsed = reference_started.elapsed();
    let mut executor = AtlasExecutor::new(&model, ExecutorConfig::default()).unwrap();
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
