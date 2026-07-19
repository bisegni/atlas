use std::{path::Path, time::Duration};

use atlas_core::QuantFormat;
use atlas_model::{
    AtlasModel,
    executor::{AtlasExecutor, ExecutorConfig, ExecutorMetrics},
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
    let model = AtlasModel::load(&fixture).unwrap();
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
