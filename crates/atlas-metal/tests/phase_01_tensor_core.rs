use std::path::Path;

use atlas_core::{Device, read_safetensors_descriptors};
use atlas_metal::{AllocationClass, MetalError, MetalRuntime};

#[test]
fn classified_pool_reuses_decode_buffers_after_warmup() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!(
                "skipping GPU allocator assertions: no Metal device is available to this process"
            );
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    let mut pool = runtime.buffer_pool();

    let weights = pool.checkout(AllocationClass::ModelWeights, 13).unwrap();
    assert_eq!(weights.capacity(), 16);
    assert_eq!(weights.class(), AllocationClass::ModelWeights);
    assert!(matches!(
        weights.storage(pool.registry_id(), true).device,
        Device::Metal { .. }
    ));
    pool.release(weights);

    let warmup = pool.checkout(AllocationClass::Activations, 1536).unwrap();
    pool.release(warmup);
    let allocations_after_warmup = pool
        .telemetry()
        .class(AllocationClass::Activations)
        .new_buffer_allocations;
    for _ in 0..1_000 {
        let activation = pool.checkout(AllocationClass::Activations, 1536).unwrap();
        pool.release(activation);
    }

    let activation_metrics = pool.telemetry().class(AllocationClass::Activations);
    let weight_metrics = pool.telemetry().class(AllocationClass::ModelWeights);
    eprintln!(
        "allocation telemetry: activations resident={} peak_active={} steady_active={} new_buffers={}; model_weights resident={} peak_active={} steady_active={}",
        activation_metrics.resident_bytes,
        activation_metrics.peak_active_bytes,
        activation_metrics.active_bytes,
        activation_metrics.new_buffer_allocations,
        weight_metrics.resident_bytes,
        weight_metrics.peak_active_bytes,
        weight_metrics.active_bytes,
    );
    assert_eq!(
        activation_metrics.new_buffer_allocations,
        allocations_after_warmup
    );
    assert_eq!(activation_metrics.active_bytes, 0);
    assert!(activation_metrics.reused_buffer_leases >= 1_000);
    assert!(activation_metrics.peak_active_bytes >= 1536);
    assert_eq!(weight_metrics.resident_bytes, 16);
}

#[test]
fn downloaded_fixture_headers_produce_read_only_weight_descriptors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = root.join("models/hf/SmolLM2-135M-Instruct/model.safetensors");
    if !model.exists() {
        eprintln!("skipping fixture descriptors: run scripts/download-models.sh first");
        return;
    }
    let descriptors = read_safetensors_descriptors(&model).unwrap();
    assert!(!descriptors.is_empty());
    assert!(
        descriptors
            .iter()
            .all(|descriptor| descriptor.tensor.storage.read_only)
    );
    assert!(
        descriptors
            .iter()
            .any(|descriptor| descriptor.name == "model.embed_tokens.weight")
    );
}
