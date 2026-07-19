use atlas_metal::{MetalError, MetalRuntime};

#[test]
fn bootstrap_kernels_match_cpu_references_and_pipelines_are_cached() {
    let runtime = match MetalRuntime::new() {
        Ok(runtime) => runtime,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping GPU assertions: no Metal device is available to this process");
            return;
        }
        Err(error) => panic!("Metal runtime should initialize: {error}"),
    };
    assert_eq!(runtime.pipeline_count(), 5);

    let lhs: Vec<f32> = (0..1024).map(|value| value as f32 * 0.25).collect();
    let rhs: Vec<f32> = (0..1024).map(|value| -(value as f32) * 0.125).collect();
    let expected: Vec<f32> = lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect();

    let (result, first_timing) = runtime.vector_add(&lhs, &rhs).unwrap();
    assert!(first_timing.wall_time.as_nanos() > 0);
    assert!(first_timing.gpu_time.is_some());
    for (actual, expected) in result.iter().zip(expected) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }

    let (scaled, timing) = runtime.scalar_multiply(&lhs, -2.0).unwrap();
    assert!(timing.gpu_time.is_some());
    assert_eq!(
        scaled,
        lhs.iter().map(|value| value * -2.0).collect::<Vec<_>>()
    );

    let silu_input = [-2.0, -0.5, 0.0, 0.5, 2.0];
    let (silu, timing) = runtime.silu(&silu_input).unwrap();
    assert!(timing.gpu_time.is_some());
    for (actual, input) in silu.iter().zip(silu_input) {
        let expected = input / (1.0 + (-input).exp());
        assert!((actual - expected).abs() < 1e-6);
    }

    let (sum, timing) = runtime.sum(&lhs).unwrap();
    assert!(timing.gpu_time.is_some());
    assert!((sum - lhs.iter().sum::<f32>()).abs() < 1e-3);

    let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (transposed, timing) = runtime.transpose(&matrix, 2, 3).unwrap();
    assert!(timing.gpu_time.is_some());
    assert_eq!(transposed, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    for _ in 0..99 {
        let (result, _) = runtime.vector_add(&lhs, &rhs).unwrap();
        assert_eq!(
            result,
            lhs.iter().zip(&rhs).map(|(a, b)| a + b).collect::<Vec<_>>()
        );
    }
    assert_eq!(runtime.pipeline_count(), 5);
}
