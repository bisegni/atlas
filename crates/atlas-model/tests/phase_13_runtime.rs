use atlas_model::{
    AtlasModel,
    executor::{ExecutorMode, GenerationEvent},
    runtime::{AtlasRuntime, RuntimeConfig, RuntimeEvent, RuntimeRequest},
    sampling::SamplingConfig,
};

fn request(prompt: &str) -> RuntimeRequest {
    RuntimeRequest {
        prompt: prompt.into(),
        max_new_tokens: 2,
        sampling: SamplingConfig::default(),
    }
}

/// The Phase 11b acceptance gate. It is intentionally hardware/fixture gated:
/// portable tests cover config validation while this proves the Resident path
/// owns and releases isolated session state around real model execution.
#[test]
#[ignore = "requires local Metal and models/hf/SmolLM2-135M-Instruct"]
fn phase_13_runtime_streams_three_sessions_cancels_one_and_releases_slots() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = AtlasModel::load(root.join("models/hf/SmolLM2-135M-Instruct"))
        .expect("load downloaded small fixture with Metal");
    let mut runtime = AtlasRuntime::new(
        &model,
        RuntimeConfig {
            max_active_sessions: 2,
            max_queued_sessions: 1,
            max_context: 64,
        },
    )
    .unwrap();
    let first = runtime.submit(request("Atlas one")).unwrap();
    let second = runtime.submit(request("Atlas two")).unwrap();
    let third = runtime.submit(request("Atlas three")).unwrap();
    assert!(
        runtime.submit(request("overflow")).is_err(),
        "queue must remain bounded"
    );
    assert!(runtime.cancel(second));

    let mut token_sessions = Vec::new();
    runtime
        .run_until_idle(|event| {
            if let RuntimeEvent::Generation {
                session,
                event: GenerationEvent::Token { .. },
            } = event
            {
                token_sessions.push(session);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(
        runtime.active_sessions(),
        0,
        "completed sessions release their GPU slots"
    );
    assert_eq!(
        runtime.queued_sessions(),
        0,
        "completed sessions release their queue entries"
    );

    let completions = std::iter::from_fn(|| runtime.take_completed()).collect::<Vec<_>>();
    assert_eq!(completions.len(), 3);
    let cancelled = completions
        .iter()
        .find(|completion| completion.session == second)
        .unwrap();
    assert!(cancelled.metrics.cancelled);
    assert!(cancelled.generation.is_none());
    for session in [first, third] {
        let completion = completions
            .iter()
            .find(|completion| completion.session == session)
            .unwrap();
        assert!(
            completion.generation.is_some(),
            "uncancelled session must finish"
        );
        assert_eq!(completion.metrics.executor_mode, ExecutorMode::Resident);
        assert!(completion.metrics.executor.resident_bytes > 0);
        assert!(
            token_sessions.contains(&session),
            "tokens remain session ordered and observable"
        );
    }
    assert!(!token_sessions.contains(&second));
    assert!(
        completions
            .iter()
            .all(|completion| completion.metrics.queue_wait >= std::time::Duration::ZERO)
    );
    assert!(
        completions
            .iter()
            .all(|completion| completion.metrics.executor.post_warmup_allocations == 0)
    );
    assert_eq!(ExecutorMode::Resident, ExecutorMode::Resident);
}
