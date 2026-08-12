use std::{fs, path::Path, sync::atomic::AtomicBool};

use atlas_model::{
    Gemma4ChatMessage, Gemma4ChatRole, Gemma4E2bModel,
    gemma4_executor::{
        Gemma4E2bExecutor, Gemma4FinishReason, Gemma4KvCacheType, Gemma4Q4AttentionMode,
    },
    render_gemma4_chat,
};
use serde_json::Value;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf")
}

fn canonical() -> Value {
    fixture("gemma4-e2b-resident-canonical.json")
}

fn fixture(file_name: &str) -> Value {
    let path = fixture_directory().join(file_name);
    serde_json::from_slice(&fs::read(path).expect("read Gemma fixture"))
        .expect("parse Gemma fixture")
}

fn captured_oracle_fixture(file_name: &str) -> Value {
    let path = fixture_directory().join(file_name);
    let bytes = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing captured llama.cpp oracle fixture {}: {error}; run `bash scripts/capture-gemma4-llama-oracle.sh` first",
            path.display()
        )
    });
    serde_json::from_slice(&bytes).expect("parse captured llama.cpp oracle fixture")
}

fn fixture_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures")
}

fn ids(value: &Value, key: &str) -> Vec<u32> {
    value[key]
        .as_array()
        .expect("token ID array")
        .iter()
        .map(|id| id.as_u64().expect("u32 token ID") as u32)
        .collect()
}

fn verified_external_oracle(fixture: &Value, expected_finish: &str) {
    let oracle = fixture["external_oracle"]
        .as_object()
        .expect("canonical fixture must be recorded by the llama.cpp oracle capture script");
    assert_eq!(oracle["engine"].as_str(), Some("llama.cpp"));
    assert_eq!(oracle["status"].as_str(), Some("verified"));
    assert_eq!(oracle["finish_reason"].as_str(), Some(expected_finish));
}

fn fixture_finish_reason(fixture: &Value) -> Gemma4FinishReason {
    match fixture["finish_reason"].as_str() {
        Some("eos") => Gemma4FinishReason::Eos,
        Some("max_tokens") => Gemma4FinishReason::MaxTokens,
        finish => panic!("unsupported oracle fixture finish reason: {finish:?}"),
    }
}

fn first_generated_token_divergence(left: &[u32], right: &[u32]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

fn first_logit_digest_divergence(left: &[[u8; 32]], right: &[[u8; 32]]) -> Option<usize> {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}

#[test]
#[ignore = "requires local Metal, the Gemma 4 E2B Q4 GGUF fixture, and per-token logit readback"]
fn flash16_matches_legacy_resident_output_logit_digests() {
    // Arithmetic-level parity gate for the exact flash16 attention candidate.
    // Beyond equal greedy tokens, every generated token must produce a
    // byte-identical fp32 logit vector, captured as the per-token SHA-256
    // digest added to `Generation.logit_digests`. Token equality alone cannot
    // catch a small logit drift that flips a later token; the digest proves
    // the flash16 kernel reproduces LegacyFused's arithmetic exactly.
    unsafe {
        std::env::set_var("ATLAS_GEMMA4_TRACE_LOGIT_DIGESTS", "1");
    }
    let canonical = canonical();
    let canonical_prompt = canonical["prompt"].as_str().expect("canonical prompt");
    let code_prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
        Gemma4ChatRole::User,
        "write an hello world main c++ function",
    )])
    .expect("render C++ chat prompt");
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let cancelled = AtomicBool::new(false);

    for (name, prompt) in [
        ("canonical", canonical_prompt),
        ("long-cpp", code_prompt.as_str()),
    ] {
        let mut fast = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
            &model,
            4096,
            Gemma4KvCacheType::Q4_0,
            Gemma4Q4AttentionMode::Flash16,
        )
        .expect("create flash16 Resident executor");
        let mut legacy = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
            &model,
            4096,
            Gemma4KvCacheType::Q4_0,
            Gemma4Q4AttentionMode::LegacyFused,
        )
        .expect("create legacy Resident executor");

        let fast_generation = fast
            .generate_greedy_chat_stream(prompt, 64, &cancelled, |_| Ok(()))
            .expect("run flash16 Resident chat");
        let legacy_generation = legacy
            .generate_greedy_chat_stream(prompt, 64, &cancelled, |_| Ok(()))
            .expect("run legacy Resident chat");

        assert_eq!(
            fast_generation.generation.logit_digests.len(),
            fast_generation.generation.generated_token_ids.len(),
            "flash16 digest count must cover every generated token for {name}"
        );
        assert_eq!(
            legacy_generation.generation.logit_digests.len(),
            legacy_generation.generation.generated_token_ids.len(),
            "legacy digest count must cover every generated token for {name}"
        );
        assert_eq!(
            fast_generation.generation.logit_digests,
            legacy_generation.generation.logit_digests,
            "flash16 per-token fp32 logit digest parity for {name} prompt {prompt:?}; first divergent token: {:?}",
            first_logit_digest_divergence(
                &fast_generation.generation.logit_digests,
                &legacy_generation.generation.logit_digests
            )
        );
    }

    let mut fast = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        4096,
        Gemma4KvCacheType::Q4_0,
        Gemma4Q4AttentionMode::Flash16,
    )
    .expect("create long-window flash16 executor");
    let mut legacy = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        4096,
        Gemma4KvCacheType::Q4_0,
        Gemma4Q4AttentionMode::LegacyFused,
    )
    .expect("create long-window legacy executor");
    let fast_generation = fast
        .generate_greedy_fixed_benchmark_window_stream(
            &code_prompt,
            256,
            64,
            &cancelled,
            |_| Ok(()),
        )
        .expect("run long-window flash16 Resident decode");
    let legacy_generation = legacy
        .generate_greedy_fixed_benchmark_window_stream(
            &code_prompt,
            256,
            64,
            &cancelled,
            |_| Ok(()),
        )
        .expect("run long-window legacy Resident decode");
    assert_eq!(
        fast_generation.generation.logit_digests,
        legacy_generation.generation.logit_digests,
        "flash16 long-window per-token fp32 logit digest parity; first divergent token: {:?}",
        first_logit_digest_divergence(
            &fast_generation.generation.logit_digests,
            &legacy_generation.generation.logit_digests
        )
    );
}

#[test]
#[ignore = "requires local Metal and the Gemma 4 E2B Q4 GGUF fixture"]
fn resident_canonical_chat_matches_pinned_tokens_and_stays_warm_after_reset() {
    let canonical = canonical();
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let mut executor = Gemma4E2bExecutor::new(&model, 4096).expect("create Resident executor");
    let cancelled = AtomicBool::new(false);
    let mut no_events = |_| Ok(());

    let first = executor
        .generate_greedy_chat_stream(
            canonical["prompt"].as_str().expect("canonical prompt"),
            32,
            &cancelled,
            &mut no_events,
        )
        .expect("run cold Resident canonical chat");
    assert_eq!(
        first.generation.prompt_token_ids,
        ids(&canonical, "prompt_token_ids")
    );
    assert_eq!(
        first.generation.generated_token_ids,
        ids(&canonical, "generated_token_ids")
    );
    assert_eq!(first.finish_reason, Gemma4FinishReason::Eos);
    assert_eq!(
        first.generation.text,
        format!(
            "{}{}<turn|>",
            canonical["prompt"].as_str().unwrap(),
            canonical["visible_text"].as_str().unwrap(),
        )
    );
    assert!(first.metrics.resident_bytes > 0);
    assert!(first.metrics.weight_upload_bytes > 0);
    assert!(first.metrics.readback_bytes <= first.generation.generated_token_ids.len() as u64 * 4);

    executor.reset();
    let second = executor
        .generate_greedy_chat_stream(
            canonical["prompt"].as_str().expect("canonical prompt"),
            32,
            &cancelled,
            &mut no_events,
        )
        .expect("run warm Resident canonical chat");
    assert_eq!(
        second.generation.generated_token_ids,
        ids(&canonical, "generated_token_ids")
    );
    assert_eq!(second.finish_reason, Gemma4FinishReason::Eos);
    assert_eq!(second.metrics.weight_upload_bytes, 0);
    assert_eq!(second.metrics.resident_bytes, first.metrics.resident_bytes);
    assert!(
        second.metrics.readback_bytes <= second.generation.generated_token_ids.len() as u64 * 4
    );
}

#[test]
#[ignore = "requires local Metal and the Gemma 4 E2B Q4 GGUF fixture"]
fn q4_kv_flash16_matches_legacy_resident_attention_across_chat_and_long_decode() {
    let canonical = canonical();
    let canonical_prompt = canonical["prompt"].as_str().expect("canonical prompt");
    let code_prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
        Gemma4ChatRole::User,
        "write an hello world main c++ function",
    )])
    .expect("render C++ chat prompt");
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let cancelled = AtomicBool::new(false);

    for (name, prompt) in [
        ("canonical", canonical_prompt),
        ("long-cpp", code_prompt.as_str()),
    ] {
        let mut fast = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
            &model,
            4096,
            Gemma4KvCacheType::Q4_0,
            Gemma4Q4AttentionMode::Flash16,
        )
        .expect("create flash16 Resident executor");
        let mut legacy = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
            &model,
            4096,
            Gemma4KvCacheType::Q4_0,
            Gemma4Q4AttentionMode::LegacyFused,
        )
        .expect("create legacy Resident executor");

        let fast_generation = fast
            .generate_greedy_chat_stream(prompt, 64, &cancelled, |_| Ok(()))
            .expect("run flash16 Resident chat");
        let legacy_generation = legacy
            .generate_greedy_chat_stream(prompt, 64, &cancelled, |_| Ok(()))
            .expect("run legacy Resident chat");

        let first_divergence = first_generated_token_divergence(
            &fast_generation.generation.generated_token_ids,
            &legacy_generation.generation.generated_token_ids,
        );

        assert_eq!(
            fast_generation.generation.generated_token_ids,
            legacy_generation.generation.generated_token_ids,
            "flash16 token parity for {name} prompt {prompt:?}; first divergent generated token: {first_divergence:?}"
        );
        assert_eq!(
            fast_generation.finish_reason, legacy_generation.finish_reason,
            "flash16 finish parity for {name} prompt {prompt:?}"
        );
        assert_eq!(
            fast_generation.metrics.q4_attention_mode,
            Gemma4Q4AttentionMode::Flash16
        );
        assert_eq!(
            legacy_generation.metrics.q4_attention_mode,
            Gemma4Q4AttentionMode::LegacyFused
        );
        assert_eq!(
            fast_generation.metrics.attention_kernel,
            "attention_decode_gemma4_simd_q4_0_flash16_exact_runtime"
        );
        assert_eq!(
            legacy_generation.metrics.attention_kernel,
            "attention_decode_fused_gemma4_simd_q4_0"
        );
        assert_eq!(
            fast_generation.metrics.resident_bytes,
            legacy_generation.metrics.resident_bytes
        );
        assert_eq!(
            fast_generation.metrics.kv_cache_bytes,
            legacy_generation.metrics.kv_cache_bytes
        );
        assert_eq!(
            fast_generation.metrics.readback_bytes,
            legacy_generation.metrics.readback_bytes
        );
    }

    let mut fast = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        4096,
        Gemma4KvCacheType::Q4_0,
        Gemma4Q4AttentionMode::Flash16,
    )
    .expect("create long-window flash16 executor");
    let mut legacy = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
        &model,
        4096,
        Gemma4KvCacheType::Q4_0,
        Gemma4Q4AttentionMode::LegacyFused,
    )
    .expect("create long-window legacy executor");
    let fast_generation = fast
        .generate_greedy_fixed_benchmark_window_stream(
            &code_prompt,
            256,
            64,
            &cancelled,
            |_| Ok(()),
        )
        .expect("run long-window flash16 Resident decode");
    let legacy_generation = legacy
        .generate_greedy_fixed_benchmark_window_stream(
            &code_prompt,
            256,
            64,
            &cancelled,
            |_| Ok(()),
        )
        .expect("run long-window legacy Resident decode");
    let first_divergence = first_generated_token_divergence(
        &fast_generation.generation.generated_token_ids,
        &legacy_generation.generation.generated_token_ids,
    );
    assert_eq!(
        fast_generation.generation.generated_token_ids,
        legacy_generation.generation.generated_token_ids,
        "flash16 long-window token parity; first divergent generated token: {first_divergence:?}"
    );
    assert_eq!(
        fast_generation.first_eos_position, legacy_generation.first_eos_position,
        "flash16 long-window EOS parity"
    );
    assert_eq!(fast_generation.metrics.decode_command_buffers, 64);
    assert_eq!(legacy_generation.metrics.decode_command_buffers, 64);
}

#[test]
#[ignore = "requires local Metal, Gemma 4 E2B Q4 GGUF, and captured llama.cpp oracle fixtures"]
fn legacy_fused_matches_captured_llama_oracles() {
    let canonical = captured_oracle_fixture("gemma4-e2b-resident-canonical.json");
    let long_cpp = captured_oracle_fixture("gemma4-e2b-resident-long-cpp.json");
    verified_external_oracle(&canonical, "<turn|>");
    verified_external_oracle(&long_cpp, "max_tokens");
    let code_prompt = render_gemma4_chat(&[Gemma4ChatMessage::new(
        Gemma4ChatRole::User,
        "write an hello world main c++ function",
    )])
    .expect("render C++ chat prompt");
    assert_eq!(
        long_cpp["prompt"].as_str(),
        Some(code_prompt.as_str()),
        "the long llama.cpp oracle must use the exact C++ chat template"
    );
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let cancelled = AtomicBool::new(false);

    for (name, oracle) in [("canonical", &canonical), ("long-cpp", &long_cpp)] {
        let mut legacy = Gemma4E2bExecutor::new_with_kv_cache_and_q4_attention_mode(
            &model,
            4096,
            Gemma4KvCacheType::Q4_0,
            Gemma4Q4AttentionMode::LegacyFused,
        )
        .expect("create legacy Resident executor");
        let generation = legacy
            .generate_greedy_chat_stream(
                oracle["prompt"].as_str().expect("oracle prompt"),
                64,
                &cancelled,
                |_| Ok(()),
            )
            .expect("run legacy Resident oracle chat");

        assert_eq!(
            generation.generation.prompt_token_ids,
            ids(oracle, "prompt_token_ids"),
            "LegacyFused prompt tokens must retain the externally verified {name} protocol"
        );
        assert_eq!(
            generation.generation.generated_token_ids,
            ids(oracle, "generated_token_ids"),
            "LegacyFused generated tokens must retain the externally verified llama.cpp stream for {name}"
        );
        assert_eq!(
            generation.finish_reason,
            fixture_finish_reason(oracle),
            "LegacyFused finish reason must retain the externally verified llama.cpp stream for {name}"
        );
        assert_eq!(generation.metrics.kv_cache_type, Gemma4KvCacheType::Q4_0);
        assert_eq!(
            generation.metrics.q4_attention_mode,
            Gemma4Q4AttentionMode::LegacyFused
        );
        assert_eq!(
            generation.metrics.attention_kernel,
            "attention_decode_fused_gemma4_simd_q4_0"
        );
        assert!(generation.metrics.resident_bytes > 0);
    }
}

#[test]
#[ignore = "requires local Metal and the Gemma 4 E2B Q4 GGUF fixture"]
fn packed_kv_cache_modes_are_deterministic_and_reduce_kv_residency() {
    let canonical = canonical();
    let prompt = canonical["prompt"].as_str().expect("canonical prompt");
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let cancelled = AtomicBool::new(false);
    let mut no_events = |_| Ok(());
    let mut f32 = Gemma4E2bExecutor::new(&model, 4096).expect("create F32 executor");
    let f32_bytes = f32.kv_cache_bytes();
    let _ = f32
        .generate_greedy_chat_stream(prompt, 16, &cancelled, &mut no_events)
        .expect("run F32 baseline");

    for cache_type in [Gemma4KvCacheType::Q8_0, Gemma4KvCacheType::Q4_0] {
        let mut executor = Gemma4E2bExecutor::new_with_kv_cache(&model, 4096, cache_type)
            .expect("create packed KV executor");
        assert!(executor.kv_cache_bytes() < f32_bytes);
        let first = executor
            .generate_greedy_chat_stream(prompt, 16, &cancelled, &mut no_events)
            .expect("run packed KV chat");
        assert_eq!(first.metrics.kv_cache_type, cache_type);
        assert_eq!(first.metrics.kv_cache_bytes, executor.kv_cache_bytes());
        assert!(
            first
                .generation
                .generated_token_ids
                .iter()
                .all(|id| *id > 0)
        );
        executor.reset();
        let second = executor
            .generate_greedy_chat_stream(prompt, 16, &cancelled, &mut no_events)
            .expect("repeat packed KV chat");
        assert_eq!(
            second.generation.generated_token_ids,
            first.generation.generated_token_ids
        );
        assert_eq!(second.metrics.weight_upload_bytes, 0);
    }
}

#[test]
#[ignore = "requires local Metal and the Gemma 4 E2B Q4 GGUF fixture"]
fn fixed_benchmark_continues_after_eos_without_changing_canonical_prefix() {
    let canonical = canonical();
    let prompt = canonical["prompt"].as_str().expect("canonical prompt");
    let expected = ids(&canonical, "generated_token_ids");
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let cancelled = AtomicBool::new(false);
    let mut no_events = |_| Ok(());
    let mut executor = Gemma4E2bExecutor::new(&model, 4096).expect("create F32 executor");

    let first = executor
        .generate_greedy_fixed_benchmark_stream(prompt, 16, &cancelled, &mut no_events)
        .expect("run fixed benchmark");
    assert_eq!(first.finish_reason, Gemma4FinishReason::MaxTokens);
    assert_eq!(first.generation.generated_token_ids.len(), 16);
    assert_eq!(
        &first.generation.generated_token_ids[..expected.len()],
        expected
    );
    assert!(
        first
            .first_eos_position
            .is_none_or(|position| (1..=16).contains(&position))
    );

    executor.reset();
    let second = executor
        .generate_greedy_fixed_benchmark_stream(prompt, 16, &cancelled, &mut no_events)
        .expect("repeat fixed benchmark");
    assert_eq!(
        second.generation.generated_token_ids,
        first.generation.generated_token_ids
    );
    assert_eq!(second.first_eos_position, first.first_eos_position);
    assert_eq!(second.metrics.weight_upload_bytes, 0);
}

#[test]
#[ignore = "requires local Metal and the Gemma 4 E2B Q4 GGUF fixture"]
fn fixed_benchmark_window_excludes_warmup_from_decode_measurement() {
    let canonical = canonical();
    let prompt = canonical["prompt"].as_str().expect("canonical prompt");
    let model = Gemma4E2bModel::load_gguf(fixture_path()).expect("load Gemma E2B GGUF");
    let cancelled = AtomicBool::new(false);
    let mut no_events = |_| Ok(());
    let mut executor = Gemma4E2bExecutor::new(&model, 4096).expect("create Resident executor");

    let first = executor
        .generate_greedy_fixed_benchmark_window_stream(prompt, 32, 16, &cancelled, &mut no_events)
        .expect("run fixed benchmark window");
    assert_eq!(first.finish_reason, Gemma4FinishReason::MaxTokens);
    assert_eq!(first.generation.generated_token_ids.len(), 48);
    assert_eq!(first.metrics.decode_command_buffers, 16);
    assert!(
        first
            .first_eos_position
            .is_none_or(|position| (1..=48).contains(&position))
    );

    executor.reset();
    let second = executor
        .generate_greedy_fixed_benchmark_window_stream(prompt, 32, 16, &cancelled, &mut no_events)
        .expect("repeat fixed benchmark window");
    assert_eq!(
        second.generation.generated_token_ids,
        first.generation.generated_token_ids
    );
    assert_eq!(second.metrics.decode_command_buffers, 16);
    assert_eq!(second.metrics.weight_upload_bytes, 0);
}
