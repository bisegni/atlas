use std::{fs, path::Path, sync::atomic::AtomicBool};

use atlas_model::{
    Gemma4E2bModel,
    gemma4_executor::{Gemma4E2bExecutor, Gemma4FinishReason, Gemma4KvCacheType},
};
use serde_json::Value;

fn fixture_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf")
}

fn canonical() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("fixtures/gemma4-e2b-resident-canonical.json");
    serde_json::from_slice(&fs::read(path).expect("read canonical Gemma fixture"))
        .expect("parse canonical Gemma fixture")
}

fn ids(value: &Value, key: &str) -> Vec<u32> {
    value[key]
        .as_array()
        .expect("token ID array")
        .iter()
        .map(|id| id.as_u64().expect("u32 token ID") as u32)
        .collect()
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
