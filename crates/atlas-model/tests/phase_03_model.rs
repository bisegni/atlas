use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};

use atlas_model::{Generation, LayerTrace, ModelConfig, validate_generation_golden};

#[test]
fn llama_config_parsing_validates_grouped_query_attention_layout() {
    let path = std::env::temp_dir().join(format!(
        "atlas-phase-03-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(
        &path,
        r#"{
        "model_type":"llama", "vocab_size":32, "hidden_size":16,
        "intermediate_size":64, "num_hidden_layers":2,
        "num_attention_heads":4, "num_key_value_heads":2,
        "rms_norm_eps":0.00001, "rope_theta":10000,
        "bos_token_id":1, "eos_token_id":2, "tie_word_embeddings":false
    }"#,
    )
    .unwrap();
    let config = ModelConfig::from_path(&path).unwrap();
    fs::remove_file(path).unwrap();
    assert_eq!(config.head_dim(), 4);
    assert_eq!(config.num_attention_heads / config.num_key_value_heads, 2);
}

#[test]
fn raw_token_golden_requires_exact_ids_and_tolerant_logits() {
    let path = std::env::temp_dir().join(format!(
        "atlas-phase-03-golden-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::write(
        &path,
        r#"{"generated_token_ids":[9,2],"final_logits":[1.0,-2.0],"logit_abs_tolerance":0.001}"#,
    )
    .unwrap();
    let generation = Generation {
        prompt_token_ids: vec![1],
        generated_token_ids: vec![9, 2],
        text: String::new(),
        trace: LayerTrace::default(),
        final_logits: vec![1.0005, -2.0005],
    };
    validate_generation_golden(&path, &generation).unwrap();
    fs::remove_file(path).unwrap();
}
