use atlas_model::Gemma4E2bModel;

#[test]
fn gemma4_embedded_tokenizer_round_trips_when_fixture_is_available() {
    let Ok(path) = std::env::var("ATLAS_GEMMA4_GGUF") else {
        return;
    };
    let model = Gemma4E2bModel::load_gguf(path).expect("load Gemma 4 E2B GGUF fixture");
    let config = &model.config;
    assert_eq!(config.vocab_size, 262_144);
    assert_eq!(config.layers, 35);
    assert_eq!(config.hidden_size, 1536);
    assert_eq!(config.per_layer_embedding_size, 256);
    assert_eq!(model.gguf().metadata["general.architecture"], "gemma4");
    let encoded = model.tokenize("Atlas runs Gemma 4.").expect("tokenize");
    assert!(!encoded.is_empty());
    let decoded = model.decode(&encoded).expect("decode token IDs");
    assert!(decoded.contains("Atlas") && decoded.contains("Gemma"));
}
