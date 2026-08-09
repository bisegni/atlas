use atlas_model::Gemma4E2bModel;

#[test]
fn dump_config() {
    let path = "/Users/bisegni/dev/github/bisegni/atlas/models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf";
    let config = Gemma4E2bModel::load_gguf_without_quantization_preflight(path).unwrap();
    eprintln!(
        "attention_heads={} key_length={} key_length_swa={} kv_heads={} hidden={} layers={}",
        config.config.attention_heads,
        config.config.key_length,
        config.config.key_length_swa,
        config.config.key_value_heads.len(),
        config.config.hidden_size,
        config.config.layers
    );
}
