use atlas_model::{gemma4_combine_ple, gemma4_shared_kv_sources, gemma4_softcap_logits};

#[test]
fn ple_combines_identity_and_context_with_sqrt_two_scaling() {
    let combined = gemma4_combine_ple(&[2.0, -4.0], &[4.0, 2.0]).expect("combine PLE");
    let expected = [6.0f32 / 2.0f32.sqrt(), -2.0f32 / 2.0f32.sqrt()];
    for (actual, expected) in combined.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn logit_softcap_bounds_large_values_without_flattening_small_values() {
    let mut logits = [-1_000.0, -1.0, 0.0, 1.0, 1_000.0];
    gemma4_softcap_logits(&mut logits, 30.0).expect("soft cap logits");
    assert!(logits.iter().all(|value| value.abs() <= 30.0));
    assert!(logits.windows(2).all(|pair| pair[0] < pair[1]));
    assert!((logits[2] - 0.0).abs() < f32::EPSILON);
}

#[test]
fn shared_kv_reuses_only_the_matching_attention_kind() {
    let sources = gemma4_shared_kv_sources(
        &[true, true, false, false, true],
        &[true, false, true, false, false],
    )
    .expect("resolve shared KV sources");
    assert_eq!(sources, vec![0, 0, 2, 2, 0]);
}
