use atlas_model::kv_cache::{
    ContiguousKvCache, KvCacheConfig, KvView, LayerKv, PagedKvCache, SessionId,
};

fn config() -> KvCacheConfig {
    KvCacheConfig {
        layers: 2,
        kv_heads: 2,
        head_dim: 2,
        capacity: 5,
        sliding_window: Some(3),
        sink_tokens: 1,
    }
}
fn token(position: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let keys = (0..2)
        .map(|layer| {
            (0..4)
                .map(|i| (position * 100 + layer * 10 + i) as f32)
                .collect()
        })
        .collect();
    let values = (0..2)
        .map(|layer| {
            (0..4)
                .map(|i| -(position as f32 * 100.0 + layer as f32 * 10.0 + i as f32))
                .collect()
        })
        .collect();
    (keys, values)
}
fn append_both(contiguous: &mut ContiguousKvCache, paged: &mut PagedKvCache, position: usize) {
    let (keys, values) = token(position);
    let layers: Vec<_> = (0..2)
        .map(|layer| LayerKv {
            keys: &keys[layer],
            values: &values[layer],
        })
        .collect();
    contiguous.append(position, &layers).unwrap();
    paged.append(position, &layers).unwrap();
}

fn decode_attention(query: &[f32], view: &KvView, head: usize, head_dim: usize) -> Vec<f32> {
    let count = view.positions.len();
    let keys = &view.keys[head * count * head_dim..(head + 1) * count * head_dim];
    let values = &view.values[head * count * head_dim..(head + 1) * count * head_dim];
    let scores: Vec<_> = (0..count)
        .map(|position| {
            (0..head_dim)
                .map(|i| query[i] * keys[position * head_dim + i])
                .sum::<f32>()
        })
        .collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let normalizer: f32 = scores.iter().map(|score| (score - max).exp()).sum();
    (0..head_dim)
        .map(|i| {
            scores
                .iter()
                .enumerate()
                .map(|(position, score)| {
                    (score - max).exp() / normalizer * values[position * head_dim + i]
                })
                .sum()
        })
        .collect()
}

#[test]
fn phase_04_kv_cache_proves_parity_bounded_bytes_positions_and_session_isolation() {
    let mut contiguous = ContiguousKvCache::new(SessionId(7), config()).unwrap();
    let mut paged = PagedKvCache::new(SessionId(7), config(), 2).unwrap();
    for position in 0..5 {
        append_both(&mut contiguous, &mut paged, position);
    }
    // Sink position 0 plus the three newest positions remain, independent of
    // recycled physical slots/pages.
    assert_eq!(contiguous.positions().collect::<Vec<_>>(), vec![0, 2, 3, 4]);
    let contiguous_view = contiguous.view(1).unwrap();
    let paged_view = paged.view(1).unwrap();
    assert_eq!(
        contiguous_view, paged_view,
        "paged cache must match contiguous cache before/after eviction"
    );
    assert_eq!(contiguous_view.positions, vec![0, 2, 3, 4]);
    assert_eq!(&contiguous_view.keys[0..2], &[10.0, 11.0]);
    assert_eq!(&contiguous_view.keys[2..4], &[210.0, 211.0]);
    let contiguous_metrics = contiguous.metrics();
    let paged_metrics = paged.metrics();
    assert_eq!(contiguous_metrics.allocated_bytes, 2 * 2 * 5 * 2 * 2 * 4);
    assert!(contiguous_metrics.eviction_count >= 1 && paged_metrics.eviction_count >= 1);
    assert!(contiguous_metrics.fragmentation >= 0.0 && paged_metrics.fragmentation >= 0.0);

    let mut other = ContiguousKvCache::new(SessionId(8), config()).unwrap();
    append_both(
        &mut other,
        &mut PagedKvCache::new(SessionId(99), config(), 2).unwrap(),
        0,
    );
    assert_eq!(other.view(1).unwrap().positions, vec![0]);
    assert_eq!(contiguous.view(1).unwrap().positions, vec![0, 2, 3, 4]);
}

#[test]
fn cached_decode_matches_full_context_before_the_window_boundary() {
    let mut cache = ContiguousKvCache::new(SessionId(12), config()).unwrap();
    let mut full_keys = Vec::new();
    let mut full_values = Vec::new();
    for position in 0..4 {
        let (keys, values) = token(position);
        let layers: Vec<_> = (0..2)
            .map(|layer| LayerKv {
                keys: &keys[layer],
                values: &values[layer],
            })
            .collect();
        cache.append(position, &layers).unwrap();
        full_keys.push(keys[0].clone());
        full_values.push(values[0].clone());
        let full = KvView {
            positions: (0..=position).collect(),
            keys: (0..2)
                .flat_map(|head| {
                    full_keys
                        .iter()
                        .flat_map(move |token| token[head * 2..head * 2 + 2].iter().copied())
                })
                .collect(),
            values: (0..2)
                .flat_map(|head| {
                    full_values
                        .iter()
                        .flat_map(move |token| token[head * 2..head * 2 + 2].iter().copied())
                })
                .collect(),
        };
        let query = [0.25, -0.5];
        assert_eq!(
            decode_attention(&query, &cache.view(0).unwrap(), 0, 2),
            decode_attention(&query, &full, 0, 2)
        );
    }
}

#[test]
fn sliding_window_stays_bounded_for_1024_tokens() {
    let mut contiguous = ContiguousKvCache::new(SessionId(11), config()).unwrap();
    let mut paged = PagedKvCache::new(SessionId(11), config(), 2).unwrap();
    for position in 0..1_024 {
        append_both(&mut contiguous, &mut paged, position);
    }
    assert_eq!(
        contiguous.positions().collect::<Vec<_>>(),
        vec![0, 1021, 1022, 1023]
    );
    assert_eq!(contiguous.view(0).unwrap(), paged.view(0).unwrap());
    assert!(contiguous.metrics().allocated_bytes <= 2 * 2 * 5 * 2 * 2 * 4);
    assert!(paged.metrics().allocated_bytes <= 3 * 2 * 2 * 2 * 2 * 2 * 4);
}
