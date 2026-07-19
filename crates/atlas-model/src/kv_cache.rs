//! Session-local key/value cache storage for decoder attention.
//!
//! Both implementations expose tokens in increasing *absolute* position
//! order.  Physical storage may be recycled, but physical indices are never
//! used as RoPE or attention positions.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Number of physical token slots.  This is a hard memory bound.
    pub capacity: usize,
    /// Number of most-recent non-sink tokens retained. `None` disables eviction.
    pub sliding_window: Option<usize>,
    /// Initial absolute positions that survive sliding-window eviction.
    pub sink_tokens: usize,
}

impl KvCacheConfig {
    pub fn validate(self) -> Result<Self> {
        ensure!(self.layers > 0, "KV cache needs at least one layer");
        ensure!(self.kv_heads > 0, "KV cache needs at least one KV head");
        ensure!(
            self.head_dim > 0,
            "KV cache needs a non-zero head dimension"
        );
        ensure!(self.capacity > 0, "KV cache needs non-zero capacity");
        if let Some(window) = self.sliding_window {
            ensure!(window > 0, "sliding window must be non-zero");
            ensure!(
                self.capacity >= window.saturating_add(self.sink_tokens),
                "capacity must hold the sliding window and sink tokens"
            );
        }
        Ok(self)
    }
    fn token_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
    fn layer_bytes(self) -> usize {
        2 * self.capacity * self.token_width() * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerKv<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvView {
    pub positions: Vec<usize>,
    /// `[head][position][dimension]`, where `position` follows `positions`.
    pub keys: Vec<f32>,
    /// `[head][position][dimension]`, where `position` follows `positions`.
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvCacheMetrics {
    pub resident_tokens: usize,
    pub allocated_bytes: usize,
    pub bytes_per_token: usize,
    pub fragmentation: f32,
    pub eviction_count: usize,
    pub eviction_time: Duration,
}

impl KvCacheMetrics {
    fn from_parts(
        config: KvCacheConfig,
        resident_tokens: usize,
        free_slots: usize,
        eviction_count: usize,
        eviction_time: Duration,
    ) -> Self {
        let allocated_bytes = config.layers * config.layer_bytes();
        Self {
            resident_tokens,
            allocated_bytes,
            bytes_per_token: if resident_tokens == 0 {
                0
            } else {
                allocated_bytes / resident_tokens
            },
            fragmentation: free_slots as f32 / config.capacity as f32,
            eviction_count,
            eviction_time,
        }
    }
}

/// A fixed, contiguous allocation with a free-slot list.  The backing layout is
/// `[layer][K|V][head][slot][dimension]`; slot metadata maps that layout to an
/// absolute token position.
#[derive(Debug)]
pub struct ContiguousKvCache {
    config: KvCacheConfig,
    session: SessionId,
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: VecDeque<usize>,
    slots: HashMap<usize, usize>,
    free_slots: Vec<usize>,
    next_position: usize,
    eviction_count: usize,
    eviction_time: Duration,
}

impl ContiguousKvCache {
    pub fn new(session: SessionId, config: KvCacheConfig) -> Result<Self> {
        let config = config.validate()?;
        let elements = config.layers * config.capacity * config.token_width();
        Ok(Self {
            config,
            session,
            keys: vec![0.0; elements],
            values: vec![0.0; elements],
            positions: VecDeque::new(),
            slots: HashMap::new(),
            free_slots: (0..config.capacity).rev().collect(),
            next_position: 0,
            eviction_count: 0,
            eviction_time: Duration::ZERO,
        })
    }
    pub fn session(&self) -> SessionId {
        self.session
    }
    pub fn next_position(&self) -> usize {
        self.next_position
    }
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    pub fn positions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }
    pub fn append(&mut self, position: usize, layers: &[LayerKv<'_>]) -> Result<()> {
        ensure!(
            position == self.next_position,
            "expected position {}, got {position}",
            self.next_position
        );
        ensure!(
            layers.len() == self.config.layers,
            "expected {} layers, got {}",
            self.config.layers,
            layers.len()
        );
        let width = self.config.token_width();
        for layer in layers {
            ensure!(
                layer.keys.len() == width && layer.values.len() == width,
                "KV token width must be {width}"
            );
        }
        self.evict_for_append()?;
        let slot = self
            .free_slots
            .pop()
            .ok_or_else(|| anyhow::anyhow!("KV cache capacity exhausted"))?;
        for (layer, token) in layers.iter().enumerate() {
            let base = self.offset(layer, slot);
            self.keys[base..base + width].copy_from_slice(token.keys);
            self.values[base..base + width].copy_from_slice(token.values);
        }
        self.positions.push_back(position);
        self.slots.insert(position, slot);
        self.next_position += 1;
        Ok(())
    }
    pub fn view(&self, layer: usize) -> Result<KvView> {
        ensure!(layer < self.config.layers, "layer {layer} is out of range");
        let count = self.len();
        let width = self.config.token_width();
        let mut keys = vec![0.0; width * count];
        let mut values = vec![0.0; width * count];
        // Convert physical `[head][slot][dim]` to logical `[head][position][dim]`.
        for head in 0..self.config.kv_heads {
            for (logical, position) in self.positions.iter().enumerate() {
                let slot = self.slots[position];
                let src = self.offset(layer, slot) + head * self.config.head_dim;
                let dst = (head * count + logical) * self.config.head_dim;
                keys[dst..dst + self.config.head_dim]
                    .copy_from_slice(&self.keys[src..src + self.config.head_dim]);
                values[dst..dst + self.config.head_dim]
                    .copy_from_slice(&self.values[src..src + self.config.head_dim]);
            }
        }
        Ok(KvView {
            positions: self.positions().collect(),
            keys,
            values,
        })
    }
    pub fn reset(&mut self) {
        self.positions.clear();
        self.slots.clear();
        self.free_slots = (0..self.config.capacity).rev().collect();
        self.next_position = 0;
        self.eviction_count = 0;
        self.eviction_time = Duration::ZERO;
    }
    pub fn metrics(&self) -> KvCacheMetrics {
        KvCacheMetrics::from_parts(
            self.config,
            self.len(),
            self.free_slots.len(),
            self.eviction_count,
            self.eviction_time,
        )
    }
    fn offset(&self, layer: usize, slot: usize) -> usize {
        (layer * self.config.capacity + slot) * self.config.token_width()
    }
    fn evict_for_append(&mut self) -> Result<()> {
        let needs_window_evict = self
            .config
            .sliding_window
            .is_some_and(|window| self.len().saturating_sub(self.config.sink_tokens) >= window);
        if !needs_window_evict && !self.free_slots.is_empty() {
            return Ok(());
        }
        ensure!(
            self.config.sliding_window.is_some(),
            "KV cache capacity exhausted; configure a sliding window to permit eviction"
        );
        let started = Instant::now();
        let candidate = self
            .positions
            .iter()
            .copied()
            .find(|position| *position >= self.config.sink_tokens)
            .ok_or_else(|| anyhow::anyhow!("KV cache is full of sink tokens"))?;
        self.positions.retain(|position| *position != candidate);
        let slot = self
            .slots
            .remove(&candidate)
            .expect("position metadata is complete");
        self.free_slots.push(slot);
        self.eviction_count += 1;
        self.eviction_time += started.elapsed();
        Ok(())
    }
}

/// Page metadata intentionally remains visible for diagnostics and future GPU
/// residency management.  A page contains KV for one layer and consecutive
/// absolute positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPageMetadata {
    pub page_id: usize,
    pub session: SessionId,
    pub layer: usize,
    pub start_position: usize,
    pub used_tokens: usize,
    pub capacity: usize,
    pub reference_count: usize,
}

#[derive(Debug)]
struct Page {
    metadata: KvPageMetadata,
    keys: Vec<f32>,
    values: Vec<f32>,
    free_offsets: Vec<usize>,
}

/// Paged storage with an explicit free-page list.  It has the same public token
/// semantics as [`ContiguousKvCache`], while page reclamation exposes real
/// fragmentation in the metrics.
#[derive(Debug)]
pub struct PagedKvCache {
    config: KvCacheConfig,
    session: SessionId,
    page_tokens: usize,
    pages: Vec<Option<Page>>,
    free_pages: Vec<usize>,
    positions: VecDeque<usize>,
    locations: BTreeMap<usize, (usize, usize)>,
    next_position: usize,
    eviction_count: usize,
    eviction_time: Duration,
}

impl PagedKvCache {
    pub fn new(session: SessionId, config: KvCacheConfig, page_tokens: usize) -> Result<Self> {
        let config = config.validate()?;
        ensure!(page_tokens > 0, "page_tokens must be non-zero");
        let page_count = config.capacity.div_ceil(page_tokens);
        Ok(Self {
            config,
            session,
            page_tokens,
            pages: (0..page_count).map(|_| None).collect(),
            free_pages: (0..page_count).rev().collect(),
            positions: VecDeque::new(),
            locations: BTreeMap::new(),
            next_position: 0,
            eviction_count: 0,
            eviction_time: Duration::ZERO,
        })
    }
    pub fn append(&mut self, position: usize, layers: &[LayerKv<'_>]) -> Result<()> {
        ensure!(
            position == self.next_position,
            "expected position {}, got {position}",
            self.next_position
        );
        ensure!(
            layers.len() == self.config.layers,
            "expected {} layers, got {}",
            self.config.layers,
            layers.len()
        );
        let width = self.config.token_width();
        for token in layers {
            ensure!(
                token.keys.len() == width && token.values.len() == width,
                "KV token width must be {width}"
            );
        }
        self.evict_for_append()?;
        let (page_id, offset) = self.writable_page(position)?;
        let page = self.pages[page_id].as_mut().expect("writable page exists");
        for (layer, token) in layers.iter().enumerate() {
            let base = (layer * self.page_tokens + offset) * width;
            page.keys[base..base + width].copy_from_slice(token.keys);
            page.values[base..base + width].copy_from_slice(token.values);
        }
        page.metadata.used_tokens += 1;
        page.metadata.reference_count += 1;
        self.positions.push_back(position);
        self.locations.insert(position, (page_id, offset));
        self.next_position += 1;
        Ok(())
    }
    pub fn session(&self) -> SessionId {
        self.session
    }
    pub fn next_position(&self) -> usize {
        self.next_position
    }
    pub fn len(&self) -> usize {
        self.positions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
    pub fn positions(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }
    pub fn view(&self, layer: usize) -> Result<KvView> {
        ensure!(layer < self.config.layers, "layer {layer} is out of range");
        let count = self.positions.len();
        let width = self.config.token_width();
        let mut keys = vec![0.0; count * width];
        let mut values = vec![0.0; count * width];
        for head in 0..self.config.kv_heads {
            for (logical, position) in self.positions.iter().enumerate() {
                let (page_id, offset) = self.locations[position];
                let page = self.pages[page_id]
                    .as_ref()
                    .expect("live location has a page");
                let src = (layer * self.page_tokens + offset) * width + head * self.config.head_dim;
                let dst = (head * count + logical) * self.config.head_dim;
                keys[dst..dst + self.config.head_dim]
                    .copy_from_slice(&page.keys[src..src + self.config.head_dim]);
                values[dst..dst + self.config.head_dim]
                    .copy_from_slice(&page.values[src..src + self.config.head_dim]);
            }
        }
        Ok(KvView {
            positions: self.positions.iter().copied().collect(),
            keys,
            values,
        })
    }
    pub fn pages(&self) -> impl Iterator<Item = &KvPageMetadata> {
        self.pages
            .iter()
            .filter_map(|page| page.as_ref().map(|page| &page.metadata))
    }
    pub fn reset(&mut self) {
        self.pages.iter_mut().for_each(|page| *page = None);
        self.free_pages = (0..self.pages.len()).rev().collect();
        self.positions.clear();
        self.locations.clear();
        self.next_position = 0;
        self.eviction_count = 0;
        self.eviction_time = Duration::ZERO;
    }
    pub fn metrics(&self) -> KvCacheMetrics {
        let allocated_bytes = self.pages.len()
            * self.config.layers
            * 2
            * self.page_tokens
            * self.config.token_width()
            * std::mem::size_of::<f32>();
        KvCacheMetrics {
            resident_tokens: self.positions.len(),
            allocated_bytes,
            bytes_per_token: if self.positions.is_empty() {
                0
            } else {
                allocated_bytes / self.positions.len()
            },
            fragmentation: self.free_pages.len() as f32 / self.pages.len() as f32,
            eviction_count: self.eviction_count,
            eviction_time: self.eviction_time,
        }
    }
    fn writable_page(&mut self, position: usize) -> Result<(usize, usize)> {
        if let Some((page_id, offset)) =
            self.pages
                .iter_mut()
                .enumerate()
                .rev()
                .find_map(|(id, page)| {
                    page.as_mut()
                        .and_then(|page| page.free_offsets.pop().map(|offset| (id, offset)))
                })
        {
            return Ok((page_id, offset));
        }
        let page_id = self
            .free_pages
            .pop()
            .ok_or_else(|| anyhow::anyhow!("KV page capacity exhausted"))?;
        let elements = self.config.layers * self.page_tokens * self.config.token_width();
        self.pages[page_id] = Some(Page {
            metadata: KvPageMetadata {
                page_id,
                session: self.session,
                layer: usize::MAX,
                start_position: position,
                used_tokens: 0,
                capacity: self.page_tokens,
                reference_count: 0,
            },
            keys: vec![0.0; elements],
            values: vec![0.0; elements],
            free_offsets: (0..self.page_tokens).rev().collect(),
        });
        let offset = self.pages[page_id]
            .as_mut()
            .expect("new page exists")
            .free_offsets
            .pop()
            .expect("new page has capacity");
        Ok((page_id, offset))
    }
    fn evict_for_append(&mut self) -> Result<()> {
        let needs_window_evict = self.config.sliding_window.is_some_and(|window| {
            self.positions.len().saturating_sub(self.config.sink_tokens) >= window
        });
        let estimated_used = self
            .pages
            .iter()
            .filter_map(|page| page.as_ref())
            .map(|page| page.metadata.used_tokens)
            .sum::<usize>();
        if !needs_window_evict && estimated_used < self.config.capacity {
            return Ok(());
        }
        ensure!(
            self.config.sliding_window.is_some(),
            "KV cache capacity exhausted; configure a sliding window to permit eviction"
        );
        let started = Instant::now();
        let candidate = self
            .positions
            .iter()
            .copied()
            .find(|position| *position >= self.config.sink_tokens)
            .ok_or_else(|| anyhow::anyhow!("KV cache is full of sink tokens"))?;
        self.positions.retain(|position| *position != candidate);
        let (page_id, offset) = self
            .locations
            .remove(&candidate)
            .expect("position metadata is complete");
        let page = self.pages[page_id]
            .as_mut()
            .expect("live location has a page");
        page.metadata.used_tokens -= 1;
        page.metadata.reference_count -= 1;
        page.free_offsets.push(offset);
        if page.metadata.reference_count == 0 {
            self.pages[page_id] = None;
            self.free_pages.push(page_id);
        } else if page.metadata.start_position == candidate {
            self.pages[page_id]
                .as_mut()
                .expect("page remains live")
                .metadata
                .start_position = self
                .locations
                .iter()
                .filter_map(|(position, (id, _))| (*id == page_id).then_some(*position))
                .min()
                .expect("live page has a location");
        }
        self.eviction_count += 1;
        self.eviction_time += started.elapsed();
        Ok(())
    }
}
