//! Optional, engine-independent profiling and bottleneck reporting.
//!
//! The recorder is intentionally usable by Metal, model executors, and future
//! backends without importing any runtime-specific types.

use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt::Write as _, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileScope {
    ModelLoad,
    WeightPreparation,
    ResidentInitialization,
    Prefill,
    DecodeWarmup,
    DecodeMeasured,
    DecodeComplete,
    TokenSelection,
    ProfilerOverhead,
    Other,
}

impl Default for ProfileScope {
    fn default() -> Self {
        Self::Other
    }
}

impl ProfileScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ModelLoad => "model_load",
            Self::WeightPreparation => "weight_preparation",
            Self::ResidentInitialization => "resident_initialization",
            Self::Prefill => "prefill",
            Self::DecodeWarmup => "decode_warmup",
            Self::DecodeMeasured => "decode_measured",
            Self::DecodeComplete => "decode_complete",
            Self::TokenSelection => "token_selection",
            Self::ProfilerOverhead => "profiler_overhead",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockDomain {
    HostMonotonic,
    MetalGpu,
    CpuProcess,
    Unavailable,
}

impl Default for ClockDomain {
    fn default() -> Self {
        Self::Unavailable
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingKind {
    HostWall,
    GpuElapsed,
    CpuEncode,
    CpuWait,
    ReadbackWait,
    UploadWait,
    ProfilerOverhead,
}

impl Default for TimingKind {
    fn default() -> Self {
        Self::HostWall
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingBoundary {
    pub clock_domain: ClockDomain,
    pub timing_kind: TimingKind,
    pub start_ns: Option<u64>,
    pub end_ns: Option<u64>,
    pub intervals_may_overlap: bool,
    pub status: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeContract {
    pub hotspot_scope: ProfileScope,
    pub includes_warmup: bool,
    pub token_selection_included: bool,
    pub readback_included: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecodeScope {
    pub warmup_decode_tokens_requested: u64,
    pub warmup_decode_tokens_completed: u64,
    pub measured_decode_tokens_requested: u64,
    pub measured_decode_tokens_completed: u64,
    pub completed_decode_tokens_total: u64,
    pub hotspot_scope: ProfileScope,
    pub physical_command_buffer_overlap: bool,
    pub physical_command_buffer_overlap_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasuredWindow {
    pub scope: ProfileScope,
    pub host_start_ns: Option<u64>,
    pub host_end_ns: Option<u64>,
    pub wall_time_ms: Option<f64>,
    pub tokens: u64,
    pub token_selection_included: bool,
    pub readback_included: bool,
    pub timing: TimingBoundary,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkCompatibility {
    pub scope_matches_normal_benchmark: bool,
    pub token_window_matches: bool,
    pub executor_matches: bool,
    pub kernel_plan_matches: bool,
    pub token_sha_matches: bool,
    pub eos_matches: bool,
    pub prompt_token_sha256: Option<String>,
    pub generated_token_sha256: Option<String>,
    pub measured_generated_token_sha256: Option<String>,
    pub first_eos_position: Option<u64>,
    pub executor: Option<String>,
    pub kv_cache_type: Option<String>,
    pub quantization_plan: Option<String>,
    pub selected_kernels: BTreeMap<String, String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfilerOverhead {
    pub record_collection_cpu_ms: f64,
    pub aggregation_cpu_ms: f64,
    pub json_serialization_ms: f64,
    pub markdown_generation_ms: f64,
    pub callback_overhead_estimate_ms: f64,
    pub callback_overhead_status: String,
    pub total_profiler_overhead_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileMode {
    Disabled,
    Benchmark,
    Diagnostic,
}

impl Default for ProfileMode {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum ProfilePhase {
    ModelLoad,
    WeightPreparation,
    ResidentInitialization,
    Prefill,
    Decode,
    TokenSelection,
    HostSynchronization,
}

impl Default for ProfilePhase {
    fn default() -> Self {
        Self::ModelLoad
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum OperationFamily {
    Embedding,
    RmsNorm,
    QkvProjection,
    AttentionScore,
    AttentionValue,
    KvAppend,
    AttentionOutputProjection,
    FfnGateUp,
    FfnActivationMultiply,
    Residual,
    LogitSoftcap,
    FfnDown,
    FinalNorm,
    PleProjection,
    OutputProjection,
    ArgmaxOrTokenSelection,
    Copy,
    Conversion,
    Synchronization,
    Other,
}

/// Attention-specific dimensions are kept out of generic operation names so
/// profiles can distinguish the global and sliding paths and each scan pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    Global,
    Sliding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionScanPass {
    Scan,
    Combine,
    SinglePass,
}

impl Default for OperationFamily {
    fn default() -> Self {
        Self::Other
    }
}

impl OperationFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::RmsNorm => "rms_norm",
            Self::QkvProjection => "qkv_projection",
            Self::AttentionScore => "attention_score",
            Self::AttentionValue => "attention_value",
            Self::KvAppend => "kv_append",
            Self::AttentionOutputProjection => "attention_output_projection",
            Self::FfnGateUp => "ffn_gate_up",
            Self::FfnActivationMultiply => "ffn_activation_multiply",
            Self::Residual => "residual",
            Self::LogitSoftcap => "logit_softcap",
            Self::FfnDown => "ffn_down",
            Self::FinalNorm => "final_norm",
            Self::PleProjection => "ple_projection",
            Self::OutputProjection => "output_projection",
            Self::ArgmaxOrTokenSelection => "argmax_or_token_selection",
            Self::Copy => "copy",
            Self::Conversion => "conversion",
            Self::Synchronization => "synchronization",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileEvent {
    pub run_id: u64,
    pub repetition: u32,
    pub phase: ProfilePhase,
    pub token_position: Option<u32>,
    pub layer_index: Option<u32>,
    pub attention_kind: Option<AttentionKind>,
    pub attention_scan_pass: Option<AttentionScanPass>,
    pub operation_family: OperationFamily,
    pub scope: ProfileScope,
    pub kernel_name: Option<String>,
    pub host_encode_ns: u64,
    pub host_wall_ns: u64,
    pub gpu_ns: Option<u64>,
    pub cpu_wait_ns: u64,
    pub command_buffer_id: Option<u64>,
    pub dispatch_id: Option<u64>,
    pub dispatch_calls: u64,
    pub threadgroups: u64,
    pub threads: u64,
    pub bytes_read_estimate: u64,
    pub bytes_written_estimate: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub upload_time_ns: u64,
    pub readback_time_ns: u64,
    pub timing: TimingBoundary,
    pub timing_boundaries: Vec<TimingBoundary>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileCounters {
    pub host_wall_ns: u64,
    pub gpu_ns: u64,
    pub production_gpu_elapsed_ns: u64,
    pub attributed_gpu_duration_ns: u64,
    pub gpu_duration_source: String,
    pub categorized_gpu_ns: u64,
    pub cpu_wait_ns: u64,
    pub host_encode_ns: u64,
    pub command_buffers: u64,
    pub dispatches: u64,
    pub threadgroups: u64,
    pub threadgroups_dispatched: u64,
    pub threads_dispatched: u64,
    pub timed_dispatches: u64,
    pub categorized_dispatches: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub allocations: u64,
    pub resizes: u64,
    pub pipeline_creations: u64,
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub kv_cache_bytes: u64,
    pub upload_time_ns: u64,
    pub readback_time_ns: u64,
    pub memory_operation_time_ns: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhaseSummary {
    pub phase: ProfilePhase,
    pub scope: ProfileScope,
    pub wall_ns: u64,
    pub gpu_ns: u64,
    pub attributed_gpu_ns: u64,
    pub cpu_wait_ns: u64,
    pub command_buffers: u64,
    pub dispatch_calls: u64,
    pub threadgroups_dispatched: u64,
    pub threads_dispatched: u64,
    pub timed_dispatches: u64,
    pub untimed_dispatches: u64,
    pub categorized_dispatches: u64,
    pub uncategorized_dispatches: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub tokens: u64,
    pub tokens_per_second: f64,
    pub categorized_gpu_ns: u64,
    pub uncategorized_gpu_ns: u64,
    pub host_encode_ns: u64,
    pub upload_time_ns: u64,
    pub readback_time_ns: u64,
    pub unexplained_ns: u64,
    pub allocations: u64,
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub kv_cache_bytes: u64,
    pub command_buffer_idle_gap_ns: Option<u64>,
    pub command_buffer_schedule_ns: Option<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageSummary {
    pub gpu_attribution: f64,
    pub dispatch_attribution: f64,
    pub operation_attribution: f64,
    pub kernel_attribution: f64,
    pub synchronization_attribution: f64,
    pub complete: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SynchronizationSummary {
    pub cpu_wait_ns: u64,
    pub command_buffer_idle_gap_ns: Option<u64>,
    pub command_buffer_schedule_ns: Option<u64>,
    pub gpu_waiting_for_cpu_ns: Option<u64>,
    pub readback_wait_ns: Option<u64>,
    pub upload_wait_ns: Option<u64>,
    pub command_buffer_count: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemorySummary {
    pub resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub kv_cache_bytes: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
    pub bytes_per_prompt_token: Option<f64>,
    pub bytes_per_generated_token: Option<f64>,
    pub effective_bandwidth_bytes_per_second: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DispatchReconciliation {
    pub dispatch_calls: u64,
    pub threadgroups_dispatched: u64,
    pub threads_dispatched: u64,
    pub timed_dispatch_calls: u64,
    pub untimed_dispatch_calls: u64,
    pub categorized_dispatch_calls: u64,
    pub uncategorized_dispatch_calls: u64,
    pub total_gpu_ns: u64,
    pub attributed_gpu_duration_ns: u64,
    pub categorized_gpu_ns: u64,
    pub uncategorized_gpu_ns: u64,
    pub dispatch_coverage: f64,
    pub gpu_timing_coverage: f64,
    pub complete: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileAggregate {
    pub key: String,
    pub phase: Option<ProfilePhase>,
    pub layer_index: Option<u32>,
    pub attention_kind: Option<AttentionKind>,
    pub attention_scan_pass: Option<AttentionScanPass>,
    pub operation_family: Option<OperationFamily>,
    pub scope: Option<ProfileScope>,
    pub kernel_name: Option<String>,
    pub events: u64,
    pub dispatches: u64,
    pub threadgroups: u64,
    pub host_encode_ns: u64,
    pub host_wall_ns: u64,
    pub gpu_ns: u64,
    pub cpu_wait_ns: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub upload_bytes: u64,
    pub readback_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileWorkload {
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prefill_ns: u64,
    pub decode_ns: u64,
    pub ttft_ns: u64,
    pub warmup_decode_tokens: u64,
    pub measured_decode_tokens: u64,
    pub completed_decode_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    pub schema_version: u32,
    pub engine: String,
    pub mode: ProfileMode,
    pub profile_status: String,
    pub scope_contract: ScopeContract,
    pub decode_scope: DecodeScope,
    pub measured_windows: BTreeMap<ProfileScope, MeasuredWindow>,
    pub benchmark_compatibility: BenchmarkCompatibility,
    pub profiler_overhead: ProfilerOverhead,
    /// Optional diagnostic-only Metal counter capability/capture record.
    pub gpu_counter_capture: Option<serde_json::Value>,
    pub workload: ProfileWorkload,
    pub counters: ProfileCounters,
    pub reconciliation: DispatchReconciliation,
    pub phase_reconciliation: Vec<DispatchReconciliation>,
    pub coverage: CoverageSummary,
    pub synchronization: SynchronizationSummary,
    pub memory: MemorySummary,
    pub phase_summaries: Vec<PhaseSummary>,
    pub scope_counters: BTreeMap<ProfileScope, ProfileCounters>,
    pub phases: Vec<ProfileAggregate>,
    pub operation_families: Vec<ProfileAggregate>,
    pub layers: Vec<ProfileAggregate>,
    pub kernels: Vec<ProfileAggregate>,
    pub attention_dispatches: Vec<ProfileAggregate>,
    pub recommendations: Vec<Recommendation>,
    pub hotspots: HotspotReport,
    pub warnings: Vec<String>,
    pub events: Vec<ProfileEvent>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HotspotReport {
    pub prefill: Vec<Recommendation>,
    pub decode_measured: Vec<Recommendation>,
    pub decode_warmup: Vec<Recommendation>,
    pub host_synchronization: Vec<Recommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub rank: usize,
    pub target: String,
    pub classification: String,
    pub scope: ProfileScope,
    pub priority_score: f64,
    pub phase_time_share: f64,
    pub evidence: Vec<String>,
    pub absolute_ms: f64,
    pub categorized_gpu_share: f64,
    pub wall_time_share: Option<f64>,
    pub wall_time_share_status: String,
    pub dispatches_per_measured_token: f64,
    pub gpu_ms_per_measured_token: f64,
    pub dispatch_calls: u64,
    pub threadgroups: u64,
    pub confidence: String,
}

#[derive(Debug, Default)]
pub struct Profiler {
    mode: ProfileMode,
    events: Vec<ProfileEvent>,
    counters: ProfileCounters,
    workload: ProfileWorkload,
    phase_summaries: Vec<PhaseSummary>,
    scope_counters: BTreeMap<ProfileScope, ProfileCounters>,
    scope_contract: ScopeContract,
    decode_scope: DecodeScope,
    measured_windows: BTreeMap<ProfileScope, MeasuredWindow>,
    benchmark_compatibility: BenchmarkCompatibility,
    profiler_overhead: ProfilerOverhead,
    gpu_counter_capture: Option<serde_json::Value>,
    collection_complete: bool,
}

impl Profiler {
    pub fn new(mode: ProfileMode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }
    pub fn mode(&self) -> ProfileMode {
        self.mode
    }
    pub fn enabled(&self) -> bool {
        self.mode != ProfileMode::Disabled
    }
    pub fn set_workload(&mut self, workload: ProfileWorkload) {
        self.workload = workload;
    }
    pub fn counters_mut(&mut self) -> &mut ProfileCounters {
        &mut self.counters
    }
    pub fn counters(&self) -> &ProfileCounters {
        &self.counters
    }
    pub fn set_counters(&mut self, counters: ProfileCounters) {
        self.counters = counters;
    }
    pub fn set_phase_summaries(&mut self, summaries: Vec<PhaseSummary>) {
        self.phase_summaries = summaries;
    }
    pub fn set_scope_counters(&mut self, counters: BTreeMap<ProfileScope, ProfileCounters>) {
        self.scope_counters = counters;
    }
    pub fn set_scope_contract(&mut self, contract: ScopeContract) {
        self.scope_contract = contract;
    }
    pub fn set_decode_scope(&mut self, scope: DecodeScope) {
        self.decode_scope = scope;
    }
    pub fn set_measured_windows(&mut self, windows: BTreeMap<ProfileScope, MeasuredWindow>) {
        self.measured_windows = windows;
    }
    pub fn set_benchmark_compatibility(&mut self, compatibility: BenchmarkCompatibility) {
        self.benchmark_compatibility = compatibility;
    }
    pub fn set_profiler_overhead(&mut self, overhead: ProfilerOverhead) {
        self.profiler_overhead = overhead;
    }
    pub fn set_gpu_counter_capture(&mut self, capture: Option<serde_json::Value>) {
        self.gpu_counter_capture = capture;
    }
    pub fn set_collection_complete(&mut self, complete: bool) {
        self.collection_complete = complete;
    }
    pub fn record(&mut self, event: ProfileEvent) {
        if !self.enabled() {
            return;
        }
        self.counters.host_encode_ns = self
            .counters
            .host_encode_ns
            .saturating_add(event.host_encode_ns);
        self.counters.host_wall_ns = self
            .counters
            .host_wall_ns
            .saturating_add(event.host_wall_ns);
        if let Some(gpu_ns) = event.gpu_ns {
            self.counters.attributed_gpu_duration_ns = self
                .counters
                .attributed_gpu_duration_ns
                .saturating_add(gpu_ns);
            self.counters.gpu_duration_source = "exact_per_dispatch_diagnostic_pass".into();
        }
        self.counters.gpu_ns = self
            .counters
            .gpu_ns
            .saturating_add(event.gpu_ns.unwrap_or_default());
        let attributed =
            event.operation_family != OperationFamily::Other && event.kernel_name.is_some();
        if attributed {
            self.counters.categorized_gpu_ns = self
                .counters
                .categorized_gpu_ns
                .saturating_add(event.gpu_ns.unwrap_or_default());
        }
        self.counters.cpu_wait_ns = self.counters.cpu_wait_ns.saturating_add(event.cpu_wait_ns);
        self.counters.upload_bytes = self
            .counters
            .upload_bytes
            .saturating_add(event.upload_bytes);
        self.counters.readback_bytes = self
            .counters
            .readback_bytes
            .saturating_add(event.readback_bytes);
        let dispatch_calls = event.dispatch_calls.max(1);
        self.counters.dispatches = self.counters.dispatches.saturating_add(dispatch_calls);
        self.counters.threadgroups_dispatched = self
            .counters
            .threadgroups_dispatched
            .saturating_add(event.threadgroups);
        self.counters.threads_dispatched = self
            .counters
            .threads_dispatched
            .saturating_add(event.threads);
        self.counters.timed_dispatches = self
            .counters
            .timed_dispatches
            .saturating_add(dispatch_calls * u64::from(event.gpu_ns.is_some()));
        self.counters.categorized_dispatches = self
            .counters
            .categorized_dispatches
            .saturating_add(dispatch_calls * u64::from(attributed));
        self.counters.upload_time_ns = self
            .counters
            .upload_time_ns
            .saturating_add(event.upload_time_ns);
        self.counters.readback_time_ns = self
            .counters
            .readback_time_ns
            .saturating_add(event.readback_time_ns);
        self.events.push(event);
    }
    pub fn report(&self) -> ProfileReport {
        let reconciliation = reconcile(&self.counters);
        let phase_reconciliation = self
            .phase_summaries
            .iter()
            .filter(|summary| matches!(summary.phase, ProfilePhase::Prefill | ProfilePhase::Decode))
            .map(|summary| reconcile_phase(summary))
            .collect::<Vec<_>>();
        let coverage = coverage_summary(&self.counters, &self.events, &phase_reconciliation);
        let mut warnings = reconciliation.warnings.clone();
        warnings.extend(coverage.warnings.clone());
        let synchronization = synchronization_summary(&self.phase_summaries);
        let memory = memory_summary(&self.counters, &self.workload);
        let mut overall_recommendations = recommendations(
            &self.events,
            Some(ProfileScope::DecodeMeasured),
            self.scope_counters
                .get(&ProfileScope::DecodeMeasured)
                .map_or(0, |x| x.categorized_gpu_ns),
            self.measured_windows
                .get(&ProfileScope::DecodeMeasured)
                .and_then(|x| x.wall_time_ms)
                .map(|x| (x * 1_000_000.0) as u64)
                .unwrap_or_default(),
            self.workload.measured_decode_tokens,
        );
        let mut prefill_hotspots = recommendations(
            &self.events,
            Some(ProfileScope::Prefill),
            self.scope_counters
                .get(&ProfileScope::Prefill)
                .map_or(0, |x| x.categorized_gpu_ns),
            self.measured_windows
                .get(&ProfileScope::Prefill)
                .and_then(|x| x.wall_time_ms)
                .map(|x| (x * 1_000_000.0) as u64)
                .unwrap_or_default(),
            self.workload.prompt_tokens,
        );
        let mut decode_measured_hotspots = recommendations(
            &self.events,
            Some(ProfileScope::DecodeMeasured),
            self.scope_counters
                .get(&ProfileScope::DecodeMeasured)
                .map_or(0, |x| x.categorized_gpu_ns),
            self.measured_windows
                .get(&ProfileScope::DecodeMeasured)
                .and_then(|x| x.wall_time_ms)
                .map(|x| (x * 1_000_000.0) as u64)
                .unwrap_or_default(),
            self.workload.measured_decode_tokens,
        );
        let mut decode_warmup_hotspots = recommendations(
            &self.events,
            Some(ProfileScope::DecodeWarmup),
            self.scope_counters
                .get(&ProfileScope::DecodeWarmup)
                .map_or(0, |x| x.categorized_gpu_ns),
            self.measured_windows
                .get(&ProfileScope::DecodeWarmup)
                .and_then(|x| x.wall_time_ms)
                .map(|x| (x * 1_000_000.0) as u64)
                .unwrap_or_default(),
            self.workload.warmup_decode_tokens,
        );
        let mut host_hotspots = host_recommendations(&self.phase_summaries);
        set_confidence(
            &mut overall_recommendations,
            scope_attribution_complete(self.scope_counters.get(&ProfileScope::DecodeMeasured)),
        );
        set_confidence(
            &mut prefill_hotspots,
            scope_attribution_complete(self.scope_counters.get(&ProfileScope::Prefill)),
        );
        set_confidence(
            &mut decode_measured_hotspots,
            scope_attribution_complete(self.scope_counters.get(&ProfileScope::DecodeMeasured)),
        );
        set_confidence(
            &mut decode_warmup_hotspots,
            scope_attribution_complete(self.scope_counters.get(&ProfileScope::DecodeWarmup)),
        );
        set_confidence(&mut host_hotspots, coverage.complete);
        ProfileReport {
            schema_version: 4,
            engine: "atlas".into(),
            mode: self.mode,
            profile_status: if self.collection_complete {
                "complete".into()
            } else {
                "incomplete".into()
            },
            scope_contract: self.scope_contract.clone(),
            decode_scope: self.decode_scope.clone(),
            measured_windows: self.measured_windows.clone(),
            benchmark_compatibility: self.benchmark_compatibility.clone(),
            profiler_overhead: self.profiler_overhead.clone(),
            gpu_counter_capture: self.gpu_counter_capture.clone(),
            workload: self.workload.clone(),
            counters: self.counters.clone(),
            reconciliation,
            phase_reconciliation,
            coverage,
            synchronization,
            memory,
            phase_summaries: self.phase_summaries.clone(),
            scope_counters: self.scope_counters.clone(),
            phases: aggregate(&self.events, Key::Phase),
            operation_families: aggregate(&self.events, Key::Operation),
            layers: aggregate(&self.events, Key::Layer),
            kernels: aggregate(&self.events, Key::Kernel),
            attention_dispatches: aggregate(&self.events, Key::AttentionDimensions),
            recommendations: overall_recommendations,
            hotspots: HotspotReport {
                prefill: prefill_hotspots,
                decode_measured: decode_measured_hotspots,
                decode_warmup: decode_warmup_hotspots,
                host_synchronization: host_hotspots,
            },
            warnings,
            events: self.events.clone(),
        }
    }
}

fn set_confidence(recommendations: &mut [Recommendation], complete: bool) {
    for recommendation in recommendations {
        recommendation.confidence = if complete { "high" } else { "low" }.into();
    }
}

/// A hotspot is only as trustworthy as its own measurement scope. Prefill may
/// intentionally retain unsupported setup kernels without invalidating an
/// exactly attributed decode-measured ranking.
fn scope_attribution_complete(counters: Option<&ProfileCounters>) -> bool {
    const MINIMUM_COVERAGE: f64 = 0.95;
    let Some(counters) = counters else {
        return false;
    };
    ratio(counters.categorized_dispatches, counters.dispatches) >= MINIMUM_COVERAGE
        && ratio(
            counters.categorized_gpu_ns,
            counters.attributed_gpu_duration_ns.max(counters.gpu_ns),
        ) >= MINIMUM_COVERAGE
}

#[derive(Clone, Copy)]
enum Key {
    Phase,
    Operation,
    Layer,
    Kernel,
    AttentionDimensions,
}

fn aggregate(events: &[ProfileEvent], key: Key) -> Vec<ProfileAggregate> {
    let mut map: BTreeMap<String, ProfileAggregate> = BTreeMap::new();
    for e in events {
        let (name, phase, layer, operation, kernel, attention_kind, attention_scan_pass) = match key
        {
            Key::Phase => (
                format!("{:?}", e.phase),
                Some(e.phase),
                None,
                None,
                None,
                None,
                None,
            ),
            Key::Operation => (
                e.operation_family.as_str().into(),
                None,
                None,
                Some(e.operation_family),
                None,
                None,
                None,
            ),
            Key::Layer => (
                e.layer_index
                    .map_or_else(|| "unattributed".into(), |v| v.to_string()),
                None,
                e.layer_index,
                None,
                None,
                None,
                None,
            ),
            Key::Kernel => (
                e.kernel_name
                    .clone()
                    .unwrap_or_else(|| "unattributed".into()),
                None,
                None,
                None,
                e.kernel_name.clone(),
                None,
                None,
            ),
            Key::AttentionDimensions => (
                format!(
                    "layer={};kind={};pass={}",
                    e.layer_index
                        .map_or_else(|| "unattributed".into(), |v| v.to_string()),
                    e.attention_kind.map_or("none", |v| match v {
                        AttentionKind::Global => "global",
                        AttentionKind::Sliding => "sliding",
                    }),
                    e.attention_scan_pass.map_or("none", |v| match v {
                        AttentionScanPass::Scan => "scan",
                        AttentionScanPass::Combine => "combine",
                        AttentionScanPass::SinglePass => "single_pass",
                    })
                ),
                None,
                e.layer_index,
                Some(e.operation_family),
                e.kernel_name.clone(),
                e.attention_kind,
                e.attention_scan_pass,
            ),
        };
        let x = map.entry(name.clone()).or_insert(ProfileAggregate {
            key: name,
            phase,
            layer_index: layer,
            attention_kind,
            attention_scan_pass,
            operation_family: operation,
            scope: Some(e.scope),
            kernel_name: kernel,
            ..Default::default()
        });
        x.events += 1;
        x.dispatches += e.dispatch_calls.max(1);
        x.threadgroups += e.threadgroups;
        x.host_encode_ns += e.host_encode_ns;
        x.host_wall_ns += e.host_wall_ns;
        x.gpu_ns += e.gpu_ns.unwrap_or_default();
        x.cpu_wait_ns += e.cpu_wait_ns;
        x.bytes_read += e.bytes_read_estimate;
        x.bytes_written += e.bytes_written_estimate;
        x.upload_bytes += e.upload_bytes;
        x.readback_bytes += e.readback_bytes;
    }
    let mut values: Vec<_> = map.into_values().collect();
    values.sort_by(|a, b| b.gpu_ns.cmp(&a.gpu_ns));
    values
}

fn reconcile(counters: &ProfileCounters) -> DispatchReconciliation {
    let uncategorized = counters
        .dispatches
        .saturating_sub(counters.categorized_dispatches);
    let untimed = counters
        .dispatches
        .saturating_sub(counters.timed_dispatches);
    let attributed_gpu =
        effective_attributed_gpu_ns(counters.gpu_ns, counters.attributed_gpu_duration_ns);
    let categorized_gpu = counters.categorized_gpu_ns.min(attributed_gpu);
    let uncategorized_gpu = counters
        .attributed_gpu_duration_ns
        .max(counters.gpu_ns)
        .saturating_sub(categorized_gpu);
    let dispatch_coverage = ratio(counters.categorized_dispatches, counters.dispatches);
    let gpu_timing_coverage = ratio(categorized_gpu, attributed_gpu);
    let mut warnings = Vec::new();
    if untimed > 0 {
        warnings.push(format!(
            "{} dispatch calls have no per-dispatch GPU timing",
            untimed
        ));
    }
    if uncategorized > 0 {
        warnings.push(format!(
            "{} dispatch calls have no operation attribution",
            uncategorized
        ));
    }
    if gpu_timing_coverage < 1.0 {
        warnings.push("GPU attribution is incomplete; uncategorized GPU time remains outside hotspot attribution".into());
    }
    DispatchReconciliation {
        dispatch_calls: counters.dispatches,
        threadgroups_dispatched: counters.threadgroups_dispatched,
        threads_dispatched: counters.threads_dispatched,
        timed_dispatch_calls: counters.timed_dispatches,
        untimed_dispatch_calls: untimed,
        categorized_dispatch_calls: counters.categorized_dispatches,
        uncategorized_dispatch_calls: uncategorized,
        total_gpu_ns: counters.gpu_ns,
        attributed_gpu_duration_ns: attributed_gpu,
        categorized_gpu_ns: categorized_gpu,
        uncategorized_gpu_ns: uncategorized_gpu,
        dispatch_coverage,
        gpu_timing_coverage,
        complete: warnings.is_empty(),
        warnings,
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn effective_attributed_gpu_ns(production_gpu_ns: u64, attributed_gpu_ns: u64) -> u64 {
    if attributed_gpu_ns == 0 {
        production_gpu_ns
    } else {
        attributed_gpu_ns
    }
}

fn reconcile_phase(summary: &PhaseSummary) -> DispatchReconciliation {
    let attributed_gpu_ns = effective_attributed_gpu_ns(summary.gpu_ns, summary.attributed_gpu_ns);
    let uncategorized_gpu_ns = attributed_gpu_ns.saturating_sub(summary.categorized_gpu_ns);
    let mut warnings = Vec::new();
    let dispatch_coverage = ratio(summary.categorized_dispatches, summary.dispatch_calls);
    let gpu_coverage = ratio(summary.categorized_gpu_ns, attributed_gpu_ns);
    if gpu_coverage < 0.90 {
        warnings.push(format!(
            "GPU attribution coverage is {:.1}%",
            gpu_coverage * 100.0
        ));
    }
    if gpu_coverage < 0.75 || dispatch_coverage < 0.75 {
        warnings.push("phase attribution is incomplete below the 75% threshold".into());
    }
    DispatchReconciliation {
        dispatch_calls: summary.dispatch_calls,
        threadgroups_dispatched: summary.threadgroups_dispatched,
        threads_dispatched: summary.threads_dispatched,
        timed_dispatch_calls: summary.timed_dispatches,
        untimed_dispatch_calls: summary.untimed_dispatches,
        categorized_dispatch_calls: summary.categorized_dispatches,
        uncategorized_dispatch_calls: summary.uncategorized_dispatches,
        total_gpu_ns: summary.gpu_ns,
        attributed_gpu_duration_ns: attributed_gpu_ns,
        categorized_gpu_ns: summary.categorized_gpu_ns,
        uncategorized_gpu_ns,
        dispatch_coverage,
        gpu_timing_coverage: gpu_coverage,
        complete: warnings.is_empty(),
        warnings,
    }
}

fn coverage_summary(
    counters: &ProfileCounters,
    events: &[ProfileEvent],
    phase_reconciliation: &[DispatchReconciliation],
) -> CoverageSummary {
    let gpu_attribution = ratio(counters.categorized_gpu_ns, counters.gpu_ns);
    let dispatch_attribution = ratio(counters.categorized_dispatches, counters.dispatches);
    let operation_attribution = ratio(
        events
            .iter()
            .filter(|event| event.operation_family != OperationFamily::Other)
            .map(|event| event.dispatch_calls.max(1))
            .sum(),
        counters.dispatches,
    );
    let kernel_attribution = ratio(
        events
            .iter()
            .filter(|event| event.kernel_name.is_some())
            .map(|event| event.dispatch_calls.max(1))
            .sum(),
        counters.dispatches,
    );
    // CPU waits are measured, but queue idle gaps, GPU-waiting-for-CPU, and
    // readback/upload waits are not universally exposed by Metal. Do not
    // report complete synchronization coverage while those sources are
    // unavailable.
    let synchronization_attribution = 0.0;
    let mut warnings = Vec::new();
    if gpu_attribution < 0.90 {
        warnings.push(format!(
            "GPU attribution coverage below 90%: {:.1}%",
            gpu_attribution * 100.0
        ));
    }
    warnings.push(
        "synchronization attribution is partial: GPU-wait-for-CPU and transfer waits are unavailable".into(),
    );
    let complete = gpu_attribution >= 0.75
        && dispatch_attribution >= 0.75
        && phase_reconciliation.iter().all(|x| x.complete);
    if !complete {
        warnings.push(
            "hotspot attribution is partial; percentages exclude uncategorized GPU duration".into(),
        );
    }
    CoverageSummary {
        gpu_attribution,
        dispatch_attribution,
        operation_attribution,
        kernel_attribution,
        synchronization_attribution,
        complete,
        warnings,
    }
}

fn synchronization_summary(summaries: &[PhaseSummary]) -> SynchronizationSummary {
    let measured = summaries
        .iter()
        .filter(|x| {
            matches!(
                x.scope,
                ProfileScope::Prefill | ProfileScope::DecodeWarmup | ProfileScope::DecodeMeasured
            )
        })
        .collect::<Vec<_>>();
    SynchronizationSummary {
        cpu_wait_ns: summaries
            .iter()
            .filter(|x| {
                matches!(
                    x.scope,
                    ProfileScope::Prefill
                        | ProfileScope::DecodeWarmup
                        | ProfileScope::DecodeMeasured
                )
            })
            .map(|x| x.cpu_wait_ns)
            .sum(),
        command_buffer_count: summaries.iter().map(|x| x.command_buffers).sum(),
        command_buffer_idle_gap_ns: measured
            .iter()
            .map(|x| x.command_buffer_idle_gap_ns)
            .sum::<Option<u64>>(),
        command_buffer_schedule_ns: measured
            .iter()
            .map(|x| x.command_buffer_schedule_ns)
            .sum::<Option<u64>>(),
        ..Default::default()
    }
}

fn memory_summary(counters: &ProfileCounters, workload: &ProfileWorkload) -> MemorySummary {
    let traffic = counters
        .upload_bytes
        .saturating_add(counters.readback_bytes);
    MemorySummary {
        resident_bytes: counters.resident_bytes,
        peak_resident_bytes: counters.peak_resident_bytes,
        kv_cache_bytes: counters.kv_cache_bytes,
        upload_bytes: counters.upload_bytes,
        readback_bytes: counters.readback_bytes,
        bytes_per_prompt_token: (workload.prompt_tokens > 0)
            .then(|| traffic as f64 / workload.prompt_tokens as f64),
        bytes_per_generated_token: (workload.generated_tokens > 0)
            .then(|| traffic as f64 / workload.generated_tokens as f64),
        effective_bandwidth_bytes_per_second: (counters.gpu_ns > 0)
            .then(|| traffic as f64 / (counters.gpu_ns as f64 / 1e9)),
    }
}

fn recommendations(
    events: &[ProfileEvent],
    scope: Option<ProfileScope>,
    gpu_ns: u64,
    wall_ns: u64,
    measured_tokens: u64,
) -> Vec<Recommendation> {
    let filtered = events
        .iter()
        .filter(|event| scope.is_none_or(|expected| event.scope == expected))
        .cloned()
        .collect::<Vec<_>>();
    let total = gpu_ns.max(1) as f64;
    let mut candidates = aggregate(&filtered, Key::Operation)
        .into_iter()
        .map(|a| {
            let share = a.gpu_ns as f64 / total;
            let mean_gpu_ns = a.gpu_ns / a.dispatches.max(1);
            let classification =
                if a.cpu_wait_ns > 0 && a.cpu_wait_ns as f64 > a.gpu_ns as f64 * 0.25 {
                    "synchronization_candidate"
                } else if mean_gpu_ns < 10_000 {
                    "dispatch_overhead_candidate"
                } else if a.bytes_read > 0 || a.bytes_written > 0 {
                    "bandwidth_candidate"
                } else {
                    "mixed"
                };
            Recommendation {
                rank: 0,
                target: a.key,
                classification: classification.into(),
                scope: scope.unwrap_or(ProfileScope::Other),
                priority_score: share,
                phase_time_share: share,
                evidence: vec![
                    format!("{} dispatches", a.dispatches),
                    format!("{:.3} ms GPU", a.gpu_ns as f64 / 1_000_000.0),
                ],
                absolute_ms: a.gpu_ns as f64 / 1_000_000.0,
                categorized_gpu_share: share,
                wall_time_share: None,
                wall_time_share_status: if wall_ns == 0 {
                    "unavailable_no_valid_non_overlapping_wall_interval".into()
                } else {
                    "unavailable_gpu_duration_is_not_host_wall_contribution".into()
                },
                dispatches_per_measured_token: if measured_tokens == 0 {
                    0.0
                } else {
                    a.dispatches as f64 / measured_tokens as f64
                },
                gpu_ms_per_measured_token: if measured_tokens == 0 {
                    0.0
                } else {
                    a.gpu_ns as f64 / measured_tokens as f64 / 1_000_000.0
                },
                dispatch_calls: a.dispatches,
                threadgroups: a.threadgroups,
                confidence: "medium".into(),
            }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.priority_score.total_cmp(&a.priority_score));
    for (i, x) in candidates.iter_mut().enumerate() {
        x.rank = i + 1;
    }
    candidates.truncate(10);
    candidates
}

fn host_recommendations(summaries: &[PhaseSummary]) -> Vec<Recommendation> {
    let wait = summaries
        .iter()
        .filter(|summary| summary.phase != ProfilePhase::HostSynchronization)
        .map(|summary| summary.cpu_wait_ns)
        .sum::<u64>();
    if wait == 0 {
        return Vec::new();
    }
    vec![Recommendation {
        rank: 1,
        target: "cpu_wait".into(),
        classification: "synchronization_candidate".into(),
        scope: ProfileScope::ProfilerOverhead,
        priority_score: 1.0,
        phase_time_share: 1.0,
        evidence: vec![format!("{:.3} ms CPU wait", wait as f64 / 1_000_000.0)],
        absolute_ms: wait as f64 / 1_000_000.0,
        categorized_gpu_share: 0.0,
        wall_time_share: None,
        wall_time_share_status: "unavailable_cpu_wait_is_not_operation_wall_time".into(),
        dispatches_per_measured_token: 0.0,
        gpu_ms_per_measured_token: 0.0,
        dispatch_calls: 0,
        threadgroups: 0,
        confidence: "medium".into(),
    }]
}

impl ProfileReport {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "# Atlas Performance Profile\n\nMode: `{:?}`\nProfile status: `{}`\nHotspot scope: `{}`\n",
            self.mode,
            self.profile_status,
            self.scope_contract.hotspot_scope.as_str()
        );
        let _ = writeln!(
            out,
            "## Workload\n\n- Prompt tokens: {}\n- Generated tokens: {}\n- Warmup decode tokens: {}\n- Measured decode tokens: {}\n- Completed decode tokens: {}\n- Prefill: {:.3} ms\n- Measured decode: {:.3} ms\n- Measured decode throughput: {:.2} tok/s\n",
            self.workload.prompt_tokens,
            self.workload.generated_tokens,
            self.workload.warmup_decode_tokens,
            self.workload.measured_decode_tokens,
            self.workload.completed_decode_tokens,
            self.workload.prefill_ns as f64 / 1e6,
            self.workload.decode_ns as f64 / 1e6,
            if self.workload.decode_ns == 0 {
                0.0
            } else {
                self.workload.measured_decode_tokens as f64 / (self.workload.decode_ns as f64 / 1e9)
            }
        );
        let _ = writeln!(
            out,
            "## Counters\n\n| Metric | Value |\n|---|---:|\n| Command buffers | {} |\n| Dispatch calls | {} |\n| Threadgroups | {} |\n| Threads | {} |\n| Upload bytes | {} |\n| Readback bytes | {} |\n| Production GPU elapsed (ms) | {:.3} |\n| Attributed GPU duration (ms) | {:.3} |\n| Categorized GPU duration (ms) | {:.3} |\n| GPU duration source | {} |\n| CPU wait (ms) | {:.3} |\n| Peak resident bytes | {} |\n| KV-cache bytes | {} |\n",
            self.counters.command_buffers,
            self.counters.dispatches,
            self.counters.threadgroups_dispatched,
            self.counters.threads_dispatched,
            self.counters.upload_bytes,
            self.counters.readback_bytes,
            self.counters.production_gpu_elapsed_ns as f64 / 1e6,
            self.counters.attributed_gpu_duration_ns as f64 / 1e6,
            self.counters.categorized_gpu_ns as f64 / 1e6,
            self.counters.gpu_duration_source,
            self.counters.cpu_wait_ns as f64 / 1e6,
            self.counters.peak_resident_bytes,
            self.counters.kv_cache_bytes
        );
        let _ = writeln!(
            out,
            "## Reconciliation\n\n- Dispatch coverage: {:.1}%\n- GPU timing coverage: {:.1}%\n- Production GPU elapsed: {:.3} ms\n- Attributed GPU duration: {:.3} ms\n- Complete: {}\n- Timed dispatches: {}\n- Untimed dispatches: {}\n- Categorized dispatches: {}\n- Uncategorized dispatches: {}\n",
            self.reconciliation.dispatch_coverage * 100.0,
            self.reconciliation.gpu_timing_coverage * 100.0,
            self.reconciliation.total_gpu_ns as f64 / 1e6,
            self.reconciliation.attributed_gpu_duration_ns as f64 / 1e6,
            self.reconciliation.complete,
            self.reconciliation.timed_dispatch_calls,
            self.reconciliation.untimed_dispatch_calls,
            self.reconciliation.categorized_dispatch_calls,
            self.reconciliation.uncategorized_dispatch_calls
        );
        let _ = writeln!(
            out,
            "## Coverage\n\n- GPU attribution: {:.1}%\n- Dispatch attribution: {:.1}%\n- Operation attribution: {:.1}%\n- Kernel attribution: {:.1}%\n- Synchronization attribution: {:.1}%\n- Complete: {}\n",
            self.coverage.gpu_attribution * 100.0,
            self.coverage.dispatch_attribution * 100.0,
            self.coverage.operation_attribution * 100.0,
            self.coverage.kernel_attribution * 100.0,
            self.coverage.synchronization_attribution * 100.0,
            self.coverage.complete
        );
        let _ = writeln!(
            out,
            "## Phase reconciliation\n\n| Phase | Wall ms | GPU ms | Categorized GPU ms | Uncategorized GPU ms | CPU encode ms | CPU wait ms | Upload ms | Readback ms | Unexplained ms |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        );
        for summary in &self.phase_summaries {
            let _ = writeln!(
                out,
                "| {:?} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                summary.phase,
                summary.wall_ns as f64 / 1e6,
                summary.gpu_ns as f64 / 1e6,
                summary.categorized_gpu_ns as f64 / 1e6,
                summary.uncategorized_gpu_ns as f64 / 1e6,
                summary.host_encode_ns as f64 / 1e6,
                summary.cpu_wait_ns as f64 / 1e6,
                summary.upload_time_ns as f64 / 1e6,
                summary.readback_time_ns as f64 / 1e6,
                summary.unexplained_ns as f64 / 1e6
            );
        }
        let _ = writeln!(
            out,
            "\n## Synchronization\n\n- CPU wait: {:.3} ms\n- Command-buffer idle gaps: {}\n- Command-buffer schedule time: {}\n- GPU waiting for CPU: {}\n- Readback waits: {}\n- Upload waits: {}\n",
            self.synchronization.cpu_wait_ns as f64 / 1e6,
            format_optional_ms(self.synchronization.command_buffer_idle_gap_ns),
            format_optional_ms(self.synchronization.command_buffer_schedule_ns),
            self.synchronization.gpu_waiting_for_cpu_ns.map_or_else(
                || "unavailable".into(),
                |v| format!("{:.3} ms", v as f64 / 1e6)
            ),
            format_optional_ms(self.synchronization.readback_wait_ns),
            format_optional_ms(self.synchronization.upload_wait_ns)
        );
        let _ = writeln!(
            out,
            "\n## Memory\n\n- Resident: {} bytes\n- Peak: {} bytes\n- KV cache: {} bytes\n- Upload: {} bytes\n- Readback: {} bytes\n- Effective bandwidth: {}\n",
            self.memory.resident_bytes,
            self.memory.peak_resident_bytes,
            self.memory.kv_cache_bytes,
            self.memory.upload_bytes,
            self.memory.readback_bytes,
            self.memory
                .effective_bandwidth_bytes_per_second
                .map_or_else(|| "unavailable".into(), |v| format!("{:.3} GB/s", v / 1e9))
        );
        let _ = writeln!(
            out,
            "## Phases\n\n| Phase | Wall ms | GPU ms | Dispatches | Threadgroups | Upload bytes | Readback bytes | Tokens/sec |\n|---|---:|---:|---:|---:|---:|---:|---:|"
        );
        for summary in &self.phase_summaries {
            let _ = writeln!(
                out,
                "| {:?} | {:.3} | {:.3} | {} | {} | {} | {} | {:.2} |",
                summary.phase,
                summary.wall_ns as f64 / 1e6,
                summary.gpu_ns as f64 / 1e6,
                summary.dispatch_calls,
                summary.threadgroups_dispatched,
                summary.upload_bytes,
                summary.readback_bytes,
                summary.tokens_per_second
            );
        }
        if let Some(counters) = &self.gpu_counter_capture {
            let _ = writeln!(
                out,
                "\n## Diagnostic GPU counters\n\n```json\n{}\n```",
                counters
            );
        }
        if !self.attention_dispatches.is_empty() {
            let _ = writeln!(
                out,
                "\n## Attention dispatch attribution\n\n| Layer | Kind | Scan pass | GPU ms | Dispatches | Threadgroups |\n|---:|---|---|---:|---:|---:|"
            );
            for aggregate in &self.attention_dispatches {
                let _ = writeln!(
                    out,
                    "| {} | {:?} | {:?} | {:.3} | {} | {} |",
                    aggregate
                        .layer_index
                        .map_or_else(|| "unattributed".into(), |layer| layer.to_string()),
                    aggregate.attention_kind,
                    aggregate.attention_scan_pass,
                    aggregate.gpu_ns as f64 / 1e6,
                    aggregate.dispatches,
                    aggregate.threadgroups
                );
            }
        }
        let _ = writeln!(
            out,
            "## Recommendations\n\n| Rank | Scope | Target | Classification | Confidence | GPU ms | Categorized GPU share | Wall share | Dispatch calls | Dispatches/token | Threadgroups |\n|---:|---|---|---|---|---:|---:|---|---:|---:|---:|\n"
        );
        for r in &self.recommendations {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {:.3} | {:.1}% | {} | {} | {:.3} | {:.3} |",
                r.rank,
                r.scope.as_str(),
                r.target,
                r.classification,
                r.confidence,
                r.absolute_ms,
                r.categorized_gpu_share * 100.0,
                r.wall_time_share
                    .map_or_else(|| "unavailable".into(), |x| format!("{:.1}%", x * 100.0)),
                r.dispatch_calls,
                r.dispatches_per_measured_token,
                r.threadgroups
            );
        }
        for (name, hotspots) in [
            ("Prefill", &self.hotspots.prefill),
            ("Decode measured", &self.hotspots.decode_measured),
            ("Decode warmup", &self.hotspots.decode_warmup),
            (
                "Host / Synchronization",
                &self.hotspots.host_synchronization,
            ),
        ] {
            let _ = writeln!(
                out,
                "\n### {name} hotspots\n\n| Rank | Scope | Target | Classification | Confidence | Absolute ms | GPU share | Wall share | Dispatch calls | Dispatches/token | Threadgroups |\n|---:|---|---|---|---|---:|---:|---|---:|---:|---:|"
            );
            for hotspot in hotspots {
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {:.3} | {:.1}% | {} | {} | {:.3} | {:.3} |",
                    hotspot.rank,
                    hotspot.scope.as_str(),
                    hotspot.target,
                    hotspot.classification,
                    hotspot.confidence,
                    hotspot.absolute_ms,
                    hotspot.categorized_gpu_share * 100.0,
                    hotspot
                        .wall_time_share
                        .map_or_else(|| "unavailable".into(), |x| format!("{:.1}%", x * 100.0)),
                    hotspot.dispatch_calls,
                    hotspot.dispatches_per_measured_token,
                    hotspot.threadgroups
                );
            }
        }
        if !self.warnings.is_empty() {
            let _ = writeln!(out, "\n## Warnings\n");
            for warning in &self.warnings {
                let _ = writeln!(out, "- {warning}");
            }
        }
        out
    }
}

pub fn duration_ns(value: Duration) -> u64 {
    value.as_nanos().min(u64::MAX as u128) as u64
}

fn format_optional_ms(value: Option<u64>) -> String {
    value.map_or_else(
        || "unavailable".into(),
        |ns| format!("{:.3} ms", ns as f64 / 1e6),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_profiler_does_not_retain_events() {
        let mut p = Profiler::new(ProfileMode::Disabled);
        p.record(ProfileEvent::default());
        assert!(p.report().events.is_empty());
    }
    #[test]
    fn aggregates_operations_and_ranks() {
        let mut p = Profiler::new(ProfileMode::Diagnostic);
        p.set_workload(ProfileWorkload {
            measured_decode_tokens: 1,
            ..Default::default()
        });
        p.set_scope_counters(BTreeMap::from([(
            ProfileScope::DecodeMeasured,
            ProfileCounters {
                categorized_gpu_ns: 30,
                ..Default::default()
            },
        )]));
        p.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            operation_family: OperationFamily::FfnDown,
            kernel_name: Some("down".into()),
            gpu_ns: Some(20),
            ..Default::default()
        });
        p.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            operation_family: OperationFamily::RmsNorm,
            kernel_name: Some("norm".into()),
            gpu_ns: Some(10),
            ..Default::default()
        });
        let r = p.report();
        assert_eq!(r.operation_families[0].key, "ffn_down");
        assert_eq!(r.recommendations[0].rank, 1);
    }
    #[test]
    fn reports_are_machine_and_human_readable() {
        let r = Profiler::new(ProfileMode::Benchmark).report();
        assert!(r.to_json().unwrap().contains("schema_version"));
        assert!(r.to_markdown().contains("Atlas Performance Profile"));
    }

    #[test]
    fn measured_scope_excludes_warmup_from_default_rankings() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.set_workload(ProfileWorkload {
            warmup_decode_tokens: 32,
            measured_decode_tokens: 128,
            completed_decode_tokens: 160,
            ..Default::default()
        });
        let mut scoped = BTreeMap::new();
        scoped.insert(
            ProfileScope::DecodeWarmup,
            ProfileCounters {
                dispatches: 32,
                categorized_dispatches: 32,
                gpu_ns: 900,
                categorized_gpu_ns: 900,
                ..Default::default()
            },
        );
        scoped.insert(
            ProfileScope::DecodeMeasured,
            ProfileCounters {
                dispatches: 128,
                categorized_dispatches: 128,
                gpu_ns: 100,
                categorized_gpu_ns: 100,
                ..Default::default()
            },
        );
        profiler.set_scope_counters(scoped);
        profiler.record(ProfileEvent {
            scope: ProfileScope::DecodeWarmup,
            operation_family: OperationFamily::FfnDown,
            kernel_name: Some("warmup".into()),
            dispatch_calls: 32,
            gpu_ns: Some(900),
            ..Default::default()
        });
        profiler.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            operation_family: OperationFamily::OutputProjection,
            kernel_name: Some("measured".into()),
            dispatch_calls: 128,
            gpu_ns: Some(100),
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(
            report.recommendations[0].scope,
            ProfileScope::DecodeMeasured
        );
        assert_eq!(report.recommendations[0].dispatches_per_measured_token, 1.0);
        assert_eq!(report.hotspots.decode_warmup.len(), 1);
        assert_eq!(report.recommendations[0].categorized_gpu_share, 1.0);
    }

    #[test]
    fn measured_decode_confidence_ignores_unrelated_prefill_attribution() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.set_workload(ProfileWorkload {
            measured_decode_tokens: 1,
            ..Default::default()
        });
        profiler.set_scope_counters(BTreeMap::from([(
            ProfileScope::DecodeMeasured,
            ProfileCounters {
                dispatches: 1,
                categorized_dispatches: 1,
                gpu_ns: 100,
                attributed_gpu_duration_ns: 100,
                categorized_gpu_ns: 100,
                ..Default::default()
            },
        )]));
        profiler.record(ProfileEvent {
            scope: ProfileScope::Prefill,
            operation_family: OperationFamily::Other,
            gpu_ns: Some(100),
            ..Default::default()
        });
        profiler.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            operation_family: OperationFamily::FfnDown,
            gpu_ns: Some(100),
            ..Default::default()
        });
        let report = profiler.report();
        assert!(!report.coverage.complete);
        assert_eq!(report.recommendations[0].confidence, "high");
    }

    #[test]
    fn physical_command_buffer_overlap_is_serialized_without_fabricating_a_buffer() {
        let mut profiler = Profiler::new(ProfileMode::Benchmark);
        profiler.set_decode_scope(DecodeScope {
            warmup_decode_tokens_requested: 32,
            warmup_decode_tokens_completed: 32,
            measured_decode_tokens_requested: 128,
            measured_decode_tokens_completed: 128,
            completed_decode_tokens_total: 160,
            hotspot_scope: ProfileScope::DecodeMeasured,
            physical_command_buffer_overlap: true,
            physical_command_buffer_overlap_reason: Some("prefill selects token one".into()),
        });
        let value: serde_json::Value =
            serde_json::from_str(&profiler.report().to_json().unwrap()).unwrap();
        assert_eq!(value["schema_version"], 4);
        assert_eq!(value["decode_scope"]["completed_decode_tokens_total"], 160);
        assert_eq!(
            value["decode_scope"]["physical_command_buffer_overlap"],
            true
        );
    }

    #[test]
    fn reconciliation_marks_partial_dispatch_and_gpu_coverage() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.set_counters(ProfileCounters {
            dispatches: 100,
            threadgroups_dispatched: 400,
            threads_dispatched: 12_800,
            timed_dispatches: 20,
            categorized_dispatches: 10,
            gpu_ns: 1_000,
            categorized_gpu_ns: 200,
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.reconciliation.untimed_dispatch_calls, 80);
        assert_eq!(report.reconciliation.uncategorized_dispatch_calls, 90);
        assert_eq!(report.reconciliation.dispatch_coverage, 0.1);
        assert_eq!(report.reconciliation.gpu_timing_coverage, 0.2);
        assert!(!report.reconciliation.complete);
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn aggregated_event_preserves_dispatch_multiplicity_and_kernel_coverage() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.record(ProfileEvent {
            operation_family: OperationFamily::FfnDown,
            kernel_name: Some("matmul_q4_0".into()),
            dispatch_calls: 12,
            threadgroups: 48,
            gpu_ns: Some(1_200),
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.reconciliation.dispatch_calls, 12);
        assert_eq!(report.reconciliation.threadgroups_dispatched, 48);
        assert_eq!(report.coverage.kernel_attribution, 1.0);
        assert_eq!(report.schema_version, 4);
        assert!(report.to_markdown().contains("Unexplained ms"));
    }

    #[test]
    fn aggregates_conservative_memory_traffic_estimates() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            phase: ProfilePhase::Decode,
            operation_family: OperationFamily::FfnDown,
            kernel_name: Some("matvec_q4_0_16row".into()),
            dispatch_calls: 2,
            bytes_read_estimate: 10_000,
            bytes_written_estimate: 2_000,
            ..Default::default()
        });
        let report = profiler.report();
        let aggregate = report
            .kernels
            .iter()
            .find(|aggregate| aggregate.key == "matvec_q4_0_16row")
            .expect("kernel traffic aggregate");
        assert_eq!(aggregate.bytes_read, 10_000);
        assert_eq!(aggregate.bytes_written, 2_000);
    }

    #[test]
    fn aggregates_attention_by_layer_kind_and_scan_pass() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            phase: ProfilePhase::Decode,
            layer_index: Some(7),
            attention_kind: Some(AttentionKind::Global),
            attention_scan_pass: Some(AttentionScanPass::Scan),
            operation_family: OperationFamily::AttentionScore,
            kernel_name: Some("attention_scan".into()),
            dispatch_calls: 8,
            threadgroups: 32,
            gpu_ns: Some(640),
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.attention_dispatches.len(), 1);
        let aggregate = &report.attention_dispatches[0];
        assert_eq!(aggregate.layer_index, Some(7));
        assert_eq!(aggregate.attention_kind, Some(AttentionKind::Global));
        assert_eq!(aggregate.attention_scan_pass, Some(AttentionScanPass::Scan));
        assert_eq!(aggregate.dispatches, 8);
        assert_eq!(aggregate.threadgroups, 32);
    }

    #[test]
    fn unknown_kernel_remains_uncategorized() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.record(ProfileEvent {
            dispatch_calls: 4,
            gpu_ns: Some(400),
            operation_family: OperationFamily::Other,
            kernel_name: None,
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.reconciliation.uncategorized_dispatch_calls, 4);
        assert!(!report.coverage.complete);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("incomplete"))
        );
    }

    #[test]
    fn phase_wall_share_and_cpu_wait_are_not_double_counted() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.set_phase_summaries(vec![
            PhaseSummary {
                phase: ProfilePhase::Decode,
                scope: ProfileScope::DecodeMeasured,
                wall_ns: 1_000,
                gpu_ns: 400,
                cpu_wait_ns: 500,
                categorized_gpu_ns: 400,
                dispatch_calls: 1,
                categorized_dispatches: 1,
                ..Default::default()
            },
            PhaseSummary {
                phase: ProfilePhase::HostSynchronization,
                scope: ProfileScope::ProfilerOverhead,
                cpu_wait_ns: 500,
                ..Default::default()
            },
        ]);
        profiler.record(ProfileEvent {
            phase: ProfilePhase::Decode,
            scope: ProfileScope::DecodeMeasured,
            operation_family: OperationFamily::FfnDown,
            kernel_name: Some("down".into()),
            gpu_ns: Some(400),
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.synchronization.cpu_wait_ns, 500);
        assert_eq!(report.hotspots.decode_measured[0].wall_time_share, None);
        assert_eq!(
            report.hotspots.decode_measured[0].wall_time_share_status,
            "unavailable_no_valid_non_overlapping_wall_interval"
        );
    }

    #[test]
    fn completed_collection_can_have_attribution_warnings() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.set_collection_complete(true);
        profiler.set_counters(ProfileCounters {
            dispatches: 10,
            categorized_dispatches: 9,
            gpu_ns: 100,
            attributed_gpu_duration_ns: 100,
            categorized_gpu_ns: 90,
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(report.profile_status, "complete");
        assert!(!report.warnings.is_empty());
        assert_eq!(report.reconciliation.attributed_gpu_duration_ns, 100);
    }

    #[test]
    fn exact_attribution_is_distinct_from_production_gpu_elapsed() {
        let mut profiler = Profiler::new(ProfileMode::Diagnostic);
        profiler.set_scope_counters(BTreeMap::from([(
            ProfileScope::DecodeMeasured,
            ProfileCounters {
                gpu_ns: 100,
                production_gpu_elapsed_ns: 100,
                attributed_gpu_duration_ns: 130,
                categorized_gpu_ns: 120,
                gpu_duration_source: "production_boundary_plus_exact_dispatch_attribution".into(),
                ..Default::default()
            },
        )]));
        profiler.set_workload(ProfileWorkload {
            measured_decode_tokens: 2,
            ..Default::default()
        });
        profiler.record(ProfileEvent {
            scope: ProfileScope::DecodeMeasured,
            operation_family: OperationFamily::FfnDown,
            kernel_name: Some("down".into()),
            dispatch_calls: 1,
            gpu_ns: Some(120),
            ..Default::default()
        });
        let report = profiler.report();
        assert_eq!(
            report.scope_counters[&ProfileScope::DecodeMeasured].gpu_ns,
            100
        );
        assert_eq!(
            report.scope_counters[&ProfileScope::DecodeMeasured].attributed_gpu_duration_ns,
            130
        );
        assert_eq!(report.recommendations[0].absolute_ms, 0.00012);
    }

    #[test]
    fn zero_warmup_keeps_measured_window_unambiguous() {
        let scope = DecodeScope {
            warmup_decode_tokens_requested: 0,
            warmup_decode_tokens_completed: 0,
            measured_decode_tokens_requested: 128,
            measured_decode_tokens_completed: 128,
            completed_decode_tokens_total: 128,
            hotspot_scope: ProfileScope::DecodeMeasured,
            ..Default::default()
        };
        let value: serde_json::Value = serde_json::to_value(scope).unwrap();
        assert_eq!(value["warmup_decode_tokens_completed"], 0);
        assert_eq!(value["measured_decode_tokens_completed"], 128);
        assert_eq!(value["completed_decode_tokens_total"], 128);
    }
}
