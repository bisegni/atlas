# Quantization Preflight Artifact and Error-Propagation Graph

## Purpose

This document extends `docs/dynamic_quantization_preflight_strategy.md` with two concrete requirements:

1. Atlas preflight should build an error-propagation graph to improve candidate ranking.
2. Atlas preflight must emit a reusable optimization artifact that normal Atlas runs can load and apply without repeating preflight.

The artifact is not only a benchmark report. It is an executable, validated, hardware-specific optimization plan for one exact model.

The intended lifecycle is:

```text
first run / explicit preflight
    -> inspect model and hardware
    -> measure baseline
    -> estimate sensitivity and error propagation
    -> evaluate candidate plans
    -> validate exact token/EOS parity
    -> select winner or safe baseline
    -> write reusable plan artifact

subsequent run
    -> load model
    -> load plan artifact
    -> validate artifact identity and compatibility
    -> materialize selected formats/layouts
    -> bind selected Resident Metal kernels
    -> run normal inference without repeating search
```

The normal runtime must never trust an incompatible or partially validated artifact.

---

## 1. Error-propagation graph

Local reconstruction error is useful, but it does not describe what happens after the perturbed group.

A small error may:

- be absorbed by the residual stream;
- be attenuated by normalization;
- remain harmless through many layers;
- be amplified by attention or FFN operations;
- become important only near a low-margin output token;
- alter EOS many decode steps later.

Atlas should therefore represent the model as an executable dependency graph and attach measured error-propagation evidence to its edges.

### Graph nodes

Recommended nodes:

```text
vocabulary.token_embedding
layer[N].attention.qkv
layer[N].attention.output
layer[N].ffn.gate_up
layer[N].ffn.down
layer[N].residual_after_attention
layer[N].residual_after_ffn
final_norm
vocabulary.output_projection
```

The first implementation may use fewer nodes, but it should preserve the real execution order.

### Graph edges

An edge describes how a perturbation at one node affects a downstream observation point.

Example:

```text
layer.5.attention.qkv
    -> layer.5.residual_after_attention
    -> layer.5.ffn.gate_up
    -> layer.5.residual_after_ffn
    -> ...
    -> vocabulary.output_projection
```

Each measured edge should record values such as:

```text
input_error_norm
output_error_norm
amplification_ratio
cosine_distance_delta
residual_relative_error
logit_delta
baseline_logit_margin
local_top1_changed
sample_count
```

A basic amplification ratio is:

```text
amplification_ratio = downstream_error_norm / (upstream_error_norm + epsilon)
```

This ratio is empirical and workload-specific. It is not a mathematical guarantee.

### Propagated risk

For a candidate quantization change at group `g`, Atlas may estimate:

```text
propagated_risk(g) =
    local_error(g)
    * measured_amplification_to_output(g)
    * low_margin_penalty(g)
```

A more conservative implementation can use the maximum observed amplification across calibration samples rather than the mean.

The final ranking can become:

```text
priority = expected_benefit / (local_risk + propagated_risk + epsilon)
```

This remains only a search heuristic.

Exact complete-model token and EOS parity remains mandatory.

### How to measure propagation cheaply

Do not replay every possible downstream path for every tensor candidate.

Recommended first implementation:

1. Run the baseline and store bounded checkpoints at selected graph nodes.
2. Apply one candidate perturbation at one executable group.
3. Resume or replay only from that group to a small set of downstream checkpoints.
4. Compare candidate and baseline states at:
   - the next residual boundary;
   - the end of the current layer;
   - one later layer boundary;
   - final hidden state;
   - output logits.
5. Accumulate amplification statistics.

If partial replay is not yet supported, run a short complete forward pass and collect the same checkpoints. The architecture should still use the graph abstraction so partial replay can be added later.

### Recommended graph data structures

```rust
pub struct ErrorPropagationGraph {
    pub schema_version: u32,
    pub nodes: Vec<ErrorNode>,
    pub edges: Vec<ErrorEdge>,
    pub calibration_corpus_id: String,
    pub sampling_policy_id: String,
}

pub struct ErrorNode {
    pub node_id: String,
    pub kind: ErrorNodeKind,
    pub layer_index: Option<u32>,
    pub executable_group_id: Option<String>,
}

pub struct ErrorEdge {
    pub from_node: String,
    pub to_node: String,
    pub measurements: ErrorPropagationMeasurement,
}

pub struct ErrorPropagationMeasurement {
    pub sample_count: usize,
    pub mean_amplification_ratio: f64,
    pub max_amplification_ratio: f64,
    pub mean_cosine_distance_delta: f64,
    pub max_residual_relative_error: f64,
    pub max_logit_delta: Option<f64>,
    pub minimum_margin_safety_ratio: Option<f64>,
    pub local_top1_changes: u32,
}
```

### Important limitation

The graph helps Atlas choose which candidates to test and which combinations are risky.

It must not be used to claim parity without actual generation.

A candidate with a low propagated-risk score can still fail exact greedy parity after a long decode window and must then be rejected.

---

## 2. The reusable preflight artifact

The output of preflight should be a versioned JSON file, for example:

```text
<model-path>.atlas-plan.json
```

or:

```text
plans/<model-sha>/<hardware-id>/resident-quantization-plan.json
```

Recommended command:

```text
atlas model preflight \
  --model /path/to/model.gguf \
  --output /path/to/model.atlas-plan.json
```

Recommended runtime command:

```text
atlas run \
  --model /path/to/model.gguf \
  --plan /path/to/model.atlas-plan.json
```

Auto-discovery may also look beside the GGUF:

```text
model.gguf
model.atlas-plan.json
```

The explicit `--plan` argument should take precedence over auto-discovery.

---

## 3. Artifact responsibilities

The plan must tell Atlas exactly how to execute the preflighted model.

It should include:

```text
model identity
hardware identity
Atlas build identity
Metal library and kernel-registry identity
executor configuration
selected format per executable group
selected physical layout per group
selected Metal kernel per group
conversion method per group
buffer-sharing rules
dependencies and compatibility rules
KV-cache configuration
prefill configuration
measured memory and throughput
parity evidence
error-propagation evidence
rejected alternatives
safe baseline description
```

The artifact should contain decisions, not raw activation samples.

Large calibration activations and temporary candidate buffers must not be stored in the production plan.

---

## 4. Artifact states

Use an explicit state machine:

```text
building
validated
ready
rejected
invalidated
```

Only `ready` plans may be consumed by normal inference.

A file left in `building` state after interruption must never be used.

Recommended write flow:

```text
1. Write temporary artifact with state=building.
2. Flush the file.
3. Re-read and validate it.
4. Change state to ready in the final representation.
5. Write final temporary file.
6. Flush and atomically rename to the requested path.
```

Normal inference should ignore temporary files.

---

## 5. Proposed top-level JSON schema

```json
{
  "schema_version": 1,
  "artifact_type": "atlas_resident_quantization_plan",
  "state": "ready",
  "plan_id": "sha256:...",
  "created_at": "...",

  "identity": {},
  "executor": {},
  "calibration": {},
  "baseline": {},
  "selected_plan": {},
  "physical_buffers": [],
  "groups": [],
  "dependencies": [],
  "error_propagation_graph": {},
  "measurements": {},
  "parity_evidence": {},
  "search_summary": {},
  "rejections": [],
  "artifact_digest_sha256": "..."
}
```

---

## 6. Identity contract

The artifact must be bound to the exact model and relevant runtime environment.

Recommended identity fields:

```json
{
  "model": {
    "sha256": "...",
    "gguf_metadata_sha256": "...",
    "tensor_manifest_sha256": "...",
    "architecture": "gemma4",
    "model_id": "..."
  },
  "hardware": {
    "device_name": "...",
    "registry_id": 0,
    "chip_identifier": "...",
    "gpu_families": [],
    "unified_memory_bytes": 0
  },
  "software": {
    "atlas_version": "...",
    "atlas_git_commit": "...",
    "build_profile": "release",
    "feature_flags": [],
    "metal_library_sha256": "...",
    "kernel_registry_sha256": "...",
    "converter_registry_sha256": "..."
  }
}
```

### Hard invalidation fields

At minimum invalidate the artifact when any of these differ:

```text
model SHA-256
tensor manifest
model architecture
Apple GPU registry/chip identity
Metal kernel registry hash
Metal library hash
conversion method version
artifact schema version
Resident executor compatibility version
KV-cache configuration when included in the plan
```

Atlas version may be handled through a dedicated compatibility version rather than invalidating for every source commit, but the first implementation should be conservative.

---

## 7. Selected executable plan

Each group entry should tell runtime how to materialize and bind that group.

```json
{
  "group_id": "layer.12.ffn.gate_up",
  "group_kind": "ffn_gate_up",
  "layer_index": 12,
  "tensor_names": ["...", "..."],

  "source_format": "q6_k",
  "selected_format": "q4_0",
  "selected_layout": "q4_0_16row",
  "selected_kernel": "gemma4_gate_up_q4_0_fused",

  "conversion": {
    "method": "q6_k_to_q4_0_v1",
    "method_version": 1,
    "deterministic": true,
    "output_sha256": "...",
    "converted_bytes": 0
  },

  "physical_buffer_id": "buffer:layer12:gate_up:q4",
  "dependencies": [],
  "fallback_group_id": "baseline:layer.12.ffn.gate_up"
}
```

### Baseline entries

Even when a candidate wins, store the safe baseline selection for every group.

This allows Atlas to:

- explain the change;
- diagnose failures;
- reconstruct the baseline for verification;
- fall back before inference if candidate materialization fails.

The normal runtime must not silently switch individual groups after generation has started.

Fallback must happen during plan loading and model construction.

---

## 8. Physical buffer plan

The plan must distinguish executable groups from physical storage.

This is especially important for tied embedding/output weights.

```json
{
  "buffer_id": "buffer:vocabulary:q6_shared",
  "format": "q6_k",
  "layout": "gguf_native",
  "byte_length": 0,
  "source": "gguf",
  "shared_by_groups": [
    "vocabulary.token_embedding",
    "vocabulary.output_projection"
  ]
}
```

A converted buffer may instead use:

```json
{
  "buffer_id": "buffer:vocabulary:q4_output",
  "format": "q4_0",
  "layout": "lm_head_q4_0_16row",
  "byte_length": 0,
  "source": "runtime_conversion",
  "conversion_method": "q6_k_to_q4_0_v1",
  "content_sha256": "...",
  "shared_by_groups": ["vocabulary.output_projection"]
}
```

This prevents the runtime from guessing whether two groups share one representation.

---

## 9. Conversion materialization policy

There are two possible artifact models.

### Decision-only artifact

The JSON stores the selected conversion method but not converted bytes.

At model load, Atlas:

```text
reads source GGUF tensor
performs deterministic conversion
verifies output checksum
creates Resident Metal buffer
binds selected kernel
```

Advantages:

```text
small artifact
portable beside the original GGUF
simple invalidation
```

Disadvantage:

```text
conversion cost is paid on each process start
```

### Artifact plus converted-weight cache

The JSON references a separate binary cache:

```text
model.atlas-plan.json
model.atlas-weights.bin
```

The binary file contains converted tensor/group representations.

At model load, Atlas verifies checksums and uploads the already converted bytes.

Advantages:

```text
lower startup cost
no repeated CPU conversion
exact reproduction of preflighted bytes
```

Disadvantages:

```text
larger disk usage
more complex atomic writes and invalidation
```

### Recommendation

Implement in two stages:

```text
Stage 1: decision-only JSON plan
Stage 2: optional converted-weight binary cache
```

The JSON schema should support both from the beginning with a field such as:

```json
{
  "materialization": {
    "kind": "runtime_conversion",
    "external_file": null,
    "offset": null,
    "length": null,
    "sha256": "..."
  }
}
```

or:

```json
{
  "materialization": {
    "kind": "external_binary",
    "external_file": "model.atlas-weights.bin",
    "offset": 1048576,
    "length": 524288,
    "sha256": "..."
  }
}
```

---

## 10. Runtime plan-loading workflow

Normal Atlas inference should use the following workflow:

```text
load GGUF header and tensor manifest
locate explicit or adjacent plan artifact
parse plan schema
require state == ready
validate artifact digest
validate model identity
validate hardware identity
validate executor compatibility
validate kernel registry and converter versions
validate every group and dependency
validate every referenced physical buffer
materialize or load converted representations
verify converted-content checksums
create Resident Metal buffers
resolve and bind selected kernels
run a short construction proof if configured
begin normal inference
```

The construction proof should verify execution plumbing, not repeat the full preflight.

Possible checks:

```text
all required Resident kernels resolved
all group bindings complete
expected Resident bytes allocated
no Reference/CPU inference fallback available
no missing converted buffer
no dependency conflict
```

---

## 11. Runtime failure behavior

### Explicit plan

When the user passes:

```text
--plan /path/to/plan.json
```

and validation fails, Atlas should fail loudly by default.

It should explain the exact reason:

```text
model hash mismatch
hardware mismatch
kernel registry mismatch
unsupported schema
missing conversion method
converted checksum mismatch
memory allocation failure
```

An explicit plan should not be silently ignored.

### Auto-discovered plan

When an adjacent plan is auto-discovered and invalid:

```text
1. Record the invalidation reason.
2. Do not use any selection from the invalid plan.
3. Use the safe source-model baseline, or run preflight when policy=auto.
4. Never mix a partially valid cached plan with guessed runtime decisions.
```

Recommended policies:

```text
--preflight-policy off
--preflight-policy use-plan
--preflight-policy auto
--preflight-policy force
```

Semantics:

```text
off       ignore plan search; use safe baseline
use-plan  require a valid plan; fail if absent/invalid
auto      use valid plan, otherwise run preflight and write one
force     rerun preflight even when a valid plan exists
```

---

## 12. Plan application must be deterministic

Given the same:

```text
model bytes
plan artifact
hardware identity
Atlas compatibility version
Metal library
```

Atlas should resolve the same:

```text
selected group formats
physical buffers
converted bytes
kernels
layouts
dispatch topology
memory accounting
```

Plan application must not rerun the search algorithm.

The only permitted work is validation, deterministic materialization, and Resident executor construction.

---

## 13. Artifact Rust structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasQuantizationArtifact {
    pub schema_version: u32,
    pub artifact_type: String,
    pub state: ArtifactState,
    pub plan_id: String,

    pub identity: ArtifactIdentity,
    pub executor: ExecutorPlanIdentity,
    pub calibration: CalibrationSummary,

    pub baseline: ExecutablePlanDescription,
    pub selected_plan: ExecutablePlanDescription,

    pub physical_buffers: Vec<PhysicalBufferPlan>,
    pub groups: Vec<ExecutableGroupPlan>,
    pub dependencies: Vec<PlanDependency>,

    pub error_propagation_graph: Option<ErrorPropagationGraph>,
    pub measurements: MeasurementSet,
    pub parity_evidence: ParityEvidenceSet,
    pub search_summary: SearchSummary,
    pub rejections: Vec<CandidateRejection>,

    pub artifact_digest_sha256: String,
}
```

Plan loading:

```rust
pub trait QuantizationArtifactLoader {
    fn load(path: &Path) -> Result<AtlasQuantizationArtifact>;
    fn validate(
        artifact: &AtlasQuantizationArtifact,
        context: &RuntimeIdentity,
    ) -> Result<ValidatedQuantizationArtifact>;
}
```

Plan application:

```rust
pub trait QuantizationPlanMaterializer {
    fn materialize(
        artifact: &ValidatedQuantizationArtifact,
        model: &GgufModel,
        runtime: &MetalRuntime,
    ) -> Result<MaterializedExecutionPlan>;
}
```

The executor should accept only a validated, materialized plan:

```rust
pub fn new_with_plan(
    model: Arc<Gemma4E2bModel>,
    runtime: Arc<MetalRuntime>,
    plan: MaterializedExecutionPlan,
) -> Result<Gemma4E2bExecutor>;
```

---

## 14. CLI proposal

### Generate or refresh artifact

```text
atlas model quantization-preflight \
  --model model.gguf \
  --output model.atlas-plan.json
```

Optional:

```text
--weights-cache model.atlas-weights.bin
--beam-width 3
--max-candidates 32
--max-preflight-seconds 300
--calibration-corpus atlas-default-v1
```

### Validate artifact without inference

```text
atlas model quantization-plan \
  --model model.gguf \
  --plan model.atlas-plan.json \
  --validate
```

### Run using the artifact

```text
atlas run \
  --model model.gguf \
  --plan model.atlas-plan.json
```

### Automatic use

```text
atlas run \
  --model model.gguf \
  --preflight-policy auto
```

This should:

```text
use an adjacent valid plan when available
otherwise run preflight once
write the plan
use the selected plan
```

---

## 15. Minimal artifact required for the first implementation

The first production artifact does not need every future feature.

Minimum fields:

```text
schema version
state=ready
model SHA-256
tensor manifest hash
hardware registry/chip identity
Atlas/Metal compatibility hashes
baseline plan
selected group format per executable group
selected kernel per group
conversion method per converted group
physical-buffer sharing map
KV-cache and executor configuration
Resident memory measurements
throughput measurements
prompt/generated/measured-token hashes
EOS positions and finish reasons
rejection reasons
artifact digest
```

The error-propagation graph may initially be stored as optional diagnostic evidence.

It should influence search ranking but not be required by the normal executor after the selected plan has already been validated.

---

## 16. Implementation order

### Step 1: Plan loading before advanced search

First ensure Atlas can:

```text
write the existing mixed/all-Q4 preflight decision as an artifact
load it on a second run
validate model/hardware/build identity
reconstruct selected Resident group bindings
skip preflight search
run exact-parity inference
```

This proves the complete artifact lifecycle.

### Step 2: Per-group plan schema

Generalize the artifact from one global weight format to per-executable-group selections.

### Step 3: Sensitivity and error graph

Add:

```text
local activation-weighted sensitivity
residual-aware error
selected downstream checkpoints
amplification edges
logit-margin propagation
```

Use the graph to rank candidate groups.

### Step 4: Guided search

Add top-N candidate testing, small beam search, and interaction moves.

### Step 5: Optional converted-weight binary cache

Avoid repeated conversion on subsequent process starts.

---

## 17. Acceptance tests

Required tests include:

```text
preflight writes a ready artifact
second run loads artifact and skips search
loaded plan resolves the same formats and kernels
model hash mismatch rejects artifact
hardware mismatch rejects artifact
kernel-registry mismatch rejects artifact
unsupported schema rejects artifact
explicit invalid plan fails loudly
auto-discovered invalid plan falls back according to policy
partial/building artifact is never consumed
converted tensor checksum is verified
selected physical buffer sharing is reproduced
no Reference or CPU inference fallback occurs
loaded-plan generation matches the preflight winner exactly
loaded-plan EOS matches the preflight winner exactly
loaded-plan throughput remains within acceptance tolerance
baseline artifact works when every candidate is rejected
```

For error propagation:

```text
zero perturbation produces zero measured propagation error
known injected perturbation appears at downstream checkpoints
amplification measurements are deterministic within tolerance
output-projection perturbation records logit-margin risk
high-risk known Q4 vocabulary candidate is ranked conservatively
error graph never overrides failed exact parity
```

---

## Final requirement

The preflight result must be reusable, not ephemeral.

The final architecture is:

```text
model.gguf
    +
model.atlas-plan.json
    + optional
model.atlas-weights.bin
    -> validated Resident Metal execution plan
```

The JSON artifact is the contract between preflight and future Atlas runs.

It says exactly:

```text
which groups use which formats
which physical representations exist
which kernels execute them
how converted weights are reconstructed or loaded
which dependencies must hold
which hardware and model identities are valid
which parity and performance evidence justified promotion
```

The governing rule is:

> Preflight searches and proves the plan once. Subsequent Atlas runs validate and apply that plan; they do not rediscover it.
