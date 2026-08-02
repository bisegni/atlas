# Sensitivity-Guided Quantization Preflight Strategy

## Purpose

This document defines the recommended next step for Atlas quantization preflight.

Atlas already has a safe baseline flow:

- the GGUF model is already quantized;
- the current baseline is mixed Q4/Q6;
- Resident Metal runs both baseline and candidate inference;
- candidates are promoted only with exact greedy-token and EOS parity;
- faster candidates that fail parity are rejected;
- a valid cached plan is written;
- the mixed Q4/Q6 baseline remains available.

The next goal is not full input-dependent or per-token dynamic quantization.

The recommended goal is:

> Use a short Resident preflight calibration pass to estimate which executable tensor groups are numerically insensitive, rank the most promising format changes, test only those candidates end to end, and cache the best exact-parity hardware-specific plan.

This is sensitivity-guided hardware autotuning of static weight formats.

---

## Core idea

Atlas should not try every possible quantization combination blindly.

For each executable tensor group, estimate two quantities:

1. **Numerical risk**: how much changing the weight format perturbs the outputs produced by that group for representative Resident inference inputs.
2. **Expected benefit**: how much the change may improve complete-model throughput, memory residency, upload cost, or kernel fusion.

Candidates should be ranked approximately as:

```text
priority = expected_benefit / (numerical_risk + epsilon)
```

This ranking is only a search heuristic.

The final promotion decision must still require:

- real complete-model Resident Metal inference;
- exact prompt-token parity;
- exact generated-token parity;
- exact measured-window parity;
- exact EOS position and finish-reason parity;
- acceptable residency, memory, upload, and readback behavior;
- a statistically credible performance improvement.

A sensitivity score must never replace end-to-end parity validation.

---

## Recommended selectable units

Do not begin with arbitrary individual tensors.

Use executable tensor groups that match actual Metal operations and possible fused dispatches.

Recommended initial groups:

```text
vocabulary.token_embedding
vocabulary.output_projection
layer[N].attention.qkv
layer[N].ffn.gate_up
layer[N].ffn.down
```

Keep normalization tensors, control tensors, rotary parameters, and other F16/F32 tensors fixed initially.

### Why groups instead of individual tensors

Individual GGUF tensors are not always independent execution units.

Examples:

- Q, K, and V often share the same input and may use a fused dispatch.
- Gate and up projections share the same input and may use one fused kernel.
- Embedding and output projection may share source storage but require different executable layouts.
- A locally faster tensor representation may require an extra buffer or prevent a faster fused path.

The optimizer should therefore operate on executable groups while the cached plan may still record every underlying tensor.

---

## Phase 1: Instrumented baseline pass

Run the safe mixed Q4/Q6 baseline once with lightweight instrumentation.

Use a small fixed calibration corpus containing at least:

- a short instruction prompt;
- a medium conversational prompt;
- a long-context prompt;
- a vocabulary-stress prompt with numbers, punctuation, Unicode, and code-like tokens;
- an EOS-sensitive prompt;
- a fixed-window prompt that does not terminate early.

During the baseline pass, sample the inputs to the selectable groups:

```text
QKV input hidden states
Gate/up input hidden states
FFN-down input hidden states
Hidden states entering output projection
Token IDs used for embedding lookup
Residual-stream norms near each sampled group
Baseline top-1 and top-2 logit margins
```

Do not store all activations.

Use bounded sampling, for example:

- a fixed maximum number of tokens per workload;
- evenly spaced token positions;
- a few early, middle, and late decode positions;
- a deterministic sampling seed;
- optional extra samples at low-logit-margin positions.

The instrumentation must not change the production numerical path.

---

## Phase 2: Local sensitivity estimation

For each legal group-format alternative, temporarily materialize the converted weights and compare the baseline group output with the candidate group output on the sampled baseline inputs.

For a group with baseline operation:

```text
y = W(x)
```

and candidate representation:

```text
y_q = W_q(x)
```

measure a normalized reconstruction error such as:

```text
relative_output_mse = sum(||y_q - y||^2) / (sum(||y||^2) + epsilon)
```

Also record:

```text
maximum_relative_error
mean_cosine_distance
maximum_cosine_distance
error_relative_to_residual_norm
sample_count
```

The output error is more useful than raw weight error because it considers the activation distribution actually seen by the model.

### Residual-aware risk

For projection outputs that are added to the residual stream, calculate:

```text
residual_relative_error = ||y_q - y|| / (||residual|| + epsilon)
```

A similar local output error may be harmless when the residual norm is large and dangerous when the residual norm is small.

### Position weighting

Do not assume that every layer has equal sensitivity.

Record group position and optionally apply a conservative weighting for groups close to the output projection, but use measured evidence rather than a fixed rule such as “all final layers must remain high precision.”

---

## Special handling for vocabulary tensors

The vocabulary groups need stronger diagnostics because they directly affect token identity.

### Token embedding

For the token IDs present in the calibration corpus, compare baseline and candidate embedding vectors.

Record:

```text
mean_embedding_error
maximum_embedding_error
frequency_weighted_error
worst_token_ids
```

Embedding and output projection must be treated as separate executable uses even when they share the same source tensor.

### Output projection

For every sampled hidden state, calculate baseline and candidate logits.

Record:

```text
baseline_top1_token
baseline_top2_token
baseline_top1_top2_margin
candidate_top1_token
candidate_top1_top2_margin
maximum_logit_delta
delta_for_baseline_top1
delta_for_baseline_top2
whether_local_top1_changed
```

A useful risk indicator is the relationship between logit perturbation and the baseline top-1/top-2 margin:

```text
margin_safety_ratio = baseline_margin / (2 * maximum_logit_delta + epsilon)
```

Interpretation:

```text
large ratio      lower local token-selection risk
ratio near 1     risky
ratio below 1    perturbation may exceed the decision margin
```

This remains a heuristic, not a proof of global token parity.

The current Q6-to-Q4 vocabulary candidate should likely score as high risk because it already produces generated-token and EOS divergence.

---

## Phase 3: Benefit estimation

For each group-format-kernel alternative, estimate the possible benefit.

Record at least:

```text
source_bytes
converted_bytes
resident_bytes_saved
additional_resident_bytes
expected_memory_bandwidth_reduction
compatible_kernel
possible_fused_dispatch
expected_dispatch_count_change
microbenchmark_speedup, if available
```

Isolated kernel timing may be used for ranking or pruning only.

It must not be used for promotion because complete-model performance also depends on:

- dispatch overhead;
- command-buffer structure;
- buffer sharing;
- memory pressure;
- residency;
- lost or gained fusion;
- prefill versus decode behavior.

A simple first benefit score can be:

```text
benefit_score =
    estimated_decode_speedup
  + weighted_memory_saving
  + fusion_bonus
  - additional_buffer_penalty
  - additional_dispatch_penalty
```

Keep the initial score simple and explainable.

---

## Phase 4: Candidate ranking

Define an opportunity record:

```rust
struct QuantizationOpportunity {
    group_id: String,
    source_format: QuantFormat,
    candidate_format: QuantFormat,
    candidate_kernel: String,
    conversion_method: String,

    relative_output_mse: f64,
    max_relative_error: f64,
    cosine_distance: f64,
    residual_relative_error: f64,
    local_top1_changes: u32,
    minimum_margin_safety_ratio: Option<f64>,

    estimated_speedup: f64,
    resident_bytes_saved: i64,
    fusion_bonus: f64,

    risk_score: f64,
    benefit_score: f64,
    priority_score: f64,
}
```

A simple risk score can combine normalized metrics:

```text
risk_score =
    w1 * relative_output_mse
  + w2 * max_relative_error
  + w3 * residual_relative_error
  + w4 * cosine_distance
  + vocabulary_risk_penalty
  + local_top1_change_penalty
```

Then:

```text
priority_score = benefit_score / (risk_score + epsilon)
```

Do not overfit the score initially.

Its job is only to decide which candidates deserve real end-to-end testing first.

---

## Candidate classes

Classify each opportunity into one of three groups.

### Green

Characteristics:

```text
very low activation-weighted error
no local top-1 logit changes
good expected speed or memory benefit
no dependency conflict
no additional high-cost buffer
```

These candidates are tested first.

### Yellow

Characteristics:

```text
moderate error
small margins at some sampled positions
meaningful expected performance benefit
possible interaction with another group
```

These candidates are tested after green candidates and require longer validation.

### Red

Characteristics:

```text
local top-1 changes
large error relative to residual norm
logit perturbation comparable to decision margin
little performance benefit
unsupported or expensive dependency
```

Red candidates are normally skipped in automatic production preflight.

They may remain available behind an explicit experimental option.

---

## Phase 5: End-to-end search

Use the sensitivity ranking to reduce the search space, but validate complete plans.

Recommended first algorithm:

> Baseline-anchored coordinate search with a beam width of 2 or 3 and explicit compound moves for known interacting groups.

### Atomic moves

Examples:

```text
one QKV group from source format to Q4
one gate/up group from source format to Q4
one FFN-down group from source format to Q4
one output-projection kernel/layout change
```

### Compound moves

Explicitly test interactions such as:

```text
token embedding + output projection
Q + K + V as one group
gate + up as one group
gate/up + FFN down within one layer
shared vocabulary buffer layout alternatives
```

### Search workflow

```text
1. Start with the mixed Q4/Q6 baseline.
2. Rank legal opportunities using sensitivity and expected benefit.
3. Test green candidates with short complete-model Resident validation.
4. Reject immediately on any token or EOS divergence.
5. Keep the best 2 or 3 parity-valid complete plans.
6. Expand those plans with the next compatible opportunities.
7. Test explicit compound moves for known interactions.
8. Stop when the candidate, memory, or wall-time budget is reached.
9. Run full long-window validation only on the best finalists.
10. Cache the fastest fully valid plan or the baseline.
```

### Important rule

Do not permanently update the winner after every local success and discard all alternatives.

Keeping a small beam is important because:

- two changes may enable a fused kernel;
- a local speedup may become a regression after another buffer is added;
- a slightly slower intermediate plan may lead to a faster valid combination.

---

## Pseudocode

```text
baseline = run_instrumented_resident_baseline(calibration_prompts)
require baseline is valid Resident Metal execution

opportunities = []

for group in executable_groups:
    for alternative in legal_alternatives(group):
        candidate_weights = materialize_temporarily(alternative)

        sensitivity = measure_group_sensitivity(
            baseline.activation_samples[group],
            baseline.residual_samples[group],
            candidate_weights
        )

        if group is output_projection:
            sensitivity += measure_logit_margin_risk(
                baseline.output_hidden_samples,
                candidate_weights
            )

        benefit = estimate_benefit(group, alternative)

        opportunities.push(
            rank(group, alternative, sensitivity, benefit)
        )

        release(candidate_weights)

opportunities.sort_by_priority_descending()

beam = [baseline.plan]
incumbent = baseline.plan

for opportunity in opportunities:
    if opportunity.class == RED:
        continue

    expanded = []

    for plan in beam:
        if not compatible(plan, opportunity):
            continue

        candidate = apply(plan, opportunity)
        result = run_short_complete_resident_validation(candidate)

        if result.exact_token_parity
           and result.exact_eos_parity
           and result.residency_valid
           and result.memory_valid:
            expanded.push(candidate, result)
        else:
            record_rejection(candidate, result)
            release_candidate_buffers(candidate)

    beam = keep_best_complete_plans(
        baseline,
        incumbent,
        beam + expanded,
        width = 3
    )

    incumbent = best_parity_valid_plan(beam)

    if search_budget_exhausted:
        break

finalists = best_non_baseline_plans(beam, maximum = 2)

for finalist in finalists:
    result = run_full_resident_validation(
        candidate = finalist,
        matched_baseline = baseline,
        short_medium_long_and_eos_workloads
    )

    if result.all_hard_gates_pass
       and result.complete_model_speedup >= promotion_threshold:
        valid_finalists.push(finalist, result)
    else:
        record_rejection(finalist, result)
        release_candidate_buffers(finalist)

winner = best_valid_finalist(valid_finalists)

if winner exists:
    winner = run_final_ablation_cleanup(winner)
    clean_result = rerun_from_clean_state(winner, baseline)

    if clean_result.all_promotion_gates_pass:
        cache_atomically(winner)
        return winner

cache_atomically(baseline.plan)
return baseline.plan
```

---

## Progressive validation windows

Use multiple validation levels to control startup cost.

### Local sensitivity stage

No full generation is required for every alternative.

Use sampled baseline activations to estimate local risk.

### Short screen

Recommended initial values:

```text
8-16 warmup decode tokens
16-32 measured decode tokens
one short prompt
one vocabulary-stress prompt for vocabulary changes
```

Reject immediately on any mismatch.

### Intermediate validation

For candidates surviving the short screen:

```text
64-128 generated tokens
short and medium prompts
one EOS-sensitive prompt when relevant
```

### Full finalist validation

For only the best one or two candidates:

```text
all calibration prompt classes
one 256-token fixed decode
one 512- or 1024-token long continuation for high-risk changes
one EOS-terminated workload
matched prefill and decode configuration
multiple timing repetitions
```

Vocabulary changes should always receive long-window and EOS validation.

---

## Exact promotion gates

A candidate may be promoted only when all conditions pass.

### Parity

```text
prompt token IDs exactly match
generated token IDs exactly match
measured-window token IDs exactly match
EOS presence exactly matches
first EOS position exactly matches
finish reason exactly matches
```

Record hashes and the first divergent token when a failure occurs.

### Resident execution

```text
executor == resident
selected kernels are production Metal kernels
real command buffers were committed
selected dispatches actually executed
no Reference fallback
no CPU inference fallback
```

Metal pipeline creation or compilation alone is not GPU execution evidence.

### Memory and residency

```text
peak resident memory below configured limit
steady-state resident memory recorded
KV-cache bytes matched
weight upload bytes recorded
no unexpected measured-window uploads
readback bytes within expected bounds
rejected candidate buffers released
```

### Performance

Use complete-model timing, not isolated kernel timing.

A reasonable initial rule is:

```text
median decode throughput at least 3% faster
weighted end-to-end score at least 2-3% better
no mandatory workload regression greater than 2%
result confirmed from a clean executor state
```

If the measurements are too noisy, increase repetitions rather than weaken the threshold.

---

## Sensitivity budget

Atlas may use a bounded cumulative risk score to prune combinations:

```text
sum(group_risk * group_weight) <= search_risk_budget
```

This can prevent combining many individually small perturbations into one high-risk plan.

However, local errors are not truly additive.

The risk budget is only a search heuristic and must never replace complete-model parity validation.

---

## Cached plan additions

Extend the quantization-plan sidecar with per-group sensitivity and search information.

Recommended fields:

```json
{
  "group_id": "layer.12.ffn.gate_up",
  "source_format": "q6_k",
  "selected_format": "q4_0",
  "selected_kernel": "...",
  "conversion_method": "q6_k_to_q4_0_v1",
  "sensitivity": {
    "relative_output_mse": 0.0,
    "max_relative_error": 0.0,
    "cosine_distance": 0.0,
    "residual_relative_error": 0.0,
    "local_top1_changes": 0,
    "minimum_margin_safety_ratio": null,
    "sample_count": 0
  },
  "estimated_benefit": {
    "estimated_speedup_percent": 0.0,
    "resident_bytes_saved": 0,
    "fusion_bonus": 0.0
  },
  "measured_result": {
    "decode_speedup_percent": 0.0,
    "resident_bytes": 0,
    "parity_passed": true
  }
}
```

Also store:

```text
calibration corpus version
activation sampling policy version
sensitivity algorithm version
risk-score weights
candidate budget
beam width
search stop reason
rejected candidate reasons
model and hardware identity
Atlas commit and Metal library hash
```

Any change to the sensitivity algorithm, conversion method, kernel registry, model hash, or relevant hardware identity should invalidate the plan or its search evidence.

---

## Rust implementation outline

### New module

Suggested module:

```text
atlas-model/src/gemma4_quantization_sensitivity.rs
```

Possible responsibilities:

```text
activation sample collection
local group replay
output-error calculation
residual-relative error
vocabulary embedding analysis
logit-margin analysis
risk scoring
opportunity ranking
```

### Core structures

```rust
pub struct ActivationSampleSet {
    pub group_id: String,
    pub samples: Vec<ActivationSample>,
}

pub struct ActivationSample {
    pub workload_id: String,
    pub token_position: usize,
    pub input: Vec<f32>,
    pub baseline_output: Option<Vec<f32>>,
    pub residual_norm: Option<f32>,
}

pub struct GroupSensitivityReport {
    pub group_id: String,
    pub source_format: QuantFormat,
    pub candidate_format: QuantFormat,
    pub candidate_kernel: String,
    pub relative_output_mse: f64,
    pub max_relative_error: f64,
    pub cosine_distance: f64,
    pub residual_relative_error: f64,
    pub local_top1_changes: u32,
    pub minimum_margin_safety_ratio: Option<f64>,
    pub sample_count: usize,
    pub risk_score: f64,
}
```

### Avoid large host storage

Prefer streaming accumulation of statistics.

Do not retain all outputs when only aggregate metrics are needed.

For example:

```rust
pub struct ErrorAccumulator {
    squared_error_sum: f64,
    baseline_energy_sum: f64,
    max_relative_error: f64,
    cosine_distance_sum: f64,
    sample_count: usize,
}
```

For output projection, storing a bounded number of hidden-state samples is acceptable, but avoid retaining complete logits for every token.

### Candidate resource lifetime

Every temporary converted buffer must be owned by an RAII candidate handle.

On rejection or after local sensitivity measurement:

```text
release candidate-only Metal buffers
release pipeline references not shared by the baseline
remove candidate residency accounting
confirm no candidate buffer remains selected
```

---

## Recommended implementation sequence

### Step 1: Group inventory

- Formalize executable group IDs.
- Map Gemma tensors to QKV, gate/up, FFN-down, embedding, and output groups.
- Represent tied vocabulary storage explicitly.
- Reproduce the current mixed-Q4/Q6 and all-Q4 plans using these groups.

### Step 2: Baseline activation sampler

- Add bounded deterministic sampling.
- Collect QKV, gate/up, down, and output-projection inputs.
- Record residual norms and top-1/top-2 margins.
- Confirm instrumentation does not alter generated tokens.

### Step 3: Local sensitivity evaluator

- Replay candidate group operations on sampled inputs.
- Calculate output reconstruction error.
- Add output-projection logit-margin analysis.
- Emit a diagnostic report only; do not change automatic selection yet.

### Step 4: Opportunity ranking

- Add simple risk and benefit scores.
- Classify candidates as green, yellow, or red.
- Record why candidates were ranked or pruned.

### Step 5: End-to-end top-N testing

- Test only the highest-ranked green opportunities.
- Use short complete-model Resident validation.
- Keep a beam width of 2 or 3.
- Preserve exact parity gates.

### Step 6: Compound interaction testing

- Test embedding/output pairs.
- Test QKV groups.
- Test gate/up groups and optional gate/up + down combinations.

### Step 7: Full validation and plan cache

- Run long-window and EOS tests for finalists.
- Confirm performance from a clean state.
- Cache the winner or the baseline.
- Store sensitivity and rejection evidence.

---

## Features that should not be implemented yet

Do not include these in the first production version:

```text
per-token precision changes
input-dependent format switching during decode
runtime activation quantization
arbitrary individual-tensor search
simultaneous tuning of KV-cache format
simultaneous tuning of attention algorithm
simultaneous tuning of prefill topology
Bayesian optimization
multi-armed-bandit online exploration
promotion based on approximate token similarity
promotion of late-diverging candidates
expansion from parity-failing states to search for numerical cancellation
automatic quantization of normalization/control tensors
cross-device reuse of cached plans
```

The first version should hold the executor configuration fixed and only search a small, explicit set of weight-group format and kernel alternatives.

---

## Final recommendation

The next Atlas implementation should be divided into two clear phases.

### Phase 12.2a: Sensitivity analysis only

Implement:

```text
baseline activation sampling
per-group activation-weighted output error
residual-relative error
embedding error analysis
output-projection logit-margin analysis
benefit estimation
ranked opportunity report
```

Do not automatically promote new per-group plans yet.

Use this phase to verify that the analysis correctly identifies the known Q4 vocabulary conversion as risky.

### Phase 12.2b: Guided per-group autotuning

Implement:

```text
top-N green candidate evaluation
small beam search
explicit interaction moves
short then long Resident validation
exact token/EOS gates
complete-model performance gates
atomic plan caching
baseline fallback
```

The governing principle is:

> Sensitivity analysis decides what Atlas should try. Only complete Resident Metal inference with exact token and EOS parity decides what Atlas may promote.
