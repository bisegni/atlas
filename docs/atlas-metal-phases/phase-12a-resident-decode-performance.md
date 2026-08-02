# Phase 12.1: Resident decode performance remediation

## Outcome

Atlas identifies and improves the measured GPU-resident Gemma 4 E2B Q4_0
decode bottleneck on Apple Silicon without weakening token parity, residency,
or normal generation semantics. The current work starts with growing-context
instrumentation before promoting an attention-kernel change.

## Baseline

The normal user-facing workload remains one-shot resident chat. Diagnostic
profiling and fixed decode windows are separate so they cannot contaminate
normal chat metrics:

```zsh
cargo run --release -p atlas-cli -- profile --model gemma4-e2b-q4_0 --kv-cache-type q4_0 --decode-tokens 1024 --max-context 2048

cargo run --release -p atlas-cli -- benchmark --model gemma4-e2b-q4_0 --kv-cache-type q4_0 --prompt 'Explain Resident decode.' --warmup-decode-tokens 1024 --decode-tokens 512 --max-context 2048
```

Gemma CLI commands now default to the promoted Q4_0 KV cache. This selects the
default no-value-barrier Q4 two-pass attention scan once the context reaches
its split threshold; `--kv-cache-type f32` remains the explicit diagnostic
oracle.

Every completed chat turn appends one JSON object to
`artifacts/chat-performance.jsonl`. This append-only artifact is the runtime
evidence path; it records actual generated-token count, EOS/max-token finish
reason, TTFT, prefill/decode throughput, host/GPU time, command buffers,
upload/readback bytes, and resident bytes.

## Work

- Keep performance measurement in normal resident `chat`. The diagnostic-only
  `profile` and fixed-window `benchmark` commands are permitted solely to
  identify long-context decode bottlenecks; Q8 parity and golden checks remain
  Rust fixture acceptance coverage rather than public runtime commands.
- Profile the same Resident Q4_0 decode path at 128, 256, 512, 1K, 2K, and 4K
  generated-token ordinals. Compare attention candidates using a deterministic
  warm-up window followed by a separately timed decode window, exact full and
  measured token hashes, first-EOS parity, stable KV residency, and no
  short-context throughput regression. Run
  `scripts/run-gemma4-decode-attention-ab.sh` to compare the production
  64-key four-way context split with an experiment-only one-SIMD Q4 scan. It
  preserves the production kernel's four interleaved partial dot products and
  their reduction order, but removes the per-key threadgroup barriers from the
  first pass. The cross-head shared-KV experiment is retained only as rejected
  evidence: its synchronization cost cut long-context throughput roughly in
  half despite exact output parity. The rejected context eight-way split
  changed the reduction order and therefore token selection; it is not
  selectable or promotable.
- The diagnostic profile labels the two-pass attention scan and final combine
  independently as `gemma_attention_split_scan` and
  `gemma_attention_split_combine`. Choose the next implementation experiment
  from the larger measured component; those labels do not alter production
  dispatches or generation semantics.
- Remove the measured dominant host/command-buffer costs without adding a CPU
  fallback: retain resident weights, KV, activations, and token selection;
  preserve the single decode command-buffer boundary and token-only default
  readback.
- Improve the measured dominant packed-kernel path (Q4_0/Q8_0 matvec or its
  surrounding layout/dispatch) using Apple-GPU occupancy and memory-access
  evidence. Do not dequantize a full weight matrix or trade resident bytes for
  an undisclosed FP32 cache.
- Profile Gemma decode projections as separate Q4 QKV, Q4 FFN gate/up, Q6
  language-model-head, and remaining projection families. Split normalization
  and positional work into RMS normalization, fused Q/K norm+RoPE, RoPE
  rotation, and RoPE layout conversion before selecting the next kernel target.

The tied vocabulary projection remains on the canonical Q6_K eight-row kernel.
`ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=cacheopt` is an opt-in candidate that
reuses the Q6_K super-block scale across each half-SIMD row without changing
packed-bit decoding or the eight-row reduction order. It is not promoted until
an exact Resident A/B demonstrates a relative long-context improvement with no
short-context regression. `scripts/run-gemma4-q6-lm-head-ab.sh` writes its
screen or five-window promotion evidence under
`artifacts/phase-12a-q6-lm-head-ab/`. The Apple-Silicon screen at
`artifacts/phase-12a-q6-lm-head-ab/20260731T164809Z/q6-lm-head-ab-summary.json`
was exact and accounting-stable but only gained +0.83% long-context decode, so
the candidate remains opt-in. The production Resident
RMS path is `rms_norm_decode_f32_vec4`: it vectorizes the 2304-wide
single-token normalization loads and stores while retaining one 32-lane
reduction and the same resident buffers. The five-window Apple-Silicon gate at
`artifacts/phase-12a-rms-norm-ab/20260729T185547Z/rms-norm-ab-summary.json`
passed exact SHA/EOS parity and stable Q4 KV/Resident accounting, with +8.84%
long-context decode (28.88 to 31.44 tok/s) and +14.02% short-context decode
(42.48 to 48.43 tok/s). Set `ATLAS_GEMMA4_RMS_NORM_EXPERIMENT=baseline` only
to select the scalar RMS diagnostic oracle.

`ATLAS_GEMMA4_WEIGHT_FORMAT=all_q4` is a separate, unpromoted Resident
candidate. At executor setup it deterministically re-quantizes the fixture's
Q6_K `token_embd.weight` and `per_layer_token_embd.weight` tables to Q4_0,
uploads those derived buffers instead of their Q6_K GPU source buffers, and
uses `embedding_lookup_q4_0` plus `matvec_q4_0_16row` for both prefill and
decode. Benchmark records expose `weight_format`,
`embedding_kernel`, and `output_projection_kernel`; an all-Q4 run must report
no Q6 vocabulary projection.

Use `scripts/run-gemma4-decode-attention-ab.sh - ATLAS_GEMMA4_WEIGHT_FORMAT=all_q4`
for the fixed five-window Resident A/B artifact. Promotion requires exact
prompt/full/measured stream SHA and EOS parity, stable Q4 KV and Resident
accounting, all-Q4 kernel selection, at least 3% sustained long-context decode
improvement, and no short-context regression. Until then, mixed Q4/Q6 remains
the production default.

For a faster preflight screen, use two runs with shorter decode windows:

```zsh
bash scripts/run-gemma4-decode-attention-ab.sh --screen \
  - ATLAS_GEMMA4_WEIGHT_FORMAT=all_q4
```

The screen is suitable for rejecting clearly slower candidates. A passing
screen is not promotion evidence; rerun the default five-window command before
recording a plan as ready.

When the all-Q4 candidate fails exact parity, run the short diagnostic
`bash scripts/run-gemma4-q4-vocabulary-diagnosis.sh`. It compares
`q4_embeddings` and `q4_lm_head` against the mixed Resident oracle in two
64-token windows. This localizes the failing vocabulary boundary; it is not a
performance or promotion gate.

Quantization decisions can be recorded in a versioned sidecar next to the
GGUF as `<model>.quantization-plan.json`. Atlas validates the schema, model
SHA, source tensor formats, supported candidate formats, positive benchmark
timings, and parity before accepting the sidecar during Resident executor
construction. Inspect a validated sidecar without requiring Metal with:

```zsh
cargo run -p atlas-cli -- model quantization-plan --model gemma4-e2b-q4_0
```

Profile and rewrite a target/oracle inventory for the upcoming Metal profiling pass with:

```zsh
cargo run -p atlas-cli -- model quantization-plan --profile \
  --model gemma4-e2b-q4_0 \
  --oracle models/gguf/gemma-4-e2b-it-f16/gemma-4-E2B_f16-it.gguf \
  --output artifacts/quantization-plans/gemma4-e2b-q4_0.json
```

Preparation writes a `pending` inventory only; it is not accepted by normal
Resident loading until GPU timings, logit bounds, and exact token parity have
been recorded.

The sidecar currently applies only a validated paired choice for the two Gemma
vocabulary tables (`token_embd.weight` and `per_layer_token_embd.weight`);
arbitrary internal per-tensor format rewrites are not applied until matching
source/oracle fixtures and format-specific Resident kernels exist. A missing
or invalid sidecar leaves the established mixed-Q4/Q6 path unchanged.

The generic 16-row Q4 projection retains the unpromoted shared-input
diagnostic candidate `matvec_q4_0_16row_shared_input`. The broader
`ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT=simdgroup_tiled` candidate applies the same
one-load-per-SIMD-group activation schedule to the generic projection, fused
QKV, fused gate/up, and batched-prefill Q4 projection families. It retains the
established packed Q4 layout, F32 output math, and eight-lane reduction order.
`baseline` (and the default) retains the established production kernels until
promotion. `scripts/run-gemma4-q4-matvec-ab.sh` records the two-run screen or
five-run promotion artifact under `artifacts/phase-12a-q4-matvec-ab/`; its
telemetry requires the selected baseline/candidate name for every affected Q4
family. The mixed-Q4/Q6 gate requires exact SHA/EOS parity, Q4 KV and Resident
accounting stability, at least 3% long-context improvement, and no
short-context regression. The optional re-quantized Q4 vocabulary stream is
covered by direct kernel parity and telemetry, not mixed-stream equality.

The production Q4 attention scan removes the first-pass barrier after each
thread updates the value partial that it exclusively owns. It retains the
four-way, 128-thread Q4 split and is selected by default as
`attention_decode_fused_gemma4_simd_q4_0_2pass_no_value_barrier`. Set
`ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=2pass64` (or `baseline`) to select the
pre-promotion diagnostic oracle.
`scripts/run-gemma4-q4-attention-barrier-ab.sh` records a two-window screen or
five-window promotion artifact under
`artifacts/phase-12a-q4-attention-barrier-ab/`. It clears unrelated rejected
experiment flags and requires exact SHA/EOS parity, stable Q4 KV and Resident
accounting, explicit baseline/candidate selection, at least 3% long-context
improvement, and no short-context regression before promotion. The five-window
Apple-Silicon artifact at
`artifacts/phase-12a-q4-attention-barrier-ab/20260730T162521Z/q4-attention-barrier-ab-summary.json`
passed exact SHA/EOS parity and stable accounting, with +6.16% long-context
decode (31.29 to 33.21 tok/s) and +1.31% short-context decode (48.93 to 49.57
tok/s).

The next opt-in attention candidate combines one Q4 scale load per SIMD-group
block with that promoted no-value-barrier schedule. Set
`ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=2pass_cache_no_value_barrier` to select
it; the default remains the promoted no-value-barrier scan.
`scripts/run-gemma4-q4-attention-cache-ab.sh` records two-window screen or
five-window promotion artifacts under `artifacts/phase-12a-q4-attention-cache-ab/`.
Its gate requires exact SHA/EOS parity, stable Q4 KV and Resident accounting,
explicit baseline/candidate selection, at least 3% long-context improvement,
and no short-context regression.
The Apple-Silicon screen at
`artifacts/phase-12a-q4-attention-cache-ab/20260731T153217Z/q4-attention-cache-ab-summary.json`
was exact and accounting-stable but only gained +0.05% long-context decode, so
it is not eligible for promotion.

The next isolated candidate explicitly unrolls pairs of consecutive KV
positions while retaining the promoted four-way 128-thread Q4 scan, exact
online-softmax order, and no-value-barrier schedule. Set
`ATLAS_GEMMA4_Q4_ATTENTION_EXPERIMENT=2pass_unroll2_no_value_barrier` to
select it. `scripts/run-gemma4-q4-attention-unroll-ab.sh` records its screen
or five-window promotion evidence under
`artifacts/phase-12a-q4-attention-unroll-ab/`; the default remains the
promoted scan unless exact parity, at least +3% long decode, and no
short-context regression all pass.

The next isolated candidate changes only Q4 FFN-down resident storage from
row-major records to `[16-row tile][block][row]`, preserving every 18-byte
Q4_0 record. Set `ATLAS_GEMMA4_FFN_DOWN_EXPERIMENT=interleaved16` to repack
only `blk.*.ffn_down.weight` during resident upload and select the matching
decode and layer-major prefill kernels; `baseline` (and the default) retains
the existing row-major layout. `scripts/run-gemma4-q4-ffn-down-ab.sh` writes
two-window screen or five-window promotion artifacts under
`artifacts/phase-12a-q4-ffn-down-ab/`. Promotion requires exact generated
SHA/EOS parity, unchanged Resident/KV/upload accounting, explicit pipeline
selection, at least 3% long-context decode improvement, and no short-context
regression.
The Apple-Silicon screen at
`artifacts/phase-12a-q4-ffn-down-ab/20260730T205512Z/q4-ffn-down-ab-summary.json`
was exact but regressed long decode by 4.48% and short decode by 5.85%, so it
remains opt-in only.

  The default FFN gate/up path combines the two same-input Q4 projections into
  one Metal dispatch. Set
  `ATLAS_GEMMA4_FFN_GATE_UP_EXPERIMENT=baseline` to recover the separate
  projection oracle when diagnosing a regression. Its promotion requires
  exact-token Resident acceptance and a like-for-like warm-throughput
  improvement.
- Normal `chat` and `generate` stop at EOS and remain on
  `ExecutorMode::Resident`.
- Preserve the Phase-12 manifest quality gates and add a regression assertion
  that the profiling path is observational only: it must not change generated
  token IDs, finish reason, resident bytes, or default readback behavior.
- Restore the legacy resident score/softmax/value path as the observable
  production `ExecutorMode::Resident` attention selection while Q8 parity is
  unresolved; it remains GPU-resident and never falls back to `Reference`.
  Keep fused `attention_decode_fused_f32` behind the hidden
  `ResidentAttentionPath::Fused` diagnostic selector. Promote fused only after
  the full Q8 suite passes exactly. The opt-in profile reports the selected
  attention implementation and fused dispatch count.
- Validate FP32 fused-vs-legacy, Q8 fused-vs-legacy, and Q8 legacy-vs-FP32
  against position zero, multi-position KV reads, and the configured capacity
  boundary. Keep one command buffer per token and selected-token default
  readback assertions in those tests.
- Q4 keeps its pinned Phase-12 manifest policy. Before promoting Q8 after a
  fusion change, run the resident FP32-vs-Q8 golden suite for 32 greedy tokens
  each: `The capital of France is`, `Atlas resident decode validation.`, and
  the recorded fixed 64-token KV prompt in the Phase-12 artifact. Require
  exact generated token IDs and finish reason for every case; logits are
  diagnostic evidence only. If it fails, retain the output artifact and run
  the opt-in resident stage-parity diagnostic to identify the first divergent
  FP32 stage rather than relaxing the gate.
- Before promoting a Gemma 4 kernel optimization, add a pinned external
  runtime revision and require exact greedy prompt/generated token parity for
  a fixed canonical short chat. Phase 12a-pre intentionally accepted the
  text-chat foundation without this external-runtime comparison; optimization
  work must restore it as a promotion gate rather than relying on semantic
  similarity.

## Exit gate

On the same Apple-Silicon hardware, pinned Gemma fixture, prompt, Q4_0 KV
cache, and context capacity, the candidate must retain exact full and
measurement-window generated-token hashes plus the same first EOS position as
the production attention path. Both modes must report `ExecutorMode::Resident`,
stable KV/resident bytes, and a positive long-context median decode-throughput
change without a short-context regression.

Record `scripts/run-gemma4-decode-attention-ab.sh` output under
`artifacts/phase-12a-decode-attention-ab/`, along with the fixture checksum,
hardware/OS, profile samples, context window, upload/readback/command-buffer
metrics, and the combined A/B summary. No attention experiment becomes the
default until this artifact passes.
