# Phase 13.0 — Resident decode to 100 tok/s

## Resume: where we are (end of phase 12.3)

Phase 12.3 moved Gemma4-E2B q4_0 Resident decode from 35.3 tok/s (long,
2048 context, 512 measured tokens) to a promotion-validated composed stack
at 56.6 tok/s long / 68.9 tok/s short (+60.7% / +54.2%, GPU 14127 -> 8641 ms
and 2745 -> 1737 ms). The old phase documentation was retired with this
phase; the composed stack is the new baseline.

Current composed stack (all Resident, env-selected during phase 12.3):

- `attention_decode_gemma4_simd_q4_0_flash16_uw` / `_swa_uw`: single-dispatch
  flash attention, force-unrolled block loops, register-cached query, 12/24
  SIMD slices (threadgroup-memory merge buffers 24 KiB, within the M2 Max
  32 KiB budget).
- `matvec_q4_0_64row_mv` / `matvec_q4_0_64row_mv_rms`,
  `matvec_q6_k_64row_mv` / `matvec_q6_k_64row_mv_rms` (phase 13.0 P4,
  promoted): llama.cpp-style mv_ext ports with 64 rows per threadgroup
  (8 SIMD groups x 8 rows, 256 threads), selected by `mv_ext_64`.
- `matmul_q4_0_qkv_32row_mv`, `matmul_q4_0_gate_up_32row_mv`: the 32-row
  mv_ext ports for the fused QKV and gate/up projections.
- `rms_norm_decode_f32_vec4` + the P1 `_rms` input fusion
  (`ATLAS_GEMMA4_RMS_MATVEC_EXPERIMENT=fused`) and P2 epilogue fusion
  (`ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT=fused`).
- P2b `kv_append_decode_fused_vnorm`, P2c post-FFN residual RMS fusion, and
  the P2a gelu/ple fusions (all `ATLAS_GEMMA4_*_EXPERIMENT=fused`, opt-in).
- `matmul_f16_batch` for the f16 per-layer model projection.

Promotion evidence: `artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T223643Z/`
(long 56.86 tok/s, short 66.41 tok/s, five-run median).

Device facts: Apple M2 Max, `metal-info` reports 32 KiB threadgroup memory
and 1024 max threads per threadgroup.

## Shared-KV query repair and Flash16 correctness gate (implementation pending Metal acceptance)

The layer-major Resident path must project a fresh Q vector for every layer,
including layers that reuse K/V from an earlier provider. The fused QKV path
already handled provider layers, but shared-KV layers could leave the Q buffer
unchanged and therefore attend with stale data. This is unrelated to
quantization-plan selection: cached and disabled preflight runs select the
same mixed Q4/Q6 formats.

`Gemma4E2bExecutor` now uses the RMS-input Q4 matvec for those shared-KV
layers. This removed the initial stale-Q failure: the C++ chat is coherent
through a 64-token run on the Resident Flash16 path. It does **not** accept
Flash16 as numerically correct yet. Apple-Silicon evidence on the C++ prompt
shows Flash16 and the prior Resident `LegacyFused` attention kernel agree for
the first 50 generated tokens and diverge at zero-based generated token 50
(the 51st generated token).

The slice-merge `_flash16_uw` kernels are therefore retired from the Flash16
selector. Flash16 selects `_flash16_exact_nb` kernels that preserve
LegacyFused's runtime FP32 Q·K accumulation, four-SIMD score reduction,
key-ordered online softmax, and per-token fp32 logit arithmetic, dropping only
the redundant per-key value barrier (phase 13.2). The
`_exact_runtime` variants retain the value barrier and remain diagnostic-only;
the compile-time head-width candidates remain diagnostic-only too.

Flash16 (the `_nb` no-value-barrier kernel) is the Resident production default
for Q4-KV decode, accepted in phase 13.2 once the per-token logit-digest and
exact-token parity gates passed on Apple Silicon. `LegacyFused` remains
selectable through the explicit `--q4-attention-mode legacy_fused` diagnostic
and performance interfaces. The ignored Gemma test
`q4_kv_flash16_matches_legacy_resident_attention_across_chat_and_long_decode`
is self-contained: it requires Flash16 exact token/finish parity with
Resident LegacyFused for canonical and C++ chat prompts plus a 256+64 long
decode window, and the digest gate
`flash16_matches_legacy_resident_output_logit_digests` requires byte-identical
per-token fp32 logits. Both pass on Apple Silicon; they remain the parity
sentinel for any future attention-kernel change.

External-oracle acceptance is deliberately separate. After fixture capture,
`legacy_fused_matches_captured_llama_oracles` requires Resident LegacyFused to
reproduce independently captured llama.cpp streams for both the short
canonical chat and the fixed 64-token C++ chat. This prevents a missing local
oracle artifact from hiding the Flash16-versus-LegacyFused kernel diagnosis.

Capture the canonical independently before running the full ignored suite:

```zsh
bash scripts/capture-gemma4-llama-oracle.sh
```

The script renders each user message through llama.cpp's Jinja single-turn
mode, while independently verifying that its expected raw Gemma protocol
rendering tokenizes identically to Atlas. It uses greedy Q4-K/V decoding for a
short terminal `<turn|>` case and a fixed 64-token C++ case. It promotes
ignored local fixtures only from the Resident `LegacyFused` result after exact
prompt-ID, visible-output, and finish checks against llama.cpp. It then captures Flash16 separately, tokenizes the
llama.cpp completion independently, and writes both its direct Flash16-versus-llama first divergence and the
LegacyFused-versus-Flash16 first divergence under
`artifacts/phase-12a-llama-oracle/`; it exits non-zero while Flash16 differs.

Run the external-oracle gate after the capture has written both verified
fixtures. The capture can still exit non-zero because Flash16 parity is
currently expected to fail:

```zsh
cargo test -p atlas-model --test phase_12a_gemma4_resident \
  legacy_fused_matches_captured_llama_oracles \
  -- --ignored --exact
```

For a direct diagnostic capture, use:

```zsh
cargo run --release -p atlas-cli -- generate \
  --model gemma4-e2b-q4_0 --chat \
  --prompt "write an hello world main c++ function" \
  --max-new-tokens 64 --greedy \
  --q4-attention-mode legacy_fused --json
```

The normal `chat`, `benchmark`, `generate`, `profile`, and `matched` defaults
now use Flash16 (the `_nb` no-value-barrier kernel, phase 13.2) and report
`q4_attention_mode` and `attention_kernel` with the selected Resident kernel.
`--q4-attention-mode legacy_fused` selects the diagnostic kernel explicitly.

## Measured hotspot baseline (new phase baseline)

Profile of the composed stack, long-context decode (1024-token attribution
window, 512 measured tokens; coverage 100%):

`artifacts/phase-13.0-baseline-profile/20260809T060610Z/diagnostic/long-profile.json`

| Rank | Kernel | GPU share | GPU ms/1024 tok | disp/tok |
|---:|---|---:|---:|---:|
| 1 | matvec_q4_0_32row_mv | 29.5% | 7177 | 239.8 |
| 2 | attention flash16_swa_uw | 19.4% | 4733 | 42.0 |
| 3 | rms_norm_decode_f32_vec4 | 19.1% | 4649 | 263.8 |
| 4 | matmul_q4_0_gate_up_32row_mv | 18.2% | 4420 | 52.5 |
| 5 | attention flash16_uw | 11.8% | 2873 | 10.5 |
| 6 | matvec_f16 | 6.2% | 1514 | 1.5 |
| 7 | matmul_q4_0_qkv_32row_mv | 2.6% | 627 | 22.5 |

Measured decode window: 8902 ms GPU for 512 tokens = 17.39 ms/token
(57.5 tok/s GPU-bound). Host encode 1470 ms/window; cpu_wait 9038 ms is on
par with GPU time, so decode is GPU-latency-bound with a meaningful
dispatch-encoding component (778 dispatches/token, ~43k threadgroups/token).

## Bottleneck analysis

1. `matvec_q4_0_32row_mv` (240 disp/tok) is the largest single kernel. It
   covers the ffn-down, attention-output, and ple projections on the same
   Q4 weights that are re-read on every token. At 32 rows per threadgroup
   the kernel is occupancy- and ILP-limited more than pure-bandwidth
   limited at the current 17.4 ms/token.
2. `rms_norm_decode_f32_vec4` at 19% is a compute-trivial kernel that has
   become launch/latency-bound: 264 dispatches/token (two RMS per layer
   plus the final norm), each re-reading and re-scaling the full 2304-wide
   hidden buffer. This is the classic small-kernel serialization cost.
3. Flash attention (31% combined) is dominated by the per-slice serial KV
   scan; slices re-read the KV range from the resident buffer, and the
   merge cost grows with slice count.
4. 778 dispatches/token feeds a 16% host-encode component; the existing but
   untested fusion experiments (gelu, qk-norm/rope, rms-epilogue,
   combined-fusions scripts under `scripts/`) target exactly this.

## Plan (ordered, each with hypothesis, kernel gate, and screen gate)

### P1. Fuse RMS into the consuming matvec kernels
Hypothesis: eliminating the 264 disp/tok rms dispatches and their
hidden-state round trips removes most of the 19.1% rms cost; llama.cpp
mul_mv kernels compute the input RMS in-kernel. Implement an RMS-input
variant of the 32-row matvec family (q4/q6/gate-up/qkv) selected by a
fused flag, prove tolerance parity (max-abs < 1e-3) against the vec4 +
unfused path, then screen. Target: decode GPU < 14.5 ms/token (~70 tok/s).
Fallback: keep vec4 but fuse only into ffn_down and attention-output.

P1 status (implemented, screen-passed): the four `_rms` kernels
(`matvec_q4_0_32row_mv_rms`, `matmul_q4_0_qkv_32row_mv_rms`,
`matmul_q4_0_gate_up_32row_mv_rms`, `matvec_q6_k_32row_mv_rms`) fold the
input RMS into the mv_ext matvec dispatch and are selected by
`ATLAS_GEMMA4_RMS_MATVEC_EXPERIMENT=fused` (default off; disabled under
`ATLAS_GEMMA4_TRACE_STAGES`). Parity is proven by
`crates/atlas-metal/tests/matvec_rms_fused_parity.rs` (max-abs ~1e-8).
Screen evidence `artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T071026Z/`
(RMS_NORM_EXPERIMENT=vec4 + flash16_uw + RMS_MATVEC_EXPERIMENT=fused):
dispatches 772 -> 721 /token (short) and 778 -> 727 (long), decode
44.4 -> 68.9 tok/s short and 34.7 -> 55.8 tok/s long vs the phase-13
baseline. The standalone-RMS removal is within run-to-run noise at screen
resolution; the dispatch savings and fused-kernel path are measured.
Promotion still requires the composed five-run gate (P6).

### P2. Reduce dispatch count by fusing small kernels
Gelu into gate-up (16-row fused variants already exist), qk-norm/rope into
the attention dispatch, residual/epilogue adds, kv-append. Each fusion is
semantics-preserving and gated by the tolerance test. Target: 778 ->
~450 dispatches/token; host-encode and small-kernel latency savings
worth ~5-8% end-to-end.

P2a status (implemented, promoted): FFN `gelu_multiply_f32`, PLE
`ple_gelu_multiply_offset_f32`, and the decode epilogue
`gemma4_rms_residual_f32` (rewritten to vec4 tiles in this phase) fold the
gelu/offset-multiply and the post-attention RMS+residual-add into single
dispatches, selected by `ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT=fused`,
`ATLAS_GEMMA4_FFN_GELU_MULTIPLY_EXPERIMENT=fused`, and
`ATLAS_GEMMA4_PLE_COMPOSITION_EXPERIMENT=fused` (all opt-in, default off).
Bitwise parity is proven by
`crates/atlas-metal/tests/dispatch_fusion_parity.rs`. Promotion evidence
`artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T173852Z/`: long 34.09 ->
54.88 tok/s (+60.97%), short 43.14 -> 67.23 tok/s (+55.8%), GPU median long
14029 -> 8495 ms, dispatches 778 -> 622/token.

P2b status (implemented, screen-passed): `kv_append_decode_{f32,q8_0,q4_0}_vnorm`
fold the single-group provider V RMS into the KV append dispatch (each block
thread recomputes the reference's sequential sumsq, so the quantized cache
bytes are bitwise identical and the raw V buffer stays untouched), selected
by `ATLAS_GEMMA4_KV_APPEND_VNORM_EXPERIMENT=fused` (opt-in, default off).
Bitwise parity: `crates/atlas-metal/tests/kv_append_vnorm_fused_parity.rs`.
Screen `artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T180850Z/`: long
34.83 -> 57.00 tok/s (+63.7%), short 43.98 -> 70.42 tok/s (+60.1%), GPU
13944 -> 8223 ms, dispatches 622 -> 486/token.

P2c status (implemented, promoted): the same `gemma4_rms_residual_f32`
kernel now also fuses the post-FFN RMS + residual-add site (label
`post_ffn_norm_residual`), under the same `ATLAS_GEMMA4_RMS_EPILOGUE_EXPERIMENT`
flag; the kernel-level parity already covers the exact norm+add reference.
The composed P2 promotion gate
`artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T183148Z/` (P2a + P2b +
P2c) passed: long 34.42 -> 56.49 tok/s (+64.1%), short 45.08 -> 68.88
tok/s (+52.8%), GPU median long 13974 -> 8222 ms, dispatches 622 ->
458/token, with all four kernel parity gates (matvec mv_ext, rms-fused,
dispatch-fusion, kv-append-vnorm bitwise) green.

P2d status (blocked, analysis recorded): qk-norm/rope cannot be folded into
the flash16 attention dispatch. Attention reads the quantized K/V cache,
while the K norm+rope must run on the raw K before quantization; the Q-side
fold alone is worth <0.3% at 512 context (qk_norm_rope 69.1 us vs 23690 us
total per token in the max-context profile) at high kernel complexity. The
k_rot round trip remains the only dispatch-level cost.

### P3. Flash16 v3: threadgroup-memory KV tiling
Stage the serial KV scan through threadgroup memory in fixed-size chunks so
each key block is read once per threadgroup instead of once per slice, and
double the effective scan parallelism within the 32 KiB budget by
splitting the value accumulator merge. Target: attention 5.4 -> 3.5 ms/token
(~11% end-to-end). Kernel gate: parity vs flash16_uw.

P3 status (blocked, analysis recorded; v3 code reverted): the dim-half split
of the V accumulator cannot be combined with a key-range split of the scan
slices. Each slice folds the online-softmax weights of its key range only
into the V dims that slice covers, so the dominant top-score key's V
contribution never reaches the other half of the output (observed output
collapsed to ~1e-6 uniform values; single-dominant-key heads matched). Every
correct alternative fails within the 32 KiB budget: shared key ranges across
slice halves keep the serial scan length identical to `_uw` (no gain, 2x K
traffic), an f16 merge state exceeds the 1e-3 tolerance by >5x (exp-rescale
rounding), and an f32 24-slice merge state needs ~48 KiB. Attention is
~0.4% of per-token time at the 512-context gate, so the remaining pursuit is
deprioritized; any future version needs a wider-visibility kernel redesign
(e.g. single-writer V accumulation across slices).

### P4. Matvec breadth and ILP
64-row threadgroups for the widest matrices (output/ple/ffn-down), Q6
64-row LM-head variant, and a check of accumulation order vs the mv_ext
parity test. Target: matvec family 8.7 -> 7.2 ms/token.

P4 status (implemented, parity green, small win, target not met): four
64-row-per-threadgroup kernels (`matvec_q4_0_64row_mv[_rms]`,
`matvec_q6_k_64row_mv[_rms]`) with byte-identical per-lane accumulation
order to the 32-row family (256 threads, 8 SIMD groups x 8 rows per
threadgroup), selected by `ATLAS_GEMMA4_Q4_MATVEC_EXPERIMENT=mv_ext_64` and
`ATLAS_GEMMA4_Q6_LM_HEAD_EXPERIMENT=mv_ext_64` (opt-in; the qkv/gate-up
32-row kernels keep the mv_ext selection so the composed stack stays
intact). Parity is proven by the extended `matvec_mv_ext_parity.rs` and
`matvec_rms_fused_parity.rs` (6 tests each, max-abs < 1e-3, including the
2048/2304 real widths). Isolated long-workload A/B on the composed stack
(2 runs each, 512 measured tokens): threadgroups/token 41380 -> 33856
(-18.2%), GPU 8325/8330 -> 8196/8292 ms, tok/s 57 -> 57/58. The 64-row
change cuts threadgroup count as designed but moves GPU time by only
~0.5-1.5%, so the kernel was not threadgroup-launch-bound; the planned
matvec 8.7 -> 7.2 ms/token target was not achieved. The composed screen
(`20260809T220812Z`, all P1-P4 flags) passed at long 35.12 -> 57.14 tok/s
(+62.7%) and short 43.92 -> 71.33 (+62.4%) vs the phase-13 baseline, with
all four parity gates green. P4 stays composed (opt-in, no regression) but
does not materially advance the 100 tok/s target.

### P5. Prefill (secondary)
 89 tok/s prefill today; flash-style tiled q4 prefill kernels and larger f16
 batch tiles target 120+ tok/s. Gate: prefill tok/s on the fixed workload,
 parity vs the current prefill path.

P5 status: the flash-style tiled q4 batch kernel
`matmul_q4_0_batch_16row_token_tiled` is implemented and composed (opt-in via
`ATLAS_GEMMA4_Q4_BATCH_EXPERIMENT=tiled`, selected for the matmul_batch Q4_0
and ffn_down Q4_0 sites). It tiles TOKEN_TILE=8 tokens per threadgroup
(128 threads, grid `ceil(batch/8) x ceil(output_width/16)`, threadgroup
input_tile[8][32] + weight_tile[16][18]), cutting weight traffic 8x while
keeping the identical per-lane accumulation order, so kernel-level parity is
exact: the `batch_matmul_parity.rs` tiled test (4 geometries x 5 batches =
20 cases) passes at max-abs ~1e-9 vs tolerance 1e-3, and a composed-stack
end-to-end A/B produces a byte-identical stream (generated_token_sha256
`2a13ef27...` both ways). Measurements on the composed stack, 59-token
prompt, same process/flags, only the batch selector differing: prefill
94.05 -> 98.23 tok/s (+4.4%); the screen run (`20260809T231154Z`) shows
94.37 -> 99.78 tok/s (+5.7%) on the fixed workload, `prefill_gate_met:
false`. Diagnostics (25-token prefill scope): wall 378.6 -> 290.7 ms while
attributed GPU time went 229.2 -> 238.9 ms (tiled matmul 126.2 ms vs 108.5
ms untiled; 8x fewer threadgroups makes the kernel latency/occupancy-bound,
not bandwidth-bound), so the wall gain comes from cheaper command
scheduling, not kernel throughput. The 120 tok/s gate is NOT met, and a
zero-cost batch matmul could at best reach ~115 tok/s on the fixed workload:
the dominant remaining prefill cost is the per-token dispatch loops
(gemma4_qk_norm_rope_fused_f32 875 dispatches / 49.5 ms, attention 875 /
30.6 ms, rms_norm_groups_in_place_unweighted_f32 375 / 25.3 ms,
matmul_f16_batch 13.3 ms, kv_append 7.4 ms) plus host-side scheduling
(cpu_wait ~= wall). P5 stays composed (opt-in, no regression, ~+5% prefill)
and is the next phase's work item: batched per-token kernels
(qk/rope, prefill attention, unweighted norm fused into kv append).

### P6. Final composed promotion A/B
Compose P1-P4 wins, run the five-run promotion gate against the phase-13
baseline (the composed phase-12.3 stack). Target >= 100 tok/s long with no
short-workload regression.

P6 status (promoted `20260809T223643Z`): the five-run composed gate
(`--with-rms-vec4 --with-flash16-uw --with-rms-fused --with-dispatch-fusion
--with-matvec-64row`) passed with `promotion_eligible: true`. Long
34.67 -> 56.86 tok/s (+64.0%), short 43.60 -> 66.41 tok/s (+52.3%), GPU
long 14269 -> 8583 ms, dispatches 778 -> 572/token, all four parity gates
green (mv_ext 6, rms-fused 6, dispatch-fusion 3, kv-append bitwise 3),
accounting stable on both workloads. Evidence:
`artifacts/phase-12.3-q4-matvec-mv-ext-ab/20260809T223643Z/`. The promoted
composed stack is the new phase baseline: long ~57 tok/s, short ~66 tok/s.
The 100 tok/s long acceptance gate is not met by this phase (56.9 vs 100);
the phase stays open and the remaining gap is the target of the next
phase's work items (matvec family bandwidth, attention scan, prefill).

## Acceptance gates (phase 13.0)

- 100 tok/s decode on the long workload (2048 context, 512 measured tokens,
  five-run median) on the Resident executor, no CPU fallback;
- no short-workload regression vs the phase-13 baseline;
- every new kernel covered by a kernel-level tolerance parity test
  (max-abs < 1e-3 vs the production pipeline);
- accounting (Resident, KV, upload, readback) unchanged and stable;
- stream hashes recorded as drift diagnostics (parity gate superseded in
  phase 12.3; correctness lives in the kernel-level tests);
- evidence recorded under `artifacts/phase-13.0*/` with the exact commands
  and environment.

## Production cleanup (20260809)

The promoted composed stack is now the only execution path. All experiment
plumbing that selected between losing kernels or legacy variants has been
removed: the `ATLAS_GEMMA4_*_EXPERIMENT` selectors, attention-baseline and
prefill-token-major switches, trace-stage instrumentation, the 16-row /
packed16 / interleaved / two-pass / cacheopt / token-tiled / q6 matvec
variants, and the main A/B runner script `run-gemma4-mv-ext-ab.sh`.

The MSL kernel set is trimmed to the kernels the codebase dispatches: the
64-entry pipeline list (`lib.rs`) matches `kernels.metal` exactly, and a
repo-wide scan of `dispatch*` call sites confirms no referenced kernel is
missing. The surviving set keeps the reference/parity oracle kernels
(`matmul_f32`, `masked_softmax_f32`, `attention_scores_f32`,
`attention_values_f32`, `logits_process_f32`) because `AtlasModel` remains
the parity oracle for the fixture-gated phase-06 gates. Experiment-era
parity tests that compared losing variants against each other were replaced
by CPU-oracle tolerance tests against the production kernels
(`attention_flash_correctness.rs`, `matvec_mv_ext_parity.rs`,
`matvec_rms_fused_parity.rs`, `batch_matmul_parity.rs`); variant-to-variant
parity tests for deleted kernels were removed.

## Command book

```zsh
# Phase baseline profile
cargo test --workspace
cargo run -p atlas-cli -- metal-info
cargo run -p atlas-cli -- fixture verify --model small   # SmolLM2 fixture sanity
# Composed-stack benchmark (phase-13 baseline, no A/B selectors)
bash scripts/run-gemma4-performance-acceptance.sh
# CPU-oracle kernel correctness (production kernels)
cargo test -p atlas-metal --test attention_flash_correctness
cargo test -p atlas-metal --test matvec_mv_ext_parity
cargo test -p atlas-metal --test matvec_rms_fused_parity
cargo test -p atlas-metal --test batch_matmul_parity
```
