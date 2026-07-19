# Phase 3: First transformer model

## Outcome

The engine loads SafeTensors, tokenizes a prompt, executes every layer on
Metal, and greedily generates tokens with oracle-compared logits.

## Work

- Implement SmolLM2/Llama config parsing, tokenizer integration, shard loading,
  embeddings, RMSNorm, RoPE, grouped-query attention, SwiGLU, final norm, LM
  head, and greedy decoding.
- Add a layer trace for embeddings, Q/K/V, attention, MLP, and final logits;
  create pinned oracle goldens from raw token IDs.

## Model fixture

Download the small model for complete generation tests. After it passes,
download the larger model for config validation and a one-layer short-prompt
forward pass.

## Exit gate

`atlas-cli generate --model small --prompt 'The capital of France is'
--max-new-tokens 8 --greedy` matches the approved golden sequence/logits.
`phase_03_model` emits per-layer drift and validates the larger model forward.

## Implementation notes

- `atlas-model` is the Llama/SmolLM2 execution path. It reads `config.json`,
  `tokenizer.json`, a single SafeTensors file or sharded index, and executes
  embeddings, RMSNorm, RoPE, grouped-query attention, SwiGLU, final norm, and
  LM head through `atlas-ops`/Metal.
- `atlas-cli generate --model small --prompt 'The capital of France is'
  `--max-new-tokens 8 --greedy` prints raw prompt/generated token IDs, text,
  and a trace for the required stages. Phase 3 intentionally recomputes the
  prompt each decode iteration; KV reuse starts in Phase 4.
- Pin an approved oracle as JSON outside Git (for example,
  `artifacts/phase-03/small-golden.json`) and pass `--golden PATH`. Its
  required `generated_token_ids` field is compared exactly; optional
  `final_logits` and `logit_abs_tolerance` make logit drift fail the command.
- `atlas-cli phase_03_model --model larger` executes one layer for the larger
  model and prints its trace. Use `--model-dir PATH` with either command for a
  separately downloaded fixture.
