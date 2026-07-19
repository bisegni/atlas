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
