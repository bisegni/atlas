# Phase 12a — Gemma-only inference architecture and correctness reset

Atlas supports one production artifact: `gemma4-e2b-q4_0`, the pinned Gemma 4
E2B Q4_0 GGUF. Historical Llama and SmolLM phase records remain evidence for
earlier work, but they are not current CLI, manifest, provider, download, or
runtime contracts.

## Production contract

- `chat` and `generate` load a manifest-backed Gemma artifact only.
- Inference is Resident-only. A rejected model or failed Resident session is
  reported directly; no Reference executor is selected.
- `atlas-model::inference` is the narrow object-safe CLI boundary. Model-family
  adapters render chat prompts, tokenize, construct a resident session, stream
  generation, reset state, and report common resident metrics. Gemma's PLE,
  shared KV, RoPE, stop handling, thought filtering, and Metal token loop stay
  concrete behind the adapter.
- `artifacts/chat-performance.jsonl` remains append-only and records the
  common resident fields plus Gemma prefill metadata.

## External exact-token oracle

The independent oracle must be generated from the locally installed
`llama-cli`, not Atlas. The current observed executable identifies itself as
`llama.cpp version 10080 (fd41bf65a)`. The fixture must have SHA-256
`fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`.

The canonical rendered prompt bytes are:

```text
<|turn>user\nhi<turn|>\n<|turn>model\n
```

Use greedy settings (`--temp 0 --top-k 1`) and record the exact prompt-token
IDs, generated-token IDs, finish reason, visible text, executable revision,
and GGUF digest in `fixtures/gemma4-e2b-oracle.json`. The checked-in
`fixtures/gemma4-e2b-resident-canonical.json` currently pins the independently
reviewable Atlas Resident side; it is explicitly marked as awaiting the valid
llama.cpp capture. The required visible
canonical response is:

```text
Hello! How can I help you today? 😊
```

`atlas-cli generate` is raw completion by default. Use `--chat` when collecting
the Atlas side of this oracle so it renders the same Gemma turn delimiters as
normal `chat`.

## Acceptance gate

With `models/gguf/gemma-4-e2b-it-q4_0/gemma-4-E2B_q4_0-it.gguf` installed,
run the ignored Metal Q6_K oracle plus the one-shot Resident canonical test.
Add the two-turn case only after recording its matching external oracle. The
one-shot test must match token IDs and finish reason exactly,
report nonzero Resident bytes, bounded readback, zero warm weight uploads,
and no Reference fallback. Do not rename this file `[done]` until that evidence
and the required performance gate are recorded.

## Recorded Resident evidence

On Apple Silicon, the normal interactive `chat` command produced the canonical
`hi` response twice in one process. The cold turn emitted 11 tokens, ended at
EOS, used 44 readback bytes and 3,489,602,512 Resident bytes, and uploaded
3,333,699,724 weight bytes. After `/reset`, the second turn produced the same
visible response with `weight_upload_bytes = 0`, proving the Gemma weights
remained resident across normal chat-session reset. Its measured short-prompt
rates were 35.20 prefill tok/s and 39.22 decode tok/s. This is correctness and
warm-residency evidence only; it does not satisfy the separate long-workload
performance gate.

The fixture-gated Metal acceptance command also passed on Apple Silicon:

```zsh
cargo test -p atlas-model --test phase_12a_gemma4_resident \
  resident_canonical_chat_matches_pinned_tokens_and_stays_warm_after_reset \
  -- --ignored --exact
```

It verifies the pinned prompt/generated token IDs, EOS finish, complete
protocol text, bounded readback, nonzero Resident allocation, and zero weight
upload on the post-reset warm turn.
