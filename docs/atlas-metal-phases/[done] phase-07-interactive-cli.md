# Phase 7: Interactive CLI

## Outcome

`atlas-cli` can run one greedy completion or maintain a terminal conversation
against the local model without an HTTP server.

## Work

- Add `atlas-cli chat --model small --prompt TEXT --max-tokens N` and REPL
  mode when `--prompt` is omitted.
- When `--max-tokens` is omitted, resolve the response budget from the
  remaining executor context; retain the explicit option for fixed workloads.
- Support `/help`, `/reset`, and `/quit`; retain conversation text across
  turns but create/reset the executor session for each completion.
- Print TTFT plus prefill/decode throughput after every completion.

## Exit gate

One-shot and equivalent REPL prompts emit the same greedy tokens, `/reset`
removes prior conversation context, and errors identify a missing fixture or
unavailable Metal device.

## Implementation notes

`atlas-cli chat` implements both modes through the Phase-6 executor.  Parser,
REPL-state, reset, and metrics-format tests run without a model fixture; live
generation parity remains conditional on a usable Metal device and fixture.
Generation metrics report the resolved `max_new_tokens` and whether the limit
came from remaining context or an explicit CLI option.
