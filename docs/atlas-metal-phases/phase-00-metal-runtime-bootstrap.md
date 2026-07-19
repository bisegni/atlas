# Phase 0: Metal runtime bootstrap

## Outcome

`atlas-metal` selects the default device, compiles and dispatches
`vector_add_f32`, and returns output plus GPU duration to Rust.

## Work

- Create the workspace with `atlas-core`, `atlas-metal`, and `atlas-cli`.
- Add direct `objc2-metal` bindings, device/queue creation, buffer
  upload/download, command-buffer error handling, pipeline caching, and timing.
- Implement vector add, scalar multiply, SiLU, reduction, and transpose with
  deterministic CPU references.

## Model fixture

Download the small SmolLM2 fixture under the shared contract. Parse
`config.json` and the SafeTensors index/header only; verify every manifest
entry without loading model weights.

## Exit gate

`cargo test -p atlas-metal --test phase_00_bootstrap` executes 100 dispatches
without resource growth, meets FP32 tolerances, and records CPU/GPU times and
model revision. Provide `atlas-cli metal-info` and `atlas-cli fixture verify
--model small`.
