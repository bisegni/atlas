# Atlas Metal phase index

Status source of truth for the Atlas Metal inference phases. A phase is
complete only when its declared runnable acceptance gate passed on Apple
Silicon with the required numerical or performance evidence recorded.

## Current phase

- [phase-13.0-resident-decode-100-toks.md](phase-13.0-resident-decode-100-toks.md) —
  Resident decode to 100 tok/s. Active: resume, hotspot baseline, ordered
  improvement plan (RMS fusion, dispatch fusion, flash16 v3 tiling, matvec
  ILP), acceptance gates.

## Retired phases

Phase 12.3 (resident decode optimization) reached its promotion gate with
the composed flash16_uw + mv_ext + rms-vec4 stack at 56.6/68.9 tok/s
long/short and was retired when phase 13.0 started. Earlier phase documents
were removed as part of that cleanup; their evidence artifacts remain under
`artifacts/`.
