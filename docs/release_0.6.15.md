# Kogwistar 0.6.15

## Typed runtime and backend contracts

- Split workflow runtime state, checkpoint, routing, invocation, join, and
  persistence responsibilities into focused internal modules while preserving
  the public runtime imports.
- Extract typed document, backend payload, lifecycle projection, OCR geometry,
  and span models from broad model modules.
- Strengthen persisted-state decoding and regression coverage for checkpoint,
  join, budget, backend response, and workflow invocation boundaries.
- Upgrade `pydantic-extension` to `0.0.8` for the intended performance
  improvement.
- Update the Rust lockfile to `rustls-webpki 0.103.16` and `webpki-roots
  1.0.10` to address RUSTSEC-2026-0382 and RUSTSEC-2026-0381.

## Verification status

- Release candidate only. Do not tag or publish until the core CPython/PyPy CI
  matrix and release checks pass on the merged commit.
- Local Ruff and focused tests should be recorded here after verification; do
  not treat an interrupted or incomplete run as passing.
