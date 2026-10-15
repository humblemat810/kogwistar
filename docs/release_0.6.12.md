# Kogwistar 0.6.12

## Runtime contract hardening

- Added a shared typed contract for persisted workflow join snapshots.
- Centralized sync and async join-state decoding with fail-closed validation of
  node IDs, counters, waiters, pending tokens, and suspended tokens.
- Routed `resume_run()` through the same decoder so resumed executions cannot
  bypass the typed join-state boundary.
- Rejected malformed, negative, or non-numeric persisted join counters instead
  of silently converting them to zero; legacy short counter vectors are padded
  to the current join-node count at the shared decoder boundary.
- Preserved the existing persisted `_rt_join` field names and compatibility
  handling for legacy synchronous token shapes.
- Narrowed workflow ownership payload decoding to validated JSON objects rather
  than unchecked mapping casts.

## Verification

- Ruff passes on the changed runtime modules.
- Pyright reports zero errors and warnings on the changed runtime modules.
- The focused join-state and ownership regressions pass locally (`10 passed`)
  with `-p no:cacheprovider`.
- The previously run runtime ownership/async contract selection passed its
  ownership and non-optional async cases; optional Chroma cases require the
  uninstalled Chroma dependency.

## Release gate

This release must be built from the merged `main` commit and published only
after the CPython 3.12-3.14 and PyPy 3.11 CI jobs pass.
