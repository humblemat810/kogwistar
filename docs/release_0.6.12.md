# Kogwistar 0.6.12

## Runtime contract hardening

- Added a shared typed contract for persisted workflow join snapshots.
- Centralized sync and async join-state decoding with fail-closed validation of
  node IDs, counters, waiters, pending tokens, and suspended tokens.
- Routed `resume_run()` through the same decoder so resumed executions cannot
  bypass the typed join-state boundary.
- Preserved the existing persisted `_rt_join` field names and compatibility
  handling for legacy synchronous token shapes.
- Narrowed workflow ownership payload decoding to validated JSON objects rather
  than unchecked mapping casts.

## Verification

- Ruff passes on the changed runtime modules.
- Pyright reports zero errors and warnings on the changed runtime modules.
- The runtime ownership, async contract, and bijection suites pass locally with
  `-p no:cacheprovider`.

## Release gate

This release must be built from the merged `main` commit and published only
after the CPython 3.12-3.14 and PyPy 3.11 CI jobs pass.
