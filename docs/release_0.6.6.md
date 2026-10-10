# Kogwistar 0.6.6

## Highlights

- Includes the merged type-contract hardening from PR #56.
- Adds explicit protocols for stable tool, file-walker, workflow-metadata,
  index-worker, and ingester-cache boundaries.
- Preserves flexible parser, MCP reflection, and third-party constructor
  boundaries where runtime signatures are intentionally adaptive.
- Keeps persisted graph, workflow, MCP, and Rust/schema payload contracts
  behavior-compatible.

## Compatibility

This is a behavior-preserving patch release. It supports CPython 3.12-3.14 and
PyPy 3.11. No persisted schema migration is introduced.

## Release Gate

Publish only after the required CI matrix is green on CPython 3.12-3.14 and
PyPy 3.11, including lint, Rust, native-wheel, and SQLite invariant checks.
The release tag must match the package version:

```text
v0.6.6
```

Downstream repositories must pin the merged `v0.6.6` commit rather than the
feature branch before preparing their own releases.
