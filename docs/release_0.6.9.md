# Kogwistar 0.6.9

## Highlights

- Makes failed nested workflow children retryable from their durable
  checkpoints instead of treating a failed child result as an accepted
  terminal result during parent resume.
- Fences Stage-1 graph reads and two-stage projection handoffs against stale
  lifecycle state, including tombstoned and stale-revision entities before
  reconciliation.
- Preserves the existing append-only and tombstone lifecycle contracts while
  keeping explicit two-stage reconciliation behavior.
- Updates the Rust dependency lockfile to `quinn-proto 0.11.18` and
  `rustls 0.23.46`, resolving the current RustSec advisories.

## Compatibility

This is a behavior-preserving patch release for existing graph, storage, and
workflow APIs. It supports CPython 3.12-3.14 and PyPy 3.11. Existing
single-stage and two-stage backends remain supported; two-stage deployments
continue to require their normal reconciliation worker for promotion and
cleanup.

## Validation

The release candidate must pass the required CI matrix on the merged commit:

- CPython 3.12, 3.13, and 3.14;
- PyPy 3.11;
- Ruff and type checks;
- Rust workspace, native-wheel, SQLite, and dependency-audit checks;
- workflow resume, lifecycle fencing, ACL, and two-stage projection tests.

Local Rust verification for this candidate passed `cargo audit` and
`cargo test --workspace`.

## Release Gate

Publish only after the merged `main` commit is green. The release tag must
match the package version:

```text
v0.6.9
```

Downstream repositories must pin the merged `v0.6.9` commit, not this feature
branch, before preparing their own releases.
