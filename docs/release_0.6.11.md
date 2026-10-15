# Kogwistar 0.6.11

## Highlights

- Releases durable workflow ownership fencing for resumable runs.
- Releases a failed run's ownership lease when a durable checkpoint exists, so
  a subsequent worker can safely resume instead of receiving a false live-owner
  conflict.
- Preserves compare-and-swap generation fencing, lease renewal, and exclusive
  ownership for concurrent resume attempts.

## Compatibility

This is a behavior-preserving patch release for existing graph, storage,
workflow, ACL, and projection APIs. It supports CPython 3.12-3.14 and PyPy
3.11. A failed run without a durable checkpoint remains terminal; only a run
with a verified checkpoint becomes resumable.

## Validation

The release candidate must pass the required CI matrix on the merged commit:

- CPython 3.12, 3.13, and 3.14;
- PyPy 3.11;
- Ruff, type checks, Rust workspace, native-wheel, SQLite, and dependency-audit
  checks;
- workflow ownership, checkpoint recovery, and package-version parity tests.

Local validation for this candidate passed the focused ownership/recovery tests
and the CI-equivalent CPython suite (`946 passed, 22 skipped, 1193 deselected`).

## Release Gate

Publish only after the merged `main` commit is green. The release tag must
match the package version:

```text
v0.6.11
```

Downstream repositories must pin the merged `v0.6.11` commit, not this feature
branch, before preparing their own releases.
