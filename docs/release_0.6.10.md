# Kogwistar 0.6.10

## Highlights

- Corrects the public `kogwistar.__version__` value so it matches the
  published package metadata and release tag.
- Keeps the workflow retry, lifecycle fencing, and Rust dependency fixes from
  `0.6.9` unchanged.

## Compatibility

This is a release-metadata patch for CPython 3.12-3.14 and PyPy 3.11. Existing
graph, storage, workflow, ACL, and two-stage projection APIs are unchanged.

## Validation

The release candidate must pass the required CI matrix on the merged commit:

- CPython 3.12, 3.13, and 3.14;
- PyPy 3.11;
- Ruff, type checks, Rust workspace, native-wheel, SQLite, and dependency-audit
  checks;
- package metadata and public module-version parity checks.

## Release Gate

Publish only after the merged `main` commit is green. The release tag must
match the package version:

```text
v0.6.10
```

Downstream repositories must pin the merged `v0.6.10` commit, not this feature
branch, before preparing their own releases.
