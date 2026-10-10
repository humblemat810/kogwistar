# Kogwistar 0.6.5

## Highlights

- Completes the cross-repository type-contract hardening for JSON, engine,
  runtime, storage, MCP, ontology, and adapter boundaries.
- Keeps recursive JSON contracts compatible with Python 3.11 and preserves the
  existing public graph, workflow, persistence, and MCP behavior.
- Retains the implementation-specific cache selection: Joblib on CPython and
  DiskCache on PyPy.

## Compatibility

This is intended as a behavior-preserving patch release. Persisted graph,
workflow, and protocol payloads remain compatible; the changes narrow static
contracts without introducing a new serialized schema.

## Release Gate

Publish only after PR #55 is merged into `main` and the required CI matrix is
green on CPython 3.12-3.14 and PyPy 3.11, including lint, Rust, native-wheel,
and SQLite invariant checks. The release tag must match the package version:

```text
v0.6.5
```

Downstream repositories must pin the merged `v0.6.5` commit rather than this
feature branch before preparing their own releases.
