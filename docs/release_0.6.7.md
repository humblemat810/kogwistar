# Kogwistar 0.6.7

## Highlights

- Adds the pure-vision RAG tutorial's curated mock, CPU CLIP, and GPU
  Qwen3-VL smoke paths.
- Documents multimodal graph grounding, profile isolation, model readiness,
  and sequential embedding/answer handoff.
- Adds TUI and Compose contract coverage without changing persisted graph,
  workflow, or embedding-profile semantics.

## Compatibility

This is a behavior-preserving patch release. It supports CPython 3.12-3.14 and
PyPy 3.11. The GPU tutorial path is optional and requires NVIDIA Container
Toolkit; the core package remains vendor-neutral.

## Release Gate

Publish only after the required CI matrix is green on CPython 3.12-3.14 and
PyPy 3.11, including lint, Rust, native-wheel, and SQLite invariant checks.
The release tag must match the package version:

```text
v0.6.7
```

Downstream repositories must pin the merged `v0.6.7` commit rather than this
feature branch before preparing their own releases.
