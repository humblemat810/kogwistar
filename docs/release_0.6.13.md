# Kogwistar 0.6.13

## Chroma metadata compatibility

- Adapt null-valued metadata at the Chroma adapter boundary. Chroma does not
  accept null metadata values; optional null keys are omitted while supported
  values are preserved.
- Preserve the caller's metadata objects and existing empty-list encoding.
- Add regressions for metadata normalization and the actual collection-call
  boundary.

## Verification

- Focused projection-capability tests pass locally (`26 passed`).
- A real ephemeral Chroma write/read with `wf_predicate=None` succeeds and
  preserves the remaining metadata.
- Ruff and `git diff --check` pass on changed files.
- Pyright reports zero errors and warnings on the changed production module.
- Full supported-interpreter CI must pass on the merged release candidate before
  tagging or publishing.
