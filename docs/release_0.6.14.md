# Kogwistar 0.6.14

## Typed backend and workflow-state contracts

- Fail closed when persisted workflow join snapshots are malformed instead of
  treating invalid state as an empty or default snapshot.
- Use explicit structural capability protocols for asynchronous storage,
  collection operations, atomic mutation, and two-stage projection dispatch.
- Keep backend-specific operations behind declared interfaces rather than
  inferring known capabilities from implementation details.
- Add regression coverage for backend dispatch, mutation capabilities,
  projection adapters, and malformed persisted join state.

## Verification

- Ruff passes on the changed core modules and focused tests.
- Focused non-Chroma core tests pass locally (`40 passed`, `5 deselected`).
- Optional Chroma integration tests require the `chromadb` extra and were not
  run in this local environment.
- Pyright and the complete CPython/PyPy CI matrix remain release gates; do not
  tag or publish until they pass on the merged release commit.
