# Test Fixture Consolidation

## Implemented

- Moved auth env constants into dedicated helper modules instead of `conftest`.
- Moved shared test network helper into `tests/net_helpers.py`.
- Moved large MCP adjudication sample payload into `tests/graph_sample_data.py`.
- Moved reusable graph seed/build helpers into `tests/graph_seed_helpers.py`.
- Kept `tests/conftest.py` as fixture wiring layer, with thin wrappers for shared helpers.

## Remaining Cleanup

- Split `tests/conftest.py` by concern:
  - env/bootstrap
  - embedding/backend fixtures
  - pg/chroma/container fixtures
  - graph sample fixtures
- Audit package-level `conftest.py` files and remove any fixture shadowing that duplicates root fixtures.
- Consolidate only truly identical `make_node` / `make_edge` helpers. Keep file-local builders when semantics differ.
- Replace direct imports from `tests.conftest` with direct imports from helper modules where practical.
- Rename typo-prone fixture names only with compatibility shims or a single cleanup pass.

## Guardrails

- Do not change fixture semantics while moving code.
- Prefer wrapper fixtures during migration to avoid broad test churn.
- Run focused regression tests after each extraction step, especially auth, MCP, workflow SSE, and bundle-render tests.
