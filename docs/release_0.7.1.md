# Kogwistar 0.7.1

## ADR-015 capability gate

- Complete the ADR-015 six-capability local CI gate across PostgreSQL event
  history, projections/run registry, queues/leases/lanes, PostgreSQL/pgvector,
  workflow runtime, and REST/SSE/MCP/CLI.
- Preserve the Python-authority rollback path and explicit
  `rust_cutover_ready` ledger semantics.
- Fix the direct-import circular boundary in `_rust_bridge` and native SQLite
  transaction ownership checks.

## Package support

- Keep `Requires-Python >=3.11`, CPython native wheels, and the PyPy 3.11
  Python-authority compatibility profile unchanged.
