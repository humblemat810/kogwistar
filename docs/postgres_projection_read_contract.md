# PostgreSQL Projection Read Contract

`PgVectorBackend` exposes Chroma-shaped collection reads for PostgreSQL and
pgvector. This document records the current read semantics; it does not grant
any authorization by itself.

## Read Shape

`get()` always selects the identity, document, and metadata columns needed to
construct the result. The `embedding` column is selected only when the caller
includes `"embeddings"` in `include`.

This applies to both synchronous `_get_flat()` and asynchronous
`_get_flat_async()` paths. Omitting embeddings is a data-minimization and
memory-safety measure; it is not an ACL decision. Graph, namespace, tenant, and
caller authorization must still be enforced before the backend is called.

## Limit Semantics

- An explicit positive `limit` adds a SQL `LIMIT` clause.
- `limit=None` deliberately omits SQL `LIMIT` and reads all rows matching the
  supplied `ids`/metadata filters.
- The collection facade keeps its normal bounded default (`200`). Unlimited
  reads therefore require an explicit `None`; they are not implicit.
- Result ordering is not a stable API guarantee unless the caller uses a
  higher-level ordered projection/query contract.

Unlimited reads are intended for trusted projection repair, rebuild, and
administrative paths with a known resource budget. REST, MCP, and other
untrusted request surfaces must validate a finite limit or expose a bounded
cursor/page API. A database read without `LIMIT` is not an authorization grant
and must never bypass ACL, tenant, workspace, project, or namespace checks.
The backend does not enforce that service-boundary policy; each exposed API
must enforce it before calling the backend.

## Security Invariants

1. Authorization and scope filtering happen before backend read execution.
2. Caller-controlled `where` values are bound through the existing SQLAlchemy
   query builder; they are not interpolated into SQL text.
3. `include` is an output projection only. Requesting or omitting embeddings
   cannot widen graph or tenant visibility.
4. Embeddings may be sensitive derived data; callers receive them only when
   the operation explicitly requires them and policy allows it.
5. Unlimited reads must be auditable and bounded by the owning workflow,
   repair job, or administrative policy for rows, memory, and wall time.

The backend remains a storage adapter. Authentication, ACL, delegation,
resource budgets, and audit/provenance belong to the caller's governed engine
or service boundary.

## Sync/Async Parity

Synchronous and asynchronous flat reads share the same selected-column,
filter, and limit semantics. Async execution changes only how the connection
is awaited; it does not provide a weaker authorization or a different result
contract.

The regression suite
`tests/pg_sql/test_postgres_unlimited_reads.py` verifies:

- `limit=None` omits `LIMIT` in sync and async SQL;
- explicit limits remain present; and
- embeddings are excluded unless requested, while remaining available when
  requested.

The suite uses SQLAlchemy statement inspection. It does not replace a live
PostgreSQL/pgvector authorization and resource-budget integration test.
