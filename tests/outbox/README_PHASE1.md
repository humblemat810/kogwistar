# Phase 1: IndexJobs Outbox + Reconciler (Join-like Indexes)

This bundle contains the Phase 1 implementation that makes join-like derived indexes crash-safe / self-healing via a durable `index_jobs` queue with leases.

## Files
- engine_sqlite.py: adds `index_jobs` table + enqueue/claim/mark APIs (SQLite)
- engine_postgres_meta.py: adds `index_jobs` table + enqueue/claim/mark APIs (Postgres)
- engine.py: adds enqueue helpers + `reconcile_indexes()` drainer + apply handlers; wires node/edge upserts and tombstone/redirect to enqueue derived index jobs.

## Semantics
- Outbox-first: enqueue jobs is the correctness mechanism.
- Fast path: enqueue + immediate reconcile is an optimization.
- Leases: prevent "DOING forever"; jobs can be stolen after `lease_until`.

## Phase 1 index kinds
- node_docs
- node_refs
- edge_refs
- edge_endpoints

Vector index work is intentionally out of Phase 1.
