# ADR-004: Phase 4 -- pgvector as Authoritative Backend

## Status

Accepted

## Context

Phase 4 formalizes pgvector as a first-class backend integrated into the
transactional contract of the system.

Previous phases established: - Unit-of-Work (UoW) transaction
boundaries - Event log append semantics - Replay and repair mechanisms -
Backend abstraction parity

This ADR selects Option A: pgvector is authoritative within the
transactional boundary.

------------------------------------------------------------------------

## Decision

pgvector rows are included in the same atomic contract as:

-   Nodes
-   Edges
-   Metadata
-   Event log entries
-   Index job enqueue operations

All must succeed or fail together inside a `uow()` transaction.

------------------------------------------------------------------------

## Transactional Invariants

### 1. Commit Success Invariant

If `uow()` commits successfully:

-   Node and edge rows exist.
-   Metadata reflects merge rules.
-   Event log entry exists.
-   pgvector row exists.
-   Index job enqueue row exists (if applicable).

These must be durable and visible after commit.

------------------------------------------------------------------------

### 2. Rollback Invariant

If `uow()` raises or fails:

-   None of the above side effects are visible.
-   No partial state may remain.

------------------------------------------------------------------------

### 3. Replay Consistency Invariant

Replay from the event log must reconstruct a consistent state.

Replay may be used for: - Repair - Reindex - Disaster recovery - Drift
correction

Replay is not the primary correctness mechanism.\
Transactional atomicity is the primary guarantee.

The event log is the ultimate audit truth.

------------------------------------------------------------------------

### 4. Shard Boundary Invariant

Atomicity is guaranteed within a shard boundary.

Future sharding strategy: - Knowledge graph may shard by domain/topic. -
Writes are slow and localized. - Reads dominate.

Cross-shard operations are out of scope for this phase.

------------------------------------------------------------------------

### 5. Read Semantics Outside UoW

Outside a UoW:

-   Reads follow PostgreSQL READ COMMITTED semantics.
-   Each statement sees the latest committed data at statement start.
-   Snapshot consistency across multiple statements requires an explicit
    transaction.

No stronger read guarantees are provided in this phase.

------------------------------------------------------------------------

## Rebuild and Repair Policy

Rebuild operations are allowed and supported.

Rebuild is used for: - Index re-creation - Repair after detected
corruption - Migration - Operational recovery

Rebuild must be idempotent and safe to resume.

Rebuild does not weaken the transactional contract.

------------------------------------------------------------------------

## Rationale

Including pgvector inside the transactional boundary provides:

-   Immediate consistency between entities and vectors
-   Elimination of projection drift
-   Strong rollback guarantees
-   Simpler mental model
-   Deterministic testable invariants

------------------------------------------------------------------------

## Phase 4 Validation Tests

1.  Commit success E2E test
2.  Forced rollback E2E test
3.  Deterministic vector ordering test
4.  Replay rebuild test
5.  Metadata merge equivalence test across backends

------------------------------------------------------------------------

## Final Statement

Phase 4 defines pgvector as an authoritative component of the
transactional contract.

Atomicity is guaranteed within a shard. Replay is the repair mechanism.
The event log is the ultimate audit truth. Consistency is enforced at
commit time.
