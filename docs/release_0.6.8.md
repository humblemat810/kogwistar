# Kogwistar 0.6.8

## Highlights

- Adds canonical create-once admission for maintenance node mutations.
- Serializes maintenance identity admission with the authoritative metadata
  transaction: SQLite uses its writer transaction, Postgres uses a transaction
  advisory lock, and in-memory engines use the enclosing process-local lock.
- Rechecks ACL visibility after entering the maintenance unit of work and
  before applying lifecycle mutations.
- Carries expected entity-revision fences through graph-patch acceptance,
  requeue, and application paths so stale maintenance work fails closed.
- Keeps ordinary `add_node()` ingestion upsert behavior unchanged; the new
  create-once operation is an explicit maintenance capability.

## Compatibility

This is a behavior-preserving patch release for existing graph and storage
APIs. It supports CPython 3.12-3.14 and PyPy 3.11. Existing ACL-aware writer
implementations remain valid because create-once admission is optional; an
engine without the canonical event-store capability cannot claim atomic
maintenance admission.

## Validation

The release candidate must pass the required CI matrix on the merged commit:

- CPython 3.12, 3.13, and 3.14;
- PyPy 3.11;
- Ruff and type checks;
- Rust workspace, native-wheel, and SQLite invariant checks;
- canonical event admission and ACL protocol regression tests.

## Release Gate

Publish only after the merged `main` commit is green. The release tag must
match the package version:

```text
v0.6.8
```

Downstream repositories must pin the merged `v0.6.8` commit, not this feature
branch, before preparing their own releases.
