# 08 Storage Backends and Parity

Audience: Builder / integrator
Time: 15-20 minutes

## What You Will Build

You will build a decision model for when to use the lightweight local stack versus stronger transactional backends, and how parity is enforced by tests instead of marketing claims.

## Why This Matters

Advanced readers will not trust backend abstraction claims unless the repo explains what stays invariant and what changes operationally.

## Run or Inspect

- Inspect backend implementations under `kogwistar/engine_core/`.
- Inspect parity-oriented tests under `tests/outbox/`, `tests/pg_sql/`, and `tests/kg_conversation/`.
- Compare this page with the persistence assumptions in [01 Hello Graph Engine](./01_hello_graph_engine.md) and the replay-oriented claims in [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md).

## Inspect The Result

- Note where the repo positions Chroma or SQLite as pragmatic local persistence.
- Note where it positions pgvector or Postgres for stronger transactional guarantees.
- Notice that replay and parity tests describe the contract more clearly than any backend slogan could.

## Invariant Demonstrated

Behavioral contracts are enforced above the storage choice. Backends may differ operationally, but the repo treats parity as a tested requirement.

The repo now also has a backend-contract smoke suite. If you are adding or swapping a backend, start there:

- `fake` is the fast CI double. It uses in-memory collections plus the real SQLite metastore on a temp path.
- `chroma` and `pg` are broader coverage paths and should keep passing the same contract checks.
- New backends should prove they satisfy the common contract before they are used by higher-level conversation, MCP, or workflow tests.

### Current Fake vs Real Behavior

The fake backend is intentionally close to Chroma-shaped behavior, but it is not a persistence clone.

- `fake` is safe for in-process tests that only need the collection contract, read/query shapes, or deterministic orchestration.
- `fake` does not survive a subprocess reopen the way `chroma` or `pg` do.
- `fake` can participate in parity tests when the assertion is structural or semantic, but it may diverge on generated IDs or backend-owned trace rows.
- Some parity cases remain real-only because the fake path does not yet reproduce the same workflow trace semantics. In this repo, that includes the stricter backbone-only conversation parity case.

Practical rule:

- use `fake` for lightweight CI coverage and structural parity
- keep `chroma` and `pg` for persistence, subprocess reopen, and backend-specific trace semantics
- when a `fake` row is added, mark it explicitly as `ci` and keep the real rows as `ci_full`

## Next Tutorial

Continue to [09 Indexing Pipeline](./09_indexing_pipeline.md).
