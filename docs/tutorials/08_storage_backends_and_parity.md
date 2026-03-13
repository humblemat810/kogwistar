# 08 Storage Backends and Parity

Audience: Builder / integrator
Time: 15-20 minutes

## What You Will Build

You will build a decision model for when to use the lightweight local stack versus stronger transactional backends, and how parity is enforced by tests instead of marketing claims.

## Why This Matters

Advanced readers will not trust backend abstraction claims unless the repo explains what stays invariant and what changes operationally.

## Run or Inspect

- Inspect backend implementations under `graph_knowledge_engine/engine_core/`.
- Inspect parity-oriented tests under `tests/outbox/`, `tests/pg_sql/`, and `tests/kg_conversation/`.
- Compare this page with the persistence assumptions in [01 Hello Graph Engine](./01_hello_graph_engine.md) and the replay-oriented claims in [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md).

## Inspect The Result

- Note where the repo positions Chroma or SQLite as pragmatic local persistence.
- Note where it positions pgvector or Postgres for stronger transactional guarantees.
- Notice that replay and parity tests describe the contract more clearly than any backend slogan could.

## Invariant Demonstrated

Behavioral contracts are enforced above the storage choice. Backends may differ operationally, but the repo treats parity as a tested requirement.

## Next Tutorial

Continue to [09 Indexing Pipeline](./09_indexing_pipeline.md).
