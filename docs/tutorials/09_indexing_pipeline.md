# 09 Indexing Pipeline

Audience: Builder / integrator
Time: 20 minutes

## What You Will Build

You will build a mental model of how indexing, replay, and rebuildable projections fit around the core graph engine.

## Why This Matters

Search and indexing are often presented as magic. In this repo they are implementation layers around authoritative graph writes, which changes how you debug and recover them.

## Run or Inspect

- Inspect the indexing and CDC-related code paths in `graph_knowledge_engine/cdc/` and `tests/outbox/`.
- Read the event and repair-oriented helpers in `scripts/claw_runtime_loop.py`.
- Compare this page with [08 Storage Backends and Parity](./08_storage_backends_and_parity.md) so indexing is understood as a projection, not the source of truth.

## Inspect The Result

- Distinguish graph entities from serving indexes.
- Notice that replay and rebuild are expected operations, not emergency-only paths.
- Notice that failure recovery talks about re-materialization instead of hand-edited index state.

## Invariant Demonstrated

Indexes are rebuildable projections. The system treats authoritative writes and serving/search layers as related but not identical concerns.

## Next Tutorial

Continue to [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md).
