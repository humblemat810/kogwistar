# 14 Architecture Deep Dive

Audience: Advanced / contributor
Time: 25-30 minutes

## What You Will Build

You will build the full repo-level picture: why knowledge graph, conversation graph, and workflow runtime are separate subsystems and how they compose.

## Why This Matters

This tutorial comes last on purpose. Once you have runnable intuition, the larger architectural split is easier to evaluate on its merits instead of sounding like abstraction for its own sake.

## Run or Inspect

- Read [docs/LEARNING_PATH.md](../LEARNING_PATH.md) again after finishing the numbered tutorials.
- Inspect `ZEN.md` for the author's framing and `kogwistar/docs/` for the formal design records.
- Revisit [11 Build a Mini GraphRAG App](./11_build_a_mini_graphrag_app.md) and [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md) while reading the deeper architecture docs.
- Follow with [26 Recovery and Durable Operational State](./26_recovery_and_durable_operational_state.md) if you want the operator-facing picture of bounded repair, redelivery, and latest-state projections.

## Inspect The Result

- Separate the workflow design graph from the conversation graph in your head.
- Separate knowledge retrieval from conversation-side projection and pinning.
- Separate authoritative truth from durable latest-state projections used for recovery and operator views.
- Notice that replay, provenance, and future wisdom layers are easier to reason about because the boundaries are explicit.

## Invariant Demonstrated

Subsystem boundaries are intentional. The architecture keeps different sources of truth separate so replay, audit, and future learning layers remain tractable.

## Next Tutorial

Continue to [26 Recovery and Durable Operational State](./26_recovery_and_durable_operational_state.md) for the restart and operator-repair view, or return to [docs/LEARNING_PATH.md](../LEARNING_PATH.md) and choose the execution ladder or subsystem area you want to go deeper on.
