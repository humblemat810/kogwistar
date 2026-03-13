# 10 Event Log Replay and CDC

Audience: Advanced / contributor
Time: 25 minutes
Companion: [scripts/tutorial_sections/10_event_log_replay_and_cdc.py](../../scripts/tutorial_sections/10_event_log_replay_and_cdc.py)

## What You Will Build

You will inspect runtime trace events, event-sourced loop control, and the viewer-facing CDC surfaces that make replay and debugging concrete.

## Why This Matters

This is one of the repo's flagship ideas. Deterministic replay, append-only execution, and viewer-friendly CDC are much more persuasive when shown together than when scattered across separate docs.

## Run or Inspect

- Run the companion file to inspect runtime level 3 output plus claw loop command hints.
- Follow the deeper execution docs:
  - [Runtime Level 3 - CDC Viewer and LangGraph Interop](./runtime-level-3-observability-interop.md)
  - [RAG Level 3 - Event-Sourced Loop Control](./level-3-event-loop-control.md)
- Inspect replay-related tests under `tests/outbox/`.

## Inspect The Result

- Confirm trace event types cover run, step, route, join, and checkpoint lifecycle.
- Confirm the bundled viewer asset path exists.
- Confirm the claw loop examples expose inbox/outbox transitions instead of hidden background behavior.

## Invariant Demonstrated

Execution history is inspectable and replay-friendly. The system exposes causal traces instead of asking you to trust invisible orchestration.

## Next Tutorial

Continue to [11 Build a Mini GraphRAG App](./11_build_a_mini_graphrag_app.md).
