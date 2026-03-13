# RAG Level 3: Reinforced Event-Sourced Loop Control

Goal: show durable event loop behavior with guardrails (`ttl`) and optional CDC observability.

This level intentionally uses the existing claw runtime script instead of duplicating runtime logic.

## What You Will Build

You will enqueue a message into the claw loop, process it through the durable inbox/outbox flow, and optionally connect the same run to the CDC viewer bridge.

## Why This Matters

This is the RAG-side bridge into event sourcing. It shows that retrieval-oriented behavior can still live inside a durable, inspectable loop instead of an opaque retry wrapper.

## Run or Inspect

## Quick Run

```powershell
python scripts/claw_runtime_loop.py init --data-dir .gke-data/claw-loop

python scripts/claw_runtime_loop.py enqueue `
  --data-dir .gke-data/claw-loop `
  --conversation-id conv-demo `
  --event-type user.message `
  --payload '{"text":"hello claw","ttl":2}'

python scripts/claw_runtime_loop.py run-once --data-dir .gke-data/claw-loop

python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction in --limit 10
python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction out --limit 10
```

Expected behavior:

- Input event transitions through `pending -> processing -> done|failed`.
- Outbox rows are appended for emitted outputs.

## Inspect The Result

- Inspect the inbox and outbox rows after `run-once`.
- Confirm the event does not recurse forever when `ttl` reaches zero.
- Compare these event rows with the richer trace sink story in [Runtime Level 3 - CDC Viewer and LangGraph Interop](./runtime-level-3-observability-interop.md).

## Inside The Engine

- Uses durable inbox/outbox rows in SQLite (`claw_events`).
- Distinguishes internal continuation (`route=self`) vs final output (`route=output`).
- Uses loop budget (`ttl`) to avoid unbounded continuation.

## Optional CDC Extension

```powershell
python scripts/claw_runtime_loop.py run-cdc-bridge --host 127.0.0.1 --port 8787 --reset-oplog

python scripts/claw_runtime_loop.py init `
  --data-dir .gke-data/claw-loop `
  --cdc-publish-endpoint http://127.0.0.1:8787/ingest

python scripts/claw_runtime_loop.py render-cdc-pages `
  --data-dir .gke-data/claw-loop `
  --out-dir .cdc_debug/pages `
  --cdc-ws-url ws://127.0.0.1:8787/changes/ws
```

## Checkpoint

Pass when:

- Event statuses advance correctly.
- Outbox entries exist after processing.
- TTL-limited runs terminate without infinite self-requeue.

## Invariant Demonstrated

Loop control is durable and inspectable. The system records state transitions instead of burying them inside an agent retry loop.

## Troubleshooting

- If events do not move, ensure `run-once` or `run-loop` was executed.
- If CDC page is static, verify `run-cdc-bridge` is active and WS URL matches.
- Use `--reset-oplog` when troubleshooting stale CDC streams.

## Next Tutorial

Return to [10 Event Log Replay and CDC](./10_event_log_replay_and_cdc.md) or continue deeper into [Runtime Ladder Overview](./runtime-ladder-overview.md).
