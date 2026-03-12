# RAG Level 3: Reinforced Event-Sourced Loop Control

Goal: show durable event loop behavior with guardrails (`ttl`) and optional CDC observability.

This level intentionally uses the existing claw runtime script instead of duplicating runtime logic.

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

## Inside The Engine

- Uses durable inbox/outbox rows in SQLite (`claw_events`).
- Distinguishes internal continuation (`route=self`) vs final output (`route=output`).
- Uses loop budget (`ttl`) to avoid unbounded continuation.

## Checkpoint (Pass/Fail)

Pass when:

- Event statuses advance correctly.
- Outbox entries exist after processing.
- TTL-limited runs terminate without infinite self-requeue.

Fail signals:

- Stuck `processing` rows.
- No outbox rows after `run-once`.

## Troubleshooting

- If events do not move, ensure `run-once` or `run-loop` was executed.
- If CDC page is static, verify `run-cdc-bridge` is active and WS URL matches.
- Use `--reset-oplog` when troubleshooting stale CDC streams.
