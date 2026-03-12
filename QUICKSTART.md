# Standalone Quickstart

This guide is for running the backend by itself (no Designer app integration).

## 1) Prerequisites

- Python `3.10+`
- `pip`
- (Optional) Docker Desktop for container setup

## 2) Local Setup (No Docker)

From repo root:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -e ".[chroma]"
```

Create `.env` (or export env vars in shell):

```env
AUTH_MODE=dev
HOST=127.0.0.1
PORT=28110
GKE_BACKEND=chroma
GKE_PERSIST_DIRECTORY=./.gke-data
```

Start server:

```powershell
knowledge-mcp
```

## 3) Verify Server Is Up

```powershell
Invoke-RestMethod http://localhost:28110/health
```

Expected: JSON with `"ok": true`.

## 4) Generate a Dev Token (Standalone Auth)

```powershell
$token = (Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:28110/auth/dev-token `
  -ContentType "application/json" `
  -Body '{"username":"dev","role":"rw","ns":"docs"}'
).token

$token
```

Validate token works:

```powershell
Invoke-RestMethod `
  -Uri http://localhost:28110/api/auth/me `
  -Headers @{ Authorization = "Bearer $token" }
```

## 5) Docker Setup (Standalone)

Run backend with embedded Chroma and dev auth:

```powershell
$env:AUTH_MODE="dev"
docker compose --profile chroma up -d app-chroma
```

Check health:

```powershell
Invoke-RestMethod http://localhost:28110/health
```

Stop:

```powershell
docker compose down
```

## 6) Common Issues

- If port `28110` is busy, set another `PORT` and use the same port in your requests.
- If auth calls fail, confirm `AUTH_MODE=dev` in your running process/container.
- If you previously changed auth schema, remove stale local `auth.sqlite` and restart.

## 7) Standalone Claw-Style Runtime Loop (No Server)

This uses a custom resolver + `WorkflowRuntime` wrapper script, with persistent input events in SQLite.

Script: `scripts/claw_runtime_loop.py`

### Setup

Use the same environment from section 2, then initialize runtime storage/design:

```powershell
python scripts/claw_runtime_loop.py init --data-dir .gke-data/claw-loop
```

### Enqueue Input Events (persisted)

```powershell
python scripts/claw_runtime_loop.py enqueue `
  --data-dir .gke-data/claw-loop `
  --conversation-id conv-demo `
  --event-type user.message `
  --payload '{"text":"hello claw"}'
```

### Run Once (process one event)

```powershell
python scripts/claw_runtime_loop.py run-once --data-dir .gke-data/claw-loop
```

### Run Continuous Loop (OpenClaw-like worker)

```powershell
python scripts/claw_runtime_loop.py run-loop `
  --data-dir .gke-data/claw-loop `
  --sleep-ms 500
```

Stop with `Ctrl+C`.

### Inspect Inbox/Outbox

```powershell
python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction in --limit 20
python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction out --limit 20
```

`direction=in` rows are ingested input events (`pending|processing|done|failed`).
`direction=out` rows are emitted output events from `ClawResolver.persist_outbox`.

### CDC Bridge + Browser Workflow Page

Start CDC bridge:

```powershell
python scripts/claw_runtime_loop.py run-cdc-bridge --host 127.0.0.1 --port 8787 --reset-oplog
```

Initialize runtime with CDC publish endpoint:

```powershell
python scripts/claw_runtime_loop.py init `
  --data-dir .gke-data/claw-loop `
  --cdc-publish-endpoint http://127.0.0.1:8787/ingest
```

Render CDC-enabled pages (includes `workflow.bundle.html`):

```powershell
python scripts/claw_runtime_loop.py render-cdc-pages `
  --data-dir .gke-data/claw-loop `
  --out-dir .cdc_debug/pages `
  --cdc-ws-url ws://127.0.0.1:8787/changes/ws
```

Seed notebook-style background hypergraph data (primitive + edge->edge + node->edge):

```powershell
python scripts/claw_runtime_loop.py seed-background --data-dir .gke-data/claw-loop
```

Run provenance/span correction using built-in extraction matching helpers:

```powershell
python scripts/claw_runtime_loop.py repair-provenance `
  --data-dir .gke-data/claw-loop `
  --doc-id doc:background:hypergraph:001
```

Run full beginner walkthrough (threaded blocking input runtime, max demo loops guardrail = 2):

```powershell
python scripts/claw_runtime_loop.py tutorial --data-dir .gke-data/claw-loop --open-browser --max-demo-loops 2
```

### TTL and Clock Events

- Use payload `ttl` to cap self-loop continuation and prevent infinite loops.
- In this tutorial, `ttl` means loop budget (times to loop), not wall-clock expiry.
- If you also want time-based expiry, add a separate payload field (for example `expires_at_ms`).
- Use `clock.tick` to trigger polling behavior:

```powershell
python scripts/claw_runtime_loop.py enqueue-clock --data-dir .gke-data/claw-loop --conversation-id __clock__ --ttl 1
```

- Or auto-generate periodic clock events in worker loop:

```powershell
python scripts/claw_runtime_loop.py run-loop `
  --data-dir .gke-data/claw-loop `
  --clock-interval-ms 3000
```
