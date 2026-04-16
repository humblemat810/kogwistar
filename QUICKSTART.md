# Standalone Quickstart

This guide is for running the backend by itself (no Designer app integration).

## 1) Prerequisites

- Python `3.13`
- `pip`
- (Optional) Docker Desktop for container setup

## 1.5) [Hypergraph-RAG QA example](./docs/tutorials/level-0-simple-rag.md)

## 2) Local Setup (No Docker)

Clone the repo and enter it:

```bash
git clone git@github.com:humblemat810/kogwistar.git
# Or use HTTPS:
# git clone https://github.com/humblemat810/kogwistar.git
cd kogwistar
```

From repo root:

```bash
python -m venv .venv
. ./.venv/bin/activate
pip install -e ".[server,chroma]"
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

```
knowledge-mcp
```

### VS Code Debug Button Equivalent

If you want a one-click VS Code launch config for local development, add this
entry to the project's
`.vscode/launch.json`:

```json
{
  "name": "Start Kogwistar Server (28110)",
  "type": "debugpy",
  "request": "launch",
  "module": "kogwistar.server_mcp_with_admin",
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "envFile": "${workspaceFolder}/.env",
  "env": {
    "HOST": "127.0.0.1",
    "PORT": "28110",
    "AUTH_MODE": "dev",
    "JWT_SECRET": "dev-secret",
    "JWT_ALG": "HS256",
    "GKE_BACKEND": "chroma",
    "GKE_PERSIST_DIRECTORY": "${workspaceFolder}/.tmp_vscode_server/gke",
    "MCP_CHROMA_DIR": "${workspaceFolder}/.tmp_vscode_server/mcp/docs",
    "MCP_CHROMA_DIR_CONVERSATION": "${workspaceFolder}/.tmp_vscode_server/mcp/conversation",
    "MCP_CHROMA_DIR_WORKFLOW": "${workspaceFolder}/.tmp_vscode_server/mcp/workflow",
    "MCP_CHROMA_DIR_WISDOM": "${workspaceFolder}/.tmp_vscode_server/mcp/wisdom",
    "DEV_AUTH_NS": "docs,conversation,workflow,wisdom",
    "DEV_AUTH_ROLE": "rw"
  }
}
```

What this does:

- Starts the same `kogwistar.server_mcp_with_admin` server surface as the
  `knowledge-mcp` command, but from the VS Code debug button.
- Uses `AUTH_MODE=dev` so you can mint local dev tokens without setting up
  OIDC first.
- Grants the common local namespaces in `DEV_AUTH_NS` so chat, workflow, and
  docs routes are all reachable.
- Binds to `127.0.0.1:28110`, which is the repo's default dev port.
- Pins persistence and Chroma directories under `.tmp_vscode_server` so the
  debug run uses an isolated local data directory.

What the host project needs:

- `kogwistar` installed in the selected Python environment.
- A `.env` file if you want to layer extra settings through `envFile`.
- If you want the default UI integration, a frontend running at
  `http://127.0.0.1:5173/`. If not, you can still use the API directly.

## 3) Verify Server Is Up

```bash
curl http://localhost:28110/health
```

Expected: JSON with `"ok": true`.

## 4) Generate a Dev Token (Standalone Auth)

```bash
token=$(curl -s \
  -X POST http://localhost:28110/auth/dev-token \
  -H "Content-Type: application/json" \
  -d '{"username":"dev","role":"rw","ns":"docs"}' | python -c 'import json,sys; print(json.load(sys.stdin)["token"])')

printf '%s\n' "$token"
```

Validate token works:

```bash
curl http://localhost:28110/api/auth/me \
  -H "Authorization: Bearer $token"
```

## 5) Docker Setup (Standalone)

Run backend with embedded Chroma and dev auth:

```bash
export AUTH_MODE=dev
docker compose --profile chroma up -d app-chroma
```

Check health:

```bash
curl http://localhost:28110/health
```

Stop:

```bash
docker compose down
```

## 6) Common Issues

- If port `28110` is busy, set another `PORT` and use the same port in your requests.
- If auth calls fail, confirm `AUTH_MODE=dev` in your running process/container.
- If you previously changed auth schema, remove stale local `auth.sqlite` and restart.

## 7) Standalone Claw-Style Runtime Loop (No Web Server, loop count capped)

This uses a custom resolver + `WorkflowRuntime` wrapper script, with persistent input events in SQLite.

Treat this as a minimal substrate example for OpenClaw-style platforms, not as a finished OpenClaw wrapper or a production-ready security product. The point of this section is to show that Kogwistar can sit underneath governed agent loops, not only underneath memory or workflow demos.

If you want the more concrete real-world example of that openclaw governance (not the openclaw like gateway loop implemnetation in this repo), see [`humblemat810/cloistar`](https://github.com/humblemat810/cloistar), which uses Kogwistar as the substrate for an OpenClaw governance layer.

Script: `scripts/claw_runtime_loop.py`

### Setup

Use the same environment from section 2, then initialize runtime storage/design:

```bash
python scripts/claw_runtime_loop.py init --data-dir .gke-data/claw-loop
```

### Enqueue Input Events (persisted)

```bash
python scripts/claw_runtime_loop.py enqueue \
  --data-dir .gke-data/claw-loop \
  --conversation-id conv-demo \
  --event-type user.message \
  --payload '{"text":"hello claw"}'
```

### Run Once (process one event)

```bash
python scripts/claw_runtime_loop.py run-once --data-dir .gke-data/claw-loop
```

### Run Continuous Loop (OpenClaw-like worker)

```bash
python scripts/claw_runtime_loop.py run-loop \
  --data-dir .gke-data/claw-loop \
  --sleep-ms 500
```

Stop with `Ctrl+C`.

### Inspect Inbox/Outbox

```bash
python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction in --limit 20
python scripts/claw_runtime_loop.py list-events --data-dir .gke-data/claw-loop --direction out --limit 20
```

`direction=in` rows are ingested input events (`pending|processing|done|failed`).
`direction=out` rows are emitted output events from `ClawResolver.persist_outbox`.

### CDC Bridge + Browser Workflow Page

Start CDC bridge:

```bash
python scripts/claw_runtime_loop.py run-cdc-bridge --host 127.0.0.1 --port 8787 --reset-oplog
```

Initialize runtime with CDC publish endpoint:

```bash
python scripts/claw_runtime_loop.py init \
  --data-dir .gke-data/claw-loop \
  --cdc-publish-endpoint http://127.0.0.1:8787/ingest
```

Render CDC-enabled pages (includes `workflow.bundle.html`):

```bash
python scripts/claw_runtime_loop.py render-cdc-pages \
  --data-dir .gke-data/claw-loop \
  --out-dir .cdc_debug/pages \
  --cdc-ws-url ws://127.0.0.1:8787/changes/ws
```

Seed notebook-style background hypergraph data (primitive + edge->edge + node->edge):

```bash
python scripts/claw_runtime_loop.py seed-background --data-dir .gke-data/claw-loop
```

Run provenance/span correction using built-in extraction matching helpers:

```bash
python scripts/claw_runtime_loop.py repair-provenance \
  --data-dir .gke-data/claw-loop \
  --doc-id doc:background:hypergraph:001
```

Run full beginner walkthrough (threaded blocking input runtime, max internal self-requeue guardrail = 2):

```bash
python scripts/claw_runtime_loop.py tutorial --data-dir .gke-data/claw-loop --open-browser --max-demo-loops 2
```

Tutorial resolver policy (current):

- `route=self` is the only path that may auto-enqueue continuation.
- `route=output` never auto-enqueues, even if `next_payload` exists.
- If `route=output` includes `next_payload`, it is stored as `deferred_next_payload` in output metadata for audit/replay.
- Deferred payload does not auto-wake the loop; a future external event (`user.message`, `clock.tick`, etc.) is required.
- If `route=self` omits `next_payload`, runtime synthesizes a minimal continuation payload and continues when TTL/budget allow.
- Internal continuation events are appended to queue tail (FIFO fairness).
- `--max-demo-loops` counts only internal self-requeues, not all processed events.

### TTL and Clock Events

- Use payload `ttl` to cap self-loop continuation and prevent infinite loops.
- In this tutorial, `ttl` means loop budget (times to loop), not wall-clock expiry.
- If you also want time-based expiry, add a separate payload field (for example `expires_at_ms`).
- Use `clock.tick` to trigger polling behavior:

```bash
python scripts/claw_runtime_loop.py enqueue-clock --data-dir .gke-data/claw-loop --conversation-id __clock__ --ttl 1
```

- Or auto-generate periodic clock events in worker loop:

```bash
python scripts/claw_runtime_loop.py run-loop \
  --data-dir .gke-data/claw-loop \
  --clock-interval-ms 3000
```

## 8) RAG Ladder and Runtime Ladder

For staged walkthroughs:

- RAG Ladder (baseline RAG, retrieval orchestration, provenance/pinning, event loop guardrails)
- Runtime Ladder (WorkflowRuntime resolver/pause-resume/CDC/LangGraph path)

- `docs/tutorials/README.md`

