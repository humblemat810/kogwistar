# FastHTML HTMX Chat UI API Guide

This document is the contract for a FastHTML + HTMX conversation UI that talks to this repository's REST API.

It is written to help a local agent implement the UI without guessing endpoint shapes, auth behavior, or run/event semantics.

## What This API Is For

The server exposes a conversation and workflow runtime backed by `ChatRunService`.

Use it for:

- Conversation list and transcript rendering
- Submitting a user turn and tracking the resulting run
- Streaming run progress into a non-chat event panel
- Loading context snapshots and workflow debug data

Do not invent alternate endpoints if an equivalent one already exists here.

## Authentication

The repo currently supports two auth paths:

- `POST /auth/dev-token` on the main server for local/dev JWT minting
- `GET /api/auth/login` and `/api/auth/callback` for the OIDC flow

For a FastHTML chat shell, the simplest local path is `POST /auth/dev-token`.

Important caveat:

- The current `/auth/dev-token` route does not validate a password.
- It mints a JWT from the JSON body you send.
- The request body must be valid JSON.
- `ns` may be a single namespace, a comma-separated string, or a JSON list.
- If the UI presents username/password fields, the password is only a UI affordance unless you add server-side verification.

Role and namespace semantics:

- `role=ro` is the minimum role for read endpoints.
- `role=rw` is the minimum role for write endpoints.
- A token with `role=rw` is acceptable for read endpoints because it exceeds the minimum.
- The token must also carry the namespace required by the endpoint.
- Dev tokens may contain multiple namespaces in the `ns` claim.

### Dev token request

```http
POST /auth/dev-token
Content-Type: application/json

{
  "username": "alice",
  "role": "rw",
  "ns": "docs,conversation,workflow,wisdom"
}
```

Equivalent JSON-list form:

```http
POST /auth/dev-token
Content-Type: application/json

{
  "username": "alice",
  "role": "rw",
  "ns": ["docs", "conversation", "workflow", "wisdom"]
}
```

### Dev token response

```json
{
  "token": "eyJhbGciOi..."
}
```

### JWT expectations

Use the returned token as a bearer token on later requests:

```http
Authorization: Bearer <token>
```

The token must carry the namespace and role needed by the endpoint:

- Conversation read endpoints require at least `role=ro`
- Conversation write endpoints require at least `role=rw`
- The conversation namespace must be allowed

Example token claims:

```json
{
  "sub": "alice",
  "role": "rw",
  "ns": ["docs", "conversation", "workflow", "wisdom"]
}
```

If the request body is empty or form-encoded instead of JSON, `/auth/dev-token` will fail before validation.

### Minimal dev-login client example

```js
async function devLogin() {
  const resp = await fetch("/auth/dev-token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: "alice",
      role: "rw",
      ns: "docs,conversation,workflow,wisdom",
    }),
  });

  if (!resp.ok) {
    throw new Error(`dev login failed: ${resp.status}`);
  }

  const data = await resp.json();
  sessionStorage.setItem("jwt", data.token);
  return data.token;
}
```

## Recommended UI Flow

1. User signs in and the UI stores the JWT in the session.
2. Sidebar loads conversations for the current user.
3. Clicking a conversation loads the transcript.
4. Sending a message posts a turn and returns a `run_id`.
5. The center panel shows the conversation transcript.
6. The right panel subscribes to run events and displays non-chat state.
7. The UI may poll run status as a lightweight badge source, not as the inspection path.

## Core Endpoints

### List conversations for the current user

`GET /api/conversations`

Authorization:

- `role=ro`
- conversation namespace required

Behavior:

- The server resolves the effective user from the token/session.
- The route does not accept a user id in the request body.

Response shape:

```json
{
  "conversations": [
    {
      "id": "conv_123",
      "start_node_id": "turn|conv_123|user-1",
      "status": "active",
      "turn_count": 14
    }
  ]
}
```

Notes:

- Results are sorted by `turn_count` descending.
- This endpoint is intended for the left sidebar.

Example client call:

```js
async function loadConversationList(token) {
  const resp = await fetch("/api/conversations", {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!resp.ok) throw new Error(`list failed: ${resp.status}`);
  return await resp.json();
}
```

### Create a conversation

`POST /api/conversations`

Authorization:

- `role=rw`
- conversation namespace required

Request body:

```json
{
  "user_id": "alice",
  "conversation_id": null,
  "start_node_id": null
}
```

Response shape:

```json
{
  "conversation_id": "conv_456",
  "user_id": "alice",
  "status": "active",
  "start_node_id": "turn|conv_456|user-1",
  "tail_node_id": "turn|conv_456|user-1",
  "turn_count": 0
}
```

Notes:

- If the token-derived user id exists, the server uses it.
- Otherwise the route falls back to `user_id` in the body.
- The response also includes `start_node_id` from the create call.

Example client call:

```js
async function createConversation(token, userId) {
  const resp = await fetch("/api/conversations", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      user_id: userId,
      conversation_id: null,
      start_node_id: null,
    }),
  });
  if (!resp.ok) throw new Error(`create failed: ${resp.status}`);
  return await resp.json();
}
```

### Load a conversation summary

`GET /api/conversations/{conversation_id}`

Authorization:

- `role=ro`
- conversation namespace required

Response shape:

```json
{
  "conversation_id": "conv_456",
  "user_id": "alice",
  "status": "active",
  "start_node_id": "turn|conv_456|user-1",
  "tail_node_id": "turn|conv_456|assistant-2",
  "turn_count": 8
}
```

Use this to refresh conversation metadata in the sidebar or header.

### Load a transcript

`GET /api/conversations/{conversation_id}/turns`

Authorization:

- `role=ro`
- conversation namespace required

Response shape:

```json
{
  "conversation_id": "conv_456",
  "turns": [
    {
      "node_id": "turn|conv_456|user-1",
      "turn_index": 1,
      "role": "user",
      "content": "Hello",
      "entity_type": "conversation_turn"
    },
    {
      "node_id": "turn|conv_456|assistant-1",
      "turn_index": 2,
      "role": "assistant",
      "content": "Hi, how can I help?",
      "entity_type": "assistant_turn"
    }
  ]
}
```

Use this to render the center chat column.

Example transcript fetch:

```js
async function loadTranscript(token, conversationId) {
  const resp = await fetch(`/api/conversations/${conversationId}/turns`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!resp.ok) throw new Error(`transcript failed: ${resp.status}`);
  return await resp.json();
}
```

### Submit a user message

`POST /api/conversations/{conversation_id}/turns:answer`

Authorization:

- `role=rw`
- conversation namespace required

Request body:

```json
{
  "user_id": "alice",
  "text": "Summarize this project",
  "workflow_id": "agentic_answering.v2"
}
```

Response:

- HTTP `202 Accepted`

Response shape:

```json
{
  "run_id": "run_789",
  "conversation_id": "conv_456",
  "workflow_id": "agentic_answering.v2",
  "status": "queued",
  "user_turn_node_id": "turn|conv_456|user-2"
}
```

Notes:

- `text` must be non-empty.
- If `user_id` is omitted, the server uses the conversation owner from the token-backed identity.
- This endpoint creates the user turn first, then starts the run asynchronously.

Example send-message flow:

```js
async function submitAnswer(token, conversationId, text, workflowId = "agentic_answering.v2") {
  const resp = await fetch(`/api/conversations/${conversationId}/turns:answer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      text,
      workflow_id: workflowId,
    }),
  });
  if (resp.status !== 202) {
    throw new Error(`answer submit failed: ${resp.status}`);
  }
  return await resp.json();
}
```

## Run APIs

These are the APIs the right panel should use for live execution state.

### Load run metadata

`GET /api/runs/{run_id}`

Authorization:

- `role=ro`
- conversation namespace required

Response shape typically includes:

```json
{
  "run_id": "run_789",
  "conversation_id": "conv_456",
  "workflow_id": "agentic_answering.v2",
  "status": "running",
  "terminal": false,
  "cancel_requested": false,
  "started_at_ms": 1710000000000,
  "finished_at_ms": null,
  "assistant_turn_node_id": null
}
```

Status values:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

### Stream run events with SSE

`GET /api/runs/{run_id}/events?after_seq=0`

Authorization:

- `role=ro`
- conversation namespace required

This returns `text/event-stream`.

Each event frame uses:

```text
id: <seq>
event: <event_type>
data: {"run_id":"...","event_type":"...","created_at_ms":...}
```

The JSON payload includes the server event fields plus the event data.

Typical event types for a chat run:

- `run.created`
- `run.started`
- `run.stage`
- `reasoning.summary`
- `output.delta`
- `output.completed`
- `run.completed`
- `run.failed`
- `run.cancelling`
- `run.cancelled`

Recommended UI behavior:

- `run.stage` updates the right panel status line
- `reasoning.summary` appends a thought/status entry
- `output.delta` streams assistant text into the center panel
- `output.completed` finalizes the assistant bubble
- `run.completed`, `run.failed`, and `run.cancelled` close the live state

Resume behavior:

- Persist the last event `id`
- Reconnect with `after_seq=<last_id>`

Example SSE hookup:

```js
function watchRunEvents(runId, afterSeq = 0) {
  const es = new EventSource(`/api/runs/${runId}/events?after_seq=${afterSeq}`);

  es.addEventListener("run.stage", (evt) => {
    const data = JSON.parse(evt.data);
    console.log("stage", data.stage);
  });

  es.addEventListener("reasoning.summary", (evt) => {
    const data = JSON.parse(evt.data);
    console.log("summary", data.summary);
  });

  es.addEventListener("output.delta", (evt) => {
    const data = JSON.parse(evt.data);
    console.log("delta", data.delta);
  });

  es.addEventListener("run.completed", () => es.close());
  es.addEventListener("run.failed", () => es.close());
  es.addEventListener("run.cancelled", () => es.close());

  return es;
}
```

### Poll run events

`GET /api/runs/{run_id}/events/poll?after_seq=0&limit=500`

Authorization:

- `role=ro`
- conversation namespace required

This is the fallback if SSE is unavailable or you want a manual polling mode.

### Inspect run steps and replay state

Use these when you need debugging detail that should not be coupled to the status call:

- `GET /api/runs/{run_id}/steps`
- `GET /api/runs/{run_id}/checkpoints`
- `GET /api/runs/{run_id}/checkpoints/{step_seq}`
- `GET /api/runs/{run_id}/replay?target_step_seq=...`

### Cancel a run

`POST /api/runs/{run_id}/cancel`

Authorization:

- `role=rw`
- conversation namespace required

Response:

- HTTP `202 Accepted`

Use this when the user stops generation from the UI.

## Snapshot and Workflow Debug APIs

These are useful for the right panel or debug drawer.

### Latest context snapshot

`GET /api/conversations/{conversation_id}/snapshots/latest?run_id=<run_id>&stage=<stage>`

Authorization:

- `role=ro`
- workflow namespace required

Response shape:

```json
{
  "snapshot_node_id": "snap_123",
  "conversation_id": "conv_456",
  "metadata": {
    "run_id": "run_789",
    "stage": "draft_answer"
  },
  "properties": {
    "messages": [...],
    "items": [...]
  }
}
```

### Workflow runtime endpoints

If the UI needs workflow debug controls, the runtime router exposes:

- `POST /api/workflow/runs`
- `GET /api/workflow/runs/{run_id}` for status only
- `GET /api/workflow/runs/{run_id}/events`
- `POST /api/workflow/runs/{run_id}/cancel`
- `GET /api/workflow/runs/{run_id}/steps`
- `GET /api/workflow/runs/{run_id}/checkpoints`
- `GET /api/workflow/runs/{run_id}/checkpoints/{step_seq}`
- `GET /api/workflow/runs/{run_id}/replay?target_step_seq=...`

## HTMX Integration Guidance

HTMX is a good fit for the shell, but the backend API is JSON and SSE-based.

Recommended pattern:

- Use HTMX for page fragments, navigation, and local form submission
- Use `fetch()` or a small JS helper for JSON API calls
- Use `EventSource` for SSE

### Sidebar refresh

On load or after login:

1. Call `GET /api/conversations`
2. Render the sidebar conversation list from the returned JSON
3. Select the first conversation or restore the last active one

### Sending a message

1. Submit the message to `POST /api/conversations/{conversation_id}/turns:answer`
2. Read the `run_id` from the `202` response
3. Clear the composer only after the request succeeds
4. Open or reuse an `EventSource` for `/api/runs/{run_id}/events`
5. Use `GET /api/runs/{run_id}` only for compact lifecycle state

### Right panel

Keep the right panel focused on live run telemetry:

- current stage
- reasoning summaries
- cancellation state
- run completion/error status
Debug data belongs in the inspection endpoints, not the status call or the main transcript.

## Suggested Client Wiring

### Conversation list item

```html
<button
  class="conversation-item"
  data-conversation-id="conv_456"
  hx-get="/ui/conversations/conv_456"
  hx-target="#chat-panel"
  hx-swap="innerHTML"
>
  Project discussion
</button>
```

The UI route can server-render the chat panel after fetching:

- `GET /api/conversations/{conversation_id}`
- `GET /api/conversations/{conversation_id}/turns`

This works well when the UI wants to render the current transcript server-side before wiring SSE.

### New conversation button

```html
<button
  hx-post="/ui/conversations/new"
  hx-target="#chat-panel"
  hx-swap="innerHTML"
>
  New chat
</button>
```

The FastHTML handler can proxy to `POST /api/conversations` and then render the new conversation.

### Message composer

```html
<form
  hx-post="/ui/messages/send"
  hx-target="#chat-panel"
  hx-swap="none"
>
  <textarea name="text"></textarea>
  <button type="submit">Send</button>
</form>
```

The FastHTML route should:

1. Call `POST /api/conversations/{conversation_id}/turns:answer`
2. Return a fragment that keeps the transcript visible
3. Let the SSE listener stream the assistant response into the UI

If the UI wants optimistic updates, it can render the user bubble locally and replace it only if the request fails.

## Example Event Handling

```js
const es = new EventSource(`/api/runs/${runId}/events?after_seq=${lastSeq}`);

es.addEventListener("run.stage", (evt) => {
  const data = JSON.parse(evt.data);
  updateRightPanel(`Stage: ${data.stage}`);
  lastSeq = Number(evt.lastEventId || lastSeq);
});

es.addEventListener("reasoning.summary", (evt) => {
  const data = JSON.parse(evt.data);
  appendEventLog(data.summary);
});

es.addEventListener("output.delta", (evt) => {
  const data = JSON.parse(evt.data);
  appendAssistantDelta(data.delta);
});

es.addEventListener("output.completed", (evt) => {
  const data = JSON.parse(evt.data);
  finalizeAssistantBubble(data.assistant_text);
});

es.addEventListener("run.completed", () => {
  es.close();
});

es.addEventListener("run.failed", () => {
  es.close();
});

es.addEventListener("run.cancelled", () => {
  es.close();
});
```

## Error Handling

The server maps common exceptions to HTTP status codes:

- `KeyError` -> `404`
- `ValueError` -> `400`
- `HTTPException` -> pass through
- unexpected errors -> `500`

Practical UI rules:

- `401` means missing or invalid token
- `403` means token lacks role or namespace access
- `404` usually means unknown conversation or run id
- `409` appears in workflow-projection rebuild cases

## Minimal End-to-End Sequence

1. `POST /auth/dev-token`
2. Store the JWT in session
3. `GET /api/conversations`
4. `POST /api/conversations` if starting fresh
5. `GET /api/conversations/{conversation_id}/turns`
6. `POST /api/conversations/{conversation_id}/turns:answer`
7. `GET /api/runs/{run_id}/events`
8. `GET /api/runs/{run_id}` until terminal for lifecycle state only

The common client pattern is:

1. Login and store the JWT.
2. Load the conversation list.
3. Render the selected conversation transcript.
4. Submit a message.
5. Stream the run into the right-hand panel.
6. Keep the transcript and the run telemetry separate.

## Implementation Notes For the Local Agent

- Prefer the existing API contract over creating new server routes.
- Treat `conversation_id` as the primary conversation key in the UI.
- Treat `run_id` as the primary live-execution key.
- Use SSE for live status and JSON fetches for initial state.
- Keep chat content in the center panel and all run telemetry in the right panel.
- If you need a more elaborate login form, adapt the FastHTML app, not the GraphRAG server, unless you intentionally want to change auth behavior.
