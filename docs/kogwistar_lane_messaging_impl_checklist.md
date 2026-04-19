# Kogwistar Lane Messaging - Implementation Checklist

Status: In progress
Audience: Local coding agents and human maintainers
Companion: `kogwistar_lane_messaging_ard.md`
Review guide: `visibility_viewing_auditing.md`

---

## 0. Background

This checklist is the execution companion for the lane-messaging ARD. It exists to keep implementation on the right abstraction layer while still letting us ship incrementally.

The intended boundary is:

- `core` owns lane messaging semantics
- the metastore abstraction owns projection behavior
- concrete stores own storage primitives only
- `meta_sqlite` remains the abstraction slot even when storage is in-memory or Postgres-backed
- graph-native message nodes remain authoritative truth
- queue mechanics remain rebuildable projections
- sample integration must stay compatible during migration

If any checklist item points at the wrong layer, fix the checklist or ARD first instead of encoding the wrong boundary in code.

## 1. Goal

Implement a graph-native message system across foreground and background lanes in Kogwistar.

The system should:

- represent lane messages as first-class graph objects
- use entity events as the authoritative write path
- allow foreground and background workers to communicate through the same conversation namespace
- keep queue mechanics in projections rather than in hot-path graph traversal
- preserve rebuildability if list pointers or serving indexes are lost

---

## 2. Execution Contract

### Must Keep True

- message truth lives in graph-native entities and authoritative events
- projection truth lives in the metastore abstraction layer
- backend choice must not change lane semantics
- visibility, audit, and repair surfaces must describe the same truth
- application integrations should depend on one stable contract

### Must Not Happen

- backend-specific semantic forks
- direct dependence on concrete store classes for lane semantics
- silent rebuild gaps that cannot be audited
- queue mechanics that are only recoverable by raw graph scan
- docs that misstate the ownership boundary

---

## 2. Final design summary

### 2.1 Authoritative truth

Authoritative truth is:

- message node creation via entity event
- semantic membership edges such as:
  - `in_conversation`
  - `in_inbox`
  - `sent_by`
  - `sent_to`
  - `reply_to`
  - `about_run`
  - `about_step`

### 2.2 Derived / projected state

Derived state is:

- inbox sequence number
- conversation-local sequence number
- claim/lease state for worker consumption
- latest message indexes
- linked list edges such as `prev` / `next`
- inbox head/tail shortcuts

### 2.3 Operational rule

Workers must not discover pending work by raw graph scan.

Workers should consume from a fast serving projection built from authoritative graph/event truth.

---

## 3. Proposed data model

### 3.1 Graph objects

#### Message node

Suggested minimum properties:

```json
{
  "kind": "lane_message",
  "message_id": "msg:...",
  "conversation_id": "conv:...",
  "inbox_id": "inbox:worker:index",
  "sender_id": "lane:foreground",
  "recipient_id": "lane:worker:index",
  "msg_type": "request.parse_document",
  "status": "pending",
  "correlation_id": "corr:...",
  "reply_to": null,
  "run_id": "run:...",
  "step_id": "step:...",
  "payload": {},
  "created_at": "...",
  "completed_at": null,
  "error": null
}
```

#### Inbox node

```json
{
  "kind": "lane_inbox",
  "inbox_id": "inbox:worker:index",
  "worker_name": "index",
  "lane_role": "background_worker"
}
```

#### Optional lane actor nodes

Examples:

- `lane:foreground`
- `lane:worker:index`
- `lane:worker:summarizer`
- `lane:supervisor`

---

### 3.2 Edge types

Required:

- `message --in_conversation--> conversation`
- `message --in_inbox--> inbox`
- `message --sent_by--> actor`
- `message --sent_to--> actor`

Useful:

- `message --reply_to--> message`
- `message --about_run--> run`
- `message --about_step--> step`
- `message --produced_result--> artifact`

Derived only:

- `message --prev--> message`
- `message --next--> message`
- `inbox --tail--> message`
- `conversation --tail_message--> message`

---

## 4. Event model

### 4.1 New event intents

These do not all need separate top-level event types if the entity event envelope already supports node/edge mutations, but the semantic intents should exist.

#### Mandatory semantic intents

- `MESSAGE_CREATED`
- `MESSAGE_STATUS_UPDATED`
- `MESSAGE_REPLY_LINKED`

#### Optional semantic intents

- `MESSAGE_CLAIMED`
- `MESSAGE_COMPLETED`
- `MESSAGE_FAILED`
- `MESSAGE_CANCELLED`

#### Projection-only intents

These should normally remain implicit / rebuildable rather than authoritative:

- `MESSAGE_PROJECTED_TO_INBOX_SEQ`
- `MESSAGE_PROJECTED_TO_CONVERSATION_SEQ`
- `MESSAGE_LINKED_PREV_NEXT`

### 4.2 Write rule

When sending a message, the caller should emit authoritative graph/entity mutations only.

The send path should not directly mutate queue-serving state unless that is part of the same durable projection pipeline.

#### Send flow

1. create message node
2. attach semantic edges
3. persist entity event
4. projector updates inbox/conversation serving views
5. worker or foreground consumers read projected pending items

---

## 5. Module-level implementation plan

### 5.1 Engine core / metastore abstraction

Likely files:

- `kogwistar/engine_core/engine.py`
- `kogwistar/engine_core/meta_*` shared helpers / mixins
- `kogwistar/engine_core/engine_sqlite.py`
- `kogwistar/engine_core/engine_postgres_meta.py`
- `kogwistar/messaging/service.py`

Tasks:

- [x] add a lane messaging facade on the engine
  - [x] `send_lane_message(...)`
  - [x] `update_lane_message_status(...)`
  - [x] `claim_projected_lane_messages(...)`
  - [x] `ack_projected_lane_message(...)`
  - [x] `requeue_projected_lane_message(...)`
  - [x] `list_projected_lane_messages(...)`
- [x] ensure engine methods route through entity event writes rather than direct ad hoc graph edits
- [x] add metastore-level projection helpers for pending message queries
- [x] concrete metastore classes share lane-message projection mixin instead of owning projection semantics individually

Notes:

- Treat `engine.meta_sqlite` as the metastore interface slot even when the concrete store is in-memory or Postgres-backed.
- Do not make the engine API depend on raw linked-list traversal for consumption.

### 5.2 Entity event path

Tasks:

- [x] add message-creation mutation helper
- [x] define canonical edge/property set for lane messages
- [x] keep message membership authoritative at creation time
- [x] avoid creating a half-valid message that only becomes real after later linked-list mutation
- [x] keep `MESSAGE_*` semantic intent implicit for now inside generic entity events plus node replacement; do not split another top-level event stream unless a future slice needs it
- [x] keep `llm-wiki` request/reply path compatible while lane messaging contract evolves

### 5.3 Projection layer

Tasks:

- [x] add a projected table or equivalent index for fast consumption
- [x] assign monotonic inbox-local `seq`
- [x] assign monotonic conversation-local `conversation_seq`
- [x] maintain pending/claimable view
- [x] keep projection rebuildable from authoritative truth
- [x] optionally materialize `prev` / `next`
- [x] optionally maintain inbox or conversation tail shortcuts

Suggested projection fields:

```text
projected_lane_messages(
  message_id,
  inbox_id,
  conversation_id,
  recipient_id,
  sender_id,
  msg_type,
  seq,
  conversation_seq,
  status,
  claimed_by,
  lease_until,
  retry_count,
  created_at,
  available_at,
  run_id,
  step_id,
  correlation_id
)
```

Notes:

- This is the place for quick search of newest messages.
- SQLite, in-memory, and Postgres support are now implemented through the metastore layer.
- in-memory, SQLite, and Postgres should keep the same lane-message projection contract through the shared mixin/base layer.
- The current contract is intentionally minimal: `send / claim / ack / requeue / list / rebuild`.
- `prev` / `next` and tail shortcuts remain optional future projection optimizations, not required for the stable contract.

### 5.4 Runtime integration

Tasks:

- [x] keep `StepContext.publish()` as same-process orchestration only

## 6. Execution Checklist

### Phase 1 - Core ownership

- [x] confirm core lane messaging API is the only semantic entrypoint
- [x] move any projection semantics out of concrete stores
- [x] keep `meta_sqlite` as abstraction slot for projections
- [x] audit shared base / mixin layers for repeated projection behavior

### Phase 2 - Projection contract

- [x] define exact projection fields for pending, claimable, retryable, and dead-letter states
- [x] make projection rebuild deterministic from graph/event truth
- [x] add repair path for queue projections without rewriting authoritative message truth
- [x] ensure inbox and conversation ordering rules are documented and tested

### Phase 3 - App compatibility

- [x] pin a sample contract-compliant object against the new stable contract
- [x] pin a sample integration against the new stable contract
- [ ] add migration notes for any renamed or split messaging surface

### Phase 4 - Docs alignment

- [x] update this checklist as implementation lands
- [ ] update `kogwistar_lane_messaging_ard.md` if semantics move
- [x] update `visibility_viewing_auditing.md` if new views or audit surfaces appear
- [ ] update `STATUS.md` once the contract becomes stable

### Phase 5 - Test the contract

- [x] core lane messaging contract tests
- [x] sample contract fixture test
- [x] metastore projection rebuild tests
- [x] backend parity tests for in-memory, SQLite, Postgres
- [x] visibility smoke test for send, claim, retry
- [x] visibility / audit smoke tests for send, claim, retry, dead-letter

### Phase 6 - Rollout

- [x] keep old path alive until new contract is pinned
- [x] do not cut over backend-specific semantics before parity is proven
- [x] promote new lane messaging contract only after app integration passes
- [x] add `StepContext.send_lane_message(...)` for durable cross-lane delivery
- [x] add `StepContext.emit_lane_message_event(...)` for lifecycle mirroring hooks
- [x] keep current in-memory queue semantics for same-run, same-process orchestration
- [x] do not silently repurpose the existing `queue.Queue` as cross-process IPC

Suggested split:

- `publish()` => same-process orchestration message
- `send_lane_message()` => durable graph-native inter-lane message

### 5.5 Background worker integration

Tasks:

- [x] add polling/claim loop against `projected_lane_messages`
- [x] implement lease/ack/retry semantics using projection/meta layer
- [x] let a worker emit replies as new authoritative message nodes
- [x] attach `reply_to` and shared `correlation_id`
- [ ] migrate additional workers beyond the first maintenance request/reply path

Notes:

- Do not overload `index_jobs` itself unless deliberately merging concepts.
- Current `llm-wiki` state: one maintenance request/reply flow is mirrored into lane messaging, while `index_jobs` still remains the authoritative scheduler for that worker.

### 5.6 Conversation / chat / run registry integration

Tasks:

- [x] mirror meaningful lane message state into run registry events when useful
- example event names:
  - `worker.requested`
  - `worker.claimed`
  - `worker.progress`
  - `worker.result`
  - `worker.failed`
- [x] expose lane message progress to UI/SSE without requiring raw graph fetch each time

Notes:

- The graph remains authoritative; registry/SSE is an observability surface.

---

## 6. Linked-list materialization guidance

### 6.1 Recommendation

If inbox or conversation linked lists are kept, treat them as derived serving structure.

Do not make correctness depend solely on `next` / `prev` edges.

### 6.2 Preferred order facts

Reliable order should come from:

- append event order
- projected monotonic sequence numbers

Linked list pointers are useful for:

- graph inspection
- locality/navigation
- tail/head shortcuts
- debugging / visualization

### 6.3 Safer pointer strategy

Prefer at most one authoritative direction if needed.

Examples:

- `message --prev--> previous_message`
- `conversation --tail_message--> latest_message`

Then derive `next` if desired.

This reduces pointer maintenance burden.

---

## 7. Suggested public API sketch

```python
class LaneMessagingService:
    def send_message(
        self,
        *,
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        payload: dict,
        run_id: str | None = None,
        step_id: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
    ) -> str: ...

    def update_message_status(
        self,
        *,
        message_id: str,
        status: str,
        error: dict | None = None,
    ) -> None: ...

    def claim_pending(
        self,
        *,
        inbox_id: str,
        claimed_by: str,
        limit: int,
        lease_seconds: int,
    ) -> list[ProjectedLaneMessage]: ...

    def ack(
        self,
        *,
        message_id: str,
        claimed_by: str,
    ) -> None: ...

    def requeue(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error: dict | None = None,
        delay_seconds: int = 0,
    ) -> None: ...

    def dead_letter(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error: dict | None = None,
    ) -> None: ...

    def list_projected(
        self,
        *,
        inbox_id: str | None = None,
        status: str | None = None,
    ) -> list[ProjectedLaneMessage]: ...
```

Stable contract:

- `send_message(...)`
- `claim_pending(...)`
- `ack(...)`
- `requeue(...)`
- `dead_letter(...)`
- `list_projected(...)`

Rules:

- message truth lives in graph/entity events
- projection truth lives in metastore abstraction
- concrete stores only provide storage primitives
- `meta_sqlite` remains slot name, not semantic owner
- same contract must work across in-memory, SQLite, Postgres

---

## 8. Checklist

- [x] core lane messaging contract tests
- [x] sample contract fixture test
- [x] metastore projection rebuild tests
- [x] backend parity tests for in-memory, SQLite, Postgres
- [x] visibility smoke test for send, claim, retry, dead-letter
- [x] `StepContext.send_lane_message(...)`
- [x] `StepContext.emit_lane_message_event(...)`
- [x] run registry / SSE can show worker lifecycle
- [x] recovery tests pass when projection is rebuilt
- [x] linked-list materialization, if added, is rebuildable
- [ ] promotion after app integration passes
