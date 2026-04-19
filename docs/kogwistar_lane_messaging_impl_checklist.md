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
- `llm-wiki` request/reply flow must stay compatible during migration

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
- [ ] optionally materialize `prev` / `next`
- [ ] optionally maintain inbox or conversation tail shortcuts

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

### 5.4 Runtime integration

Tasks:

- [x] keep `StepContext.publish()` as same-process orchestration only

## 6. Execution Checklist

### Phase 1 - Core ownership

- [ ] confirm core lane messaging API is the only semantic entrypoint
- [ ] move any projection semantics out of concrete stores
- [ ] keep `meta_sqlite` as abstraction slot for projections
- [ ] audit shared base / mixin layers for repeated projection behavior

### Phase 2 - Projection contract

- [ ] define exact projection fields for pending, claimable, retryable, and dead-letter states
- [ ] make projection rebuild deterministic from graph/event truth
- [ ] add repair path for queue projections without rewriting authoritative message truth
- [ ] ensure inbox and conversation ordering rules are documented and tested

### Phase 3 - App compatibility

- [ ] keep `llm-wiki` request/reply behavior stable
- [ ] pin a sample integration against the new stable contract
- [ ] keep existing public request/reply paths compatible during migration
- [ ] add migration notes for any renamed or split messaging surface

### Phase 4 - Docs alignment

- [ ] update this checklist as implementation lands
- [ ] update `kogwistar_lane_messaging_ard.md` if semantics move
- [ ] update `visibility_viewing_auditing.md` if new views or audit surfaces appear
- [ ] update `STATUS.md` once the contract becomes stable

### Phase 5 - Test the contract

- [ ] core lane messaging contract tests
- [ ] metastore projection rebuild tests
- [ ] backend parity tests for in-memory, SQLite, Postgres
- [ ] `llm-wiki` orchestration regression tests
- [ ] visibility / audit smoke tests for send, claim, retry, dead-letter

### Phase 6 - Rollout

- [ ] keep old path alive until new contract is pinned
- [ ] do not cut over backend-specific semantics before parity is proven
- [ ] promote new lane messaging contract only after app integration passes
- [ ] optionally add `StepContext.send_lane_message(...)` for durable cross-lane delivery
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

- [ ] mirror meaningful lane message state into run registry events when useful
- example event names:
  - `worker.requested`
  - `worker.claimed`
  - `worker.progress`
  - `worker.result`
  - `worker.failed`
- [ ] expose lane message progress to UI/SSE without requiring raw graph fetch each time

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
```

---

## 8. Migration plan

### Phase 1 - minimal end-to-end

Deliver the smallest usable path.

- [x] message node + semantic edges
- [x] projected pending table/index
- [x] background worker can claim and reply
- [x] no linked-list materialization yet
- [ ] run registry emits worker progress events

Success criteria:

- foreground sends durable message
- worker claims it
- worker emits reply message
- UI / run event feed can show lifecycle

### Phase 2 - conversation ordering

- [x] add conversation-local sequence projection
- [x] add inbox-local sequence projection
- [x] expose latest message queries
- [ ] optionally add derived `prev` / `next`

### Phase 3 - advanced coordination

- [ ] cancellations
- [ ] timeouts
- [ ] supervisor lane
- [ ] approval / governance lane messages
- [x] routing rules and priority
- [ ] dead-letter handling

---

## 9. Test plan

### 9.1 Unit tests

#### Message creation

- [x] create lane message writes expected node properties
- [x] message creation attaches required semantic edges
- [x] message is queryable by conversation and inbox immediately after authoritative write

#### Projection

- [x] projected row appears after message creation
- [x] inbox seq increments monotonically
- [x] conversation seq increments monotonically
- [x] latest-message query returns expected result

#### Claim / ack

- [x] one worker claims pending message successfully
- [x] second worker cannot claim an already leased message
- [x] ack marks message completed
- [x] expired lease allows reclaim

#### Retry

- [x] requeue increments retry count
- [x] delayed retry respects `available_at`
- [ ] dead-letter or terminal-failed behavior works if implemented

#### Reply linkage

- [x] worker reply creates a new message node
- [x] reply preserves correlation id
- [x] reply attaches `reply_to`

### 9.2 Integration tests

#### Foreground to worker round-trip

1. create conversation
2. send message from foreground to worker inbox
3. worker loop claims projected message
4. worker processes and emits reply
5. reply becomes visible in same conversation
6. run registry/SSE-facing events reflect progress

Current status:

- [x] steps 1-5 are pinned for the `llm-wiki` maintenance request/reply path
- [ ] step 6 is still missing

#### Recovery / rebuild

- [ ] delete or corrupt linked-list pointers
- [ ] rebuild projections
- [ ] verify ordering still reconstructs from authoritative message membership + sequence facts

#### Multi-sender concurrency

- [ ] multiple foreground senders send to same inbox
- [ ] projected sequence remains total and stable
- [ ] no lost messages

#### Worker crash / restart

- [ ] worker claims message
- [ ] worker crashes before ack
- [ ] lease expires
- [ ] another worker reclaims safely

### 9.3 End-to-end tests worth having

- [ ] chat-triggered background summarization via lane message
- [ ] workflow step requests async document parse via lane message
- [ ] governance/approval uses same messaging substrate with different lane actors

---

## 10. Non-goals for first slice

Do not require these in the first implementation:

- fully general actor system semantics
- priority scheduling across all worker classes
- distributed cross-backend exactly-once guarantees
- complex graph traversal as worker inbox scanning
- rich linked-list editing in hot path

---

## 11. Key design rules for local agents

1. Entity event is the authoritative write path.
2. Message membership is authoritative at creation time.
3. Queue mechanics belong in projections.
4. Workers consume projected pending items, not raw graph scans.
5. Linked-list pointers are derived convenience, not sole truth.
6. Reply is a new message node, not mutation of the old one into a response.
7. Use correlation IDs and reply links consistently.
8. Keep same-process runtime publish separate from durable cross-process messaging.

---

## 12. Recommended first coding slice

If only one slice is implemented now, do this:

- [x] add `send_lane_message()` using entity event-backed message node creation
- [x] add projected pending lane message table/index in meta sqlite
- [x] add claim/ack/requeue APIs
- [x] update one background worker to consume that projection or mirror into it
- [x] emit reply messages into the same conversation
- [ ] mirror worker lifecycle into run registry events

This is the smallest slice that makes the system materially more OS-like.

---

## 13. Acceptance checklist

- [x] foreground can send durable message into a conversation-linked worker inbox
- [x] authoritative write is append/event based
- [x] message is semantically complete on creation
- [x] projected pending index supports fast claim
- [x] worker can ack/requeue safely
- [x] reply is represented as a new message node
- [x] correlation / reply linkage is preserved
- [ ] run registry or SSE can show worker lifecycle
- [ ] linked-list materialization, if added, is rebuildable
- [ ] recovery tests pass when projection is rebuilt

---

## 14. Landed slice notes

The current implementation now exists in core and is wired into one `llm-wiki`
foreground/background flow.

### Core landed

- `kogwistar.messaging` provides:
  - `LaneMessagingService`
  - projected row/result models
- engine facade methods now expose:
  - `send_lane_message(...)`
  - `update_lane_message_status(...)`
  - `claim_projected_lane_messages(...)`
  - `ack_projected_lane_message(...)`
  - `requeue_projected_lane_message(...)`
  - `list_projected_lane_messages(...)`
- in-memory, SQLite, and Postgres meta stores now project lane-message queue rows
- message nodes are written with semantic edges:
  - `in_conversation`
  - `in_inbox`
  - `sent_by`
  - `sent_to`
  - optional `reply_to`
  - optional `about_run`
  - optional `about_step`

### `llm-wiki` landed

- maintenance request creation now emits a durable foreground-to-worker lane message
- maintenance worker completion/failure now emits a reply lane message back to foreground
- reply messages preserve `reply_to_message_id` and `correlation_id`
- the original request message status is updated on completion/failure

### Tests landed

- core lane-messaging creation and projection tests
- meta-store contract coverage for in-memory and SQLite projections
- `llm-wiki` orchestration test pinning request/reply round-trip semantics

### Remaining gaps and ideas to revisit

- explicit run-registry / SSE surfacing is still missing
- a durable runtime-facing `StepContext.send_lane_message(...)` API is still missing
- linked-list materialization and rebuild tests are still intentionally deferred
- only one `llm-wiki` worker flow is migrated so far
- Postgres projection support is implemented but still needs focused behavioral test coverage

---

## 15. Final guidance

Do not try to make a pure graph linked list solve both semantics and queue mechanics at once.

Use the graph to express meaning.
Use projections to express operational speed and leasing.

That balance is the most robust path for Kogwistar.
