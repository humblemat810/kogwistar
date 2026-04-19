# ARD: Graph-Native Lane Messaging for Foreground and Background Workers

Status: Proposed  
Target repo: `kogwistar`  
Audience: local agents and human implementers  

---

## 1. Summary

This ARD proposes a graph-native message system for communication across lanes such as:

- foreground runtime
- background workers
- agents
- supervisors
- approval/governance components

The key design choice is:

**messages are first-class graph objects in the conversation namespace, while queue mechanics remain a derived operational projection.**

This allows Kogwistar to move closer to an AI operating system shape:

- durable inter-lane communication
- replayable coordination
- graph-native observability
- worker-safe operational dequeue
- unified provenance across conversation, workflow, and background execution

---

## 2. Problem

Kogwistar already has:

- in-process runtime messaging for a workflow run
- durable background job processing patterns
- graph-native conversation and workflow structures
- event-sourced semantics and rebuildable projections

What is missing is a **first-class inter-lane message system** that can represent foreground/background worker communication inside the substrate itself.

Today, background activity risks appearing as hidden infrastructure behavior rather than graph-native system behavior.

That weakens the claim that Kogwistar is becoming an AI operating system instead of only a graph store plus runtime.

---

## 3. Goals

1. Represent worker/agent/system messages as first-class graph objects.
2. Allow foreground and background workers to share the same conversation namespace.
3. Preserve append-only, event-first semantics.
4. Support reliable dequeue/claim/retry for background workers.
5. Keep queue mechanics rebuildable from authoritative graph/event truth.
6. Make messages queryable by conversation, run, worker, correlation, and reply chain.
7. Enable SSE/UI/debug projections from the same authoritative source.

---

## 4. Non-goals

1. This is not a replacement for every existing in-process queue.
2. This is not a requirement that all message order be recovered only by raw graph traversal.
3. This is not a mandate that linked-list pointers are the only truth.
4. This does not attempt to solve distributed consensus across arbitrary remote nodes.
5. This does not redesign the whole workflow engine in one step.

---

## 5. Architectural decision

### 5.1 Core decision

Use **entity events** to create semantically complete message nodes in the graph.

Then derive:

- conversation-local order
- inbox-local order
- head/tail shortcuts
- optional linked-list edges
- claimable pending queue projection

### 5.2 Critical boundary

**Authoritative truth:**

- message exists
- message belongs to conversation/inbox
- sender
- recipient
- run/step/correlation references
- lifecycle state transitions

**Derived projection state:**

- queue sequence numbers
- lease/claim state
- fast pending lookup
- linked-list navigation shortcuts
- unread/latest shortcuts
- worker-local inbox scans

### 5.3 Why this choice

This preserves the Kogwistar pattern:

- append facts first
- project serving structures later
- rebuild projections when needed

It also avoids making fragile pointer structures the sole source of correctness.

---

## 6. Conceptual model

### 6.1 Main idea

Foreground and background workers share the **same conversation graph namespace**, but not a single mutable conversation node used as a mailbox blob.

Instead:

- each message is its own node
- each worker can have an inbox node
- each message belongs to both a conversation and optionally an inbox
- ordering and claimability are projected

### 6.2 Recommended graph entities

#### Conversation node
Represents the shared conversational/system namespace.

#### Worker inbox node
Represents a logical mailbox for a worker or lane.

Examples:

- `inbox:foreground`
- `inbox:worker:index`
- `inbox:worker:projection`
- `inbox:governance`

#### Message node
Represents one unit of inter-lane communication.

Examples:

- request from foreground to worker
- progress update from worker to foreground
- result reply
- cancellation
- escalation
- approval request

#### Participant nodes
Optional but recommended. These can represent:

- foreground lane
- worker lane
- human actor
- tool actor
- supervisor actor

---

## 7. Data model

### 7.1 Message node properties

Recommended message node payload:

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
  "priority": 0,
  "correlation_id": "corr:...",
  "reply_to_message_id": null,
  "run_id": "run:...",
  "step_id": "step:...",
  "payload": {},
  "error": null,
  "created_at": "...",
  "updated_at": "...",
  "completed_at": null
}
```

### 7.2 Recommended message lifecycle states

Authoritative semantic states:

- `pending`
- `claimed`
- `completed`
- `failed`
- `cancelled`

Projection-only worker mechanics may additionally track:

- `lease_until`
- `claimed_by`
- `retry_count`
- `available_at`

### 7.3 Graph edges

Recommended edges:

- `message --in_conversation--> conversation`
- `message --in_inbox--> inbox`
- `message --sent_by--> participant`
- `message --sent_to--> participant`
- `message --reply_to--> message`
- `message --about_run--> run`
- `message --about_step--> step`

Optional ordering edges:

- `message --prev_in_inbox--> previous_message`
- `message --next_in_inbox--> next_message`
- `message --prev_in_conversation--> previous_message`
- `message --next_in_conversation--> next_message`

These ordering edges are **not required to make the message real**.

---

## 8. Event model

### 8.1 Authoritative events

Use entity events for the authoritative state transitions.

Recommended event types:

- `MESSAGE_CREATED`
- `MESSAGE_CLAIMED`
- `MESSAGE_COMPLETED`
- `MESSAGE_FAILED`
- `MESSAGE_CANCELLED`
- `MESSAGE_REPLIED`

### 8.2 Important rule

A message must be semantically complete at creation time.

That means `MESSAGE_CREATED` must already establish:

- which conversation it belongs to
- which inbox it belongs to, if any
- sender
- recipient
- type
- payload
- initial status

Do **not** rely on a later linked-list mutation to make the message logically present.

### 8.3 Example event flow

Foreground sends request:

1. append `MESSAGE_CREATED`
2. projector derives queue row and order
3. worker claims message
4. append `MESSAGE_CLAIMED`
5. worker finishes
6. append `MESSAGE_COMPLETED`
7. worker emits reply via another `MESSAGE_CREATED`
8. projector updates pending and latest views

---

## 9. Linked-list inbox option

### 9.1 Proposal

Each worker may have an inbox represented as a graph-linked sequence of messages.

Potential structure:

- inbox node stores `head_message_id` and/or `tail_message_id`
- messages may carry `prev_in_inbox` and/or `next_in_inbox` edges

### 9.2 Evaluation

This is a good **semantic structure** for inbox navigation and graph-native modeling.

However, it should **not** be the only operational mechanism for dequeue.

### 9.3 Reason

Tail append to a linked list is concurrency-sensitive:

- two senders may race on the same tail
- pointer updates can fail halfway
- traversal can break if one pointer is wrong

Therefore the linked list should be treated as one of:

- a derived serving structure
- an optional convenience structure
- a rebuildable ordering view

not the sole source of correctness.

### 9.4 Preferred usage

- use membership + append event as truth
- derive monotonic `seq`
- optionally derive `prev`/`next`
- use `seq` and pending indexes for worker consumption

---

## 10. Projection model

### 10.1 Required projection

Create an operational projection for worker-safe dequeue.

Suggested relational projection:

```text
projected_inbox_messages(
  inbox_id,
  message_id,
  conversation_id,
  recipient_id,
  seq,
  status,
  priority,
  claimed_by,
  lease_until,
  retry_count,
  available_at,
  created_at,
  updated_at
)
```

### 10.2 Projection responsibilities

The projection should support:

- newest/oldest message lookup
- pending message scan by inbox or recipient
- claim with lease
- retry/requeue
- efficient latest-message serving
- conversation-local order lookup

### 10.3 Generic projection abstraction

`meta_sqlite` can use the generic projection abstraction to maintain:

- monotonic `seq`
- current claimability state
- latest per inbox
- latest per conversation

This is preferable to raw graph traversal on every worker tick.

---

## 11. Claim / lease semantics

### 11.1 Required behavior

Workers need safe consumption semantics.

Projection layer should support:

- claim pending messages for a recipient inbox
- set lease expiration
- mark completion/failure
- requeue timed-out work
- retry count increments

### 11.2 Why this belongs in projection

Claim/lease/retry is an operational concern.

It is useful to expose high-level lifecycle in the graph, but hot-path leasing should be stored in serving state for performance and simplicity.

### 11.3 Recommended split

**Graph/event truth:**

- message created
- message claimed
- message completed
- message failed

**Projection serving state:**

- current lease expiration
- queue ordering
- claim filters
- retry scheduling

---

## 12. Why this matters for AI operating system direction

A first-class lane message system is one of the system features that makes Kogwistar feel OS-like.

It provides the equivalent of durable IPC across:

- foreground runtime
- background worker pool
- agents
- tools
- supervisors
- governance/approval components

Without this, Kogwistar can still be a strong substrate. But with it, the platform becomes closer to an AI operating system because processes/lanes communicate through a native system model rather than ad hoc hidden mechanics.

This is not the whole AI OS story, but it is an important subsystem within it.

---

## 13. Recommended implementation plan

### Phase 1: graph schema and event types

Implement:

- message node type
- inbox node type
- message lifecycle event types
- message graph relations (`in_conversation`, `in_inbox`, `sent_by`, `sent_to`, `reply_to`)

Deliverable:

- foreground can emit a message node through entity events
- worker replies can also be emitted as message nodes

### Phase 2: projection support

Implement projection tables/views for:

- pending per inbox
- sequence ordering per inbox and conversation
- latest message per inbox/conversation
- claim/lease/retry mechanics

Deliverable:

- worker can efficiently poll/claim without graph traversal

### Phase 3: worker integration

Integrate worker loops so they:

- consume from projected inbox queue
- append lifecycle entity events
- emit reply message nodes

Deliverable:

- foreground -> worker -> foreground round trip through graph-native messages

### Phase 4: serving and UI

Project message activity to:

- SSE run stream
- conversation view
- worker diagnostics view
- CDC/debug tooling

Deliverable:

- users can see system messages and worker progress in one unified view

### Phase 5: linked-list ordering shortcuts

Optionally add:

- `prev_in_inbox`
- `next_in_inbox`
- `prev_in_conversation`
- `next_in_conversation`
- head/tail shortcuts

Deliverable:

- richer graph navigation without changing the authoritative model

---

## 14. Suggested implementation constraints

1. Do not make raw linked-list pointer correctness the only source of order truth.
2. Do not require every worker tick to traverse the graph.
3. Do not treat message creation as incomplete until a later list pointer mutation happens.
4. Do not overload one conversation node property with an array of messages.
5. Do not mix indexing jobs and lane messages into the same queue abstraction unless there is a very clear reason.

---

## 15. Suggested APIs

These are conceptual APIs only.

### 15.1 Message emission

```python
emit_lane_message(
    conversation_id: str,
    inbox_id: str | None,
    sender_id: str,
    recipient_id: str,
    msg_type: str,
    payload: dict,
    *,
    run_id: str | None = None,
    step_id: str | None = None,
    correlation_id: str | None = None,
    reply_to_message_id: str | None = None,
    priority: int = 0,
) -> str
```

### 15.2 Projection dequeue

```python
claim_pending_messages(
    inbox_id: str,
    worker_id: str,
    limit: int = 1,
    lease_seconds: int = 30,
) -> list[ProjectedInboxMessage]
```

### 15.3 Completion

```python
complete_lane_message(message_id: str, result_payload: dict | None = None) -> None
fail_lane_message(message_id: str, error_payload: dict) -> None
```

---

## 16. Testing strategy

### 16.1 Unit tests

- create message node through entity event
- verify semantic edges are present at creation time
- project inbox membership and sequence
- claim pending messages from projection
- complete and fail transitions
- reply chain linkage

### 16.2 Concurrency tests

- multiple foreground senders append to same inbox
- verify sequence projection remains valid
- verify no lost pending messages
- verify lease expiration requeues safely

### 16.3 Recovery tests

- rebuild projection from event stream
- rebuild linked-list shortcuts from message membership + sequence
- verify latest and pending views after rebuild

### 16.4 End-to-end tests

- foreground sends background request
- worker claims and processes
- worker emits reply
- conversation view shows both messages
- run stream shows lifecycle transitions

---

## 17. Migration / rollout notes

1. Start with one worker inbox type, not all workers at once.
2. Keep current background job systems running while the new lane messaging path is introduced.
3. Use message projection in parallel with existing worker observability until confidence is high.
4. Add graph-native message support first, then switch selected flows over incrementally.

---

## 18. Open questions

1. Should inboxes be per worker type, per worker instance, or both?
2. Should conversation ordering and inbox ordering share the same sequence or use separate projected sequences?
3. Should lifecycle state transitions be only graph events, or also cached directly on projected rows?
4. Should linked-list edges be authoritative secondary structure or fully derived and disposable?
5. Should human/user messages and worker/system messages converge on the same message node schema long term?

---

## 19. Final decision

Adopt the following direction:

- **Use entity events to create semantically complete message nodes.**
- **Place messages in the same conversation graph namespace as foreground/background activity.**
- **Represent worker inboxes as graph objects.**
- **Use projection tables for sequence, claim, lease, retry, and fast newest/pending lookup.**
- **Treat linked-list message ordering as a derived or secondary structure, not the only source of correctness.**

This gives Kogwistar a graph-native inter-lane message system that is much more consistent with its AI operating system direction, while preserving operational safety and rebuildability.

