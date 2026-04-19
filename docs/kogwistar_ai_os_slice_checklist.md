# Kogwistar Core — Slice Checklist Toward an AI Operating System

## Purpose

This checklist turns the broad “AI operating system” idea into concrete implementation slices.

It assumes:
- Kogwistar **does not replace Linux**
- Linux / containers / cloud infra remain the machine execution layer
- Kogwistar becomes the **agent/process/memory/governance operating layer**

The goal is not “be a kernel.”
The goal is to make Kogwistar the native environment for:
- agents
- workflows
- background workers
- tools
- approvals
- durable memory
- replayable execution

---

## Core Invariants

These invariants must hold across all slices.

### 1. Process truth reuses existing runtime truth

Do **not** introduce a second execution truth.

- `workflow_run` = authoritative process truth
- `workflow_step_exec` = authoritative process execution event truth
- `workflow_checkpoint` = authoritative process checkpoint truth
- any future “process table” is a **projection** over existing runtime truth

Implication:
- no separate top-level `process` entity as a second authoritative execution record
- “process” is an OS-facing model layered on existing runtime truth

### 2. Lane messages are first-class authoritative graph entities

Lane/system messages are **authoritative graph entities**, not mere serving projections.

Authoritative truth includes:
- message existence
- sender / recipient / inbox membership
- conversation linkage
- reply linkage
- run / step linkage when applicable
- lifecycle transitions as authoritative events

`meta_sql` stores only **assisting projections**, such as:
- per-inbox sequence indexes
- claimable / leased message views
- newest/latest lookup helpers
- operator query helpers
- unread / failed / dead-letter summaries

Implication:
- `meta_sql` MUST NOT become hidden message truth
- graph/event truth must remain sufficient to rebuild assisting projections

### 3. IPC truth is graph/event-authoritative; serving views are derived

Lane-message correctness must not depend on serving projection freshness.

Workers consume from authoritative message truth and/or authoritative lifecycle state, aided by `meta_sql` indexes.

Derived views may include:
- inbox newest-first views
- queue depth summaries
- operator dashboards
- SSE-friendly serving projections

Derived views MUST be disposable and rebuildable.

### 4. Namespace meanings are distinct

Do **not** overload one namespace concept for all purposes.

There are three separate meanings:

- **storage namespace**
  - existing storage / view partitioning (`scoped_namespace`-style concept)
- **execution namespace**
  - runtime / branch-safe execution scoping
- **security scope**
  - tenant / visibility / permission boundary

Implication:
- storage partitioning is not automatically a security boundary
- execution branch scoping is not automatically a tenant boundary

### 5. Checkpoint truth already exists and must be reused

Do **not** reinvent checkpoint/resume from scratch.

The OS-facing checkpoint/resume model must layer on the existing runtime checkpoint truth.

What must be formalized instead:
- persisted vs ephemeral state
- resume keys
- compatibility rules across design/runtime evolution

### 6. Operator surfaces must not outrun visibility controls

Until isolation / capabilities are implemented, operator views are **admin/internal only**.

Implication:
- early dashboards / SSE / inspection surfaces must be clearly labeled internal
- default public/user-facing introspection should wait for visibility filtering

---

## IPC Contract (Authoritative)

This contract must be written down before implementation continues.

### Delivery semantics
- at-least-once delivery
- duplicate delivery is possible under retry / lease expiry / recovery
- consumers/handlers MUST be idempotent

### Ordering semantics
- ordering guarantee is **per inbox**, not global
- per-inbox order is represented by authoritative sequence semantics
- assisting projections may materialize the order for fast scans

### Claim / lease semantics
- messages may be claimed with a lease
- leases may expire
- expired leases may be stolen / reclaimed
- lease steal must be explicit and auditable

### Lifecycle semantics
At minimum:
- pending
- claimed
- completed / acked
- failed
- cancelled
- requeued
- dead-lettered (recommended)

### Recovery semantics
- authoritative message truth must allow projection rebuild
- orphaned claims must be repairable
- duplicate redelivery after crash/restart must be expected

### Truth boundary
- authoritative: message entities + authoritative lifecycle events
- assisting projection: dequeue indexes, seq scans, dashboards, convenience views

---

## Checkpoint / Resume Contract (Authoritative)

Checkpoint/resume slices must respect existing runtime truth.

### Persisted vs ephemeral
This boundary must be explicit.

Persisted examples:
- run identity
- token state
- suspended location
- join/barrier/runtime coordination state that must survive restart
- checkpoint payloads needed for semantic continuity

Ephemeral examples:
- injected runtime deps
- process-local helper objects
- in-memory caches that can be reconstructed

### Resume keys
Resume requires explicit keys/identity sufficient to continue safely.
At minimum, define and preserve the existing runtime resume identity contract.

### Compatibility rules
When workflow design or runtime shape changes:
- either old checkpoints are migrated explicitly
- or they are marked non-resumable explicitly
- silent best-effort resume is not acceptable when semantics changed incompatibly

### Failure mode requirement
Invalid-after-design-change resume must fail loudly and inspectably, not silently drift.

---

## Acceptance Matrix (Must Exist Early)

Before later slices build on these primitives, acceptance coverage must exist for:

### Messaging
- [x] duplicate delivery is tolerated correctly
- [x] lease expiry and lease steal behave correctly
- [x] ack / requeue are idempotent
- [x] per-inbox ordering remains correct
- [x] recovery after worker crash preserves message truth
- [x] assisting projections rebuild correctly from authoritative truth

### Isolation / visibility
- [x] cross-security-scope message send deny
- [x] cross-security-scope read deny
- [x] shared-memory / shared-inbox cases work only when explicit

### Checkpoint / resume
- [x] resume after restart works when compatible
- [x] invalid-after-design-change resume fails safely
- [x] join/fanout restore behaves correctly
- [x] persisted vs ephemeral state boundary is honored

### Migration / compatibility
- [x] workflow/runtime artifacts remain compatible with the process projection model
- [x] lane-message assisting projections can be rebuilt/migrated safely
- [x] operator views do not become a second truth source

---

## Maturity Model

### Stage 0 — Strong substrate
Already broadly present in direction:
- graph-native knowledge and workflow substrate
- event-sourced provenance / replay orientation
- workflow execution runtime
- worker patterns
- projections / serving views
- governance ecosystem direction

### Stage 1 — OS-shaped substrate
Needs:
- first-class messaging across lanes
- process identity and lifecycle projection
- operator views
- namespace / isolation semantics

### Stage 2 — AI operating core
Needs:
- capability kernel
- service supervision
- checkpoint / resume / recovery contract formalization
- policy-driven scheduling and quotas
- stable syscall-like API surface

### Stage 3 — Credible agent OS
Needs:
- long-running agents as services
- safe multi-tenant memory model
- durable tool/device subsystem
- strong observability and control plane UX

---

# Slice Checklist

---

## Slice 1 — Lane Messaging / Native IPC

### Goal
Foreground, background workers, tools, agents, and supervisors communicate through first-class authoritative system messages.

Status: done.

### Add
- [x] First-class authoritative lane message entity family
- [x] Inbox/outbox concept per worker / process / agent
- [x] Authoritative message relations:
  - [x] `sent_by`
  - [x] `sent_to`
  - [x] `reply_to`
  - [x] `in_conversation`
  - [x] `about_run`
  - [x] `about_step`
  - [x] `in_inbox` or equivalent authoritative inbox membership
- [x] Authoritative lifecycle events / state transitions:
  - [x] pending
  - [x] claimed
  - [x] completed / acked
  - [x] failed
  - [x] cancelled
  - [x] requeued
  - [x] dead-lettered (recommended)
- [x] Correlation IDs for request/reply chains
- [x] Lease expiry / lease steal contract
- [x] Per-inbox ordering contract with sequence semantics
- [x] `meta_sql` assisting projections for:
  - [x] fast inbox dequeue / newest-first access
  - [x] claimable views
  - [x] lease expiry scans
  - [x] failed/dead-letter summaries
- [x] SSE / operator visibility for lane messages

### Done when
- [x] A foreground path can send an authoritative message to a background worker
- [x] A worker can claim, process, ack, and reply
- [x] Duplicate delivery does not corrupt correctness
- [x] Lease steal / recovery is test-covered
- [x] The full conversation + run + reply chain is queryable from authoritative graph truth
- [x] `meta_sql` projections can be rebuilt from authoritative truth

### Why it matters
This is the IPC layer. Without it, Kogwistar feels like a framework with jobs. With it, communication becomes part of the substrate itself.

---

## Slice 2 — Universal Process Model

### Goal
Agents, workers, runs, and services become explicit OS-facing process-like views without introducing a second execution truth.

Status: done.

### Add
- [x] Explicit mapping:
  - [x] `workflow_run` => process truth
  - [x] `workflow_step_exec` => process execution event truth
  - [x] `workflow_checkpoint` => process checkpoint truth
- [x] Process projection / operator-facing process table
- [x] Process kinds (projection/model classification), e.g.:
  - [x] workflow run
  - [x] agent
  - [x] background worker
  - [x] service / daemon
  - [x] tool execution
- [x] Parent-child process relationships where they can be derived truthfully
- [x] Process states (projected from existing truth), e.g.:
  - [x] created
  - [x] runnable
  - [x] waiting
  - [x] blocked
  - [x] suspended
  - [x] completed
  - [x] failed
  - [x] terminated
- [x] Process metadata / serving fields:
  - [x] owner / security scope
  - [x] storage namespace
  - [x] execution namespace
  - [x] priority class
  - [x] policy class
  - [x] creation time
  - [x] exit reason
- [x] Process event stream / registry projection

### Done when
- [x] Every long-lived execution can be represented through the process model without inventing a second truth
- [x] Child process spawn is explicit where semantically real
- [x] Waiting-on relationships are visible
- [x] Operators can inspect current process state through a projection over runtime truth

### Why it matters
This is the foundation for “agent OS” language. A system without a process model is not OS-like enough.

---

## Slice 3 — Namespaces and Isolation

### Goal
Kogwistar can isolate memory, communication, and process visibility between tenants / agents / workspaces without overloading existing namespace terms.

Status: done.

### Add
- [x] Explicit distinction between:
  - [x] storage namespace
  - [x] execution namespace
  - [x] security scope
- [x] Security scope model for tenant / workspace / project isolation
- [x] Agent-private memory areas
- [x] Shared memory areas
- [x] Cross-scope projection / pin / ref rules
- [x] Message routing restrictions across security scopes
- [x] Default visibility rules for reads/writes
- [x] Clear mapping rules between storage partitioning and security scope

### Done when
- [x] An agent can have private state not globally visible
- [x] Shared memory can be explicit, not accidental
- [x] Cross-tenant leakage is structurally harder
- [x] Message delivery respects security scope boundaries
- [x] Existing storage namespace semantics remain intact

### Why it matters
OS language implies memory and communication isolation.

---

## Slice 4 — Capability Kernel / Permissions

### Goal
Actions are governed by explicit capabilities rather than only external policy layers.

Status: done.

### Add
- [x] Capability model for processes / agents / workers
- [x] Capability types, e.g.:
  - [x] read_graph
  - [x] write_graph
  - [x] send_message
  - [x] spawn_process
  - [x] invoke_tool
  - [x] read_security_scope
  - [x] project_view
- [x] approve_action
- [x] Capability checks at execution boundaries
- [x] Deny / allow / require-approval outcomes
- [x] Capability inheritance for child processes
- [x] Capability revocation support
- [x] Audit trail for capability decisions

### Done when
- [x] Any significant action can be capability-checked
- [x] Capability denials are durable and inspectable
- [x] Child processes do not automatically get everything

### Why it matters
This is the kernel boundary for an AI OS.

---

## Slice 5 — System Call Surface

### Goal
Expose a minimal, stable API surface that other runtimes and agents can target.

Status: done.

### Add
- [x] Core syscall-like operations, e.g.:
  - [x] `spawn_process`
  - [x] `terminate_process`
  - [x] `send_message`
  - [x] `receive_message`
  - [x] `mount_memory`
  - [x] `project_view`
  - [x] `invoke_tool`
  - [x] `checkpoint`
  - [x] `resume`
  - [x] `request_approval`
- [x] Clear success / failure / blocked contracts
- [x] Stable request/response schemas
- [x] Versioning strategy for syscall surface

### Done when
- [x] A simple external runtime can target Kogwistar through this surface
- [x] The operations feel like system primitives, not app-specific helpers

### Why it matters
An operating system needs a usable system surface, not only internal modules.

---

## Slice 6 — Checkpoint / Suspend / Resume

### Goal
Processes can pause and continue later with durable continuity by reusing existing runtime checkpoint truth.

Status: done.

### Add
- [x] Formalized checkpoint contract over existing runtime truth
- [x] Explicit persisted vs ephemeral state rules
- [x] Explicit resume identity / key contract
- [x] Suspend semantics
- [x] Resume semantics
- [x] Waiting reasons:
  - [x] approval wait
  - [x] message wait
  - [x] schedule delay
  - [x] external callback wait
  - [x] dependency wait
- [x] Restart / resume from durable state
- [x] Replay compatibility rules
- [x] Design-change compatibility rules

### Done when
- [x] A process can pause without losing semantic continuity
- [x] A worker can resume work after restart
- [x] Approval-blocked flows can resume correctly later
- [x] Invalid-after-design-change resumes fail safely and inspectably
- [x] Persisted vs ephemeral boundaries are test-covered

### Why it matters
This is the step from workflow execution into process runtime.

---

## Slice 7 — Policy-Driven Scheduler

### Goal
Kogwistar gains AI-layer scheduling policy, even if low-level CPU control stays with Linux.

Status: done.

### Add
- [x] Priority classes:
  - [x] foreground
  - [x] background
  - [x] latency-sensitive
  - [x] batch
- [x] Admission control
- [x] Concurrency limits per tenant / process class
- [x] Retry/backoff policy
- [x] Queue fairness policy
- [x] Dead-letter handling for failed work
- [x] Optional policy hooks for external scheduler integration

### Done when
- [x] Kogwistar can decide what should run first
- [x] Kogwistar can defer, throttle, or reject work at the AI layer
- [x] Different workload classes behave differently by policy

### Why it matters
You do not need hardware scheduling to have meaningful OS scheduling.

---

## Slice 8 — Resource Accounting and Budgets

### Goal
Processes and tenants have measurable budgets and usage records.

Status: done.

### Add
- [x] Token budget tracking
- [x] Tool/API cost tracking
- [x] Active run count limits
- [x] Message queue depth metrics
- [x] Storage usage accounting
- [x] Time budget / timeout classes
- [x] Optional CPU/memory policy integration fields for host infra

### Done when
- [x] The system can say who consumed what
- [x] Policies can use budgets to allow, throttle, or stop work
- [x] Operators can inspect current resource pressure

### Why it matters
An OS allocates and constrains resources, even if enforcement is delegated downward.

---

## Slice 9 — Tool / Device Subsystem

### Goal
Tools become first-class managed resources, similar to drivers/devices at the AI layer.

Status: done.

### Add
- [x] Tool registry with stable identities
- [x] Tool capability requirements
- [x] Uniform invocation receipt model
- [x] Tool side-effect logging
- [x] Sync vs async tool contract support
- [x] Tool classes:
  - [x] pure/query tool
  - [x] side-effecting tool
  - [x] long-running tool
  - [x] human approval tool
- [x] Tool execution as process or child-process entity

### Done when
- [x] Tool invocations are auditable and governable
- [x] Tool behavior is not opaque to the substrate
- [x] Long-running tools integrate with checkpoint/message flow

### Why it matters
For an AI OS, tools are the equivalent of devices and drivers.

---

## Slice 10 — Service / Daemon Model

### Goal
Always-on agents and workers become managed services.

### Add
- [ ] Service node / definition model
- [ ] Service lifecycle:
  - [ ] enabled
  - [ ] starting
  - [ ] healthy
  - [ ] degraded
  - [ ] restarting
  - [ ] stopped
- [ ] Restart policy
- [ ] Health check / heartbeat model
- [ ] Auto-start policy
- [ ] Trigger sources:
  - [ ] schedule
  - [ ] message arrival
  - [ ] graph change
  - [ ] external event

### Done when
- [ ] A long-running agent can be declared and supervised
- [ ] Crashed services can be restarted with policy
- [ ] Health is visible to operators

### Why it matters
This is how “agent 1 always running” becomes real.

---

## Slice 11 — Operator Views / System Introspection

### Goal
Kogwistar becomes operable like a system, not just inspectable by ad hoc graph queries.

Status: in progress.

### Add
- [x] Process table view
- [ ] Inbox/outbox view
- [ ] Queue depth and failed message views
- [ ] Waiting / blocked-on graph
- [ ] Capability inspection view
- [ ] Service health dashboard
- [ ] Budget / quota dashboard
- [x] Event timeline / trace explorer
- [x] Internal/admin-only visibility guard for early versions

### Done when
- [x] An operator can answer “what is happening now?” quickly
- [ ] Stuck flows can be located without code diving
- [ ] System state is legible
- [ ] Early operator surfaces are not accidentally user-facing before isolation/capabilities land

### Why it matters
Operating systems need operator ergonomics, not only correctness.

---

## Slice 12 — Recovery / Repair Utilities

### Goal
The system can repair projections, recover failed queues, and reconcile durable state.

### Add
- [ ] Rebuild inbox projections from authoritative graph/events
- [ ] Rebuild process table projections
- [ ] Repair orphaned claimed messages / leases
- [ ] Replay selected process histories
- [ ] Dead-letter inspection and replay
- [ ] Migration safety tools for schema evolution

### Done when
- [ ] Operators can recover from projection corruption
- [ ] The system is resilient to partial failures
- [ ] “Authoritative truth vs serving projection” is operationally real

### Why it matters
This is essential if Kogwistar is serious about evented, replayable system behavior.

---

# Recommended Order

## Near-term order
1. [x] Slice 1 — Lane Messaging / Native IPC
2. [x] Slice 2 — Universal Process Model
3. [ ] Slice 3 — Namespaces and Isolation
4. [x] Slice 4 — Capability Kernel / Permissions
5. [x] Slice 5 — System Call Surface
6. [ ] Slice 11 — Operator Views / System Introspection (internal/admin-only first)

## Mid-term order
6. [x] Slice 6 — Checkpoint / Suspend / Resume
7. [x] Slice 7 — Policy-Driven Scheduler
8. [x] Slice 8 — Resource Accounting and Budgets
9. [x] Slice 9 — Tool / Device Subsystem

## Later OS-hardening order
10. [ ] Slice 10 — Service / Daemon Model
11. [ ] Slice 12 — Recovery / Repair Utilities

Note: Slice 5 can start earlier as an experimental API, but it should not be called stable too soon.

---

# Minimal Credible “AI Operating System” Bar

Kogwistar starts to have a credible claim once these are present together:

- [x] lane messaging / IPC
- [x] process model
- [ ] namespaces / isolation
- [ ] capability checks
- [x] checkpoint / resume
- [ ] service supervision
- [x] operator views

Without the remaining service supervision / recovery utilities, it is still better described as a strong substrate / harness.

With them, “agent OS” becomes a structurally defensible description.

---

# Practical Positioning

Kogwistar does not need to become Linux.

It should aim to become:

> the AI-native operating layer that manages processes, messages, memory, tools, and governance on top of Linux / containers / cloud infrastructure.

That is both ambitious and realistic.

