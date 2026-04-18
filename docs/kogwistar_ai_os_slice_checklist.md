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
- process identity and lifecycle
- operator views
- namespace / isolation semantics

### Stage 2 — AI operating core
Needs:
- capability kernel
- service supervision
- checkpoint / resume / recovery model
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
Foreground, background workers, tools, agents, and supervisors communicate through first-class system messages.

### Add
- [ ] Message node type for lane/system messages
- [ ] Inbox/outbox concept per worker / process / agent
- [ ] Message edges:
  - [ ] `sent_by`
  - [ ] `sent_to`
  - [ ] `reply_to`
  - [ ] `in_conversation`
  - [ ] `about_run`
  - [ ] `about_step`
- [ ] Durable message lifecycle states:
  - [ ] pending
  - [ ] claimed
  - [ ] completed
  - [ ] failed
  - [ ] cancelled
- [ ] Projection for fast inbox dequeue / newest-first access
- [ ] Correlation IDs for request/reply chains
- [ ] SSE / operator visibility for lane messages

### Done when
- [ ] A foreground path can send a message to a background worker
- [ ] A worker can claim, process, and reply
- [ ] The full conversation + run + reply chain is queryable
- [ ] Recovery is possible from durable state

### Why it matters
This is the IPC layer. Without it, Kogwistar feels like a framework with jobs. With it, it starts feeling like a system.

---

## Slice 2 — Universal Process Model

### Goal
Agents, workers, runs, and services become explicit process-like entities rather than ad hoc runtime concepts.

### Add
- [ ] Process node type
- [ ] Process kinds:
  - [ ] workflow run
  - [ ] agent
  - [ ] background worker
  - [ ] service / daemon
  - [ ] tool execution
- [ ] Parent-child process relationships
- [ ] Process states:
  - [ ] created
  - [ ] runnable
  - [ ] waiting
  - [ ] blocked
  - [ ] suspended
  - [ ] completed
  - [ ] failed
  - [ ] terminated
- [ ] Process metadata:
  - [ ] owner / tenant
  - [ ] namespace
  - [ ] priority class
  - [ ] policy class
  - [ ] creation time
  - [ ] exit reason
- [ ] Process event stream / registry projection

### Done when
- [ ] Every long-lived execution can be represented as a process entity
- [ ] Child process spawn is explicit
- [ ] Waiting-on relationships are visible
- [ ] Operators can inspect current process state

### Why it matters
This is the foundation for “agent OS” language. A system without a process model is not OS-like enough.

---

## Slice 3 — Namespaces and Isolation

### Goal
Kogwistar can isolate memory, communication, and process visibility between tenants / agents / workspaces.

### Add
- [ ] Namespace model for graph regions
- [ ] Tenant / workspace / project scopes
- [ ] Agent-private memory areas
- [ ] Shared memory areas
- [ ] Cross-namespace projection / pin / ref rules
- [ ] Message routing restrictions across namespaces
- [ ] Default visibility rules for reads/writes

### Done when
- [ ] An agent can have private state not globally visible
- [ ] Shared memory can be explicit, not accidental
- [ ] Cross-tenant leakage is structurally harder
- [ ] Message delivery respects namespace boundaries

### Why it matters
OS language implies memory and communication isolation.

---

## Slice 4 — Capability Kernel / Permissions

### Goal
Actions are governed by explicit capabilities rather than only external policy layers.

### Add
- [ ] Capability model for processes / agents / workers
- [ ] Capability types, e.g.:
  - [ ] read_graph
  - [ ] write_graph
  - [ ] send_message
  - [ ] spawn_process
  - [ ] invoke_tool
  - [ ] read_namespace
  - [ ] project_view
  - [ ] approve_action
- [ ] Capability checks at execution boundaries
- [ ] Deny / allow / require-approval outcomes
- [ ] Capability inheritance for child processes
- [ ] Capability revocation support
- [ ] Audit trail for capability decisions

### Done when
- [ ] Any significant action can be capability-checked
- [ ] Capability denials are durable and inspectable
- [ ] Child processes do not automatically get everything

### Why it matters
This is the kernel boundary for an AI OS.

---

## Slice 5 — System Call Surface

### Goal
Expose a minimal, stable API surface that other runtimes and agents can target.

### Add
- [ ] Core syscall-like operations, e.g.:
  - [ ] `spawn_process`
  - [ ] `terminate_process`
  - [ ] `send_message`
  - [ ] `receive_message`
  - [ ] `mount_memory`
  - [ ] `project_view`
  - [ ] `invoke_tool`
  - [ ] `checkpoint`
  - [ ] `resume`
  - [ ] `request_approval`
- [ ] Clear success / failure / blocked contracts
- [ ] Stable request/response schemas
- [ ] Versioning strategy for syscall surface

### Done when
- [ ] A simple external runtime can target Kogwistar through this surface
- [ ] The operations feel like system primitives, not app-specific helpers

### Why it matters
An operating system needs a usable system surface, not only internal modules.

---

## Slice 6 — Checkpoint / Suspend / Resume

### Goal
Processes can pause and continue later with durable continuity.

### Add
- [ ] Checkpoint model for process state
- [ ] Suspend semantics
- [ ] Resume semantics
- [ ] Waiting reasons:
  - [ ] approval wait
  - [ ] message wait
  - [ ] schedule delay
  - [ ] external callback wait
  - [ ] dependency wait
- [ ] Restart / resume from durable state
- [ ] Replay compatibility rules

### Done when
- [ ] A process can pause without losing semantic continuity
- [ ] A worker can resume work after restart
- [ ] Approval-blocked flows can resume correctly later

### Why it matters
This is the step from workflow execution into process runtime.

---

## Slice 7 — Policy-Driven Scheduler

### Goal
Kogwistar gains AI-layer scheduling policy, even if low-level CPU control stays with Linux.

### Add
- [ ] Priority classes:
  - [ ] foreground
  - [ ] background
  - [ ] latency-sensitive
  - [ ] batch
- [ ] Admission control
- [ ] Concurrency limits per tenant / process class
- [ ] Retry/backoff policy
- [ ] Queue fairness policy
- [ ] Dead-letter handling for failed work
- [ ] Optional policy hooks for external scheduler integration

### Done when
- [ ] Kogwistar can decide what should run first
- [ ] Kogwistar can defer, throttle, or reject work at the AI layer
- [ ] Different workload classes behave differently by policy

### Why it matters
You do not need hardware scheduling to have meaningful OS scheduling.

---

## Slice 8 — Resource Accounting and Budgets

### Goal
Processes and tenants have measurable budgets and usage records.

### Add
- [ ] Token budget tracking
- [ ] Tool/API cost tracking
- [ ] Active run count limits
- [ ] Message queue depth metrics
- [ ] Storage usage accounting
- [ ] Time budget / timeout classes
- [ ] Optional CPU/memory policy integration fields for host infra

### Done when
- [ ] The system can say who consumed what
- [ ] Policies can use budgets to allow, throttle, or stop work
- [ ] Operators can inspect current resource pressure

### Why it matters
An OS allocates and constrains resources, even if enforcement is delegated downward.

---

## Slice 9 — Tool / Device Subsystem

### Goal
Tools become first-class managed resources, similar to drivers/devices at the AI layer.

### Add
- [ ] Tool registry with stable identities
- [ ] Tool capability requirements
- [ ] Uniform invocation receipt model
- [ ] Tool side-effect logging
- [ ] Sync vs async tool contract support
- [ ] Tool classes:
  - [ ] pure/query tool
  - [ ] side-effecting tool
  - [ ] long-running tool
  - [ ] human approval tool
- [ ] Tool execution as process or child-process entity

### Done when
- [ ] Tool invocations are auditable and governable
- [ ] Tool behavior is not opaque to the substrate
- [ ] Long-running tools integrate with checkpoint/message flow

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

### Add
- [ ] Process table view
- [ ] Inbox/outbox view
- [ ] Queue depth and failed message views
- [ ] Waiting / blocked-on graph
- [ ] Capability inspection view
- [ ] Service health dashboard
- [ ] Budget / quota dashboard
- [ ] Event timeline / trace explorer

### Done when
- [ ] An operator can answer “what is happening now?” quickly
- [ ] Stuck flows can be located without code diving
- [ ] System state is legible

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
1. [ ] Slice 1 — Lane Messaging / Native IPC
2. [ ] Slice 2 — Universal Process Model
3. [ ] Slice 11 — Operator Views / System Introspection
4. [ ] Slice 3 — Namespaces and Isolation
5. [ ] Slice 4 — Capability Kernel / Permissions

## Mid-term order
6. [ ] Slice 6 — Checkpoint / Suspend / Resume
7. [ ] Slice 7 — Policy-Driven Scheduler
8. [ ] Slice 8 — Resource Accounting and Budgets
9. [ ] Slice 9 — Tool / Device Subsystem

## Later OS-hardening order
10. [ ] Slice 10 — Service / Daemon Model
11. [ ] Slice 12 — Recovery / Repair Utilities
12. [ ] Slice 5 — System Call Surface

Note: Slice 5 can start earlier as an experimental API, but it should not be called stable too soon.

---

# Minimal Credible “AI Operating System” Bar

Kogwistar starts to have a credible claim once these are present together:

- [ ] lane messaging / IPC
- [ ] process model
- [ ] namespaces / isolation
- [ ] capability checks
- [ ] checkpoint / resume
- [ ] service supervision
- [ ] operator views

Without all of them, it is still better described as a strong substrate / harness.

With them, “agent OS” becomes a structurally defensible description.

---

# Practical Positioning

Kogwistar does not need to become Linux.

It should aim to become:

> the AI-native operating layer that manages processes, messages, memory, tools, and governance on top of Linux / containers / cloud infrastructure.

That is both ambitious and realistic.

