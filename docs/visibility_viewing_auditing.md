# Visibility / Viewing / Auditing

This doc ties slice semantics to what operators can see and how actions are audited.

## Canonical Shape

```text
authoritative truth
  -> runtime events
  -> operator projections
  -> CLI / MCP views
```

Visibility is not truth. Views are projections.

## Slice Mapping

### Slice 3 - Namespaces and Isolation

- visibility boundary on reads and writes
- explicit `project_view` / shared memory / cross-scope pin-ref semantics
- audit boundary for denied access

### Slice 4 - Capability Kernel

- subject-local approvals
- allow / deny / revoke are auditable facts
- effective capability snapshot is view, not source of truth

### Slice 5 - System Call Surface

- syscall entrypoints must be logged as intent and result
- `invoke_tool` and `request_approval` need branch-aware audit trail
- denied calls remain inspectable

### Slice 6 - Checkpoint / Suspend / Resume

- checkpoint, suspend, resume, replay, and contract views must line up
- resume keys must be inspectable without exposing hidden mutable state
- invalid-after-design-change resume failures must be visible and auditable

### Slice 7 - Policy-Driven Scheduler

- scheduler decisions must be observable as policy + state, not opaque side effect
- pause/resume, priority, and lane selection must be inspectable
- blocked / deferred / resumed runs must be visible in operator views

### Slice 8 - Resource Accounting and Budgets

- budget snapshots must be visible as ledger state
- runtime token/time/cost debit must be auditable per step or per run
- exhausted / deferred / replenished budget state must be inspectable

### Slice 9 - Tool / Device Subsystem

- tool invocation records must show request, permission, result, side effect
- device/tool capability and approval state must be inspectable
- subworkflow / device action traces must not vanish into backend-only logs

### Slice 10 - Service / Daemon

- service enabled/disabled/heartbeat/trigger/restart events are visible
- service health is derived from latest heartbeat + policy
- operator can inspect child run linkage without mutating run truth

### Slice 11 - Operator Views / Introspection

- process table
- inbox / blocked / timeline
- dashboard and health views
- internal/admin-only until visibility rules mature

### Slice 12 - Recovery / Repair

- repair actions are auditable
- replay / dead-letter / orphan repair must show before/after effect
- rebuilding projections must not rewrite authoritative history

## Required View Classes

- `read` view: safe inspection, no mutation
- `operator` view: internal projection, broader than user view
- `audit` view: immutable event trail
- `repair` view: privileged corrective action trail

## Slice To View Matrix

| Slice | read | operator | audit | repair |
| --- | --- | --- | --- | --- |
| 3 namespaces / isolation | yes | yes | yes | no |
| 4 capability kernel | yes | yes | yes | no |
| 5 syscall surface | yes | yes | yes | no |
| 6 checkpoint / resume | yes | yes | yes | limited |
| 7 scheduler | yes | yes | yes | limited |
| 8 budgets | yes | yes | yes | limited |
| 9 tool / device | yes | yes | yes | no |
| 10 service / daemon | yes | yes | yes | yes |
| 11 operator views | yes | yes | yes | yes |
| 12 recovery / repair | yes | yes | yes | yes |

## Audit Facts To Keep

- actor subject
- scope / namespace
- action name
- target object id
- allow / deny / revoke decision
- source of trigger
- timestamp
- correlation ids: conversation, run, service, workflow
- before / after projection hash when repair or replay happens
- policy decision inputs for scheduler and budget deferrals
- checkpoint identity and resume contract identity
- tool call id / result id / side-effect node id

## Anti-Patterns

- hidden auto-tick in read path
- view as second truth
- operator surface that bypasses namespace / capability rules
- repair path that silently mutates history
- scheduler that mutates without trace
- budget debit that cannot be explained from ledger events
- tool execution that leaves no traceable call/result pair
