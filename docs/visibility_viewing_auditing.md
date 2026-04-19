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

Principles:
- every visible action needs a traceable source event
- every operator view needs a declared scope
- every repair path needs a before/after audit trail
- no read path may silently mutate truth

## Slice Mapping

### Slice 3 - Namespaces and Isolation

- visibility boundary on reads and writes
- explicit `project_view` / shared memory / cross-scope pin-ref semantics
- audit boundary for denied access
- per-namespace read filters must be deterministic
- cross-scope access must emit allow/deny evidence
- shared views must still preserve subject ownership metadata

### Slice 4 - Capability Kernel

- subject-local approvals
- allow / deny / revoke are auditable facts
- effective capability snapshot is view, not source of truth
- approvals are keyed by subject, not global role alone
- revocation wins over prior grant
- capability snapshot must show grant source and revocation source

### Slice 5 - System Call Surface

- syscall entrypoints must be logged as intent and result
- `invoke_tool` and `request_approval` need branch-aware audit trail
- denied calls remain inspectable
- syscall logs must include actor, target, result, and error when present
- branching syscalls must preserve branch id / resume id / decision id
- tool invocation must not be recorded only in backend logs

### Slice 6 - Checkpoint / Suspend / Resume

- checkpoint, suspend, resume, replay, and contract views must line up
- resume keys must be inspectable without exposing hidden mutable state
- invalid-after-design-change resume failures must be visible and auditable
- checkpoints must identify the exact resume contract version
- replay should explain divergence, not hide it
- suspend state should show why it paused and who can resume it

### Slice 7 - Policy-Driven Scheduler

- scheduler decisions must be observable as policy + state, not opaque side effect
- pause/resume, priority, and lane selection must be inspectable
- blocked / deferred / resumed runs must be visible in operator views
- scheduler decisions must show policy input and chosen lane
- a paused run must remain queryable while paused
- priority inversion / starvation should be visible in timeline views

### Slice 8 - Resource Accounting and Budgets

- budget snapshots must be visible as ledger state
- runtime token/time/cost debit must be auditable per step or per run
- exhausted / deferred / replenished budget state must be inspectable
- budget view must distinguish workflow runtime budget from model token budget
- debit/credit events should be attributable to the exact step or service tick
- replenishment must be visible as a state transition, not implied

### Slice 9 - Tool / Device Subsystem

- tool invocation records must show request, permission, result, side effect
- device/tool capability and approval state must be inspectable
- subworkflow / device action traces must not vanish into backend-only logs
- pure/query, side-effecting, long-running, and human-approval tools should render differently
- side effects need node linkage when they create graph truth
- tool failures must preserve request payload and failure cause

### Slice 10 - Service / Daemon

- service enabled/disabled/heartbeat/trigger/restart events are visible
- service health is derived from latest heartbeat + policy
- operator can inspect child run linkage without mutating run truth
- service trigger source should be visible per event
- service repair must show projection rebuild, not rewrite execution truth
- child workflow runs remain the only execution truth

### Slice 11 - Operator Views / Introspection

- process table
- inbox / blocked / timeline
- dashboard and health views
- internal/admin-only until visibility rules mature
- operator views should be enough to answer "what is happening now?"
- operator views must remain projections, not a second truth source

### Slice 12 - Recovery / Repair

- repair actions are auditable
- replay / dead-letter / orphan repair must show before/after effect
- rebuilding projections must not rewrite authoritative history
- repair should be idempotent or safely repeatable
- dead-letter replay must keep original failure evidence
- orphan repair must not create phantom ownership

## Required View Classes

- `read` view: safe inspection, no mutation
- `operator` view: internal projection, broader than user view
- `audit` view: immutable event trail
- `repair` view: privileged corrective action trail

## Surface Map

- `read`: user-safe inspection endpoints
- `operator`: internal dashboards, inboxes, process tables, timeline views
- `audit`: immutable event history and decision trail
- `repair`: admin-only repair/replay/rebuild actions
- `CLI/MCP`: thin wrappers over the same view classes, not separate truth

## Concrete Query Surfaces

- `GET /api/workflow/visibility`
- `GET /api/workflow/scheduler/timeline`
- `GET /api/workflow/budget`
- `GET /api/workflow/budget/history`
- `GET /api/workflow/tools/audit`
- `GET /api/syscall/v1/audit`
- `workflow.visibility_snapshot`
- `workflow.scheduler_timeline`
- `workflow.budget_snapshot`
- `workflow.budget_history`
- `workflow.tool_audit`

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

## Event Layout

Each visible event should try to include:
- `event_type`
- `subject`
- `scope`
- `target_id`
- `run_id` / `conversation_id` / `service_id`
- `decision`
- `reason`
- `created_at_ms`

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
- syscall audit entry
- scheduler timeline entry
- budget history entry

## Anti-Patterns

- hidden auto-tick in read path
- view as second truth
- operator surface that bypasses namespace / capability rules
- repair path that silently mutates history
- scheduler that mutates without trace
- budget debit that cannot be explained from ledger events
- tool execution that leaves no traceable call/result pair
