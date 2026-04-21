# Kogwistar Async Runtime Implementation Checklist

Companion ARD: `docs/kogwistar_async_runtime_ard.md`

Status: Draft checklist. The existing `WorkflowRuntime` is the semantic reference. `AsyncWorkflowRuntime` is not implemented yet.

---

## Phase 0 - Current-State Audit

- [x] identify the existing sync runtime reference: `kogwistar.runtime.runtime.WorkflowRuntime`
- [x] identify current resolver registry: `kogwistar.runtime.resolvers.MappingStepResolver`
- [x] identify current workflow graph contract: `kogwistar.runtime.contract`
- [x] identify existing replay/state reducer reference: `kogwistar.runtime.replay`
- [ ] inventory all runtime entrypoints that currently instantiate `WorkflowRuntime`
- [ ] inventory tests that must pass unchanged for both sync and async runtimes
- [ ] decide public export name and import path for `AsyncWorkflowRuntime`
- [ ] inventory suspend/resume, cancellation, nested workflow, and trace tests that define sync semantics
- [ ] inventory LangGraph semantics-mode tests that must stay reducer-compatible

Notes:

- Current runtime uses a `ThreadPoolExecutor` scheduler.
- Current resolver contract is sync-shaped and returns `StepRunResult`.
- No repo-native `AsyncWorkflowRuntime` class appears to exist yet.

---

## Phase 1 - Shared Runtime Contract

- [ ] define a runtime-neutral workflow executor interface
- [ ] define `SyncStepFn` and `AsyncStepFn` type aliases
- [ ] define how sync functions are adapted into async runtime
- [ ] define how async functions are rejected or wrapped by sync runtime
- [ ] keep `StepContext` shape identical across runtimes
- [ ] keep `StepRunResult` semantics identical across runtimes
- [ ] keep `WorkflowRun -> Node -> ExecutionAttempt -> Executor` terminology consistent in docs and code
- [ ] include `Token` in shared architecture terminology
- [ ] define runtime-neutral terminal status vocabulary
- [ ] add tests proving both runtimes accept the same workflow graph model

Acceptance tests:

- [ ] sync and async runtimes load the same `WorkflowSpec`
- [ ] sync and async runtimes produce equivalent terminal status for a linear workflow
- [ ] sync and async runtimes produce equivalent state for branching and joins
- [ ] async runtime can execute sync resolvers through an adapter
- [ ] async runtime can execute native async resolvers

---

## Phase 2 - Resolver Compatibility

- [ ] add `AsyncMappingStepResolver` or extend `MappingStepResolver` with async-safe resolution
- [ ] expose `ops` for sync and async resolver registries
- [ ] add a canonical check that `default_sync_ops == default_async_ops`
- [ ] preserve sandboxed op behavior or explicitly mark it unsupported in async first slice
- [ ] preserve nested workflow invocation semantics
- [ ] preserve state schema inference or provide equivalent async metadata
- [ ] ensure resolver exceptions become `RunFailure`, not leaked task crashes
- [ ] ensure blocking sync handlers do not block unrelated async tasks
- [ ] define adapter behavior for CPU-bound versus IO-bound sync handlers
- [ ] preserve dependency injection via `_deps` as process-local state

Acceptance tests:

- [ ] registered sync op appears in async op set
- [ ] registered async op appears in async op set
- [ ] missing op behavior matches sync runtime
- [ ] exception-to-`RunFailure` behavior matches sync runtime
- [ ] legacy `update` warning behavior is either preserved or intentionally retired with docs
- [ ] async runtime runs two independent awaited handlers concurrently
- [ ] `_deps` is available during live execution but omitted from checkpoints

---

## Phase 3 - Async Scheduler Core

- [ ] implement `AsyncWorkflowRuntime`
- [ ] replace thread worker scheduling with task-based scheduling
- [ ] preserve deterministic edge ordering by priority
- [ ] preserve token ids, parent token ids, and branch masks
- [ ] preserve token nesting semantics: first branch continues current token, later branches get child tokens
- [ ] preserve fanout behavior
- [ ] preserve join/barrier behavior
- [ ] preserve explicit-join semantics: convergence without `wf_join` may execute once per token
- [ ] preserve route-next behavior
- [ ] preserve cancellation checkpoints
- [ ] preserve terminal status artifacts
- [ ] enforce max concurrent tasks
- [ ] ensure task cancellation does not corrupt persisted run state
- [ ] ensure parent run does not finish while required child tasks remain pending
- [ ] define deterministic scheduler acceptance order for completed async tasks

Acceptance tests:

- [ ] async linear workflow reaches terminal
- [ ] async branching workflow converges at join
- [ ] async fanout respects `wf_fanout` and `wf_multiplicity`
- [ ] async convergence without explicit join can execute downstream once per arriving token
- [ ] async route-next matches sync route-next
- [ ] async cancellation persists cancelled terminal state
- [ ] async runtime does not leave orphan pending tasks on terminal exit
- [ ] async completed-task race still applies state in deterministic acceptance order

---

## Phase 4 - State and Replay Equivalence

- [ ] share state update application logic with `WorkflowRuntime`
- [ ] verify async runtime uses the same merge semantics as `WorkflowRuntime.apply_state_update`
- [ ] verify async runtime matches `kogwistar.runtime.replay` reducer semantics
- [ ] keep checkpoint payload schema compatible
- [ ] keep run and step sequence semantics compatible
- [ ] checkpoint token ids and parent token ids
- [ ] checkpoint join/barrier bookkeeping needed for resume
- [ ] checkpoint pending/scheduled tokens and in-flight obligations
- [ ] preserve `_deps` as non-checkpointed process-local data
- [ ] ensure wall-clock timestamps are audit facts, not replay ordering inputs
- [ ] ensure replay order is `seq` / `step_seq`, not `created_at_ms`
- [ ] define graph side-effect normalization for sync/async comparisons
- [ ] compare accepted node and edge side effects, not only final state

Acceptance tests:

- [ ] sync and async runtimes produce identical final state for the same deterministic workflow
- [ ] replay of async checkpoints yields the same state as live async execution
- [ ] async checkpoint restore resumes from the same next pending node as sync restore
- [ ] changing `created_at_ms` does not change replay result
- [ ] checkpoint restore preserves token nesting and join counters
- [ ] replay does not require `_deps` to be serialized
- [ ] side-by-side sync/async run produces the same normalized persisted node set
- [ ] side-by-side sync/async run produces the same normalized persisted edge set
- [ ] side-by-side sync/async run produces the same terminal status artifact

---

## Phase 4A - Suspend, Resume, and Cancellation

- [ ] preserve `RunSuspended` as a normal runtime result
- [ ] persist suspended token id, node id, wait reason, and checkpoint state
- [ ] resume from suspended token with client-provided result
- [ ] ensure resume applies state update through shared merge semantics
- [ ] ensure cancellation stops new scheduling after acceptance
- [ ] define policy for already in-flight async tasks during cancellation
- [ ] persist exactly one cancelled terminal artifact
- [ ] ensure cancellation is idempotent
- [ ] classify cooperative task cancellation markers as async diagnostics, not semantic workflow side effects

Acceptance tests:

- [ ] async suspended run can resume to success
- [ ] async suspended run can resume to failure when client result fails
- [ ] async cancellation before scheduling persists cancelled state
- [ ] async cancellation while tasks are in-flight leaves no corrupt checkpoint
- [ ] repeated cancellation request is idempotent
- [ ] async-only cancellation diagnostics do not affect normalized graph side-effect parity

---

## Phase 4B - Nested Workflow Semantics

- [ ] support child workflow invocation from async runtime
- [ ] apply child result to parent through shared merge semantics
- [ ] preserve parent/child run ids and trace linkage
- [ ] ensure parent waits for required child completion
- [ ] define cancellation propagation between parent and child
- [ ] define failure propagation from child to parent
- [ ] avoid duplicate trace/event sinks for nested async runs

Acceptance tests:

- [ ] async parent workflow invokes child workflow and merges child result
- [ ] child failure produces deterministic parent failure
- [ ] parent cancellation cancels or isolates child according to policy
- [ ] nested async traces are linked and non-duplicated

---

## Phase 5 - Backend-Aware Durability

- [ ] define async runtime write unit boundaries
- [ ] preserve Postgres strong transaction behavior where the backend supports it
- [ ] preserve Chroma event-first eventual repair behavior
- [ ] preserve in-memory semantics without pretending crash recovery exists
- [ ] share transaction policy decisions with sync runtime where possible
- [ ] make per-step commit behavior explicit
- [ ] ensure event log and index jobs are not duplicated during replay/rebuild
- [ ] test mid-step failure behavior by backend family
- [ ] ensure unit-of-work state is task-local for concurrent async steps
- [ ] ensure nested UoW in the same task joins the outer scope
- [ ] ensure rollback in one async task does not roll back another task's committed scope

Acceptance tests:

- [ ] Postgres-backed async step commit is atomic or rolls back cleanly
- [ ] Chroma-backed async step failure is repairable from event truth
- [ ] in-memory backend documents volatile recovery limits
- [ ] async replay does not append new authoritative history
- [ ] async runtime respects backend unit-of-work boundaries
- [ ] concurrent async UoWs are isolated by task context
- [ ] nested async UoW joins inside one task

---

## Phase 6 - Trace, Observability, and Events

- [ ] reuse `EventEmitter` semantics or provide async equivalent
- [ ] preserve workflow trace node and edge shape
- [ ] preserve `WorkflowRunNode`, `WorkflowStepExecNode`, and checkpoint artifacts
- [ ] preserve trace fast-path behavior where safe
- [ ] ensure nested async runtimes do not duplicate trace sinks
- [ ] ensure task durations and status reporting are best-effort and non-fatal
- [ ] add structured async task lifecycle events
- [ ] expose scheduled, started, completed, failed, suspended, and cancelled task states
- [ ] ensure trace write failures follow explicit policy
- [ ] separate durable semantic runtime events from async-only diagnostic telemetry

Acceptance tests:

- [ ] async runtime writes trace artifacts with the same metadata fields as sync runtime
- [ ] async nested workflow invocation reuses or safely isolates trace emitter
- [ ] trace write failure does not corrupt workflow state unless policy says fatal
- [ ] async runtime exposes enough events for server progress APIs
- [ ] async diagnostic telemetry can be ignored without changing replay or side-effect parity

---

## Phase 7 - API and Integration

- [ ] export `AsyncWorkflowRuntime` from `kogwistar.runtime`
- [ ] add server runner option for async runtime
- [ ] add conversation service injection point for runtime class selection
- [ ] ensure existing sync runtime remains default until parity is proven
- [ ] add compatibility shims for tests and demos
- [ ] document how callers choose sync versus async runtime
- [ ] document unsupported async first-slice features, if any

Acceptance tests:

- [ ] package import exposes `AsyncWorkflowRuntime`
- [ ] server/runtime path can run one workflow with async runtime when explicitly selected
- [ ] conversation orchestrator can inject async runtime without changing workflow design
- [ ] sync runtime entrypoints continue to work unchanged
- [ ] sync runtime remains default when no async runtime option is supplied

---

## Phase 8 - Parity Matrix

- [ ] accepted graph node side-effect parity
- [ ] accepted graph edge side-effect parity
- [ ] linear workflow parity
- [ ] branching parity
- [ ] join/barrier parity
- [ ] fanout parity
- [ ] route-next parity
- [ ] suspend/resume parity
- [ ] cancellation parity
- [ ] nested workflow parity
- [ ] sandboxed op parity or documented gap
- [ ] checkpoint replay parity
- [ ] backend durability parity
- [ ] trace artifact parity
- [ ] server progress event parity
- [ ] LangGraph semantics-mode reducer parity
- [ ] visual-mode non-parity documented

Parity rule:

For a deterministic workflow and fixed resolver outputs, sync and async runtimes should produce the same terminal status, accepted state, persisted run artifacts, and replay result. Timing fields may differ and must not participate in semantic equality.

LangGraph parity applies only to `execution="semantics"`. Visual export is not a replay or durability contract.

Graph side-effect parity applies to accepted semantic artifacts: run nodes, step execution nodes, checkpoint nodes, runtime edges, terminal artifacts, and durable events used by replay or server progress. Async-only task lifecycle diagnostics may differ if they are explicitly classified as diagnostics and excluded from replay semantics.

---

## Phase 9 - CI Coverage

- [ ] add lightweight `ci` tests for contract and import boundaries
- [ ] add `core` tests for resolver compatibility and state reducer parity
- [ ] add `workflow` tests for runtime execution parity
- [ ] add `ci_full` tests for backend-specific durability and Chroma/Postgres behavior
- [ ] add a small smoke test that proves `default_sync_ops == default_async_ops`
- [ ] avoid making slow backend tests the only guard for semantic parity
- [ ] add a small CI test proving timestamp fields do not affect replay ordering
- [ ] add a small CI test proving `_deps` stays out of checkpoints
- [ ] add side-by-side sync/async graph side-effect parity tests with normalized volatile fields

Suggested initial tests:

- [ ] `tests/runtime/test_async_runtime_contract.py`
- [ ] `tests/runtime/test_async_runtime_parity.py`
- [ ] `tests/runtime/test_async_runtime_replay.py`
- [ ] `tests/runtime/test_async_runtime_backend_durability.py`
- [ ] `tests/runtime/test_async_runtime_suspend_resume.py`
- [ ] `tests/runtime/test_async_runtime_nested.py`
- [ ] `tests/runtime/test_async_runtime_graph_side_effect_parity.py`

---

## Acceptance Checklist

- [ ] one workflow design model runs on both runtimes
- [ ] sync and async resolvers share one semantic contract
- [ ] `default_sync_ops == default_async_ops` is enforced by tests
- [ ] async runtime state updates match sync runtime state updates
- [ ] async runtime replay matches live execution
- [ ] backend durability semantics are explicit by backend family
- [ ] trace artifacts and progress events remain compatible
- [ ] wall-clock timestamps are audit facts only
- [ ] token nesting, explicit join, and checkpoint resume semantics match sync runtime
- [ ] suspend, resume, cancellation, and nested workflow behavior match sync runtime or are explicitly documented as first-slice gaps
- [ ] existing workflows have identical normalized graph node/edge side effects under sync and async runtimes
- [ ] async-only cooperative cancellation diagnostics are excluded from semantic side-effect parity
- [ ] async task-local UoW isolation is tested
- [ ] LangGraph parity claims are limited to semantics mode
- [ ] sync runtime remains stable while async runtime lands incrementally
