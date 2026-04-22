# Kogwistar Async Runtime Implementation Checklist

Companion ARD: `docs/kogwistar_async_runtime_ard.md`

Status: Active implementation.
Current snapshot:
- Phase 0: completed
- Phase 1: completed (first-slice delegation parity scope)
- Phase 2: completed
- Phase 3: completed (experimental native async scheduler path landed for linear/fanout/route-next/join/cancel/resume basics with bounded concurrency and deterministic acceptance order)
- Phase 4+: not started

---

## Phase 0 - Current-State Audit

- [x] identify the existing sync runtime reference: `kogwistar.runtime.runtime.WorkflowRuntime`
- [x] identify current resolver registry: `kogwistar.runtime.resolvers.MappingStepResolver`
- [x] identify current workflow graph contract: `kogwistar.runtime.contract`
- [x] identify existing replay/state reducer reference: `kogwistar.runtime.replay`
- [x] inventory all runtime entrypoints that currently instantiate `WorkflowRuntime`
- [x] inventory tests that must pass unchanged for both sync and async runtimes
- [x] decide public export name and import path for `AsyncWorkflowRuntime` (`kogwistar.runtime.AsyncWorkflowRuntime`)
- [x] inventory suspend/resume, cancellation, nested workflow, and trace tests that define sync semantics
- [x] inventory LangGraph semantics-mode tests that must stay reducer-compatible

Notes:

- Current runtime uses a `ThreadPoolExecutor` scheduler.
- Current resolver contract is sync-shaped and returns `StepRunResult`.
- First-slice `AsyncWorkflowRuntime` exists at `kogwistar.runtime.async_runtime.AsyncWorkflowRuntime`.
- Phase 1 parity assertions here are first-slice delegation parity (facade contract), not async task scheduler parity.
- Phase 2 nested-invocation compatibility in first slice is preserved by resolver adapter metadata forwarding
  (`nested_ops`, `_state_schema`, `close_sandbox_run`) so sync runtime policy hooks still apply.
- Runtime entrypoints inventory (production paths):
  - `kogwistar/conversation/conversation_orchestrator.py`
  - `kogwistar/conversation/agentic_answering.py`
  - `kogwistar/server/chat_service_run_execution.py` (run + resume default runtime runner)
  - `kogwistar/runtime/perf_profile.py`
  - Demo entrypoints under `kogwistar/demo/*` using `WorkflowRuntime(...)`
- Sync-semantic test inventory (must remain semantic baseline):
  - Suspend/Resume: `tests/runtime/test_workflow_suspend_resume.py`, `tests/runtime/test_resume_wait_reasons.py`
  - Cancellation: `tests/runtime/test_workflow_cancel_event_sourced.py`
  - Nested/Trace: `tests/runtime/test_trace_sink_parallel_nested_minimal.py`, `tests/workflow/test_tracing_e2e.py`
  - Core run semantics: `tests/runtime/test_workflow_invocation_and_route_next.py`, `tests/workflow/test_workflow_join.py`, `tests/workflow/test_save_load_progress.py`, `tests/runtime/test_workflow_terminal_status.py`
- LangGraph semantics/reducer-compatible inventory:
  - `tests/workflow/test_langgraph_converter.py`
  - `tests/workflow/test_langgraph_converter_fanout.py`
  - `tests/workflow/test_langgraph_converter_parallel_integration.py`
  - `tests/workflow/test_langgraph_converter_real_parallel.py`
  - `tests/workflow/test_langgraph_blob_state.py`
  - `tests/workflow/test_langgraph_blobstate_seq.py`

Test evidence:

- Phase 0 is audit-only. No direct acceptance test. Evidence is inventory + baseline semantic test sets above.

---

## Phase 1 - Shared Runtime Contract

- [x] define a runtime-neutral workflow executor interface
- [x] define `SyncStepFn` and `AsyncStepFn` type aliases
- [x] define how sync functions are adapted into async runtime
- [x] define how async functions are rejected or wrapped by sync runtime
- [x] keep `StepContext` shape identical across runtimes
- [x] keep `StepRunResult` semantics identical across runtimes
- [x] keep `WorkflowRun -> Node -> ExecutionAttempt -> Executor` terminology consistent in docs and code
- [x] include `Token` in shared architecture terminology
- [x] define runtime-neutral terminal status vocabulary
- [x] add tests proving both runtimes accept the same workflow graph model

Acceptance tests:

- [x] sync and async runtimes load the same `WorkflowSpec` -> `test_sync_and_async_runtime_accept_same_workflow_graph_model`
- [x] sync and async runtimes produce equivalent terminal status for a linear workflow -> `test_async_runtime_linear_terminal_status_equivalent_to_sync`
- [x] sync and async runtimes produce equivalent state for branching and joins -> `test_async_runtime_branch_join_status_and_state_equivalent_to_sync`
- [x] async runtime can execute sync resolvers through an adapter -> `test_async_adapter_accepts_sync_handler`
- [x] async runtime can execute native async resolvers -> `test_async_adapter_accepts_async_handler`

---

## Phase 2 - Resolver Compatibility

- [x] add `AsyncMappingStepResolver` or extend `MappingStepResolver` with async-safe resolution
- [x] expose `ops` for sync and async resolver registries
- [x] add a canonical check that `default_sync_ops == default_async_ops`
- [x] preserve sandboxed op behavior or explicitly mark it unsupported in async first slice
- [x] preserve nested workflow invocation semantics
- [x] preserve state schema inference or provide equivalent async metadata
- [x] ensure resolver exceptions become `RunFailure`, not leaked task crashes
- [x] ensure blocking sync handlers do not block unrelated async tasks
- [x] define adapter behavior for CPU-bound versus IO-bound sync handlers
- [x] preserve dependency injection via `_deps` as process-local state

Acceptance tests:

- [x] registered sync op appears in async op set -> `test_default_sync_ops_equal_default_async_ops`
- [x] registered async op appears in async op set -> `test_registered_async_op_appears_in_async_op_set`
- [x] missing op behavior matches sync runtime -> `test_missing_op_behavior_matches_sync_resolver`
- [x] exception-to-`RunFailure` behavior matches sync runtime -> `test_exception_to_runfailure_matches_sync_resolver`
- [x] legacy `update` warning behavior is either preserved or intentionally retired with docs -> `test_async_resolver_legacy_update_warning_preserved`
- [x] async runtime runs two independent awaited handlers concurrently -> `test_async_resolver_runs_two_awaited_handlers_concurrently`
- [x] `_deps` is available during live execution but omitted from checkpoints -> `test_async_resolver_deps_available_in_handler`, `test_async_runtime_deps_live_but_omitted_from_checkpoint_payload`

---

## Phase 3 - Async Scheduler Core

Phase 3 status: completed. `AsyncWorkflowRuntime` now has an experimental native async scheduler path (`experimental_native_scheduler=True`) for linear/branch/fanout/route-next plus first-slice join-merge behavior. Checkpoint/resume and cancellation durability are covered for this slice via sync delegation and native persistence hooks. Contract file last verified green at `35 passed`.

- [x] implement `AsyncWorkflowRuntime`
- [ ] replace thread worker scheduling with task-based scheduling
- [x] preserve deterministic edge ordering by priority
- [x] preserve token ids, parent token ids, and branch masks
- [x] preserve token nesting semantics: first branch continues current token, later branches get child tokens
- [x] preserve fanout behavior
- [x] preserve join/barrier behavior
- [x] preserve explicit-join semantics: convergence without `wf_join` may execute once per token
- [x] preserve route-next behavior
- [x] preserve cancellation checkpoints
- [x] preserve terminal status artifacts
- [x] enforce max concurrent tasks
- [x] ensure task cancellation does not corrupt persisted run state
- [x] ensure parent run does not finish while required child tasks remain pending
- [x] define deterministic scheduler acceptance order for completed async tasks

Acceptance tests:

- [x] async linear workflow reaches terminal -> `test_async_runtime_native_scheduler_linear_success`
- [x] async branching workflow converges at join -> `test_async_runtime_native_scheduler_join_merge_runs_once`
- [x] async fanout respects `wf_fanout` and `wf_multiplicity` -> `test_async_runtime_native_scheduler_fanout_appends`, `test_async_runtime_native_scheduler_respects_many_multiplicity`
- [x] async convergence without explicit join can execute downstream once per arriving token -> `test_async_runtime_native_scheduler_without_join_executes_once_per_token`
- [x] async route-next matches sync route-next -> `test_async_runtime_native_scheduler_route_next_and_priority`
- [x] async cancellation persists cancelled terminal state -> `test_async_runtime_native_scheduler_cancellation_drains_inflight`, `test_async_runtime_native_scheduler_persists_cancelled_terminal`
- [x] async runtime does not leave orphan pending tasks on terminal exit -> `test_async_runtime_native_scheduler_cancellation_drains_inflight`
- [x] async completed-task race still applies state in deterministic acceptance order -> `test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order`

Phase 3 progress notes:

- Implemented:
  - Experimental native async scheduling loop using `asyncio` tasks and semaphore concurrency limits.
  - Deterministic acceptance order for completed tasks by local step sequence in the current first slice.
  - Next-edge selection with predicate/default/fanout plus `route-next` filtering, with priority-first ordering.
  - Native-path cancellation now drains in-flight tasks to avoid orphan task leakage on terminal exit.
  - Token nesting first slice: first routed branch keeps token id; later fanout branches mint child token ids.
  - Token spawn events are emitted (`type=token.spawn`) with parent/child token linkage for observability.
  - First-slice join merge support (`wf_join`/`join` op) with static incoming-edge quorum and single downstream continuation.
  - `wf_terminal` metadata now short-circuits routing in native async path.
  - Native `_rt_join` snapshot now carries `join_node_ids`, `join_outstanding`, `join_waiters`, `pending`, and `suspended`.
  - `wf_multiplicity="many"` now keeps multiple routed edges even when `wf_fanout` is false.
  - `resume_run(...)` now delegates to sync runtime resume path, and native `run(... _resume_*)` markers also delegate to sync runtime to preserve checkpoint resume semantics in this slice.
  - Native cancellation now writes cancelled terminal persistence through sync runtime persistence hook when available.

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

- [x] export `AsyncWorkflowRuntime` from `kogwistar.runtime`
- [ ] add server runner option for async runtime
- [ ] add conversation service injection point for runtime class selection
- [ ] ensure existing sync runtime remains default until parity is proven
- [ ] add compatibility shims for tests and demos
- [ ] document how callers choose sync versus async runtime
- [ ] document unsupported async first-slice features, if any

Acceptance tests:

- [x] package import exposes `AsyncWorkflowRuntime`
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
- [ ] maintain a sync-to-async test mapping rule: each sync runtime semantic test should have an async counterpart or be explicitly waived with rationale

Suggested initial tests:

- [x] `tests/runtime/test_async_runtime_contract.py`
- [ ] `tests/runtime/test_async_runtime_parity.py`
- [ ] `tests/runtime/test_async_runtime_replay.py`
- [ ] `tests/runtime/test_async_runtime_backend_durability.py`
- [ ] `tests/runtime/test_async_runtime_suspend_resume.py`
- [ ] `tests/runtime/test_async_runtime_nested.py`
- [ ] `tests/runtime/test_async_runtime_graph_side_effect_parity.py`

Sync/Async test-mapping policy:

- Default rule: every sync runtime semantic test should map to an async counterpart.
- Non-injective allowance: some sync tests may not have a 1:1 async mapping.
- For each waived mapping, record:
  - sync test id/path
  - waiver reason (why no direct async equivalent)
  - risk note (what semantic gap remains)
  - replacement guard (if covered by another async test)

---

## Acceptance Checklist

- [x] one workflow design model runs on both runtimes
- [x] sync and async resolvers share one semantic contract
- [x] `default_sync_ops == default_async_ops` is enforced by tests
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

Recent note (2026-04-22):

- Fixed lazy-export gap in `kogwistar.runtime.__init__` so `kogwistar.runtime.async_runtime` is resolvable as a submodule attribute (required by monkeypatch string paths in async contract tests).
