# Kogwistar Async Runtime Implementation Checklist

Companion ARD: `docs/kogwistar_async_runtime_ard.md`

Status: Active implementation.
Current snapshot:
- Phase 0: completed
- Phase 1: completed (first-slice delegation parity scope)
- Phase 2: completed
- Phase 3: completed (experimental native async scheduler path landed for linear/fanout/route-next/join/cancel/resume basics with bounded concurrency and deterministic acceptance order)
- Phase 4+: in progress (state/replay reducer unification started)
- Phase 7: in progress (runtime selection/integration wiring started)

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

Phase 3 status: completed. `AsyncWorkflowRuntime` now has an experimental native async scheduler path (`experimental_native_scheduler=True`) for linear/branch/fanout/route-next plus first-slice join-merge behavior. It is task-based and thread-free by default for step execution; any extra threading must be explicit inside resolver or handler code. Checkpoint/resume and cancellation durability are covered for this slice via sync delegation and native persistence hooks. Contract file last verified green at `40 passed`.

- [x] implement `AsyncWorkflowRuntime`
- [x] replace thread worker scheduling with task-based scheduling -> `test_async_resolver_runs_sync_handlers_inline_without_to_thread`, `test_async_runtime_native_scheduler_sync_handlers_run_inline_without_to_thread`
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

- [x] share state update application logic with `WorkflowRuntime`
- [x] verify async runtime uses the same merge semantics as `WorkflowRuntime.apply_state_update` -> `test_async_runtime_native_scheduler_uses_shared_state_merge_semantics`
- [x] verify async runtime matches `kogwistar.runtime.replay` reducer semantics -> `test_replay_state_reducer_matches_sync_runtime_merge_semantics`
- [x] keep checkpoint payload schema compatible
- [x] keep run and step sequence semantics compatible
- [x] checkpoint token ids and parent token ids
- [x] checkpoint join/barrier bookkeeping needed for resume
- [x] checkpoint pending/scheduled tokens and in-flight obligations
- [x] preserve `_deps` as non-checkpointed process-local data
- [x] ensure wall-clock timestamps are audit facts, not replay ordering inputs
- [x] ensure replay order is `seq` / `step_seq`, not `created_at_ms`
- [x] define graph side-effect normalization for sync/async comparisons
- [x] compare accepted node and edge side effects, not only final state

Acceptance tests:

- [x] sync and async runtimes produce identical final state for the same deterministic workflow -> `test_async_runtime_linear_terminal_status_equivalent_to_sync`, `test_async_runtime_branch_join_status_and_state_equivalent_to_sync`, `test_async_runtime_native_scheduler_uses_shared_state_merge_semantics`
- [x] replay of async checkpoints yields the same state as live async execution -> `test_replay_state_reducer_matches_sync_runtime_merge_semantics`
- [x] async checkpoint restore resumes from the same next pending node as sync restore -> `test_async_runtime_resume_run_delegates_to_sync_resume`, `test_async_runtime_run_with_resume_markers_delegates_to_sync_run`, `test_resume_run_failure_can_route_to_recovery_branch_async_backends`
- [x] changing `created_at_ms` does not change replay result -> `test_replay_ignores_created_at_ms_and_orders_by_step_seq`
- [x] checkpoint restore preserves token nesting and join counters -> `test_async_runtime_native_scheduler_token_nesting_and_spawn_events`, `test_async_runtime_native_scheduler_persists_pending_token_parent_links_on_cancel`, `test_async_runtime_native_scheduler_persists_rt_join_frontier_shape`
- [x] replay does not require `_deps` to be serialized -> `test_async_runtime_deps_live_but_omitted_from_checkpoint_payload`
- [x] side-by-side sync/async run produces the same normalized persisted node set -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] side-by-side sync/async run produces the same normalized persisted edge set -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] side-by-side sync/async run produces the same terminal status artifact -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`

Phase 4 progress note:

- Shared reducer now lives in `kogwistar.runtime.runtime.apply_state_update_inplace` and is reused by sync runtime, async runtime, and replay.
- Replay ordering is keyed by `step_seq`; `created_at_ms` stays audit-only.
- Side-by-side parity now checks normalized node, edge, and terminal artifacts for sync vs async runs.
- Phase 4 core is done; Phase 4A and Phase 4B acceptance coverage is in place.

---

## Phase 4A - Suspend, Resume, and Cancellation

- [x] preserve `RunSuspended` as a normal runtime result
- [x] persist suspended token id, node id, wait reason, and checkpoint state
- [x] resume from suspended token with client-provided result
- [x] ensure resume applies state update through shared merge semantics
- [x] ensure cancellation stops new scheduling after acceptance
- [x] define policy for already in-flight async tasks during cancellation
- [x] persist exactly one cancelled terminal artifact
- [x] ensure cancellation is idempotent
- [x] classify cooperative task cancellation markers as async diagnostics, not semantic workflow side effects

Acceptance tests:

- [x] async suspended run can resume to success
- [x] async suspended run can resume to failure when client result fails
- [x] async cancellation before scheduling persists cancelled state
- [x] async cancellation while tasks are in-flight leaves no corrupt checkpoint
- [x] repeated cancellation request is idempotent
- [x] async-only cancellation diagnostics do not affect normalized graph side-effect parity

---

## Phase 4B - Nested Workflow Semantics

- [x] support child workflow invocation from async runtime
- [x] apply child result to parent through shared merge semantics
- [x] preserve parent/child run ids and trace linkage
- [x] ensure parent waits for required child completion
- [x] define cancellation propagation between parent and child
- [x] define failure propagation from child to parent
- [x] avoid duplicate trace/event sinks for nested async runs

Acceptance tests:

- [x] async parent workflow invokes child workflow and merges child result
- [x] child failure produces deterministic parent failure
- [x] parent cancellation cancels child according to policy
- [x] nested async traces are linked and non-duplicated

Notes:

- Parent/child cancellation propagation now follows parent-cancel propagates into child runs. Current code covers suspend/resume, child invocation, failure propagation, cancellation propagation, and trace linkage.

---

## Phase 5 - Backend-Aware Durability

- [x] define async runtime write unit boundaries
- [x] preserve Postgres strong transaction behavior where the backend supports it -> `test_pg_transaction_rollback_power_out_simulation`, `test_async_pg_backend_transaction_rollback`, `test_async_pg_engine_uow_rolls_back_writes_together`
- [x] preserve Chroma event-first eventual repair behavior -> `test_phase3_chroma_replay_repairs_missing_vector_state`, `test_phase3b_chroma_replay_repair_overwrites_tampered_row`
- [x] preserve in-memory semantics without pretending crash recovery exists
- [x] share transaction policy decisions with sync runtime where possible -> `test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend`, `test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend`
- [x] make per-step commit behavior explicit -> `test_async_runtime_native_scheduler_uses_step_uow_boundary`, `test_async_postgres_uow_nested_transaction_joins_outer_scope`
- [x] ensure event log and index jobs are not duplicated during replay/rebuild -> `test_phase3b_replay_does_not_emit_new_entity_events_normal_or_repair`, `test_pgvector_replay_is_idempotent_no_new_events`
- [x] test mid-step failure behavior by backend family -> `test_pg_transaction_rollback_power_out_simulation`, `test_async_pg_backend_transaction_rollback`, `test_phase3_chroma_replay_repairs_missing_vector_state`
- [x] ensure unit-of-work state is task-local for concurrent async steps
- [x] ensure nested UoW in the same task joins the outer scope
- [x] ensure rollback in one async task does not roll back another task's committed scope -> `test_async_postgres_uow_rollback_in_one_task_does_not_touch_other_commit`

Acceptance tests:

- [x] Postgres-backed async step commit is atomic or rolls back cleanly -> `test_async_pg_backend_invariants`, `test_async_pg_backend_transaction_rollback`, `test_async_pg_engine_uow_rolls_back_writes_together`
- [x] Chroma-backed async step failure is repairable from event truth -> `test_phase3_chroma_replay_repairs_missing_vector_state`, `test_phase3b_chroma_replay_repair_overwrites_tampered_row`
- [x] in-memory backend documents volatile recovery limits -> `test_in_memory_meta_new_store_has_no_recovery_memory`
- [x] async replay does not append new authoritative history -> `test_replay_to_is_read_only_and_does_not_append_new_history`
- [x] async runtime respects backend unit-of-work boundaries -> `test_async_runtime_native_scheduler_uses_step_uow_boundary`, `test_async_pg_engine_uow_rolls_back_writes_together`
- [x] concurrent async UoWs are isolated by task context -> `test_async_postgres_uow_task_local_context_isolated_across_tasks`
- [x] nested async UoW joins inside one task -> `test_async_postgres_uow_nested_transaction_joins_outer_scope`

---

## Phase 6 - Trace, Observability, and Events

- [x] reuse `EventEmitter` semantics or provide async equivalent
- [x] preserve workflow trace node and edge shape -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] preserve `WorkflowRunNode`, `WorkflowStepExecNode`, and checkpoint artifacts -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`, `test_replay_to_is_read_only_and_does_not_append_new_history`
- [x] preserve trace fast-path behavior where safe -> `test_runtime_trace_writes_disable_eager_index_reconcile_for_in_memory_backend`, `test_async_runtime_trace_fast_path_configuration_matches_sync_runtime`
- [x] ensure nested async runtimes do not duplicate trace sinks -> `test_async_runtime_native_scheduler_nested_invocation_reuses_trace_emitter`
- [x] ensure task durations and status reporting are best-effort and non-fatal -> `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`
- [x] add structured async task lifecycle events -> `test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`
- [x] expose scheduled, started, completed, failed, suspended, and cancelled task states -> `test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`, `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`, `test_async_runtime_native_scheduler_cancellation_drains_inflight`, `test_async_runtime_native_scheduler_persists_cancelled_terminal`, `test_async_runtime_suspend_and_resume_roundtrip`
- [x] ensure trace write failures follow explicit policy -> `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`
- [x] separate durable semantic runtime events from async-only diagnostic telemetry -> `test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`, `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`

Acceptance tests:

- [x] async runtime writes trace artifacts with the same metadata fields as sync runtime -> `test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`, `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] async nested workflow invocation reuses or safely isolates trace emitter -> `test_async_runtime_native_scheduler_nested_invocation_reuses_trace_emitter`
- [x] trace write failure does not corrupt workflow state unless policy says fatal -> `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`
- [x] async runtime exposes enough events for server progress APIs -> `test_chat_rest_events_poll_sees_live_updates_for_async_backends`, `test_mcp_run_events_sees_live_updates_for_async_backends`, `test_workflow_runtime_sse_cancel_after_sleep_ticks_for_async_backends`
- [x] async diagnostic telemetry can be ignored without changing replay or side-effect parity -> `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`, `test_async_runtime_side_by_side_node_edge_and_terminal_parity`

---

## Phase 7 - API and Integration

- [x] export `AsyncWorkflowRuntime` from `kogwistar.runtime`
- [x] add server runner option for async runtime
- [x] add conversation service injection point for runtime class selection
- [x] ensure existing sync runtime remains default until parity is proven
- [x] add compatibility shims for tests and demos
- [x] document how callers choose sync versus async runtime
- [x] document unsupported async first-slice features, if any

Acceptance tests:

- [x] package import exposes `AsyncWorkflowRuntime`
- [x] server/runtime path can run one workflow with async runtime when explicitly selected -> `test_workflow_run_submit_accepts_runtime_kind_and_defaults_to_sync`
- [x] conversation orchestrator can inject async runtime without changing workflow design -> `test_default_runtime_runner_uses_async_or_sync_runtime_by_kind`
- [x] sync runtime entrypoints continue to work unchanged -> `test_chat_service_reexports_public_symbols`, `test_chat_run_service_async_wrappers_delegate_through_to_thread`, `test_async_sse_routes_use_async_service_methods`
- [x] sync runtime remains default when no async runtime option is supplied -> `test_workflow_run_submit_accepts_runtime_kind_and_defaults_to_sync`, `test_default_runtime_runner_uses_async_or_sync_runtime_by_kind`

Notes:

- Callers can pick runtime by passing `runtime_kind="async"` to workflow submission.
- Service owners can set `default_runtime_kind="async"` when they want async as local default.
- Existing `runtime_runner` injection remains supported for tests and demos.
- Async selection is opt-in; sync stays default until later parity gates say otherwise.

---

## Phase 8 - Parity Matrix

- [x] accepted graph node side-effect parity -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] accepted graph edge side-effect parity -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] linear workflow parity -> `test_async_runtime_native_scheduler_linear_success`
- [x] branching parity -> `test_async_runtime_branch_join_status_and_state_equivalent_to_sync`, `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
- [x] join/barrier parity -> `test_async_runtime_native_scheduler_join_merge_runs_once`, `test_async_runtime_branch_join_status_and_state_equivalent_to_sync`
- [x] fanout parity -> `test_async_runtime_native_scheduler_fanout_appends`, `test_async_runtime_native_scheduler_respects_many_multiplicity`
- [x] route-next parity -> `test_async_runtime_native_scheduler_route_next_and_priority`
- [x] suspend/resume parity -> `test_async_runtime_suspend_and_resume_roundtrip`, `test_async_runtime_resume_run_delegates_to_sync_resume`, `test_async_runtime_run_with_resume_markers_delegates_to_sync_run`
- [x] cancellation parity -> `test_async_runtime_native_scheduler_cancellation_drains_inflight`, `test_async_runtime_native_scheduler_persists_cancelled_terminal`, `test_async_runtime_parent_cancellation_propagates_to_child`
- [x] nested workflow parity -> `test_async_runtime_nested_workflow_invocation_matches_sync`, `test_async_runtime_nested_workflow_child_failure_fails_parent`
- [x] sandboxed op parity or documented gap -> documented first-slice gap in `Phase 2` notes; no async sandboxed-op divergence introduced
- [x] checkpoint replay parity -> `test_replay_state_reducer_matches_sync_runtime_merge_semantics`, `test_replay_to_is_read_only_and_does_not_append_new_history`
- [x] backend durability parity -> `test_async_pg_backend_transaction_rollback`, `test_async_pg_engine_uow_rolls_back_writes_together`, `test_phase3_chroma_replay_repairs_missing_vector_state`, `test_in_memory_meta_new_store_has_no_recovery_memory`
- [x] trace artifact parity -> `test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`, `test_async_runtime_native_scheduler_nested_invocation_reuses_trace_emitter`, `test_async_runtime_native_scheduler_trace_emitter_failure_is_best_effort`
- [x] server progress event parity -> `test_chat_rest_events_poll_sees_live_updates_for_async_backends`, `test_mcp_run_events_sees_live_updates_for_async_backends`, `test_workflow_runtime_sse_cancel_after_sleep_ticks_for_async_backends`
- [x] LangGraph semantics-mode reducer parity -> `test_replay_state_reducer_matches_sync_runtime_merge_semantics`, `test_async_runtime_native_scheduler_uses_shared_state_merge_semantics`
- [x] visual-mode non-parity documented

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
- Each async acceptance item should name its corresponding sync test(s), or explicitly mark a documented gap.
- Prefer a test-local docstring on each async semantic test that names its sync mirror.
- Add a meta-test that reads the first docstring line for each mapped sync/async case and fails on missing mirror text.
- Add a naming-convention test so mapped sync cases stay `test_...` and mapped async cases stay `test_async_...`.
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
- [x] async runtime state updates match sync runtime state updates -> `test_async_runtime_native_scheduler_uses_shared_state_merge_semantics`, `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
  - sync counterpart(s): `tests/workflow/test_workflow_native_update.py`, `tests/workflow/test_workflow_join.py`
- [x] async runtime replay matches live execution -> `test_replay_state_reducer_matches_sync_runtime_merge_semantics`, `test_replay_to_is_read_only_and_does_not_append_new_history`
  - sync counterpart(s): `tests/workflow/test_workflow_suspend_resume.py`, `tests/runtime/test_workflow_invocation_and_route_next.py`
- [x] backend durability semantics are explicit by backend family -> `test_async_pg_backend_transaction_rollback`, `test_async_pg_engine_uow_rolls_back_writes_together`, `test_phase3_chroma_replay_repairs_missing_vector_state`, `test_in_memory_meta_new_store_has_no_recovery_memory`
  - sync counterpart(s): `tests/primitives/test_document_rollback.py`, `tests/primitives/test_edge_endpoints_rollback.py`, `tests/core/test_in_memory_meta_store.py`
- [x] trace artifacts and progress events remain compatible -> `test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`, `test_chat_rest_events_poll_sees_live_updates_for_async_backends`, `test_mcp_run_events_sees_live_updates_for_async_backends`, `test_workflow_runtime_sse_cancel_after_sleep_ticks_for_async_backends`
  - sync counterpart(s): `tests/workflow/test_tracing_e2e.py`, `tests/runtime/test_trace_sink_parallel_nested_minimal.py`
- [x] wall-clock timestamps are audit facts only -> `test_replay_ignores_created_at_ms_and_orders_by_step_seq`
  - sync counterpart(s): `tests/runtime/test_checkpoint_resume_contract.py`, `tests/workflow/test_save_load_progress.py`
- [x] token nesting, explicit join, and checkpoint resume semantics match sync runtime -> `test_async_runtime_native_scheduler_token_nesting_and_spawn_events`, `test_async_runtime_native_scheduler_join_merge_runs_once`, `test_async_runtime_resume_run_delegates_to_sync_resume`, `test_async_runtime_run_with_resume_markers_delegates_to_sync_run`, `test_async_runtime_native_scheduler_persists_rt_join_frontier_shape`
  - sync counterpart(s): `tests/workflow/test_workflow_join.py`, `tests/runtime/test_workflow_suspend_resume.py`, `tests/runtime/test_resume_wait_reasons.py`
- [x] suspend, resume, cancellation, and nested workflow behavior match sync runtime or are explicitly documented as first-slice gaps -> `test_async_runtime_suspend_and_resume_roundtrip`, `test_async_runtime_parent_cancellation_propagates_to_child`, `test_async_runtime_nested_workflow_child_failure_fails_parent`, `test_async_runtime_native_scheduler_cancellation_drains_inflight`, `test_async_runtime_native_scheduler_persists_cancelled_terminal`
  - sync counterpart(s): `tests/runtime/test_workflow_suspend_resume.py`, `tests/runtime/test_workflow_cancel_event_sourced.py`, `tests/runtime/test_trace_sink_parallel_nested_minimal.py`
- [x] existing workflows have identical normalized graph node/edge side effects under sync and async runtimes -> `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
  - sync counterpart(s): `tests/workflow/test_workflow_join.py`, `tests/workflow/test_workflow_native_update.py`
- [x] async-only cooperative cancellation diagnostics are excluded from semantic side-effect parity -> `test_async_runtime_native_scheduler_cancellation_drains_inflight`, `test_async_runtime_native_scheduler_persists_cancelled_terminal`, `test_async_runtime_side_by_side_node_edge_and_terminal_parity`
  - sync counterpart(s): `tests/runtime/test_workflow_cancel_event_sourced.py`, `tests/workflow/test_tracing_e2e.py`
- [x] async task-local UoW isolation is tested
  - sync counterpart(s): `tests/core/test_in_memory_meta_store.py`, `tests/pg_sql/test_async_pgvector_backend.py`
- [x] LangGraph parity claims are limited to semantics mode -> documented in Phase 8 notes
  - sync counterpart(s): `tests/workflow/test_langgraph_converter.py`, `tests/workflow/test_langgraph_converter_parallel_integration.py`, `tests/workflow/test_langgraph_blob_state.py`
- [x] sync runtime remains stable while async runtime lands incrementally -> `test_workflow_run_submit_accepts_runtime_kind_and_defaults_to_sync`, `test_default_runtime_runner_uses_async_or_sync_runtime_by_kind`, `test_chat_service_split_smoke.py`
  - sync counterpart(s): `tests/server/test_chat_server_api.py`, `tests/server/test_chat_service_split_smoke.py`

Recent note (2026-04-22):

- Fixed lazy-export gap in `kogwistar.runtime.__init__` so `kogwistar.runtime.async_runtime` is resolvable as a submodule attribute (required by monkeypatch string paths in async contract tests).
