# Kogwistar Async Runtime Bijection Refactor Plan

Companion checklist: `docs/kogwistar_async_runtime_impl_checklist.md`

Purpose: make sync/async runtime test bijection refactors mechanical and case-by-case. Do not infer from file name alone.

Status:

- Active refactor plan.
- Current bijection subset is guarded by `tests/runtime/test_runtime_bijection_contract.py`.
- This file is the human triage ledger; guard tests are the executable enforcement.

## Rules

Use exact words consistently:

- `Moved from path::test_name`: exact test case moved into a bijection file. Source test must be deleted in same patch.
- `Refactored from path::test_name`: only bijection slice extracted. Source test must remain because it owns broader semantics.
- `New sync mirror ...` / `New async mirror ...`: no source test existed at the right granularity.
- `Waived`: no 1:1 bijection mapping yet; always record concrete why and name replacement guard.

Never:

- Call a copied test `Moved from`.
- Delete a `Refactored from` source.
- Point sync-side provenance at async-runtime source.
- Move broad integration/backend/replay/server tests wholesale into bijection.
- Add only one side of a bijection suffix.
- Write `waived` without concrete rationale and replacement guard.

Done means:

- Matching suffix exists in both bijection files.
- First docstring line names mirror case.
- Provenance line is `Moved from`, `Refactored from`, or `New ... mirror`.
- `Moved from` source no longer exists.
- `Refactored from` source still exists and has `Source retained: ...`.
- Guard passes:

```powershell
.venv\Scripts\python.exe -m pytest tests\runtime\test_runtime_bijection_contract.py -q
```

## Decision Table

| Case shape | Action | Source action | Docstring |
| --- | --- | --- | --- |
| Existing sync test is exactly one bijection semantic case | Move sync into `test_sync_runtime_bijection_contract.py`; implement async mirror if missing | Delete old sync source | Sync `Moved from ...`; async `New async mirror ...` or `Moved from ...` |
| Existing async test is exactly one bijection semantic case | Move async into `test_async_runtime_bijection_contract.py`; implement sync mirror if missing | Delete old async source | Async `Moved from ...`; sync `New sync mirror ...` or `Moved from ...` |
| Both exact sync and async tests exist | Move both into bijection files | Delete both old sources | Both `Moved from ...` |
| Source owns broader semantics | Extract bijection assertion only | Keep source | `Refactored from ...` plus `Source retained: ...` |
| No source at right granularity | Create bijection pair | No deletion | `New sync mirror ...` and/or `New async mirror ...` |
| Many-to-one, slow, backend-parametrized, or nondeterministic | Do not move yet | Keep source | Waiver entry |

## Current Bijection Subset

These cases are already done in `test_sync_runtime_bijection_contract.py` and `test_async_runtime_bijection_contract.py`.

- [x] `default_ops_equal`: `test_sync_runtime_default_ops_equal` / `test_async_runtime_default_ops_equal`
- [x] `state_merge_semantics`: `test_sync_runtime_state_merge_semantics` / `test_async_runtime_state_merge_semantics`
- [x] `step_context_and_result_contract`: `test_sync_runtime_step_context_and_result_contract` / `test_async_runtime_step_context_and_result_contract`
- [x] `preserves_nested_ops_and_state_schema_in_adapter`: `test_sync_runtime_preserves_nested_ops_and_state_schema_in_adapter` / `test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter`
- [x] `trace_fast_path_configuration`: `test_sync_runtime_trace_fast_path_configuration` / `test_async_runtime_trace_fast_path_configuration`
- [x] `auto_transaction_mode_defaults_to_none_for_non_pg_backend`: `test_sync_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend` / `test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend`
- [x] `auto_transaction_mode_uses_step_for_pg_backend`: `test_sync_runtime_auto_transaction_mode_uses_step_for_pg_backend` / `test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend`
- [x] `missing_op_behavior`: `test_sync_runtime_missing_op_behavior` / `test_async_runtime_missing_op_behavior`
- [x] `exception_to_runfailure`: `test_sync_runtime_exception_to_runfailure` / `test_async_runtime_exception_to_runfailure`
- [x] `async_handler_callability`: `test_sync_runtime_async_handler_callability` / `test_async_runtime_async_handler_callability`
- [x] `sync_handler_callability`: `test_sync_runtime_sync_handler_callability` / `test_async_runtime_sync_handler_callability`
- [x] `registered_op_appears_in_op_set`: `test_sync_runtime_registered_op_appears_in_op_set` / `test_async_runtime_registered_op_appears_in_op_set`
- [x] `preserves_sandboxed_behavior`: `test_sync_runtime_preserves_sandboxed_behavior` / `test_async_runtime_preserves_sandboxed_behavior`
- [x] `legacy_update_warning_preserved`: `test_sync_runtime_legacy_update_warning_preserved` / `test_async_runtime_legacy_update_warning_preserved`
- [x] `step_context_send_lane_message_delegates`: `test_sync_runtime_step_context_send_lane_message_delegates_to_sender` / `test_async_runtime_step_context_send_lane_message_delegates_to_sender`
- [x] `step_context_send_lane_message_requires_sender`: `test_sync_runtime_step_context_send_lane_message_requires_sender` / `test_async_runtime_step_context_send_lane_message_requires_sender`
- [x] `step_context_emit_lane_message_event_delegates`: `test_sync_runtime_step_context_emit_lane_message_event_delegates_to_sink` / `test_async_runtime_step_context_emit_lane_message_event_delegates_to_sink`
- [x] `step_context_emit_lane_message_event_requires_sink`: `test_sync_runtime_step_context_emit_lane_message_event_requires_sink` / `test_async_runtime_step_context_emit_lane_message_event_requires_sink`
- [x] `state_schema_metadata_available`: `test_sync_runtime_state_schema_metadata_available` / `test_async_runtime_state_schema_metadata_available`
- [x] `deps_available_in_handler`: `test_sync_runtime_deps_available_in_handler` / `test_async_runtime_deps_available_in_handler`
- [x] `deps_live_not_checkpointed`: `test_sync_runtime_deps_live_but_omitted_from_checkpoint_payload` / `test_async_runtime_deps_live_but_omitted_from_checkpoint_payload`
- [x] `linear_terminal_status_equivalent`: `test_sync_runtime_linear_terminal_status_equivalent` / `test_async_runtime_linear_terminal_status_equivalent`
- [x] `branch_join_status_and_state_equivalent`: `test_sync_runtime_branch_join_status_and_state_equivalent` / `test_async_runtime_branch_join_status_and_state_equivalent`
- [x] `route_next_alias_can_fan_out_multiple_branches`: `test_sync_runtime_route_next_alias_can_fan_out_multiple_branches` / `test_async_runtime_route_next_alias_can_fan_out_multiple_branches`
- [x] `native_update_schema_applies_known_and_falls_back_unknown`: `test_sync_runtime_native_update_schema_applies_known_and_falls_back_unknown` / `test_async_runtime_native_update_schema_applies_known_and_falls_back_unknown`
- [x] `non_sandboxed_op_does_not_prepare_sandbox`: `test_sync_runtime_non_sandboxed_op_does_not_prepare_sandbox` / `test_async_runtime_non_sandboxed_op_does_not_prepare_sandbox`
- [x] `runtime_exported`: `test_sync_runtime_exported` / `test_async_runtime_exported`
- [x] `adapter_forwards_close_sandbox_run`: `test_sync_runtime_adapter_forwards_close_sandbox_run` / `test_async_runtime_adapter_forwards_close_sandbox_run`

Move next:

## Candidate Case Backlog

This is case-by-case, not file-by-file. Classification may change after reading each test body.

## Useful Smoke Gaps To Add Next

These are high-signal, low-cost smoke tests that are not yet in the current bijection subset. Prefer these before moving broad workflow tests.

| Priority | Missing smoke | Why useful | Proposed action |
| --- | --- | --- | --- |
| P0 | Resolver state-schema metadata smoke | Catches sync/async resolver metadata drift without running workflow graph. | Done; keep bijection pair. |
| P0 | Live `_deps` handler smoke | Catches DI/process-local state contract before checkpoint complexity. | Done; keep bijection pair. |
| P0 | StepContext lane-message delegation smoke | `StepContext` is shared runtime surface; cheap and not backend-bound. | Done; keep bijection pair. |
| P0 | StepContext lane-message missing-sender smoke | Guards failure shape for shared context helpers. | Done; keep bijection pair. |
| P0 | StepContext lane-event delegation smoke | Shared progress/event helper; useful for server progress compatibility. | Done; keep bijection pair. |
| P0 | StepContext lane-event missing-sink smoke | Guards failure shape for shared event helper. | Done; keep bijection pair. |
| P1 | Sync runtime rejects awaitable handler smoke | Prevents accidentally making sync runtime async-capable by hidden await. | Keep as sync contract / waiver; do not force into bijection. |
| P1 | Async adapter `close_sandbox_run` forwarding smoke | Protects sandbox cleanup path when async facade delegates to sync runtime. | Done; keep bijection pair. |
| P1 | Non-sandboxed op does not prepare sandbox | Small resolver policy smoke, now bijection-ready. | Done; keep as current bijection subset. |
| P1 | Same workflow graph model accepted by both runtimes | Guards constructor/graph-model contract without real workflow execution. | Done; keep bijection pair. |
| P2 | Runtime public export/import smoke | Useful package boundary check but partly async-only. | Done; keep bijection pair. |
| P2 | Default resolver execution smoke | Valuable end-to-end smoke but too broad for bijection. | Keep `test_workflow_runtime_uses_default_resolver` as integration/CI-full; do not move wholesale. |

### Move async + implement sync mirror

These look like small async-side contract cases. Inspect before moving.

- [x] `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_sync_handlers_inline_without_to_thread` -> `sync_handlers_inline_without_to_thread` | waived; async-only thread-free scheduler invariant
- [x] `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_two_awaited_handlers_concurrently` -> `awaited_handlers_concurrently` | waived; async-only awaited concurrency invariant, because sync runtime has no concurrent awaited-task analogue
- [x] `tests/runtime/test_async_runtime_contract.py::test_async_resolver_state_schema_metadata_available` -> `state_schema_metadata_available` | moved into bijection pair
- [x] `tests/runtime/test_async_runtime_contract.py::test_async_resolver_deps_available_in_handler` -> `deps_available_in_handler` | moved into bijection pair

### Move sync + implement async mirror

These are sync-side baseline cases likely worth bijection triage.

- [x] `tests/runtime/test_resume_wait_reasons.py::test_wait_reason_can_suspend_and_resume` -> `wait_reason_suspend_resume` | reclassified to parity bridge; source retained for resume wait-reason integration
- [x] `tests/runtime/test_trace_sink_parallel_nested_minimal.py::test_trace_sink_parallel_and_nested_minimal_sync` -> `trace_sink_parallel_nested_minimal` | reclassified to parity bridge; source retained for trace sink integration
- [x] `tests/runtime/test_workflow_terminal_status.py::test_runtime_persists_completed_terminal_for_leaf_node` -> `persists_completed_terminal_for_leaf_node` | reclassified to backend acceptance/backend-family matrix; source retained for backend-param terminal persistence
- [x] `tests/runtime/test_workflow_terminal_status.py::test_runtime_persists_failed_terminal_for_leaf_node` -> `persists_failed_terminal_for_leaf_node` | reclassified to backend acceptance/backend-family matrix; source retained and async backend counterpart added
  - async backend counterparts now pass for both `async-chroma` and `async-pg`
- [x] `tests/runtime/test_checkpoint_resume_contract.py::test_load_checkpoint_accepts_current_schema` -> `load_checkpoint_accepts_current_schema` | waived, because this is schema-version acceptance guard for checkpoint loader, not sync-vs-async runtime semantic split
- [x] `tests/runtime/test_checkpoint_resume_contract.py::test_load_checkpoint_rejects_future_schema` -> `load_checkpoint_rejects_future_schema` | waived, because this is schema-version rejection guard for checkpoint loader, not sync-vs-async runtime semantic split
- [x] `tests/runtime/test_checkpoint_resume_contract.py::test_replay_rejects_future_schema` -> `replay_rejects_future_schema` | waived, because this is replay schema safety guard, not runtime-kind semantic split
- [x] `tests/runtime/test_checkpoint_resume_contract.py::test_replay_ignores_created_at_ms_and_orders_by_step_seq` -> `replay_orders_by_step_seq` | reclassified to acceptance mapping retained; source retained and mapped to async frontier-shape acceptance
- [x] `tests/runtime/test_checkpoint_resume_contract.py::test_replay_to_is_read_only_and_does_not_append_new_history` -> `replay_read_only_no_new_history` | reclassified to acceptance mapping retained; source retained and mapped to async resume delegation acceptance
- [x] `tests/runtime/test_sandbox.py::test_mapping_resolver_does_not_prepare_sandbox_for_non_sandboxed_op` -> `non_sandboxed_op_does_not_prepare_sandbox` | moved into bijection pair
- [x] `tests/runtime/test_step_context_lane_message.py::test_step_context_send_lane_message_delegates_to_sender` -> `step_context_send_lane_message_delegates` | moved into bijection pair
- [x] `tests/runtime/test_step_context_lane_message.py::test_step_context_send_lane_message_requires_sender` -> `step_context_send_lane_message_requires_sender` | moved into bijection pair
- [x] `tests/runtime/test_step_context_lane_message_events.py::test_step_context_emit_lane_message_event_delegates_to_sink` -> `step_context_emit_lane_message_event_delegates` | moved into bijection pair
- [x] `tests/runtime/test_step_context_lane_message_events.py::test_step_context_emit_lane_message_event_requires_sink` -> `step_context_emit_lane_message_event_requires_sink` | moved into bijection pair
- [x] `tests/workflow/test_workflow_join.py::test_join_barrier_waits_for_all_arrivals` -> `join_barrier_waits_for_all_arrivals` | reclassified to parity bridge; source retained for thread-worker sync join integration
- [x] `tests/workflow/test_workflow_join.py::test_join_does_not_wait_for_branch_that_can_no_longer_reach_it` -> `join_unreachable_branch_not_required` | reclassified to acceptance/parity backlog; source retained for reachability semantics
- [x] `tests/workflow/test_save_load_progress.py::test_runtime_checkpoint_load_and_replay` -> `checkpoint_load_and_replay` | reclassified to parity-bridge backlog; source retained for save/load progress
- [x] `tests/workflow/test_save_load_progress.py::test_runtime_resume_from_checkpoint` -> `resume_from_checkpoint` | reclassified to parity-bridge backlog; source retained for save/load progress
- [x] `tests/workflow/test_save_load_progress.py::test_runtime_resume_from_checkpoint_frontier` -> `resume_from_checkpoint_frontier` | reclassified to parity-bridge backlog; source retained for frontier semantics

### Keep as acceptance or integration tests, not bijection subset

These are intentionally broader than one bijection semantic case.

| Source case | Reason |
| --- | --- |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_sync_handlers_run_inline_without_to_thread` | Async-only thread-free scheduler invariant. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_linear_success` | Native scheduler end-to-end smoke; keep as acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_fanout_appends` | Native scheduler fanout acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_cancellation_drains_inflight` | Async cancellation acceptance and diagnostics. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_route_next_and_priority` | Native scheduler route-next acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_token_nesting_and_spawn_events` | Token nesting plus events; too broad for bijection. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_join_merge_runs_once` | Join acceptance; source for smaller future bijection extraction. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_without_join_executes_once_per_token` | Explicit no-join behavior; acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_respects_many_multiplicity` | Native routing multiplicity acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_persists_rt_join_frontier_shape` | Checkpoint shape acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_persists_pending_token_parent_links_on_cancel` | Cancellation checkpoint acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_side_by_side_node_edge_and_terminal_parity` | Main side-by-side semantic parity; keep whole. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_suspend_and_resume_roundtrip` | Suspend/resume acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_nested_workflow_invocation_matches_sync` | Nested workflow acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_nested_workflow_child_failure_fails_parent` | Nested failure acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_parent_cancellation_propagates_to_child` | Parent-child cancellation acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_cancel_idempotent_terminal_persistence` | Cancellation persistence acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_resume_run_delegates_to_sync_resume` | First-slice delegation acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_run_with_resume_markers_delegates_to_sync_run` | First-slice delegation acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_persists_cancelled_terminal` | Cancellation terminal acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_enforces_max_concurrent_tasks` | Async scheduler concurrency invariant. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order` | Async deterministic acceptance-order invariant. |
| `tests/runtime/test_workflow_cancel_event_sourced.py::test_runtime_event_sourced_cancel_reconciles_and_replay_is_stable` | Event-sourced cancel and replay integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_suspend_and_resume` | Broad sync suspend/resume integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_suspend_and_resume_branching` | Broad sync branching suspend/resume integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_does_not_route_to_terminal` | Failure routing integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_waits_for_inflight_branch_drain` | Sync thread-worker branch drain behavior. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_then_downstream_before_join_is_routed_when_handled` | Failure recovery routing integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_then_downstream_before_join_is_skipped_when_unhandled` | Failure recovery routing integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_allows_existing_join_to_finish_but_blocks_downstream` | Failure/join integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_can_route_to_recovery_branch` | Recovery branch integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_workflow_failure_does_not_take_default_recovery_edge` | Recovery edge selection integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_resume_run_failure_can_route_to_recovery_branch` | Resume failure routing integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_resume_run_can_resuspend_same_token_with_updated_payload` | Resume/resuspend integration. |
| `tests/runtime/test_workflow_suspend_resume.py::test_sandbox_recoverable_error_can_suspend_then_resume_success` | Sandbox + suspend/resume integration. |
| `tests/workflow/test_tracing_e2e.py::test_tracing_routing_decision_end_to_end` | Trace E2E integration. |
| `tests/workflow/test_tracing_e2e.py::test_tracing_join_events_end_to_end` | Trace E2E integration. |
| `tests/server/test_chat_server_api.py::test_workflow_run_submit_accepts_runtime_kind_and_defaults_to_sync` | Server runtime selection integration. |
| `tests/server/test_chat_server_api.py::test_default_runtime_runner_uses_async_or_sync_runtime_by_kind` | Server runtime runner integration. |
| `tests/server/test_chat_server_async_events.py::test_chat_rest_events_poll_sees_live_updates_for_async_backends` | Server progress event integration. |

## Backend E2E Smoke Matrix

Goal: pin the runtime slice across all backend families with one small end-to-end smoke per family. Keep these outside bijection unless a tiny contract can be extracted.

- [x] `memory` backend E2E smoke on sync runtime
- [x] `memory` backend E2E smoke on async runtime
- [x] `chroma` backend E2E smoke on sync runtime
- [x] `chroma` backend E2E smoke on async runtime
- [x] `pg` backend E2E smoke on sync runtime
- [x] `pg` backend E2E smoke on async runtime

Suggested invariant for each row:

- one tiny workflow graph
- one insert / one step / one trace or checkpoint roundtrip
- assert same visible terminal state shape
- assert backend-specific durability mode only where relevant

## Parity Bridge Tests

These are not 1:1 bijection cases. One test should exercise sync, then async, and assert the same visible invariant. Keep them in a third contract file family, e.g. `tests/runtime/test_runtime_parity_bridge_contract.py`.

- [x] `nested_workflow_synthesized_design_is_persisted_and_used`
- [x] `nested_workflow_child_failure_fails_parent`
- [x] `workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown`
- [x] `runtime_checkpoint_load_and_replay`
- [x] `runtime_resume_from_checkpoint`
- [x] `runtime_resume_from_checkpoint_frontier`
- [x] `workflow_suspend_and_resume_roundtrip`
- [x] `wait_reason_suspend_resume`
- [x] `workflow_suspend_and_resume_branching_roundtrip`
- [x] `parent_cancellation_propagates_to_child`
- [x] `route_next_shared_semantics`
- [x] `join_barrier_waits_for_all_arrivals`
- [x] `join_unreachable_branch_not_required`
- [x] `trace_sink_parallel_nested_minimal`
- [x] `runtime_trace_events_metadata_shape`
- [x] `side_by_side_node_edge_and_terminal_parity`

Rule:

- If a test is naturally "run sync, then run async, compare", do not force it into bijection files.
- Keep its source in a parity bridge file, with docstring naming both runtimes and the shared invariant.

Selection rule:

- Same workflow design can run under both runtimes.
- Same assertion target is visible at API/result/checkpoint/trace level.
- Difference is scheduler path, not business semantics.
- Test can compare normalized outputs and ignore async-only diagnostics.

Bridge shape:

- build one workflow graph
- run `WorkflowRuntime`
- run `AsyncWorkflowRuntime`
- normalize artifact/result/checkpoint/trace payload
- compare only semantic fields

Good normalization targets:

- `status`
- normalized `final_state`
- persisted child design presence/count
- checkpoint frontier shape
- terminal node/edge multiset
- trace event kind/order where order is semantic

Exclude from bridge equality:

- wall-clock timestamps
- async-only cancellation diagnostics
- thread/task scheduling incidental order
- backend-specific persistence ids

### Parity-Bridge Inventory

These are strongest current candidates, with source and invariant.

| Candidate suffix | Current source(s) | Shared invariant |
| --- | --- | --- |
| `nested_workflow_synthesized_design_is_persisted_and_used` | `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_synthesized_design_is_persisted_and_used`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_nested_workflow_invocation_matches_sync` | synthesized child workflow design persists and child result is reused identically |
| `nested_workflow_child_failure_fails_parent` | `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_failure_short_circuits_parent_routing`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_nested_workflow_child_failure_fails_parent` | child failure short-circuits parent and normalized final state matches |
| `workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown` | `tests/workflow/test_workflow_native_update.py::test_workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_uses_shared_state_merge_semantics` | known schema merges and unknown keys fall back compatibly |
| `runtime_checkpoint_load_and_replay` | `tests/workflow/test_save_load_progress.py::test_runtime_checkpoint_load_and_replay`, `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_checkpoint_load_and_replay` | checkpoint payload and replayed state match after normalization |
| `runtime_resume_from_checkpoint` | `tests/workflow/test_save_load_progress.py::test_runtime_resume_from_checkpoint`, `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_resume_from_checkpoint` | resumed run skips already-completed work and lands same final state |
| `runtime_resume_from_checkpoint_frontier` | `tests/workflow/test_save_load_progress.py::test_runtime_resume_from_checkpoint_frontier`, `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_resume_from_checkpoint_frontier` | resumed frontier shape and terminal result match |
| `workflow_suspend_and_resume_roundtrip` | `tests/runtime/test_workflow_suspend_resume.py::test_workflow_suspend_and_resume`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_suspend_and_resume_roundtrip` | suspend checkpoint then resume yields same normalized status/final_state |
| `wait_reason_suspend_resume` | `tests/runtime/test_resume_wait_reasons.py::test_wait_reason_can_suspend_and_resume`, `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_wait_reason_suspend_resume` | wait_reason, suspended token shape, and resumed terminal state normalize equal |
| `workflow_suspend_and_resume_branching_roundtrip` | `tests/runtime/test_workflow_suspend_resume.py::test_workflow_suspend_and_resume_branching`, async equivalent to add or extract | suspended branch/join frontier resumes to same semantic result |
| `parent_cancellation_propagates_to_child` | `tests/runtime/test_async_runtime_contract.py::test_async_runtime_parent_cancellation_propagates_to_child` plus sync bridge source to extract | parent cancel reaches child and final cancelled state normalizes equal |
| `route_next_shared_semantics` | `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_route_next_shared_semantics` | sync rich `_route_next` and async native scheduler choose same routes for explicit alias, fanout, and default fallback |
| `join_barrier_waits_for_all_arrivals` | `tests/workflow/test_workflow_join.py::test_join_barrier_waits_for_all_arrivals`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_join_merge_runs_once`, `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_join_barrier_waits_for_all_arrivals` | join releases once after all required arrivals |
| `join_unreachable_branch_not_required` | `tests/workflow/test_workflow_join.py::test_join_does_not_wait_for_branch_that_can_no_longer_reach_it`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_without_join_executes_once_per_token` | join does not wait on branch outside reachable join region |
| `trace_sink_parallel_nested_minimal` | `tests/runtime/test_trace_sink_parallel_nested_minimal.py::test_trace_sink_parallel_and_nested_minimal_sync`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata`, `tests/runtime/test_runtime_parity_bridge_contract.py::test_runtime_parity_bridge_trace_sink_parallel_nested_minimal` | normalized trace event shape and nested metadata remain compatible |
| `runtime_trace_events_metadata_shape` | `tests/workflow/test_tracing_e2e.py::*`, `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata` | trace event metadata contract stays compatible across runtimes |
| `side_by_side_node_edge_and_terminal_parity` | `tests/runtime/test_async_runtime_contract.py::test_async_runtime_side_by_side_node_edge_and_terminal_parity` with sync source extraction | normalized conversation node/edge side effects and terminal nodes match |

### Best Next Bridge Extractions

Start here. Highest signal, lowest ambiguity.

- [x] `nested_workflow_synthesized_design_is_persisted_and_used`
- [x] `workflow_suspend_and_resume_roundtrip`
- [x] `wait_reason_suspend_resume`
- [x] `join_barrier_waits_for_all_arrivals`
- [x] `workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown`
- [x] `trace_sink_parallel_nested_minimal`

### Bridgeable But Keep Existing Acceptance Too

Do not delete source acceptance when first bridging these.
All items in this section are now checked. Next bridge targets live in parity backlog, mainly checkpoint/resume and frontier/cancellation gaps.

- [x] `runtime_checkpoint_load_and_replay`
- [x] `runtime_resume_from_checkpoint`
- [x] `runtime_resume_from_checkpoint_frontier`
- [x] `workflow_suspend_and_resume_branching_roundtrip`
- [x] `side_by_side_node_edge_and_terminal_parity`
- [x] `parent_cancellation_propagates_to_child`

### Not Bridge First

These are related, but first bridge would be too distorted or too async-specific.
They are bridgeable only after extracting shared scheduler hooks or timing normalizers. Until then, keep them as async-focused acceptance guards.

- `test_async_runtime_native_scheduler_enforces_max_concurrent_tasks`
- `test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order`
- `test_async_resolver_runs_two_awaited_handlers_concurrently`
- `test_async_runtime_native_scheduler_cancellation_drains_inflight`

### Non-runtime or low-priority inventory

These appeared in runtime/workflow scans but are not first candidates for sync/async runtime bijection.

| Source case group | Reason |
| --- | --- |
| `tests/runtime/test_budget_ledger.py::*` | Budget model/ledger semantics, not runtime scheduler bijection. |
| `tests/runtime/test_budget_adapters.py::*` | Usage adapter semantics, not runtime scheduler bijection. |
| `tests/runtime/test_execution_*` | Analytics/template semantics, not runtime scheduler bijection. |
| `tests/runtime/test_grouped_artifacts.py::*` | Artifact helper semantics. |
| `tests/runtime/test_maintenance_template.py::*` | Maintenance template semantics. |
| `tests/runtime/test_versioned_artifacts.py::*` | Versioned artifact helper semantics. |
| `tests/workflow/test_langgraph_*` | LangGraph semantics-mode parity, tracked outside bijection subset. |
| `tests/workflow/*demo*` | Demo coverage, keep outside bijection subset unless extracting a tiny semantic case. |
| `tests/server/test_*design*` | Runtime design projection/history semantics, keep as server/design tests. |

## Waiver Ledger

Use this table when a runtime semantic case cannot be bijected 1:1.
Each waiver entry must answer three things:

- why exact bijection is wrong or premature
- which existing test remains replacement guard
- what regression risk stays if that guard is weakened

| Source case | Waiver reason | Replacement guard | Risk |
| --- | --- | --- | --- |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_two_awaited_handlers_concurrently` | Async-only concurrency; sync runtime cannot mirror awaited-task concurrency. | `test_async_resolver_runs_two_awaited_handlers_concurrently` | Async scheduler may regress without async-specific guard. |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_sync_handlers_inline_without_to_thread` | Async-only thread-free scheduler contract; sync runtime has no awaited handler lane to mirror. | `test_async_runtime_native_scheduler_sync_handlers_run_inline_without_to_thread` | Hidden thread offload may regress if async-only guard is removed. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_load_checkpoint_accepts_current_schema` | Schema compatibility guard, not runtime-kind semantic split. | `test_load_checkpoint_accepts_current_schema` | Future schema acceptance could regress silently. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_load_checkpoint_rejects_future_schema` | Schema compatibility guard, not runtime-kind semantic split. | `test_load_checkpoint_rejects_future_schema` | Future schema rejection could regress silently. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_replay_rejects_future_schema` | Schema compatibility guard, not runtime-kind semantic split. | `test_replay_rejects_future_schema` | Replay may accept unsupported schema and corrupt behavior. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_enforces_max_concurrent_tasks` | Async-only semaphore policy. | Same test. | Async overload behavior may regress. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order` | Async-only race acceptance-order policy. | Same test. | Nondeterministic state merge if removed. |

## Inventory Command

Refresh candidate inventory before each refactor batch:

```bash
rg -n "^def test_" tests/runtime tests/workflow tests/server | rg "runtime|workflow|resume|cancel|trace|checkpoint|replay|resolver|sandbox|join|fanout|route|transaction|uow|async"
```

It scans test defs, then filters names by runtime/workflow semantic keywords, so each refactor batch starts from fresh candidate list.

Then update this file case-by-case before moving tests.
