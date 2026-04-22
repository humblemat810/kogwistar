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
- `Waived`: no 1:1 bijection mapping yet; record why and name replacement guard.

Never:

- Call a copied test `Moved from`.
- Delete a `Refactored from` source.
- Point sync-side provenance at async-runtime source.
- Move broad integration/backend/replay/server tests wholesale into bijection.
- Add only one side of a bijection suffix.

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

These cases are already in `test_sync_runtime_bijection_contract.py` and `test_async_runtime_bijection_contract.py`.

| Suffix | Sync case | Async case | Sync provenance | Async provenance | Source action |
| --- | --- | --- | --- | --- | --- |
| `default_ops_equal` | `test_sync_runtime_default_ops_equal` | `test_async_runtime_default_ops_equal` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_default_sync_ops_equal_default_async_ops` | Old async source removed. |
| `state_merge_semantics` | `test_sync_runtime_state_merge_semantics` | `test_async_runtime_state_merge_semantics` | Refactored from `tests/runtime/test_checkpoint_resume_contract.py::test_replay_state_reducer_matches_sync_runtime_merge_semantics` | Refactored from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_uses_shared_state_merge_semantics` | Both sources retained. |
| `step_context_and_result_contract` | `test_sync_runtime_step_context_and_result_contract` | `test_async_runtime_step_context_and_result_contract` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_step_context_and_result_contract` | Old async source removed. |
| `preserves_nested_ops_and_state_schema_in_adapter` | `test_sync_runtime_preserves_nested_ops_and_state_schema_in_adapter` | `test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter` | New sync mirror | Refactored from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_preserves_nested_ops_and_state_schema_in_adapter` | Async source retained. |
| `trace_fast_path_configuration` | `test_sync_runtime_trace_fast_path_configuration` | `test_async_runtime_trace_fast_path_configuration` | Refactored from `tests/runtime/test_workflow_suspend_resume.py::test_runtime_trace_writes_disable_eager_index_reconcile_for_in_memory_backend` | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_trace_fast_path_configuration_matches_sync_runtime` | Sync source retained; old async source removed. |
| `auto_transaction_mode_defaults_to_none_for_non_pg_backend` | `test_sync_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend` | `test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_auto_transaction_mode_defaults_to_none_for_non_pg_backend` | Old async source removed. |
| `auto_transaction_mode_uses_step_for_pg_backend` | `test_sync_runtime_auto_transaction_mode_uses_step_for_pg_backend` | `test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_runtime_auto_transaction_mode_uses_step_for_pg_backend` | Old async source removed. |
| `missing_op_behavior` | `test_sync_runtime_missing_op_behavior` | `test_async_runtime_missing_op_behavior` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_missing_op_behavior_matches_sync_resolver` | Old async source removed. |
| `exception_to_runfailure` | `test_sync_runtime_exception_to_runfailure` | `test_async_runtime_exception_to_runfailure` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_exception_to_runfailure_matches_sync_resolver` | Old async source removed. |
| `async_handler_callability` | `test_sync_runtime_async_handler_callability` | `test_async_runtime_async_handler_callability` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_adapter_accepts_async_handler` | Old async source removed. |
| `sync_handler_callability` | `test_sync_runtime_sync_handler_callability` | `test_async_runtime_sync_handler_callability` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_adapter_accepts_sync_handler` | Old async source removed. |
| `registered_op_appears_in_op_set` | `test_sync_runtime_registered_op_appears_in_op_set` | `test_async_runtime_registered_op_appears_in_op_set` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_registered_async_op_appears_in_async_op_set` | Old async source removed. |
| `preserves_sandboxed_behavior` | `test_sync_runtime_preserves_sandboxed_behavior` | `test_async_runtime_preserves_sandboxed_behavior` | Refactored from `tests/runtime/test_sandbox.py::test_mapping_resolver_executes_sandboxed_code_with_run_context` | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_resolver_preserves_sandboxed_behavior` | Sync source retained; old async source removed. |
| `legacy_update_warning_preserved` | `test_sync_runtime_legacy_update_warning_preserved` | `test_async_runtime_legacy_update_warning_preserved` | New sync mirror | Moved from `tests/runtime/test_async_runtime_contract.py::test_async_resolver_legacy_update_warning_preserved` | Old async source removed. |

## Candidate Case Backlog

This is case-by-case, not file-by-file. Classification may change after reading each test body.

## Useful Smoke Gaps To Add Next

These are high-signal, low-cost smoke tests that are not yet in the current bijection subset. Prefer these before moving broad workflow tests.

| Priority | Missing smoke | Why useful | Proposed action |
| --- | --- | --- | --- |
| P0 | Resolver state-schema metadata smoke | Catches sync/async resolver metadata drift without running workflow graph. | Move/refactor `test_async_resolver_state_schema_metadata_available`; add sync mirror with suffix `state_schema_metadata_available`. |
| P0 | Live `_deps` handler smoke | Catches DI/process-local state contract before checkpoint complexity. | Move/refactor `test_async_resolver_deps_available_in_handler`; add sync mirror with suffix `deps_available_in_handler`. |
| P0 | StepContext lane-message delegation smoke | `StepContext` is shared runtime surface; cheap and not backend-bound. | Move/refactor `test_step_context_send_lane_message_delegates_to_sender`; add async mirror with suffix `step_context_send_lane_message_delegates`. |
| P0 | StepContext lane-message missing-sender smoke | Guards failure shape for shared context helpers. | Move/refactor `test_step_context_send_lane_message_requires_sender`; add async mirror with suffix `step_context_send_lane_message_requires_sender`. |
| P0 | StepContext lane-event delegation smoke | Shared progress/event helper; useful for server progress compatibility. | Move/refactor `test_step_context_emit_lane_message_event_delegates_to_sink`; add async mirror with suffix `step_context_emit_lane_message_event_delegates`. |
| P0 | StepContext lane-event missing-sink smoke | Guards failure shape for shared event helper. | Move/refactor `test_step_context_emit_lane_message_event_requires_sink`; add async mirror with suffix `step_context_emit_lane_message_event_requires_sink`. |
| P1 | Sync runtime rejects awaitable handler smoke | Prevents accidentally making sync runtime async-capable by hidden await. | Move/refactor `test_sync_runtime_rejects_async_handler_by_runfailure`; async side should be explicit mirror/waiver because async accepts awaitables. |
| P1 | Async adapter `close_sandbox_run` forwarding smoke | Protects sandbox cleanup path when async facade delegates to sync runtime. | Move/refactor `test_async_runtime_adapter_forwards_close_sandbox_run`; add sync mirror or mark adapter-specific. |
| P1 | Non-sandboxed op does not prepare sandbox | Small resolver policy smoke, likely bijectable. | Refactor `test_mapping_resolver_does_not_prepare_sandbox_for_non_sandboxed_op`; add async mirror. |
| P1 | Same workflow graph model accepted by both runtimes | Guards constructor/graph-model contract without real workflow execution. | Refactor `test_sync_and_async_runtime_accept_same_workflow_graph_model`; source may remain if it also guards monkeypatch path. |
| P2 | Runtime public export/import smoke | Useful package boundary check but partly async-only. | Keep `test_async_runtime_exported`; add sync import smoke only if valuable, otherwise waiver. |
| P2 | Default resolver execution smoke | Valuable end-to-end smoke but too broad for bijection. | Keep `test_workflow_runtime_uses_default_resolver` as integration/CI-full; do not move wholesale. |

### Move async + implement sync mirror

These look like small async-side contract cases. Inspect before moving.

| Source case | Proposed suffix | Initial action |
| --- | --- | --- |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_exported` | `runtime_exported` | Move async if exact; add sync import/export mirror if useful, else waive as async-only public API. |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_sync_handlers_inline_without_to_thread` | `sync_handlers_inline_without_to_thread` | Probably async-only thread-free diagnostic; likely waive, not bijection. |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_two_awaited_handlers_concurrently` | `awaited_handlers_concurrently` | Async-only concurrency; waive from sync bijection. |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_state_schema_metadata_available` | `state_schema_metadata_available` | Move async + implement sync mirror. |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_deps_available_in_handler` | `deps_available_in_handler` | Move async + implement sync mirror if exact. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_adapter_forwards_close_sandbox_run` | `adapter_forwards_close_sandbox_run` | Move async + implement sync mirror or refactor from sandbox close semantics. |
| `tests/runtime/test_async_runtime_contract.py::test_sync_runtime_rejects_async_handler_by_runfailure` | `rejects_async_handler_by_runfailure` | Move sync-like case to sync bijection + implement async mirror or mark sync-only rejection. |
| `tests/runtime/test_async_runtime_contract.py::test_sync_and_async_runtime_accept_same_workflow_graph_model` | `accept_same_workflow_graph_model` | Refactor into bijection; source may remain if broader runtime constructor behavior. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_linear_terminal_status_equivalent_to_sync` | `linear_terminal_status_equivalent` | Move/refactor async; add sync mirror or rely on existing side-by-side parity if too fake-runtime specific. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_branch_join_status_and_state_equivalent_to_sync` | `branch_join_status_and_state_equivalent` | Move/refactor async; add sync mirror or keep as parity acceptance. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_deps_live_but_omitted_from_checkpoint_payload` | `deps_live_not_checkpointed` | Refactor; source likely retained for checkpoint semantics. |

### Move sync + implement async mirror

These are sync-side baseline cases likely worth bijection triage.

| Source case | Proposed suffix | Initial action |
| --- | --- | --- |
| `tests/runtime/test_resume_wait_reasons.py::test_wait_reason_can_suspend_and_resume` | `wait_reason_suspend_resume` | Refactor from sync source; source retained for resume wait-reason integration. |
| `tests/runtime/test_trace_sink_parallel_nested_minimal.py::test_trace_sink_parallel_and_nested_minimal_sync` | `trace_sink_parallel_nested_minimal` | Refactor; source retained for trace sink integration. |
| `tests/runtime/test_workflow_terminal_status.py::test_runtime_persists_completed_terminal_for_leaf_node` | `persists_completed_terminal_for_leaf_node` | Refactor; source retained for backend-param terminal persistence. |
| `tests/runtime/test_workflow_terminal_status.py::test_runtime_persists_failed_terminal_for_leaf_node` | `persists_failed_terminal_for_leaf_node` | Refactor; source retained for backend-param terminal persistence. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_load_checkpoint_accepts_current_schema` | `load_checkpoint_accepts_current_schema` | Likely replay/checkpoint contract, not runtime bijection; maybe waive. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_load_checkpoint_rejects_future_schema` | `load_checkpoint_rejects_future_schema` | Likely replay/checkpoint contract, not runtime bijection; maybe waive. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_replay_rejects_future_schema` | `replay_rejects_future_schema` | Likely replay-only; waive or acceptance coverage. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_replay_ignores_created_at_ms_and_orders_by_step_seq` | `replay_orders_by_step_seq` | Refactor/waive; source retained for replay. |
| `tests/runtime/test_checkpoint_resume_contract.py::test_replay_to_is_read_only_and_does_not_append_new_history` | `replay_read_only_no_new_history` | Refactor/waive; source retained for event history semantics. |
| `tests/runtime/test_sandbox.py::test_mapping_resolver_does_not_prepare_sandbox_for_non_sandboxed_op` | `non_sandboxed_op_does_not_prepare_sandbox` | Move/refactor sync + implement async mirror if async resolver supports same sandbox metadata. |
| `tests/runtime/test_step_context_lane_message.py::test_step_context_send_lane_message_delegates_to_sender` | `step_context_send_lane_message_delegates` | Move/refactor sync + async mirror if StepContext shared. |
| `tests/runtime/test_step_context_lane_message.py::test_step_context_send_lane_message_requires_sender` | `step_context_send_lane_message_requires_sender` | Move/refactor sync + async mirror if StepContext shared. |
| `tests/runtime/test_step_context_lane_message_events.py::test_step_context_emit_lane_message_event_delegates_to_sink` | `step_context_emit_lane_message_event_delegates` | Move/refactor sync + async mirror if StepContext shared. |
| `tests/runtime/test_step_context_lane_message_events.py::test_step_context_emit_lane_message_event_requires_sink` | `step_context_emit_lane_message_event_requires_sink` | Move/refactor sync + async mirror if StepContext shared. |
| `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_synthesized_design_is_persisted_and_used` | `nested_workflow_synthesized_design` | Refactor; source retained for persistence/design semantics. |
| `tests/runtime/test_workflow_invocation_and_route_next.py::test_route_next_alias_can_fan_out_multiple_branches` | `route_next_alias_fanout` | Refactor into async mirror; source retained for sync route-next integration. |
| `tests/runtime/test_workflow_invocation_and_route_next.py::test_nested_workflow_failure_short_circuits_parent_routing` | `nested_failure_short_circuits_parent` | Refactor; source retained for nested routing integration. |
| `tests/workflow/test_workflow_native_update.py::test_workflow_runtime_native_update_schema_applies_known_and_falls_back_unknown` | `native_update_schema_known_and_unknown` | Refactor; source retained for native update schema. |
| `tests/workflow/test_workflow_join.py::test_join_barrier_waits_for_all_arrivals` | `join_barrier_waits_for_all_arrivals` | Refactor; source retained for thread-worker sync join integration. |
| `tests/workflow/test_workflow_join.py::test_join_does_not_wait_for_branch_that_can_no_longer_reach_it` | `join_unreachable_branch_not_required` | Refactor; source retained for reachability semantics. |
| `tests/workflow/test_save_load_progress.py::test_runtime_checkpoint_load_and_replay` | `checkpoint_load_and_replay` | Refactor/waive; source retained for save/load progress. |
| `tests/workflow/test_save_load_progress.py::test_runtime_resume_from_checkpoint` | `resume_from_checkpoint` | Refactor; source retained for save/load progress. |
| `tests/workflow/test_save_load_progress.py::test_runtime_resume_from_checkpoint_frontier` | `resume_from_checkpoint_frontier` | Refactor; source retained for frontier semantics. |

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

| Source case | Waiver reason | Replacement guard | Risk |
| --- | --- | --- | --- |
| `tests/runtime/test_async_runtime_contract.py::test_async_resolver_runs_two_awaited_handlers_concurrently` | Async-only concurrency; sync runtime cannot mirror awaited-task concurrency. | `test_async_resolver_runs_two_awaited_handlers_concurrently` | Async scheduler may regress without async-specific guard. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_enforces_max_concurrent_tasks` | Async-only semaphore policy. | Same test. | Async overload behavior may regress. |
| `tests/runtime/test_async_runtime_contract.py::test_async_runtime_native_scheduler_applies_completed_tasks_in_step_order` | Async-only race acceptance-order policy. | Same test. | Nondeterministic state merge if removed. |

## Inventory Command

Refresh candidate inventory before each refactor batch:

```powershell
rg -n "^def test_" tests/runtime tests/workflow tests/server | rg "runtime|workflow|resume|cancel|trace|checkpoint|replay|resolver|sandbox|join|fanout|route|transaction|uow|async"
```

Then update this file case-by-case before moving tests.
