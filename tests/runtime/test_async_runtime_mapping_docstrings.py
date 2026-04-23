from __future__ import annotations

import importlib
import inspect

import pytest

pytestmark = [pytest.mark.ci, pytest.mark.runtime, pytest.mark.runtime_async]


MAPPINGS = [
    (
        "tests.workflow.test_workflow_join",
        "test_join_barrier_waits_for_all_arrivals",
        "tests.runtime.test_async_runtime_bijection_contract",
        "test_async_runtime_branch_join_status_and_state_equivalent",
    ),
    (
        "tests.workflow.test_workflow_join",
        "test_join_does_not_wait_for_branch_that_can_no_longer_reach_it",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_native_scheduler_without_join_executes_once_per_token",
    ),
    (
        "tests.workflow.test_workflow_join",
        "test_nested_joins_human_debug",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_native_scheduler_token_nesting_and_spawn_events",
    ),
    (
        "tests.runtime.test_checkpoint_resume_contract",
        "test_replay_state_reducer_matches_sync_runtime_merge_semantics",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_native_scheduler_uses_shared_state_merge_semantics",
    ),
    (
        "tests.runtime.test_checkpoint_resume_contract",
        "test_replay_ignores_created_at_ms_and_orders_by_step_seq",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_native_scheduler_persists_rt_join_frontier_shape",
    ),
    (
        "tests.runtime.test_checkpoint_resume_contract",
        "test_replay_to_is_read_only_and_does_not_append_new_history",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_resume_run_is_not_supported",
    ),
    (
        "tests.runtime.test_workflow_suspend_resume",
        "test_workflow_suspend_and_resume",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_suspend_and_resume_roundtrip",
    ),
    (
        "tests.runtime.test_workflow_suspend_resume",
        "test_workflow_suspend_and_resume_branching",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_suspend_and_resume_roundtrip",
    ),
    (
        "tests.runtime.test_workflow_suspend_resume",
        "test_resume_run_failure_can_route_to_recovery_branch",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_parent_cancellation_propagates_to_child",
    ),
    (
        "tests.runtime.test_workflow_suspend_resume",
        "test_resume_run_can_resuspend_same_token_with_updated_payload",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_run_with_resume_markers_is_not_supported",
    ),
    (
        "tests.runtime.test_workflow_suspend_resume",
        "test_sandbox_recoverable_error_can_suspend_then_resume_success",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_suspend_and_resume_roundtrip",
    ),
    (
        "tests.runtime.test_workflow_cancel_event_sourced",
        "test_runtime_event_sourced_cancel_reconciles_and_replay_is_stable",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_native_scheduler_cancellation_drains_inflight",
    ),
    (
        "tests.runtime.test_workflow_invocation_and_route_next",
        "test_nested_workflow_synthesized_design_is_persisted_and_used",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_nested_workflow_invocation_matches_sync",
    ),
    (
        "tests.runtime.test_workflow_invocation_and_route_next",
        "test_nested_workflow_failure_short_circuits_parent_routing",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_nested_workflow_child_failure_fails_parent",
    ),
    (
        "tests.runtime.test_trace_sink_parallel_nested_minimal",
        "test_trace_sink_parallel_and_nested_minimal_sync",
        "tests.runtime.test_async_runtime_contract",
        "test_async_runtime_native_scheduler_emits_trace_events_with_expected_metadata",
    ),
]


@pytest.mark.parametrize("sync_mod,sync_name,async_mod,async_name", MAPPINGS)
def test_sync_async_mapping_docstrings_have_first_line_pairs(
    sync_mod: str, sync_name: str, async_mod: str, async_name: str
):
    sync_fn = getattr(importlib.import_module(sync_mod), sync_name)
    async_fn = getattr(importlib.import_module(async_mod), async_name)

    sync_doc = inspect.getdoc(sync_fn)
    async_doc = inspect.getdoc(async_fn)

    assert sync_doc, f"missing docstring: {sync_mod}::{sync_name}"
    assert async_doc, f"missing docstring: {async_mod}::{async_name}"

    sync_first = sync_doc.splitlines()[0]
    async_first = async_doc.splitlines()[0]
    sync_file = sync_mod.replace(".", "/") + ".py"
    async_file = async_mod.replace(".", "/") + ".py"
    allowed_sync_files = {sync_file}
    if async_mod.endswith("test_async_runtime_bijection_contract"):
        allowed_sync_files.add(
            "tests/runtime/test_sync_runtime_bijection_contract.py"
        )

    assert sync_first.startswith("Async mirror:"), sync_first
    assert async_first.startswith("Sync mirror:"), async_first
    assert async_file in sync_first, sync_first
    assert any(path in async_first for path in allowed_sync_files), async_first


@pytest.mark.parametrize("sync_mod,sync_name,async_mod,async_name", MAPPINGS)
def test_sync_async_mapping_names_follow_prefix_convention(
    sync_mod: str, sync_name: str, async_mod: str, async_name: str
):
    sync_fn = getattr(importlib.import_module(sync_mod), sync_name)
    async_fn = getattr(importlib.import_module(async_mod), async_name)

    assert sync_fn.__name__.startswith("test_"), sync_fn.__name__
    assert not sync_fn.__name__.startswith("test_async_"), sync_fn.__name__
    assert async_fn.__name__.startswith("test_async_"), async_fn.__name__
