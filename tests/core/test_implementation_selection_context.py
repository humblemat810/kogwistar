from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest
from kogwistar._rust_bridge import (
    contract_implementation_mode,
    graph_store_implementation_mode,
    meta_store_implementation_mode,
    postgres_authority_implementation_mode,
    runtime_implementation_mode,
    server_implementation_mode,
)
from kogwistar.implementation_selection import (
    implementation_selection_bound,
    implementation_selection_scope,
)
from kogwistar.runtime import AsyncWorkflowRuntime, WorkflowRuntime

pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_selector_values_are_frozen_and_nested_scope_does_not_recapture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    monkeypatch.delenv("KOGWISTAR_IMPL_RUNTIME", raising=False)

    with implementation_selection_scope():
        assert runtime_implementation_mode() == "python"
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
        with implementation_selection_scope():
            assert runtime_implementation_mode() == "python"
        assert runtime_implementation_mode() == "python"

    assert runtime_implementation_mode() == "rust"


def test_all_selector_families_use_the_same_run_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectors = (
        contract_implementation_mode,
        graph_store_implementation_mode,
        meta_store_implementation_mode,
        postgres_authority_implementation_mode,
        runtime_implementation_mode,
        server_implementation_mode,
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    for name in (
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.delenv(name, raising=False)

    with implementation_selection_scope():
        for name in (
            "KOGWISTAR_IMPL_MODE",
            "KOGWISTAR_IMPL_CONTRACTS",
            "KOGWISTAR_IMPL_GRAPH_STORE",
            "KOGWISTAR_IMPL_META_STORE",
            "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
            "KOGWISTAR_IMPL_RUNTIME",
            "KOGWISTAR_IMPL_SERVER",
        ):
            monkeypatch.setenv(name, "rust")
        assert [selector() for selector in selectors] == ["python"] * len(selectors)

    assert [selector() for selector in selectors] == ["rust"] * len(selectors)


def test_runtime_entrypoints_keep_sync_and_async_call_contracts() -> None:
    assert not inspect.iscoroutinefunction(WorkflowRuntime.run)
    assert inspect.iscoroutinefunction(AsyncWorkflowRuntime.run)


def test_selector_snapshot_propagates_to_async_tasks_and_worker_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "python")

    async def check() -> tuple[str, str]:
        with implementation_selection_scope():
            monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")
            child_value = await asyncio.create_task(asyncio.to_thread(server_implementation_mode))
            return server_implementation_mode(), child_value

    assert asyncio.run(check()) == ("python", "python")
    assert server_implementation_mode() == "rust"


def test_snapshot_can_be_copied_to_sync_step_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    with implementation_selection_scope():
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        copied = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(copied.run, runtime_implementation_mode).result()

    assert result == "python"
    assert runtime_implementation_mode() == "rust"


def test_bound_async_run_holds_snapshot_until_await_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")

    @implementation_selection_bound
    async def run() -> tuple[str, str]:
        before = runtime_implementation_mode()
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        await asyncio.sleep(0)
        return before, runtime_implementation_mode()

    assert asyncio.run(run()) == ("python", "python")
    assert runtime_implementation_mode() == "rust"
