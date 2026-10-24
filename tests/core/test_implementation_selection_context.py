from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest
from kogwistar._rust_bridge import (
    contract_implementation_mode,
    graph_store_implementation_mode,
    meta_store_implementation_mode,
    postgres_authority_implementation_mode,
    runtime_implementation_mode,
    server_implementation_mode,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.implementation_selection import (
    capture_implementation_selection,
    implementation_selection_metadata,
    implementation_selection_bound,
    implementation_selection_scope,
    postgres_authority_selectors_aligned,
)
from kogwistar.runtime import AsyncWorkflowRuntime, WorkflowRuntime
from tests._helpers.embeddings import ConstantEmbeddingFunction

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


def test_selection_metadata_records_effective_defaults_and_transaction_coherence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_META_STORE", raising=False)
    monkeypatch.delenv("KOGWISTAR_IMPL_GRAPH_STORE", raising=False)
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)

    metadata = implementation_selection_metadata(capture_implementation_selection())

    assert metadata["selectors"]["KOGWISTAR_IMPL_META_STORE"] is None
    assert metadata["effective_selectors"]["KOGWISTAR_IMPL_META_STORE"] == "rust"
    assert metadata["effective_selectors"]["KOGWISTAR_IMPL_POSTGRES_AUTHORITY"] == "python"
    assert metadata["invariants"]["postgres_transaction_coherent"] is True
    assert metadata["invariants"]["postgres_selector_alignment"] is False


def test_selection_metadata_flags_incoherent_postgres_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")

    metadata = implementation_selection_metadata(capture_implementation_selection())

    assert metadata["invariants"]["postgres_transaction_coherent"] is False
    assert metadata["invariants"]["postgres_selector_alignment"] is False


@pytest.mark.parametrize(
    ("meta_mode", "graph_mode", "authority_mode", "expected"),
    [
        ("python", "python", "python", True),
        ("rust", "rust", "rust", True),
        ("rust", "python", "python", False),
        ("shadow", "shadow", "python", False),
    ],
)
def test_postgres_selectors_cannot_silently_split_authority(
    meta_mode: str, graph_mode: str, authority_mode: str, expected: bool
) -> None:
    assert (
        postgres_authority_selectors_aligned(
            meta_mode=meta_mode,
            graph_mode=graph_mode,
            authority_mode=authority_mode,
        )
        is expected
    )


def test_runtime_entrypoints_keep_sync_and_async_call_contracts() -> None:
    assert not inspect.iscoroutinefunction(WorkflowRuntime.run)
    assert inspect.iscoroutinefunction(AsyncWorkflowRuntime.run)


def test_bound_runtime_uses_its_construction_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    runtime = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )

    @implementation_selection_bound
    def run(owner: SimpleNamespace) -> tuple[str, str]:
        before = runtime_implementation_mode()
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        return before, runtime_implementation_mode()

    assert run(runtime) == ("python", "python")
    assert runtime_implementation_mode() == "rust"


def test_graph_engine_captures_selectors_when_constructed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=ConstantEmbeddingFunction(dim=4),
        backend_factory=build_in_memory_backend,
    )

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    with implementation_selection_scope(engine._implementation_selection_snapshot):
        assert runtime_implementation_mode() == "python"
    assert runtime_implementation_mode() == "rust"


@pytest.mark.parametrize("runtime_type", [WorkflowRuntime, AsyncWorkflowRuntime])
def test_runtime_rejects_engines_with_different_selection_snapshots(
    monkeypatch: pytest.MonkeyPatch, runtime_type: Callable[..., object]
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    workflow_engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    conversation_engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )

    with pytest.raises(ValueError, match="different implementation selections"):
        runtime_type(
            workflow_engine=workflow_engine,
            conversation_engine=conversation_engine,
            step_resolver=None,
            predicate_registry={},
        )


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
