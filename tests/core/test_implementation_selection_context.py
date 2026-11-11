from __future__ import annotations

import asyncio
import inspect
import sqlite3
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from kogwistar._rust_bridge import (
    api_health,
    contract_implementation_mode,
    contract_canonical_json,
    graph_store_implementation_mode,
    meta_store_implementation_mode,
    postgres_authority_implementation_mode,
    runtime_implementation_mode,
    server_implementation_mode,
    runtime_apply_state_update,
    store_memory_read,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine, engine_context
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.in_memory_backend import InMemoryBackend, build_in_memory_backend
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.json_types import JsonValue
from kogwistar.engine_core.postgres_backend import (
    CollectionSpec,
    PgCollectionFacade,
    PgVectorBackend,
    PostgresUnitOfWork,
)
from kogwistar.engine_core.subsystems.write import WriteSubsystem
from kogwistar.engine_core.two_stage_async import AsyncPostgresTwoStageProjectionAdapter
from kogwistar.agent import HookRegistry, HookSpec
from kogwistar.implementation_selection import (
    active_transaction_authority,
    callback_execution_scope,
    capture_implementation_selection,
    implementation_selection_metadata,
    implementation_selection_bound,
    implementation_selection_scope,
    assert_python_transaction_entry_allowed,
    postgres_authority_selectors_aligned,
    resolve_implementation_selection,
    selected_transaction_authority,
    transaction_authority_for_selection,
    transaction_authority_scope,
)
from kogwistar.runtime import AsyncWorkflowRuntime, WorkflowRuntime
from tests._helpers.embeddings import ConstantEmbeddingFunction

pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(autouse=True)
def _isolate_implementation_selector_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep selector tests independent of CI's Python-authority overrides."""
    for name in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_python_selection_snapshot_prevents_late_native_bridge_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    for name in (
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
    ):
        monkeypatch.delenv(name, raising=False)
    snapshot = capture_implementation_selection()

    def unexpected_native_load() -> None:
        pytest.fail("Python-selected bridge attempted to load the Rust extension")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unexpected_native_load)
    with implementation_selection_scope(snapshot):
        for name in (
            "KOGWISTAR_IMPL_MODE",
            "KOGWISTAR_IMPL_CONTRACTS",
            "KOGWISTAR_IMPL_GRAPH_STORE",
            "KOGWISTAR_IMPL_META_STORE",
            "KOGWISTAR_IMPL_RUNTIME",
            "KOGWISTAR_IMPL_SERVER",
        ):
            monkeypatch.setenv(name, "rust")

        python_json = '{"selected":"python"}'
        python_update: list[object] = []
        python_nodes: list[object] = []
        python_status = {"ok": True, "implementation": "python"}
        assert contract_canonical_json(
            value={"selected": "python"}, python_value=python_json
        ) == python_json
        assert runtime_apply_state_update(
            payload={"updates": []}, python_value=python_update
        ) is python_update
        assert store_memory_read(
            snapshot={},
            operation={"kind": "list_nodes"},
            python_value=python_nodes,
            store="graph",
        ) is python_nodes
        assert api_health(payload={}, python_value=python_status) is python_status


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


def test_global_rust_fallback_and_transaction_owner_use_the_same_effective_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_META_STORE", raising=False)
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)

    snapshot = capture_implementation_selection()
    metadata = implementation_selection_metadata(snapshot)

    assert metadata["effective_selectors"]["KOGWISTAR_IMPL_META_STORE"] == "rust"
    assert transaction_authority_for_selection(snapshot) == "rust"
    with implementation_selection_scope(snapshot):
        with pytest.raises(RuntimeError, match="does not match frozen implementation selection"):
            with transaction_authority_scope("python"):
                pass
        with transaction_authority_scope("rust"):
            assert active_transaction_authority() == "rust"


def test_empty_selector_is_not_reported_as_an_unset_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "")

    metadata = implementation_selection_metadata(capture_implementation_selection())

    assert metadata["effective_selectors"]["KOGWISTAR_IMPL_RUNTIME"] == ""


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
    assert getattr(
        WorkflowRuntime.resume_run,
        "__implementation_selection_bound__",
        False,
    )
    assert getattr(
        WorkflowRuntime.resume_from_latest_checkpoint,
        "__implementation_selection_bound__",
        False,
    )
    assert getattr(
        AsyncWorkflowRuntime.resume_from_latest_checkpoint,
        "__implementation_selection_bound__",
        False,
    )


def test_every_public_lane_messaging_operation_binds_its_engine_selection() -> None:
    from kogwistar.messaging.service import LaneMessagingService

    unbound = [
        name
        for name, member in inspect.getmembers(LaneMessagingService, callable)
        if not name.startswith("_")
        and not getattr(member, "__implementation_selection_bound__", False)
    ]

    assert unbound == []


def test_public_recovery_and_indexing_operations_bind_engine_selection() -> None:
    from kogwistar.engine_core.recovery import RecoverySubsystem
    from kogwistar.engine_core.subsystems.persist import PersistSubsystem
    from kogwistar.engine_core.subsystems.read import ReadSubsystem
    from kogwistar.engine_core.subsystems.rollback import RollbackSubsystem
    from kogwistar.engine_core.subsystems.acl import ACLSubsystem
    from kogwistar.engine_core.subsystems.acl import ACLAwareReadSubsystem
    from kogwistar.engine_core.subsystems.acl import ACLAwareWriteSubsystem
    from kogwistar.engine_core.subsystems.ingest import IngestSubsystem
    from kogwistar.engine_core.search_index.service import SearchIndexService
    from kogwistar.messaging.service import LaneMessagingService
    from kogwistar.engine_core.jobs import JobQueueSubsystem
    from kogwistar.server.chat_service_run_execution import _RunExecutionService
    from kogwistar.server.chat_service_run_inspection import _RunInspectionService
    from kogwistar.server.chat_service_workflow_design import _WorkflowDesignService
    from kogwistar.server.chat_service_conversation_queries import (
        _ConversationQueryService,
    )
    from kogwistar.server.service_daemon import ServiceSupervisor

    unbound: list[str] = []
    for subsystem in (
        RecoverySubsystem,
        IndexingSubsystem,
        JobQueueSubsystem,
        LaneMessagingService,
        ReadSubsystem,
        RollbackSubsystem,
        ACLAwareReadSubsystem,
        IngestSubsystem,
        SearchIndexService,
    ):
        unbound.extend(
            f"{subsystem.__name__}.{name}"
            for name, member in inspect.getmembers(subsystem, callable)
            if not name.startswith("_")
            and not getattr(member, "__implementation_selection_bound__", False)
        )

    persist_entrypoints = (
        "persist_graph",
        "persist_graph_extraction",
        "persist_document_graph_extraction",
        "ingest_with_toposort",
    )
    unbound.extend(
        f"PersistSubsystem.{name}"
        for name in persist_entrypoints
        if not getattr(
            getattr(PersistSubsystem, name),
            "__implementation_selection_bound__",
            False,
        )
    )
    acl_entrypoints = (
        "append_canonical_write_events_for_item",
        "record_default_acl_for_item",
        "repair_default_acl_for_item",
        "repair_acl_records_from_events",
        "record_acl",
        "repair_missing_default_acls",
        "prefetch_acl_neighborhood",
        "rebuild_acl_graph_from_truth",
        "decide_acl",
        "decide_acl_usage",
        "decide_acl_node_read",
        "get_node_acl_checked",
        "get_edge_acl_checked",
    )
    unbound.extend(
        f"ACLSubsystem.{name}"
        for name in acl_entrypoints
        if not getattr(
            getattr(ACLSubsystem, name),
            "__implementation_selection_bound__",
            False,
        )
    )

    write_entrypoints = (
        "add_node",
        "create_node_if_absent",
        "add_edge",
        "add_node_async",
        "add_edge_async",
        "patch_base_projection_metadata",
        "rust_postgres_delete_existing",
        "rust_postgres_replace_existing",
        "add_pure_node",
        "add_pure_edge",
        "add_document",
        "add_domain",
        "index_node_docs",
        "index_node_refs",
        "index_edge_refs",
        "delete_edge_ref_rows",
        "delete_node_ref_rows",
        "maybe_reindex_edge_refs",
        "maybe_reindex_node_refs",
        "prune_node_refs_for_doc",
        "rebuild_edge_refs_for_doc",
        "rebuild_all_edge_refs",
        "rebuild_node_refs_for_doc",
        "rebuild_all_node_refs",
        "delete_edges_by_ids",
    )
    unbound.extend(
        f"WriteSubsystem.{name}"
        for name in write_entrypoints
        if not getattr(
            getattr(WriteSubsystem, name),
            "__implementation_selection_bound__",
            False,
        )
    )

    acl_write_entrypoints = (
        "add_document",
        "add_node_async",
        "add_edge_async",
        "add_pure_node",
        "add_pure_edge",
        "add_domain",
        "maybe_reindex_edge_refs",
        "maybe_reindex_node_refs",
        "prune_node_refs_for_doc",
        "rebuild_edge_refs_for_doc",
        "rebuild_all_edge_refs",
        "rebuild_node_refs_for_doc",
        "rebuild_all_node_refs",
        "delete_edges_by_ids",
        "rust_postgres_delete_existing",
        "rust_postgres_replace_existing",
        "uses_rust_postgres_authority",
        "add_node",
        "create_node_if_absent",
        "add_edge",
        "index_node_docs",
        "index_node_refs",
        "index_edge_refs",
        "delete_edge_ref_rows",
        "delete_node_ref_rows",
    )
    unbound.extend(
        f"ACLAwareWriteSubsystem.{name}"
        for name in acl_write_entrypoints
        if not getattr(
            getattr(ACLAwareWriteSubsystem, name),
            "__implementation_selection_bound__",
            False,
        )
    )

    # Reflect over public server methods so newly added engine-facing entrypoints
    # cannot silently bypass implementation-selection binding. These exceptions
    # only forward to an already-bound owner or return static catalog data.
    safe_unbound_methods = {
        _RunExecutionService: {"list_steps"},
        _RunInspectionService: {"list_steps"},
        _WorkflowDesignService: {"list_steps", "workflow_catalog_ops"},
        _ConversationQueryService: {"list_steps"},
        ServiceSupervisor: set(),
    }
    for service_type, exceptions in safe_unbound_methods.items():
        unbound.extend(
            f"{service_type.__name__}.{name}"
            for name, method in inspect.getmembers(service_type, callable)
            if not name.startswith("_")
            and name not in exceptions
            and not getattr(method, "__implementation_selection_bound__", False)
        )

    assert unbound == []


def test_server_resume_runner_uses_conversation_engine_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.server.chat_service_run_execution import _RunExecutionService

    class ResumeReachedCheckpoint(Exception):
        pass

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    owner = SimpleNamespace(_conversation_engine=lambda: engine)
    service = _RunExecutionService(owner)
    request = SimpleNamespace(
        client_result={"step_seq": 0},
        conversation_engine=engine,
        run_id="resume-selector-test",
    )

    def stop_at_checkpoint(**_kwargs: object) -> dict[str, object]:
        assert runtime_implementation_mode() == "python"
        raise ResumeReachedCheckpoint

    monkeypatch.setattr(
        "kogwistar.server.chat_service_run_execution.load_checkpoint",
        stop_at_checkpoint,
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")

    with pytest.raises(ResumeReachedCheckpoint):
        service._default_resume_runner(request)

    assert runtime_implementation_mode() == "rust"


def test_checkpoint_resume_cannot_switch_transaction_owner_mid_continuation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    store = EngineSQLite(tmp_path / "resume-owner.sqlite")
    store.ensure_initialized()
    store.set_user_seq("resume-owner", 7)

    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)
    runtime = WorkflowRuntime.__new__(WorkflowRuntime)
    runtime._implementation_selection_snapshot = capture_implementation_selection()
    checkpoint = SimpleNamespace(
        metadata={
            "workflow_id": "workflow",
            "step_seq": 3,
            "state_json": "{}",
        }
    )

    def latest_checkpoint(**_kwargs: object) -> object:
        monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
        return checkpoint

    observed: list[str] = []

    def run(**_kwargs: object) -> str:
        observed.append(runtime_implementation_mode())
        with pytest.raises(
            RuntimeError,
            match="frozen Rust transaction authority",
        ):
            store.set_user_seq("resume-owner", 8)
        return "resumed"

    runtime._latest_checkpoint_for_run = latest_checkpoint
    runtime.run = run
    result = runtime.resume_from_latest_checkpoint(
        run_id="run",
        workflow_id="workflow",
        conversation_id="conversation",
        turn_node_id="turn",
    )

    assert result == "resumed"
    assert observed == ["python"]
    assert runtime_implementation_mode() == "python"
    assert store.current_user_seq("resume-owner") == 7


def test_resume_contract_keeps_conversation_engine_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.server.chat_service_run_inspection import _RunInspectionService

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    observed: list[str] = []

    def get_nodes(**_kwargs: object) -> list[object]:
        observed.append(runtime_implementation_mode())
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        return []

    engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection(),
        meta_sqlite=object(),
        get_nodes=get_nodes,
    )
    owner = SimpleNamespace(_conversation_engine=lambda: engine)
    service = _RunInspectionService(owner)

    result = service.resume_contract("inspection-selector-test")

    assert result["compatible"] is False
    assert observed == ["python", "python"]
    assert runtime_implementation_mode() == "rust"


def test_workflow_design_history_uses_workflow_engine_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.server.chat_service_workflow_design import _WorkflowDesignService

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    workflow_engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    conversation_engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    owner = SimpleNamespace(
        _workflow_engine=lambda: workflow_engine,
        _conversation_engine=lambda: conversation_engine,
        _workflow_history_lock=threading.Lock(),
    )
    service = _WorkflowDesignService(owner)
    observed: list[str] = []

    def projection_status(**_kwargs: object) -> str:
        observed.append(runtime_implementation_mode())
        return "ready"

    def sync_projection(**_kwargs: object) -> dict[str, object]:
        observed.append(runtime_implementation_mode())
        return {}

    monkeypatch.setattr(service, "_workflow_projection_status", projection_status)
    monkeypatch.setattr(service, "_workflow_sync_projection_locked", sync_projection)

    result = service.workflow_design_history(workflow_id="wf-selector-test")

    assert result["materialization_status"] == "ready"
    assert observed == ["python", "python", "python"]
    assert runtime_implementation_mode() == "rust"


def test_conversation_query_uses_conversation_engine_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.server.chat_service_conversation_queries import (
        _ConversationQueryService,
    )

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    owner = SimpleNamespace(
        _conversation_engine=lambda: engine,
        _conversation_service=lambda: SimpleNamespace(
            get_conversation_tail=lambda **_kwargs: SimpleNamespace(id="tail")
        ),
    )
    service = _ConversationQueryService(owner)
    observed: list[str] = []
    start_node = SimpleNamespace(
        metadata={"entity_type": "conversation_start"},
        properties={"status": "active"},
        id="start",
        user_id="user-1",
        turn_index=0,
    )

    def conversation_nodes(_conversation_id: str) -> list[SimpleNamespace]:
        observed.append(runtime_implementation_mode())
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        return [start_node]

    monkeypatch.setattr(service, "_conversation_nodes", conversation_nodes)

    result = service.get_conversation("conversation-selector-test")

    assert result["tail_node_id"] == "tail"
    assert result["turn_count"] == 0
    assert observed == ["python", "python"]
    assert runtime_implementation_mode() == "rust"


def test_service_supervisor_uses_workflow_engine_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.server.service_daemon import ServiceSupervisor

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    workflow_engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    conversation_engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    supervisor = ServiceSupervisor(
        get_workflow_engine=lambda: workflow_engine,
        get_conversation_engine=lambda: conversation_engine,
        run_registry=SimpleNamespace(),
        spawn_workflow_run=lambda **_kwargs: {},
        scope_snapshot=lambda: {},
    )
    observed: list[str] = []

    def trigger_service(**_kwargs: object) -> dict[str, object]:
        observed.append(runtime_implementation_mode())
        return {"service_id": "service-selector-test"}

    monkeypatch.setattr(supervisor, "_trigger_service", trigger_service)

    result = supervisor.trigger_service(
        "service-selector-test", trigger_type="external event"
    )

    assert result["service_id"] == "service-selector-test"
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_indexing_compound_enqueue_keeps_one_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []

    class Metadata:
        def enqueue_index_job(self, **_values: object) -> str:
            observed.append(runtime_implementation_mode())
            return f"job-{len(observed)}"

    engine = SimpleNamespace(
        _phase1_enable_index_jobs=True,
        namespace="default",
        metadata=Metadata(),
        persistence_mode="single_stage",
    )
    subsystem = IndexingSubsystem(engine=engine)

    def source_payload(**_values: object) -> str:
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
        return "{}"

    monkeypatch.setattr(subsystem, "canonical_source_payload", source_payload)
    subsystem.enqueue_index_jobs_for_node("node-1", op="UPSERT")

    assert observed == ["python", "python"]
    assert runtime_implementation_mode() == "rust"


def test_job_queue_operations_keep_engine_selection_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
    )
    observed_authorities: list[str] = []

    def claim_jobs(**kwargs: object) -> list[object]:
        observed_authorities.append(selected_transaction_authority())
        return []

    monkeypatch.setattr(engine.metadata, "claim_index_jobs", claim_jobs)
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")

    assert engine.jobs.claim(namespace="jobs:selection") == []
    assert observed_authorities == ["python"]


def test_server_background_run_keeps_conversation_engine_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.server.chat_service_run_execution import _RunExecutionService
    from kogwistar.server.chat_service_shared import AnswerRunRequest
    from kogwistar.server.run_registry import RunRegistry

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
    )
    registry = RunRegistry(engine.metadata)
    run_id = "run-selection-snapshot"
    registry.create_run(
        run_id=run_id,
        conversation_id="conversation-selection-snapshot",
        workflow_id="answer.v1",
        user_id="user-1",
        user_turn_node_id="turn-1",
    )
    observed_authorities: list[str] = []

    def answer_runner(_request: AnswerRunRequest) -> dict[str, JsonValue]:
        observed_authorities.append(selected_transaction_authority())
        return {
            "assistant_text": "answer",
            "assistant_turn_node_id": "assistant-turn-1",
            "workflow_status": "succeeded",
        }

    owner = SimpleNamespace(
        _conversation_engine=lambda: engine,
        run_registry=registry,
        answer_runner=answer_runner,
        _publish=lambda rid, event_type, payload=None: registry.append_event(
            rid, event_type, payload
        ),
    )
    execution = _RunExecutionService(owner)
    request = AnswerRunRequest(
        run_id=run_id,
        conversation_id="conversation-selection-snapshot",
        user_id="user-1",
        user_text="hello",
        user_turn_node_id="turn-1",
        workflow_id="answer.v1",
        knowledge_engine=engine,
        conversation_engine=engine,
        workflow_engine=engine,
        prev_turn_meta_summary=SimpleNamespace(),
        registry=registry,
        publish=lambda event_type, payload=None: registry.append_event(
            run_id, event_type, payload
        ),
        is_cancel_requested=lambda: False,
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")

    execution._run_answer(request)

    assert observed_authorities == ["python"]
    assert registry.get_run(run_id)["status"] == "succeeded"


def test_recovery_inspection_keeps_one_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.engine_core.recovery import RecoverySubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []

    class Metadata:
        def list_index_jobs(self, **_values: object) -> list[object]:
            monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
            observed.append(runtime_implementation_mode())
            return []

    recovery = RecoverySubsystem(SimpleNamespace(metadata=Metadata()))
    assert recovery.inspect_queues(["alpha"]) == []
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_read_operation_keeps_one_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.engine_core.models import Node
    from kogwistar.engine_core.subsystems.read import ReadSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []
    engine = SimpleNamespace(
        _resolve_redirect_chain=lambda **values: values["initial_items"],
        _filter_items_by_resolve_mode=lambda items, _mode: items,
    )
    subsystem = ReadSubsystem(engine)
    original_convert = subsystem.nodes_from_single_or_id_query_result

    def fetch(**_values: object) -> dict[str, object]:
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
        return {"ids": [], "documents": [], "metadatas": []}

    def observe_conversion(
        got: object, **values: object
    ) -> list[Node]:
        observed.append(runtime_implementation_mode())
        return original_convert(
            cast(Mapping[str, JsonValue], got),
            node_type=cast(type[Node], values["node_type"]),
        )

    monkeypatch.setattr(subsystem, "_node_get_raw", fetch)
    monkeypatch.setattr(
        subsystem, "nodes_from_single_or_id_query_result", observe_conversion
    )

    assert subsystem.get_nodes(node_type=Node) == []
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_multi_document_rollback_keeps_one_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.engine_core.subsystems.rollback import RollbackSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []
    rollback = RollbackSubsystem(SimpleNamespace())
    result = {
        "deleted_docs": 0,
        "deleted_node_ids": [],
        "deleted_edge_ids": [],
        "updated_edges": 0,
    }

    def rollback_one(_document_id: str) -> dict[str, object]:
        observed.append(runtime_implementation_mode())
        return result

    class SelectorChangingIds(list[str]):
        def __iter__(self) -> Iterator[str]:
            monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
            return super().__iter__()

    monkeypatch.setattr(rollback, "rollback_document", rollback_one)
    assert rollback.rollback_many_documents(SelectorChangingIds(["doc-1"])) == {
        "deleted_docs": 1,
        "deleted_nodes": 0,
        "deleted_edges": 0,
        "updated_edges": 0,
    }
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_persist_graph_keeps_one_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.engine_core.subsystems.persist import PersistSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []
    persist = PersistSubsystem(SimpleNamespace())

    def preflight(_parsed: object, *, alias_key: str) -> None:
        assert alias_key == "run-1"
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")

    def build_deps(_parsed: object) -> tuple[list[str], dict[str, str], dict[str, object]]:
        observed.append(runtime_implementation_mode())
        return [], {}, {}

    monkeypatch.setattr(persist, "preflight_validate", preflight)
    monkeypatch.setattr(persist, "_alloc_real_ids", lambda _parsed: None)
    monkeypatch.setattr(persist, "build_deps", build_deps)

    result = persist.persist_graph(parsed=SimpleNamespace(), session_id="run-1")

    assert result["nodes_added"] == result["edges_added"] == 0
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_acl_write_keeps_one_selector_snapshot_across_unit_of_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    from kogwistar.engine_core.subsystems.acl import ACLSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []

    class ACLGraph:
        def bind_loaders(self, **_values: object) -> None:
            return None

        def invalidate(self, **_values: object) -> None:
            return None

    @contextmanager
    def unit_of_work():
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
        yield

    engine = SimpleNamespace(
        acl_graph=ACLGraph(),
        metadata=SimpleNamespace(lock_entity_identity=lambda **_values: None),
        raw_write=SimpleNamespace(
            add_node=lambda _node: observed.append(runtime_implementation_mode()),
            add_edge=lambda _edge: None,
        ),
        uow=unit_of_work,
        _emit_change=lambda **_values: None,
        namespace="default",
        kg_graph_type="knowledge",
    )
    acl = ACLSubsystem(engine)
    monkeypatch.setattr(acl, "_latest_persisted_acl_record", lambda **_values: None)
    monkeypatch.setattr(acl, "_build_acl_record_node", lambda _record: object())

    record = acl.record_acl(
        truth_graph="knowledge",
        entity_id="node-1",
        version=1,
        mode="private",
    )

    assert record.target.entity_id == "node-1"
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_acl_aware_write_facade_keeps_snapshot_across_acl_and_raw_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    from kogwistar.engine_core.subsystems.acl import ACLAwareWriteSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []

    @contextmanager
    def unit_of_work():
        monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
        yield

    class ACL:
        def writes_can_share_backend_transaction(self) -> bool:
            return True

        def record_default_acl_for_item(self, _item: object, *, grain: str) -> None:
            observed.append(f"acl:{grain}:{runtime_implementation_mode()}")

    engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection(),
        acl=ACL(),
        uow=unit_of_work,
    )
    raw_write = SimpleNamespace(
        add_node=lambda _node, **_kwargs: observed.append(
            f"write:{runtime_implementation_mode()}"
        )
    )
    write = ACLAwareWriteSubsystem(engine, raw_write)

    write.add_node(SimpleNamespace())

    assert observed == ["write:python", "acl:node:python"]
    assert runtime_implementation_mode() == "rust"


def test_acl_aware_read_keeps_one_selector_snapshot_through_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.engine_core.subsystems.acl import ACLAwareReadSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []
    entity = object()

    class RawRead:
        def get_nodes(self, **_values: object) -> list[object]:
            monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
            return [entity]

    acl_read = ACLAwareReadSubsystem(SimpleNamespace(), RawRead())
    monkeypatch.setattr(
        acl_read,
        "_node_visible",
        lambda _node: observed.append(runtime_implementation_mode()) or True,
    )

    assert acl_read.get_nodes() == [entity]
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_multi_page_ingest_keeps_one_selector_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.engine_core.subsystems.ingest import IngestSubsystem

    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "python")
    observed: list[str] = []

    def ingest_page(**_values: object) -> dict[str, object]:
        observed.append(runtime_implementation_mode())
        if len(observed) == 1:
            monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")
        return {"nodes_added": 0, "edges_added": 0, "raw": None}

    ingest = IngestSubsystem(SimpleNamespace(_ingest_text_with_llm=ingest_page))
    result = ingest.add_page(
        document_id="document-1",
        page_text={
            "pages": [
                {"page_number": 1, "text": "first"},
                {"page_number": 2, "text": "second"},
            ]
        },
    )

    assert result["nodes_added"] == result["edges_added"] == 0
    assert observed == ["python", "python"]
    assert runtime_implementation_mode() == "rust"


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


def test_engine_context_uses_the_owning_engine_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    owner = SimpleNamespace(
        kg_graph_type="knowledge",
        engine_id=None,
        _implementation_selection_snapshot=capture_implementation_selection(),
    )

    @engine_context
    def boundary(engine: SimpleNamespace) -> str:
        return runtime_implementation_mode()

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    assert boundary(owner) == "python"
    assert runtime_implementation_mode() == "rust"


def test_write_subsystem_proxy_uses_the_owning_engine_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    engine = SimpleNamespace(
        _implementation_selection_snapshot=capture_implementation_selection()
    )
    write = WriteSubsystem(engine)

    observed: list[str] = []
    write._add_node_impl = lambda *_args, **_kwargs: observed.append(
        runtime_implementation_mode()
    )

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    write.add_node(SimpleNamespace())
    assert observed == ["python"]
    assert runtime_implementation_mode() == "rust"


def test_transaction_authority_scope_rejects_nested_owner_switch() -> None:
    assert active_transaction_authority() is None


def test_callback_execution_scope_fails_closed_inside_authoritative_transaction() -> None:
    with transaction_authority_scope("python"):
        with pytest.raises(
            RuntimeError,
            match="callback execution is not allowed while a transaction is active",
        ):
            with callback_execution_scope():
                pass
    with callback_execution_scope():
        assert active_transaction_authority() is None


def test_worker_callback_cannot_open_transaction_authority() -> None:
    with callback_execution_scope():
        with pytest.raises(
            RuntimeError,
            match="transaction authority cannot be opened from a worker callback",
        ):
            with transaction_authority_scope("python"):
                pytest.fail("worker callbacks must return effects, not write directly")

    with transaction_authority_scope("python"):
        assert active_transaction_authority() == "python"


@pytest.mark.asyncio
async def test_callback_guard_propagates_through_asyncio_to_thread(
    tmp_path: Path,
) -> None:
    store = EngineSQLite(tmp_path / "callback-to-thread.sqlite")
    store.ensure_initialized()
    store.set_user_seq("callback-user", 7)

    with callback_execution_scope():
        with pytest.raises(
            RuntimeError,
            match="transaction authority cannot be opened from a worker callback",
        ):
            await asyncio.to_thread(store.set_user_seq, "callback-user", 8)

    assert store.current_user_seq("callback-user") == 7


def test_retained_sqlite_uow_connection_cannot_write_after_owner_scope(
    tmp_path: Path,
) -> None:
    store = EngineSQLite(tmp_path / "retained-uow.sqlite")
    store.ensure_initialized()

    with store.transaction() as conn:
        conn.execute("CREATE TABLE retained_handle_probe (value TEXT NOT NULL)")
        retained_conn = conn

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        retained_conn.execute(
            "INSERT INTO retained_handle_probe (value) VALUES ('late')"
        )

    with store.transaction() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM retained_handle_probe"
        ).fetchone()[0]
    assert count == 0


@pytest.mark.parametrize("async_dispatch", [False, True])
def test_framework_hook_thread_preserves_callback_write_guard(
    tmp_path: Path, async_dispatch: bool
) -> None:
    store = EngineSQLite(tmp_path / f"hook-callback-{async_dispatch}.sqlite")
    store.ensure_initialized()
    store.set_user_seq("callback-user", 7)

    hooks = HookRegistry()
    hooks.register(
        HookSpec(
            hook_id="attempt-write",
            callback=lambda _payload: store.set_user_seq("callback-user", 8),
            failure_mode="fail_closed",
        )
    )

    with callback_execution_scope():
        if async_dispatch:
            with pytest.raises(
                RuntimeError,
                match="transaction authority cannot be opened from a worker callback",
            ):
                asyncio.run(hooks.arun({}))
        else:
            with pytest.raises(
                RuntimeError,
                match="transaction authority cannot be opened from a worker callback",
            ):
                hooks.run({})

    assert store.current_user_seq("callback-user") == 7


@pytest.mark.parametrize("async_dispatch", [False, True])
def test_timed_out_framework_hook_keeps_guard_after_parent_scope_exits(
    tmp_path: Path, async_dispatch: bool
) -> None:
    store = EngineSQLite(tmp_path / f"timed-out-hook-{async_dispatch}.sqlite")
    store.ensure_initialized()
    store.set_user_seq("callback-user", 7)
    callback_started = threading.Event()
    release_callback = threading.Event()
    callback_finished = threading.Event()
    callback_errors: list[str] = []

    def callback(_payload: Mapping[str, JsonValue]) -> None:
        callback_started.set()
        assert release_callback.wait(timeout=2)
        try:
            store.set_user_seq("callback-user", 8)
        except RuntimeError as error:
            callback_errors.append(str(error))
        finally:
            callback_finished.set()

    hooks = HookRegistry()
    hooks.register(
        HookSpec(
            hook_id="delayed-write",
            callback=callback,
            timeout_ms=20,
            failure_mode="fail_open",
        )
    )

    try:
        with callback_execution_scope():
            if async_dispatch:
                results = asyncio.run(hooks.arun({}))
            else:
                results = hooks.run({})
        assert callback_started.is_set()
        assert results[0].status == "failed"
        release_callback.set()
        assert callback_finished.wait(timeout=2)
        assert callback_errors == [
            "transaction authority cannot be opened from a worker callback"
        ]
        assert store.current_user_seq("callback-user") == 7
    finally:
        release_callback.set()


def test_callback_execution_marker_is_cleared_after_callback_failure() -> None:
    with pytest.raises(ValueError, match="callback failure"):
        with callback_execution_scope():
            raise ValueError("callback failure")

    with transaction_authority_scope("python"):
        assert active_transaction_authority() == "python"


def test_engine_unit_of_work_cannot_be_opened_from_worker_callback(
    tmp_path: Path,
) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=ConstantEmbeddingFunction(dim=4),
        backend_factory=build_in_memory_backend,
    )
    expected = "transaction authority cannot be opened from a worker callback"

    with callback_execution_scope():
        with pytest.raises(RuntimeError, match=expected):
            with engine.uow():
                pytest.fail("callback must not enter the engine UoW")


def test_backend_transactions_cannot_be_opened_from_worker_callback(
    tmp_path: Path,
) -> None:
    class EngineWithoutTransactions:
        def begin(self) -> None:
            pytest.fail("backend transaction started before callback guard")

    contexts = (
        EngineSQLite(tmp_path / "guarded.sqlite").transaction(),
        InMemoryMetaStore().transaction(),
        PostgresUnitOfWork(engine=EngineWithoutTransactions()).transaction(),
    )
    expected = "transaction authority cannot be opened from a worker callback"

    for context in contexts:
        with callback_execution_scope():
            with pytest.raises(RuntimeError, match=expected):
                with context:
                    pytest.fail("backend transaction must not start in callback")


def test_raw_sqlite_connection_cannot_write_inside_worker_callback(
    tmp_path: Path,
) -> None:
    store = EngineSQLite(tmp_path / "raw-sqlite-callback.sqlite")
    store.ensure_initialized()
    store.set_user_seq("callback-user", 7)
    setup_connection = sqlite3.connect(store.db_path)
    try:
        setup_connection.execute("PRAGMA journal_mode=DELETE")
    finally:
        setup_connection.close()

    with callback_execution_scope():
        with store.connect() as connection:
            row = connection.execute(
                "SELECT value FROM user_seq WHERE user_id = ?",
                ("callback-user",),
            ).fetchone()
            assert row[0] == 7

            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                connection.execute(
                    "UPDATE user_seq SET value = 8 WHERE user_id = ?",
                    ("callback-user",),
                )
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                connection.execute("BEGIN")

    leaked_callback_connection: sqlite3.Connection | None = None
    with callback_execution_scope():
        leaked_callback_connection = store.connect()
    assert leaked_callback_connection is not None
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            leaked_callback_connection.execute(
                "UPDATE user_seq SET value = 8 WHERE user_id = ?",
                ("callback-user",),
            )
    finally:
        leaked_callback_connection.close()

    inspection_connection = sqlite3.connect(store.db_path)
    try:
        journal_mode = inspection_connection.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
    finally:
        inspection_connection.close()
    assert journal_mode.lower() == "delete"

    assert store.current_user_seq("callback-user") == 7
    store.set_user_seq("callback-user", 8)
    assert store.current_user_seq("callback-user") == 8


def test_in_memory_meta_connection_cannot_mutate_from_worker_callback() -> None:
    store = InMemoryMetaStore()
    store.enqueue_index_job(
        job_id="callback-job",
        namespace="ns",
        entity_kind="node",
        entity_id="node-1",
        index_kind="node_docs",
        op="UPSERT",
    )
    claimed = store.claim_index_jobs(limit=1, namespace="ns")
    assert len(claimed) == 1
    lease_until = claimed[0].lease_until
    assert lease_until is not None

    with callback_execution_scope():
        with store.connect() as connection:
            with pytest.raises(
                RuntimeError,
                match="transaction authority cannot be opened from a worker callback",
            ):
                connection.execute(
                    "UPDATE INDEX_JOBS SET LEASE_UNTIL = 0 WHERE JOB_ID = ?",
                    ("callback-job",),
                )

    row = store.list_index_jobs(namespace="ns", limit=1)[0]
    assert row.lease_until == lease_until


def test_backend_collection_mutations_cannot_run_from_worker_callback() -> None:
    class PostgresBackendWithoutIO:
        _is_async_engine = False

        def _upsert(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("PostgreSQL write started before callback guard")

    memory = InMemoryBackend(engine=SimpleNamespace())
    postgres = PgCollectionFacade(
        backend=PostgresBackendWithoutIO(),  # type: ignore[arg-type]
        table=SimpleNamespace(),  # type: ignore[arg-type]
        spec=CollectionSpec(vector=True),
    )
    writes = (
        lambda: memory.node_add(ids=["n"], documents=["node"], metadatas=[{}]),
        lambda: memory.node_upsert(ids=["n"], documents=["node"], metadatas=[{}]),
        lambda: memory.node_update(ids=["n"], metadatas=[{"title": "x"}]),
        lambda: memory.node_delete(ids=["n"]),
        lambda: memory.node_clear_embeddings(ids=["n"]),
        lambda: postgres.add(ids=["n"], documents=["node"], metadatas=[{}]),
        lambda: postgres.upsert(ids=["n"], documents=["node"], metadatas=[{}]),
        lambda: postgres.update(ids=["n"], metadatas=[{"title": "x"}]),
        lambda: postgres.delete(ids=["n"]),
    )
    expected = "transaction authority cannot be opened from a worker callback"

    for write in writes:
        with callback_execution_scope():
            with pytest.raises(RuntimeError, match=expected):
                write()

    assert memory.node.count() == 0


def test_chroma_collection_mutations_cannot_run_from_worker_callback() -> None:
    from kogwistar.engine_core.chroma_backend import (
        AsyncChromaBackend,
        ChromaBackend,
    )

    calls: list[dict[str, object]] = []

    async def upsert(**kwargs: object) -> None:
        calls.append(kwargs)

    collection = SimpleNamespace(
        upsert=upsert,
        get=lambda **_kwargs: {"ids": []},
    )
    backend = object.__new__(ChromaBackend)
    backend._collections = {"node": collection}
    async_backend = object.__new__(AsyncChromaBackend)
    async_backend._collections = {"node": collection}

    backend.call("node", "upsert", ids=["outside-callback"])
    assert len(calls) == 1
    assert backend.call("node", "get")["ids"] == []  # type: ignore[index]

    expected = "Chroma collection mutation is not allowed from a worker callback"
    with callback_execution_scope():
        with pytest.raises(RuntimeError, match=expected):
            backend.call("node", "upsert", ids=["blocked"])

        async def attempt_async_write() -> None:
            with pytest.raises(RuntimeError, match=expected):
                await async_backend.async_call("node", "upsert", ids=["blocked"])

        asyncio.run(attempt_async_write())

    assert len(calls) == 1


def test_native_store_bridges_cannot_be_called_from_worker_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kogwistar import _rust_bridge
    from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite

    native_calls: list[str] = []
    monkeypatch.setattr(
        _rust_bridge,
        "_load_extension",
        lambda: native_calls.append("loaded"),
    )
    operations = (
        lambda: _rust_bridge.store_sqlite(
            path=tmp_path / "guarded.sqlite", operation={"kind": "add_node"}
        ),
        lambda: RustEngineSQLite(tmp_path, "guarded-facade.sqlite").set_user_seq(
            "callback-user", 1
        ),
        lambda: _rust_bridge.store_postgres(
            dsn="postgresql://example/db",
            schema="test",
            operation={"kind": "upsert_node"},
        ),
    )
    expected = "transaction authority cannot be opened from a worker callback"

    for operation in operations:
        with callback_execution_scope():
            with pytest.raises(RuntimeError, match=expected):
                operation()

    with callback_execution_scope():
        with pytest.raises(RuntimeError, match=expected):
            with RustEngineSQLite(tmp_path, "guarded-transaction.sqlite").transaction():
                pytest.fail("Rust SQLite facade must reject callback transaction entry")

    assert native_calls == []


@pytest.mark.requires_rust
def test_native_extension_store_entrypoints_reject_worker_callback() -> None:
    from kogwistar import _rust

    entrypoints = (
        (_rust.store_sqlite_json, ("{}",)),
        (_rust.store_postgres_json, ("{}",)),
        (_rust.api_run_server, ()),
    )
    expected = "transaction authority cannot be opened from a worker callback"
    for entrypoint, args in entrypoints:
        with callback_execution_scope():
            with pytest.raises(RuntimeError, match=expected):
                entrypoint(*args)


def test_postgres_sql_write_primitives_cannot_be_called_from_worker_callback() -> None:
    backend = object.__new__(PgVectorBackend)
    writes = (
        lambda: backend._upsert(
            object(), ids=[], documents=[], metadatas=[]  # type: ignore[arg-type]
        ),
        lambda: backend.stage1_projection_upsert(
            namespace="ns",
            entity_kind="node",
            entity_id="n",
            document="{}",
            metadata={},
            source_fingerprint="fp",
        ),
    )
    expected = "transaction authority cannot be opened from a worker callback"

    for write in writes:
        with callback_execution_scope():
            with pytest.raises(RuntimeError, match=expected):
                write()


def test_transaction_authority_scope_matches_frozen_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)

    with implementation_selection_scope():
        with pytest.raises(
            RuntimeError,
            match="does not match frozen implementation selection",
        ):
            with transaction_authority_scope("python"):
                pass
        with transaction_authority_scope("rust"):
            assert active_transaction_authority() == "rust"


def test_python_transaction_entry_fails_closed_for_rust_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)

    with implementation_selection_scope():
        with pytest.raises(RuntimeError, match="frozen Rust transaction authority"):
            assert_python_transaction_entry_allowed()

    with transaction_authority_scope("rust"):
        with pytest.raises(RuntimeError, match="while Rust owns transaction authority"):
            assert_python_transaction_entry_allowed()

    with pytest.raises(RuntimeError, match="frozen Rust transaction authority"):
        assert_python_transaction_entry_allowed()


def test_rust_transaction_authority_propagates_to_asyncio_tasks_and_threads() -> None:
    def assert_python_entry_denied() -> None:
        with pytest.raises(
            RuntimeError,
            match="forbidden while Rust owns transaction authority",
        ):
            assert_python_transaction_entry_allowed()

    async def child_task() -> None:
        assert_python_entry_denied()

    async def verify_propagation() -> None:
        with transaction_authority_scope("rust"):
            await asyncio.create_task(child_task())
            await asyncio.to_thread(assert_python_entry_denied)

    asyncio.run(verify_propagation())


def test_python_transaction_entry_remains_available_for_python_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")
    with implementation_selection_scope():
        assert_python_transaction_entry_allowed()


def test_python_postgres_uow_never_begins_under_rust_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EngineWithoutTransactions:
        def begin(self) -> None:
            pytest.fail("Python SQLAlchemy transaction started under Rust authority")

    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)
    with implementation_selection_scope():
        uow = PostgresUnitOfWork(engine=EngineWithoutTransactions())  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="frozen Rust transaction authority"):
            with uow.transaction():
                pytest.fail("Python UoW must not enter the transaction")


def test_raw_sqlalchemy_transaction_cannot_nest_inside_rust_owner() -> None:
    import sqlalchemy as sa

    engine = sa.create_engine("sqlite://")
    EnginePostgresMetaStore(engine=engine)
    try:
        with callback_execution_scope():
            with pytest.raises(
                RuntimeError,
                match="transaction authority cannot be opened from a worker callback",
            ):
                with engine.begin():
                    pytest.fail("raw SQLAlchemy transaction began inside callback")

        with transaction_authority_scope("rust"):
            with pytest.raises(RuntimeError, match="Rust owns the active unit"):
                with engine.begin():
                    pytest.fail("SQLAlchemy transaction began under Rust authority")

        with transaction_authority_scope("python"):
            with engine.begin():
                pass
    finally:
        engine.dispose()


def test_python_sqlite_and_memory_transactions_reject_rust_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)
    contexts = (
        EngineSQLite(tmp_path / "python-meta.sqlite").transaction(),
        InMemoryMetaStore().transaction(),
    )
    sqlite_store = EngineSQLite(tmp_path / "raw-python-meta.sqlite")
    with implementation_selection_scope():
        with pytest.raises(RuntimeError, match="frozen Rust transaction authority"):
            sqlite_store.connect()
        for context in contexts:
            with pytest.raises(RuntimeError, match="frozen Rust transaction authority"):
                with context:
                    pytest.fail("Python metadata transaction started under Rust selection")


@pytest.mark.requires_rust
def test_engine_rejects_python_metadata_under_rust_transaction_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)

    def mismatched_backend(engine):
        backend = build_in_memory_backend(engine)
        engine.metadata = InMemoryMetaStore()
        return backend

    with pytest.raises(
        RuntimeError,
        match="metadata transaction implementation does not match",
    ):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            embedding_function=ConstantEmbeddingFunction(dim=4),
            backend_factory=mismatched_backend,
        )


@pytest.mark.asyncio
async def test_rust_async_uow_uses_native_metadata_transaction_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RustMetadata:
        @contextmanager
        def transaction(self):
            assert active_transaction_authority() == "rust"
            yield object()

    class PythonAsyncUow:
        def transaction(self):
            pytest.fail("Python async UoW entered under Rust transaction authority")

    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)
    engine = object.__new__(GraphKnowledgeEngine)
    engine.metadata = RustMetadata()
    engine._async_backend_uow = PythonAsyncUow()
    engine._implementation_selection_snapshot = capture_implementation_selection()

    async with engine.async_uow():
        assert active_transaction_authority() == "rust"


def test_async_projection_transaction_uses_engine_authority_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")

    class Backend:
        _is_async_engine = True

    class Uow:
        @asynccontextmanager
        async def transaction(self):
            yield

    engine = SimpleNamespace(
        backend=Backend(),
        _async_backend_uow=Uow(),
        _implementation_selection_snapshot=capture_implementation_selection(),
    )
    adapter = AsyncPostgresTwoStageProjectionAdapter(engine)

    with adapter._transaction_authority_scope():
        assert active_transaction_authority() == "rust"
    assert active_transaction_authority() is None
    with transaction_authority_scope("python"):
        assert active_transaction_authority() == "python"
        with transaction_authority_scope("python"):
            assert active_transaction_authority() == "python"
        with pytest.raises(RuntimeError, match="different implementation authority"):
            with transaction_authority_scope("rust"):
                pass
    assert active_transaction_authority() is None


def test_resolve_implementation_selection_rejects_split_authority() -> None:
    python_snapshot = (("KOGWISTAR_IMPL_MODE", "python"),)
    rust_snapshot = (("KOGWISTAR_IMPL_MODE", "rust"),)

    assert resolve_implementation_selection(
        SimpleNamespace(_implementation_selection_snapshot=python_snapshot),
        SimpleNamespace(),
    ) == python_snapshot
    with pytest.raises(ValueError, match="different implementation selections"):
        resolve_implementation_selection(
            SimpleNamespace(_implementation_selection_snapshot=python_snapshot),
            SimpleNamespace(_implementation_selection_snapshot=rust_snapshot),
        )


def test_resolve_implementation_selection_includes_subsystem_owner_snapshot() -> None:
    python_snapshot = (("KOGWISTAR_IMPL_MODE", "python"),)
    rust_snapshot = (("KOGWISTAR_IMPL_MODE", "rust"),)
    engine = SimpleNamespace(_implementation_selection_snapshot=python_snapshot)
    subsystem = SimpleNamespace(_e=engine)
    other_engine = SimpleNamespace(_implementation_selection_snapshot=rust_snapshot)

    assert resolve_implementation_selection(subsystem, engine) == python_snapshot
    with pytest.raises(ValueError, match="different implementation selections"):
        resolve_implementation_selection(subsystem, other_engine)


def test_indexing_boundary_uses_owning_engine_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Meta:
        def enqueue_index_job(self, **_: object) -> str:
            return runtime_implementation_mode()

    snapshot = capture_implementation_selection()
    engine = SimpleNamespace(
        _implementation_selection_snapshot=snapshot,
        _phase1_enable_index_jobs=True,
        metadata=Meta(),
        namespace="default",
    )
    subsystem = IndexingSubsystem(engine=engine)
    monkeypatch.setenv("KOGWISTAR_IMPL_MODE", "rust")

    assert subsystem.enqueue_index_job(
        entity_kind="node",
        entity_id="node-1",
        index_kind="node_embedding",
        op="UPSERT",
    ) == "python"


def test_indexing_subsystem_rejects_legacy_only_metadata_alias() -> None:
    engine = cast(
        GraphKnowledgeEngine,
        SimpleNamespace(
            meta_sqlite=object(),
            _phase1_enable_index_jobs=True,
            namespace="default",
        ),
    )
    subsystem = IndexingSubsystem(engine=engine)

    with pytest.raises(TypeError, match="backend-neutral engine.metadata"):
        subsystem.enqueue_index_job(
            entity_kind="node",
            entity_id="node-1",
            index_kind="node_embedding",
            op="UPSERT",
        )


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
