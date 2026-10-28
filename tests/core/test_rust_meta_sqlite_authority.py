from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from kogwistar._rust_bridge import RustParityError
from kogwistar.agent import (
    DurableCatalogStore,
    DurableSkillCatalogMaterializer,
    DurableSkillProjectionStore,
)
from kogwistar.agent.skills import parse_skill_text
from kogwistar.engine_core import rust_meta_sqlite
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.event_envelope import EntityEventEnvelope
from kogwistar.engine_core.in_memory_backend import InMemoryBackend, _DummyLock
from kogwistar.engine_core.rust_meta_sqlite import (
    RustEngineSQLite,
    RustSQLiteConnectionUnavailable,
    _RustTransactionToken,
    build_sqlite_meta_store,
)
from tests._helpers.graph_builders import build_entity_node

pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension():
    return pytest.importorskip("kogwistar._rust")


def test_public_selector_keeps_python_rollback_and_routes_rust_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    assert type(build_sqlite_meta_store(tmp_path)).__name__ == "EngineSQLite"
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "shadow")
    assert type(build_sqlite_meta_store(tmp_path)).__name__ == "EngineSQLite"
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    assert isinstance(build_sqlite_meta_store(tmp_path), RustEngineSQLite)


def test_rust_authority_marks_calls_as_sole_writer_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _store_sqlite(**kwargs: object) -> dict[str, bool]:
        calls.append(kwargs)
        return {"initialized": True}

    monkeypatch.setattr(rust_meta_sqlite, "store_sqlite", _store_sqlite)
    RustEngineSQLite(tmp_path).ensure_initialized()

    assert calls == [
        {
            "path": tmp_path / "engine.db",
            "operation": {"kind": "open_init"},
            "transaction_id": None,
            "reuse_session": False,
        }
    ]


def test_rust_authority_uow_is_atomic_and_raw_python_writer_is_closed(
    tmp_path: Path,
) -> None:
    store = RustEngineSQLite(tmp_path)
    store.ensure_initialized()
    with pytest.raises(RustSQLiteConnectionUnavailable):
        store.connect()

    with store.transaction() as transaction:
        assert store.next_global_seq_conn(transaction) == 1
        assert store.append_entity_event(
            namespace="authority",
            event_id="committed",
            entity_kind="node",
            entity_id="n1",
            op="UPSERT",
            payload_json="{}",
        ) == 1
    assert store.current_global_seq() == 1
    assert [row[0] for row in store.iter_entity_events(namespace="authority")] == [1]

    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction():
            assert store.next_global_seq() == 2
            store.append_entity_event(
                namespace="authority",
                event_id="rolled-back",
                entity_kind="node",
                entity_id="n2",
                op="UPSERT",
                payload_json="{}",
            )
            raise RuntimeError("rollback")
    assert store.current_global_seq() == 1
    assert [row[0] for row in store.iter_entity_events(namespace="authority")] == [1]


def test_rust_sqlite_identity_lock_requires_native_transaction(tmp_path: Path) -> None:
    store = RustEngineSQLite(tmp_path, "identity-lock.sqlite")
    store.ensure_initialized()

    with pytest.raises(
        RustSQLiteConnectionUnavailable, match="active Rust SQLite transaction"
    ):
        store.lock_entity_identity(
            namespace="ns", entity_kind="node", entity_id="node-1"
        )

    with store.transaction():
        store.lock_entity_identity(
            namespace="ns", entity_kind="node", entity_id="node-1"
        )


def test_sqlite_transaction_token_cannot_cross_facade_owner(
    tmp_path: Path,
) -> None:
    first = RustEngineSQLite(tmp_path, "owner.sqlite")
    second = RustEngineSQLite(tmp_path, "owner.sqlite")
    token = _RustTransactionToken("same-transaction", first._authority_id)
    active = second._transaction_id.set(token.value)
    try:
        with pytest.raises(RustSQLiteConnectionUnavailable, match="stale"):
            second._require_token(token)
    finally:
        second._transaction_id.reset(active)


def test_rust_open_during_active_write_reads_previous_committed_projection(
    tmp_path: Path,
) -> None:
    store = RustEngineSQLite(tmp_path, "concurrent-read.sqlite")
    store.ensure_initialized()
    store.replace_named_projection(
        "concurrent",
        "head",
        {"revision": 0},
        last_authoritative_seq=0,
        last_materialized_seq=0,
        projection_schema_version=1,
        materialization_status="ready",
    )
    transaction_open = Event()
    reader_done = Event()

    def write_uncommitted() -> None:
        with store.transaction():
            seq = store.append_entity_event(
                namespace="concurrent",
                event_id="uncommitted",
                entity_kind="node",
                entity_id="n1",
                op="UPSERT",
                payload_json="{}",
            )
            store.replace_named_projection(
                "concurrent",
                "head",
                {"revision": 1},
                last_authoritative_seq=seq,
                last_materialized_seq=seq,
                projection_schema_version=1,
                materialization_status="ready",
            )
            transaction_open.set()
            if not reader_done.wait(timeout=5):
                raise AssertionError("reader could not finish before writer commit")

    def read_committed() -> dict[str, object] | None:
        if not transaction_open.wait(timeout=5):
            raise AssertionError("writer did not open its transaction")
        try:
            return store.get_named_projection("concurrent", "head")
        finally:
            reader_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        writer = executor.submit(write_uncommitted)
        reader = executor.submit(read_committed)
        projection = reader.result(timeout=7)
        writer.result(timeout=7)

    assert projection is not None
    assert projection["payload"] == {"revision": 0}
    assert store.get_named_projection("concurrent", "head")["payload"] == {
        "revision": 1
    }


def test_rust_session_reuses_connection_across_reads_and_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_store_sqlite(**kwargs: object) -> object:
        calls.append(kwargs)
        operation = kwargs["operation"]
        assert isinstance(operation, dict)
        if operation["kind"] == "next_global_seq":
            return 1
        return None

    monkeypatch.setattr(rust_meta_sqlite, "store_sqlite", fake_store_sqlite)
    store = RustEngineSQLite(tmp_path, "session.sqlite")

    with store.session():
        assert store.next_global_seq() == 1
        with store.transaction():
            assert store.next_global_seq() == 1
        assert store.next_global_seq() == 1
        assert calls[-1]["operation"] == {"kind": "next_global_seq"}

    assert [call["operation"] for call in calls] == [
        {"kind": "open_init"},
        {"kind": "next_global_seq"},
        {"kind": "begin_transaction"},
        {"kind": "next_global_seq"},
        {"kind": "commit_transaction"},
        {"kind": "next_global_seq"},
        {"kind": "close"},
    ]
    assert all(call["reuse_session"] is True for call in calls)


@pytest.mark.asyncio
async def test_rust_async_transaction_cancelled_waiter_does_not_leak_lock(
    tmp_path: Path,
) -> None:
    store = RustEngineSQLite(tmp_path, "async-transaction-lock.sqlite")
    store.ensure_initialized()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_transaction() -> None:
        async with store.async_transaction():
            holder_entered.set()
            await release_holder.wait()

    async def wait_for_transaction() -> None:
        async with store.async_transaction():
            raise AssertionError("cancelled waiter must not enter the transaction")

    holder = asyncio.create_task(hold_transaction())
    await holder_entered.wait()
    waiter = asyncio.create_task(wait_for_transaction())
    await asyncio.sleep(0.02)
    waiter.cancel()
    release_holder.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await holder

    async with store.async_transaction():
        assert store.transaction_active
    assert not store.transaction_active
    store.close()


def test_rust_batch_facade_uses_one_native_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_store_sqlite(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        return [1, None]

    monkeypatch.setattr(rust_meta_sqlite, "store_sqlite", fake_store_sqlite)
    store = RustEngineSQLite(tmp_path, "batch.sqlite")
    operations = [{"kind": "next_global_seq"}, {"kind": "current_global_seq"}]

    assert store.execute_batch(operations) == [1, None]
    assert calls == [
        {
            "path": tmp_path / "batch.sqlite",
            "operation": {
                "kind": "batch",
                "operations": operations,
                "abort": False,
            },
            "transaction_id": None,
            "reuse_session": False,
        }
    ]


def test_rust_batch_facade_empty_input_does_not_cross_native_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_store_sqlite(**kwargs: object) -> object:
        raise AssertionError(f"empty batch unexpectedly called Rust: {kwargs}")

    monkeypatch.setattr(rust_meta_sqlite, "store_sqlite", fail_store_sqlite)
    assert RustEngineSQLite(tmp_path).execute_batch([]) == []


def test_rust_batch_facade_rejects_nested_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = RustEngineSQLite(tmp_path, "nested-batch.sqlite")
    token = store._transaction_id.set("active")
    try:
        with pytest.raises(RuntimeError, match="cannot be nested"):
            store.execute_batch([{"kind": "next_global_seq"}])
    finally:
        store._transaction_id.reset(token)


def test_rust_batch_facade_preserves_atomic_rollback(tmp_path: Path) -> None:
    store = RustEngineSQLite(tmp_path, "batch-atomic.sqlite")
    store.ensure_initialized()
    operations = [
        {
            "kind": "raw_append",
            "namespace": "batch",
            "event_id": "committed",
            "entity_kind": "node",
            "entity_id": "n1",
            "op": "UPSERT",
            "payload_json": "{}",
        },
        {
            "kind": "replace_named_projection",
            "namespace": "batch",
            "key": "head",
            "payload": {"revision": 1},
            "last_authoritative_seq": 1,
            "last_materialized_seq": 1,
            "projection_schema_version": 1,
            "materialization_status": "ready",
        },
    ]

    result = store.execute_batch(operations)
    assert isinstance(result, list)
    assert store.get_latest_entity_event_seq(namespace="batch") == 1
    assert store.get_named_projection("batch", "head")["payload"] == {"revision": 1}

    aborted = [dict(operations[0], event_id="rolled-back"), dict(operations[1], key="rollback")]
    with pytest.raises(RustParityError) as exc:
        store.execute_batch(aborted, abort=True)
    assert exc.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert store.get_latest_entity_event_seq(namespace="batch") == 1
    assert store.get_named_projection("batch", "rollback") is None


def test_rust_authority_database_is_readable_after_python_rollback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rust = RustEngineSQLite(tmp_path, "rollback.sqlite")
    rust.ensure_initialized()
    rust.set_index_applied_fingerprint(
        namespace="ns", coalesce_key="node:n:doc", applied_fingerprint="rust"
    )
    rust.replace_named_projection(
        "projection",
        "key",
        {"owner": "rust"},
        last_authoritative_seq=1,
        last_materialized_seq=1,
        projection_schema_version=1,
        materialization_status="ready",
    )

    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    python = build_sqlite_meta_store(tmp_path, "rollback.sqlite")
    assert python.get_index_applied_fingerprint(
        namespace="ns", coalesce_key="node:n:doc"
    ) == "rust"
    assert python.get_named_projection("projection", "key")["payload"] == {
        "owner": "rust"
    }


def test_rust_authority_named_projection_batch_cas_is_atomic(tmp_path: Path) -> None:
    store = RustEngineSQLite(tmp_path, "batch-cas.sqlite")
    store.ensure_initialized()
    updates = [
        {
            "namespace": "skills",
            "key": "provider",
            "payload": {"owner": "rust"},
            "expected_last_authoritative_seq": None,
            "expected_last_materialized_seq": None,
            "last_authoritative_seq": 1,
            "last_materialized_seq": 1,
            "projection_schema_version": 1,
            "materialization_status": "ready",
        },
        {
            "namespace": "catalog",
            "key": "provider",
            "payload": {"owner": "rust"},
            "expected_last_authoritative_seq": 99,
            "expected_last_materialized_seq": 99,
            "last_authoritative_seq": 1,
            "last_materialized_seq": 1,
            "projection_schema_version": 1,
            "materialization_status": "ready",
        },
    ]
    assert not store.compare_and_swap_named_projections(updates)
    assert store.get_named_projection("skills", "provider") is None

    updates[1]["expected_last_authoritative_seq"] = None
    updates[1]["expected_last_materialized_seq"] = None
    assert store.compare_and_swap_named_projections(updates)
    assert store.get_named_projection("skills", "provider")["payload"] == {
        "owner": "rust"
    }
    assert store.get_named_projection("catalog", "provider")["payload"] == {
        "owner": "rust"
    }


def test_rust_authority_skill_projection_reconciles_after_partial_commit(
    tmp_path: Path,
) -> None:
    metadata = RustEngineSQLite(tmp_path, "skill-reconcile.sqlite")
    metadata.ensure_initialized()
    artifact = parse_skill_text(
        "# Deploy\n\n1. validate\n",
        provider_id="project",
        provider_local_id="deploy",
        skill_version="v1",
    ).model_copy(
        update={"projection_revision": 1, "tenant_id": "tenant-1", "project_id": "project-1"}
    )
    projections = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    materializer = DurableSkillCatalogMaterializer(
        projections=projections, catalog=catalog
    )
    _, prepared = projections.prepare_upsert_update(artifact)
    assert metadata.compare_and_swap_named_projections([prepared[0]])

    restarted_projections = DurableSkillProjectionStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    restarted_catalog = DurableCatalogStore(
        metadata,
        tenant_id="tenant-1",
        project_id="project-1",
        acl_enabled=False,
    )
    restarted = DurableSkillCatalogMaterializer(
        projections=restarted_projections, catalog=restarted_catalog
    )
    assert restarted_projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None
    assert restarted.reconcile() == 1
    assert restarted_projections.graph(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    assert restarted_catalog.get(
        "project:deploy", tenant_id="tenant-1", project_id="project-1"
    ) is not None
    restarted.remove_provider("project")
    assert restarted_projections.get(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None
    assert restarted_projections.graph(
        "project", "deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None
    assert restarted_catalog.get(
        "project:deploy", tenant_id="tenant-1", project_id="project-1"
    ) is None


def test_rust_authority_event_envelope_is_lossless_and_idempotent(tmp_path: Path) -> None:
    store = RustEngineSQLite(tmp_path, "envelope.sqlite")
    store.ensure_initialized()
    event = EntityEventEnvelope(
        namespace="archive/source",
        seq=1,
        event_id="event-1",
        entity_kind="source",
        entity_id="source-1",
        op="UPSERT",
        payload_json='{"title":"rust"}',
        created_at=123456789,
    )

    assert store.append_entity_event_envelope(event) == 1
    assert store.append_entity_event_envelope(event) == 1
    assert list(store.iter_entity_event_envelopes(namespace=event.namespace)) == [event]

    with pytest.raises(RustParityError, match="conflicts"):
        store.append_entity_event_envelope(
            EntityEventEnvelope(
                namespace=event.namespace,
                seq=1,
                event_id=event.event_id,
                entity_kind=event.entity_kind,
                entity_id=event.entity_id,
                op=event.op,
                payload_json='{"title":"changed"}',
                created_at=event.created_at,
            )
        )


def test_public_graph_engine_uses_rust_meta_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def backend_factory(engine):
        backend = InMemoryBackend(engine)
        engine.backend_kind = backend.backend_kind = "memory"
        names = (
            "node_index",
            "node",
            "edge",
            "edge_endpoints",
            "document",
            "domain",
            "node_docs",
            "node_refs",
            "edge_refs",
        )
        engine.collection_lock = {name: _DummyLock() for name in names}
        for name in names:
            setattr(engine, f"{name}_collection", getattr(backend, name))
        return backend

    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        kg_graph_type="knowledge",
        embedding_function=lambda texts: [[0.0, 1.0] for _ in texts],
        backend_factory=backend_factory,
    )
    assert isinstance(engine.meta_sqlite, RustEngineSQLite)
    with engine.uow():
        engine.meta_sqlite.append_entity_event(
            namespace="product",
            event_id="one",
            entity_kind="node",
            entity_id="n",
            op="UPSERT",
            payload_json="{}",
        )
        engine.meta_sqlite.enqueue_index_job(
            job_id="job",
            namespace="product",
            entity_kind="node",
            entity_id="n",
            index_kind="node_docs",
            op="UPSERT",
        )
    assert len(list(engine.meta_sqlite.iter_entity_events(namespace="product"))) == 1
    assert engine.meta_sqlite.list_index_jobs(namespace="product")[0].job_id == "job"

    node = build_entity_node(
        node_id="product-node", doc_id="product-doc", embedding=[0.0, 1.0]
    )
    engine.write.add_node(node)
    assert any(
        row[2] == "product-node"
        for row in engine.meta_sqlite.iter_entity_events(namespace=engine.namespace)
    )
