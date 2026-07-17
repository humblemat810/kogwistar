from __future__ import annotations

from pathlib import Path

import pytest

from kogwistar.engine_core.rust_meta_sqlite import (
    RustEngineSQLite,
    RustSQLiteConnectionUnavailable,
    build_sqlite_meta_store,
)
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import InMemoryBackend, _DummyLock
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
