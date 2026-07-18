from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from kogwistar._rust_bridge import RustParityError, store_postgres
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    Document,
    Domain,
    Edge,
    GraphExtractionWithIDs,
    Grounding,
    Node,
    PureChromaEdge,
    PureChromaNode,
)
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.engine_core.rust_postgres_session import RustEnginePostgresMetaStore
from tests._helpers.graph_builders import (
    build_entity_node,
    build_relationship_edge,
    mk_document_span,
)


pytestmark = [pytest.mark.ci_full, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def _call(
    dsn: str,
    schema: str,
    operation: dict[str, Any],
    *,
    transaction_id: str | None = None,
) -> Any:
    return store_postgres(
        dsn=dsn,
        schema=schema,
        operation=operation,
        transaction_id=transaction_id,
    )


def _append(event_id: str) -> dict[str, Any]:
    return {
        "kind": "raw_append",
        "namespace": "native-session",
        "event_id": event_id,
        "entity_kind": "node",
        "entity_id": event_id,
        "op": "UPSERT",
        "payload_json": "{}",
    }


def test_native_postgres_session_commit_rollback_and_fail_closed(
    pg_dsn: str | None, pg_schema: str | None, sa_engine: Any
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    _call(dsn, schema, {"kind": "ensure_schema"})

    committed = uuid.uuid4().hex
    transaction_id = uuid.uuid4().hex
    assert _call(
        dsn,
        schema,
        {"kind": "begin_transaction"},
        transaction_id=transaction_id,
    ) is None
    assert _call(
        dsn,
        schema,
        _append(committed),
        transaction_id=transaction_id,
    )["seq"] == 1
    assert _call(
        dsn,
        schema,
        {"kind": "next_global_seq"},
        transaction_id=transaction_id,
    ) == 1
    assert _call(
        dsn,
        schema,
        {"kind": "next_user_seq", "user_id": "user"},
        transaction_id=transaction_id,
    ) == 1
    _call(
        dsn,
        schema,
        {
            "kind": "set_index_applied_fingerprint",
            "namespace": "native-session",
            "coalesce_key": "node:n1",
            "applied_fingerprint": "committed",
            "last_job_id": None,
        },
        transaction_id=transaction_id,
    )

    with pytest.raises(RustParityError) as second_writer:
        _call(dsn, schema, _append("no-owner"))
    assert second_writer.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"

    with pytest.raises(RustParityError) as second_transaction:
        _call(
            dsn,
            schema,
            {"kind": "begin_transaction"},
            transaction_id=uuid.uuid4().hex,
        )
    assert second_transaction.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"

    assert _call(
        dsn,
        schema,
        {"kind": "commit_transaction"},
        transaction_id=transaction_id,
    ) is None
    assert _call(
        dsn,
        schema,
        {"kind": "latest_retained_event_seq", "namespace": "native-session"},
    ) == 1
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    assert python.current_global_seq() == 1
    assert python.current_user_seq("user") == 1
    assert python.get_index_applied_fingerprint(
        namespace="native-session", coalesce_key="node:n1"
    ) == "committed"

    with pytest.raises(RustParityError) as stale:
        _call(
            dsn,
            schema,
            _append("stale"),
            transaction_id=transaction_id,
        )
    assert stale.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"

    rolled_back = uuid.uuid4().hex
    rollback_id = uuid.uuid4().hex
    _call(
        dsn,
        schema,
        {"kind": "begin_transaction"},
        transaction_id=rollback_id,
    )
    assert _call(
        dsn,
        schema,
        _append(rolled_back),
        transaction_id=rollback_id,
    )["seq"] == 2
    assert _call(
        dsn,
        schema,
        {"kind": "next_global_seq"},
        transaction_id=rollback_id,
    ) == 2
    _call(
        dsn,
        schema,
        {"kind": "set_user_seq", "user_id": "user", "value": 99},
        transaction_id=rollback_id,
    )
    _call(
        dsn,
        schema,
        {
            "kind": "set_index_applied_fingerprint",
            "namespace": "native-session",
            "coalesce_key": "node:n1",
            "applied_fingerprint": "rolled-back",
            "last_job_id": None,
        },
        transaction_id=rollback_id,
    )
    _call(
        dsn,
        schema,
        {"kind": "rollback_transaction"},
        transaction_id=rollback_id,
    )
    assert _call(
        dsn,
        schema,
        {"kind": "latest_retained_event_seq", "namespace": "native-session"},
    ) == 1
    assert python.current_global_seq() == 1
    assert python.current_user_seq("user") == 1
    assert python.get_index_applied_fingerprint(
        namespace="native-session", coalesce_key="node:n1"
    ) == "committed"


def test_native_postgres_sessions_are_isolated_by_schema(
    pg_dsn: str | None, pg_schema: str | None, sa_engine: Any
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    other_schema = f"{schema}_other"
    try:
        for selected in (schema, other_schema):
            _call(dsn, selected, {"kind": "ensure_schema"})
        first_id = uuid.uuid4().hex
        second_id = uuid.uuid4().hex
        _call(dsn, schema, {"kind": "begin_transaction"}, transaction_id=first_id)
        _call(
            dsn,
            other_schema,
            {"kind": "begin_transaction"},
            transaction_id=second_id,
        )
        _call(dsn, schema, _append("first"), transaction_id=first_id)
        _call(dsn, other_schema, _append("second"), transaction_id=second_id)
        _call(dsn, schema, {"kind": "commit_transaction"}, transaction_id=first_id)
        _call(
            dsn,
            other_schema,
            {"kind": "commit_transaction"},
            transaction_id=second_id,
        )
        assert _call(
            dsn,
            schema,
            {"kind": "latest_retained_event_seq", "namespace": "native-session"},
        ) == 1
        assert _call(
            dsn,
            other_schema,
            {"kind": "latest_retained_event_seq", "namespace": "native-session"},
        ) == 1
    finally:
        with sa_engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{other_schema}" CASCADE')


def test_rust_postgres_meta_facade_commits_and_rolls_back_shared_capabilities(
    pg_dsn: str | None, pg_schema: str | None, sa_engine: Any
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    rust = RustEnginePostgresMetaStore(dsn=dsn, schema=schema)
    rust.ensure_initialized()

    with rust.transaction() as token:
        assert rust.next_global_seq_conn(token) == 1
        assert rust.next_scoped_seq("scope") == 1
        assert rust.append_entity_event(
            namespace="facade",
            event_id="committed",
            entity_kind="node",
            entity_id="n1",
            op="UPSERT",
            payload_json="{}",
        ) == 1
        rust.replace_named_projection(
            "facade",
            "state",
            {"value": 1},
            last_authoritative_seq=1,
            last_materialized_seq=1,
            projection_schema_version=1,
            materialization_status="ready",
        )
        assert rust.enqueue_index_job(
            job_id="committed-job",
            namespace="facade",
            entity_kind="node",
            entity_id="n1",
            index_kind="node_docs",
            op="UPSERT",
        ) == "committed-job"

    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    assert python.current_global_seq() == 1
    assert python.current_scoped_seq("scope") == 1
    assert [row[0] for row in python.iter_entity_events(namespace="facade")] == [1]
    assert python.get_named_projection("facade", "state")["payload"] == {"value": 1}
    assert python.list_index_jobs(namespace="facade")[0].job_id == "committed-job"

    with pytest.raises(RuntimeError, match="rollback"):
        with rust.transaction():
            assert rust.next_global_seq() == 2
            rust.append_entity_event(
                namespace="facade",
                event_id="rolled-back",
                entity_kind="node",
                entity_id="n2",
                op="UPSERT",
                payload_json="{}",
            )
            rust.replace_named_projection(
                "facade",
                "state",
                {"value": 2},
                last_authoritative_seq=2,
                last_materialized_seq=2,
                projection_schema_version=1,
                materialization_status="ready",
            )
            rust.enqueue_index_job(
                job_id="rolled-back-job",
                namespace="facade",
                entity_kind="node",
                entity_id="n2",
                index_kind="node_docs",
                op="UPSERT",
            )
            raise RuntimeError("rollback")

    assert python.current_global_seq() == 1
    assert [row[0] for row in python.iter_entity_events(namespace="facade")] == [1]
    assert python.get_named_projection("facade", "state")["payload"] == {"value": 1}
    assert [row.job_id for row in python.list_index_jobs(namespace="facade")] == [
        "committed-job"
    ]


def test_public_rust_postgres_node_add_is_atomic_with_event_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    assert isinstance(engine.meta_sqlite, RustEnginePostgresMetaStore)
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)

    node = build_entity_node(
        node_id="public-committed", doc_id="doc", embedding=[1.0, 0.0, 0.0]
    )
    engine.write.add_node(node)

    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    row = backend.node_get(ids=["public-committed"], include=["documents"])
    assert row["ids"] == ["public-committed"]

    def reject_python_read(**_values: Any) -> None:
        raise AssertionError("exact singleton Rust graph read used Python backend")

    with monkeypatch.context() as native_read_guard:
        native_read_guard.setattr(backend, "node_get", reject_python_read)
        native_nodes = engine.read.get_nodes(ids=["public-committed"])
        assert [item.id for item in native_nodes] == ["public-committed"]

    with monkeypatch.context() as native_query_guard:
        native_query_guard.setattr(backend, "node_query", reject_python_read)
        native_matches = engine.read.query_nodes(
            query_embeddings=[[1.0, 0.0, 0.0]],
            where={"doc_id": "doc"},
            n_results=5,
        )
        assert [item.id for item in native_matches[0]] == ["public-committed"]

    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(kind, entity_id, op) for _, kind, entity_id, op, _ in events] == [
        ("node", "public-committed", "ADD")
    ]
    jobs = python.list_index_jobs(namespace=engine.namespace)
    assert {(job.entity_id, job.index_kind) for job in jobs} == {
        ("public-committed", "node_docs"),
        ("public-committed", "node_refs"),
    }

    def fail_enqueue(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(engine, "enqueue_index_jobs_for_node", fail_enqueue)
    failed = build_entity_node(
        node_id="public-rolled-back", doc_id="doc", embedding=[1.0, 0.0, 0.0]
    )
    with pytest.raises(RuntimeError, match="enqueue failed"):
        engine.write.add_node(failed)

    assert backend.node_get(ids=["public-rolled-back"], include=[])["ids"] == []
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [entity_id for _, _, entity_id, _, _ in events] == ["public-committed"]
    assert all(job.entity_id != "public-rolled-back" for job in python.list_index_jobs())


def test_public_rust_postgres_edge_add_is_atomic_with_event_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    for node_id in ("edge-source", "edge-target"):
        backend.node_add(
            ids=[node_id],
            documents=[node_id],
            metadatas=[{}],
            embeddings=[[1.0, 0.0, 0.0]],
        )
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)

    edge = build_relationship_edge(
        edge_id="public-edge-committed",
        src="edge-source",
        tgt="edge-target",
        doc_id="doc",
        embedding=[1.0, 0.0, 0.0],
    )
    engine.write.add_edge(edge)

    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    assert backend.edge_get(ids=[edge.id], include=[])["ids"] == [edge.id]
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(kind, entity_id, op) for _, kind, entity_id, op, _ in events] == [
        ("edge", edge.id, "ADD")
    ]
    jobs = python.list_index_jobs(namespace=engine.namespace)
    assert {(job.entity_id, job.index_kind) for job in jobs} == {
        (edge.id, "edge_refs"),
        (edge.id, "edge_endpoints"),
    }

    def fail_enqueue(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("edge enqueue failed")

    monkeypatch.setattr(engine, "enqueue_index_jobs_for_edge", fail_enqueue)
    failed = build_relationship_edge(
        edge_id="public-edge-rolled-back",
        src="edge-source",
        tgt="edge-target",
        doc_id="doc",
        embedding=[1.0, 0.0, 0.0],
    )
    with pytest.raises(RuntimeError, match="edge enqueue failed"):
        engine.write.add_edge(failed)

    assert backend.edge_get(ids=[failed.id], include=[])["ids"] == []
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [entity_id for _, _, entity_id, _, _ in events] == [edge.id]
    assert all(job.entity_id != failed.id for job in python.list_index_jobs())


def test_public_rust_postgres_duplicate_add_matches_python_upsert_semantics(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)

    first = build_entity_node(
        node_id="public-upsert", doc_id="doc", summary="first",
        embedding=[1.0, 0.0, 0.0],
    )
    replacement = build_entity_node(
        node_id="public-upsert", doc_id="doc", summary="replacement",
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(first)
    engine.write.add_node(replacement)

    row = backend.node_get(
        ids=["public-upsert"], include=["documents", "embeddings"]
    )
    assert row["ids"] == ["public-upsert"]
    assert "replacement" in row["documents"][0]
    assert row["embeddings"][0] == pytest.approx([0.0, 1.0, 0.0])

    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(entity_id, op) for _, _, entity_id, op, _ in events] == [
        ("public-upsert", "ADD"),
        ("public-upsert", "ADD"),
    ]
    jobs = python.list_index_jobs(namespace=engine.namespace)
    assert {(job.entity_id, job.index_kind) for job in jobs} == {
        ("public-upsert", "node_docs"),
        ("public-upsert", "node_refs"),
    }


def test_public_rust_postgres_lifecycle_patch_is_atomic_and_preserves_row(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    node = build_entity_node(
        node_id="lifecycle-node",
        doc_id="doc",
        summary="must survive",
        metadata={"custom": "kept"},
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(node)

    assert engine.lifecycle.tombstone_node(
        node.id, reason="merged", deleted_by="test"
    )
    row = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    document = json.loads(row["documents"][0])
    assert document["summary"] == "must survive"
    assert document["metadata"]["lifecycle_status"] == "tombstoned"
    assert row["metadatas"][0]["custom"] == "kept"
    assert row["metadatas"][0]["delete_reason"] == "merged"
    assert row["embeddings"][0] == pytest.approx([0.0, 1.0, 0.0])

    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(entity_id, op) for _, _, entity_id, op, _ in events] == [
        (node.id, "ADD"),
        (node.id, "TOMBSTONE"),
    ]
    jobs = python.list_index_jobs(namespace=engine.namespace)
    assert {(job.index_kind, job.op) for job in jobs} == {
        ("node_docs", "DELETE"),
        ("node_refs", "DELETE"),
    }
    assert not engine.lifecycle.tombstone_node("missing-node")
    assert len(list(python.iter_entity_events(namespace=engine.namespace))) == 2


def test_public_rust_postgres_lifecycle_failure_rolls_back_patch_event_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    node = build_entity_node(
        node_id="lifecycle-rollback",
        doc_id="doc",
        metadata={"custom": "kept"},
        embedding=[1.0, 0.0, 0.0],
    )
    engine.write.add_node(node)
    before = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )

    def fail_enqueue(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("lifecycle enqueue failed")

    monkeypatch.setattr(engine, "enqueue_index_jobs_for_node", fail_enqueue)
    with pytest.raises(RuntimeError, match="lifecycle enqueue failed"):
        engine.lifecycle.tombstone_node(node.id)

    row = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    assert row["documents"] == before["documents"]
    assert row["metadatas"] == before["metadatas"]
    assert row["embeddings"] == before["embeddings"]
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(entity_id, op) for _, _, entity_id, op, _ in events] == [
        (node.id, "ADD")
    ]
    assert {(job.index_kind, job.op) for job in python.list_index_jobs()} == {
        ("node_docs", "UPSERT"),
        ("node_refs", "UPSERT"),
    }


def test_public_rust_postgres_legacy_default_row_is_claimed_but_custom_scope_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    backend.node_add(
        ids=["legacy-default", "legacy-unknown-scope"],
        documents=[
            json.dumps({"id": "legacy-default", "metadata": {"old": True}}),
            json.dumps({"id": "legacy-unknown-scope", "metadata": {"old": True}}),
        ],
        metadatas=[{"old": True}, {"old": True}],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")

    default_engine = GraphKnowledgeEngine(
        backend=backend,
        namespace="default",
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(default_engine, "reconcile_indexes", lambda **kwargs: 0)
    assert default_engine.lifecycle.tombstone_node("legacy-default")
    claimed = backend.node_get(ids=["legacy-default"], include=["metadatas"])
    assert claimed["metadatas"][0]["namespace"] == "default"

    custom_engine = GraphKnowledgeEngine(
        backend=backend,
        namespace="custom",
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(custom_engine, "reconcile_indexes", lambda **kwargs: 0)
    with pytest.raises(RustParityError) as ambiguous:
        custom_engine.lifecycle.tombstone_node("legacy-unknown-scope")
    assert ambiguous.value.code == "KOGWISTAR_STORE_PERSISTENCE_FAILED"
    unchanged = backend.node_get(
        ids=["legacy-unknown-scope"], include=["documents", "metadatas"]
    )
    assert unchanged["metadatas"] == [{"old": True}]
    assert json.loads(unchanged["documents"][0])["metadata"] == {"old": True}


def test_public_rust_postgres_document_and_domain_add_are_event_atomic(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    document = Document(
        id="public-document",
        content="document body",
        type="text",
        metadata={"source": "test"},
        embeddings=[1.0, 0.0, 0.0],
    )
    domain = Domain(id="public-domain", name="Domain", description="description")
    engine.write.add_document(document)
    engine.write.add_domain(domain)

    assert backend.document_get(ids=[document.id], include=["documents"])[
        "documents"
    ] == ["document body"]
    with monkeypatch.context() as native_read_guard:
        native_read_guard.setattr(
            backend,
            "document_get",
            lambda **_values: (_ for _ in ()).throw(
                AssertionError("exact document Rust read used Python backend")
            ),
        )
        assert engine.get_document(document.id).content == "document body"
    assert backend.domain_get(ids=[domain.id], include=["documents"])["ids"] == [
        domain.id
    ]
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(kind, entity_id, op) for _, kind, entity_id, op, _ in events] == [
        ("document", document.id, "ADD"),
        ("domain", domain.id, "ADD"),
    ]

    meta = engine.meta_sqlite
    assert isinstance(meta, RustEnginePostgresMetaStore)
    original = meta.apply_graph_mutation

    def fail_after_write(**values: Any) -> dict[str, Any]:
        original(**values)
        raise RuntimeError("after native graph write")

    monkeypatch.setattr(meta, "apply_graph_mutation", fail_after_write)
    failed = Document(
        id="public-document-rolled-back",
        content="must roll back",
        type="text",
        metadata={},
        embeddings=[0.0, 1.0, 0.0],
    )
    with pytest.raises(RuntimeError, match="after native graph write"):
        engine.write.add_document(failed)
    assert backend.document_get(ids=[failed.id], include=[])["ids"] == []
    assert [entity_id for _, _, entity_id, _, _ in python.iter_entity_events()] == [
        document.id,
        domain.id,
    ]


@pytest.mark.parametrize(
    ("meta_mode", "graph_mode"), [("rust", "python"), ("python", "rust")]
)
def test_public_sync_postgres_rejects_partial_rust_authority(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
    meta_mode: str,
    graph_mode: str,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", meta_mode)
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", graph_mode)
    with pytest.raises(ValueError, match="requires both"):
        GraphKnowledgeEngine(
            backend=backend,
            embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
        )


def test_native_graph_delete_is_atomic_idempotent_and_scope_safe(
    pg_dsn: str | None, pg_schema: str | None, sa_engine: Any
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    _call(dsn, schema, {"kind": "ensure_schema"})
    backend.node_add(
        ids=["delete-me", "wrong-scope"],
        documents=["delete", "wrong"],
        metadatas=[
            {"namespace": "delete"},
            {"namespace": "other"},
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    operation = {
        "kind": "graph_delete_mutation",
        "namespace": "delete",
        "table": backend.nodes.name,
        "entity_kind": "node",
        "event_id": "delete-event",
        "entity_id": "delete-me",
        "payload": {"reason": "test"},
    }
    deleted = _call(dsn, schema, operation)
    assert deleted["inserted"] is True
    assert deleted["mutated"] is True
    assert backend.node_get(ids=["delete-me"], include=[])["ids"] == []
    retry = _call(dsn, schema, operation)
    assert retry["event"] == deleted["event"]
    assert retry["inserted"] is False
    assert retry["mutated"] is False

    assert _call(
        dsn,
        schema,
        {**operation, "event_id": "missing-delete", "entity_id": "missing"},
    ) is None
    assert _call(
        dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "delete"}
    ) == 1

    with pytest.raises(RustParityError) as wrong_scope:
        _call(
            dsn,
            schema,
            {
                **operation,
                "event_id": "wrong-scope-delete",
                "entity_id": "wrong-scope",
            },
        )
    assert wrong_scope.value.code == "KOGWISTAR_STORE_PERSISTENCE_FAILED"
    assert backend.node_get(ids=["wrong-scope"], include=[])["ids"] == [
        "wrong-scope"
    ]

    backend.node_add(
        ids=["rollback-delete"],
        documents=["rollback"],
        metadatas=[{"namespace": "delete"}],
        embeddings=[[0.0, 0.0, 1.0]],
    )
    transaction_id = uuid.uuid4().hex
    _call(
        dsn,
        schema,
        {"kind": "begin_transaction"},
        transaction_id=transaction_id,
    )
    _call(
        dsn,
        schema,
        {
            **operation,
            "event_id": "rollback-delete",
            "entity_id": "rollback-delete",
        },
        transaction_id=transaction_id,
    )
    _call(
        dsn,
        schema,
        {"kind": "rollback_transaction"},
        transaction_id=transaction_id,
    )
    assert backend.node_get(ids=["rollback-delete"], include=[])["ids"] == [
        "rollback-delete"
    ]
    assert _call(
        dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "delete"}
    ) == 1


def test_public_rust_postgres_edge_delete_is_atomic_with_event_and_jobs(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    for node_id in ("delete-source", "delete-target"):
        engine.write.add_node(
            build_entity_node(
                node_id=node_id,
                doc_id="doc",
                embedding=[1.0, 0.0, 0.0],
            )
        )
    for edge_id in ("public-delete-edge", "public-rollback-edge"):
        engine.write.add_edge(
            build_relationship_edge(
                edge_id=edge_id,
                src="delete-source",
                tgt="delete-target",
                doc_id="doc",
                embedding=[1.0, 0.0, 0.0],
            )
        )

    engine.write.delete_edges_by_ids(["public-delete-edge", "missing-edge"])
    assert backend.edge_get(ids=["public-delete-edge"], include=[])["ids"] == []
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(kind, entity_id, op) for _, kind, entity_id, op, _ in events][-1] == (
        "edge",
        "public-delete-edge",
        "DELETE",
    )
    assert not any(entity_id == "missing-edge" for _, _, entity_id, _, _ in events)
    delete_jobs = [
        job
        for job in python.list_index_jobs(namespace=engine.namespace)
        if job.entity_id == "public-delete-edge"
    ]
    assert {(job.index_kind, job.op) for job in delete_jobs} == {
        ("edge_refs", "DELETE"),
        ("edge_endpoints", "DELETE"),
    }

    def fail_enqueue(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("delete enqueue failed")

    monkeypatch.setattr(engine, "enqueue_index_jobs_for_edge", fail_enqueue)
    before_events = list(python.iter_entity_events(namespace=engine.namespace))
    with pytest.raises(RuntimeError, match="delete enqueue failed"):
        engine.write.delete_edges_by_ids(["public-rollback-edge"])
    assert backend.edge_get(ids=["public-rollback-edge"], include=[])["ids"] == [
        "public-rollback-edge"
    ]
    assert list(python.iter_entity_events(namespace=engine.namespace)) == before_events


def test_public_rust_postgres_replace_existing_preserves_embedding_and_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    node = build_entity_node(
        node_id="replace-existing",
        doc_id="doc",
        summary="before",
        metadata={"kept": "yes"},
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(node)
    replacement = node.model_copy(deep=True)
    replacement.summary = "after"
    replacement.mentions = list(node.mentions)
    replacement_document = replacement.model_dump_json(field_mode="backend")
    assert engine.write.rust_postgres_replace_existing(
        entity_kind="node",
        entity_id=node.id,
        document=replacement_document,
        metadata_patch={"references": "merged", "new_key": "new"},
        payload=replacement.model_dump(field_mode="backend", exclude=["embedding"]),
    )

    row = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    assert json.loads(row["documents"][0])["summary"] == "after"
    assert row["metadatas"][0]["kept"] == "yes"
    assert row["metadatas"][0]["new_key"] == "new"
    assert row["embeddings"][0] == pytest.approx([0.0, 1.0, 0.0])
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events = list(python.iter_entity_events(namespace=engine.namespace))
    assert [(entity_id, op) for _, _, entity_id, op, _ in events] == [
        (node.id, "ADD"),
        (node.id, "REPLACE"),
    ]
    assert {
        (job.index_kind, job.op)
        for job in python.list_index_jobs(namespace=engine.namespace)
    } == {("node_docs", "UPSERT"), ("node_refs", "UPSERT")}

    def fail_enqueue(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("replace enqueue failed")

    monkeypatch.setattr(engine, "enqueue_index_jobs_for_node", fail_enqueue)
    before = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    before_events = list(python.iter_entity_events(namespace=engine.namespace))
    with pytest.raises(RuntimeError, match="replace enqueue failed"):
        engine.write.rust_postgres_replace_existing(
            entity_kind="node",
            entity_id=node.id,
            document=json.dumps({"id": node.id, "summary": "rolled back"}),
            metadata_patch={"new_key": "rolled back"},
            payload={"id": node.id},
        )
    after = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    assert after == before
    assert list(python.iter_entity_events(namespace=engine.namespace)) == before_events


def test_public_rust_postgres_existing_ingest_merges_references_via_native_replace(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    for doc_id in ("merge-doc-one", "merge-doc-two"):
        engine.write.add_document(
            Document(
                id=doc_id,
                content="x",
                type="text",
                metadata={},
                domain_id=None,
                processed=False,
                embeddings=[1.0, 0.0, 0.0],
                source_map=None,
            )
        )
    original = build_entity_node(
        node_id="merge-existing",
        doc_id="merge-doc-one",
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(original)
    incoming = build_entity_node(
        node_id="merge-existing",
        doc_id="merge-doc-two",
        embedding=[1.0, 0.0, 0.0],
    )

    result = engine.persist.ingest_with_toposort(
        GraphExtractionWithIDs(nodes=[incoming], edges=[]),
        doc_id="merge-doc-two",
    )
    assert result["nodes_added"] == 0
    row = backend.node_get(
        ids=["merge-existing"], include=["documents", "embeddings"]
    )
    merged = Node.model_validate_json(row["documents"][0])
    assert {span.doc_id for grounding in merged.mentions for span in grounding.spans} == {
        "merge-doc-one",
        "merge-doc-two",
    }
    assert row["embeddings"][0] == pytest.approx([0.0, 1.0, 0.0])
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    node_events = [
        (entity_id, op)
        for _, kind, entity_id, op, _ in python.iter_entity_events(
            namespace=engine.namespace
        )
        if kind == "node"
    ]
    assert node_events == [("merge-existing", "ADD"), ("merge-existing", "REPLACE")]


def test_public_rust_postgres_rollback_document_uses_native_delete_event(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    document = Document(
        id="rollback-native-document",
        content="rollback body",
        type="text",
        metadata={},
        domain_id=None,
        processed=False,
        embeddings=[1.0, 0.0, 0.0],
        source_map=None,
    )
    engine.write.add_document(document)
    result = engine.rollback.rollback_document(document.id)

    assert result["rolled_back_doc_ids"] == [document.id]
    assert backend.document_get(ids=[document.id], include=[])["ids"] == []
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    document_events = [
        (entity_id, op)
        for _, kind, entity_id, op, _ in python.iter_entity_events(
            namespace=engine.namespace
        )
        if kind == "document"
    ]
    assert document_events == [(document.id, "ADD"), (document.id, "DELETE")]


def test_public_rust_postgres_replay_repairs_projection_without_new_events(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    node = build_entity_node(
        node_id="replay-node",
        doc_id="doc",
        summary="authoritative",
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(node)
    assert engine.lifecycle.tombstone_node(node.id, reason="replay-test")

    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events_before = list(python.iter_entity_events(namespace=engine.namespace))
    assert [op for _, _, _, op, _ in events_before] == ["ADD", "TOMBSTONE"]
    with sa_engine.begin() as connection:
        connection.exec_driver_sql(f'TRUNCATE TABLE "{schema}"."{backend.nodes.name}"')
    assert backend.node_get(ids=[node.id], include=[])["ids"] == []

    last_seq = engine.persist.replay_namespace(
        namespace=engine.namespace,
        apply_indexes=True,
        repair_backend=True,
    )
    assert last_seq == events_before[-1][0]
    rebuilt = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    assert rebuilt["ids"] == [node.id]
    assert json.loads(rebuilt["documents"][0])["summary"] == "authoritative"
    assert rebuilt["metadatas"][0]["lifecycle_status"] == "tombstoned"
    assert rebuilt["metadatas"][0]["delete_reason"] == "replay-test"
    assert list(python.iter_entity_events(namespace=engine.namespace)) == events_before


def test_public_rust_postgres_prune_node_routes_edge_replace_and_delete(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    for node_id in ("prune-remove", "prune-survivor", "prune-target"):
        engine.write.add_node(
            build_entity_node(
                node_id=node_id,
                doc_id="doc",
                embedding=[1.0, 0.0, 0.0],
            )
        )
    deleted_edge = build_relationship_edge(
        edge_id="prune-delete-edge",
        src="prune-remove",
        tgt="prune-target",
        doc_id="doc",
        embedding=[1.0, 0.0, 0.0],
    )
    updated_edge = build_relationship_edge(
        edge_id="prune-update-edge",
        src="prune-remove",
        tgt="prune-target",
        doc_id="doc",
        embedding=[0.0, 1.0, 0.0],
    )
    updated_edge.source_ids.append("prune-survivor")
    engine.write.add_edge(deleted_edge)
    engine.write.add_edge(updated_edge)
    endpoint_rows = [
        {
            "id": f"{edge_id}::src::prune-remove",
            "edge_id": edge_id,
            "endpoint_id": "prune-remove",
            "endpoint_type": "node",
            "node_id": "prune-remove",
            "role": "src",
        }
        for edge_id in (deleted_edge.id, updated_edge.id)
    ]
    backend.edge_endpoints_add(
        ids=[row["id"] for row in endpoint_rows],
        documents=[json.dumps(row) for row in endpoint_rows],
        metadatas=endpoint_rows,
        embeddings=[[1.0, 0.0, 0.0] for _ in endpoint_rows],
    )

    result = engine.rollback.prune_node_from_edges("prune-remove")
    assert result["deleted_edges"] == {deleted_edge.id}
    assert result["updated_edges"] == {updated_edge.id}
    assert backend.edge_get(ids=[deleted_edge.id], include=[])["ids"] == []
    updated = backend.edge_get(
        ids=[updated_edge.id], include=["documents", "embeddings"]
    )
    parsed = Edge.model_validate_json(updated["documents"][0])
    assert parsed.source_ids == ["prune-survivor"]
    assert updated["embeddings"][0] == pytest.approx([0.0, 1.0, 0.0])
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    edge_events = [
        (entity_id, op)
        for _, kind, entity_id, op, _ in python.iter_entity_events(
            namespace=engine.namespace
        )
        if kind == "edge"
    ]
    assert edge_events == [
        (deleted_edge.id, "ADD"),
        (updated_edge.id, "ADD"),
        (deleted_edge.id, "DELETE"),
        (updated_edge.id, "REPLACE"),
    ]


def test_public_rust_postgres_derived_metadata_patch_preserves_document_and_history(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    node = build_entity_node(
        node_id="derived-patch-node",
        doc_id="doc",
        embedding=[1.0, 0.0, 0.0],
    )
    engine.write.add_node(node)
    before = backend.node_get(ids=[node.id], include=["documents", "metadatas"])
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    events_before = list(python.iter_entity_events(namespace=engine.namespace))

    engine.write.patch_base_projection_metadata(
        entity_kind="node",
        entity_id=node.id,
        metadata_patch={"doc_ids": '["doc"]', "node_refs_fp": "fingerprint"},
    )
    after = backend.node_get(ids=[node.id], include=["documents", "metadatas"])
    assert after["documents"] == before["documents"]
    assert after["metadatas"][0]["doc_ids"] == '["doc"]'
    assert after["metadatas"][0]["node_refs_fp"] == "fingerprint"
    assert list(python.iter_entity_events(namespace=engine.namespace)) == events_before


def test_public_rust_postgres_pure_adds_use_native_events_without_fanout_jobs(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    source = PureChromaNode(
        id="pure-source",
        label="source",
        type="entity",
        summary="source",
        doc_id=None,
        metadata={},
        embedding=[1.0, 0.0, 0.0],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )
    target = source.model_copy(update={"id": "pure-target", "label": "target"})
    edge = PureChromaEdge(
        id="pure-edge",
        label="related",
        type="relationship",
        summary="related",
        relation="related_to",
        source_ids=[source.id],
        target_ids=[target.id],
        source_edge_ids=[],
        target_edge_ids=[],
        doc_id=None,
        metadata={},
        embedding=[0.0, 1.0, 0.0],
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )

    engine.write.add_pure_node(source)
    engine.write.add_pure_node(target)
    engine.write.add_pure_edge(edge)

    assert backend.node_get(ids=[source.id, target.id], include=[])["ids"]
    assert backend.edge_get(ids=[edge.id], include=[])["ids"] == [edge.id]
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    assert [
        (kind, entity_id, op)
        for _, kind, entity_id, op, _ in python.iter_entity_events(
            namespace=engine.namespace
        )
    ] == [
        ("node", source.id, "ADD"),
        ("node", target.id, "ADD"),
        ("edge", edge.id, "ADD"),
    ]
    assert python.list_index_jobs(namespace=engine.namespace) == []


def test_public_rust_postgres_prune_node_refs_replaces_atomically(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    node = build_entity_node(
        node_id="prune-node-refs",
        doc_id="doc-one",
        mentions=[
            Grounding(
                spans=[mk_document_span("doc-one"), mk_document_span("doc-two")]
            )
        ],
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(node)

    assert engine.write.prune_node_refs_for_doc(node.id, "doc-one")

    row = backend.node_get(
        ids=[node.id], include=["documents", "metadatas", "embeddings"]
    )
    pruned = Node.model_validate_json(row["documents"][0])
    assert [span.doc_id for grounding in pruned.mentions for span in grounding.spans] == [
        "doc-two"
    ]
    assert row["embeddings"][0] == pytest.approx([0.0, 1.0, 0.0])
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    assert [
        (entity_id, op)
        for _, kind, entity_id, op, _ in python.iter_entity_events(
            namespace=engine.namespace
        )
        if kind == "node"
    ] == [(node.id, "ADD"), (node.id, "REPLACE")]


def test_public_rust_postgres_extraction_rollback_deletes_atomically(
    monkeypatch: pytest.MonkeyPatch,
    pg_dsn: str | None,
    pg_schema: str | None,
    sa_engine: Any,
) -> None:
    _, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "rust")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "rust")
    engine = GraphKnowledgeEngine(
        backend=backend,
        embedding_function=lambda documents_or_texts: [
            [1.0, 0.0, 0.0] for _ in documents_or_texts
        ],
    )
    monkeypatch.setattr(engine, "reconcile_indexes", lambda **kwargs: 0)
    span = mk_document_span("extraction-doc")
    span.insertion_method = "llm_graph_extraction"
    node = build_entity_node(
        node_id="extraction-delete-node",
        doc_id="extraction-doc",
        mentions=[Grounding(spans=[span])],
        embedding=[0.0, 1.0, 0.0],
    )
    engine.write.add_node(node)

    result = engine.rollback_document_extraction(
        "extraction-doc", "llm_graph_extraction"
    )

    assert result["deleted_nodes"] == 1
    assert backend.node_get(ids=[node.id], include=[])["ids"] == []
    python = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    assert [
        (entity_id, op)
        for _, kind, entity_id, op, _ in python.iter_entity_events(
            namespace=engine.namespace
        )
        if kind == "node"
    ] == [(node.id, "ADD"), (node.id, "DELETE")]

    failed = build_entity_node(
        node_id="extraction-delete-failure",
        doc_id="extraction-doc",
        mentions=[Grounding(spans=[span.model_copy(deep=True)])],
        embedding=[1.0, 0.0, 0.0],
    )
    engine.write.add_node(failed)

    def fail_enqueue(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("extraction delete enqueue failed")

    monkeypatch.setattr(engine, "enqueue_index_jobs_for_node", fail_enqueue)
    with pytest.raises(RuntimeError, match="extraction delete enqueue failed"):
        engine.rollback_document_extraction(
            "extraction-doc", "llm_graph_extraction"
        )
    assert backend.node_get(ids=[failed.id], include=[])["ids"] == [failed.id]
