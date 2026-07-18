from __future__ import annotations

from types import SimpleNamespace

import pytest

from kogwistar.engine_core.subsystems.read import ReadSubsystem
from kogwistar.engine_core.rust_postgres_session import (
    RustEnginePostgresMetaStore,
    RustPostgresConnectionUnavailable,
    RustPostgresSession,
)


pytestmark = [pytest.mark.ci, pytest.mark.core]


def test_postgres_session_nested_commit_uses_one_native_transaction(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")

    with session.transaction() as outer:
        session.call("raw_append")
        with session.transaction() as inner:
            session.require_token(inner)
            assert inner.value == outer.value
            session.call("replace_named_projection")

    transaction_ids = {value for _, value in calls if value is not None}
    assert len(transaction_ids) == 1
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "raw_append",
        "replace_named_projection",
        "commit_transaction",
    ]


def test_postgres_session_rolls_back_and_rejects_raw_or_stale_writer(monkeypatch):
    calls: list[str] = []

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        calls.append(operation["kind"])
        return None

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    session = RustPostgresSession(dsn="postgresql://example/db", schema="test")
    with pytest.raises(RustPostgresConnectionUnavailable):
        session.connect()

    stale = None
    with pytest.raises(RuntimeError, match="stop"):
        with session.transaction() as token:
            stale = token
            raise RuntimeError("stop")
    assert calls == ["begin_transaction", "rollback_transaction"]
    assert stale is not None
    with pytest.raises(RustPostgresConnectionUnavailable, match="stale"):
        session.require_token(stale)


def test_postgres_meta_facade_routes_nested_sequence_and_queue_calls(monkeypatch):
    calls: list[tuple[str, str | None]] = []
    results = {
        "next_global_seq": 1,
        "next_user_seq": 1,
        "enqueue_index_job": "job",
    }

    def fake_store_postgres(*, operation, transaction_id, **kwargs):
        del kwargs
        calls.append((operation["kind"], transaction_id))
        return results.get(operation["kind"])

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    with store.transaction() as token:
        assert store.next_global_seq_conn(token) == 1
        assert store.next_scoped_seq("scope") == 1
        assert store.enqueue_index_job(
            job_id="job",
            entity_kind="node",
            entity_id="n",
            index_kind="node_docs",
            op="UPSERT",
        ) == "job"
    assert [kind for kind, _ in calls] == [
        "begin_transaction",
        "next_global_seq",
        "next_user_seq",
        "enqueue_index_job",
        "commit_transaction",
    ]
    assert len({value for _, value in calls if value is not None}) == 1


def test_postgres_meta_facade_validates_native_graph_read_shapes(monkeypatch):
    results = {
        "graph_projection_records": [{"id": "n1"}],
        "graph_projection_vector_query": [
            {"record": {"id": "n1"}, "distance": 0.0}
        ],
    }

    def fake_store_postgres(*, operation, **kwargs):
        del kwargs
        return results[operation["kind"]]

    monkeypatch.setattr(
        "kogwistar.engine_core.rust_postgres_session.store_postgres",
        fake_store_postgres,
    )
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    assert store.graph_projection_records(table="gke_nodes") == [{"id": "n1"}]
    assert store.graph_projection_vector_query(table="gke_nodes")[0][
        "distance"
    ] == 0.0

    results["graph_projection_records"] = {"id": "wrong"}
    with pytest.raises(RuntimeError, match="invalid records"):
        store.graph_projection_records(table="gke_nodes")


def test_read_subsystem_routes_simple_base_reads_to_native_projection(monkeypatch):
    store = RustEnginePostgresMetaStore(
        dsn="postgresql://example/db", schema="test"
    )
    calls: list[dict] = []

    def native_records(**values):
        calls.append(values)
        return [
            {
                "id": "n1",
                "document": '{"id":"n1","summary":"native"}',
                "metadata": {"namespace": "alpha", "team": "red"},
                "embedding": [1.0, 0.0, 0.0],
            }
        ]

    monkeypatch.setattr(store, "graph_projection_records", native_records)

    class Backend:
        nodes = SimpleNamespace(name="gke_nodes")
        edges = SimpleNamespace(name="gke_edges")
        documents = SimpleNamespace(name="gke_documents")
        embedding_dim = 3
        distance = "cosine"

        async def node_get(self, **_values):
            return {"ids": ["python"]}

        async def edge_get(self, **_values):
            return {"ids": ["python-edge"]}

        async def node_query(self, **_values):
            return {"ids": [["python-query"]]}

        async def document_get(self, **_values):
            raise AssertionError("exact document read used Python backend")

    reader = ReadSubsystem(
        SimpleNamespace(meta_sqlite=store, backend=Backend(), namespace="alpha")
    )
    result = reader._node_get_raw(
        ids=["n1"], where={"team": "red"}, limit=5
    )
    assert result == {
        "ids": ["n1"],
        "documents": ['{"id":"n1","summary":"native"}'],
        "metadatas": [{"namespace": "alpha", "team": "red"}],
        "embeddings": [[1.0, 0.0, 0.0]],
    }
    assert calls == [
        {
            "namespace": "alpha",
            "workspace_id": None,
            "graph_space": None,
            "table": "gke_nodes",
            "ids": ["n1"],
            "metadata": {"team": "red"},
            "limit": 5,
        }
    ]

    no_include = reader._node_get_raw(ids=["n1"], include=[], limit=5)
    assert "documents" in no_include and "metadatas" in no_include
    assert "embeddings" not in no_include

    fallback = reader._node_get_raw(
        where={"team": {"$in": ["red"]}}, include=[], limit=5
    )
    assert fallback == {"ids": ["python"]}
    assert reader._node_get_raw(where={"team": "red"}, include=[], limit=5) == {
        "ids": ["python"]
    }

    native_records_result = {
        "id": "doc-1",
        "document": "content",
        "metadata": {"type": "text", "processed": True},
        "embedding": None,
    }
    monkeypatch.setattr(store, "graph_projection_records", lambda **_values: [native_records_result])
    document = reader.get_document("doc-1")
    assert document.id == "doc-1" and document.content == "content"

    monkeypatch.setattr(
        store,
        "graph_projection_vector_query",
        lambda **_values: [
            {"record": native_records_result, "distance": 0.25}
        ],
    )
    native_query = reader._rust_postgres_projection_query(
        entity_kind="node",
        query_embeddings=[[1.0, 0.0, 0.0]],
        where={"team": "red"},
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    assert native_query == {
        "ids": [["doc-1"]],
        "documents": [["content"]],
        "metadatas": [[{"type": "text", "processed": True}]],
        "distances": [[0.25]],
    }
    assert reader._rust_postgres_projection_query(
        entity_kind="node",
        query_embeddings=[[1.0, 0.0, 0.0]],
        where={"team": {"$in": ["red"]}},
        n_results=3,
        include=[],
    ) is None
