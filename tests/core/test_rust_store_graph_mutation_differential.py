from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("pgvector")
pytest.importorskip("sqlalchemy")

from kogwistar._rust_bridge import RustParityError, store_postgres
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.postgres_backend import PgVectorBackend


pytestmark = [pytest.mark.ci_full, pytest.mark.core]


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _rust(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def _mutation(
    *,
    event_id: str,
    op: str,
    namespace: str = "alpha",
    workspace_id: str = "ws",
    graph_space: str = "facts",
    **record: Any,
) -> dict[str, Any]:
    return {
        "kind": "graph_mutation",
        "namespace": namespace,
        "workspace_id": workspace_id,
        "graph_space": graph_space,
        "table": "gke_nodes",
        "entity_kind": "node",
        "event_id": event_id,
        "op": op,
        "payload": {"id": record["id"], "version": event_id},
        "embedding_dim": 3,
        "record": record,
    }


def test_graph_mutation_event_history_pgvector_and_python_mutual_readability(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    meta = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    meta.ensure_initialized()

    add = _mutation(
        event_id="node-add",
        op="ADD",
        id="n1",
        document="first",
        metadata={"team": "red"},
        embedding=[1.0, 0.0, 0.0],
    )
    assert _rust(dsn, schema, add)["event"]["seq"] == 1
    retry = _rust(dsn, schema, add)
    assert retry["inserted"] is False
    assert retry["mutated"] is False

    replace = _mutation(
        event_id="node-replace",
        op="REPLACE",
        id="n1",
        document="second",
        metadata={"team": "red", "rank": 2},
        embedding=[0.9, 0.1, 0.0],
    )
    tombstone = _mutation(
        event_id="node-tombstone",
        op="TOMBSTONE",
        id="n1",
        document="second",
        metadata={"team": "red", "lifecycle_status": "tombstoned"},
        embedding=[0.9, 0.1, 0.0],
    )
    assert _rust(dsn, schema, replace)["event"]["seq"] == 2
    assert _rust(dsn, schema, tombstone)["event"]["seq"] == 3
    assert [row[3] for row in meta.iter_entity_events(namespace="alpha", from_seq=1)] == [
        "ADD",
        "REPLACE",
        "TOMBSTONE",
    ]

    # Python sees Rust materialization, including scope metadata, and Rust sees
    # a Python-written row in the same pre-existing pgvector relation.
    python_row = backend.node_get(ids=["n1"], include=["documents", "metadatas", "embeddings"])
    assert python_row["documents"] == ["second"]
    assert python_row["metadatas"][0]["namespace"] == "alpha"
    assert python_row["metadatas"][0]["lifecycle_status"] == "tombstoned"
    backend.node_upsert(
        ids=["python-row"],
        documents=["from python"],
        metadatas=[{"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts", "team": "red"}],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    rows = _rust(
        dsn,
        schema,
        {
            "kind": "graph_projection_records",
            "namespace": "alpha",
            "workspace_id": "ws",
            "graph_space": "facts",
            "table": "gke_nodes",
            "metadata": {"team": "red"},
            "limit": 10,
        },
    )
    assert [row["id"] for row in rows] == ["n1", "python-row"]


def test_graph_mutation_scope_vector_contract_and_fault_rollback(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    EnginePostgresMetaStore(engine=sa_engine, schema=schema).ensure_initialized()

    for row_id, vector in (("b", [1.0, 0.0, 0.0]), ("a", [1.0, 0.0, 0.0])):
        _rust(
            dsn,
            schema,
            _mutation(
                event_id=f"vector-{row_id}",
                op="ADD",
                id=row_id,
                document=row_id,
                metadata={"team": "red"},
                embedding=vector,
            ),
        )
    matches = _rust(
        dsn,
        schema,
        {
            "kind": "graph_projection_vector_query",
            "namespace": "alpha",
            "workspace_id": "ws",
            "graph_space": "facts",
            "table": "gke_nodes",
            "embedding": [1.0, 0.0, 0.0],
            "embedding_dim": 3,
            "metadata": {"team": "red"},
            "metric": "cosine",
            "limit": 10,
        },
    )
    assert [item["record"]["id"] for item in matches] == ["a", "b"]

    isolated = _mutation(
        event_id="other-scope",
        op="ADD",
        namespace="beta",
        id="other",
        document="other",
        metadata={"team": "red"},
        embedding=[1.0, 0.0, 0.0],
    )
    assert _rust(dsn, schema, isolated)["event"]["seq"] == 1
    assert all(item["record"]["id"] != "other" for item in matches)

    with pytest.raises(RustParityError) as dimension:
        _rust(
            dsn,
            schema,
            _mutation(
                event_id="bad-dimension",
                op="ADD",
                id="bad",
                document="bad",
                metadata={},
                embedding=[1.0, 0.0],
            ),
        )
    assert dimension.value.code == "KOGWISTAR_STORE_PERSISTENCE_FAILED"

    rollback = _mutation(
        event_id="rolled-back",
        op="ADD",
        id="rolled-back",
        document="rollback",
        metadata={"team": "red"},
        embedding=[1.0, 0.0, 0.0],
    )
    with pytest.raises(RustParityError) as aborted:
        _rust(dsn, schema, {"kind": "batch", "operations": [rollback], "abort": True})
    assert aborted.value.code == "KOGWISTAR_STORE_TRANSACTION_ABORTED"
    assert _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "alpha"}) == 2
    assert backend.node_get(ids=["rolled-back"], include=["documents"])["ids"] == []

    with pytest.raises(RustParityError) as unknown:
        _rust(
            dsn,
            schema,
            {**_mutation(event_id="unknown", op="ADD", id="unknown", document="", metadata={}, embedding=[1.0, 0.0, 0.0]), "nope": True},
        )
    assert unknown.value.code == "KOGWISTAR_STORE_INVALID_PAYLOAD"


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_graph_mutation_vector_metric_parity_with_python_pgvector(
    sa_engine, pg_dsn: str | None, pg_schema: str | None, metric: str
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, distance=metric, schema=schema)
    backend.ensure_schema()
    EnginePostgresMetaStore(engine=sa_engine, schema=schema).ensure_initialized()
    for row_id, embedding in (("b", [1.0, 0.0, 0.0]), ("a", [1.0, 0.0, 0.0]), ("c", [0.0, 1.0, 0.0])):
        _rust(
            dsn,
            schema,
            _mutation(
                event_id=f"{metric}-{row_id}", op="ADD", id=row_id, document=row_id,
                metadata={"team": "metric"}, embedding=embedding,
            ),
        )
    python = backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]], n_results=10,
        where={"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts", "team": "metric"},
        include=["distances"],
    )
    rust = _rust(
        dsn,
        schema,
        {
            "kind": "graph_projection_vector_query", "namespace": "alpha", "workspace_id": "ws",
            "graph_space": "facts", "table": "gke_nodes", "embedding": [1.0, 0.0, 0.0],
            "embedding_dim": 3, "metadata": {"team": "metric"}, "metric": metric, "limit": 10,
        },
    )
    assert [item["record"]["id"] for item in rust] == python["ids"][0]
    assert [item["distance"] for item in rust] == pytest.approx(python["distances"][0])


def test_graph_mutation_fresh_rust_schema_and_validation_rollback(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, _ = _require_pg(pg_dsn, pg_schema)
    schema = f"gke_rust_graph_{uuid.uuid4().hex}"
    try:
        assert _rust(dsn, schema, {"kind": "create_graph_schema", "embedding_dim": 3}) == {"initialized": True}
        quoted = "O'Reilly\\雪"
        mutation = _mutation(
            event_id="fresh", op="ADD", id="fresh", document=quoted,
            metadata={"literal": quoted}, embedding=[1.0, 0.0, 0.0],
        )
        assert _rust(dsn, schema, mutation)["event"]["seq"] == 1
        backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
        row = backend.node_get(ids=["fresh"], include=["documents", "metadatas"])
        assert row["documents"] == [quoted]
        assert row["metadatas"][0]["literal"] == quoted
        backend.node_upsert(
            ids=["python"], documents=["python"],
            metadatas=[{"namespace": "alpha", "workspace_id": "ws", "graph_space": "facts", "literal": quoted}],
            embeddings=[[0.0, 1.0, 0.0]],
        )
        rows = _rust(dsn, schema, {
            "kind": "graph_projection_records", "namespace": "alpha", "workspace_id": "ws",
            "graph_space": "facts", "table": "gke_nodes", "metadata": {"literal": quoted}, "limit": 10,
        })
        assert [row["id"] for row in rows] == ["fresh", "python"]
        assert _rust(dsn, schema, {
            "kind": "graph_projection_records", "namespace": "alpha", "workspace_id": "ws",
            "graph_space": "facts", "table": "gke_nodes", "metadata": {}, "limit": 0,
        }) == []

        # Python's broader init can safely add all remaining backend/meta relations.
        backend.ensure_schema()
        EnginePostgresMetaStore(engine=sa_engine, schema=schema).ensure_initialized()

        before = _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "alpha"})
        conflict = _mutation(
            event_id="scope-conflict", op="REPLACE", id="fresh", document="bad", metadata={},
            embedding=[1.0, 0.0, 0.0], workspace_id="other",
        )
        with pytest.raises(RustParityError):
            _rust(dsn, schema, conflict)
        assert _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "alpha"}) == before
        with pytest.raises(RustParityError):
            _rust(dsn, schema, {
                "kind": "graph_mutation", "namespace": "alpha", "workspace_id": "ws", "graph_space": "facts",
                "table": "not-valid;drop", "entity_kind": "node", "event_id": "bad-table", "op": "ADD",
                "payload": {}, "embedding_dim": 3, "record": {"id": "x", "metadata": {}, "embedding": [1.0, 0.0, 0.0]},
            })
        assert _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "alpha"}) == before
        with pytest.raises(RustParityError) as bad_schema:
            _rust(dsn, "bad-schema;drop", {"kind": "create_graph_schema", "embedding_dim": 3})
        assert bad_schema.value.code == "KOGWISTAR_STORE_PERSISTENCE_FAILED"
    finally:
        with sa_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
