from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pgvector")
sa = pytest.importorskip("sqlalchemy")

from kogwistar._rust_bridge import RustParityError, store_postgres
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.engine_postgres_meta import EnginePostgresMetaStore
from kogwistar.engine_core.models import Edge, Node
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from tests._helpers.embeddings import ConstantEmbeddingFunction
from tests._helpers.graph_builders import build_entity_node

pytestmark = [pytest.mark.ci_full, pytest.mark.core]

ROOT = Path(__file__).resolve().parents[2]
PYTHON_GRAPH_MODEL_FIXTURE = json.loads(
    (
        ROOT
        / "contracts"
        / "golden"
        / "python-graph-model-payloads-v1.json"
    ).read_text(encoding="utf-8")
)


@pytest.fixture(scope="module", autouse=True)
def _native_extension() -> Any:
    return pytest.importorskip("kogwistar._rust")


def _rust(dsn: str | None, schema: str, operation: dict[str, Any]) -> Any:
    return store_postgres(dsn=dsn, schema=schema, operation=operation)


def _require_pg(pg_dsn: str | None, pg_schema: str | None) -> tuple[str, str]:
    if pg_dsn is None or pg_schema is None:
        pytest.skip("PostgreSQL fixture/DSN unavailable")
    return pg_dsn, pg_schema


def _drop_pg_schema(sa_engine: Any, schema: str) -> None:
    bootstrap_engine = sa_engine.execution_options(kogwistar_schema_bootstrap=True)
    with bootstrap_engine.begin() as connection:
        connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


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


def test_python_and_rust_postgres_writers_serialize_shared_event_stream(
    sa_engine,
    pg_dsn: str | None,
    pg_schema: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent Python/Rust transactions commit into one ordered graph log."""
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    pytest.importorskip("kogwistar._rust")
    namespace = f"mixed-writer-{uuid.uuid4().hex}"
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    meta = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    meta.ensure_initialized()

    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_META_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_GRAPH_STORE", "python")
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "python")
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "mixed_python_rust_pg_writers"),
        embedding_function=ConstantEmbeddingFunction(dim=3),
        backend=backend,
        namespace=namespace,
    )
    python_node = build_entity_node(
        node_id="python-writer-node",
        doc_id="python-writer-doc",
        summary="committed through Python authority",
    )
    rust_operation = _mutation(
        event_id="rust-writer-event",
        op="ADD",
        namespace=namespace,
        id="rust-writer-node",
        document="committed through Rust authority",
        metadata={"lifecycle_status": "active"},
        embedding=[0.0, 1.0, 0.0],
    )
    start = Barrier(2)

    def write_python() -> str:
        start.wait(timeout=10)
        engine.write.add_node(python_node)
        return python_node.id

    def write_rust() -> str:
        start.wait(timeout=10)
        result = _rust(dsn, schema, rust_operation)
        assert result["inserted"] is True
        return "rust-writer-node"

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write_python), pool.submit(write_rust)]
            assert {future.result(timeout=30) for future in futures} == {
                "python-writer-node",
                "rust-writer-node",
            }

        events = list(meta.iter_entity_events(namespace=namespace, from_seq=1))
        assert [event[0] for event in events] == [1, 2]
        assert {event[2] for event in events} == {
            "python-writer-node",
            "rust-writer-node",
        }
        assert backend.node_get(
            ids=["python-writer-node", "rust-writer-node"],
            include=["documents", "metadatas"],
        )["ids"] == ["python-writer-node", "rust-writer-node"]
        assert _rust(
            dsn,
            schema,
            {
                "kind": "latest_retained_event_seq",
                "namespace": namespace,
            },
        ) == 2
    finally:
        engine.close()


def test_python_node_and_edge_payloads_round_trip_through_rust_graph_mutation(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    meta = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    meta.ensure_initialized()

    assert PYTHON_GRAPH_MODEL_FIXTURE["schema_version"] == 1
    assert [case["kind"] for case in PYTHON_GRAPH_MODEL_FIXTURE["cases"]] == [
        "node",
        "edge",
        "edge_with_edge_endpoints",
    ]
    entities = [
        (
            case["kind"],
            (Node if case["kind"] == "node" else Edge).model_validate(case["payload"]),
        )
        for case in PYTHON_GRAPH_MODEL_FIXTURE["cases"]
    ]
    node = next(entity for kind, entity in entities if kind == "node")
    edges = [entity for kind, entity in entities if kind != "node"]
    expected_documents: dict[str, str] = {}
    for fixture_kind, entity in entities:
        entity_kind = "node" if fixture_kind == "node" else "edge"
        table = "gke_nodes" if entity_kind == "node" else "gke_edges"
        payload = entity.model_dump(field_mode="backend", dump_format="json")
        excluded_fields = (
            ["embedding", "metadata"] if entity_kind == "node" else ["embedding"]
        )
        document = entity.model_dump_json(
            field_mode="backend",
            exclude=excluded_fields,
        )
        expected_documents[entity.id] = document
        fixture_payload = next(
            case["payload"]
            for case in PYTHON_GRAPH_MODEL_FIXTURE["cases"]
            if case["kind"] == fixture_kind
        )
        assert payload == fixture_payload
        mutation = _mutation(
            event_id=f"{fixture_kind}-model-payload",
            op="ADD",
            id=entity.id,
            document=document,
            metadata=entity.metadata,
            embedding=list(entity.embedding or []),
        )
        mutation.update(
            entity_kind=entity_kind,
            table=table,
            payload=payload,
        )
        _rust(dsn, schema, mutation)

        event = list(meta.iter_entity_events(namespace="alpha", from_seq=1))[-1]
        restored_payload = json.loads(event[4])
        assert restored_payload == payload
        restored = (Node if entity_kind == "node" else Edge).model_validate(
            restored_payload
        )
        assert restored.model_dump(field_mode="backend", dump_format="json") == payload

    assert backend.node_get(
        ids=[node.id], include=["documents", "metadatas"]
    )["documents"] == [expected_documents[node.id]]
    assert backend.edge_get(
        ids=[edge.id for edge in edges], include=["documents", "metadatas"]
    )["documents"] == [expected_documents[edge.id] for edge in edges]


def test_graph_space_scope_preserves_shared_python_rust_node_edge_storage(
    sa_engine, pg_dsn: str | None, pg_schema: str | None
) -> None:
    """Graph-space values scope records without changing the entity contract."""
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    EnginePostgresMetaStore(engine=sa_engine, schema=schema).ensure_initialized()

    graph_spaces = ("facts", "custom:legal", "urn:tenant:graph:6")
    for index, graph_space in enumerate(graph_spaces):
        scope_metadata = {
            "namespace": "alpha",
            "workspace_id": "ws",
            "graph_space": graph_space,
        }
        python_node_id = f"python-{graph_space}-node-{index}"
        python_edge_id = f"python-{graph_space}-edge-{index}"
        backend.node_upsert(
            ids=[python_node_id],
            documents=[python_node_id],
            metadatas=[scope_metadata],
            embeddings=[[1.0, 0.0, 0.0]],
        )
        backend.edge_upsert(
            ids=[python_edge_id],
            documents=[python_edge_id],
            metadatas=[
                {
                    **scope_metadata,
                    "relation": "connects",
                    "source_ids": [python_node_id],
                    "target_ids": [python_node_id],
                }
            ],
            embeddings=[[1.0, 0.0, 0.0]],
        )

        for entity_kind, entity_id, table in (
            ("node", python_node_id, "gke_nodes"),
            ("edge", python_edge_id, "gke_edges"),
        ):
            rust_rows = _rust(
                dsn,
                schema,
                {
                    "kind": "graph_projection_records",
                    "namespace": "alpha",
                    "workspace_id": "ws",
                    "graph_space": graph_space,
                    "table": table,
                    "metadata": {},
                    "limit": 10,
                },
            )
            assert entity_id in {row["id"] for row in rust_rows}

            rust_id = f"rust-{graph_space}-{entity_kind}-{index}"
            mutation = _mutation(
                event_id=f"event-{rust_id}",
                op="ADD",
                graph_space=graph_space,
                id=rust_id,
                document=rust_id,
                metadata={
                    "namespace": "alpha",
                    "workspace_id": "ws",
                    "graph_space": graph_space,
                    **(
                        {
                            "relation": "connects",
                            "source_ids": [python_node_id],
                            "target_ids": [python_node_id],
                            "source_edge_ids": [],
                            "target_edge_ids": [],
                        }
                        if entity_kind == "edge"
                        else {}
                    ),
                },
                embedding=[1.0, 0.0, 0.0],
            )
            mutation.update(
                entity_kind=entity_kind,
                table=table,
                payload={"id": rust_id},
            )
            _rust(dsn, schema, mutation)

            read = (
                backend.node_get
                if entity_kind == "node"
                else backend.edge_get
            )
            python_rows = read(ids=[rust_id], include=["metadatas"])
            assert python_rows["ids"] == [rust_id]
            assert python_rows["metadatas"][0]["graph_space"] == graph_space


def test_rust_reads_python_persisted_wisdom_graph_from_role_schema(
    request: pytest.FixtureRequest,
    sa_engine,
    pg_dsn: str | None,
    pg_schema: str | None,
) -> None:
    """Rust can read Wisdom rows persisted by Python's role-specific backend."""
    dsn, base_schema = _require_pg(pg_dsn, pg_schema)
    wisdom_schema = f"{base_schema}_wisdom"
    request.addfinalizer(lambda: _drop_pg_schema(sa_engine, wisdom_schema))
    backend = PgVectorBackend(
        engine=sa_engine, embedding_dim=3, schema=wisdom_schema
    )
    backend.ensure_schema()

    node_id = "python-wisdom-node"
    edge_id = "python-wisdom-edge"
    node_document = json.dumps(
        {
            "id": node_id,
            "label": "Legacy Python lesson",
            "summary": "A persisted Wisdom node from the Python graph engine.",
            "type": "wisdom_lesson",
        },
        separators=(",", ":"),
    )
    edge_document = json.dumps(
        {
            "id": edge_id,
            "label": "supports",
            "summary": "The lesson supports the observed pattern.",
            "relation": "supports",
            "source_ids": [node_id],
            "target_ids": [node_id],
        },
        separators=(",", ":"),
    )
    backend.node_upsert(
        ids=[node_id],
        documents=[node_document],
        metadatas=[{"namespace": "wisdom", "_class_name": "PureChromaNode"}],
        embeddings=[[1.0, 0.0, 0.0]],
    )
    backend.edge_upsert(
        ids=[edge_id],
        documents=[edge_document],
        metadatas=[
            {
                "namespace": "wisdom",
                "_class_name": "PureChromaEdge",
                "relation": "supports",
                "source_ids": [node_id],
                "target_ids": [node_id],
            }
        ],
        embeddings=[[0.0, 1.0, 0.0]],
    )

    for table, entity_id, expected_document in (
        ("gke_nodes", node_id, node_document),
        ("gke_edges", edge_id, edge_document),
    ):
        rust_rows = _rust(
            dsn,
            wisdom_schema,
            {
                "kind": "graph_projection_records",
                "namespace": "wisdom",
                "table": table,
                "ids": [entity_id],
                "limit": 1,
            },
        )
        assert len(rust_rows) == 1
        assert rust_rows[0]["id"] == entity_id
        assert rust_rows[0]["document"] == expected_document
        assert rust_rows[0]["metadata"]["namespace"] == "wisdom"


@pytest.mark.parametrize("level", [1 << 200, -(1 << 200)])
def test_arbitrary_precision_node_level_round_trips_python_rust_postgres_python(
    sa_engine, pg_dsn: str | None, pg_schema: str | None, level: int
) -> None:
    dsn, schema = _require_pg(pg_dsn, pg_schema)
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=3, schema=schema)
    backend.ensure_schema()
    meta = EnginePostgresMetaStore(engine=sa_engine, schema=schema)
    meta.ensure_initialized()

    base_payload = next(
        case["payload"]
        for case in PYTHON_GRAPH_MODEL_FIXTURE["cases"]
        if case["kind"] == "node"
    )
    node_payload = json.loads(json.dumps(base_payload))
    node_payload["id"] = "arbitrary-precision-level"
    node_payload["level_from_root"] = level
    node_payload["metadata"]["level_from_root"] = level
    node = Node.model_validate(node_payload)
    payload = node.model_dump(field_mode="backend", dump_format="json")
    mutation = _mutation(
        event_id=f"node-level-{level}",
        op="ADD",
        id=node.id,
        document=node.summary,
        metadata=node.metadata,
        embedding=list(node.embedding or []),
    )
    mutation["payload"] = payload

    _rust(dsn, schema, mutation)

    event_payload = next(
        json.loads(event[4])
        for event in meta.iter_entity_events(namespace="alpha", from_seq=1)
        if json.loads(event[4]).get("id") == node.id
    )
    restored = Node.model_validate(event_payload)
    assert restored.level_from_root == level
    assert restored.metadata["level_from_root"] == level

    row = backend.node_get(ids=[node.id], include=["documents", "metadatas"])
    assert row["metadatas"][0]["level_from_root"] == level
    rust_rows = _rust(
        dsn,
        schema,
        {
            "kind": "graph_projection_records",
            "namespace": "alpha",
            "workspace_id": "ws",
            "graph_space": "facts",
            "table": "gke_nodes",
            "metadata": {"level_from_root": level},
            "limit": 10,
        },
    )
    assert rust_rows[0]["metadata"]["level_from_root"] == level
    vector_rows = _rust(
        dsn,
        schema,
        {
            "kind": "graph_projection_vector_query",
            "namespace": "alpha",
            "workspace_id": "ws",
            "graph_space": "facts",
            "table": "gke_nodes",
            "embedding": list(node.embedding or []),
            "embedding_dim": 3,
            "metadata": {"level_from_root": level},
            "metric": "cosine",
            "limit": 10,
        },
    )
    assert vector_rows[0]["record"]["metadata"]["level_from_root"] == level


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
    isolated = _mutation(
        event_id="other-namespace-scope",
        op="ADD",
        namespace="beta",
        id="other-namespace",
        document="other namespace",
        metadata={"team": "red"},
        embedding=[1.0, 0.0, 0.0],
    )
    assert _rust(dsn, schema, isolated)["event"]["seq"] == 1
    for event_id, node_id, workspace_id, graph_space in (
        ("other-workspace-scope", "other-workspace", "other-ws", "facts"),
        ("other-graph-space-scope", "other-graph-space", "ws", "other-facts"),
    ):
        _rust(
            dsn,
            schema,
            _mutation(
                event_id=event_id,
                op="ADD",
                id=node_id,
                document=node_id,
                metadata={"team": "red"},
                embedding=[1.0, 0.0, 0.0],
                workspace_id=workspace_id,
                graph_space=graph_space,
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
    python = backend.node_query(
        query_embeddings=[[1.0, 0.0, 0.0]],
        n_results=10,
        where={
            "namespace": "alpha",
            "workspace_id": "ws",
            "graph_space": "facts",
            "team": "red",
        },
        include=["distances"],
    )
    assert [item["record"]["id"] for item in matches] == python["ids"][0] == ["a", "b"]
    assert [item["distance"] for item in matches] == pytest.approx(python["distances"][0])

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
    assert _rust(dsn, schema, {"kind": "latest_retained_event_seq", "namespace": "alpha"}) == 4
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


def test_python_and_rust_bootstrap_serialize_vector_extension_creation(
    pg_dsn: str | None, pg_container: Any
) -> None:
    if pg_dsn is None or pg_container is None:
        pytest.skip("fresh disposable PostgreSQL database requires the local test container")

    database_name = f"gke_bootstrap_{uuid.uuid4().hex[:16]}"
    python_schema = f"gke_python_{uuid.uuid4().hex[:16]}"
    rust_schema = f"gke_rust_{uuid.uuid4().hex[:16]}"
    admin_url = sa.engine.make_url(pg_dsn).set(database="postgres")
    admin_engine = sa.create_engine(admin_url)
    database_created = False
    try:
        with admin_engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            connection.exec_driver_sql(f'CREATE DATABASE "{database_name}"')
        database_created = True
        target_dsn = admin_url.set(database=database_name).render_as_string(
            hide_password=False
        )
        start = Barrier(2)

        def bootstrap_python() -> None:
            engine = sa.create_engine(target_dsn)
            try:
                backend = PgVectorBackend(
                    engine=engine, embedding_dim=3, schema=python_schema
                )
                start.wait(timeout=10)
                backend.ensure_schema()
            finally:
                engine.dispose()

        def bootstrap_rust() -> Any:
            start.wait(timeout=10)
            return _rust(
                target_dsn,
                rust_schema,
                {"kind": "create_graph_schema", "embedding_dim": 3},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            python_future = executor.submit(bootstrap_python)
            rust_future = executor.submit(bootstrap_rust)
            python_future.result(timeout=60)
            assert rust_future.result(timeout=60) == {"initialized": True}

        target_engine = sa.create_engine(target_dsn)
        try:
            with target_engine.connect() as connection:
                assert connection.scalar(
                    sa.text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ) == 1
                for schema in (python_schema, rust_schema):
                    table_name = f"{schema}.gke_nodes"
                    assert connection.scalar(
                        sa.text("SELECT to_regclass(:table_name) IS NOT NULL").bindparams(
                            table_name=table_name
                        )
                    )
        finally:
            target_engine.dispose()
    finally:
        if database_created:
            with admin_engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as connection:
                connection.execute(
                    sa.text(
                        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                        "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                    ),
                    {"database_name": database_name},
                )
                connection.exec_driver_sql(f'DROP DATABASE "{database_name}"')
        admin_engine.dispose()
