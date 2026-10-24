from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

from kogwistar.engine_core import models
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import (
    PgVectorBackend,
    PostgresUnitOfWork,
)

pytestmark = pytest.mark.ci_full


def _dummy_grounding() -> models.Grounding:
    sp = models.Span.from_dummy_for_document()
    return models.Grounding(spans=[sp])


def _make_node(*, summary: str, label: str, doc_id: str) -> models.Node:
    return models.Node(
        label=label,
        type="entity",
        summary=summary,
        domain_id=None,
        canonical_entity_id=None,
        mentions=[_dummy_grounding()],
        doc_id=doc_id,
        metadata={},
        embedding=None,
    )


def _make_edge(*, src: str, tgt: str, relation: str, doc_id: str) -> models.Edge:
    return models.Edge(
        label=relation,
        type="relationship",
        summary=f"{src} {relation} {tgt}",
        domain_id=None,
        canonical_entity_id=None,
        mentions=[_dummy_grounding()],
        doc_id=doc_id,
        metadata={},
        embedding=None,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=None,
        target_edge_ids=None,
        relation=relation,
    )


def _dummy_embed(dim: int):
    def _emb(texts: list[str]) -> list[list[float]]:
        return [[0.0] * dim for _ in texts]

    return _emb


def test_phase3_e2e_engine_edges_by_doc_pg(
    tmp_path: Path, sa_engine, pg_schema
) -> None:
    if sa_engine is None or pg_schema is None:
        pytest.skip("PostgreSQL fixtures are unavailable")
    dim = 8  # small for tests

    be = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=pg_schema)
    be.ensure_schema()

    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "chroma"),
        embedding_function=_dummy_embed(dim),
        backend=be,
    )

    # Swap to Postgres backend + backend UoW
    eng._backend_uow = PostgresUnitOfWork(engine=sa_engine)

    doc_id = "docA"
    n1 = _make_node(summary="A", label="A", doc_id=doc_id)
    n2 = _make_node(summary="B", label="B", doc_id=doc_id)

    eng.add_node(n1)
    eng.add_node(n2)

    e = _make_edge(
        src=n1.safe_get_id(), tgt=n2.safe_get_id(), relation="links_to", doc_id=doc_id
    )
    eng.add_edge(e)

    got = eng.edges_by_doc(doc_id)
    assert e.safe_get_id() in set(got)


def test_incident_edge_lookup_is_canonical_and_joins_postgres_transaction(
    tmp_path: Path, sa_engine, pg_schema
) -> None:
    import json

    from sqlalchemy import text

    if sa_engine is None or pg_schema is None:
        pytest.skip("PostgreSQL fixtures are unavailable")
    dim = 8
    backend = PgVectorBackend(engine=sa_engine, embedding_dim=dim, schema=pg_schema)
    backend.ensure_schema()
    eng = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "incident_edges_pg"),
        embedding_function=_dummy_embed(dim),
        backend=backend,
    )
    eng._backend_uow = PostgresUnitOfWork(engine=sa_engine)

    left = _make_node(summary="left", label="left", doc_id="incident-doc")
    middle = _make_node(summary="middle", label="middle", doc_id="incident-doc")
    right = _make_node(summary="right", label="right", doc_id="incident-doc")
    eng.write.add_node(left)
    eng.write.add_node(middle)
    eng.write.add_node(right)

    ordinary = _make_edge(
        src=left.safe_get_id(),
        tgt=middle.safe_get_id(),
        relation="ordinary",
        doc_id="incident-doc",
    )
    single_endpoint = _make_edge(
        src=left.safe_get_id(),
        tgt=middle.safe_get_id(),
        relation="single_endpoint",
        doc_id="incident-doc",
    )
    single_endpoint.target_ids = []
    eng.write.add_edge(ordinary)
    eng.write.add_edge(single_endpoint)

    hyperedge = _make_edge(
        src=middle.safe_get_id(),
        tgt=right.safe_get_id(),
        relation="hyperedge",
        doc_id="incident-doc",
    )
    hyperedge.source_edge_ids = [ordinary.safe_get_id()]
    hyperedge.target_edge_ids = [ordinary.safe_get_id()]
    eng.write.add_edge(hyperedge)

    raw_incident = backend.edge_get_by_endpoint(
        endpoint_id=left.safe_get_id(), endpoint_type="node"
    )
    assert ordinary.safe_get_id() in raw_incident["ids"], raw_incident
    edge_table = backend.edges.name
    explain_sql = text(
        f"""EXPLAIN SELECT id FROM "{pg_schema}"."{edge_table}"
        WHERE ((metadata ->> 'source_ids')::jsonb) @> CAST(:endpoint AS jsonb)
           OR ((metadata ->> 'target_ids')::jsonb) @> CAST(:endpoint AS jsonb)"""
    )
    with sa_engine.begin() as connection:
        connection.execute(text("SET LOCAL enable_seqscan = off"))
        plan = connection.execute(
            explain_sql, {"endpoint": json.dumps([left.safe_get_id()])}
        ).scalars().all()
    assert any(
        f"idx_{edge_table}_source_ids_gin" in str(line)
        or f"idx_{edge_table}_target_ids_gin" in str(line)
        for line in plan
    ), plan
    backend.edge_endpoints_delete(where={"edge_id": ordinary.safe_get_id()})
    by_left = eng.read.get_incident_edges(left.safe_get_id())
    assert {edge.safe_get_id() for edge in by_left} == {
        ordinary.safe_get_id(),
        single_endpoint.safe_get_id(),
    }
    by_edge = eng.read.get_incident_edges(
        ordinary.safe_get_id(), endpoint_type="edge"
    )
    assert [edge.safe_get_id() for edge in by_edge] == [hyperedge.safe_get_id()]

    transient = _make_edge(
        src=left.safe_get_id(),
        tgt=right.safe_get_id(),
        relation="transaction-local",
        doc_id="incident-doc",
    )
    with pytest.raises(RuntimeError, match="rollback probe"):
        with eng.uow():
            eng.write.add_edge(transient)
            inside_transaction = eng.read.get_incident_edges(right.safe_get_id())
            assert transient.safe_get_id() in {
                edge.safe_get_id() for edge in inside_transaction
            }
            raise RuntimeError("rollback probe")

    after_rollback = eng.read.get_incident_edges(right.safe_get_id())
    assert transient.safe_get_id() not in {
        edge.safe_get_id() for edge in after_rollback
    }
