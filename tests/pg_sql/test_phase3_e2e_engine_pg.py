from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
pytest.importorskip("sqlalchemy")

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core import models
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
    def _emb(texts: List[str]) -> List[List[float]]:
        return [[0.0] * dim for _ in texts]

    return _emb


def test_phase3_e2e_engine_edges_by_doc_pg(
    tmp_path: Path, sa_engine, pg_schema
) -> None:
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
