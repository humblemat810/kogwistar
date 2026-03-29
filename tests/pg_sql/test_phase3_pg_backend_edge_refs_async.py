from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytestmark = pytest.mark.ci_full
pytest.importorskip("sqlalchemy")

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from tests.conftest import _make_async_engine


@pytest.fixture(params=[pytest.param("pg", id="async-pg")])
def async_pg_engine(
    request: pytest.FixtureRequest, tmp_path
) -> GraphKnowledgeEngine:
    eng = _make_async_engine(
        backend_kind=str(request.param),
        request=request,
        tmp_path=tmp_path,
        dim=8,
        graph_kind="knowledge",
    )
    return eng


def _dummy_ref(
    *, doc_id: str, edge_id: str
) -> tuple[str, str, Dict[str, Any], List[float] | None]:
    _id = f"ref|{doc_id}|{edge_id}"
    doc = f'{{"doc_id":"{doc_id}","edge_id":"{edge_id}"}}'
    meta = {"doc_id": doc_id, "edge_id": edge_id, "entity_type": "edge_ref"}
    emb = None
    return _id, doc, meta, emb


def test_phase3_pg_backend_edge_refs_roundtrip_async(async_pg_engine) -> None:
    be = async_pg_engine.backend
    assert isinstance(be, PgVectorBackend)
    be.ensure_schema()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for edge_id in ("e1", "e2", "e3"):
        _id, doc, meta, _ = _dummy_ref(doc_id="docA", edge_id=edge_id)
        ids.append(_id)
        docs.append(doc)
        metas.append(meta)

    be.edge_refs_add(ids=ids, documents=docs, metadatas=metas, embeddings=None)

    got = be.edge_refs_get(
        where={"$and": [{"doc_id": "docA"}, {"edge_id": {"$in": ["e1", "e3"]}}]},
        include=["documents", "metadatas"],
        limit=100,
    )

    got_ids = set(got.get("ids") or [])
    assert any("e1" in _id for _id in got_ids)
    assert any("e3" in _id for _id in got_ids)
    assert all("e2" not in _id for _id in got_ids)

    metas_out = got.get("metadatas") or []
    assert all(isinstance(m, dict) for m in metas_out)


def test_phase3_pg_backend_node_update_metadata_merge_async(async_pg_engine) -> None:
    """Smoke test for JSONB merge update on vector collections."""

    be = async_pg_engine.backend
    assert isinstance(be, PgVectorBackend)
    be.ensure_schema()

    _id = "n1"
    doc = "node one"
    meta0: Dict[str, Any] = {"a": 1, "keep": "x", "nested": {"x": 1}}

    be.node_add(ids=[_id], documents=[doc], metadatas=[meta0], embeddings=None)

    patch: Dict[str, Any] = {"a": 2, "b": 3, "nested": {"y": 2}}
    be.node_update(ids=[_id], metadatas=[patch])

    got = be.node_get(ids=[_id], include=["metadatas", "documents"], limit=10)
    metas_out: List[Dict[str, Any]] = got.get("metadatas") or []
    assert len(metas_out) == 1

    m = metas_out[0]
    assert m["keep"] == "x"
    assert m["a"] == 2
    assert m["b"] == 3
    assert m["nested"] == {"y": 2}
