from __future__ import annotations

from typing import Any, Dict, List

from graph_knowledge_engine.postgres_backend import PgVectorBackend


def _dummy_ref(*, doc_id: str, edge_id: str) -> tuple[str, str, Dict[str, Any], List[float] | None]:
    _id = f"ref|{doc_id}|{edge_id}"
    doc = f'{{"doc_id":"{doc_id}","edge_id":"{edge_id}"}}'
    meta = {"doc_id": doc_id, "edge_id": edge_id, "entity_type": "edge_ref"}
    emb = None
    return _id, doc, meta, emb


def test_phase3_pg_backend_edge_refs_roundtrip(sa_engine, pg_schema) -> None:
    be = PgVectorBackend(engine=sa_engine, embedding_dim=8, schema=pg_schema)
    be.ensure_schema()

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for edge_id in ("e1", "e2", "e3"):
        _id, doc, meta, _ = _dummy_ref(doc_id="docA", edge_id=edge_id)
        ids.append(_id)
        docs.append(doc)
        metas.append(meta)

    # embeddings kwarg is accepted and ignored for edge_refs in our backend
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