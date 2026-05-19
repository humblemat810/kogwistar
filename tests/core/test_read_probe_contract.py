from __future__ import annotations

import pytest

from kogwistar.engine_core import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Edge, Grounding, MentionVerification, Node, Span


pytestmark = [pytest.mark.core]


class _TinyEmbeddingFunction:
    def name(self) -> str:
        return "tiny-test-embedding"

    def __call__(self, documents: list[str]) -> list[list[float]]:
        return [[float(len(text) + 1), float(sum(ord(ch) for ch in text) % 97 + 1)] for text in documents]


def _engine(tmp_path) -> GraphKnowledgeEngine:
    return GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_in_memory_backend,
        embedding_function=_TinyEmbeddingFunction(),
    )


def _span(doc_id: str) -> Span:
    return Span(
        doc_id=doc_id,
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human",
            is_verified=True,
            score=1.0,
            notes="test",
        ),
        collection_page_url="N/A",
        document_page_url="N/A",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
    )


def _grounding(doc_id: str) -> Grounding:
    return Grounding(spans=[_span(doc_id)])


def _node(
    *,
    node_id: str,
    doc_id: str,
    label: str,
    summary: str,
    entity_type: str,
    embedding: list[float],
    metadata: dict[str, object],
) -> Node:
    return Node(
        id=node_id,
        label=label,
        type="entity",
        summary=summary,
        doc_id=doc_id,
        mentions=[_grounding(doc_id)],
        metadata={"level_from_root": 0, "entity_type": entity_type, **metadata},
        embedding=embedding,
        level_from_root=0,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def _edge(
    *,
    edge_id: str,
    src: str,
    tgt: str,
    doc_id: str,
    label: str,
    summary: str,
    relation: str,
    entity_type: str,
    embedding: list[float],
    metadata: dict[str, object],
) -> Edge:
    return Edge(
        id=edge_id,
        label=label,
        type="relationship",
        summary=summary,
        relation=relation,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        doc_id=doc_id,
        mentions=[_grounding(doc_id)],
        metadata={"level_from_root": 0, "entity_type": entity_type, **metadata},
        embedding=embedding,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
    )


def test_probe_reads_work_without_embeddings(tmp_path, monkeypatch):
    engine = _engine(tmp_path)

    async def node_get(*, ids=None, where=None, limit=None, include=None):
        del limit, include
        if ids is not None:
            ids = [str(item) for item in ids]
            if "probe-node" not in ids:
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": None}
        if where is not None and where.get("entity_type") != "probe":
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": None}
        return {
            "ids": ["probe-node"],
            "documents": [None],
            "metadatas": [{"entity_type": "probe", "kind": "node"}],
            "embeddings": None,
        }

    async def edge_get(*, ids=None, where=None, limit=None, include=None):
        del limit, include
        if ids is not None:
            ids = [str(item) for item in ids]
            if "probe-edge" not in ids:
                return {"ids": [], "documents": [], "metadatas": [], "embeddings": None}
        if where is not None and where.get("entity_type") != "probe":
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": None}
        return {
            "ids": ["probe-edge"],
            "documents": [None],
            "metadatas": [{"entity_type": "probe", "kind": "edge"}],
            "embeddings": None,
        }

    monkeypatch.setattr(engine.backend, "node_get", node_get)
    monkeypatch.setattr(engine.backend, "edge_get", edge_get)

    assert engine.read.node_exists(ids=["probe-node"]) is True
    assert engine.read.node_exists(where={"entity_type": "probe"}) is True
    assert engine.read.get_node_metadatas(ids=["probe-node"]) == [{"entity_type": "probe", "kind": "node"}]

    assert engine.read.edge_exists(ids=["probe-edge"]) is True
    assert engine.read.edge_exists(where={"entity_type": "probe"}) is True
    assert engine.read.get_edge_metadatas(ids=["probe-edge"]) == [{"entity_type": "probe", "kind": "edge"}]


def test_probe_id_existence_uses_numeric_limit(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    seen_limits: list[int | None] = []

    async def node_get(*, ids=None, where=None, limit=None, include=None):
        del where, include
        seen_limits.append(limit)
        return {"ids": [str(item) for item in ids or []], "documents": [], "metadatas": [], "embeddings": None}

    async def edge_get(*, ids=None, where=None, limit=None, include=None):
        del where, include
        seen_limits.append(limit)
        return {"ids": [str(item) for item in ids or []], "documents": [], "metadatas": [], "embeddings": None}

    monkeypatch.setattr(engine.backend, "node_get", node_get)
    monkeypatch.setattr(engine.backend, "edge_get", edge_get)

    assert engine.read.node_exists(ids=["probe-node-a", "probe-node-b"]) is True
    assert engine.read.edge_exists(ids=["probe-edge-a", "probe-edge-b"]) is True
    assert seen_limits == [2, 2]


def test_hydrated_reads_remain_unchanged(tmp_path):
    engine = _engine(tmp_path)
    node = _node(
        node_id="hydrated-node",
        doc_id="doc-1",
        label="Hydrated node",
        summary="Hydrated summary",
        entity_type="probe",
        embedding=[0.2, 0.8],
        metadata={"probe_kind": "node"},
    )
    edge = _edge(
        edge_id="hydrated-edge",
        src=node.id,
        tgt=node.id,
        doc_id="doc-1",
        label="Hydrated edge",
        summary="Hydrated summary",
        relation="related_to",
        entity_type="probe",
        embedding=[0.4, 0.6],
        metadata={"probe_kind": "edge"},
    )

    engine.write.add_node(node)
    engine.write.add_edge(edge)

    hydrated_nodes = engine.read.get_nodes(ids=[node.id])
    hydrated_edges = engine.read.get_edges(ids=[edge.id])

    assert len(hydrated_nodes) == 1
    assert hydrated_nodes[0].id == node.id
    assert hydrated_nodes[0].label == "Hydrated node"
    assert hydrated_nodes[0].embedding == [0.2, 0.8]
    assert hydrated_nodes[0].metadata["probe_kind"] == "node"

    assert len(hydrated_edges) == 1
    assert hydrated_edges[0].id == edge.id
    assert hydrated_edges[0].label == "Hydrated edge"
    assert hydrated_edges[0].relation == "related_to"
    assert hydrated_edges[0].source_ids == [node.id]
    assert hydrated_edges[0].target_ids == [node.id]
    assert hydrated_edges[0].embedding == [0.4, 0.6]
