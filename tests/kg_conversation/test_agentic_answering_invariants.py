from __future__ import annotations
import pytest
pytestmark = pytest.mark.ci_full

from dataclasses import dataclass
from typing import Any, Dict, List


from kogwistar.conversation.models import MetaFromLastSummary
from kogwistar.conversation.agentic_answering import pointer_id
from kogwistar.engine_core.models import (
    Span,
    MentionVerification,
    Grounding,
)


def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="DOC_TEST",
        insertion_method="unit_test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


class _Backend:
    """In-memory backend stub for conversation + KG backends used by projection tests."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}

    # ---- Nodes
    def node_get(self, *, ids: List[str], include: List[str]) -> Dict[str, Any]:
        out_ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        for _id in ids:
            if _id in self.nodes:
                out_ids.append(_id)
                if "metadatas" in include:
                    metas.append(self.nodes[_id].get("metadata", {}))
        resp: Dict[str, Any] = {"ids": out_ids}
        if "metadatas" in include:
            resp["metadatas"] = metas
        return resp

    def node_add(
        self,
        *,
        ids: List[str],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
        embeddings=None,
    ):
        for _id, md, doc in zip(ids, metadatas, documents):
            self.nodes[_id] = {"metadata": md, "document": doc}

    # ---- Edges (binary edge storage for invariants tests)
    def edge_get(self, *, ids: List[str], include: List[str]) -> Dict[str, Any]:
        out_ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        for _id in ids:
            if _id in self.edges:
                out_ids.append(_id)
                if "metadatas" in include:
                    metas.append(self.edges[_id].get("metadata", {}))
        resp: Dict[str, Any] = {"ids": out_ids}
        if "metadatas" in include:
            resp["metadatas"] = metas
        return resp

    def edge_add(
        self,
        *,
        ids: List[str],
        sources: List[str],
        targets: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        for _id, s, t, md in zip(ids, sources, targets, metadatas):
            self.edges[_id] = {"source": s, "target": t, "metadata": md}


@dataclass
class _Engine:
    backend: _Backend

    def __init__(self, _backend=None):
        self.read = self
        self.write = self
        self.edges = []
        self.nodes = []
        if _backend:
            self.backend = _backend

    def add_edge(self, edge):
        self.edges.append(edge)
        # self.backend.edge_add(edge)

    def get_edge(self, *arg, **kwargs):
        return self.edges

    def get_edges(self, *arg, **kwargs):
        return self.edges

    def get_node(self, *arg, **kwargs):
        return self.nodes

    def get_nodes(self, *arg, **kwargs):
        return self.nodes

    def add_node(self, node):
        self.nodes.append(node)
        # self.backend.node_add(node)


class _AgentWithProjection:
    """Minimal agent shim exposing the projection methods we want to enforce.

    In the real repo, these are implemented on AgenticAnsweringAgent.
    """

    def __init__(self) -> None:
        self.conversation_engine = _Engine(_Backend())
        self.knowledge_engine = _Engine(_Backend())

    # Note: we do NOT implement _project_kg_node here; tests import the real agent if available.


def test_pointer_id_is_deterministic() -> None:
    a = pointer_id(
        scope="conv:abc", pointer_kind="kg_node", target_kind="node", target_id="N1"
    )
    b = pointer_id(
        scope="conv:abc", pointer_kind="kg_node", target_kind="node", target_id="N1"
    )
    c = pointer_id(
        scope="conv:abc", pointer_kind="kg_node", target_kind="node", target_id="N2"
    )
    assert a == b
    assert a != c


def test_edge_endpoint_projection_uses_same_node_projection_function() -> None:
    """Invariant: projected edge endpoints must be the projected node ids derived from KG node ids.

    This test enforces the contract by *construction*: we compute the expected
    projected endpoint ids using pointer_id(...) and require the projection
    helper to use the same mapping.
    """
    scope = "conv:conv1"
    kg_src = "KG_NODE_SRC"
    kg_dst = "KG_NODE_DST"

    expected_src = pointer_id(
        scope=scope, pointer_kind="kg_node", target_kind="node", target_id=kg_src
    )
    expected_dst = pointer_id(
        scope=scope, pointer_kind="kg_node", target_kind="node", target_id=kg_dst
    )

    # A projected edge should use those exact ids as its endpoints.
    # We assert this by checking the endpoint ids the agent writes.
    from kogwistar.conversation.agentic_answering import (
        AgenticAnsweringAgent,
    )

    # If the repo hasn't implemented _project_kg_edge yet, this should fail loudly.
    assert hasattr(AgenticAnsweringAgent, "_project_kg_edge"), (
        "Missing AgenticAnsweringAgent._project_kg_edge. "
        "Implement edge projection with deterministic endpoints (dangling allowed)."
    )

    # Create a minimal AgenticAnsweringAgent instance by bypassing heavy init:
    agent = object.__new__(AgenticAnsweringAgent)
    agent.conversation_engine = _Engine(_Backend())
    agent.knowledge_engine = _Engine(_Backend())

    # Seed KG edge in knowledge backend (must include metadatas for endpoints resolution)
    kg_edge_id = "KG_EDGE_1"
    agent.knowledge_engine.backend.edges[kg_edge_id] = {
        "source_ids": kg_src,
        "target_ids": kg_dst,
        "metadata": {
            "source_ids": [kg_src],
            "target_ids": [kg_dst],
            "kg_edge_id": kg_edge_id,
        },
    }

    run_node_id = "RUN1"
    pid_edge = agent._project_kg_edge(
        conversation_id="conv1",
        run_node_id=run_node_id,
        kg_edge_id=kg_edge_id,
        provenance_span=_span(),
        prev_turn_meta_summary=MetaFromLastSummary(
            0, 0, 0
        ),  # or provide real MetaFromLastSummary in repo tests
    )

    # The projection should have created a projected edge (or edge pointer) using deterministic ids.
    # We check that its stored endpoints are the expected projected node ids.

    assert pid_edge in [i.id for i in agent.conversation_engine.get_edges()], (
        "Projected edge must exist in conversation engine"
    )
    stored = [i for i in agent.conversation_engine.get_edges() if i.id == pid_edge][0]
    assert stored.source_ids[0] == expected_src
    assert stored.target_ids[0] == expected_dst


def test_edge_projection_allows_dangling_endpoints() -> None:
    """Invariant: edge projection must NOT require endpoint nodes to already exist."""
    from kogwistar.conversation.agentic_answering import (
        AgenticAnsweringAgent,
    )

    assert hasattr(AgenticAnsweringAgent, "_project_kg_edge"), (
        "Missing AgenticAnsweringAgent._project_kg_edge. "
        "Dangling endpoints should be allowed (do not auto-create endpoint nodes)."
    )

    agent = object.__new__(AgenticAnsweringAgent)
    agent.conversation_engine = _Engine(_Backend())
    agent.knowledge_engine = _Engine(_Backend())

    kg_src = "KG_NODE_SRC"
    kg_dst = "KG_NODE_DST"
    kg_edge_id = "KG_EDGE_2"
    agent.knowledge_engine.backend.edges[kg_edge_id] = {
        "source_ids": [kg_src],
        "target_ids": [kg_dst],
        "metadata": {
            "source_ids": [kg_src],
            "target_ids": [kg_dst],
            "kg_edge_id": kg_edge_id,
        },
    }

    pid_edge = agent._project_kg_edge(
        conversation_id="conv2",
        run_node_id="RUN2",
        kg_edge_id=kg_edge_id,
        provenance_span=_span(),
        prev_turn_meta_summary=MetaFromLastSummary(0, 0, 0),
    )

    # Ensure we did not create endpoint nodes automatically
    engine = agent.conversation_engine
    assert len(engine.get_nodes()) == 0, (
        "Edge projection should not auto-create endpoint pointer nodes"
    )
    assert pid_edge in [i.id for i in engine.get_edges()]


def test_projected_kg_node_is_pointer_only_and_kg_targeted() -> None:
    """Invariant: conversation projection of KG nodes must stay pointer-only."""
    from kogwistar.conversation.agentic_answering import AgenticAnsweringAgent

    assert hasattr(AgenticAnsweringAgent, "_project_kg_node"), (
        "Missing AgenticAnsweringAgent._project_kg_node. "
        "Project KG nodes into conversation as pointer nodes only."
    )

    agent = object.__new__(AgenticAnsweringAgent)
    agent.conversation_engine = _Engine(_Backend())
    agent.knowledge_engine = _Engine(_Backend())

    kg_node_id = "KG_NODE_POINTER_1"
    agent.knowledge_engine.backend.nodes[kg_node_id] = {
        "metadata": {
            "label": "KG Node",
            "summary": "Durable knowledge node",
            "type": "entity",
            "canonical_entity_id": kg_node_id,
        }
    }

    run_node_id = "RUN3"
    pid_node = agent._project_kg_node(
        conversation_id="conv3",
        run_node_id=run_node_id,
        kg_node_id=kg_node_id,
        provenance_span=_span(),
        prev_turn_meta_summary=MetaFromLastSummary(0, 0, 0),
    )

    nodes = agent.conversation_engine.get_nodes()
    assert pid_node in [i.id for i in nodes]
    stored = [i for i in nodes if i.id == pid_node][0]
    assert stored.type == "reference_pointer"
    assert stored.properties["target_namespace"] == "kg"
    assert stored.properties["target_id"] == kg_node_id
    assert stored.properties["refers_to_collection"] == "nodes"


def test_evidence_pack_digest_includes_edges_if_supported() -> None:
    """Invariant: if EvidencePackDigest supports edge_ids, materialization must propagate them.

    This enforces the 'edges are evidence' design. If your repo intentionally
    keeps edges as non-evidence, delete this test.
    """
    from kogwistar.conversation.models import EvidencePackDigest  # type: ignore

    d = EvidencePackDigest(
        node_ids=["N1"],
        edge_ids=["E1"],
        depth="shallow",
        max_chars_per_item=200,
        max_total_chars=500,
        evidence_pack_hash="h",
    )
    assert d.edge_ids == ["E1"]
