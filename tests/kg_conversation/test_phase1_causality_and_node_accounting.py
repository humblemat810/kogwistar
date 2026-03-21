import pytest

pytestmark = pytest.mark.ci
from pydantic import ValidationError

from kogwistar.conversation.models import (
    ConversationEdge,
    ConversationNodeMetadata,
)
from kogwistar.engine_core.models import Span, Grounding
from kogwistar.engine_core.engine import GraphKnowledgeEngine


def test_node_metadata_forbids_summary_distance_fields():
    with pytest.raises(ValidationError):
        ConversationNodeMetadata.model_validate(
            {
                "entity_type": "conversation_turn",
                "level_from_root": 0,
                "in_conversation_chain": True,
                "char_distance_from_last_summary": 10,
                "turn_distance_from_last_summary": 2,
            }
        )


def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return sp


def _mk_edge(src: str, tgt: str) -> ConversationEdge:
    # Avoid heavy validation; we only need fields used by Phase-1 invariants.
    return ConversationEdge.model_construct(
        id="e",
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="next_turn",
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        doc_id="conv:c",
        mentions=[Grounding(spans=[_mk_span(f"{src}->{tgt}")])],
        properties={},
        embedding=None,
        metadata={"relation": "next_turn", "target_id": tgt, "causal_type": "chain"},
        domain_id=None,
        canonical_entity_id=None,
    )


def _flatten_and(where: dict | None) -> dict:
    if not where:
        return {}
    if "$and" not in where:
        return dict(where)
    out: dict = {}
    for term in where.get("$and", []):
        if isinstance(term, dict):
            out.update(term)
    return out


def _bind(eng: GraphKnowledgeEngine, *, existing_endpoints: list[dict]):
    from unittest.mock import MagicMock
    from kogwistar.conversation.service import ConversationService

    # Mock backend to support edge_endpoints_get
    class MockBackend:
        def edge_endpoints_get(self, *, where: dict, **kwargs):
            w = _flatten_and(where)
            found = []
            for row in existing_endpoints:
                ok = True
                for k, v in w.items():
                    if row.get(k) != v:
                        ok = False
                        break
                if ok:
                    found.append(row)
            return {"ids": [f"ep_{i}" for i in range(len(found))], "metadatas": found}

        def node_get(self, **kwargs):
            return {"ids": [], "metadatas": []}

        def edge_get(self, **kwargs):
            return {"ids": [], "metadatas": [], "documents": []}

    eng.backend = MockBackend()  # type: ignore
    eng.kg_graph_type = "conversation"
    eng.llm_tasks = MagicMock()

    svc = ConversationService(
        conversation_engine=eng,
        knowledge_engine=eng,
    )
    return svc


def test_conversation_engine_next_turn_outgoing_uniqueness_enforced():
    eng = GraphKnowledgeEngine.__new__(GraphKnowledgeEngine)

    # Existing edge A -> B implies an outgoing next_turn endpoint for src=A
    existing_endpoints = [
        {
            "doc_id": "conv:c",
            "relation": "next_turn",
            "role": "src",
            "endpoint_type": "node",
            "endpoint_id": "A",
        },
        {
            "doc_id": "conv:c",
            "relation": "next_turn",
            "role": "tgt",
            "endpoint_type": "node",
            "endpoint_id": "B",
        },
    ]
    svc = _bind(eng, existing_endpoints=existing_endpoints)
    with pytest.raises(ValueError):
        svc._validate_conversation_edge_add(_mk_edge("A", "C"))


def test_conversation_engine_next_turn_incoming_uniqueness_enforced():
    eng = GraphKnowledgeEngine.__new__(GraphKnowledgeEngine)

    # Existing edge A -> B implies an incoming next_turn endpoint for tgt=B
    existing_endpoints = [
        {
            "doc_id": "conv:c",
            "relation": "next_turn",
            "role": "src",
            "endpoint_type": "node",
            "endpoint_id": "A",
        },
        {
            "doc_id": "conv:c",
            "relation": "next_turn",
            "role": "tgt",
            "endpoint_type": "node",
            "endpoint_id": "B",
        },
    ]
    svc = _bind(eng, existing_endpoints=existing_endpoints)
    with pytest.raises(ValueError):
        svc._validate_conversation_edge_add(_mk_edge("C", "B"))
