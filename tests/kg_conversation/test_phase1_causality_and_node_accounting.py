import pytest
from pydantic import ValidationError

from graph_knowledge_engine.models import ConversationNodeMetadata, ConversationEdge
from graph_knowledge_engine.engine import GraphKnowledgeEngine


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
        mentions=[],
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
    # Bind the real normalization + validation implementations
    eng._normalize_conversation_edge_metadata = GraphKnowledgeEngine._normalize_conversation_edge_metadata.__get__(eng)
    eng._validate_conversation_edge_add = GraphKnowledgeEngine._validate_conversation_edge_add.__get__(eng)

    # Minimal where-and helper (matches engine semantics for Phase-1)
    if not hasattr(eng, "_where_and"):
        def _where_and(*parts):
            terms = [p for p in parts if isinstance(p, dict) and p]
            return {"$and": terms} if len(terms) > 1 else (terms[0] if terms else {})
        eng._where_and = _where_and  # type: ignore

    # Endpoint existence check is what Phase-1 validator uses (no full scans)
    def _edge_endpoints_exists(*, where: dict) -> bool:
        w = _flatten_and(where)
        for row in existing_endpoints:
            ok = True
            for k, v in w.items():
                if row.get(k) != v:
                    ok = False
                    break
            if ok:
                return True
        return False

    eng._edge_endpoints_exists = _edge_endpoints_exists  # type: ignore


def test_conversation_engine_next_turn_outgoing_uniqueness_enforced():
    eng = GraphKnowledgeEngine.__new__(GraphKnowledgeEngine)

    # Existing edge A -> B implies an outgoing next_turn endpoint for src=A
    existing_endpoints = [
        {"doc_id": "conv:c", "relation": "next_turn", "role": "src", "endpoint_type": "node", "endpoint_id": "A"},
        {"doc_id": "conv:c", "relation": "next_turn", "role": "tgt", "endpoint_type": "node", "endpoint_id": "B"},
    ]
    _bind(eng, existing_endpoints=existing_endpoints)
    eng.kg_graph_type = "conversation"
    with pytest.raises(ValueError):
        eng._validate_conversation_edge_add(_mk_edge("A", "C"))


def test_conversation_engine_next_turn_incoming_uniqueness_enforced():
    eng = GraphKnowledgeEngine.__new__(GraphKnowledgeEngine)

    # Existing edge A -> B implies an incoming next_turn endpoint for tgt=B
    existing_endpoints = [
        {"doc_id": "conv:c", "relation": "next_turn", "role": "src", "endpoint_type": "node", "endpoint_id": "A"},
        {"doc_id": "conv:c", "relation": "next_turn", "role": "tgt", "endpoint_type": "node", "endpoint_id": "B"},
    ]
    _bind(eng, existing_endpoints=existing_endpoints)
    eng.kg_graph_type = "conversation"
    with pytest.raises(ValueError):
        eng._validate_conversation_edge_add(_mk_edge("C", "B"))
