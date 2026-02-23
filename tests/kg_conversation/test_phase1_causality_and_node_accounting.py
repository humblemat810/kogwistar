
import pytest
from pydantic import ValidationError

from graph_knowledge_engine.models import ConversationNodeMetadata, ConversationEdge
from graph_knowledge_engine.engine import GraphKnowledgeEngine


def test_node_metadata_forbids_summary_distance_fields():
    with pytest.raises(ValidationError):
        ConversationNodeMetadata.model_validate({
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
            "char_distance_from_last_summary": 10,
            "turn_distance_from_last_summary": 2,
        })


def _mk_edge(src, tgt):
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


def _bind(eng: GraphKnowledgeEngine):
    eng._normalize_conversation_edge_metadata = GraphKnowledgeEngine._normalize_conversation_edge_metadata.__get__(eng)
    eng._validate_conversation_edge_add = GraphKnowledgeEngine._validate_conversation_edge_add.__get__(eng)


def test_next_turn_outgoing_uniqueness_enforced():
    eng = GraphKnowledgeEngine.__new__(GraphKnowledgeEngine)

    existing = [_mk_edge("A", "B")]
    eng.get_edges = lambda where=None, limit=20000, **kw: existing if (where == {"relation": "next_turn"}) else existing
    _bind(eng)

    with pytest.raises(ValueError):
        eng._validate_conversation_edge_add(_mk_edge("A", "C"))


def test_next_turn_incoming_uniqueness_enforced():
    eng = GraphKnowledgeEngine.__new__(GraphKnowledgeEngine)

    existing = [_mk_edge("A", "B")]
    eng.get_edges = lambda where=None, limit=20000, **kw: existing if (where == {"relation": "next_turn"}) else existing
    _bind(eng)

    with pytest.raises(ValueError):
        eng._validate_conversation_edge_add(_mk_edge("C", "B"))
