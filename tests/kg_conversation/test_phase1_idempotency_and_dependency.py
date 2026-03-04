# tests/kg_conversation/test_phase1_idempotency_and_dependency.py
from __future__ import annotations

import json
import pytest
from typing import Literal
from conversation.models import ConversationEdge, ConversationNode
from engine_core.models import (
    Grounding,
    Span,
    MentionVerification,
)

def _mk_grounding(doc_id: str, excerpt: str = "x") -> Grounding:
    # Minimal valid span/grounding for GraphEntityRefBase validators.
    sp = Span(
        collection_page_url="N/A",
        document_page_url="N/A",
        doc_id=doc_id,
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=len(excerpt),
        excerpt=excerpt,
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test"),
    )
    return Grounding(spans=[sp])

def _mk_turn(*, conversation_id: str, user_id: str, turn_id: str, role: Literal['user', 'assistant', 'system', 'tool'], turn_index: int) -> ConversationNode:
    doc_id = f"conv:{conversation_id}"
    return ConversationNode(
        id=turn_id,
        label=turn_id,
        type="entity",
        summary=f"{role}:{turn_index}",
        mentions=[_mk_grounding(doc_id)],
        doc_id=doc_id,
        metadata={
            "entity_type": f"{role}_turn",
            "in_conversation_chain": True,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "level_from_root":0,
        },
        role=role,
        turn_index=turn_index,
        conversation_id=conversation_id,
        user_id=user_id,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        level_from_root=0
    )

def _mk_next_turn_edge(*, conversation_id: str, src: str, tgt: str) -> ConversationEdge:
    doc_id = f"conv:{conversation_id}"
    return ConversationEdge(
        id=None,  # let engine assign if it wants
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        mentions=[_mk_grounding(doc_id)],
        doc_id=doc_id,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="next_turn",
        metadata={
            "causal_type": "chain",
            "conversation_id": conversation_id,
        },
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )

def _mk_dependency_edge(*, conversation_id: str, src: str, tgt: str) -> ConversationEdge:
    doc_id = f"conv:{conversation_id}"
    return ConversationEdge(
        id=None,
        label="depends_on",
        type="relationship",
        summary="Dependency",
        mentions=[_mk_grounding(doc_id)],
        doc_id=doc_id,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        relation="depends_on",
        metadata={
            "causal_type": "dependency",
            "conversation_id": conversation_id,
        },
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
    )

def _count_next_turn_edges(conversation_engine, *, conversation_id: str, src: str, tgt: str) -> int:
    got = conversation_engine.backend.edge_get(where={"relation": "next_turn"})
    docs = got.get("documents") or []
    cnt = 0
    for d in docs:
        try:
            obj = json.loads(d)
        except Exception:
            continue
        if obj.get("doc_id") != f"conv:{conversation_id}":
            continue
        if obj.get("source_ids") == [src] and obj.get("target_ids") == [tgt]:
            cnt += 1
    return cnt

def test_next_turn_duplicate_is_idempotent_in_add_edge(conversation_engine):
    """Adding the exact same next_turn edge twice should be a NOOP (idempotent), not a failure."""
    conversation_id = "conv_idem_edge"
    user_id = "u"
    t1 = _mk_turn(conversation_id=conversation_id, user_id=user_id, turn_id="turn1", role="user", turn_index=1)
    t2 = _mk_turn(conversation_id=conversation_id, user_id=user_id, turn_id="turn2", role="assistant", turn_index=2)

    conversation_engine.add_node(t1)
    conversation_engine.add_node(t2)

    e = _mk_next_turn_edge(conversation_id=conversation_id, src=t1.id, tgt=t2.id)
    conversation_engine.add_edge(e)

    # Duplicate write attempt: should NOT raise.
    e_dup = _mk_next_turn_edge(conversation_id=conversation_id, src=t1.id, tgt=t2.id)
    conversation_engine.add_edge(e_dup)

    assert _count_next_turn_edges(conversation_engine, conversation_id=conversation_id, src=t1.id, tgt=t2.id) == 1

def test_next_turn_duplicate_is_not_idempotent_in_add_pure_edge(conversation_engine):
    conversation_id = "conv_idem_pure"
    user_id = "u"
    t1 = _mk_turn(conversation_id=conversation_id, user_id=user_id, turn_id="turn1", role="user", turn_index=1)
    t2 = _mk_turn(conversation_id=conversation_id, user_id=user_id, turn_id="turn2", role="assistant", turn_index=2)

    conversation_engine.add_node(t1)
    conversation_engine.add_node(t2)

    e = _mk_next_turn_edge(conversation_id=conversation_id, src=t1.id, tgt=t2.id)
    conversation_engine.add_pure_edge(e)

    e_dup = _mk_next_turn_edge(conversation_id=conversation_id, src=t1.id, tgt=t2.id)
    conversation_engine.add_pure_edge(e_dup)

    assert _count_next_turn_edges(conversation_engine, conversation_id=conversation_id, src=t1.id, tgt=t2.id) == 2

def test_dependency_freeze_rejects_new_incoming_into_used_node_without_scanning_all_edges(conversation_engine, monkeypatch):
    """If a node has produced outputs (chain/dependency outgoing), it is 'used'.
    Adding NEW dependency-incoming edges into it must fail, and validation must not scan all edges."""
    conversation_id = "conv_dep_freeze_v2"
    user_id = "u"
    t1 = _mk_turn(conversation_id=conversation_id, user_id=user_id, turn_id="turn1", role="user", turn_index=1)
    t2 = _mk_turn(conversation_id=conversation_id, user_id=user_id, turn_id="turn2", role="assistant", turn_index=2)

    conversation_engine.add_node(t1)
    conversation_engine.add_node(t2)

    # Mark t1 as 'used' by giving it an outgoing chain edge.
    conversation_engine.add_edge(_mk_next_turn_edge(conversation_id=conversation_id, src=t1.id, tgt=t2.id))

    def _fail_get_edges(*args, **kwargs):
        raise AssertionError("get_edges() should not be called for dependency-freeze validation")

    monkeypatch.setattr(conversation_engine, "get_edges", _fail_get_edges, raising=True)

    # New dependency INCOMING into used node t1 must be rejected.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(_mk_dependency_edge(conversation_id=conversation_id, src=t2.id, tgt=t1.id))
