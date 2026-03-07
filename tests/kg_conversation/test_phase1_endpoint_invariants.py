# tests/kg_conversation/test_phase1_endpoint_invariants.py
from __future__ import annotations

import pytest

from graph_knowledge_engine.conversation.models import ConversationEdge, MetaFromLastSummary
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.engine_core.models import Span, Grounding
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
def _mk_span(doc_id: str) -> Span:
    sp = Span.from_dummy_for_document()
    sp.doc_id = doc_id
    return sp


def _mk_edge(*, src: str, tgt: str, relation: str, causal_type: str, doc_id: str) -> ConversationEdge:
    # ConversationEdgeMetadata requires distance fields; for Phase-1 we keep them on edges.
    return ConversationEdge(
        id=None,
        label=relation,
        type="relationship",
        summary=relation,
        source_ids=[src],
        target_ids=[tgt],
        source_edge_ids=[],
        target_edge_ids=[],
        relation=relation,
        doc_id=doc_id,
        mentions=[Grounding(spans=[_mk_span(f"{src}->{tgt}")])],
        properties={},
        metadata={
            "entity_type": "conversation_edge",
            "causal_type": causal_type,
            "char_distance_from_last_summary": 0,
            "turn_distance_from_last_summary": 0,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    )

def _noop_filtering_callback(*_args, **_kwargs):
    from graph_knowledge_engine.conversation.models import FilteringResult

    return FilteringResult(node_ids=[], edge_ids=[]), "noop"


def _mk_three_turns(conversation_engine: GraphKnowledgeEngine, kg_engine: GraphKnowledgeEngine, *, user_id: str, conv_id: str,
                    #causal_type: str="chain"
                    ):
    from graph_knowledge_engine.id_provider import stable_id
    conversation_engine.tool_call_id_factory = stable_id
    kg_engine.tool_call_id_factory = stable_id
    """Create 3 turns using the real public API + orchestrator.

    We intentionally use add_conversation_turn (not direct node creation) to satisfy
    ConversationNode shape requirements and to ensure any automatic next_turn edge
    creation happens as in production.
    """
    svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=kg_engine,
    )
    conv_id, _start_id = svc.create_conversation(user_id, conv_id)

    # Turn 1 (user)
    r1 = svc.add_conversation_turn(
        user_id,
        conv_id,
        turn_id="turn_1_user",
        mem_id="mem_1",
        role="user",
        content="u1",
        ref_knowledge_engine=kg_engine,
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
    )

    # Turn 2 (assistant/system)
    r2 = svc.add_conversation_turn(
        user_id,
        conv_id,
        turn_id="turn_2_assistant",
        mem_id="mem_2",
        role="assistant",
        content="a1",
        ref_knowledge_engine=kg_engine,
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
        prev_turn_meta_summary=r1.prev_turn_meta_summary,
    )

    # Turn 3 (user)
    r3 = svc.add_conversation_turn(
        user_id,
        conv_id,
        turn_id="turn_3_user",
        mem_id="mem_3",
        role="user",
        content="u2",
        ref_knowledge_engine=kg_engine,
        filtering_callback=_noop_filtering_callback,
        add_turn_only=True,
        prev_turn_meta_summary=r2.prev_turn_meta_summary,
    )

    return r1.user_turn_node_id, r2.user_turn_node_id, r3.user_turn_node_id


def test_next_turn_outgoing_uniqueness_enforced(conversation_engine, engine):
    """If a turn already has a next_turn outgoing, adding a second must fail.

    This test assumes Phase-1 validation is wired into BOTH add_edge and add_pure_edge.
    """
    t1, t2, t3 = _mk_three_turns(conversation_engine, engine, user_id="u", conv_id="conv_out_unique")

    # Orchestrator should already have created t1 -> t2 next_turn.
    # Adding another next_turn out of t1 must be rejected.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_edge(src=t1, tgt=t3, relation="next_turn", causal_type="chain", doc_id=f"conv:{'conv_out_unique'}")
        )


def test_next_turn_incoming_uniqueness_enforced(conversation_engine, engine):
    t1, t2, t3 = _mk_three_turns(conversation_engine, engine, user_id="u", conv_id="conv_in_unique")

    # Orchestrator should already have created t1 -> t2 next_turn.
    # Adding another next_turn into t2 must be rejected.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_edge(src=t3, tgt=t2, relation="next_turn", causal_type="chain", doc_id=f"conv:{'conv_in_unique'}")
        )


def test_next_turn_validated_in_add_pure_edge(conversation_engine, engine):
    t1, t2, t3 = _mk_three_turns(conversation_engine, engine, user_id="u", conv_id="conv_pure_edge")

    with pytest.raises(ValueError):
        conversation_engine.add_pure_edge(
            _mk_edge(src=t1, tgt=t3, relation="next_turn", causal_type="chain", doc_id=f"conv:{'conv_pure_edge'}")
        )


def test_dependency_freeze_rule_does_not_scan_all_edges(conversation_engine, engine, monkeypatch):
    """Freeze rule should be implementable via endpoint existence checks.

    We assert that get_edges() is not called during validation (regression guard).
    """
    t1, t2, t3 = _mk_three_turns(conversation_engine, engine, user_id="u", conv_id="conv_dep_freeze"
                                 #, causal_type='chain'
                                 )

    # If implementation regresses to scanning, fail fast.
    if hasattr(conversation_engine, "get_edges"):
        monkeypatch.setattr(
            conversation_engine,
            "get_edges",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("get_edges() must not be used for Phase-1 validation")),
            raising=True,
        )

    # t1 is 'used' (has outgoing next_turn). Adding dependency incoming into t1 must fail.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_edge(src=t3, tgt=t1, relation="depends_on", causal_type="dependency", doc_id=f"conv:{'conv_dep_freeze'}")
        )
