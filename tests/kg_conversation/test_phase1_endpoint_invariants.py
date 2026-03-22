# tests/kg_conversation/test_phase1_endpoint_invariants.py
from __future__ import annotations

import pytest

from kogwistar.conversation.models import (
    ConversationEdge,
    ConversationNode,
)
from kogwistar.engine_core.models import (
    Grounding,
    MentionVerification,
    Span,
)
from tests.conftest import _make_engine_pair
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.engine_core.engine import GraphKnowledgeEngine


pytestmark = pytest.mark.parametrize(
    "phase1_engine_pair",
    [
        pytest.param(
            {
                "backend_kind": "fake",
                "embedding_kind": "constant",
                "dim": 384,
            },
            id="fake_backend_constant",
            marks=pytest.mark.ci,
        ),
        pytest.param(
            {
                "backend_kind": "chroma",
                "embedding_kind": "provider",
                "dim": 384,
            },
            id="real_chroma_provider",
            marks=pytest.mark.ci_full,
        ),
    ],
    indirect=True,
)


def _mk_span(doc_id: str) -> Span:
    return Span(
        collection_page_url="N/A",
        document_page_url="N/A",
        doc_id=doc_id,
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=1,
        excerpt="x",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(
            method="human", is_verified=True, score=1.0, notes="test"
        ),
    )


def _mk_turn(
    *,
    conversation_id: str,
    user_id: str,
    turn_id: str,
    role: str,
    turn_index: int,
) -> ConversationNode:
    doc_id = f"conv:{conversation_id}"
    return ConversationNode(
        id=turn_id,
        label=turn_id,
        type="entity",
        summary=f"{role}:{turn_index}",
        mentions=[Grounding(spans=[_mk_span(doc_id)])],
        doc_id=doc_id,
        metadata={
            "entity_type": f"{role}_turn",
            "in_conversation_chain": True,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "level_from_root": 0,
        },
        role=role,  # type: ignore[arg-type]
        turn_index=turn_index,
        conversation_id=conversation_id,
        user_id=user_id,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        level_from_root=0,
    )


def _mk_edge(
    *, src: str, tgt: str, relation: str, causal_type: str, doc_id: str
) -> ConversationEdge:
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
    from kogwistar.conversation.models import FilteringResult

    return FilteringResult(node_ids=[], edge_ids=[]), "noop"


@pytest.fixture
def phase1_engine_pair(request, tmp_path):
    """Build one conversation/knowledge engine pair for this file.

    The module is parametrized into two variants:
    - `fake_backend_constant`: cheap CI coverage
    - `real_chroma_provider`: fuller backend/embedding coverage

    To add another variant later, extend the `pytestmark` parametrization above.
    """

    cfg = dict(request.param)
    return _disable_phase1_index_jobs(
        _make_engine_pair(
            backend_kind=cfg["backend_kind"],
            tmp_path=tmp_path,
            sa_engine=None,
            pg_schema=None,
            dim=int(cfg.get("dim", 384)),
            embedding_kind=str(cfg["embedding_kind"]),
        )
    )


def _disable_phase1_index_jobs(engine_pair):
    kg_engine, conversation_engine = engine_pair
    kg_engine._phase1_enable_index_jobs = False
    conversation_engine._phase1_enable_index_jobs = False
    return kg_engine, conversation_engine


def _mk_three_turns(
    conversation_engine: GraphKnowledgeEngine,
    kg_engine: GraphKnowledgeEngine,
    *,
    user_id: str,
    conv_id: str,
    # causal_type: str="chain"
):
    t1 = _mk_turn(
        conversation_id=conv_id,
        user_id=user_id,
        turn_id="turn_1_user",
        role="user",
        turn_index=1,
    )
    t2 = _mk_turn(
        conversation_id=conv_id,
        user_id=user_id,
        turn_id="turn_2_assistant",
        role="assistant",
        turn_index=2,
    )
    t3 = _mk_turn(
        conversation_id=conv_id,
        user_id=user_id,
        turn_id="turn_3_user",
        role="user",
        turn_index=3,
    )
    conversation_engine.add_node(t1)
    conversation_engine.add_node(t2)
    conversation_engine.add_node(t3)
    return t1.id, t2.id, t3.id


def _seed_chain_edge(
    conversation_engine: GraphKnowledgeEngine,
    *,
    conversation_id: str,
    src: str,
    tgt: str,
) -> None:
    conversation_engine.add_edge(
        _mk_edge(
            src=src,
            tgt=tgt,
            relation="next_turn",
            causal_type="chain",
            doc_id=f"conv:{conversation_id}",
        )
    )


def test_next_turn_outgoing_uniqueness_enforced(phase1_engine_pair):
    """If a turn already has a next_turn outgoing, adding a second must fail.

    This test assumes Phase-1 validation is wired into BOTH add_edge and add_pure_edge.
    """
    engine, conversation_engine = phase1_engine_pair
    t1, t2, t3 = _mk_three_turns(
        conversation_engine, engine, user_id="u", conv_id="conv_out_unique"
    )

    _seed_chain_edge(
        conversation_engine, conversation_id="conv_out_unique", src=t1, tgt=t2
    )

    # Adding another next_turn out of t1 must be rejected.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_edge(
                src=t1,
                tgt=t3,
                relation="next_turn",
                causal_type="chain",
                doc_id=f"conv:{'conv_out_unique'}",
            )
        )


def test_next_turn_incoming_uniqueness_enforced(phase1_engine_pair):
    engine, conversation_engine = phase1_engine_pair
    t1, t2, t3 = _mk_three_turns(
        conversation_engine, engine, user_id="u", conv_id="conv_in_unique"
    )

    _seed_chain_edge(
        conversation_engine, conversation_id="conv_in_unique", src=t1, tgt=t2
    )

    # Adding another next_turn into t2 must be rejected.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_edge(
                src=t3,
                tgt=t2,
                relation="next_turn",
                causal_type="chain",
                doc_id=f"conv:{'conv_in_unique'}",
            )
        )


def test_next_turn_validated_in_add_pure_edge(phase1_engine_pair):
    engine, conversation_engine = phase1_engine_pair
    t1, t2, t3 = _mk_three_turns(
        conversation_engine, engine, user_id="u", conv_id="conv_pure_edge"
    )

    _seed_chain_edge(
        conversation_engine, conversation_id="conv_pure_edge", src=t1, tgt=t2
    )

    with pytest.raises(ValueError):
        conversation_engine.add_pure_edge(
            _mk_edge(
                src=t1,
                tgt=t3,
                relation="next_turn",
                causal_type="chain",
                doc_id=f"conv:{'conv_pure_edge'}",
            )
        )


def test_dependency_freeze_rule_does_not_scan_all_edges(
    phase1_engine_pair, monkeypatch
):
    """Freeze rule should be implementable via endpoint existence checks.

    We assert that get_edges() is not called during validation (regression guard).
    """
    engine, conversation_engine = phase1_engine_pair
    t1, t2, t3 = _mk_three_turns(
        conversation_engine,
        engine,
        user_id="u",
        conv_id="conv_dep_freeze",
        # , causal_type='chain'
    )
    _seed_chain_edge(
        conversation_engine, conversation_id="conv_dep_freeze", src=t1, tgt=t2
    )

    # If implementation regresses to scanning, fail fast.
    if hasattr(conversation_engine, "get_edges"):
        monkeypatch.setattr(
            conversation_engine,
            "get_edges",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("get_edges() must not be used for Phase-1 validation")
            ),
            raising=True,
        )

    # t1 is 'used' (has outgoing next_turn). Adding dependency incoming into t1 must fail.
    with pytest.raises(ValueError):
        conversation_engine.add_edge(
            _mk_edge(
                src=t3,
                tgt=t1,
                relation="depends_on",
                causal_type="dependency",
                doc_id=f"conv:{'conv_dep_freeze'}",
            )
        )
