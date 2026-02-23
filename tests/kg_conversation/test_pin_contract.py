from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import ConversationNode, Grounding, Span, MentionVerification


def test_conversation_pin_knowledge_reference_is_noop(tmp_path):
    """Contract: once a knowledge_reference pointer node is first pinned, it is never overwritten.

    A second attempt to pin the same pointer must be a no-op (not error, not update).
    """
    conv_dir = tmp_path / "conv"
    eng = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")

    conversation_id = "conv:test_conversation_pin_knowledge_reference_is_noop"
    user_id = "u1"

    eng.create_conversation(user_id=user_id, conversation_id=conversation_id, start_turn_id="start")

    span = Span(
        collection_page_url=f"conversation/{conversation_id}",
        document_page_url=f"conversation/{conversation_id}#pin",
        doc_id=f"conv:{conversation_id}",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=3,
        excerpt="pin",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test"),
    )

    nid = "kr:pointer:1"
    n1 = ConversationNode(
        user_id=user_id,
        id=nid,
        label="Pinned knowledge",
        type="entity",
        doc_id=nid,
        summary="ORIGINAL",
        role="system",
        turn_index=0,
        conversation_id=conversation_id,
        mentions=[Grounding(spans=[span])],
        properties={},
        metadata={
            "entity_type": "knowledge_reference",
            "level_from_root": 0,
            "in_conversation_chain": False,
            "conversation_id": conversation_id,
        },
        domain_id=None,
        canonical_entity_id=None,
    )

    eng.add_node(n1)

    n2 = n1.model_copy(deep=True)
    n2.summary = "MUTATED"
    eng.add_node(n2)

    got = eng.get_node(nid)
    assert got is not None
    assert got.summary == "ORIGINAL"
