from __future__ import annotations

from typing import Any, Optional, Sequence

from kogwistar.conversation.models import ConversationEdge, ConversationNode
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Edge, Grounding, MentionVerification, Node, Span


def mk_verification(
    *,
    method: str = "human",
    is_verified: bool = True,
    score: float = 1.0,
    notes: str = "seed",
) -> MentionVerification:
    return MentionVerification(
        method=method,
        is_verified=is_verified,
        score=score,
        notes=notes,
    )


def mk_span(
    *,
    doc_id: str,
    full_text: str,
    start_char: int = 0,
    end_char: Optional[int] = None,
    page_number: int = 1,
    insertion_method: str = "seed",
    collection_page_url: str = "url",
    document_page_url: str = "url",
    context_before: str = "",
    context_after: str = "",
    chunk_id: Optional[str] = None,
    source_cluster_id: Optional[str] = None,
    verification: Optional[MentionVerification] = None,
) -> Span:
    if end_char is None:
        end_char = len(full_text)
    excerpt = full_text[start_char:end_char]
    return Span(
        collection_page_url=collection_page_url,
        document_page_url=document_page_url,
        doc_id=doc_id,
        insertion_method=insertion_method,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
        excerpt=excerpt,
        context_before=context_before,
        context_after=context_after,
        chunk_id=chunk_id,
        source_cluster_id=source_cluster_id,
        verification=verification or mk_verification(notes=f"seed:{insertion_method}"),
    )


def mk_grounding(*spans: Span) -> Grounding:
    return Grounding(spans=list(spans))


def add_node_raw(
    engine: GraphKnowledgeEngine,
    node: Node | ConversationNode,
    *,
    embedding_dim: int = 384,
    embedding: Optional[Sequence[float]] = None,
) -> None:
    doc, meta = engine._node_doc_and_meta(node)
    if embedding is None and getattr(node, "embedding", None) is None:
        embedding = [0.1] * embedding_dim
    if getattr(node, "embedding", None) is None:
        node.embedding = embedding  # type: ignore[assignment]

    engine.node_collection.add(
        ids=[node.id],
        documents=[doc],
        embeddings=[list(node.embedding)],  # type: ignore[arg-type]
        metadatas=[meta],
    )


def add_edge_raw(
    engine: Any,
    edge: Edge | ConversationEdge,
    *,
    embedding_dim: int = 384,
    embedding: Optional[Sequence[float]] = None,
) -> None:
    doc, meta = engine._edge_doc_and_meta(edge)
    if embedding is None and getattr(edge, "embedding", None) is None:
        embedding = [0.1] * embedding_dim
    if getattr(edge, "embedding", None) is None:
        edge.embedding = embedding  # type: ignore[assignment]

    engine.edge_collection.add(
        ids=[edge.id],
        documents=[doc],
        embeddings=[list(edge.embedding)],  # type: ignore[arg-type]
        metadatas=[meta],
    )


def seed_kg_graph(
    *, kg_engine: GraphKnowledgeEngine, kg_doc_id: str = "D_KG_001"
) -> dict[str, Any]:
    text1 = "Project KGE stores entities and relations with provenance spans."
    text2 = "Conversation graph nodes can reference KG nodes/edges for grounding."

    n1 = Node(
        id="KG_N1",
        label="KGE provenance",
        type="entity",
        summary="KGE stores entities/relations with spans for provenance.",
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text=text1,
                    insertion_method="seed_kg_node",
                    document_page_url=f"doc/{kg_doc_id}#KG_N1",
                    collection_page_url=f"collection/{kg_doc_id}",
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=kg_doc_id,
        level_from_root=0,
    )

    n2 = Node(
        id="KG_N2",
        label="Conversation grounding",
        type="entity",
        summary="Conversation graph nodes can reference KG items for grounding.",
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text=text2,
                    insertion_method="seed_kg_node",
                    document_page_url=f"doc/{kg_doc_id}#KG_N2",
                    collection_page_url=f"collection/{kg_doc_id}",
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_entity"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_node"},
        embedding=None,
        doc_id=kg_doc_id,
        level_from_root=0,
    )

    e1 = Edge(
        id="KG_E1",
        label="supports",
        type="relationship",
        summary="Provenance spans support conversation grounding.",
        source_ids=[n1.id],
        target_ids=[n2.id],
        relation="supports",
        source_edge_ids=[],
        target_edge_ids=[],
        mentions=[
            mk_grounding(
                mk_span(
                    doc_id=kg_doc_id,
                    full_text="supports",
                    insertion_method="seed_kg_edge",
                    document_page_url=f"doc/{kg_doc_id}#KG_E1",
                    collection_page_url=f"collection/{kg_doc_id}",
                    start_char=0,
                    end_char=8,
                )
            )
        ],
        metadata={"level_from_root": 0, "entity_type": "kg_edge"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"kind": "kg_edge"},
        embedding=None,
        doc_id=kg_doc_id,
    )

    add_node_raw(kg_engine, n1)
    add_node_raw(kg_engine, n2)
    add_edge_raw(kg_engine, e1)

    return {
        "doc_id": kg_doc_id,
        "node_ids": (n1.id, n2.id),
        "edge_ids": (e1.id,),
        "n1": n1,
        "n2": n2,
        "e1": e1,
    }


def seed_conversation_graph(
    *,
    conversation_engine: GraphKnowledgeEngine,
    user_id: str = "U_TEST",
    conversation_id: str = "CONV_TEST_001",
    start_node_id: str = "CONV_START_001",
    kg_seed: dict[str, Any],
) -> dict[str, Any]:
    conv_svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=conversation_engine,
    )
    conv_id, start_id = conv_svc.create_conversation(
        user_id,
        conversation_id,
        start_node_id,
    )
    assert conv_id == conversation_id
    assert start_id == start_node_id

    t0_text = "Show me what happened in the graph engine."
    t0_id = "TURN_000"
    t0_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=t0_text,
        insertion_method="conversation_turn",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{t0_id}",
        page_number=1,
    )
    turn0 = ConversationNode(
        user_id=user_id,
        id=t0_id,
        label="Turn 0 (user)",
        type="entity",
        doc_id=t0_id,
        summary=t0_text,
        role="user",  # type: ignore[arg-type]
        turn_index=0,
        conversation_id=conv_id,
        mentions=[mk_grounding(t0_span)],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(turn0, None)

    t1_text = "Here are the relevant KG nodes and the conversation timeline."
    t1_id = "TURN_001"
    t1_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=t1_text,
        insertion_method="conversation_turn",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{t1_id}",
        page_number=1,
    )
    turn1 = ConversationNode(
        user_id=user_id,
        id=t1_id,
        label="Turn 1 (assistant)",
        type="entity",
        doc_id=t1_id,
        summary=t1_text,
        role="assistant",  # type: ignore[arg-type]
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(t1_span)],
        properties={},
        metadata={
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(turn1, None)

    next_edge = ConversationEdge(
        id="EDGE_NEXT_000_001",
        source_ids=[turn0.safe_get_id()],
        target_ids=[turn1.safe_get_id()],
        relation="next_turn",
        label="next_turn",
        type="relationship",
        summary="Sequential flow",
        doc_id=f"conv:{conv_id}",
        mentions=[mk_grounding(t1_span)],
        metadata={"causal_type": "chain"},
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "conversation_edge"},
        embedding=None,
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.write.add_edge(next_edge)

    memctx_id = "MEMCTX_001"
    memctx_text = "Active memory context: user wants graph debugging view."
    memctx_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=memctx_text,
        insertion_method="memory_context",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{memctx_id}",
    )
    memctx = ConversationNode(
        user_id=user_id,
        id=memctx_id,
        label="Memory context (turn 1)",
        type="entity",
        doc_id=memctx_id,
        summary=memctx_text,
        role="system",  # type: ignore[arg-type]
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(memctx_span)],
        properties={
            "user_id": user_id,
            "source_memory_nodes_ids": [],
            "source_memory_edges_ids": [],
        },
        metadata={
            "entity_type": "memory_context",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(memctx, None)

    summ_id = "SUMM_001"
    summ_text = (
        "Summary: user asks to inspect graph flow; assistant will show KG + conversation links."
    )
    summ_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=summ_text,
        insertion_method="conversation_summary",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{summ_id}",
    )
    summ = ConversationNode(
        user_id=user_id,
        id=summ_id,
        label="Summary 0-1",
        type="entity",
        doc_id=summ_id,
        summary=summ_text,  # type: ignore[arg-type]
        role="system",  # type: ignore[arg-type]
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(summ_span)],
        properties={"content": summ_text},
        metadata={
            "entity_type": "conversation_summary",
            "level_from_root": 1,
            "in_conversation_chain": True,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=1,
    )
    conversation_engine.write.add_node(summ, None)

    summ_edge = ConversationEdge(
        id="EDGE_SUMM_001",
        source_ids=[summ.safe_get_id()],
        target_ids=[turn0.safe_get_id(), turn1.safe_get_id()],
        relation="summarizes",
        label="summarizes",
        type="relationship",
        summary="Memory summarization",
        doc_id=f"conv:{conv_id}",
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        mentions=[mk_grounding(summ_span)],
        metadata={"causal_type": "summary"},
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.write.add_edge(summ_edge)

    kg_ref_id = "KGREF_001"
    kg_ref_text = f"KG ref: node={kg_seed['node_ids'][0]} edge={kg_seed['edge_ids'][0]}"
    kg_ref_span = mk_span(
        doc_id=f"conv:{conv_id}",
        full_text=kg_ref_text,
        insertion_method="kg_ref",
        collection_page_url=f"conversation/{conv_id}",
        document_page_url=f"conversation/{conv_id}#{kg_ref_id}",
    )
    kg_ref_node = ConversationNode(
        user_id=user_id,
        id=kg_ref_id,
        label="KG reference",
        type="entity",
        doc_id=kg_ref_id,
        summary=kg_ref_text,
        role="system",  # type: ignore[arg-type]
        turn_index=1,
        conversation_id=conv_id,
        mentions=[mk_grounding(kg_ref_span)],
        properties={
            "ref_kind": "kg",
            "ref_doc_id": kg_seed["doc_id"],
            "ref_node_ids": list(kg_seed["node_ids"]),
            "ref_edge_ids": list(kg_seed["edge_ids"]),
        },
        metadata={
            "entity_type": "kg_ref",
            "level_from_root": 0,
            "in_conversation_chain": False,
        },
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
        level_from_root=0,
    )
    conversation_engine.write.add_node(kg_ref_node, None)

    ref_edge = ConversationEdge(
        id="EDGE_TURN1_KGREF",
        source_ids=[turn1.safe_get_id()],
        target_ids=[kg_ref_node.safe_get_id()],
        relation="mentions_kg",
        label="mentions_kg",
        type="relationship",
        summary="Assistant mentions KG refs",
        doc_id=f"conv:{conv_id}",
        domain_id=None,
        canonical_entity_id=None,
        properties={"ref_kind": "kg"},
        embedding=None,
        mentions=[mk_grounding(kg_ref_span)],
        metadata={"causal_type": "reference"},
        source_edge_ids=[],
        target_edge_ids=[],
    )
    conversation_engine.write.add_edge(ref_edge)

    conv_id = "conv_test_1"
    user_id = "user_test_1"
    kg_target_id = "KG_N1"

    kg_ref_node = ConversationNode(
        id="CONV_REF_KG_N1",
        label="KG ref -> KG_N1",
        type="reference_pointer",
        summary="Conversation-side pointer to a KG node (for testing focus/openRef).",
        doc_id="CONV_REF_KG_N1",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
        properties={
            "ref_target_kind": "kg_node",
            "ref_target_id": kg_target_id,
        },
        metadata={
            "level_from_root": 0,
            "entity_type": "kg_ref",
            "in_conversation_chain": False,
            "role": "system",
            "turn_index": 1,
            "conversation_id": conv_id,
            "user_id": user_id,
        },
        role="system",
        turn_index=1,
        conversation_id=conv_id,
        user_id=user_id,
        embedding=None,
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
    )

    conversation_engine.write.add_node(kg_ref_node)
    return {
        "conversation_id": conv_id,
        "start_node_id": start_id,
        "turn_ids": (turn0.id, turn1.id),
        "edge_ids": (next_edge.id, summ_edge.id, ref_edge.id),
        "memctx_id": memctx.id,
        "summary_id": summ.id,
        "kg_ref_id": kg_ref_node.id,
    }
