from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from kogwistar.conversation.conversation_context import ContextSources
from kogwistar.conversation.conversation_state_contracts import PrevTurnMetaSummaryModel
from kogwistar.conversation.memory_retriever import MemoryRetriever
from kogwistar.conversation.models import ConversationEdge, ConversationNode, RetrievalResult
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine, scoped_namespace
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.server.auth_middleware import claims_ctx
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.core


def _make_engine() -> tuple[GraphKnowledgeEngine, Path]:
    test_db_dir = Path.cwd() / ".tmp_memory_visibility" / str(uuid.uuid4())
    test_db_dir.mkdir(parents=True, exist_ok=True)
    engine = GraphKnowledgeEngine(
        persist_directory=str(test_db_dir),
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
    )
    return engine, test_db_dir


def _mk_turn_node(*, conversation_id: str, user_id: str, turn_id: str, turn_index: int):
    return ConversationNode(
        id=turn_id,
        label=f"turn-{turn_index}",
        type="entity",
        summary=f"turn-{turn_index}",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation(conversation_id)])],
        doc_id=f"conv:{conversation_id}",
        metadata={
            "entity_type": "conversation_turn",
            "in_conversation_chain": True,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "level_from_root": 0,
        },
        role="user",
        turn_index=turn_index,
        conversation_id=conversation_id,
        user_id=user_id,
        domain_id=None,
        canonical_entity_id=None,
        properties=None,
        embedding=None,
        level_from_root=0,
    )


def _mk_reference_node(
    *,
    conversation_id: str,
    user_id: str,
    ref_id: str,
    refers_to_id: str,
    visibility: str,
    security_scope: str,
    agent_id: str,
    shared_with_agents: list[str] | None = None,
):
    return ConversationNode(
        id=ref_id,
        label="ref-node",
        type="reference_pointer",
        summary="knowledge ref",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation(conversation_id)])],
        doc_id=f"conv:{conversation_id}",
        metadata={
            "entity_type": "reference_pointer",
            "in_conversation_chain": True,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "level_from_root": 0,
            "visibility": visibility,
            "security_scope": security_scope,
            "owner_security_scope": security_scope,
            "owner_agent_id": agent_id,
            "agent_id": agent_id,
            "shared_with_agents": shared_with_agents or [],
            "shared_with": [],
            "refers_to_id": refers_to_id,
        },
        role="system",
        turn_index=0,
        conversation_id=conversation_id,
        user_id=user_id,
        domain_id=None,
        canonical_entity_id=None,
        properties={"refers_to_id": refers_to_id},
        embedding=None,
        level_from_root=0,
    )


def _pin_memory(
    *,
    service: ConversationService,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    security_scope: str,
    visibility: str,
    shared_with_agents: list[str] | None = None,
):
    retriever = MemoryRetriever(
        conversation_engine=service.conversation_engine,
        llm_tasks=service.llm_tasks,
        filtering_callback=lambda *args, **kwargs: ({"node_ids": [], "edge_ids": []}, ""),
    )
    turn_id = f"turn:{uuid.uuid4()}"
    source_id = f"src:{uuid.uuid4()}"
    span = Span.from_dummy_for_conversation(conversation_id)
    turn = _mk_turn_node(
        conversation_id=conversation_id,
        user_id=user_id,
        turn_id=turn_id,
        turn_index=0,
    )
    source = _mk_turn_node(
        conversation_id=conversation_id,
        user_id=user_id,
        turn_id=source_id,
        turn_index=-1,
    )
    with scoped_namespace(service.conversation_engine, "conversation"):
        service.conversation_engine.add_node(turn)
        service.conversation_engine.add_node(source)
        out = retriever.pin_selected(
            user_id=user_id,
            current_conversation_id=conversation_id,
            mem_id=f"mem:{uuid.uuid4()}",
            turn_node_id=turn_id,
            turn_index=0,
            self_span=span,
            selected_memory=RetrievalResult(nodes=[source], edges=[]),
            memory_context_text="secret memory",
            prev_turn_meta_summary=PrevTurnMetaSummaryModel(
                prev_node_char_distance_from_last_summary=0,
                prev_node_distance_from_last_summary=0,
                tail_turn_index=0,
            ),
            visibility=visibility,
            shared_with_agents=shared_with_agents or [],
            security_scope=security_scope,
            agent_id=agent_id,
        )
    return out, turn_id


def _pin_reference(
    *,
    service: ConversationService,
    conversation_id: str,
    user_id: str,
    agent_id: str,
    security_scope: str,
    visibility: str,
    shared_with_agents: list[str] | None = None,
):
    turn_id = f"turn:{uuid.uuid4()}"
    ref_id = f"ref:{uuid.uuid4()}"
    turn = _mk_turn_node(
        conversation_id=conversation_id,
        user_id=user_id,
        turn_id=turn_id,
        turn_index=0,
    )
    ref = _mk_reference_node(
        conversation_id=conversation_id,
        user_id=user_id,
        ref_id=ref_id,
        refers_to_id=f"kg:{uuid.uuid4()}",
        visibility=visibility,
        security_scope=security_scope,
        agent_id=agent_id,
        shared_with_agents=shared_with_agents or [],
    )
    edge = ConversationEdge(
        id=f"{turn_id}::ref::{ref_id}",
        source_ids=[turn_id],
        target_ids=[ref_id],
        relation="references",
        label="references",
        type="relationship",
        summary="turn references pointer",
        doc_id=f"conv:{conversation_id}",
        mentions=[Grounding(spans=[Span.from_dummy_for_conversation(conversation_id)])],
        domain_id=None,
        canonical_entity_id=None,
        properties={"entity_type": "conversation_edge"},
        embedding=None,
        metadata={
            "entity_type": "conversation_edge",
            "conversation_id": conversation_id,
            "visibility": visibility,
            "security_scope": security_scope,
            "owner_agent_id": agent_id,
            "agent_id": agent_id,
        },
        source_edge_ids=[],
        target_edge_ids=[],
    )
    with scoped_namespace(service.conversation_engine, "conversation"):
        service.conversation_engine.add_node(turn)
        service.conversation_engine.add_node(ref)
        service.conversation_engine.add_edge(edge)
    return ref


def test_private_memory_hidden_from_other_agent():
    engine, test_db_dir = _make_engine()
    try:
        service = ConversationService.from_engine(engine, knowledge_engine=engine)
        conversation_id = "conv-private-memory"
        user_id = "user-1"

        token_a = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-a",
            }
        )
        try:
            out, turn_id = _pin_memory(
                service=service,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id="agent-a",
                security_scope="tenant-a",
                visibility="private",
            )
        finally:
            claims_ctx.reset(token_a)

        token_b = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-b",
            }
        )
        try:
            view = service.get_conversation_view(conversation_id=conversation_id, purpose="answer")
        finally:
            claims_ctx.reset(token_b)

        assert out is not None
        assert out.visibility == "private"
        assert out.security_scope == "tenant-a"
        assert out.memory_context_node.metadata["owner_agent_id"] == "agent-a"
        assert out.memory_context_node.id not in view.active_memory_context_ids
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_shared_memory_visible_to_explicit_agent():
    engine, test_db_dir = _make_engine()
    try:
        service = ConversationService.from_engine(engine, knowledge_engine=engine)
        conversation_id = "conv-shared-memory"
        user_id = "user-1"

        token_a = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-a",
            }
        )
        try:
            out, turn_id = _pin_memory(
                service=service,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id="agent-a",
                security_scope="tenant-a",
                visibility="shared",
                shared_with_agents=["agent-b"],
            )
        finally:
            claims_ctx.reset(token_a)

        token_b = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-b",
            }
        )
        try:
            view = service.get_conversation_view(conversation_id=conversation_id, purpose="answer")
        finally:
            claims_ctx.reset(token_b)

        assert out is not None
        assert out.visibility == "shared"
        assert out.memory_context_node.metadata["shared_with_agents"] == ["agent-b"]
        assert out.memory_context_node.id in view.active_memory_context_ids
        assert any(
            item.node_id == out.memory_context_node.id and item.kind == "memory_context"
            for item in view.items
        )
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_private_pinned_ref_hidden_from_other_agent():
    engine, test_db_dir = _make_engine()
    try:
        service = ConversationService.from_engine(engine, knowledge_engine=engine)
        conversation_id = "conv-private-ref"
        user_id = "user-1"

        token_a = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-a",
            }
        )
        try:
            ref = _pin_reference(
                service=service,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id="agent-a",
                security_scope="tenant-a",
                visibility="private",
            )
        finally:
            claims_ctx.reset(token_a)

        token_b = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-b",
            }
        )
        try:
            view = service.get_conversation_view(conversation_id=conversation_id, purpose="answer")
        finally:
            claims_ctx.reset(token_b)

        assert ref.id not in view.pinned_kg_ref_ids
        assert all(
            item.node_id != ref.id or item.kind != "pinned_kg_ref"
            for item in view.items
        )
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)


def test_shared_pinned_ref_visible_to_explicit_agent():
    engine, test_db_dir = _make_engine()
    try:
        service = ConversationService.from_engine(engine, knowledge_engine=engine)
        conversation_id = "conv-shared-ref"
        user_id = "user-1"

        token_a = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-a",
            }
        )
        try:
            ref = _pin_reference(
                service=service,
                conversation_id=conversation_id,
                user_id=user_id,
                agent_id="agent-a",
                security_scope="tenant-a",
                visibility="shared",
                shared_with_agents=["agent-b"],
            )
        finally:
            claims_ctx.reset(token_a)

        token_b = claims_ctx.set(
            {
                "ns": "conversation",
                "security_scope": "tenant-a",
                "agent_id": "agent-b",
            }
        )
        try:
            view = service.get_conversation_view(conversation_id=conversation_id, purpose="answer")
        finally:
            claims_ctx.reset(token_b)

        assert ref.id in view.pinned_kg_ref_ids
        assert any(
            item.node_id == ref.id and item.kind == "pinned_kg_ref"
            for item in view.items
        )
    finally:
        shutil.rmtree(test_db_dir, ignore_errors=True)
