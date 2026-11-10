from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from kogwistar.conversation.models import ConversationNode
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.server.auth_middleware import claims_ctx
from kogwistar.server.chat_service_conversation_queries import _ConversationQueryService
from tests._helpers.fake_backend import build_fake_backend


class _ConversationOwner:
    def _conversation_engine(self) -> SimpleNamespace:
        nodes = [
            SimpleNamespace(
                id="turn-1",
                metadata={"entity_type": "conversation_turn"},
                turn_index=1,
                role="user",
                summary="first",
            ),
            SimpleNamespace(
                id="turn-2",
                metadata={"entity_type": "conversation_turn"},
                turn_index=2,
                role="assistant",
                summary="second",
            )
        ]
        return SimpleNamespace(
            read=SimpleNamespace(get_nodes=lambda **_kwargs: nodes)
        )


def test_python_receive_message_result_matches_shared_wire_fixture() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/receive-message-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    queries = _ConversationQueryService(cast(Any, _ConversationOwner()))

    result = {
        "conversation_id": "c1",
        "messages": queries.list_transcript("c1"),
    }

    assert result == fixture


def test_python_receive_message_applies_engine_acl_before_projection(
    tmp_path: Path,
) -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/receive-message-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        kg_graph_type="conversation",
        namespace="conversation",
        acl_enabled=True,
    )
    queries = _ConversationQueryService(
        cast(Any, SimpleNamespace(_conversation_engine=lambda: engine))
    )
    try:
        owner_claims = claims_ctx.set(
            {"ns": "conversation", "agent_id": "owner", "security_scope": "tenant-a"}
        )
        try:
            for node_id, turn_index, role, summary, visibility in (
                ("turn-1", 1, "user", "first", "public"),
                ("turn-2", 2, "assistant", "second", "public"),
                ("turn-private", 3, "assistant", "private", "private"),
            ):
                engine.write.add_node(
                    ConversationNode(
                        id=node_id,
                        label=f"turn-{turn_index}",
                        type="entity",
                        summary=summary,
                        mentions=[
                            Grounding(spans=[Span.from_dummy_for_conversation("c1")])
                        ],
                        doc_id="conv:c1",
                        metadata={
                            "entity_type": "conversation_turn",
                            "conversation_id": "c1",
                            "in_conversation_chain": True,
                            "visibility": visibility,
                            "owner_agent_id": "owner",
                            "security_scope": "tenant-a",
                            "level_from_root": 0,
                        },
                        role=role,
                        turn_index=turn_index,
                        conversation_id="c1",
                        user_id="owner",
                    )
                )
        finally:
            claims_ctx.reset(owner_claims)

        reader_claims = claims_ctx.set(
            {"ns": "conversation", "agent_id": "reader", "security_scope": "tenant-a"}
        )
        try:
            result = {
                "conversation_id": "c1",
                "messages": queries.list_transcript("c1"),
            }
        finally:
            claims_ctx.reset(reader_claims)
    finally:
        engine.close()

    assert result == fixture
