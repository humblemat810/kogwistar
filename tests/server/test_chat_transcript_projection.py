from __future__ import annotations

from types import SimpleNamespace

from kogwistar.server.chat_service_conversation_queries import _ConversationQueryService


def test_transcript_projection_excludes_durable_agent_event_nodes(monkeypatch) -> None:
    nodes = [
        SimpleNamespace(
            id="user-turn",
            turn_index=1,
            role="user",
            summary="Question",
            metadata={"entity_type": "conversation_turn"},
        ),
        SimpleNamespace(
            id="provider-event",
            turn_index=1,
            role="system",
            summary='{"kind":"reasoning_summary","text":"internal event"}',
            metadata={"entity_type": "agent_provider_event"},
        ),
        SimpleNamespace(
            id="assistant-turn",
            turn_index=2,
            role="assistant",
            summary="Answer",
            metadata={"entity_type": "assistant_turn"},
        ),
    ]
    monkeypatch.setattr(
        _ConversationQueryService,
        "_conversation_nodes",
        lambda _self, _conversation_id: nodes,
    )

    service = object.__new__(_ConversationQueryService)
    turns = service.list_transcript("conversation-1")

    assert [turn["node_id"] for turn in turns] == ["user-turn", "assistant-turn"]
    assert all(turn["node_id"] != "provider-event" for turn in turns)
