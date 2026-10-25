from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from kogwistar.server.chat_service_conversation_queries import _ConversationQueryService


class _ConversationOwner:
    def _conversation_engine(self) -> SimpleNamespace:
        nodes = [
            SimpleNamespace(
                id="turn-1",
                metadata={"entity_type": "conversation_turn"},
                turn_index=0,
                role="user",
                summary="hello",
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
        "conversation_id": "conversation-1",
        "messages": queries.list_transcript("conversation-1"),
    }

    assert result == fixture
