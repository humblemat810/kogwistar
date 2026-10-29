from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kogwistar.conversation.models import ConversationNode
from kogwistar.conversation.service import ConversationService


ROOT = Path(__file__).resolve().parents[2]


def test_python_conversation_create_event_payload_matches_shared_golden() -> None:
    fixture = json.loads(
        (
            ROOT
            / "contracts"
            / "golden"
            / "conversation-create-start-node-v1.json"
        ).read_text(encoding="utf-8")
    )
    captured: dict[str, object] = {}

    def capture(node: ConversationNode, doc_id: str | None = None) -> None:
        _ = doc_id
        captured["payload"] = node.model_dump(field_mode="backend", exclude=["embedding"])

    service = object.__new__(ConversationService)
    service.conversation_engine = SimpleNamespace(
        kg_graph_type="conversation",
        write=SimpleNamespace(add_node=capture),
    )

    conversation_id, node_id = service._create_conversation_primitive(
        "alice", "conv-create-test", "conv-create-start"
    )

    assert (conversation_id, node_id) == ("conv-create-test", "conv-create-start")
    assert captured["payload"] == fixture["payload"]
