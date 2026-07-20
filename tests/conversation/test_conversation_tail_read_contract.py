from __future__ import annotations

import pytest

from kogwistar.conversation.models import ConversationNode
from kogwistar.conversation.policy import get_chat_tail
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Span
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.regression


def test_conversation_tail_does_not_request_irrelevant_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Tail lookup is metadata/document serving work, not vector retrieval."""
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "conversation"),
        kg_graph_type="conversation",
        backend_factory=build_fake_backend,
    )
    try:
        node = ConversationNode(
            id="turn-1",
            label="Turn 1",
            type="entity",
            summary="hello",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation("conv-1")])],
            doc_id="conv:conv-1",
            metadata={
                "entity_type": "conversation_turn",
                "in_conversation_chain": True,
                "level_from_root": 0,
            },
            role="user",
            turn_index=1,
            conversation_id="conv-1",
            domain_id=None,
            canonical_entity_id=None,
            properties={},
            embedding=None,
        )
        engine.write.add_node(node)

        original_node_get = engine.backend.node_get
        requested_includes: list[list[str] | None] = []

        def spy_node_get(**kwargs):
            requested_includes.append(kwargs.get("include"))
            return original_node_get(**kwargs)

        monkeypatch.setattr(engine.backend, "node_get", spy_node_get)

        tail = get_chat_tail(engine, conversation_id="conv-1")

        assert tail is not None
        assert tail.id == "turn-1"
        assert requested_includes == [["documents", "metadatas"]]
    finally:
        engine.close()
