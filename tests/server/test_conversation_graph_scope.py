from __future__ import annotations

import pytest
from types import SimpleNamespace
from typing import cast

from kogwistar.server.auth_middleware import (
    get_current_graph_space,
    get_current_workspace_id,
    reset_claims_ctx,
    set_claims_ctx,
)
from kogwistar.server.chat_service_conversation_queries import (
    _ConversationQueryService,
    _conversation_node_matches_scope,
)
from kogwistar.server.chat_api import create_chat_router
from kogwistar.json_types import JsonValue
from kogwistar.conversation.models import ConversationNode
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.in_memory_backend import build_in_memory_backend
from kogwistar.engine_core.models import Grounding, Span
from kogwistar.server.chat_service import ChatRunService
from tests._helpers.embeddings import ConstantEmbeddingFunction


@pytest.mark.parametrize(
    ("metadata", "workspace_id", "graph_space", "expected"),
    [
        ({}, None, None, True),
        (
            {"workspace_id": "w1", "graph_space": "conversation"},
            "w1",
            "conversation",
            True,
        ),
        (
            {"workspace_id": "w2", "graph_space": "conversation"},
            "w1",
            "conversation",
            False,
        ),
        (
            {"workspace_id": "w1", "graph_space": "knowledge"},
            "w1",
            "conversation",
            False,
        ),
        ({"graph_space": "conversation"}, "w1", "conversation", False),
        ({"workspace_id": "w1"}, "w1", "conversation", False),
        ({}, "w1", None, False),
    ],
)
def test_conversation_node_scope_matches_rust_filter_semantics(
    metadata: dict[str, JsonValue],
    workspace_id: str | None,
    graph_space: str | None,
    expected: bool,
) -> None:
    assert (
        _conversation_node_matches_scope(
            metadata, workspace_id=workspace_id, graph_space=graph_space
        )
        is expected
    )


def test_request_scope_accessors_accept_rust_workspace_claim_alias() -> None:
    token = set_claims_ctx(
        {
            "workspace_id": "workspace-a",
            "graph_space": "conversation",
        }
    )
    try:
        assert get_current_workspace_id() == "workspace-a"
        assert get_current_graph_space() == "conversation"
    finally:
        reset_claims_ctx(token)


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        ({"workspace": "legacy-workspace"}, "legacy-workspace"),
        (
            {"workspace_id": "canonical-workspace", "workspace": "legacy-workspace"},
            "canonical-workspace",
        ),
        ({"workspace_id": None, "workspace": "legacy-workspace"}, None),
        ({"workspace_id": ""}, ""),
    ],
)
def test_workspace_claim_precedence_matches_rust(
    claims: dict[str, object], expected: str | None
) -> None:
    token = set_claims_ctx(claims)
    try:
        assert get_current_workspace_id() == expected
    finally:
        reset_claims_ctx(token)


class _ScopedConversationRead:
    def __init__(self, nodes: list[SimpleNamespace]) -> None:
        self.nodes = nodes

    def get_nodes(self, **kwargs: object) -> list[SimpleNamespace]:
        where = kwargs.get("where")

        def matches(node: SimpleNamespace, clause: object) -> bool:
            if not isinstance(clause, dict):
                return True
            if "$and" in clause:
                return all(matches(node, item) for item in clause["$and"])
            metadata = getattr(node, "metadata", {}) or {}
            return all(
                getattr(node, key, metadata.get(key)) == value
                for key, value in clause.items()
            )

        return [node for node in self.nodes if matches(node, where)]


class _ScopedConversationOwner:
    def __init__(self, nodes: list[SimpleNamespace]) -> None:
        self.engine = SimpleNamespace(read=_ScopedConversationRead(nodes))

    def _conversation_engine(self) -> SimpleNamespace:
        return self.engine

    def _conversation_service(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_conversation_tail=lambda **_kwargs: SimpleNamespace(id="turn-foreign")
        )


def _scoped_query_service() -> _ConversationQueryService:
    nodes = [
        SimpleNamespace(
            id="start-a",
            conversation_id="c1",
            user_id="alice",
            turn_index=0,
            role="system",
            summary="start",
            properties={"status": "active"},
            entity_type="conversation_start",
            metadata={
                "entity_type": "conversation_start",
                "workspace_id": "w1",
                "graph_space": "conversation",
            },
        ),
        SimpleNamespace(
            id="turn-a",
            conversation_id="c1",
            user_id="alice",
            turn_index=1,
            role="user",
            summary="visible",
            properties={},
            entity_type="conversation_turn",
            metadata={
                "entity_type": "conversation_turn",
                "workspace_id": "w1",
                "graph_space": "conversation",
            },
        ),
        SimpleNamespace(
            id="turn-foreign",
            conversation_id="c1",
            user_id="alice",
            turn_index=2,
            role="assistant",
            summary="must not leak",
            properties={},
            entity_type="assistant_turn",
            metadata={
                "entity_type": "assistant_turn",
                "workspace_id": "w2",
                "graph_space": "conversation",
            },
        ),
    ]
    return _ConversationQueryService(_ScopedConversationOwner(nodes))  # type: ignore[arg-type]


def test_transcript_and_detail_filter_foreign_workspace_and_hide_foreign_tail() -> None:
    service = _scoped_query_service()

    transcript = service.list_transcript(
        "c1", workspace_id="w1", graph_space="conversation"
    )
    detail = service.get_conversation(
        "c1", workspace_id="w1", graph_space="conversation"
    )

    assert [turn["node_id"] for turn in transcript] == ["turn-a"]
    assert detail["turn_count"] == 1
    assert detail["tail_node_id"] == ""


def test_conversation_detail_is_not_found_when_all_nodes_are_out_of_scope() -> None:
    service = _scoped_query_service()

    with pytest.raises(KeyError, match="Unknown conversation_id"):
        service.get_conversation("c1", workspace_id="other")


def test_conversation_list_filters_start_node_and_scoped_transcript() -> None:
    service = _scoped_query_service()

    listed = service.list_conversations_for_user(
        "alice", workspace_id="w1", graph_space="conversation"
    )

    assert len(listed) == 1
    assert listed[0]["id"] == "c1"
    assert listed[0]["turn_count"] == 1


def test_conversation_http_reads_forward_authenticated_graph_scope() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    observed: list[tuple[str, str | None, str | None]] = []

    class Queries:
        def list_conversations_for_user(
            self, *, user_id: str, workspace_id: str | None, graph_space: str | None
        ) -> list[object]:
            observed.append(("list", workspace_id, graph_space))
            return []

    class Service:
        _conversation_queries = Queries()

        def get_conversation(
            self,
            conversation_id: str,
            *,
            workspace_id: str | None,
            graph_space: str | None,
        ) -> dict[str, object]:
            observed.append(("detail", workspace_id, graph_space))
            return {}

        def list_transcript(
            self,
            conversation_id: str,
            *,
            workspace_id: str | None,
            graph_space: str | None,
        ) -> list[object]:
            observed.append(("transcript", workspace_id, graph_space))
            return []

    app = FastAPI()
    app.include_router(
        create_chat_router(
            get_service=lambda: cast(ChatRunService, Service()),
            require_role=lambda _role: None,
            require_namespace=lambda _namespace: None,
            conversation_namespace="conversation",
            workflow_namespaces={"workflow"},
            get_user_id=lambda: "alice",
            get_workspace_id=lambda: "w1",
            get_graph_space=lambda: "conversation",
        )
    )
    with TestClient(app) as client:
        assert client.get("/api/conversations").status_code == 200
        assert client.get("/api/conversations/c1").status_code == 200
        assert client.get("/api/conversations/c1/turns").status_code == 200

    assert observed == [
        ("list", "w1", "conversation"),
        ("detail", "w1", "conversation"),
        ("transcript", "w1", "conversation"),
    ]


def test_acl_hidden_turn_is_filtered_by_python_conversation_queries(tmp_path) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "acl-conversation"),
        kg_graph_type="conversation",
        embedding_function=ConstantEmbeddingFunction(dim=3),
        backend_factory=build_in_memory_backend,
        acl_enabled=True,
    )
    conversation = ConversationService.from_engine(engine)
    owner_token = set_claims_ctx(
        {"sub": "alice", "user_id": "alice", "ns": "conversation"}
    )
    try:
        conversation_id, start_id = conversation.create_conversation(
            user_id="alice", conv_id="acl-scope-conversation"
        )
        engine.write.add_node(
            ConversationNode(
                id="acl-private-turn",
                label="Private answer",
                type="entity",
                summary="secret answer",
                conversation_id=conversation_id,
                user_id="alice",
                role="assistant",  # type: ignore[arg-type]
                turn_index=1,
                properties={
                    "content": "secret answer",
                    "entity_type": "assistant_turn",
                },
                mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                metadata={
                    "level_from_root": 0,
                    "entity_type": "assistant_turn",
                    "in_conversation_chain": True,
                },
                domain_id=None,
                canonical_entity_id=None,
            )
        )
        engine.acl.record_acl(
            grain="node",
            truth_graph="conversation",
            entity_id="acl-private-turn",
            version=2,
            mode="private",
            owner_id="bob",
            supersedes_version=1,
        )
    finally:
        reset_claims_ctx(owner_token)

    owner_token = set_claims_ctx(
        {"sub": "alice", "user_id": "alice", "ns": "conversation"}
    )
    try:
        queries = _ConversationQueryService(
            cast(
                ChatRunService,
                SimpleNamespace(
                    _conversation_engine=lambda: engine,
                    _conversation_service=lambda: conversation,
                ),
            )
        )
        detail = queries.get_conversation(conversation_id)
        transcript = queries.list_transcript(conversation_id)
        listed = queries.list_conversations_for_user("alice")
        assert detail["turn_count"] == 0
        assert transcript == []
        assert listed == [
            {
                "id": conversation_id,
                "start_node_id": start_id,
                "status": "active",
                "turn_count": 0,
            }
        ]
    finally:
        reset_claims_ctx(owner_token)

    outsider_token = set_claims_ctx(
        {"sub": "bob", "user_id": "bob", "ns": "conversation"}
    )
    try:
        with pytest.raises(KeyError, match="Unknown conversation_id"):
            queries.get_conversation(conversation_id)
        with pytest.raises(KeyError, match="Unknown conversation_id"):
            queries.list_transcript(conversation_id)
        assert queries.list_conversations_for_user("bob") == []
    finally:
        reset_claims_ctx(outsider_token)
        engine.close()
