"""Semantic test group extracted from the former monolithic API suite."""
# ruff: noqa: F403,F405
from __future__ import annotations

from .chat_mcp_tools_support import *  # noqa: F403

@pytest.mark.asyncio
async def test_mcp_conversation_create_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-create-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("rw", "conversation"):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert _structured(result) == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_ask_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-ask-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("rw", "conversation"):
        await server.mcp.call_tool(
            "conversation.create",
            {
                "user_id": contract["arguments"]["user_id"],
                "conversation_id": contract["arguments"]["conversation_id"],
            },
        )
        result = _structured(
            await server.mcp.call_tool(contract["tool"], contract["arguments"])
        )

    assert result["run_id"]
    assert result["user_turn_node_id"]
    result["run_id"] = "<generated>"
    result["user_turn_node_id"] = "<generated>"
    assert result == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_cancel_run_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-cancel-run-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )

    with _claims("rw", "conversation"):
        await server.mcp.call_tool(
            "conversation.create",
            {"user_id": "alice", "conversation_id": "conv-cancel-contract"},
        )
        service.run_registry.create_run(
            run_id=contract["arguments"]["run_id"],
            conversation_id="conv-cancel-contract",
            workflow_id="agentic_answering.v2",
            user_id="alice",
            user_turn_node_id="turn-cancel-contract",
        )
        result = _structured(
            await server.mcp.call_tool(
                contract["tool"], contract["arguments"]
            )
        )

    assert result["status"] == "cancelling"
    assert result["cancel_requested"] is True
    result["created_at_ms"] = "<generated>"
    result["updated_at_ms"] = "<generated>"
    assert result == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_cancel_run_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _cancel_runner
    )
    service.run_registry.create_run(
        run_id="cancel-auth-run",
        conversation_id="cancel-auth-conversation",
        workflow_id="agentic_answering.v2",
        user_id="cancel-auth-owner",
        user_turn_node_id="cancel-auth-turn",
    )
    args = {"run_id": "cancel-auth-run"}

    with _claims(
        "ro",
        "conversation",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": ["workflow.run.write"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.cancel_run", args, enforce_visibility=True
            )

    with _claims(
        "rw",
        "workflow",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": ["workflow.run.write"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.cancel_run", args, enforce_visibility=True
            )

    with _claims(
        "rw",
        "conversation",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": []},
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.cancel_run", args, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    run = service.run_registry.get_run("cancel-auth-run")
    assert run is not None
    assert run["terminal"] is False
    assert run["cancel_requested"] is False

    service.run_registry.create_run(
        run_id="cancel-terminal-retry-run",
        conversation_id="cancel-terminal-retry-conversation",
        workflow_id="agentic_answering.v2",
        user_id="cancel-auth-owner",
        user_turn_node_id="cancel-terminal-retry-turn",
        status="cancelled",
    )
    terminal_events_before = service.run_registry.list_events(
        "cancel-terminal-retry-run"
    )
    with _claims(
        "rw",
        "conversation",
        sub="cancel-auth-owner",
        scope_parts={"capabilities": []},
    ):
        terminal_retry = _structured(
            await server.mcp._execute(
                "conversation.cancel_run",
                {"run_id": "cancel-terminal-retry-run"},
                enforce_visibility=True,
            )
        )
    assert terminal_retry["status"] == "cancelled"
    assert terminal_retry["terminal"] is True
    assert terminal_retry["cancel_requested"] is False
    assert service.run_registry.list_events("cancel-terminal-retry-run") == terminal_events_before
    assert conversation_engine.get_nodes(
        where={
            "$and": [
                {"entity_type": "workflow_cancel_request"},
                {"run_id": "cancel-terminal-retry-run"},
            ]
        },
        limit=10,
    ) == []


@pytest.mark.asyncio
async def test_mcp_conversation_run_status_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-run-status-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    service.run_registry.create_run(
        run_id=contract["arguments"]["run_id"],
        conversation_id=contract["result"]["conversation_id"],
        workflow_id=contract["result"]["workflow_id"],
        user_id=contract["result"]["user_id"],
        user_turn_node_id=contract["result"]["user_turn_node_id"],
        status=contract["result"]["status"],
    )

    with _claims(
        "ro",
        "conversation",
        sub="alice",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        result = _structured(
            await server.mcp.call_tool(contract["tool"], contract["arguments"])
        )

    result["created_at_ms"] = "<generated>"
    result["updated_at_ms"] = "<generated>"
    assert result == contract["result"]


@pytest.mark.asyncio
async def test_mcp_conversation_run_status_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    import pytest
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    service.run_registry.create_run(
        run_id="status-auth-run",
        conversation_id="status-auth-conversation",
        workflow_id="agentic_answering.v2",
        user_id="status-auth-owner",
        user_turn_node_id="status-auth-turn",
        status="running",
    )
    args = {"run_id": "status-auth-run"}

    with _claims(
        "ro",
        "conversation",
        sub="status-auth-owner",
        scope_parts={"capabilities": []},
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.run_status", args, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "ro",
        "workflow",
        sub="status-auth-owner",
        scope_parts={"capabilities": ["workflow.run.read"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.run_status", args, enforce_visibility=True
            )

    run = service.run_registry.get_run("status-auth-run")
    assert run is not None
    assert run["status"] == "running"


@pytest.mark.asyncio
async def test_mcp_conversation_create_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    tool_args = {
        "user_id": "unauthorized-create",
        "conversation_id": "unauthorized-create-conversation",
        "start_node_id": "unauthorized-create-start",
    }

    with _claims(
        "rw",
        "conversation",
        sub="unauthorized-create",
        scope_parts={"capabilities": []},
    ):
        with pytest.raises(HTTPException):
            await server.mcp._execute(
                "conversation.create", tool_args, enforce_visibility=True
            )

    with _claims(
        "rw",
        "workflow",
        sub="unauthorized-create",
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.create", tool_args, enforce_visibility=True
            )

    assert not conversation_engine.read.get_nodes(
        where={"conversation_id": tool_args["conversation_id"]}, limit=10
    )


@pytest.mark.asyncio
async def test_mcp_conversation_ask_denies_unauthorized_effects(
    monkeypatch, engine_triplet
):
    from fastapi import HTTPException

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    owner = "unauthorized-ask-owner"
    conversation_id = "unauthorized-ask-conversation"
    with _claims(
        "rw",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        await server.mcp.call_tool(
            "conversation.create",
            {"user_id": owner, "conversation_id": conversation_id},
        )

    ask_args = {
        "conversation_id": conversation_id,
        "user_id": owner,
        "text": "This must not be admitted",
    }
    with _claims(
        "rw", "conversation", sub=owner, scope_parts={"capabilities": []}
    ):
        with pytest.raises(HTTPException) as missing_capability:
            await server.mcp._execute(
                "conversation.ask", ask_args, enforce_visibility=True
            )
        assert missing_capability.value.status_code == 403

    with _claims(
        "rw",
        "workflow",
        sub=owner,
        scope_parts={"capabilities": ["send_message"]},
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.ask", ask_args, enforce_visibility=True
            )

    conversation_nodes = conversation_engine.read.get_nodes(
        where={"conversation_id": conversation_id}, limit=10
    )
    assert len(conversation_nodes) == 1
    assert service.run_registry.meta_store.list_server_runs(
        conversation_id=conversation_id, limit=10
    ) == []


@pytest.mark.asyncio
async def test_mcp_conversation_get_transcript_matches_shared_contract(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts/golden/conversation-get-transcript-mcp-v1.json"
        ).read_text(encoding="utf-8")
    )
    owner = "transcript-contract-owner"

    with _claims(
        "rw",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        await server.mcp.call_tool(
            "conversation.create",
            {
                "user_id": owner,
                "conversation_id": contract["arguments"]["conversation_id"],
            },
        )

    with _claims(
        "ro",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["read_graph"]},
    ):
        result = await server.mcp.call_tool(contract["tool"], contract["arguments"])

    assert _structured(result) == contract["result"]

    from fastapi.testclient import TestClient
    from starlette.routing import Mount
    from tests._helpers.server_http_helpers import token_header

    saved_mcp_app = server.mcp_app
    streamable_mcp_app = server.mcp.http_app(path="/mcp")
    server.mcp_app = streamable_mcp_app
    for route in server.app.router.routes:
        if isinstance(route, Mount) and getattr(route, "app", None) is saved_mcp_app:
            route.app = streamable_mcp_app
            break
    try:
        with TestClient(server.app) as client:
            headers = token_header(
                client,
                role="ro",
                ns="conversation",
                username=owner,
                capabilities="read_graph",
            )
            headers["Accept"] = "application/json, text/event-stream"
            initialized = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "contract-test", "version": "1"},
                    },
                },
            )
            assert initialized.status_code == 200, initialized.text
            headers["Mcp-Session-Id"] = initialized.headers["mcp-session-id"]
            client.post(
                "/mcp",
                headers=headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            response = client.post(
                "/mcp",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": contract["tool"],
                        "arguments": contract["arguments"],
                    },
                },
            )
            assert response.status_code == 200, response.text
            wire_message = json.loads(
                next(
                    line.removeprefix("data: ")
                    for line in response.text.splitlines()
                    if line.startswith("data: ")
                )
            )
            tool_result = wire_message["result"]
            assert tool_result["isError"] is False
            assert tool_result["structuredContent"] == contract["result"]
    finally:
        server.mcp_app = saved_mcp_app
        for route in server.app.router.routes:
            if isinstance(route, Mount) and getattr(route, "app", None) is streamable_mcp_app:
                route.app = saved_mcp_app
                break


@pytest.mark.asyncio
async def test_mcp_conversation_get_transcript_denies_missing_capability_and_namespace(
    monkeypatch, engine_triplet
):
    import pytest

    engine, conversation_engine, workflow_engine = engine_triplet
    service = _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    conversation_id = "transcript-authorization-check"
    owner = "transcript-authorization-owner"
    original_list_transcript = service.list_transcript
    with _claims(
        "rw",
        "conversation",
        sub=owner,
        scope_parts={"capabilities": ["spawn_process"]},
    ):
        await server.mcp.call_tool(
            "conversation.create",
            {"user_id": owner, "conversation_id": conversation_id},
        )

    def denied_transcript_read(_conversation_id: str) -> list[dict[str, object]]:
        raise AssertionError("denied MCP request reached transcript service")

    monkeypatch.setattr(service, "list_transcript", denied_transcript_read)
    arguments = {"conversation_id": conversation_id}
    with _claims(
        "ro", "conversation", sub=owner, scope_parts={"capabilities": []}
    ):
        with pytest.raises(PermissionError) as missing_capability:
            await server.mcp._execute(
                "conversation.get_transcript", arguments, enforce_visibility=True
            )
        assert "not permitted" in str(missing_capability.value)

    with _claims(
        "ro", "workflow", sub=owner, scope_parts={"capabilities": ["read_graph"]}
    ):
        with pytest.raises(PermissionError):
            await server.mcp._execute(
                "conversation.get_transcript", arguments, enforce_visibility=True
            )

    monkeypatch.setattr(service, "list_transcript", original_list_transcript)
    with _claims(
        "ro", "conversation", sub=owner, scope_parts={"capabilities": ["read_graph"]}
    ):
        result = await server.mcp.call_tool("conversation.get_transcript", arguments)
    assert _structured(result) == {"conversation_id": conversation_id, "turns": []}

    with _claims("ro", "conversation", sub=owner):
        legacy_result = await server.mcp.call_tool(
            "conversation.get_transcript", arguments
        )
    assert _structured(legacy_result) == {
        "conversation_id": conversation_id,
        "turns": [],
    }


@pytest.mark.asyncio
async def test_wisdom_graph_upsert_mcp_requires_authorized_wisdom_writer(
    monkeypatch, engine_triplet
):
    engine, conversation_engine, workflow_engine = engine_triplet
    _configure_server(
        monkeypatch, engine, conversation_engine, workflow_engine, _success_runner
    )
    from kogwistar.server import mcp_tools

    persist_calls: list[tuple[int, int, str]] = []

    class WisdomStore:
        def persist_graph(self, *, parsed, session_id: str) -> dict[str, object]:
            persist_calls.append((len(parsed.nodes), len(parsed.edges), session_id))
            return {
                "node_ids": [],
                "edge_ids": [],
                "nodes_added": 0,
                "edges_added": 0,
            }

    monkeypatch.setattr(mcp_tools, "wisdom_engine", _FixedResource(WisdomStore()))
    contract_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "wisdom-graph-upsert-mcp-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["operation"] == "wisdom.kg_upsert_graph"
    arguments = contract["arguments"]

    for role, namespace in (("ro", "wisdom"), ("rw", "docs")):
        with _claims(role, namespace):
            names = {tool.name for tool in await server.mcp.list_tools()}
            assert "wisdom.kg_upsert_graph" not in names
            with pytest.raises(PermissionError):
                await server.mcp._execute(
                    "wisdom.kg_upsert_graph", arguments, enforce_visibility=True
                )
    assert persist_calls == []

    with _claims("rw", "wisdom"):
        names = {tool.name for tool in await server.mcp.list_tools()}
        assert "wisdom.kg_upsert_graph" in names
        result = await server.mcp._execute(
            "wisdom.kg_upsert_graph", arguments, enforce_visibility=True
        )

    assert result.structured_content == contract["expected_structured_content"]
    assert len(persist_calls) == 1
    assert persist_calls[0][:2] == (0, 0)
    assert persist_calls[0][2].startswith("wisdom:")
