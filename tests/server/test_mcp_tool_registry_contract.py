from __future__ import annotations

import ast
import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import BaseModel

pytestmark = [pytest.mark.ci, pytest.mark.server, pytest.mark.unit]


class _StructuredResult(BaseModel):
    value: int


def test_registry_accepts_structured_output_compatibility_option() -> None:
    from kogwistar.server.mcp_registry import McpRegistry

    registry = McpRegistry("structured-output-test")

    @registry.tool(structured_output=True)
    def answer() -> _StructuredResult:
        return _StructuredResult(value=42)

    async def exercise() -> None:
        tools = await registry.list_tools()
        tool = next(item for item in tools if item.name == "answer")
        assert tool.output_schema is not None

        result = await registry.call_tool("answer")
        assert result.structured_content == {"value": 42}

    asyncio.run(exercise())


def test_registry_sync_invocation_context_is_scoped_to_tool_body() -> None:
    from kogwistar.server.mcp_registry import McpRegistry

    entered: list[str] = []

    @contextmanager
    def context():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    registry = McpRegistry(
        "sync-context-test", sync_invocation_context=lambda _name: context()
    )

    @registry.tool()
    def answer() -> dict[str, int]:
        assert entered == ["enter"]
        return {"value": 42}

    async def exercise() -> None:
        result = await registry.call_tool("answer")
        assert result.structured_content == {"value": 42}

    asyncio.run(exercise())
    assert entered == ["enter", "exit"]


def test_registry_does_not_hold_sync_context_across_async_tool_await() -> None:
    from kogwistar.server.mcp_registry import McpRegistry

    entered: list[str] = []

    @contextmanager
    def context():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    registry = McpRegistry(
        "async-context-test", sync_invocation_context=lambda _name: context()
    )

    @registry.tool()
    async def answer() -> dict[str, int]:
        assert entered == ["enter", "exit"]
        return {"value": 42}

    async def exercise() -> None:
        result = await registry.call_tool("answer")
        assert result.structured_content == {"value": 42}

    asyncio.run(exercise())
    assert entered == ["enter", "exit"]


def test_registry_authorization_precedes_sync_invocation_context(monkeypatch) -> None:
    from kogwistar.server import mcp_tools
    from kogwistar.server.mcp_registry import McpRegistry

    entered: list[str] = []

    @contextmanager
    def context():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    registry = McpRegistry(
        "authorization-order-test", sync_invocation_context=lambda _name: context()
    )

    @registry.tool()
    def answer() -> dict[str, int]:
        return {"value": 42}

    monkeypatch.setattr(
        mcp_tools, "_tool_allowed", lambda *_args, **_kwargs: False
    )

    async def exercise() -> None:
        with pytest.raises(PermissionError):
            await registry._execute("answer", {}, enforce_visibility=True)

    asyncio.run(exercise())
    assert entered == []


def test_production_sync_context_is_knowledge_only_and_unknown_fails_closed(monkeypatch) -> None:
    from kogwistar.server import mcp_tools
    from kogwistar.server.auth_middleware import NameSpace

    def forbidden_engine_access():
        raise AssertionError("non-knowledge tool opened the knowledge engine")

    # Patch the lazy factory itself; attribute proxying would initialize the
    # resource before the tested non-knowledge tool path is entered.
    monkeypatch.setattr(mcp_tools.engine, "_factory", forbidden_engine_access)
    mcp_tools.TOOL_NAMESPACE["test_wisdom_tool"] = {NameSpace.WISDOM.value}
    mcp_tools.TOOL_NAMESPACE["test_workflow_tool"] = {NameSpace.WORKFLOW.value}
    try:
        for name in ("test_wisdom_tool", "test_workflow_tool", "test_unknown_tool"):
            context = mcp_tools._knowledge_sync_invocation_context(name)
            with context:
                pass
    finally:
        for name in ("test_wisdom_tool", "test_workflow_tool"):
            mcp_tools.TOOL_NAMESPACE.pop(name, None)


def test_workflow_mcp_tool_names_are_declared_once() -> None:
    source = Path("kogwistar/server/chat_mcp.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.keywords:
                continue
            function = decorator.func
            if not (
                isinstance(function, ast.Attribute)
                and function.attr == "tool"
                and isinstance(function.value, ast.Name)
                and function.value.id == "mcp"
            ):
                continue
            keyword = next((item for item in decorator.keywords if item.arg == "name"), None)
            if keyword and isinstance(keyword.value, ast.Constant):
                names.append(str(keyword.value.value))

    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == []


def test_committed_mcp_tool_schema_matches_live_registry() -> None:
    from kogwistar.server.mcp_tools import conversation_mcp, workflow_mcp
    from kogwistar.server_mcp_with_admin import mcp

    async def contracts(registry) -> list[dict]:
        tools = await registry.list_tools()
        return sorted(
            (
                {
                    "name": tool.name,
                    "title": tool.title,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                    "outputSchema": tool.output_schema,
                }
                for tool in tools
            ),
            key=lambda item: item["name"],
        )

    async def surfaces() -> dict[str, list[dict]]:
        return {
            "root": await contracts(mcp),
            "conversation": await contracts(conversation_mcp),
            "workflow": await contracts(workflow_mcp),
        }

    live = asyncio.run(surfaces())
    frozen = json.loads(
        Path("contracts/golden/mcp-tools.json").read_text(encoding="utf-8")
    )
    assert frozen == {"contract_version": "1.0.0", "surfaces": live}
