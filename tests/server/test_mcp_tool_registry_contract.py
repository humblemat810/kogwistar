from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ci]


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
                    "inputSchema": tool.parameters,
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
