from __future__ import annotations

import asyncio
from typing import Any

from mcp import types

from kogwistar.diagnostic.mcp_diag_seed import _call_tool_json, _find_tool


class _FakeSession:
    async def list_tools(
        self, *, params: types.PaginatedRequestParams | None = None
    ) -> types.ListToolsResult:
        del params
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="demo",
                    inputSchema={"type": "object", "properties": {}},
                    outputSchema={"type": "object"},
                )
            ]
        )

    async def call_tool(
        self, name: str, *, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        assert name == "demo"
        assert arguments == {"value": 1}
        return types.CallToolResult(
            content=[types.TextContent(text='{"ok": true}')],
            structuredContent={"ok": True},
        )


def test_mcp_v2_tool_schema_and_structured_result() -> None:
    async def run() -> None:
        session = _FakeSession()
        tool = await _find_tool(session, "demo")
        assert tool == {
            "name": "demo",
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object"},
        }
        assert await _call_tool_json(session, "demo", {"value": 1}) == {"ok": True}

    asyncio.run(run())
