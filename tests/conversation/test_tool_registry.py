from __future__ import annotations

from kogwistar.conversation.tool_registry import ToolDefinition, ToolRegistry


def test_tool_registry_register_list_and_requirements() -> None:
    registry = ToolRegistry()
    tool = ToolDefinition(
        tool_id="tool|demo|echo",
        name="demo.echo",
        kind="pure/query",
        capability="echo.read",
        supports_async=False,
        side_effects=[],
        provider="demo",
    )
    registry.register(tool)

    listed = registry.list()
    assert listed == [tool]
    assert registry.get("tool|demo|echo") == tool
    assert registry.requires("tool|demo|echo").capability == "echo.read"

    registry.revoke("tool|demo|echo")
    assert registry.get("tool|demo|echo") is None
