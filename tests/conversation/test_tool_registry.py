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


def test_tool_registry_filters_by_kind_async_and_provider() -> None:
    registry = ToolRegistry()
    pure = ToolDefinition(
        tool_id="tool|demo|pure",
        name="demo.pure",
        kind="pure/query",
        capability="demo.read",
        supports_async=False,
        side_effects=[],
        provider="demo",
    )
    side_effecting = ToolDefinition(
        tool_id="tool|demo|write",
        name="demo.write",
        kind="side-effecting",
        capability="demo.write",
        supports_async=True,
        side_effects=["graph_write"],
        provider="demo",
    )
    human = ToolDefinition(
        tool_id="tool|ops|approve",
        name="ops.approve",
        kind="human-approval",
        capability="ops.approve",
        supports_async=False,
        side_effects=[],
        provider="ops",
    )
    registry.register(pure)
    registry.register(side_effecting)
    registry.register(human)

    assert registry.list(kind="pure/query") == [pure]
    assert registry.list(kind="side-effecting") == [side_effecting]
    assert registry.list(supports_async=True) == [side_effecting]
    assert registry.list(provider="ops") == [human]
