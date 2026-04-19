from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolRequirement:
    capability: str
    mode: str = "required"


@dataclass(frozen=True)
class ToolDefinition:
    tool_id: str
    name: str
    kind: str = "pure/query"
    capability: str = ""
    supports_async: bool = False
    side_effects: list[str] = field(default_factory=list)
    provider: str = "conversation"


@dataclass(frozen=True)
class ToolReceipt:
    tool_id: str
    tool_name: str
    kind: str
    capability: str
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    side_effects: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> ToolDefinition:
        self._tools[tool.tool_id] = tool
        return tool

    def revoke(self, tool_id: str) -> None:
        self._tools.pop(tool_id, None)

    def get(self, tool_id: str) -> ToolDefinition | None:
        return self._tools.get(tool_id)

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def requires(self, tool_id: str) -> ToolRequirement | None:
        tool = self.get(tool_id)
        if tool is None or not tool.capability:
            return None
        return ToolRequirement(capability=tool.capability)
