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
    capability: str = ""
    status: str = "completed"
    input: dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "inline"
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

    def list(
        self,
        *,
        kind: str | None = None,
        supports_async: bool | None = None,
        provider: str | None = None,
    ) -> list[ToolDefinition]:
        out = list(self._tools.values())
        if kind is not None:
            out = [tool for tool in out if tool.kind == kind]
        if supports_async is not None:
            out = [tool for tool in out if bool(tool.supports_async) == bool(supports_async)]
        if provider is not None:
            out = [tool for tool in out if tool.provider == provider]
        return out

    def requires(self, tool_id: str) -> ToolRequirement | None:
        tool = self.get(tool_id)
        if tool is None or not tool.capability:
            return None
        return ToolRequirement(capability=tool.capability)
