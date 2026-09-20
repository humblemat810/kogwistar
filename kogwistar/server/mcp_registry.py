"""Small official-MCP registry used by Kogwistar's FastAPI server.

This keeps tool declarations close to their domain functions while delegating
protocol framing and transports to the official MCP Python SDK.
"""

from __future__ import annotations

import inspect
import json
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, get_type_hints

from mcp import types
from mcp.server.lowlevel import Server
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model
from starlette.applications import Starlette
from starlette.routing import Mount


def _inline_refs(value: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            resolved = definitions.get(reference.rsplit("/", 1)[-1], {})
            return _inline_refs(resolved, definitions)
        return {
            key: _inline_refs(item, definitions)
            for key, item in value.items()
            if key not in {"$defs", "title"}
        }
    if isinstance(value, list):
        return [_inline_refs(item, definitions) for item in value]
    return value


def _input_model(name: str, function: Callable[..., Any]) -> type[BaseModel]:
    hints = get_type_hints(function)
    fields: dict[str, tuple[Any, Any]] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.kind not in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            raise TypeError(f"MCP tool {name!r} has unsupported parameter {parameter.name!r}")
        annotation = hints.get(parameter.name, Any)
        default = (
            ...
            if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        fields[parameter.name] = (annotation, default)
    return create_model(
        f"{name.replace('.', '_').replace('-', '_')}Input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _schema_for_model(model: type[BaseModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    return _inline_refs(raw, raw.get("$defs", {}))


def _output_schema(function: Callable[..., Any]) -> dict[str, Any] | None:
    annotation = get_type_hints(function).get("return")
    if annotation is None or annotation is type(None):
        return None
    raw = TypeAdapter(annotation).json_schema()
    if not raw:
        return None
    return _inline_refs(raw, raw.get("$defs", {}))


@dataclass(frozen=True, slots=True)
class _ToolRecord:
    name: str
    function: Callable[..., Any]
    input_model: type[BaseModel]
    tool: types.Tool


class McpRegistry:
    """Decorator-friendly registry backed by an official low-level Server."""

    def __init__(self, name: str, *, filter_tools: bool = False) -> None:
        self.server = Server(name)
        self._filter_tools = filter_tools
        self._records: dict[str, _ToolRecord] = {}
        self._children: list[McpRegistry] = []
        self._register_handlers()

    def tool(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        structured_output: bool = False,
    ) -> Callable[..., Any]:
        # Keep accepting the former decorator keyword.  The official SDK
        # carries the equivalent contract in Tool.outputSchema, which this
        # registry derives from the function return annotation below.
        del structured_output

        def register(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or getattr(fn, "name", None) or fn.__name__
            input_model = _input_model(tool_name, fn)
            record = _ToolRecord(
                name=tool_name,
                function=fn,
                input_model=input_model,
                tool=types.Tool(
                    name=tool_name,
                    description=description,
                    inputSchema=_schema_for_model(input_model),
                    outputSchema=_output_schema(fn),
                ),
            )
            self._records[tool_name] = record
            # Role/namespace decorators are applied outside @mcp.tool().
            # Exposing the explicit name lets those decorators keep working.
            setattr(fn, "name", tool_name)
            return fn

        return register(function) if function is not None else register

    def mount(self, child: "McpRegistry") -> None:
        self._children.append(child)

    def _record_for(self, name: str) -> _ToolRecord | None:
        record = self._records.get(name)
        if record is not None:
            return record
        for child in self._children:
            record = child._record_for(name)
            if record is not None:
                return record
        return None

    def _register_handlers(self) -> None:
        @self.server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            return await self._visible_tools()

        @self.server.call_tool()
        async def _call_tool(
            name: str, arguments: dict[str, Any]
        ) -> types.CallToolResult:
            return await self._execute(name, arguments, enforce_visibility=True)

    def _local_tools(self) -> list[types.Tool]:
        tools = [record.tool for record in self._records.values()]
        for child in self._children:
            tools.extend(child._local_tools())
        return tools

    async def list_tools(self) -> list[types.Tool]:
        """Return the registry's currently visible tool surface.

        The application root has historically exposed a role/namespace-filtered
        read-only surface, while the mounted conversation and workflow
        registries expose their complete local declarations.  Keeping that
        distinction here preserves the old FastMCP contract for both direct
        callers and protocol clients.
        """

        if self._filter_tools:
            return await self._visible_tools()
        return [record.tool for record in self._records.values()]

    async def _visible_tools(self) -> list[types.Tool]:
        if not self._filter_tools:
            return await self.list_tools()
        from kogwistar.server.mcp_tools import _tool_allowed
        from kogwistar.server.auth_middleware import get_current_namespaces, get_current_role

        role = get_current_role()
        namespaces = get_current_namespaces()
        return [
            tool
            for tool in self._local_tools()
            if _tool_allowed(tool.name, role=role, namespaces=namespaces)
        ]

    async def _execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        enforce_visibility: bool = False,
    ) -> types.CallToolResult:
        from kogwistar.server.mcp_tools import _tool_allowed
        from kogwistar.server.auth_middleware import get_current_namespaces, get_current_role

        record = self._record_for(name)
        if record is None:
            raise ValueError(f"Unknown MCP tool: {name}")
        if enforce_visibility:
            role = get_current_role()
            namespaces = get_current_namespaces()
            if not _tool_allowed(name, role=role, namespaces=namespaces):
                raise PermissionError(
                    f"Tool {name!r} not permitted for role {role!r} in namespaces {namespaces}"
                )
        values = record.input_model.model_validate(arguments)
        kwargs = {
            field: getattr(values, field)
            for field in record.input_model.model_fields
        }
        result = record.function(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        if isinstance(result, dict):
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text", text=json.dumps(result, default=str)
                    )
                ],
                structuredContent=result,
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, default=str))]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        return await self._execute(name, arguments or {})

    def http_app(self, *, path: str = "/mcp") -> Starlette:
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.responses import PlainTextResponse

        endpoint = path.rstrip("/") or "/"
        manager_holder: dict[str, StreamableHTTPSessionManager] = {}

        @asynccontextmanager
        async def lifespan(app: Any):
            manager = StreamableHTTPSessionManager(self.server)
            manager_holder["manager"] = manager
            async with manager.run():
                try:
                    yield
                finally:
                    manager_holder.pop("manager", None)

        async def scoped_handler(scope: Any, receive: Any, send: Any) -> None:
            if scope.get("type") == "http":
                request_path = str(scope.get("path") or "")
                if request_path not in {endpoint, endpoint + "/"}:
                    await PlainTextResponse("Not Found", status_code=404)(
                        scope, receive, send
                    )
                    return
            manager = manager_holder.get("manager")
            if manager is None:
                await PlainTextResponse("MCP server is not started", status_code=503)(
                    scope, receive, send
                )
                return
            await manager.handle_request(scope, receive, send)

        app = Starlette(
            routes=[Mount("/", app=scoped_handler)],
            lifespan=lifespan,
        )
        # Kogwistar's combined FastAPI lifespan enters this context explicitly.
        app.lifespan = lifespan  # type: ignore[attr-defined]
        return app

    def streamable_http_app(self, *, path: str = "/mcp") -> Starlette:
        """Compatibility factory backed by the official streamable HTTP app."""

        return self.http_app(path=path)

    def run(
        self,
        *,
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
        path: str = "/mcp",
    ) -> None:
        """Run an official-SDK transport for the standalone compatibility entrypoint."""

        if transport == "stdio":
            from mcp.server.stdio import stdio_server

            async def _run() -> None:
                async with stdio_server() as (read_stream, write_stream):
                    await self.server.run(
                        read_stream,
                        write_stream,
                        self.server.create_initialization_options(),
                    )

            asyncio.run(_run())
            return
        import uvicorn

        if transport != "streamable-http":
            raise ValueError(f"unsupported MCP transport: {transport}")
        uvicorn.run(self.http_app(path=path), host=host, port=port)


__all__ = ["McpRegistry"]
