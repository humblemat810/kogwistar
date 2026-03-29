from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from kogwistar.server.mcp_tools import MCPRoleMiddleware
import kogwistar.server_mcp_with_admin as server


@pytest.mark.asyncio
async def test_mcp_role_middleware_passthrough_for_non_mcp_streaming_routes():
    messages = [
        {"type": "http.response.start", "status": 200, "headers": []},
        {"type": "http.response.body", "body": b"first", "more_body": True},
        {"type": "http.response.body", "body": b"second", "more_body": False},
    ]

    async def app(scope, receive, send):
        for message in messages:
            await send(message)

    middleware = MCPRoleMiddleware(app)
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "path": "/api/runs/demo/events",
        "headers": [],
        "method": "GET",
    }

    await middleware(scope, receive, send)

    assert sent == messages


@contextmanager
def _patched_stream_route(app: FastAPI, path: str):
    async def _stream():
        yield b"first\n"
        await asyncio.sleep(0.05)
        yield b"second\n"

    def _endpoint() -> StreamingResponse:
        return StreamingResponse(_stream(), media_type="text/plain")

    route = APIRoute(path, _endpoint, methods=["GET"], include_in_schema=False)
    app.router.routes.insert(0, route)
    try:
        yield
    finally:
        app.router.routes = [existing for existing in app.router.routes if existing is not route]


def test_mcp_role_middleware_allows_real_app_streaming_routes():
    app = server.app
    with _patched_stream_route(app, "/__test__/stream"):
        with TestClient(app) as client:
            with client.stream("GET", "/__test__/stream") as resp:
                resp.raise_for_status()
                lines = [
                    line.decode() if isinstance(line, bytes) else str(line)
                    for line in resp.iter_lines()
                ]

    assert lines == ["first", "second"]
