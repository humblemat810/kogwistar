from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jose import jwt

from kogwistar.server.auth_middleware import (
    JWTProtectMiddleware,
    claims_ctx,
    get_current_role,
    get_current_subject,
    set_auth_app,
)
from kogwistar.server.mcp_tools import MCPRoleMiddleware
import kogwistar.server_mcp_with_admin as server

pytestmark = pytest.mark.ci


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


def test_jwt_protect_middleware_preserves_claims_context(monkeypatch):
    monkeypatch.setenv("JWT_ALG", "HS256")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("JWT_ISS", "local")

    app = FastAPI()
    set_auth_app(app)
    app.add_middleware(JWTProtectMiddleware)

    @app.get("/probe")
    def probe():
        claims = claims_ctx.get() or {}
        return {
            "role": get_current_role(),
            "subject": get_current_subject(),
            "claims_role": claims.get("role"),
            "claims_sub": claims.get("sub"),
        }

    token = jwt.encode(
        {"sub": "tester", "role": "rw", "ns": "workflow", "iss": "local"},
        "test-secret",
        algorithm="HS256",
    )

    with TestClient(app) as client:
        resp = client.get("/probe", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "role": "rw",
        "subject": "tester",
        "claims_role": "rw",
        "claims_sub": "tester",
    }


def test_jwt_protect_middleware_uses_frozen_app_jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_ALG", "HS256")
    monkeypatch.setenv("JWT_SECRET", "first-secret")
    monkeypatch.setenv("JWT_ISS", "local")

    app = FastAPI()
    set_auth_app(app)
    app.add_middleware(JWTProtectMiddleware)

    @app.get("/probe")
    def probe():
        claims = claims_ctx.get() or {}
        return {"role": claims.get("role"), "subject": claims.get("sub")}

    token = jwt.encode(
        {"sub": "tester", "role": "rw", "ns": "workflow", "iss": "local"},
        "first-secret",
        algorithm="HS256",
    )

    monkeypatch.setenv("JWT_SECRET", "second-one")

    with TestClient(app) as client:
        resp = client.get("/probe", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"role": "rw", "subject": "tester"}
