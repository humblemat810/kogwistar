from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
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

pytestmark = pytest.mark.ci


def test_jwt_constants_follow_runtime_env_and_repo_dotenv(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "dev-secret")
    monkeypatch.setenv("JWT_ALG", "HS256")

    auth_middleware = importlib.import_module("kogwistar.server.auth_middleware")
    auth_middleware = importlib.reload(auth_middleware)

    # Constants should reflect runtime env, not an internal hardcoded fallback.
    assert auth_middleware.JWT_SECRET == "dev-secret"
    assert auth_middleware.JWT_ALG == "HS256"


def test_jwt_claims_context_survives_auth_middleware_reload(monkeypatch):
    monkeypatch.setenv("JWT_ALG", "HS256")
    monkeypatch.setenv("JWT_SECRET", "reload-secret")
    monkeypatch.setenv("JWT_ISS", "local")

    auth_middleware = importlib.import_module("kogwistar.server.auth_middleware")
    auth_middleware = importlib.reload(auth_middleware)

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
        "reload-secret",
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


def test_mcp_role_middleware_survives_auth_middleware_reload(monkeypatch):
    monkeypatch.setenv("JWT_ALG", "HS256")
    monkeypatch.setenv("JWT_SECRET", "reload-secret")
    monkeypatch.setenv("JWT_ISS", "local")

    auth_middleware = importlib.import_module("kogwistar.server.auth_middleware")
    auth_middleware = importlib.reload(auth_middleware)

    app = FastAPI()
    set_auth_app(app)
    app.add_middleware(JWTProtectMiddleware)
    app.add_middleware(MCPRoleMiddleware)

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
        "reload-secret",
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
