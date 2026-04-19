from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.ci


def test_jwt_constants_follow_runtime_env_and_repo_dotenv(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "dev-secret")
    monkeypatch.setenv("JWT_ALG", "HS256")

    auth_middleware = importlib.import_module("kogwistar.server.auth_middleware")
    auth_middleware = importlib.reload(auth_middleware)

    # Constants should reflect runtime env, not an internal hardcoded fallback.
    assert auth_middleware.JWT_SECRET == "dev-secret"
    assert auth_middleware.JWT_ALG == "HS256"
