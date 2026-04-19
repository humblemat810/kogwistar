from __future__ import annotations

import importlib
import os

import pytest

pytestmark = pytest.mark.ci


def test_jwt_constants_follow_runtime_env_and_repo_dotenv(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_ALG", raising=False)

    auth_middleware = importlib.import_module("kogwistar.server.auth_middleware")
    auth_middleware = importlib.reload(auth_middleware)

    # Module import loads repo .env, so constants should reflect dotenv/env values,
    # not an internal hardcoded fallback.
    assert auth_middleware.JWT_SECRET == os.getenv("JWT_SECRET") == "dev-secret"
    assert auth_middleware.JWT_ALG == "HS256"
