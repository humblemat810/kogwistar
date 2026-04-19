from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.ci


def test_jwt_constants_have_dev_fallback_when_env_missing(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_ALG", raising=False)

    auth_middleware = importlib.import_module("kogwistar.server.auth_middleware")
    auth_middleware = importlib.reload(auth_middleware)

    assert auth_middleware.JWT_SECRET == "dev-secret"
    assert auth_middleware.JWT_ALG == "HS256"

