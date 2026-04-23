from __future__ import annotations

from typing import MutableMapping

TEST_JWT_SECRET = "dev-secret"
TEST_JWT_ALG = "HS256"


def ensure_test_jwt_env(env: MutableMapping[str, str]) -> None:
    env.setdefault("JWT_SECRET", TEST_JWT_SECRET)
    env.setdefault("JWT_ALG", TEST_JWT_ALG)
