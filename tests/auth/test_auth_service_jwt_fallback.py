from __future__ import annotations

import pytest

from kogwistar.server.auth.db import create_auth_engine, get_session, init_auth_db
from kogwistar.server.auth.service import AuthService


pytestmark = pytest.mark.ci


def test_auth_service_falls_back_when_secret_not_passed(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("JWT_ISS", raising=False)
    monkeypatch.delenv("JWT_AUD", raising=False)

    engine = create_auth_engine("sqlite:///:memory:", allow_in_memory=True)
    init_auth_db(engine)
    with pytest.raises(RuntimeError, match="JWT secret is required"):
        AuthService(get_session(), jwt_secret=None)  # type: ignore[arg-type]
