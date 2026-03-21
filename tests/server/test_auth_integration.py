from __future__ import annotations
import json
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from sqlalchemy.orm import sessionmaker

import kogwistar.server_mcp_with_admin as server
from kogwistar.server.auth.db import (
    create_auth_engine,
    init_auth_db,
)
from kogwistar.server.auth.seeding import seed_auth_data
from kogwistar.server.auth.service import AuthService
from kogwistar.server.auth.models import User, ExternalIdentity

pytestmark = [pytest.mark.ci_full]


@pytest.fixture(params=["memory", "file"])
def db_url(request, tmp_path):
    if request.param == "memory":
        return "sqlite:///:memory:"
    else:
        db_file = tmp_path / "test_auth.sqlite"
        return f"sqlite:///{db_file}"


@pytest.fixture
def auth_engine(db_url):
    # create_auth_engine handles StaticPool/check_same_thread for in-memory cases in tests
    engine = create_auth_engine(db_url)
    init_auth_db(engine)
    return engine


@pytest.fixture
def auth_db(auth_engine):
    # This session will share the same underlying connection/file
    Session = sessionmaker(bind=auth_engine)
    session = Session()
    yield session
    session.close()


def test_seed_auth_data(auth_db):
    seed_json = json.dumps(
        [
            {
                "user_id": "alice-id",
                "email": "alice@example.com",
                "display_name": "Alice Admin",
                "global_role": "rw",
                "global_ns": "docs,workflow",
                "identities": [{"issuer": "http://keycloak", "subject": "alice-sub"}],
            }
        ]
    )

    with patch.dict(os.environ, {"DEV_AUTH_SEED_JSON": seed_json}):
        seed_auth_data(auth_db)

    user = auth_db.query(User).filter(User.user_id == "alice-id").first()
    assert user is not None
    assert user.email == "alice@example.com"
    assert user.global_role == "rw"
    assert user.global_ns == "docs,workflow"

    identity = (
        auth_db.query(ExternalIdentity)
        .filter(ExternalIdentity.subject == "alice-sub")
        .first()
    )
    assert identity is not None
    assert identity.user_id == "alice-id"


def test_auth_service_mint_token_with_seeded_data(auth_db):
    user = User(
        user_id="bob-id", email="bob@example.com", global_role="ro", global_ns="wisdom"
    )
    auth_db.add(user)
    auth_db.commit()

    service = AuthService(auth_db, jwt_secret="secret")
    token = service.mint_token("bob-id")

    from jose import jwt

    claims = jwt.decode(token, "secret", algorithms=["HS256"])
    assert claims["role"] == "ro"
    assert claims["ns"] == "wisdom"


def test_auth_service_mint_token_with_multi_ns(auth_db):
    user = User(
        user_id="multi-id",
        email="multi@example.com",
        global_role="rw",
        global_ns="docs,workflow,conversation",
    )
    auth_db.add(user)
    auth_db.commit()

    service = AuthService(auth_db, jwt_secret="secret")
    token = service.mint_token("multi-id")

    from jose import jwt

    claims = jwt.decode(token, "secret", algorithms=["HS256"])
    assert isinstance(claims["ns"], list)
    assert set(claims["ns"]) == {"docs", "workflow", "conversation"}


def test_oidc_flow_integration(monkeypatch, auth_engine, auth_db):
    # Setup server state to use our shared test engine
    server.auth_engine_resource._value = auth_engine
    server.auth_engine_resource._state = "initialized"

    with patch(
        "kogwistar.server.auth.router._get_auth_mode", return_value="oidc"
    ):
        mock_oidc = AsyncMock()
        mock_oidc.discovery_url = "http://mock-issuer"
        mock_oidc.get_auth_url.return_value = "http://mock-auth-url"
        mock_oidc.exchange_code.return_value = {
            "access_token": "mock-access-token",
            "id_token": "mock-id-token",
        }
        mock_oidc.validate_id_token.return_value = {
            "sub": "oidc-sub",
            "email": "oidc@example.com",
            "name": "OIDC User",
            "iss": "http://mock-issuer/realm",
            "aud": "kge-local",
            "nonce": "nonce-123",
        }
        mock_oidc.get_userinfo.return_value = {
            "sub": "oidc-sub",
            "email": "oidc@example.com",
            "name": "OIDC User",
        }

        # Setup AuthService in app state
        service = AuthService(auth_db, jwt_secret=server.JWT_SECRET)
        server.app.state.auth_service = service
        server.app.state.auth_mode = "oidc"
        server.app.state.oidc_clients = {"test": mock_oidc}
        server.app.state.oidc_default_provider = "test"
        server.app.state.oidc_provider_configs = {}

        client = TestClient(server.app)

        # 1. Login redirect
        resp = client.get("/api/auth/login", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "http://mock-auth-url"
        state_cookie = resp.cookies.get("auth_state")
        pkce_cookie = resp.cookies.get("auth_pkce_verifier")
        nonce_cookie = resp.cookies.get("auth_nonce")
        provider_cookie = resp.cookies.get("auth_provider")

        # 2. Callback
        callback_resp = client.get(
            f"/api/auth/callback?code=mock-code&state={state_cookie}",
            cookies={
                "auth_state": state_cookie,
                "auth_pkce_verifier": pkce_cookie,
                "auth_nonce": nonce_cookie,
                "auth_provider": provider_cookie,
            },
            follow_redirects=False,
        )
        assert callback_resp.status_code == 307
        location = callback_resp.headers["location"]
        assert "?token=" in location

        # 3. Verify user created in DB
        auth_db.expire_all()  # Ensure we fetch fresh data
        user = auth_db.query(User).filter(User.email == "oidc@example.com").first()
        assert user is not None
        assert user.display_name == "OIDC User"

        identity = (
            auth_db.query(ExternalIdentity)
            .filter(ExternalIdentity.subject == "oidc-sub")
            .first()
        )
        assert identity is not None
        assert identity.user_id == user.user_id
