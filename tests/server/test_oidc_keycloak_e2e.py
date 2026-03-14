from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

import graph_knowledge_engine.server_mcp_with_admin as server
from graph_knowledge_engine.server.auth.db import create_auth_engine, init_auth_db
from graph_knowledge_engine.server.auth.models import ExternalIdentity, User
from graph_knowledge_engine.server.auth.oidc import OIDCClient
from graph_knowledge_engine.server.auth.seeding import seed_auth_data
from graph_knowledge_engine.server.auth.service import AuthService
from tests.server.oidc_test_support import extract_login_action


pytestmark = [pytest.mark.integration]


@pytest.fixture
def oidc_test_client(monkeypatch, keycloak_container: dict[str, str], oidc_test_identity):
    engine = create_auth_engine("sqlite:///:memory:", allow_in_memory=True)
    init_auth_db(engine)
    session = sessionmaker(bind=engine)()
    seed_auth_data(session, seed_json=oidc_test_identity["seed_json"])

    saved_auth_mode = getattr(server.app.state, "auth_mode", None)
    saved_oidc_client = getattr(server.app.state, "oidc_client", None)
    saved_auth_service = getattr(server.app.state, "auth_service", None)
    saved_seed_auth_data = server.seed_auth_data

    monkeypatch.setenv("AUTH_MODE", "oidc")
    monkeypatch.setenv("UI_URL", "http://ui.local/")
    monkeypatch.setenv("OIDC_CLIENT_ID", "kge-local")
    monkeypatch.setenv("OIDC_CLIENT_SECRET", "")
    monkeypatch.setenv(
        "OIDC_REDIRECT_URI", "http://localhost:28110/api/auth/callback"
    )

    server.app.state.auth_mode = "oidc"
    server.seed_auth_data = lambda _session: None
    server.app.state.oidc_client = OIDCClient(
        client_id="kge-local",
        client_secret="",
        discovery_url=keycloak_container["discovery_url"],
        redirect_uri="http://localhost:28110/api/auth/callback",
    )
    server.app.state.auth_service = AuthService(session, jwt_secret=server.JWT_SECRET)

    try:
        with TestClient(server.app) as client:
            yield (
                client,
                session,
                keycloak_container["base_url"],
                keycloak_container["issuer"],
            )
    finally:
        session.close()
        engine.dispose()
        server.app.state.auth_mode = saved_auth_mode
        server.app.state.oidc_client = saved_oidc_client
        server.app.state.auth_service = saved_auth_service
        server.seed_auth_data = saved_seed_auth_data


def test_oidc_keycloak_container_e2e(oidc_test_client, oidc_test_identity):
    client, auth_db, keycloak_base_url, issuer = oidc_test_client

    login_resp = client.get("/api/auth/login", follow_redirects=False)
    assert login_resp.status_code == 307
    auth_url = login_resp.headers["location"]
    assert auth_url.startswith(keycloak_base_url)

    state_cookie = client.cookies.get("auth_state")
    verifier_cookie = client.cookies.get("auth_pkce_verifier")
    nonce_cookie = client.cookies.get("auth_nonce")
    assert state_cookie
    assert verifier_cookie
    assert nonce_cookie

    with httpx.Client(follow_redirects=False, timeout=20.0) as browser:
        auth_page = browser.get(auth_url)
        assert auth_page.status_code == 200
        login_action = extract_login_action(auth_page.text, str(auth_page.url))

        submit_resp = browser.post(
            login_action,
            data={
                "username": oidc_test_identity["username"],
                "password": oidc_test_identity["password"],
                "credentialId": "",
            },
        )

    assert submit_resp.status_code in {302, 303}
    callback_url = submit_resp.headers["location"]
    parsed = urlparse(callback_url)
    assert parsed.path == "/api/auth/callback"
    params = parse_qs(parsed.query)
    assert "code" in params
    assert params.get("state", [None])[0] == state_cookie

    callback_resp = client.get(
        f"{parsed.path}?{parsed.query}",
        follow_redirects=False,
    )
    assert callback_resp.status_code == 307, callback_resp.text
    ui_location = callback_resp.headers["location"]
    assert ui_location.startswith("http://ui.local/")
    assert "?token=" in ui_location

    token = parse_qs(urlparse(ui_location).query)["token"][0]
    me_resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    me = me_resp.json()
    assert me["email"] == oidc_test_identity["email"]
    assert me["role"] == oidc_test_identity["role"]

    auth_db.expire_all()
    user = auth_db.query(User).filter(User.email == oidc_test_identity["email"]).first()
    assert user is not None

    identity = (
        auth_db.query(ExternalIdentity)
        .filter(ExternalIdentity.user_id == user.user_id)
        .filter(ExternalIdentity.issuer == issuer)
        .first()
    )
    assert identity is not None
    assert identity.email == oidc_test_identity["email"]
