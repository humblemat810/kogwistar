import pytest
from fastapi.testclient import TestClient
from graph_knowledge_engine.server_mcp_with_admin import app, JWT_SECRET, JWT_ALG
from graph_knowledge_engine.server.auth.db import (
    create_auth_engine,
    init_auth_db,
    get_session,
)
from graph_knowledge_engine.server.auth.service import AuthService
from jose import jwt
from urllib.parse import urlparse, parse_qs
from unittest.mock import AsyncMock


@pytest.fixture(scope="module")
def client():
    # Setup in-memory auth DB for testing
    engine = create_auth_engine("sqlite:///:memory:")
    init_auth_db(engine)
    mock_oidc = AsyncMock()

    # Manually initialize app state for tests
    app.state.auth_service = AuthService(get_session(), jwt_secret=JWT_SECRET)
    app.state.oidc_clients = {"test": mock_oidc}
    app.state.oidc_default_provider = "test"
    app.state.oidc_provider_configs = {}

    with TestClient(app) as c:
        yield c


def test_auth_me_unauthorized(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_auth_me_authorized(client):
    # Create a user and a token
    auth_service = app.state.auth_service
    user_id = auth_service.resolve_user_from_external("iss", "sub", "test@example.com")
    token = auth_service.mint_token(user_id)

    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == user_id
    assert data["email"] == "test@example.com"
    assert data["role"] == "ro"
    assert data["ns"] == "docs"


def test_login_dev_mode_redirects_with_token_and_requested_redirect(
    client, monkeypatch
):
    client.cookies.clear()
    app.state.auth_mode = "dev"
    monkeypatch.setenv("DEV_AUTH_EMAIL", "dev-user@example.com")
    monkeypatch.setenv("DEV_AUTH_ROLE", "rw")
    monkeypatch.setenv("DEV_AUTH_NS", "workflow,conversation")

    response = client.get(
        "/api/auth/login?redirect_uri=https://ui.example.local/welcome",
        follow_redirects=False,
    )

    assert response.status_code == 307
    location = response.headers["location"]
    parsed = urlparse(location)
    assert parsed.scheme == "https"
    assert parsed.netloc == "ui.example.local"
    assert parsed.path == "/welcome"
    params = parse_qs(parsed.query)
    assert "token" in params

    token = params["token"][0]
    claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    assert claims["sub"] == "dev-user@example.com"
    assert claims["role"] == "rw"
    assert set(claims["ns"]) == {"workflow", "conversation"}


def test_login_rejects_redirect_override_outside_dev(client):
    app.state.auth_mode = "oidc"
    response = client.get(
        "/api/auth/login?redirect_uri=https://ui.example.local/welcome",
        follow_redirects=False,
    )
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "redirect_uri override is only allowed when AUTH_MODE=dev"
    )


def test_login_redirect(client):
    app.state.auth_mode = "oidc"
    # Mock OIDC client discovery
    app.state.oidc_clients["test"].get_auth_url = AsyncMock(
        return_value="https://example.com/auth"
    )

    response = client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://example.com/auth"
    # Check if cookies are set
    assert "auth_state" in response.cookies
    assert "auth_pkce_verifier" in response.cookies
    assert "auth_nonce" in response.cookies
    assert response.cookies["auth_provider"] == "test"


def test_logout(client):
    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_callback_success_redirects_with_token(client, monkeypatch):
    client.cookies.clear()
    app.state.auth_mode = "oidc"
    monkeypatch.setenv("UI_URL", "https://ui.example.local/")
    app.state.oidc_clients["test"].exchange_code = AsyncMock(
        return_value={"access_token": "access", "id_token": "id-token"}
    )
    app.state.oidc_clients["test"].validate_id_token = AsyncMock(
        return_value={
            "sub": "sub-123",
            "email": "user@example.com",
            "name": "Test User",
            "iss": "https://issuer.example.local",
            "aud": "kge-local",
            "nonce": "nonce-123",
        }
    )
    app.state.oidc_clients["test"].get_userinfo = AsyncMock(
        return_value={
            "sub": "sub-123",
            "email": "user@example.com",
            "name": "Test User",
        }
    )

    client.cookies.set("auth_state", "state-123")
    client.cookies.set("auth_pkce_verifier", "verifier-123")
    client.cookies.set("auth_nonce", "nonce-123")
    client.cookies.set("auth_provider", "test")
    response = client.get(
        "/api/auth/callback?code=abc&state=state-123", follow_redirects=False
    )

    assert response.status_code == 307
    location = response.headers["location"]
    parsed = urlparse(location)
    assert parsed.scheme == "https"
    assert parsed.netloc == "ui.example.local"
    params = parse_qs(parsed.query)
    assert "token" in params
    token = params["token"][0]
    claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    assert claims["sub"] == "user@example.com"
    app.state.oidc_clients["test"].validate_id_token.assert_awaited_once_with(
        "id-token", nonce="nonce-123"
    )


def test_callback_rejects_invalid_state(client):
    client.cookies.clear()
    app.state.auth_mode = "oidc"
    client.cookies.set("auth_state", "state-abc")
    client.cookies.set("auth_pkce_verifier", "verifier-abc")
    response = client.get(
        "/api/auth/callback?code=abc&state=state-wrong", follow_redirects=False
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid state"


def test_callback_rejects_missing_verifier(client):
    client.cookies.clear()
    app.state.auth_mode = "oidc"
    client.cookies.set("auth_state", "state-abc")
    response = client.get(
        "/api/auth/callback?code=abc&state=state-abc", follow_redirects=False
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing PKCE verifier"


def test_callback_rejects_missing_nonce(client):
    client.cookies.clear()
    app.state.auth_mode = "oidc"
    client.cookies.set("auth_state", "state-abc")
    client.cookies.set("auth_pkce_verifier", "verifier-abc")
    response = client.get(
        "/api/auth/callback?code=abc&state=state-abc", follow_redirects=False
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing nonce"


def test_callback_rejects_userinfo_subject_mismatch(client):
    client.cookies.clear()
    app.state.auth_mode = "oidc"
    app.state.oidc_clients["test"].exchange_code = AsyncMock(
        return_value={"access_token": "access", "id_token": "id-token"}
    )
    app.state.oidc_clients["test"].validate_id_token = AsyncMock(
        return_value={
            "sub": "sub-123",
            "email": "user@example.com",
            "name": "Test User",
            "iss": "https://issuer.example.local",
            "aud": "kge-local",
            "nonce": "nonce-abc",
        }
    )
    app.state.oidc_clients["test"].get_userinfo = AsyncMock(
        return_value={
            "sub": "sub-wrong",
            "email": "user@example.com",
            "name": "Test User",
        }
    )

    client.cookies.set("auth_state", "state-abc")
    client.cookies.set("auth_pkce_verifier", "verifier-abc")
    client.cookies.set("auth_nonce", "nonce-abc")
    client.cookies.set("auth_provider", "test")
    response = client.get(
        "/api/auth/callback?code=abc&state=state-abc", follow_redirects=False
    )
    assert response.status_code == 401
    assert (
        response.json()["detail"]
        == "userinfo subject does not match validated id_token subject"
    )
