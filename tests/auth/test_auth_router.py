import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from graph_knowledge_engine.server_mcp_with_admin import app, JWT_SECRET, JWT_ALG
from graph_knowledge_engine.server.auth.db import init_auth_db, get_session
from graph_knowledge_engine.server.auth.service import AuthService
from jose import jwt
import os

@pytest.fixture(scope="module")
def client():
    # Setup in-memory auth DB for testing
    engine = create_engine("sqlite:///:memory:")
    init_auth_db(engine)
    
    # Manually initialize app state for tests
    app.state.auth_service = AuthService(get_session(), jwt_secret=JWT_SECRET)
    
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

def test_login_redirect(client):
    # Mock OIDC client discovery
    from unittest.mock import AsyncMock
    app.state.oidc_client.get_auth_url = AsyncMock(return_value="https://example.com/auth")
    
    response = client.get("/api/auth/login", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "https://example.com/auth"
    # Check if cookies are set
    assert "auth_state" in response.cookies
    assert "auth_pkce_verifier" in response.cookies

def test_logout(client):
    response = client.post("/api/auth/logout")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
