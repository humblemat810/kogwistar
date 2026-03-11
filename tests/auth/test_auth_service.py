import pytest
from sqlalchemy import create_engine
from graph_knowledge_engine.server.auth.db import init_auth_db, get_session
from graph_knowledge_engine.server.auth.service import AuthService
from graph_knowledge_engine.server.auth.models import Base

@pytest.fixture
def auth_service():
    engine = create_engine("sqlite:///:memory:")
    init_auth_db(engine)
    session = get_session()
    service = AuthService(session, jwt_secret="test-secret")
    return service

def test_resolve_user_new(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub",
        email="test@example.com",
        display_name="Test User"
    )
    assert user_id is not None
    user = auth_service.get_user(user_id)
    assert user["email"] == "test@example.com"
    assert user["display_name"] == "Test User"

def test_resolve_user_existing_identity(auth_service):
    user_id1 = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub",
        email="test@example.com"
    )
    user_id2 = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub",
        email="test@example.com"
    )
    assert user_id1 == user_id2

def test_mint_token(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub",
        email="test@example.com"
    )
    token = auth_service.mint_token(user_id, role="rw", ns="workflow")
    assert token is not None
    
    from jose import jwt
    claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert claims["user_id"] == user_id
    assert claims["role"] == "rw"
    assert claims["ns"] == "workflow"

def test_workflow_acl(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub",
        email="test@example.com"
    )
    
    # Initially no access
    assert auth_service.check_workflow_access("wf1", user_id, "ro") is False
    
    # Grant access
    auth_service.repo.set_workflow_acl("wf1", user_id, "ro")
    assert auth_service.check_workflow_access("wf1", user_id, "ro") is True
    assert auth_service.check_workflow_access("wf1", user_id, "rw") is False
    
    # Upgrade access
    auth_service.repo.set_workflow_acl("wf1", user_id, "rw")
    assert auth_service.check_workflow_access("wf1", user_id, "rw") is True
