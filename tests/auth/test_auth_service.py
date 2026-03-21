import pytest

pytestmark = pytest.mark.ci
from graph_knowledge_engine.server.auth.db import (
    create_auth_engine,
    init_auth_db,
    get_session,
)
from graph_knowledge_engine.server.auth.service import AuthService


@pytest.fixture
def auth_service():
    engine = create_auth_engine("sqlite:///:memory:")
    init_auth_db(engine)
    session = get_session()
    service = AuthService(session, jwt_secret="test-secret")
    return service


def test_resolve_user_new(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub",
        email="test@example.com",
        display_name="Test User",
    )
    assert user_id is not None
    user = auth_service.get_user(user_id)
    assert user["email"] == "test@example.com"
    assert user["display_name"] == "Test User"


def test_resolve_user_existing_identity(auth_service):
    user_id1 = auth_service.resolve_user_from_external(
        issuer="test-issuer", subject="test-sub", email="test@example.com"
    )
    user_id2 = auth_service.resolve_user_from_external(
        issuer="test-issuer", subject="test-sub", email="test@example.com"
    )
    assert user_id1 == user_id2


def test_mint_token(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer", subject="test-sub", email="test@example.com"
    )
    token = auth_service.mint_token(user_id, role="rw", ns="workflow")
    assert token is not None

    from jose import jwt

    claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert claims["user_id"] == user_id
    assert claims["role"] == "rw"
    assert claims["ns"] == "workflow"


def test_mint_token_uses_persisted_defaults_when_overrides_omitted(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer",
        subject="test-sub-defaults",
        email="defaults@example.com",
        default_role="rw",
        default_ns="docs,workflow",
    )

    token = auth_service.mint_token(user_id)

    from jose import jwt

    claims = jwt.decode(token, "test-secret", algorithms=["HS256"])
    assert claims["role"] == "rw"
    assert claims["ns"] == ["docs", "workflow"]


def test_mint_token_unknown_user_raises(auth_service):
    with pytest.raises(ValueError, match="not found"):
        auth_service.mint_token("missing-user-id")


def test_workflow_acl(auth_service):
    user_id = auth_service.resolve_user_from_external(
        issuer="test-issuer", subject="test-sub", email="test@example.com"
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
