from __future__ import annotations

import pytest

from kogwistar.server.auth_middleware import (
    can_access_security_scope,
    describe_storage_security_mapping,
    claims_ctx,
    get_execution_namespace,
    get_current_capabilities,
    get_security_scope,
    get_security_scope_parts,
    get_storage_namespace,
    require_namespace,
    require_capability,
    require_security_scope,
    require_security_scope_access,
)
from kogwistar.conversation.policy import can_access_memory_metadata


def test_scope_helpers_default_to_claim_namespace():
    token = claims_ctx.set({"ns": "conversation"})
    try:
        assert get_storage_namespace() == "conversation"
        assert get_execution_namespace() == "conversation"
        assert get_security_scope() == "conversation"
    finally:
        claims_ctx.reset(token)


def test_scope_helpers_prefer_explicit_scope_claims():
    token = claims_ctx.set(
        {
            "ns": ["docs", "conversation"],
            "storage_ns": "tenant-a-store",
            "execution_ns": "tenant-a-exec",
            "security_scope": "tenant-a",
        }
    )
    try:
        assert get_storage_namespace() == "tenant-a-store"
        assert get_execution_namespace() == "tenant-a-exec"
        assert get_security_scope() == "tenant-a"
    finally:
        claims_ctx.reset(token)


def test_scope_requires_match_or_denies():
    token = claims_ctx.set({"ns": "conversation", "security_scope": "tenant-a"})
    try:
        assert require_namespace("conversation") == "conversation"
        assert require_security_scope({"tenant-a", "tenant-b"}) == "tenant-a"
        with pytest.raises(Exception):
            require_namespace("workflow")
        with pytest.raises(Exception):
            require_security_scope("tenant-b")
    finally:
        claims_ctx.reset(token)


def test_capability_helpers_use_claim_capabilities():
    token = claims_ctx.set({"capabilities": ["workflow.design.inspect", "workflow.run.read"]})
    try:
        assert get_current_capabilities() == {
            "workflow.design.inspect",
            "workflow.run.read",
        }
        assert require_capability("workflow.design.inspect") == "workflow.design.inspect"
        assert require_capability({"workflow.run.read", "workflow.run.write"}) == "workflow.run.read"
        with pytest.raises(Exception):
            require_capability("workflow.admin")
    finally:
        claims_ctx.reset(token)


def test_security_scope_access_requires_match_or_explicit_share():
    token = claims_ctx.set({"ns": "conversation", "security_scope": "tenant-a"})
    try:
        assert can_access_security_scope("tenant-a") is True
        assert can_access_security_scope("tenant-b") is False
        assert can_access_security_scope("tenant-b", shared=True) is True
        assert require_security_scope_access("tenant-a") == "tenant-a"
        with pytest.raises(Exception):
            require_security_scope_access("tenant-b")
        assert (
            require_security_scope_access(
                "tenant-b", shared=True, action="read message from"
            )
            == "tenant-a"
        )
    finally:
        claims_ctx.reset(token)


def test_memory_visibility_requires_agent_match_for_private_memory():
    private_md = {
        "visibility": "private",
        "owner_agent_id": "agent-a",
        "owner_security_scope": "tenant-a",
        "security_scope": "tenant-a",
    }
    shared_md = {
        "visibility": "shared",
        "owner_agent_id": "agent-a",
        "owner_security_scope": "tenant-a",
        "security_scope": "tenant-a",
        "shared_with_agents": ["agent-b"],
    }
    assert (
        can_access_memory_metadata(
            private_md, current_security_scope="tenant-a", current_agent_id="agent-a"
        )
        is True
    )
    assert (
        can_access_memory_metadata(
            private_md, current_security_scope="tenant-a", current_agent_id="agent-b"
        )
        is False
    )
    assert (
        can_access_memory_metadata(
            shared_md, current_security_scope="tenant-a", current_agent_id="agent-b"
        )
        is True
    )


def test_security_scope_parts_support_tenant_workspace_project():
    token = claims_ctx.set(
        {
            "storage_ns": "store-a",
            "execution_ns": "exec-a",
            "tenant": "tenant-a",
            "workspace": "workspace-a",
            "project": "project-a",
        }
    )
    try:
        parts = get_security_scope_parts()
        assert parts == {
            "tenant": "tenant-a",
            "workspace": "workspace-a",
            "project": "project-a",
            "path": "tenant-a/workspace-a/project-a",
        }
        mapping = describe_storage_security_mapping()
        assert mapping["storage_namespace"] == "store-a"
        assert mapping["execution_namespace"] == "exec-a"
        assert mapping["security_scope"] == "tenant-a"
        assert mapping["security_scope_path"] == "tenant-a/workspace-a/project-a"
        assert mapping["tenant"] == "tenant-a"
        assert mapping["workspace"] == "workspace-a"
        assert mapping["project"] == "project-a"
    finally:
        claims_ctx.reset(token)
