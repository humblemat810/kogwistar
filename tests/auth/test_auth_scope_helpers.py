from __future__ import annotations

import pytest

from kogwistar.server.auth_middleware import (
    claims_ctx,
    get_execution_namespace,
    get_current_capabilities,
    get_security_scope,
    get_storage_namespace,
    require_namespace,
    require_capability,
    require_security_scope,
)


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
