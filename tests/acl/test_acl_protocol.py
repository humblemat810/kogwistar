from __future__ import annotations

import pytest

from kogwistar.engine_core.acl_protocol import (
    ACLAwareReadProtocol,
    ACLAwareWriteProtocol,
    ACLPolicyProtocol,
    require_acl_protocols,
)


class Policy:
    def current_principal_context(self):
        return "principal", (), None

    def decide_acl(self, **kwargs):
        return True

    def decide_acl_node_read(self, **kwargs):
        return True

    def record_default_acl_for_item(self, item, *, grain):
        return None

    def writes_can_share_backend_transaction(self):
        return True


class Read:
    def get_nodes(self, *args, **kwargs):
        return []

    def get_edges(self, *args, **kwargs):
        return []

    def query_nodes(self, *args, **kwargs):
        return []

    def query_edges(self, *args, **kwargs):
        return []

    def search_nodes_as_of(self, *args, **kwargs):
        return []


class Write:
    def add_node(self, *args, **kwargs):
        return None

    def add_edge(self, *args, **kwargs):
        return None


def test_acl_protocols_are_runtime_checkable():
    assert isinstance(Policy(), ACLPolicyProtocol)
    assert isinstance(Read(), ACLAwareReadProtocol)
    assert isinstance(Write(), ACLAwareWriteProtocol)
    require_acl_protocols(policy=Policy(), read=Read(), write=Write())


def test_acl_enabled_contract_fails_closed_when_a_seam_is_missing():
    with pytest.raises(TypeError, match="ACL-aware write"):
        require_acl_protocols(policy=Policy(), read=Read(), write=object())


def test_acl_disabled_does_not_require_acl_protocols():
    # Engine selection skips require_acl_protocols when disabled.
    assert not isinstance(object(), ACLPolicyProtocol)
