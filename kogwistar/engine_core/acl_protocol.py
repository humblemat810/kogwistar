"""Conditional ACL integration contracts for the engine boundary.

ACL policy is owned by the engine, not duplicated by storage backends. These
protocols validate that the engine has installed policy and guarded read/write
surfaces when ACL is enabled.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ACLPolicyProtocol(Protocol):
    """Minimum engine-owned policy surface required when ACL is enabled."""

    def current_principal_context(self) -> Any: ...
    def decide_acl(self, **kwargs: Any) -> Any: ...
    def decide_acl_node_read(self, **kwargs: Any) -> Any: ...
    def record_default_acl_for_item(self, item: Any, *, grain: str) -> Any: ...
    def writes_can_share_backend_transaction(self) -> bool: ...


@runtime_checkable
class ACLAwareReadProtocol(Protocol):
    """Read operations that must remain policy guarded when ACL is enabled."""

    def get_nodes(self, *args: Any, **kwargs: Any) -> Any: ...
    def get_edges(self, *args: Any, **kwargs: Any) -> Any: ...
    def query_nodes(self, *args: Any, **kwargs: Any) -> Any: ...
    def query_edges(self, *args: Any, **kwargs: Any) -> Any: ...
    def search_nodes_as_of(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ACLAwareWriteProtocol(Protocol):
    """Mutations that must create/check ACL state when enabled."""

    def add_node(self, *args: Any, **kwargs: Any) -> Any: ...
    def add_edge(self, *args: Any, **kwargs: Any) -> Any: ...


def require_acl_protocols(*, policy: Any, read: Any, write: Any) -> None:
    """Fail closed if ACL was requested without all required engine seams."""

    requirements = (
        ("ACL policy", policy, ACLPolicyProtocol),
        ("ACL-aware read", read, ACLAwareReadProtocol),
        ("ACL-aware write", write, ACLAwareWriteProtocol),
    )
    missing = [
        label for label, value, protocol in requirements if not isinstance(value, protocol)
    ]
    if missing:
        raise TypeError(
            "acl_enabled=True requires implemented ACL protocols: "
            + ", ".join(missing)
        )


__all__ = [
    "ACLPolicyProtocol",
    "ACLAwareReadProtocol",
    "ACLAwareWriteProtocol",
    "require_acl_protocols",
]
