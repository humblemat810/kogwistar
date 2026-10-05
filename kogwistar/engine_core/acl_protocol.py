"""Conditional ACL integration contracts for the engine boundary.

ACL policy is owned by the engine, not duplicated by storage backends. These
protocols validate that the engine has installed policy and guarded read/write
surfaces when ACL is enabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .models import Edge, Node
from .vector_search import VectorSearchHit


@runtime_checkable
class ACLPolicyProtocol(Protocol):
    """Minimum engine-owned policy surface required when ACL is enabled."""

    def current_principal_context(self) -> object: ...
    def decide_acl(self, **kwargs: Any) -> object: ...
    def decide_acl_node_read(self, **kwargs: Any) -> object: ...
    def record_default_acl_for_item(self, item: object, *, grain: str) -> object: ...
    def writes_can_share_backend_transaction(self) -> bool: ...


@runtime_checkable
class ACLAwareReadProtocol(Protocol):
    """Read operations that must remain policy guarded when ACL is enabled."""

    def get_nodes(self, *args: Any, **kwargs: Any) -> Sequence[Node]: ...
    def get_edges(self, *args: Any, **kwargs: Any) -> Sequence[Edge]: ...
    def query_nodes(
        self, *args: Any, **kwargs: Any
    ) -> Sequence[Sequence[Node]]: ...
    def query_edges(
        self, *args: Any, **kwargs: Any
    ) -> Sequence[Sequence[Edge]]: ...
    def search_nodes_as_of(self, *args: Any, **kwargs: Any) -> Sequence[Node]: ...
    def search_nodes_as_of_scored(
        self, *args: Any, **kwargs: Any
    ) -> Sequence[VectorSearchHit[Node]]: ...


@runtime_checkable
class ACLAwareWriteProtocol(Protocol):
    """Mutations that must create/check ACL state when enabled."""

    def add_node(self, *args: Any, **kwargs: Any) -> None: ...
    def add_edge(self, *args: Any, **kwargs: Any) -> None: ...


def require_acl_protocols(*, policy: object, read: object, write: object) -> None:
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
