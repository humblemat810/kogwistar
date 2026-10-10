"""Conditional ACL integration contracts for the engine boundary.

ACL policy is owned by the engine, not duplicated by storage backends. These
protocols validate that the engine has installed policy and guarded read/write
surfaces when ACL is enabled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..acl.graph import ACLDecision, ACLNodeReadDecision
from .models import Edge, Node
from .vector_search import VectorSearchHit


@runtime_checkable
class ACLPolicyProtocol(Protocol):
    """Minimum engine-owned policy surface required when ACL is enabled."""

    def current_principal_context(self) -> tuple[str, tuple[str, ...], str | None]: ...

    def decide_acl(
        self,
        *,
        grain: str | None = None,
        truth_graph: str,
        entity_id: str,
        target_item_id: str | None = None,
        principal_id: str,
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
    ) -> ACLDecision: ...

    def decide_acl_node_read(
        self,
        *,
        item_grain: str,
        truth_graph: str,
        entity_id: str,
        target_item_ids: Sequence[str],
        principal_id: str,
        grounding_item_ids: Sequence[str] = (),
        principal_groups: Sequence[str] = (),
        security_scope: str | None = None,
    ) -> ACLNodeReadDecision: ...

    def record_default_acl_for_item(self, item: Node | Edge, *, grain: str) -> None: ...
    def writes_can_share_backend_transaction(self) -> bool: ...


@runtime_checkable
class ACLAwareReadProtocol(Protocol):
    """Read operations that must remain policy guarded when ACL is enabled."""

    def get_nodes(self, *args: object, **kwargs: object) -> Sequence[Node]: ...
    def get_edges(self, *args: object, **kwargs: object) -> Sequence[Edge]: ...
    def query_nodes(
        self, *args: object, **kwargs: object
    ) -> Sequence[Sequence[Node]]: ...
    def query_edges(
        self, *args: object, **kwargs: object
    ) -> Sequence[Sequence[Edge]]: ...
    def search_nodes_as_of(self, *args: object, **kwargs: object) -> Sequence[Node]: ...
    def search_nodes_as_of_scored(
        self, *args: object, **kwargs: object
    ) -> Sequence[VectorSearchHit[Node]]: ...


@runtime_checkable
class ACLAwareWriteProtocol(Protocol):
    """Mutations that must create/check ACL state when enabled."""

    def add_node(self, *args: object, **kwargs: object) -> None: ...
    def add_edge(self, *args: object, **kwargs: object) -> None: ...


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
    "ACLAwareReadProtocol",
    "ACLAwareWriteProtocol",
    "ACLPolicyProtocol",
    "require_acl_protocols",
]
