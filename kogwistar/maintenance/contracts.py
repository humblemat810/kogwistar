"""Reusable callback contracts for maintenance write boundaries."""

from __future__ import annotations

from typing import Protocol, TypeVar

from kogwistar.engine_core.models import Node

TWriteItem = TypeVar("TWriteItem", contravariant=True)


class BeforeWrite(Protocol[TWriteItem]):
    """Authorize or observe one item immediately before it is written."""

    def __call__(self, item: TWriteItem, /) -> None: ...


class ArtifactNodeBuilder(Protocol):
    """Build one replacement node from the currently matching nodes."""

    def __call__(self, existing: list[Node], created_at_ms: int, /) -> Node: ...


class GroupKeyForNode(Protocol):
    """Return the stable grouping key for a source node."""

    def __call__(self, node: Node, /) -> str: ...


class GroupedArtifactNodeBuilder(Protocol):
    """Build one node from a source group and existing replacement nodes."""

    def __call__(
        self,
        group_key: str,
        source_nodes: list[Node],
        existing: list[Node],
        created_at_ms: int,
        /,
    ) -> Node: ...


class MatchWhereForGroup(Protocol):
    """Build the read predicate for one grouped replacement."""

    def __call__(self, group_key: str, /) -> dict[str, object]: ...


__all__ = [
    "ArtifactNodeBuilder",
    "BeforeWrite",
    "GroupKeyForNode",
    "GroupedArtifactNodeBuilder",
    "MatchWhereForGroup",
]
