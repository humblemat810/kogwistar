from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from .models import Edge, Node

# Keep this broad in core; domain-specific kind tokens live outside engine_core.
EngineType = str
ExtractionSchemaMode = Literal[
    "auto", "full", "lean", "flattened_lean", "flattened_full"
]
ResolvedExtractionSchemaMode = Literal[
    "full", "lean", "flattened_lean", "flattened_full"
]
OffsetMismatchPolicy = Literal["strict", "exact", "exact_fuzzy"]


class OffsetRepairScorer(Protocol):
    """Score a candidate repaired offset against the original text."""

    def __call__(self, original: str, candidate: str, /) -> float: ...


class NodePreAddHook(Protocol):
    """Inspect a node before the engine submits it to storage."""

    def __call__(self, node: Node, /) -> None: ...


class EdgePreAddHook(Protocol):
    """Accept or reject an edge before the engine submits it to storage."""

    def __call__(self, edge: Edge, /) -> bool: ...


class ToolCallIdFactory(Protocol):
    """Create a stable identifier for an engine tool invocation."""

    def __call__(self, *parts: str) -> str: ...
