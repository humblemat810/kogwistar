from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

if TYPE_CHECKING:
    from .models import Edge, Node

# Engine names remain broad; graph entity kinds are structural node/edge tokens.
EngineType = str
GraphEntityKind = Literal["node", "edge"]
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


class PersistedGraphResult(TypedDict):
    """IDs and counts returned when a graph extraction is persisted."""

    node_ids: list[str]
    edge_ids: list[str]
    nodes_added: int
    edges_added: int


class PersistedDocumentGraphResult(PersistedGraphResult):
    """Graph persistence result associated with a document identity."""

    document_id: str


class ToposortIngestResult(TypedDict):
    """Legacy topological-ingest result; ID fields remain counts for compatibility."""

    document_id: str
    node_ids: int
    edge_ids: int
    nodes_added: int
    edges_added: int


class IngestTextResult(ToposortIngestResult):
    """Topological-ingest result with the provider's raw response."""

    raw: object


class RawPageIngestResult(TypedDict):
    page: int | None
    raw: object | None


class AddPageResult(TypedDict):
    document_id: str
    nodes_added: int
    edges_added: int
    raw_by_page: NotRequired[list[RawPageIngestResult]]
