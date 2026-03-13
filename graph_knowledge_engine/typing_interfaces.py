# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
    TYPE_CHECKING,
)

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - py<3.10 compatibility
    from typing_extensions import TypeAlias

from .engine_core.models import (
    AdjudicationTarget as GraphAdjudicationTarget,
    AdjudicationVerdict,
    Document as EngineDoc,
    Edge as GraphEdge,
    Node as GraphNode,
)
from .engine_core.storage_backend import StorageBackend

# -------------------------
# Collection / Vector store
# -------------------------

try:
    from chromadb.api.types import (
        Embedding,
        PyEmbedding,
        Document as ChromaDocument,
        Image,
        URI,
        ID,
        Include,
        QueryResult,
    )
    from chromadb.base_types import Where, WhereDocument
    from chromadb.api.types import IDs, OneOrMany, GetResult, Metadata
except Exception:  # pragma: no cover - optional dependency
    Embedding = Any  # type: ignore
    PyEmbedding = Any  # type: ignore
    ChromaDocument = str  # type: ignore
    Image = Any  # type: ignore
    URI = str  # type: ignore
    ID = str  # type: ignore
    Include = list[str]  # type: ignore
    QueryResult = Dict[str, Any]  # type: ignore
    Where = Dict[str, Any]  # type: ignore
    WhereDocument = Dict[str, Any]  # type: ignore
    IDs = List[str]  # type: ignore
    OneOrMany = Any  # type: ignore
    GetResult = Dict[str, Any]  # type: ignore
    Metadata = Dict[str, Any]  # type: ignore


class CollectionLike(Protocol):
    def add(
        self,
        ids: OneOrMany[ID],
        embeddings: Any = None,
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[ChromaDocument] | None = None,
        images: Any = None,
        uris: OneOrMany[URI] | None = None,
    ) -> None: ...

    def update(
        self,
        ids: OneOrMany[ID],
        embeddings: Any = None,
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[ChromaDocument] | None = None,
        images: Any = None,
        uris: OneOrMany[URI] | None = None,
    ) -> None: ...

    def get(
        self,
        ids: OneOrMany[ID] | None = None,
        where: Where | None = None,
        limit: int | None = None,
        offset: int | None = None,
        where_document: WhereDocument | None = None,
        include: Include = ["metadatas", "documents"],
    ) -> GetResult: ...

    def delete(
        self,
        ids: IDs | None = None,
        where: Where | None = None,
        where_document: WhereDocument | None = None,
    ) -> None: ...

    def query(
        self,
        query_embeddings: OneOrMany[Embedding] | OneOrMany[PyEmbedding] | None = None,
        query_texts: OneOrMany[ChromaDocument] | None = None,
        query_images: OneOrMany[Image] | None = None,
        query_uris: OneOrMany[URI] | None = None,
        ids: OneOrMany[ID] | None = None,
        n_results: int = 10,
        where: Where | None = None,
        where_document: WhereDocument | None = None,
        include: Include = ["metadatas", "documents", "distances"],
    ) -> QueryResult: ...


if TYPE_CHECKING:
    from .llm_tasks import LLMTaskSet

# -------------------------
# Graph objects (structural)
# -------------------------


@runtime_checkable
class NodeLike(Protocol):
    id: str
    label: str
    type: str
    summary: str
    domain_id: Optional[str]
    canonical_entity_id: Optional[str]
    properties: Optional[Dict[str, Any]]
    mentions: Optional[List[Any]]
    embedding: Optional[List[float]]
    doc_id: Optional[str]

    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self) -> str: ...


@runtime_checkable
class EdgeLike(NodeLike, Protocol):
    relation: str
    source_ids: Optional[List[str]]
    target_ids: Optional[List[str]]
    source_edge_ids: Optional[List[str]]
    target_edge_ids: Optional[List[str]]


AdjudicationTarget: TypeAlias = Union[NodeLike, EdgeLike]

# -------------------------
# Shared engine surface
# -------------------------


class EmbeddingFunctionLike(Protocol):
    def __call__(self, documents_or_texts: list[str]) -> Any: ...


class ReadLike(Protocol):
    def get_document(self, doc_id: str) -> EngineDoc: ...

    def node_ids_by_doc(
        self,
        doc_id: str,
        insertion_method: str | None = None,
    ) -> list[str]: ...

    def edge_ids_by_doc(
        self,
        doc_id: str,
        insertion_method: str | None = None,
    ) -> list[str]: ...

    def extract_reference_contexts(
        self,
        node_or_id: GraphNode | GraphEdge | str,
        *,
        window_chars: int = 120,
        max_contexts: int | None = None,
        prefer_label_fallback: bool = True,
    ) -> list[dict[str, Any]]: ...


class WriteLike(Protocol):
    def add_node(self, node: GraphNode, doc_id: str | None = None) -> Any: ...
    def add_edge(self, edge: GraphEdge, doc_id: str | None = None) -> Any: ...

    def node_doc_and_meta(self, node: GraphNode) -> tuple[str, dict[str, Any]]: ...
    def edge_doc_and_meta(self, edge: GraphEdge) -> tuple[str, dict[str, Any]]: ...

    def strip_none(self, data: dict[str, Any]) -> dict[str, Any]: ...
    def json_or_none(self, value: Any) -> str | None: ...

    def index_node_docs(self, node: GraphNode) -> list[str]: ...
    def index_node_refs(self, node: GraphNode) -> list[str]: ...
    def index_edge_refs(self, edge: GraphEdge) -> list[str]: ...


class ExtractLike(Protocol):
    def fetch_document_text(self, document_id: str) -> str: ...


class EmbedLike(Protocol):
    def iterative_defensive_emb(self, emb_text0: str) -> Any: ...


class AdjudicateLike(Protocol):
    def target_from_node(self, n: GraphNode) -> GraphAdjudicationTarget: ...
    def target_from_edge(self, e: GraphEdge) -> GraphAdjudicationTarget: ...
    def fetch_target(self, t: GraphAdjudicationTarget) -> GraphNode | GraphEdge: ...

    def split_endpoints(
        self,
        src_ids: list[str] | None,
        tgt_ids: list[str] | None,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]: ...


class EngineLike(Protocol):
    """Public read-oriented engine contract used by lightweight helpers."""

    backend: StorageBackend
    read: ReadLike


class StrategyEngineLike(EngineLike, Protocol):
    """Public engine contract used by strategy modules."""

    read: ReadLike
    write: WriteLike
    extract: ExtractLike
    embed: EmbedLike
    adjudicate: AdjudicateLike

    llm_tasks: "LLMTaskSet"
    allow_cross_kind_adjudication: bool
    cross_kind_strategy: str

    def commit_merge(
        self,
        left: GraphNode,
        right: GraphNode,
        verdict: AdjudicationVerdict,
        method: str = "unspecified",
    ) -> str: ...

    def commit_merge_target(
        self,
        left: GraphAdjudicationTarget,
        right: GraphAdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str: ...
