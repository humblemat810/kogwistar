# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
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

if TYPE_CHECKING:
    from .engine_core.vector_search import VectorSearchHit

# -------------------------
# Collection / Vector store
# -------------------------

# Keep the public collection contract dependency-free.  Chroma is optional at
# runtime, and these aliases describe the stable data shapes used at this
# boundary without forcing the optional package into every type-checking env.
ChromaScalar: TypeAlias = str | int | float | bool | None
Embedding: TypeAlias = Sequence[float]
PyEmbedding: TypeAlias = Sequence[float]
ChromaDocument: TypeAlias = str
Image: TypeAlias = object
URI: TypeAlias = str
ID: TypeAlias = str
Include: TypeAlias = list[str]
QueryResult: TypeAlias = Dict[str, Any]
Where: TypeAlias = Dict[str, Any]
WhereDocument: TypeAlias = Dict[str, Any]
IDs: TypeAlias = List[str]
_TCollectionValue = TypeVar("_TCollectionValue")
OneOrMany: TypeAlias = _TCollectionValue | Sequence[_TCollectionValue]
GetResult: TypeAlias = Dict[str, Any]
Metadata: TypeAlias = Dict[str, ChromaScalar]


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


class NamedCollectionLike(Protocol):
    """Minimal name-bearing collection surface used by native projections."""

    name: str


class ProjectionBackendLike(Protocol):
    """Optional backend attributes used by native projection fast paths.

    These handles are intentionally not members of ``StorageBackend`` because
    lightweight and third-party backends are not required to expose them.
    Callers must narrow to this protocol only at the fast-path boundary.
    """

    nodes: NamedCollectionLike
    edges: NamedCollectionLike
    documents: NamedCollectionLike
    embedding_dim: int
    distance: str


if TYPE_CHECKING:
    from .llm_tasks import LLMTaskSet

# -------------------------
# Graph objects (structural)
# -------------------------


@runtime_checkable
class NodeLike(Protocol):
    @property
    def id(self) -> str | None: ...

    @property
    def label(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def domain_id(self) -> Optional[str]: ...

    @property
    def canonical_entity_id(self) -> Optional[str]: ...

    @property
    def properties(self) -> Optional[Mapping[str, object]]: ...

    @property
    def mentions(self) -> Optional[Sequence[object]]: ...

    @property
    def embedding(self) -> Optional[Sequence[float]]: ...

    @property
    def doc_id(self) -> Optional[str]: ...

    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self) -> str: ...


@runtime_checkable
class EdgeLike(NodeLike, Protocol):
    @property
    def relation(self) -> str: ...

    @property
    def source_ids(self) -> Optional[Sequence[str]]: ...

    @property
    def target_ids(self) -> Optional[Sequence[str]]: ...

    @property
    def source_edge_ids(self) -> Optional[Sequence[str]]: ...

    @property
    def target_edge_ids(self) -> Optional[Sequence[str]]: ...


AdjudicationTarget: TypeAlias = Union[NodeLike, EdgeLike]

# -------------------------
# Shared engine surface
# -------------------------

class EmbeddingFunctionLike(Protocol):
    def __call__(self, documents_or_texts: list[str]) -> list[list[float]]: ...


class ReadLike(Protocol):
    def get_nodes(self, *args: Any, **kwargs: Any) -> Sequence[GraphNode]: ...
    def get_edges(self, *args: Any, **kwargs: Any) -> Sequence[GraphEdge]: ...
    def query_nodes(self, *args: Any, **kwargs: Any) -> Sequence[Sequence[GraphNode]]: ...
    def query_edges(self, *args: Any, **kwargs: Any) -> Sequence[Sequence[GraphEdge]]: ...
    def search_nodes_as_of(self, *args: Any, **kwargs: Any) -> Sequence[GraphNode]: ...

    def search_nodes_as_of_scored(
        self, *args: Any, **kwargs: Any
    ) -> list["VectorSearchHit[GraphNode]"]: ...

    def get_document(self, doc_id: str) -> EngineDoc: ...

    def node_exists(
        self,
        ids: Sequence[str] | None = None,
        where: Dict[str, Any] | None = None,
    ) -> bool: ...

    def edge_exists(
        self,
        ids: Sequence[str] | None = None,
        where: Dict[str, Any] | None = None,
    ) -> bool: ...

    def get_node_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where: Dict[str, Any] | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, Any]]: ...

    def get_edge_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where: Dict[str, Any] | None = None,
        limit: int | None = 400,
    ) -> list[dict[str, Any]]: ...

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


class LifecycleLike(Protocol):
    """Stable lifecycle mutation surface shared by engine implementations."""

    def tombstone_node(self, node_id: str, **kwargs: object) -> bool: ...
    def redirect_node(self, from_id: str, to_id: str, **kwargs: object) -> bool: ...
    def tombstone_edge(self, edge_id: str, **kwargs: object) -> bool: ...
    def redirect_edge(self, from_id: str, to_id: str, **kwargs: object) -> bool: ...


class WriteLike(Protocol):
    def add_document(self, document: EngineDoc) -> None: ...

    def add_node(self, node: GraphNode, doc_id: str | None = None) -> None: ...
    def add_edge(self, edge: GraphEdge, doc_id: str | None = None) -> None: ...

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
    def iterative_defensive_emb(self, emb_text0: str) -> list[float]: ...


@runtime_checkable
class TokenAwareEmbeddingFunction(Protocol):
    """Optional provider capability for safe token-bounded embedding input."""

    max_input_tokens: int

    def count_tokens(self, text: str) -> int: ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str: ...


class AdjudicateLike(Protocol):
    def target_from_node(self, n: GraphNode) -> GraphAdjudicationTarget: ...
    def target_from_edge(self, e: GraphEdge) -> GraphAdjudicationTarget: ...
    def fetch_target(self, t: GraphAdjudicationTarget) -> GraphNode | GraphEdge: ...

    def split_endpoints(
        self,
        src_ids: list[str] | None,
        tgt_ids: list[str] | None,
    ) -> tuple[list[str], list[str], list[str], list[str]]: ...


class EngineLike(Protocol):
    """Public read-oriented engine contract used by lightweight helpers."""

    @property
    def backend(self) -> StorageBackend: ...

    @property
    def read(self) -> ReadLike: ...


class StrategyEngineLike(EngineLike, Protocol):
    """Public engine contract used by strategy modules."""

    @property
    def read(self) -> ReadLike: ...

    @property
    def write(self) -> WriteLike: ...

    @property
    def extract(self) -> ExtractLike: ...

    @property
    def embed(self) -> EmbedLike: ...

    @property
    def adjudicate(self) -> AdjudicateLike: ...

    @property
    def lifecycle(self) -> LifecycleLike: ...

    @property
    def llm_tasks(self) -> "LLMTaskSet": ...
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
