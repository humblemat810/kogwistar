# -*- coding: utf-8 -*-
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - py<3.10 compatibility
    from typing import TypeAlias

from .json_types import JsonObject, JsonValue

if TYPE_CHECKING:
    from .engine_core.models import (
        AdjudicationTarget as GraphAdjudicationTarget,
    )
    from .engine_core.models import (
        AdjudicationVerdict,
        PureChromaEdge,
        PureChromaNode,
    )
    from .engine_core.models import (
        Document as EngineDoc,
    )
    from .engine_core.models import (
        Domain as GraphDomain,
    )
    from .engine_core.models import (
        Edge as GraphEdge,
    )
    from .engine_core.models import (
        Node as GraphNode,
    )
    from .engine_core.storage_backend import StorageBackend
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
QueryResult: TypeAlias = dict[str, JsonValue]
Where: TypeAlias = dict[str, JsonValue]
WhereDocument: TypeAlias = dict[str, JsonValue]
IDs: TypeAlias = list[str]
_TCollectionValue = TypeVar("_TCollectionValue")
OneOrMany: TypeAlias = _TCollectionValue | Sequence[_TCollectionValue]
GetResult: TypeAlias = dict[str, JsonValue]
Metadata: TypeAlias = dict[str, ChromaScalar]


class CollectionLike(Protocol):
    def add(
        self,
        ids: OneOrMany[ID],
        embeddings: OneOrMany[Embedding] | None = None,
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[ChromaDocument] | None = None,
        images: OneOrMany[Image] | None = None,
        uris: OneOrMany[URI] | None = None,
    ) -> None: ...

    def update(
        self,
        ids: OneOrMany[ID],
        embeddings: OneOrMany[Embedding] | None = None,
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[ChromaDocument] | None = None,
        images: OneOrMany[Image] | None = None,
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
    domains: NamedCollectionLike
    embedding_dim: int
    distance: str


class SqlAlchemyEngineLike(Protocol):
    """Minimal lifecycle surface shared by optional SQLAlchemy adapters."""

    def begin(self) -> AbstractContextManager[object]: ...

    def connect(self) -> AbstractContextManager[object]: ...

    def dispose(self) -> None: ...


class SqlAlchemyScalarResultLike(Protocol):
    """Small scalar-result surface used by optional SQLAlchemy adapters."""

    def all(self) -> Sequence[object]: ...


class SqlAlchemyMappingResultLike(Protocol):
    def first(self) -> Mapping[str, object] | None: ...

    def all(self) -> Sequence[Mapping[str, object]]: ...


class SqlAlchemyResultLike(Protocol):
    """Dependency-light result surface for typed SQLAlchemy integration."""

    def scalar_one(self) -> object: ...

    def scalar_one_or_none(self) -> object | None: ...

    def scalars(self) -> SqlAlchemyScalarResultLike: ...

    def mappings(self) -> SqlAlchemyMappingResultLike: ...

    def first(self) -> object | None: ...

    @property
    def rowcount(self) -> int: ...

    def fetchone(self) -> object | None: ...

    def fetchall(self) -> Sequence[Any]: ...

    def __iter__(self) -> Iterator[Any]: ...


class SqlAlchemyConnectionLike(Protocol):
    """Minimal connection operations used without importing SQLAlchemy types."""

    def execute(
        self,
        statement: object,
        *args: object,
        **kwargs: object,
    ) -> SqlAlchemyResultLike: ...

    def exec_driver_sql(
        self,
        statement: str,
        *args: object,
        **kwargs: object,
    ) -> SqlAlchemyResultLike: ...


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
    def domain_id(self) -> str | None: ...

    @property
    def canonical_entity_id(self) -> str | None: ...

    @property
    def properties(self) -> Mapping[str, object] | None: ...

    @property
    def mentions(self) -> Sequence[object] | None: ...

    @property
    def embedding(self) -> Sequence[float] | None: ...

    @property
    def doc_id(self) -> str | None: ...

    def model_dump(self) -> dict[str, object]: ...
    def model_dump_json(self) -> str: ...


@runtime_checkable
class EdgeLike(NodeLike, Protocol):
    @property
    def relation(self) -> str: ...

    @property
    def source_ids(self) -> Sequence[str] | None: ...

    @property
    def target_ids(self) -> Sequence[str] | None: ...

    @property
    def source_edge_ids(self) -> Sequence[str] | None: ...

    @property
    def target_edge_ids(self) -> Sequence[str] | None: ...


AdjudicationTarget: TypeAlias = NodeLike | EdgeLike

# -------------------------
# Shared engine surface
# -------------------------

class EmbeddingFunctionLike(Protocol):
    def __call__(self, documents_or_texts: list[str]) -> list[list[float]]: ...


QueryEmbeddingInput: TypeAlias = Sequence[float] | Sequence[Sequence[float]]


class ReadLike(Protocol):
    def load_node_map(
        self,
        ids: Sequence[str],
        *,
        node_type: type[GraphNode] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, GraphNode]: ...

    def load_edge_map(
        self,
        ids: Sequence[str],
        *,
        edge_type: type[GraphEdge] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, GraphEdge]: ...

    def get_nodes(
        self,
        ids: Sequence[str] | None = None,
        node_type: type[GraphNode] | None = None,
        include: list[str] | None = None,
        where: object = None,
        limit: int | None = 200,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> Sequence[GraphNode]: ...

    def get_edges(
        self,
        ids: Sequence[str] | None = None,
        edge_type: type[GraphEdge] | None = None,
        where: object = None,
        limit: int | None = 400,
        include: list[str] | None = None,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"] = "active_only",
    ) -> Sequence[GraphEdge]: ...
    def query_nodes(
        self,
        *args: object,
        query: str | None = None,
        query_embeddings: QueryEmbeddingInput | None = None,
        include: list[str] = ["documents", "embeddings", "metadatas"],
        node_type: type[GraphNode] | None = None,
        **kwargs: object,
    ) -> Sequence[Sequence[GraphNode]]: ...

    def query_edges(
        self,
        *args: object,
        query: str | None = None,
        query_embeddings: QueryEmbeddingInput | None = None,
        include: list[str] = ["documents", "embeddings", "metadatas"],
        edge_type: type[GraphEdge] | None = None,
        **kwargs: object,
    ) -> Sequence[Sequence[GraphEdge]]: ...
    def search_nodes_as_of(
        self,
        *,
        query: str | None = None,
        query_embeddings: Sequence[float] | Sequence[Sequence[float]] | None = None,
        as_of_ts: datetime | str,
        where: dict[str, JsonValue] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: type[GraphNode] = ...,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        similarity_threshold: float | None = None,
        **kwargs: object,
    ) -> Sequence[GraphNode]: ...

    def search_nodes_as_of_scored(
        self,
        *,
        query: str | None = None,
        query_embeddings: Sequence[float] | Sequence[Sequence[float]] | None = None,
        as_of_ts: datetime | str,
        where: dict[str, JsonValue] | None = None,
        n_results: int = 20,
        follow_redirects: bool = True,
        node_type: type[GraphNode] = ...,
        include: list[str] | None = None,
        max_redirect_hops: int = 16,
        similarity_threshold: float | None = None,
        **kwargs: object,
    ) -> list[VectorSearchHit[GraphNode]]: ...

    def get_document(self, doc_id: str) -> EngineDoc: ...

    def node_exists(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
    ) -> bool: ...

    def edge_exists(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
    ) -> bool: ...

    def get_node_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 200,
    ) -> list[dict[str, JsonValue]]: ...

    def get_edge_metadatas(
        self,
        ids: Sequence[str] | None = None,
        where: dict[str, JsonValue] | None = None,
        limit: int | None = 400,
    ) -> list[dict[str, JsonValue]]: ...

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

    def edges_by_doc(
        self, doc_id: str, where: dict[str, JsonValue] | None = None
    ) -> list[str]: ...

    def list_edges_with_ref_filter(
        self, doc_id: str, where: dict[str, JsonValue] | None = None
    ) -> list[GraphEdge]: ...

    def nodes_by_doc(
        self, doc_id: str, *, where: dict[str, JsonValue] | None = None
    ) -> list[str]: ...

    def list_nodes_with_ref_filter(
        self, doc_id: str, *, where: dict[str, JsonValue] | None = None
    ) -> list[GraphNode]: ...

    def ids_with_insertion_method(
        self,
        *,
        kind: str,
        insertion_method: str,
        ids: Sequence[str] | None = None,
        doc_id: str | None = None,
    ) -> list[str]: ...

    def extract_reference_contexts(
        self,
        node_or_id: GraphNode | GraphEdge | str,
        *,
        window_chars: int = 120,
        max_contexts: int | None = None,
        prefer_label_fallback: bool = True,
    ) -> list[dict[str, object]]: ...

    def nodes_from_single_or_id_query_result(
        self,
        got: Mapping[str, JsonValue],
        node_type: type[GraphNode] = ...,
    ) -> list[GraphNode]: ...

    def edges_from_single_or_id_query_result(
        self,
        got: Mapping[str, JsonValue],
        edge_type: type[GraphEdge] = ...,
        include: Sequence[str] | None = None,
    ) -> list[GraphEdge]: ...

    def nodes_from_query_result(
        self,
        gots: Mapping[str, JsonValue],
        node_type: type[GraphNode] = ...,
    ) -> list[list[GraphNode]]: ...

    def edges_from_query_result(
        self,
        gots: Mapping[str, JsonValue],
        edge_type: type[GraphEdge] = ...,
    ) -> list[list[GraphEdge]]: ...

    def where_update_from_resolve_mode(
        self,
        resolve_mode: Literal["active_only", "redirect", "include_tombstones"],
    ) -> dict[str, str]: ...


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

    async def add_node_async(
        self, node: GraphNode, doc_id: str | None = None
    ) -> None: ...

    async def add_edge_async(
        self, edge: GraphEdge, doc_id: str | None = None
    ) -> None: ...

    def add_pure_node(self, node: PureChromaNode) -> None: ...
    def add_pure_edge(self, edge: PureChromaEdge) -> None: ...
    def add_domain(self, domain: GraphDomain) -> None: ...
    def enrich_edge_meta(self, edge: GraphEdge) -> dict[str, object]: ...
    def fanout_endpoints_rows(
        self, edge: GraphEdge, doc_id: str | None
    ) -> object: ...

    def node_doc_and_meta(self, node: GraphNode) -> tuple[str, JsonObject]: ...
    def edge_doc_and_meta(self, edge: GraphEdge) -> tuple[str, JsonObject]: ...

    def strip_none(self, data: dict[str, JsonValue]) -> dict[str, JsonValue]: ...
    def json_or_none(self, value: object) -> str | None: ...

    def index_node_docs(self, node: GraphNode) -> list[str]: ...
    def index_node_refs(self, node: GraphNode) -> list[str]: ...
    def index_edge_refs(self, edge: GraphEdge) -> list[str]: ...

    def delete_edge_ref_rows(self, edge_id: str) -> None: ...
    def delete_node_ref_rows(self, node_id: str) -> None: ...

    def maybe_reindex_edge_refs(
        self, edge: GraphEdge, *, force: bool = False
    ) -> None: ...

    def maybe_reindex_node_refs(
        self, node: GraphNode, *, force: bool = False
    ) -> None: ...

    def prune_node_refs_for_doc(self, node_id: str, doc_id: str) -> bool: ...
    def rebuild_edge_refs_for_doc(self, doc_id: str) -> int: ...
    def rebuild_all_edge_refs(self) -> int: ...
    def rebuild_node_refs_for_doc(self, doc_id: str) -> int: ...
    def rebuild_all_node_refs(self) -> int: ...
    def delete_edges_by_ids(self, edge_ids: list[str]) -> None: ...

    def rust_postgres_delete_existing(
        self, *, entity_kind: str, entity_ids: list[str]
    ) -> bool: ...

    def rust_postgres_replace_existing(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        document: str,
        metadata_patch: JsonObject,
        payload: JsonObject,
    ) -> bool: ...

    def uses_rust_postgres_authority(self) -> bool: ...


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
    def llm_tasks(self) -> LLMTaskSet: ...
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
