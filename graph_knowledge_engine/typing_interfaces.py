# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, TypedDict, TypeVar, Sequence, Union, runtime_checkable, TYPE_CHECKING
try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - py<3.10 compatibility
    from typing_extensions import TypeAlias
from .engine_core.models import AdjudicationVerdict, Document as EngineDoc
# -------------------------
# Collection / Vector store
# -------------------------

# class GetResult(TypedDict, total=False):
#     ids: List[str]
#     documents: List[str]
#     metadatas: List[Dict[str, Any]]
#     embeddings: List[List[float]]
#     distances: List[List[float]]
#     uris: List[str]
#     data: Any  # some backends return extra buckets
# class QueryResult(TypedDict, total=False):
#     ids: List[List[str]]
#     documents: List[List[str]]
#     metadatas: List[List[Dict[str, Any]]]
#     distances: List[List[float]]

try:
    from chromadb.api.types import Embedding, PyEmbedding, Document as ChromaDocument, Image, URI, ID, Include, QueryResult
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
    type: str  # "entity" or "relationship" (when you treat Edge-as-Node)
    summary: str
    domain_id: Optional[str]
    canonical_entity_id: Optional[str]
    properties: Optional[Dict[str, Any]]
    mentions: Optional[List[Any]]  # ReferenceSession, but keep it loose here
    embedding: Optional[List[float]]
    doc_id: Optional[str]

    # pydantic convenience (your models expose these)
    def model_dump(self) -> Dict[str, Any]: ...
    def model_dump_json(self) -> str: ...

@runtime_checkable
class EdgeLike(NodeLike, Protocol):
    # Hyperedge endpoints
    relation: str
    source_ids: Optional[List[str]]
    target_ids: Optional[List[str]]
    source_edge_ids: Optional[List[str]]
    target_edge_ids: Optional[List[str]]

# Union the adjudication target
AdjudicationTarget = Union[NodeLike, EdgeLike]

# -------------------------
# Engine surface used by strategies
# -------------------------
class EmbeddingFunctionLike(Protocol):
    def __call__(self, documents_or_texts: list[str]) -> Any: ...
class EngineLike(Protocol):
    """
    The minimal 'engine' surface strategies depend on.
    Keep this lean so Pylance/mypy can reason about it without importing
    heavy backends.
    """
    # vector-store collections
    node_collection: CollectionLike
    edge_collection: CollectionLike
    edge_endpoints_collection: CollectionLike
    document_collection: CollectionLike
    # _ef : EmbeddingFunctionLike
    
    _ef: EmbeddingFunctionLike
    # optional indexes
    node_docs_collection: CollectionLike

    def get_document(self, doc_id: str) -> EngineDoc: ...
    # helpers the strategies call
    def chroma_sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]: ...
    def _json_or_none(self, obj: Any) -> Optional[str]: ...
    
    # llm task runner
    llm_tasks: "LLMTaskSet"

    # -------------------------
    # adjudication / commit hooks
    # -------------------------
    
    """
    Merge-or-link two objects of the same *kind* (node↔node or edge↔edge).
    Returns a canonical id (or link edge id) used/created.
    """

    def commit_merge_target(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        verdict: AdjudicationVerdict,  # AdjudicationVerdict
    ) -> str: ...
    """
    Cross-kind linker (node↔edge or edge↔node). Should create a meta-edge
    like 'reifies' / 'equivalent_node_edge' based on engine policy.
    Returns the id of the linking edge created.
    """
