# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict, Union, runtime_checkable
from .models import AdjudicationVerdict
# -------------------------
# Collection / Vector store
# -------------------------

class GetResult(TypedDict, total=False):
    ids: List[str]
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    embeddings: List[List[float]]
    distances: List[List[float]]
    uris: List[str]
    data: Any  # some backends return extra buckets

class QueryResult(TypedDict, total=False):
    ids: List[List[str]]
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]
    distances: List[List[float]]

class CollectionLike(Protocol):
    """Minimal shape of a Chroma-like collection used in engine/strategies."""

    def add(
        self,
        *,
        ids: List[str],
        documents: Optional[List[str]] = ...,
        embeddings: Optional[List[List[float]]] = ...,
        metadatas: Optional[List[Dict[str, Any]]] = ...,
        uris: Optional[List[str]] = ...,
    ) -> Any: ...

    def update(
        self,
        *,
        ids: List[str],
        documents: Optional[List[str]] = ...,
        embeddings: Optional[List[List[float]]] = ...,
        metadatas: Optional[List[Dict[str, Any]]] = ...,
        uris: Optional[List[str]] = ...,
    ) -> Any: ...

    def get(
        self,
        *,
        ids: Optional[List[str]] = ...,
        where: Optional[Dict[str, Any]] = ...,
        include: Optional[List[str]] = ...,
        limit: Optional[int] = ...,
        offset: Optional[int] = ...,
    ) -> GetResult: ...

    def delete(
        self,
        *,
        ids: Optional[List[str]] = ...,
        where: Optional[Dict[str, Any]] = ...,
    ) -> Any: ...

    def query(
        self,
        *,
        query_embeddings: Optional[List[List[float]]] = ...,
        n_results: int = ...,
        where: Optional[Dict[str, Any]] = ...,
        include: Optional[List[str]] = ...,
    ) -> QueryResult: ...

# -------------------------
# LangChain LLM surface
# -------------------------

class RunnableLike(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> Any: ...

@runtime_checkable
class ChatModelLike(Protocol):
    """Just the piece we call: llm.with_structured_output(...)."""
    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = ...,
        strict: Optional[bool] = ...,
    ) -> RunnableLike: ...

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
    references: Optional[List[Any]]  # ReferenceSession, but keep it loose here
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
    _ef : Callable
    # optional indexes
    node_docs_collection: CollectionLike


    # helpers the strategies call
    def chroma_sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]: ...
    def _json_or_none(self, obj: Any) -> Optional[str]: ...
    
    # llm runner
    llm: ChatModelLike

    # -------------------------
    # adjudication / commit hooks
    # -------------------------
    def commit_merge_target(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        verdict: AdjudicationVerdict,  # AdjudicationVerdict
    ) -> str: ...
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