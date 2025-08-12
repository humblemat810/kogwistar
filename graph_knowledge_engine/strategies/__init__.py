# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Protocol, runtime_checkable, List, Tuple, Any, Dict, Optional, Iterable
from pydantic import BaseModel

from ..models import (
    Node,
    Edge,
    AdjudicationVerdict,
    LLMMergeAdjudication,
    AdjudicationQuestionCode,
)
from typing import Any, List, Protocol, Tuple, Sequence

from ..typing_interfaces import (
    ChatModelLike,
    NodeLike,
    EdgeLike,
    AdjudicationTarget,
)
from pydantic import BaseModel
from graph_knowledge_engine.strategies.proposer import CompositeProposer, VectorProposer
from adjudicators import LLMPairAdjudicatorImpl, LLMBatchAdjudicatorImpl
from verifiers import DefaultVerifier, VerifierConfig
from merge_policies import PreferExistingCanonical


# ---------- Proposer ----------
@runtime_checkable
class MergeCandidateProposer(Protocol):
    """Unified proposer interface the engine can depend on."""

    # Single new-node vector search (existing behavior)
    def for_new_node(
        self,
        engine: "EngineLike",
        new_node: Node,
        top_k: int = 5,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Node, Node]]: ...

    # Batch proposal within a document: same-kind pairs (node↔node & edge↔edge)
    def same_kind_in_doc(
        self,
        engine: "EngineLike",
        doc_id: str, kind: str
    ) -> List[Tuple[Any, Any]]: ...

    # Batch proposal within a document: cross-kind pairs (node↔edge)
    def cross_kind_in_doc(
        self,
        engine: "EngineLike",
        doc_id: str,
    ) -> List[Tuple[Any, Any]]: ...


# ---------- Adjudicator ----------
@runtime_checkable
class PairAdjudicator(Protocol):
    def adjudicate(self, engine: EngineLike, left: Any, right: Any) -> AdjudicationVerdict: ...


@runtime_checkable
class BatchAdjudicator(Protocol):
    def batch_adjudicate(
        self,
        engine: EngineLike,
        pairs: List[Tuple[Any, Any]],
        question_code: AdjudicationQuestionCode = AdjudicationQuestionCode.SAME_ENTITY,
    ) -> Tuple[List[LLMMergeAdjudication], str]: ...


# ---------- Merge policy ----------
@runtime_checkable
class MergePolicy(Protocol):
    def commit(self, engine: EngineLike, left: Node, right: Node, verdict: AdjudicationVerdict) -> str: ...


@runtime_checkable
class CrossKindPolicy(Protocol):
    def commit(self, engine: EngineLike, left: Node | Edge, right: Node | Edge, verdict: AdjudicationVerdict) -> str: ...


# ---------- Verifier ----------
class VerificationReport(BaseModel):
    updated_node_ids: List[str] = []
    updated_edge_ids: List[str] = []


@runtime_checkable
class Verifier(Protocol):
    def verify_document(self, engine: EngineLike, document_id: str, method: str = "levenshtein") -> VerificationReport: ...

from ..typing_interfaces import CollectionLike
class EngineLike(Protocol):
    """Narrow surface area your strategies depend on."""

    # helpers the strategies call
    def chroma_sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]: ...
    def _json_or_none(self, obj: Any) -> Optional[str]: ...
    
    # node, edge to common type
    def _target_from_node(self, n : Node) -> AdjudicationTarget: ...
    def _target_from_edge(self, e: Edge) -> AdjudicationTarget: ...
    
    # llm runner
    llm: ChatModelLike
    # --- Graph reads ---
    def get_nodes(self, ids: Sequence[str]) -> List[Node]: ...
    def get_edges(self, ids: Sequence[str]) -> List[Edge]: ...

    # doc-scoped indexes (public wrappers for your old _node_ids_by_doc/_edge_ids_by_doc)
    def node_ids_by_doc(self, doc_id: str) -> List[str]: ...
    def edge_ids_by_doc(self, doc_id: str) -> List[str]: ...

    # optional convenience (keeps strategies simple)
    def all_nodes_for_doc(self, doc_id: str) -> List[Node]: ...
    def all_edges_for_doc(self, doc_id: str) -> List[Edge]: ...

    # --- Write paths used by adjudication/merge policies ---
    def commit_merge(self, left: Node, right: Node, verdict: AdjudicationVerdict) -> str: ...
    def commit_merge_target(self, left: AdjudicationTarget, right: AdjudicationTarget, verdict: AdjudicationVerdict) -> str: ...

    def embedding_function(self, documents_or_texts: list[str]) -> list[list[float]]: ...
        # vector-store collections
    node_collection: CollectionLike
    edge_collection: CollectionLike
    edge_endpoints_collection: CollectionLike
    document_collection: CollectionLike
    
    # optional indexes
    node_docs_collection: CollectionLike


__all__ = ['DefaultProposer', "LLMPairAdjudicatorImpl", "LLMBatchAdjudicatorImpl",
           "DefaultVerifier", "VerifierConfig", "PreferExistingCanonical", "EngineLike",
            "NodeLike",
            "EdgeLike", "AdjudicationTarget"]
