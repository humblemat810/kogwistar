# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, List, Protocol, Tuple, Sequence

from ..typing_interfaces import (
    ChatModelLike,
    NodeLike,
    EdgeLike,
    AdjudicationTarget,
)

from candidate_proposals import DefaultProposer
from adjudicators import LLMPairAdjudicatorImpl, LLMBatchAdjudicatorImpl
from verifiers import DefaultVerifier, VerifierConfig
from merge_policies import PreferExistingCanonical


from typing import Protocol, List, Tuple, Any, Optional, Literal, Iterable, runtime_checkable, Dict
from ..models import LLMMergeAdjudication, AdjudicationVerdict, Node, Edge

@runtime_checkable
class Proposer(Protocol):
    def generate_merge_candidates(
        self,
        *,
        kind: Literal["node", "edge"],
        scope_doc_id: Optional[str] = None,
        top_k: int = 50,
        similarity_threshold: float = 0.85,
    ) -> List[Tuple[Any, Any]]:
        ...

    def generate_cross_kind_candidates(
        self,
        *,
        scope_doc_id: Optional[str] = None,
        limit_per_bucket: int = 50,
    ) -> List[Tuple[Any, Any]]:
        ...

@runtime_checkable
class PairAdjudicator(Protocol):
    def adjudicate_pair(
        self,
        left: Any,
        right: Any,
        *,
        question: str = "same_entity",
    ) -> LLMMergeAdjudication:
        ...

@runtime_checkable
class BatchAdjudicator(Protocol):
    def adjudicate_batch(
        self,
        pairs: List[Tuple[Any, Any]],
        *,
        question_code: int,
    ) -> Tuple[List[LLMMergeAdjudication], str]:
        ...

@runtime_checkable
class Verifier(Protocol):
    def verify_mentions_for_doc(self, document_id: str) -> dict:
        ...
        
@runtime_checkable
class MergePolicy(Protocol):
    """Apply a positive verdict (node↔node, edge↔edge or cross-kind link)."""
    def merge(self, left: Any, right: Any, verdict: AdjudicationVerdict) -> str: ...

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
