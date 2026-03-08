
from typing import List, Tuple, Any, Dict, runtime_checkable, Protocol, Sequence, Optional, Callable

from pydantic import BaseModel
from ..engine_core.models import Node, Edge,AdjudicationQuestionCode, AdjudicationVerdict, AdjudicationTarget, LLMMergeAdjudication, Span
from ..typing_interfaces import StrategyEngineLike as EngineLike
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
class IAdjudicator(Protocol):
    def batch_adjudicate_merges(
        self,
        pairs: List[Tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    )-> list[Any] | tuple[list[Any], str] | tuple[list[None], str]: ...# 
    def adjudicate_pair(self, left: AdjudicationTarget, right: AdjudicationTarget, question: str)-> Dict[Any, Any] | BaseModel: ...
    def adjudicate_merge(self, left_node: Node | Edge, right_node: Node | Edge) -> Dict[Any, Any] | BaseModel:...
    
@runtime_checkable
class PairAdjudicator(Protocol):
    def adjudicate(self, engine: "EngineLike", left: Any, right: Any) -> AdjudicationVerdict: ...


@runtime_checkable
class BatchAdjudicator(Protocol):
    def batch_adjudicate(
        self,
        engine: "EngineLike",
        pairs: List[Tuple[Any, Any]],
        question_code: AdjudicationQuestionCode = AdjudicationQuestionCode.SAME_ENTITY,
    ) -> Tuple[List[LLMMergeAdjudication], str]: ...


# ---------- Merge policy ----------
@runtime_checkable
class MergePolicy(Protocol):
    def commit_merge_target(self, left: AdjudicationTarget, right: AdjudicationTarget, verdict: AdjudicationVerdict) -> str:...


@runtime_checkable
class CrossKindPolicy(Protocol):
    def commit(self, engine: "EngineLike", left: Node | Edge, right: Node | Edge, verdict: AdjudicationVerdict) -> str: ...


# ---------- Verifier ----------
class VerificationReport(BaseModel):
    updated_node_ids: List[str] = []
    updated_edge_ids: List[str] = []


@runtime_checkable
class Verifier(Protocol):
    # def verify_document(self, engine: EngineLike, document_id: str, method: str = "levenshtein") -> VerificationReport: ...
    def _verify_one_reference(
        self,
        extracted_text: str,
        full_text: str,
        ref: Span,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        threshold: float = 0.70,
    ) -> Span: ...
    def verify_mentions_for_doc(
        self,
        document_id: str,
        *,
        source_text: Optional[str] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        update_edges: bool = True,
    ) -> Dict[str, int]: ...
    def verify_mentions_for_items(
        self,
        items: List[Tuple[str, str]],  # list of ("node"|"edge", id)
        *,
        source_text_by_doc: Optional[Dict[str, str]] = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
    ) -> Dict[str, int]: ...
