
from typing import List, Tuple, Any, Dict, runtime_checkable, Protocol, Sequence, Optional

from pydantic import BaseModel
from ..models import Node, Edge,AdjudicationQuestionCode, AdjudicationVerdict, AdjudicationTarget, LLMMergeAdjudication, ReferenceSession
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
        ref: ReferenceSession,
        *,
        min_ngram: int = 5,
        weights: Dict[str, float] = {"rapidfuzz": 0.5, "coverage": 0.3, "embedding": 0.2},
        threshold: float = 0.70,
    ) -> ReferenceSession: ...
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
from ..typing_interfaces import CollectionLike
from langchain_core.language_models.chat_models import BaseChatModel
class EngineLike(Protocol):
    """Narrow surface area your strategies depend on."""
    def _nodes_by_doc(self, doc_id: str, insertion_method: Optional[str] = None) -> list[str]:...
    def _edge_ids_by_doc(self, doc_id: str, insertion_method: Optional[str] = None) -> list[str]:...
    def _fetch_target(self, t: AdjudicationTarget) -> Node | Edge:...
    # helpers the strategies call
    @staticmethod
    def chroma_sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]: ...
    @staticmethod
    def _json_or_none(obj: Any) -> Optional[str]: ...
    @staticmethod
    def _default_ref(doc_id: str, snippet: Optional[str] = None) -> ReferenceSession: ...
    def _split_endpoints(self, src_ids: list[str] | None, tgt_ids: list[str] | None) -> tuple[list[Any], list[Any], list[Any], list[Any]]: ...
    def _index_node_docs(self, node: Node) -> list[str]:...
    def _index_node_refs(self, node: Node) -> None:...
    def _index_edge_refs(self, e: Edge) -> list[str]:...
    def _best_ref(self, n: Node | Edge) -> ReferenceSession:...
    @staticmethod
    def _strip_none(d: dict) -> dict: ...
    def add_edge(self, edge: Edge, doc_id: Optional[str] = None): ...
    def add_node(self, node: Node, doc_id: Optional[str] = None): ...
    # node, edge to common type
    def _target_from_node(self, n : Node) -> AdjudicationTarget: ...
    def _target_from_edge(self, e: Edge) -> AdjudicationTarget: ...
    
    # llm runner
    llm: BaseChatModel
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
    def _fetch_document_text(self, document_id: str) -> str:...
    def embedding_function(self, documents_or_texts: list[str]) -> list[list[float]]: ...
    @staticmethod
    def _node_doc_and_meta(n: "Node") -> tuple[str, dict]: ...
    @staticmethod
    def _edge_doc_and_meta(e: "Edge") -> tuple[str, dict]:...
        # vector-store collections
        
    allow_cross_kind_adjudication: bool
    node_collection: CollectionLike
    edge_collection: CollectionLike
    edge_endpoints_collection: CollectionLike
    document_collection: CollectionLike
    cross_kind_strategy: str
    # optional indexes
    node_docs_collection: CollectionLike