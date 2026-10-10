from os import PathLike
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from ..engine_core.models import (
    AdjudicationQuestionCode,
    AdjudicationTarget,
    AdjudicationVerdict,
    Edge,
    LLMMergeAdjudication,
    Node,
    Span,
)
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
    ) -> list[tuple[Node, Node]]: ...

    # Batch proposal within a document: same-kind pairs (node↔node & edge↔edge)
    def same_kind_in_doc(
        self,
        *,
        engine: "EngineLike",
        doc_id: str,
        kind: str = "node",
    ) -> list[tuple[Any, Any]]: ...

    # Batch proposal within a document: cross-kind pairs (node↔edge)
    def cross_kind_in_doc(
        self,
        *,
        engine: "EngineLike",
        doc_id: str,
        limit_per_bucket: int = 200,
    ) -> list[tuple[Any, Any]]: ...

    def propose_any_kind_any_doc(
        self,
        *,
        engine: "EngineLike",
        pair_kind: Literal[
            "node_node",
            "edge_edge",
            "node_edge",
            "any_any",
            "any_node",
            "any_edge",
            "node_any",
            "edge_any",
        ],
        allowed_docs: list[str] | None = None,
        anchor_doc_id: str | None = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
        limit_per_bucket: int | None = None,
    ) -> list[tuple[Node | Edge, Node | Edge]]: ...

    def generate_merge_candidates(
        self,
        engine: "EngineLike",
        new_node: Node | Edge | str | Sequence[Node | Edge | str] | None,
        new_edge: Node | Edge | str | Sequence[Node | Edge | str] | None = None,
        top_k: int = 10,
        *,
        allowed_docs: list[str] | None = None,
        anchor_doc_id: str | None = None,
        cross_doc_only: bool = False,
        anchor_only: bool = True,
        score_mode: Literal["distance", "similarity"] = "distance",
        max_distance: float = 0.25,
        min_similarity: float = 0.85,
        include_nodes: bool = True,
        include_edges: bool = True,
        where: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], tuple[Node | Edge, Node | Edge, float]]: ...


# ---------- Adjudicator ----------
@runtime_checkable
class IPairAdjudicationTrace(Protocol):
    @property
    def adjudication(self) -> LLMMergeAdjudication | None: ...

    @property
    def raw(self) -> object | None: ...

    @property
    def parsing_error(self) -> str | None: ...


@runtime_checkable
class IAdjudicator(Protocol):
    def batch_adjudicate_merges(
        self,
        pairs: list[tuple["Node", "Node"]],
        question_code: "AdjudicationQuestionCode" = AdjudicationQuestionCode.SAME_ENTITY,
    ) -> list[Any] | tuple[list[Any], str] | tuple[list[None], str]: ...  #
    def adjudicate_pair(
        self, left: AdjudicationTarget, right: AdjudicationTarget, question: str
    ) -> dict[Any, Any] | BaseModel: ...
    def adjudicate_merge(
        self, left_node: Node | Edge, right_node: Node | Edge
    ) -> dict[Any, Any] | BaseModel: ...
    def adjudicate_pair_trace(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        question: str,
        *,
        cache_dir: str | PathLike[str] | None = None,
    ) -> IPairAdjudicationTrace: ...


@runtime_checkable
class PairAdjudicator(Protocol):
    def adjudicate(
        self,
        engine: "EngineLike",
        left: AdjudicationTarget,
        right: AdjudicationTarget,
    ) -> AdjudicationVerdict: ...


@runtime_checkable
class BatchAdjudicator(Protocol):
    def batch_adjudicate(
        self,
        engine: "EngineLike",
        pairs: list[tuple[Any, Any]],
        question_code: AdjudicationQuestionCode = AdjudicationQuestionCode.SAME_ENTITY,
    ) -> tuple[list[LLMMergeAdjudication], str]: ...


# ---------- Merge policy ----------
@runtime_checkable
class MergePolicy(Protocol):
    def commit_merge(
        self,
        left: Node,
        right: Node,
        verdict: AdjudicationVerdict,
        method: str = "unspecified",
    ) -> str: ...

    def commit_any_kind(
        self,
        node_or_edge_l: AdjudicationTarget,
        node_or_edge_r: AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str: ...

    def commit_merge_target(
        self,
        left: AdjudicationTarget,
        right: AdjudicationTarget,
        verdict: AdjudicationVerdict,
    ) -> str: ...


@runtime_checkable
class CrossKindPolicy(Protocol):
    def commit(
        self,
        engine: "EngineLike",
        left: Node | Edge,
        right: Node | Edge,
        verdict: AdjudicationVerdict,
    ) -> str: ...


# ---------- Verifier ----------
class VerificationReport(BaseModel):
    updated_node_ids: list[str] = []
    updated_edge_ids: list[str] = []


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
        weights: dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
        threshold: float = 0.70,
    ) -> Span: ...
    def verify_mentions_for_doc(
        self,
        document_id: str,
        *,
        source_text: str | None = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
        update_edges: bool = True,
    ) -> dict[str, int]: ...
    def verify_mentions_for_items(
        self,
        items: list[tuple[str, str]],  # list of ("node"|"edge", id)
        *,
        source_text_by_doc: dict[str, str] | None = None,
        min_ngram: int = 5,
        threshold: float = 0.70,
        weights: dict[str, float] = {
            "rapidfuzz": 0.5,
            "coverage": 0.3,
            "embedding": 0.2,
        },
    ) -> dict[str, int]: ...
