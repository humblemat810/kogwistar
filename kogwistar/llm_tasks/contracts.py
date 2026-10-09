from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

try:
    from typing import TypeAlias
except ImportError:  # pragma: no cover - py<3.10 compatibility
    from typing import TypeAlias

from pydantic import BaseModel

from ..json_types import JsonValue
from .errors import MissingTaskError

JsonObject = dict[str, JsonValue]

ProviderKind: TypeAlias = Literal[
    "gemini", "openai", "ollama", "custom", "unknown"
]
ExtractionSchemaMode: TypeAlias = Literal[
    "full", "lean", "flattened_lean", "flattened_full"
]


@dataclass(frozen=True)
class LLMTaskProviderHints:
    extract_graph_provider: ProviderKind = "unknown"
    adjudicate_pair_provider: ProviderKind = "unknown"
    adjudicate_batch_provider: ProviderKind = "unknown"
    filter_candidates_provider: ProviderKind = "unknown"
    summarize_context_provider: ProviderKind = "unknown"
    answer_with_citations_provider: ProviderKind = "unknown"
    repair_citations_provider: ProviderKind = "unknown"


@dataclass(frozen=True)
class ExtractGraphTaskRequest:
    content: str
    alias_nodes: str
    alias_edges: str
    doc_alias: str
    instruction: str
    prompt_rules: str
    schema_mode: ExtractionSchemaMode
    last_parsed: Mapping[str, JsonValue] | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class ExtractGraphTaskResult:
    raw: object | None
    parsed_payload: JsonObject | None
    parsing_error: str | None


@dataclass(frozen=True)
class AdjudicatePairTaskRequest:
    question: str
    left: Mapping[str, JsonValue]
    right: Mapping[str, JsonValue]


@dataclass(frozen=True)
class AdjudicatePairTaskResult:
    verdict_payload: JsonObject | None
    raw: object | None
    parsing_error: str | None


@dataclass(frozen=True)
class AdjudicateBatchTaskRequest:
    mapping: Sequence[Mapping[str, JsonValue]]
    pairs: Sequence[Mapping[str, JsonValue]]


@dataclass(frozen=True)
class AdjudicateBatchTaskResult:
    verdict_payloads: Sequence[JsonObject]
    raw: object | None
    parsing_error: str | None


@dataclass(frozen=True)
class FilterCandidatesTaskRequest:
    conversation_content: str
    context_text: str
    candidate_nodes_text: str
    candidate_edges_text: str
    candidate_node_ids: Sequence[str]
    candidate_edge_ids: Sequence[str]
    retry_error_messages: Sequence[str] = ()


@dataclass(frozen=True)
class FilterCandidatesTaskResult:
    node_ids: Sequence[str]
    edge_ids: Sequence[str]
    reasoning: str
    raw: object | None = None
    parsing_error: str | None = None


@dataclass(frozen=True)
class SummarizeContextTaskRequest:
    full_text: str


@dataclass(frozen=True)
class SummarizeContextTaskResult:
    text: str


@dataclass(frozen=True)
class AnswerWithCitationsTaskRequest:
    system_prompt: str
    question: str
    evidence: str
    response_model: type[BaseModel]


@dataclass(frozen=True)
class AnswerWithCitationsTaskResult:
    answer_payload: JsonObject | None
    raw: object | None
    parsing_error: str | None


@dataclass(frozen=True)
class RepairCitationsTaskRequest:
    system_prompt: str
    question: str
    evidence: str
    answer_text: str
    response_model: type[BaseModel]


@dataclass(frozen=True)
class RepairCitationsTaskResult:
    answer_payload: JsonObject | None
    raw: object | None
    parsing_error: str | None


class ExtractGraphTask(Protocol):
    def __call__(self, request: ExtractGraphTaskRequest) -> ExtractGraphTaskResult: ...


class AdjudicatePairTask(Protocol):
    def __call__(
        self, request: AdjudicatePairTaskRequest
    ) -> AdjudicatePairTaskResult: ...


class AdjudicateBatchTask(Protocol):
    def __call__(
        self, request: AdjudicateBatchTaskRequest
    ) -> AdjudicateBatchTaskResult: ...


class FilterCandidatesTask(Protocol):
    def __call__(
        self, request: FilterCandidatesTaskRequest
    ) -> FilterCandidatesTaskResult: ...


class SummarizeContextTask(Protocol):
    def __call__(
        self, request: SummarizeContextTaskRequest
    ) -> SummarizeContextTaskResult: ...


class AnswerWithCitationsTask(Protocol):
    def __call__(
        self, request: AnswerWithCitationsTaskRequest
    ) -> AnswerWithCitationsTaskResult: ...


class RepairCitationsTask(Protocol):
    def __call__(
        self, request: RepairCitationsTaskRequest
    ) -> RepairCitationsTaskResult: ...


@dataclass(frozen=True)
class LLMTaskSet:
    extract_graph: ExtractGraphTask
    adjudicate_pair: AdjudicatePairTask
    adjudicate_batch: AdjudicateBatchTask
    filter_candidates: FilterCandidatesTask
    summarize_context: SummarizeContextTask
    answer_with_citations: AnswerWithCitationsTask
    repair_citations: RepairCitationsTask
    provider_hints: LLMTaskProviderHints = field(default_factory=LLMTaskProviderHints)


def validate_llm_task_set(task_set: LLMTaskSet) -> LLMTaskSet:
    required: dict[str, Callable[..., object]] = {
        "extract_graph": task_set.extract_graph,
        "adjudicate_pair": task_set.adjudicate_pair,
        "adjudicate_batch": task_set.adjudicate_batch,
        "filter_candidates": task_set.filter_candidates,
        "summarize_context": task_set.summarize_context,
        "answer_with_citations": task_set.answer_with_citations,
        "repair_citations": task_set.repair_citations,
    }
    for name, fn in required.items():
        if fn is None or not callable(fn):
            raise MissingTaskError(f"Missing llm task: {name}")
    return task_set
