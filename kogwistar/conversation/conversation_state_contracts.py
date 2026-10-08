from __future__ import annotations

from typing import Optional, TypedDict, cast
from typing_extensions import NotRequired
from pydantic import BaseModel, ConfigDict, Field

from ..engine_core.models import Span
from ..json_types import JsonValue
from .models import (
    ConversationAIResponse,
    KnowledgeRetrievalResult,
    MemoryPinResult,
    MemoryRetrievalResult,
)

class PrevTurnMetaSummaryModel(BaseModel):
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int
    tail_turn_index: int


class SummaryStateModel(BaseModel):
    should_summarize: bool = False
    did_summarize: bool = False
    summary_node_id: Optional[str] = None


class BudgetStateModel(BaseModel):
    token_budget: int = 0
    token_used: int = 0
    time_budget_ms: int = 0
    time_used_ms: int = 0
    cost_budget: float = 0.0
    cost_used: float = 0.0
    budget_kind: str = "token"
    budget_scope: str = "run"


class WorkflowStateModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conversation_id: str
    user_id: str
    turn_node_id: str
    turn_index: int
    mem_id: str
    self_span: Span

    role: str
    user_text: str
    embedding: list[float] | None = None

    memory: MemoryRetrievalResult | None = None
    memory_raw: object | None = None
    kg: KnowledgeRetrievalResult | None = None
    kg_raw: object | None = None
    memory_pin: MemoryPinResult | None = None
    memory_pin_raw: object | None = None
    kg_pin: JsonValue | None = None
    answer: ConversationAIResponse | None = None
    answer_raw: object | None = None

    summary: SummaryStateModel = Field(default_factory=SummaryStateModel)
    budget: BudgetStateModel = Field(default_factory=BudgetStateModel)
    prev_turn_meta_summary: PrevTurnMetaSummaryModel
    # Runtime dependencies are process-local and deliberately absent here.
    # Pydantic treats leading-underscore attributes as private, so they must be
    # attached after ``model_dump`` at the runtime admission boundary.

    def dump_state(self) -> ConversationWorkflowState:
        return cast(ConversationWorkflowState, self.model_dump(exclude=set(["_deps"])))


class ConversationPrevTurnMetaSummaryDict(TypedDict):
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int
    tail_turn_index: int


class ConversationSummaryStateDict(TypedDict):
    should_summarize: bool
    did_summarize: bool
    summary_node_id: Optional[str]


class ConversationBudgetStateDict(TypedDict):
    token_budget: int
    token_used: int
    time_budget_ms: int
    time_used_ms: int
    cost_budget: float
    cost_used: float
    budget_kind: str
    budget_scope: str


# ---- Persisted / checkpointed state (JSON-friendly) ----
class ConversationWorkflowState(TypedDict):
    """Narrow JSON-friendly persisted shape for conversation workflows.

    Generic runtime state is intentionally an open ``dict[str, object]`` because
    user workflows may define arbitrary state keys.  This TypedDict describes
    only conversation's persisted/checkpointed contract.
    """

    conversation_id: str
    user_id: str
    turn_node_id: NotRequired[str]
    turn_index: NotRequired[int]
    role: NotRequired[str]
    user_text: NotRequired[str]
    mem_id: NotRequired[str]
    self_span: NotRequired[Span]
    embedding: NotRequired[list[float] | None]
    memory: NotRequired[JsonValue | None]
    memory_raw: NotRequired[object]
    kg: NotRequired[JsonValue | None]
    memory_pin: NotRequired[JsonValue | None]
    kg_pin: NotRequired[JsonValue | None]
    answer: NotRequired[JsonValue | None]
    _rt_join: NotRequired[dict[str, JsonValue]]

    # identity
    # conversation_id: str
    # user_id: str
    # turn_node_id: str
    # turn_index: int
    # role: str
    # user_text: str

    # # required for pinning + tools
    # mem_id: str
    # self_span: Span
    summary: ConversationSummaryStateDict
    budget: ConversationBudgetStateDict
    prev_turn_meta_summary: ConversationPrevTurnMetaSummaryDict
    # _deps:dict
    # _rt_join:dict
