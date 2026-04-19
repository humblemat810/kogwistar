from __future__ import annotations

from typing import Any, Optional, TypedDict, cast
from pydantic import BaseModel, ConfigDict, Field

from ..engine_core.models import Span
from kogwistar.runtime.models import WorkflowState

Json = Any


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
    embedding: Any

    memory: Optional[Any] = None
    # memory_raw: Optional[Any] = None # not serializable, not used
    kg: Optional[Any] = None
    # kg_raw: Optional[Any] = None # not serializable, not used
    memory_pin: Optional[Any] = None
    # memory_pin_raw: Optional[Any] = None # not serializable, not used
    kg_pin: Optional[Any] = None
    answer: Optional[Any] = None
    # answer_raw: Optional[Any] = None # not serializable, not used

    summary: SummaryStateModel = Field(default_factory=SummaryStateModel)
    budget: BudgetStateModel = Field(default_factory=BudgetStateModel)
    prev_turn_meta_summary: PrevTurnMetaSummaryModel
    _deps: dict[str, Any]

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
class ConversationWorkflowState(WorkflowState):
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
    # embedding: Any  # List[float] ideally, but keep as Any if your embeddings vary

    # # step outputs (mirrors your step handlers)
    # memory: Optional[Json]
    # memory_raw: Optional[Any]  # not JSON; only safe if you accept non-serializable checkpoints
    # kg: Optional[Json]
    # # kg_raw: Optional[Any]
    # memory_pin: Optional[Json]
    # # memory_pin_raw: Optional[Any]
    # kg_pin: Optional[Json]
    # answer: Optional[Json]
    # # answer_raw: Optional[Any]

    summary: ConversationSummaryStateDict
    budget: ConversationBudgetStateDict
    prev_turn_meta_summary: ConversationPrevTurnMetaSummaryDict
    # _deps:dict
    # _rt_join:dict
