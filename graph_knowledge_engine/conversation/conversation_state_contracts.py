from __future__ import annotations

from typing import Any, Optional, Callable, TypedDict, cast
from pydantic import BaseModel, ConfigDict, Field

from .models import MetaFromLastSummary
from .models import RetrievalResult
from ..engine_core.models import Span
Json = Any

class PrevTurnMetaSummaryModel(BaseModel):
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int
    tail_turn_index: int

class SummaryStateModel(BaseModel):
    should_summarize: bool = False
    did_summarize: bool = False
    summary_node_id: Optional[str] = None

import threading
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
    prev_turn_meta_summary: PrevTurnMetaSummaryModel
    _deps : dict[str,Any]
    def dump_state(self) -> WorkflowState:
        return cast(WorkflowState, self.model_dump(exclude=set(['_deps'])))
class PrevTurnMetaSummaryDict(TypedDict):
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int
    tail_turn_index: int

class SummaryStateDict(TypedDict):
    should_summarize: bool
    did_summarize: bool
    summary_node_id: Optional[str]


# ---- Persisted / checkpointed state (JSON-friendly) ----
class WorkflowState(TypedDict):
    # identity
    conversation_id: str
    user_id: str
    turn_node_id: str
    turn_index: int
    role: str
    user_text: str

    # required for pinning + tools
    mem_id: str
    self_span: Span
    embedding: Any  # List[float] ideally, but keep as Any if your embeddings vary

    # step outputs (mirrors your step handlers)
    memory: Optional[Json]
    memory_raw: Optional[Any]  # not JSON; only safe if you accept non-serializable checkpoints
    kg: Optional[Json]
    # kg_raw: Optional[Any]
    memory_pin: Optional[Json]
    # memory_pin_raw: Optional[Any]
    kg_pin: Optional[Json]
    answer: Optional[Json]
    # answer_raw: Optional[Any]

    summary: SummaryStateDict
    prev_turn_meta_summary: PrevTurnMetaSummaryDict
    _deps:dict
    _rt_join:dict