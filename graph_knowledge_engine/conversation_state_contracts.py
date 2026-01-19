from __future__ import annotations

from typing import Any, Optional, Callable, TypedDict
from pydantic import BaseModel, ConfigDict, Field

from graph_knowledge_engine.models import MetaFromLastSummary, RetrievalResult
from .models import Span
Json = Any

class PrevTurnMetaSummaryModel(BaseModel):
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int


class SummaryStateModel(BaseModel):
    should_summarize: bool = False
    did_summarize: bool = False
    summary_node_id: Optional[str] = None

import threading
class WorkflowStateModel(BaseModel):
    _lock: threading.Lock = Field(default_factory=threading.Lock)
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
    memory_raw: Optional[Any] = None
    kg: Optional[Any] = None
    kg_raw: Optional[Any] = None
    memory_pin: Optional[Any] = None
    memory_pin_raw: Optional[Any] = None
    kg_pin: Optional[Any] = None
    answer: Optional[Any] = None
    answer_raw: Optional[Any] = None

    summary: SummaryStateModel = Field(default_factory=SummaryStateModel)
    prev_turn_meta_summary: PrevTurnMetaSummaryModel
    # _deps : dict[str,Any]
    
class PrevTurnMetaSummaryDict(TypedDict):
    prev_node_char_distance_from_last_summary: int
    prev_node_distance_from_last_summary: int


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
    kg_raw: Optional[Any]
    memory_pin: Optional[Json]
    memory_pin_raw: Optional[Any]
    kg_pin: Optional[Json]
    answer: Optional[Json]
    answer_raw: Optional[Any]

    summary: SummaryStateDict
    prev_turn_meta_summary: PrevTurnMetaSummaryDict
    _deps:dict