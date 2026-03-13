from __future__ import annotations

import contextlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from graph_knowledge_engine.conversation.models import MetaFromLastSummary
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine

from .run_registry import RunRegistry


class RunCancelledError(RuntimeError):
    """Raised when a submitted chat run is cancelled cooperatively."""


class WorkflowProjectionRebuildingError(RuntimeError):
    """Raised when a workflow design projection is being rebuilt."""


@dataclass
class AnswerRunRequest:
    run_id: str
    conversation_id: str
    user_id: str
    user_text: str
    user_turn_node_id: str
    workflow_id: str
    knowledge_engine: Any
    conversation_engine: Any
    workflow_engine: Any
    prev_turn_meta_summary: MetaFromLastSummary
    registry: RunRegistry
    publish: Callable[[str, dict[str, Any] | None], dict[str, Any]]
    is_cancel_requested: Callable[[], bool]


@dataclass
class RuntimeRunRequest:
    run_id: str
    workflow_id: str
    conversation_id: str
    turn_node_id: str
    user_id: str | None
    initial_state: dict[str, Any]
    knowledge_engine: Any
    conversation_engine: Any
    workflow_engine: Any
    registry: RunRegistry
    publish: Callable[[str, dict[str, Any] | None], dict[str, Any]]
    is_cancel_requested: Callable[[], bool]


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def now_ms() -> int:
    return int(time.time() * 1000)


def workflow_namespace(workflow_id: str) -> str:
    return f"wf_design:{str(workflow_id)}"


@runtime_checkable
class ChatRunServiceOwner(Protocol):
    _DESIGN_CONTROL_KIND: str
    _CTRL_UNDO_APPLIED: str
    _CTRL_REDO_APPLIED: str
    _CTRL_BRANCH_DROPPED: str
    _CTRL_MUTATION_COMMITTED: str
    _PROJECTION_SCHEMA_VERSION: int
    _SNAPSHOT_SCHEMA_VERSION: int
    _DELTA_SCHEMA_VERSION: int
    _SNAPSHOT_INTERVAL: int

    run_registry: RunRegistry
    answer_runner: Callable[[AnswerRunRequest], dict[str, Any]]
    runtime_runner: Callable[[RuntimeRunRequest], dict[str, Any]]
    _workflow_history_lock: threading.Lock

    def _knowledge_engine(self) -> Any: ...

    def _conversation_engine(self) -> Any: ...

    def _workflow_engine(self) -> Any: ...

    def _conversation_service(self) -> ConversationService: ...

    def _publish(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...

    @contextlib.contextmanager
    def _workflow_namespace_scope(self, workflow_id: str): ...


class _BaseComponent:
    """Base collaborator that forwards shared helpers to the ChatRunService facade."""

    def __init__(self, owner: ChatRunServiceOwner) -> None:
        self._owner = owner

    @property
    def run_registry(self) -> RunRegistry:
        return self._owner.run_registry

    @property
    def answer_runner(self) -> Callable[[AnswerRunRequest], dict[str, Any]]:
        return self._owner.answer_runner

    @property
    def runtime_runner(self) -> Callable[[RuntimeRunRequest], dict[str, Any]]:
        return self._owner.runtime_runner

    @property
    def _workflow_history_lock(self) -> threading.Lock:
        return self._owner._workflow_history_lock

    def _knowledge_engine(self) -> GraphKnowledgeEngine:
        return self._owner._knowledge_engine()

    def _conversation_engine(self) -> GraphKnowledgeEngine:
        return self._owner._conversation_engine()

    def _workflow_engine(self) -> GraphKnowledgeEngine:
        return self._owner._workflow_engine()

    def _conversation_service(self) -> ConversationService:
        return self._owner._conversation_service()

    def _publish(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._owner._publish(run_id, event_type, payload)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json_safe(value)

    @staticmethod
    def _now_ms() -> int:
        return now_ms()

    @staticmethod
    def _workflow_namespace(workflow_id: str) -> str:
        return workflow_namespace(workflow_id)

    @contextlib.contextmanager
    def _workflow_namespace_scope(self, workflow_id: str):
        with self._owner._workflow_namespace_scope(workflow_id) as eng:
            yield eng
