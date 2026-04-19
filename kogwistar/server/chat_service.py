"""Top-level facade that composes the split chat and workflow service modules.

This module wires together conversation queries, run execution, run
inspection/replay, and workflow design services behind the public
``ChatRunService`` interface. It should stay orchestration-focused and delegate
specialized behavior to the narrower collaborator modules.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any, Callable

from kogwistar.conversation.models import ConversationNode
from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.models import Node

from .chat_service_conversation_queries import _ConversationQueryService
from .chat_service_run_execution import _RunExecutionService
from .chat_service_run_inspection import _RunInspectionService
from .chat_service_shared import (
    AnswerRunRequest,
    RunCancelledError,
    RuntimeRunRequest,
    RuntimeResumeRequest,
    WorkflowProjectionRebuildingError,
    json_safe,
    now_ms,
    workflow_namespace,
)
from .auth_middleware import (
    get_execution_namespace,
    get_security_scope,
    get_storage_namespace,
)
from .chat_service_workflow_design import _WorkflowDesignService
from .run_registry import RunRegistry


class ChatRunService:
    """Facade that composes workflow design, conversation, run execution, and replay collaborators."""

    _DESIGN_CONTROL_KIND = "design_control"
    _CTRL_UNDO_APPLIED = "UNDO_APPLIED"
    _CTRL_REDO_APPLIED = "REDO_APPLIED"
    _CTRL_BRANCH_DROPPED = "BRANCH_DROPPED"
    _CTRL_MUTATION_COMMITTED = "MUTATION_COMMITTED"
    _PROJECTION_SCHEMA_VERSION = 1
    _SNAPSHOT_SCHEMA_VERSION = 1
    _DELTA_SCHEMA_VERSION = 1
    _SNAPSHOT_INTERVAL = 50

    def __init__(
        self,
        *,
        get_knowledge_engine: Callable[[], Any],
        get_conversation_engine: Callable[[], Any],
        get_workflow_engine: Callable[[], Any],
        run_registry: RunRegistry,
        answer_runner: Callable[[AnswerRunRequest], dict[str, Any]] | None = None,
        runtime_runner: Callable[[RuntimeRunRequest], dict[str, Any]] | None = None,
    ) -> None:
        self._get_knowledge_engine = get_knowledge_engine
        self._get_conversation_engine = get_conversation_engine
        self._get_workflow_engine = get_workflow_engine
        self.run_registry = run_registry
        self._workflow_history_lock = threading.Lock()

        self._workflow_design = _WorkflowDesignService(self)
        self._conversation_queries = _ConversationQueryService(self)
        self._run_execution = _RunExecutionService(self)
        self._run_inspection = _RunInspectionService(self)

        self.answer_runner = answer_runner or self._run_execution._default_answer_runner
        self.runtime_runner = (
            runtime_runner or self._run_execution._default_runtime_runner
        )

    def _knowledge_engine(self) -> Any:
        return self._get_knowledge_engine()

    def _conversation_engine(self) -> Any:
        return self._get_conversation_engine()

    def _workflow_engine(self) -> Any:
        return self._get_workflow_engine()

    def _conversation_service(self) -> ConversationService:
        return ConversationService.from_engine(
            self._conversation_engine(),
            knowledge_engine=self._knowledge_engine(),
            workflow_engine=self._workflow_engine(),
        )

    def _publish(
        self, run_id: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.run_registry.append_event(run_id, event_type, payload)

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
        eng = self._workflow_engine()
        prev_ns = str(getattr(eng, "namespace", "default") or "default")
        target_ns = self._workflow_namespace(workflow_id)
        eng.namespace = target_ns
        try:
            yield eng
        finally:
            eng.namespace = prev_ns

    def _conversation_nodes(self, conversation_id: str) -> list[ConversationNode]:
        return self._conversation_queries._conversation_nodes(conversation_id)

    def _conversation_owner(self, conversation_id: str) -> str | None:
        return self._conversation_queries._conversation_owner(conversation_id)

    def _assert_workflow_projection_not_rebuilding(self, *, workflow_id: str) -> None:
        return self._workflow_design._assert_workflow_projection_not_rebuilding(
            workflow_id=workflow_id
        )

    def _workflow_capture_visible_snapshot(self, *, workflow_id: str) -> dict[str, Any]:
        return self._workflow_design._workflow_capture_visible_snapshot(
            workflow_id=workflow_id
        )

    def workflow_design_history(self, *, workflow_id: str) -> dict[str, Any]:
        return self._workflow_design.workflow_design_history(workflow_id=workflow_id)

    def refresh_workflow_design_projection(self, *, workflow_id: str) -> dict[str, Any]:
        return self._workflow_design.refresh_workflow_design_projection(
            workflow_id=workflow_id
        )

    def workflow_design_upsert_node(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        node_id: str | None,
        label: str,
        op: str | None = None,
        start: bool = False,
        terminal: bool = False,
        fanout: bool = False,
        metadata: dict[str, Any] | None = None,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        return self._workflow_design.workflow_design_upsert_node(
            workflow_id=workflow_id,
            designer_id=designer_id,
            node_id=node_id,
            label=label,
            op=op,
            start=start,
            terminal=terminal,
            fanout=fanout,
            metadata=metadata,
            actor_sub=actor_sub,
            source=source,
        )

    def workflow_design_upsert_edge(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        edge_id: str | None,
        src: str,
        dst: str,
        relation: str = "wf_next",
        predicate: str | None = None,
        priority: int = 100,
        is_default: bool = False,
        multiplicity: str = "one",
        metadata: dict[str, Any] | None = None,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        return self._workflow_design.workflow_design_upsert_edge(
            workflow_id=workflow_id,
            designer_id=designer_id,
            edge_id=edge_id,
            src=src,
            dst=dst,
            relation=relation,
            predicate=predicate,
            priority=priority,
            is_default=is_default,
            multiplicity=multiplicity,
            metadata=metadata,
            actor_sub=actor_sub,
            source=source,
        )

    def workflow_design_delete_node(
        self,
        *,
        workflow_id: str,
        node_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        return self._workflow_design.workflow_design_delete_node(
            workflow_id=workflow_id,
            node_id=node_id,
            designer_id=designer_id,
            actor_sub=actor_sub,
            source=source,
        )

    def workflow_design_delete_edge(
        self,
        *,
        workflow_id: str,
        edge_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        return self._workflow_design.workflow_design_delete_edge(
            workflow_id=workflow_id,
            edge_id=edge_id,
            designer_id=designer_id,
            actor_sub=actor_sub,
            source=source,
        )

    def workflow_design_undo(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        return self._workflow_design.workflow_design_undo(
            workflow_id=workflow_id,
            designer_id=designer_id,
            actor_sub=actor_sub,
            source=source,
        )

    def workflow_design_redo(
        self,
        *,
        workflow_id: str,
        designer_id: str,
        actor_sub: str | None = None,
        source: str = "rest",
    ) -> dict[str, Any]:
        return self._workflow_design.workflow_design_redo(
            workflow_id=workflow_id,
            designer_id=designer_id,
            actor_sub=actor_sub,
            source=source,
        )

    def create_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        start_node_id: str | None = None,
    ) -> dict[str, Any]:
        return self._conversation_queries.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            start_node_id=start_node_id,
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._conversation_queries.get_conversation(conversation_id)

    def list_transcript(self, conversation_id: str) -> list[dict[str, Any]]:
        return self._conversation_queries.list_transcript(conversation_id)

    def latest_snapshot(
        self,
        conversation_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        return self._conversation_queries.latest_snapshot(
            conversation_id, run_id=run_id, stage=stage
        )

    def submit_turn_for_answer(
        self,
        *,
        conversation_id: str,
        user_id: str | None,
        text: str,
        workflow_id: str = "agentic_answering.v2",
    ) -> dict[str, Any]:
        return self._run_execution.submit_turn_for_answer(
            conversation_id=conversation_id,
            user_id=user_id,
            text=text,
            workflow_id=workflow_id,
        )

    def submit_workflow_run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        initial_state: dict[str, Any] | None = None,
        turn_node_id: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        return self._run_execution.submit_workflow_run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            initial_state=initial_state,
            turn_node_id=turn_node_id,
            user_id=user_id,
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._run_execution.get_run(run_id)

    async def aget_run(self, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.get_run, run_id)

    def list_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        return self._run_execution.list_run_events(
            run_id, after_seq=after_seq, limit=limit
        )

    async def alist_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.list_run_events, run_id, after_seq=after_seq, limit=limit
        )

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._run_execution.cancel_run(run_id)

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        return self._run_inspection.list_steps(run_id)

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        return self._run_inspection.list_checkpoints(run_id)

    def get_checkpoint(self, run_id: str, step_seq: int) -> dict[str, Any]:
        return self._run_inspection.get_checkpoint(run_id, step_seq)

    def replay_run(self, run_id: str, target_step_seq: int) -> dict[str, Any]:
        return self._run_inspection.replay_run(run_id, target_step_seq)

    def resume_contract(self, run_id: str) -> dict[str, Any]:
        return self._run_inspection.resume_contract(run_id)

    def resume_run(
        self,
        *,
        run_id: str,
        suspended_node_id: str,
        suspended_token_id: str,
        client_result: dict[str, Any],
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        return self._run_execution._default_resume_runner(
            RuntimeResumeRequest(
                run_id=run_id,
                suspended_node_id=suspended_node_id,
                suspended_token_id=suspended_token_id,
                client_result=client_result,
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                user_id=user_id,
                knowledge_engine=self._knowledge_engine(),
                conversation_engine=self._conversation_engine(),
                workflow_engine=self._workflow_engine(),
                registry=self.run_registry,
                publish=lambda event_type, payload=None: self._publish(
                    run_id, event_type, payload
                ),
                is_cancel_requested=lambda: self.run_registry.is_cancel_requested(run_id),
            )
        )

    def workflow_design_graph(
        self, workflow_id: str, refresh: bool = False
    ) -> dict[str, object]:
        return self._workflow_design.workflow_design_graph(workflow_id, refresh)

    def workflow_catalog_ops(self) -> list[dict[str, object]]:
        return self._workflow_design.workflow_catalog_ops()

    def _execution_meta_store(self) -> Any:
        conversation_engine = self._conversation_engine()
        meta_store = getattr(conversation_engine, "meta_sqlite", None)
        if meta_store is None:
            raise AttributeError("conversation_engine does not expose meta_sqlite")
        return meta_store

    @staticmethod
    def _scope_snapshot() -> dict[str, str]:
        return {
            "storage_namespace": get_storage_namespace(),
            "execution_namespace": get_execution_namespace(),
            "security_scope": get_security_scope(),
        }

    def list_process_table(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        meta_store = self._execution_meta_store()
        list_runs = getattr(meta_store, "list_server_runs", None)
        runs = []
        if callable(list_runs):
            runs = list_runs(
                status=status,
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                limit=limit,
            )
        out: list[dict[str, Any]] = []
        scope = self._scope_snapshot()
        for run in runs:
            run_id = str(run.get("run_id") or "")
            events = []
            list_events = getattr(meta_store, "list_server_run_events", None)
            if callable(list_events):
                events = list_events(run_id, after_seq=0, limit=50)
            last_event = events[-1] if events else None
            status_val = str(run.get("status") or "")
            out.append(
                {
                    "process_id": run_id,
                    "process_kind": "workflow_run",
                    "status": status_val,
                    "workflow_id": run.get("workflow_id"),
                    "conversation_id": run.get("conversation_id"),
                    "user_id": run.get("user_id"),
                    "user_turn_node_id": run.get("user_turn_node_id"),
                    "assistant_turn_node_id": run.get("assistant_turn_node_id"),
                    "created_at_ms": run.get("created_at_ms"),
                    "updated_at_ms": run.get("updated_at_ms"),
                    "started_at_ms": run.get("started_at_ms"),
                    "finished_at_ms": run.get("finished_at_ms"),
                    "terminal": bool(run.get("terminal")),
                    "event_count": len(events),
                    "last_event_type": None if last_event is None else last_event.get("event_type"),
                    "last_event_seq": None if last_event is None else last_event.get("seq"),
                    "owner": run.get("user_id"),
                    "namespace": scope["execution_namespace"],
                    "storage_namespace": scope["storage_namespace"],
                    "security_scope": scope["security_scope"],
                }
            )
        return out

    def list_operator_inbox(
        self,
        *,
        inbox_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        meta_store = self._execution_meta_store()
        list_fn = getattr(meta_store, "list_projected_lane_messages", None)
        if not callable(list_fn):
            return []
        rows = list_fn(
            namespace=self._scope_snapshot()["storage_namespace"],
            inbox_id=inbox_id,
            status=status,
        )
        rows = rows[: int(limit)]
        scope = self._scope_snapshot()
        return [
            {
                "message_id": row.message_id,
                "inbox_id": row.inbox_id,
                "conversation_id": row.conversation_id,
                "recipient_id": row.recipient_id,
                "sender_id": row.sender_id,
                "msg_type": row.msg_type,
                "status": row.status,
                "seq": row.seq,
                "conversation_seq": row.conversation_seq,
                "claimed_by": row.claimed_by,
                "lease_until": row.lease_until,
                "retry_count": row.retry_count,
                "created_at": row.created_at,
                "available_at": row.available_at,
                "run_id": row.run_id,
                "step_id": row.step_id,
                "correlation_id": row.correlation_id,
                "storage_namespace": scope["storage_namespace"],
                "execution_namespace": scope["execution_namespace"],
                "security_scope": scope["security_scope"],
            }
            for row in rows
        ]

    def list_blocked_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.list_process_table(limit=limit)
        blocked_statuses = {"blocked", "waiting", "suspended", "cancelling"}
        out = [row for row in rows if str(row.get("status") or "") in blocked_statuses]
        return out[: int(limit)]

    def list_process_timeline(
        self, *, run_id: str, after_seq: int = 0, limit: int = 200
    ) -> list[dict[str, Any]]:
        meta_store = self._execution_meta_store()
        list_events = getattr(meta_store, "list_server_run_events", None)
        if not callable(list_events):
            return []
        return list_events(str(run_id), after_seq=int(after_seq), limit=int(limit))


__all__ = [
    "AnswerRunRequest",
    "ChatRunService",
    "RunCancelledError",
    "RuntimeRunRequest",
    "WorkflowProjectionRebuildingError",
]
