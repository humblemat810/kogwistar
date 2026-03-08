from __future__ import annotations

import json
import logging
import pathlib
import threading
from dataclasses import dataclass
from typing import Any, Callable

from graph_knowledge_engine.conversation.agentic_answering import AgenticAnsweringAgent
from graph_knowledge_engine.conversation.models import (
    ConversationNode,
    FilteringResult,
    MetaFromLastSummary,
)
from graph_knowledge_engine.conversation.service import ConversationService
from graph_knowledge_engine.id_provider import new_id_str
from graph_knowledge_engine.runtime.replay import load_checkpoint, replay_to
from graph_knowledge_engine.runtime.runtime import _get_shared_sqlite_sink
from graph_knowledge_engine.runtime.telemetry import EventEmitter

from .run_registry import RunRegistry, RunRegistryTraceBridge


class RunCancelledError(RuntimeError):
    """Raised when a submitted chat run is cancelled cooperatively."""


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


class ChatRunService:
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
        self.answer_runner = answer_runner or self._default_answer_runner
        self.runtime_runner = runtime_runner or self._default_runtime_runner

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

    def _publish(self, run_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.run_registry.append_event(run_id, event_type, payload)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))

    def create_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        start_node_id: str | None = None,
    ) -> dict[str, Any]:
        svc = self._conversation_service()
        conv_id, start_id = svc.create_conversation(user_id=user_id, conv_id=conversation_id, node_id=start_node_id)
        return self.get_conversation(conv_id) | {"start_node_id": start_id}

    def _conversation_nodes(self, conversation_id: str) -> list[ConversationNode]:
        nodes = self._conversation_engine().get_nodes(
            where={"conversation_id": conversation_id},
            node_type=ConversationNode,
            limit=10_000,
        )
        if not nodes:
            raise KeyError(f"Unknown conversation_id: {conversation_id}")
        return nodes

    def _conversation_owner(self, conversation_id: str) -> str | None:
        starts = [
            node
            for node in self._conversation_nodes(conversation_id)
            if str((getattr(node, "metadata", {}) or {}).get("entity_type") or "") == "conversation_start"
        ]
        if not starts:
            return None
        starts.sort(key=lambda node: int(getattr(node, "turn_index", -1) or -1))
        return str(getattr(starts[0], "user_id", None) or "")

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        nodes = self._conversation_nodes(conversation_id)
        svc = self._conversation_service()
        tail = svc.get_conversation_tail(conversation_id=conversation_id)
        starts = [
            node
            for node in nodes
            if str((getattr(node, "metadata", {}) or {}).get("entity_type") or "") == "conversation_start"
        ]
        turns = self.list_transcript(conversation_id)
        start_node = starts[0] if starts else None
        return {
            "conversation_id": conversation_id,
            "user_id": str(getattr(start_node, "user_id", None) or ""),
            "status": str((getattr(start_node, "properties", {}) or {}).get("status") or "active"),
            "start_node_id": str(getattr(start_node, "id", None) or ""),
            "tail_node_id": str(getattr(tail, "id", None) or ""),
            "turn_count": len(turns),
        }

    def list_transcript(self, conversation_id: str) -> list[dict[str, Any]]:
        nodes = self._conversation_nodes(conversation_id)
        turns: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            entity_type = str(metadata.get("entity_type") or "")
            if entity_type not in {"conversation_turn", "assistant_turn"}:
                continue
            turn_index = getattr(node, "turn_index", None)
            if turn_index is None:
                continue
            turns.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "turn_index": int(turn_index),
                    "role": str(getattr(node, "role", "") or ""),
                    "content": str(getattr(node, "summary", "") or ""),
                    "entity_type": entity_type,
                }
            )
        turns.sort(key=lambda item: (int(item["turn_index"]), str(item["node_id"])))
        return turns

    def latest_snapshot(
        self,
        conversation_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        svc = self._conversation_service()
        snap = svc.latest_context_snapshot_node(conversation_id=conversation_id, run_id=run_id, stage=stage)
        if snap is None:
            raise KeyError(f"No context snapshot found for conversation_id={conversation_id!r}")
        payload = svc.get_context_snapshot_payload(snapshot_node_id=str(snap.id))
        return {
            "snapshot_node_id": str(snap.id),
            "conversation_id": conversation_id,
            "metadata": payload.get("metadata") or {},
            "properties": payload.get("properties") or {},
        }

    def submit_turn_for_answer(
        self,
        *,
        conversation_id: str,
        user_id: str | None,
        text: str,
        workflow_id: str = "agentic_answering.v2",
    ) -> dict[str, Any]:
        text = str(text or "").strip()
        if not text:
            raise ValueError("text must be non-empty")

        resolved_user_id = str(user_id or self._conversation_owner(conversation_id) or "")
        if not resolved_user_id:
            raise ValueError("user_id is required for this conversation")

        svc = self._conversation_service()
        prev_turn_meta_summary = MetaFromLastSummary(0, 0)
        add_turn = svc.add_conversation_turn(
            user_id=resolved_user_id,
            conversation_id=conversation_id,
            turn_id=str(new_id_str()),
            mem_id=str(new_id_str()),
            role="user",
            content=text,
            ref_knowledge_engine=self._knowledge_engine(),
            filtering_callback=lambda *args, **kwargs: (FilteringResult(node_ids=[], edge_ids=[]), ""),
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only=True,
        )

        run_id = str(new_id_str())
        self.run_registry.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            user_id=resolved_user_id,
            user_turn_node_id=str(add_turn.user_turn_node_id),
            status="queued",
        )
        self._publish(
            run_id,
            "run.created",
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "workflow_id": workflow_id,
                "status": "queued",
                "user_turn_node_id": str(add_turn.user_turn_node_id),
            },
        )

        req = AnswerRunRequest(
            run_id=run_id,
            conversation_id=conversation_id,
            user_id=resolved_user_id,
            user_text=text,
            user_turn_node_id=str(add_turn.user_turn_node_id),
            workflow_id=workflow_id,
            knowledge_engine=self._knowledge_engine(),
            conversation_engine=self._conversation_engine(),
            workflow_engine=self._workflow_engine(),
            prev_turn_meta_summary=add_turn.prev_turn_meta_summary,
            registry=self.run_registry,
            publish=lambda event_type, payload=None: self._publish(run_id, event_type, payload),
            is_cancel_requested=lambda: self.run_registry.is_cancel_requested(run_id),
        )

        thread = threading.Thread(
            target=self._run_answer,
            args=(req,),
            daemon=True,
            name=f"chat-run-{run_id}",
        )
        thread.start()

        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "status": "queued",
            "user_turn_node_id": str(add_turn.user_turn_node_id),
        }

    def _output_chunks(self, text: str, *, chunk_size: int = 160) -> list[str]:
        if not text:
            return []
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    def _run_answer(self, req: AnswerRunRequest) -> None:
        self._publish(req.run_id, "run.started", {"run_id": req.run_id, "status": "running"})
        self.run_registry.update_status(req.run_id, status="running", started=True)
        try:
            out = self.answer_runner(req) or {}
            workflow_status = str(out.get("workflow_status") or "succeeded")
            if workflow_status == "cancelled":
                self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
                self.run_registry.update_status(req.run_id, status="cancelled", result=out, finished=True)
                return

            assistant_text = str(out.get("assistant_text") or "")
            for idx, chunk in enumerate(self._output_chunks(assistant_text)):
                self._publish(
                    req.run_id,
                    "output.delta",
                    {
                        "run_id": req.run_id,
                        "delta": chunk,
                        "chunk_index": idx,
                    },
                )
            self._publish(
                req.run_id,
                "output.completed",
                {
                    "run_id": req.run_id,
                    "assistant_text": assistant_text,
                    "assistant_turn_node_id": str(out.get("assistant_turn_node_id") or ""),
                },
            )
            self._publish(
                req.run_id,
                "run.completed",
                {
                    "run_id": req.run_id,
                    "status": "succeeded",
                    "assistant_turn_node_id": str(out.get("assistant_turn_node_id") or ""),
                },
            )
            self.run_registry.update_status(
                req.run_id,
                status="succeeded",
                assistant_turn_node_id=str(out.get("assistant_turn_node_id") or "") or None,
                result=out,
                finished=True,
            )
        except RunCancelledError:
            self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
            self.run_registry.update_status(req.run_id, status="cancelled", finished=True)
        except Exception as exc:
            err = {"message": str(exc)}
            logging.getLogger(__name__).exception("chat run failed: run_id=%s", req.run_id)
            self._publish(req.run_id, "run.failed", {"run_id": req.run_id, "status": "failed", "error": err})
            self.run_registry.update_status(req.run_id, status="failed", error=err, finished=True)

    def _default_answer_runner(self, req: AnswerRunRequest) -> dict[str, Any]:
        trace_db_path = pathlib.Path(str(getattr(req.workflow_engine, "persist_directory", "."))) / "wf_trace.sqlite"
        shared_sink = _get_shared_sqlite_sink(str(trace_db_path), drop_when_full=True)
        sink = RunRegistryTraceBridge(registry=req.registry, run_id=req.run_id, delegate=shared_sink)
        events = EventEmitter(sink=sink, logger=logging.getLogger("workflow.trace"))
        agent = AgenticAnsweringAgent(
            conversation_engine=req.conversation_engine,
            knowledge_engine=req.knowledge_engine,
            llm_tasks=req.conversation_engine.llm_tasks,
        )
        return agent.answer_workflow_v2(
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            prev_turn_meta_summary=req.prev_turn_meta_summary,
            workflow_engine=req.workflow_engine,
            workflow_id=req.workflow_id,
            run_id=req.run_id,
            events=events,
            trace=True,
            cancel_requested=lambda rid: req.is_cancel_requested(),
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
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        conversation_id = str(conversation_id or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id is required")

        # Validate conversation existence early.
        self._conversation_nodes(conversation_id)

        resolved_turn_node_id = str(turn_node_id or "").strip()
        if not resolved_turn_node_id:
            tail = self._conversation_service().get_conversation_tail(conversation_id=conversation_id)
            resolved_turn_node_id = str(getattr(tail, "id", None) or "").strip()
        if not resolved_turn_node_id:
            resolved_turn_node_id = f"wf_turn|{new_id_str()}"

        run_id = str(new_id_str())
        self.run_registry.create_run(
            run_id=run_id,
            conversation_id=conversation_id,
            workflow_id=workflow_id,
            user_id=(str(user_id) if user_id is not None else None),
            user_turn_node_id=resolved_turn_node_id,
            status="queued",
        )
        self._publish(
            run_id,
            "run.created",
            {
                "run_id": run_id,
                "run_kind": "workflow_runtime",
                "conversation_id": conversation_id,
                "workflow_id": workflow_id,
                "status": "queued",
                "turn_node_id": resolved_turn_node_id,
            },
        )

        req = RuntimeRunRequest(
            run_id=run_id,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=resolved_turn_node_id,
            user_id=(str(user_id) if user_id is not None else None),
            initial_state=dict(initial_state or {}),
            knowledge_engine=self._knowledge_engine(),
            conversation_engine=self._conversation_engine(),
            workflow_engine=self._workflow_engine(),
            registry=self.run_registry,
            publish=lambda event_type, payload=None: self._publish(run_id, event_type, payload),
            is_cancel_requested=lambda: self.run_registry.is_cancel_requested(run_id),
        )
        thread = threading.Thread(
            target=self._run_workflow,
            args=(req,),
            daemon=True,
            name=f"workflow-run-{run_id}",
        )
        thread.start()
        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "turn_node_id": resolved_turn_node_id,
            "status": "queued",
        }

    def _run_workflow(self, req: RuntimeRunRequest) -> None:
        self._publish(
            req.run_id,
            "run.started",
            {
                "run_id": req.run_id,
                "run_kind": "workflow_runtime",
                "status": "running",
            },
        )
        self.run_registry.update_status(req.run_id, status="running", started=True)
        try:
            out = self._json_safe(self.runtime_runner(req) or {})
            workflow_status = str(out.get("workflow_status") or out.get("status") or "succeeded")
            if workflow_status == "cancelled":
                self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
                self.run_registry.update_status(req.run_id, status="cancelled", result=out, finished=True)
                return
            if workflow_status in {"failed", "error"}:
                err = out.get("error")
                if not isinstance(err, dict):
                    err = {"message": f"Workflow runtime failed: status={workflow_status}"}
                self._publish(req.run_id, "run.failed", {"run_id": req.run_id, "status": "failed", "error": err})
                self.run_registry.update_status(req.run_id, status="failed", result=out, error=err, finished=True)
                return
            self._publish(req.run_id, "run.completed", {"run_id": req.run_id, "status": "succeeded"})
            self.run_registry.update_status(req.run_id, status="succeeded", result=out, finished=True)
        except RunCancelledError:
            self._publish(req.run_id, "run.cancelled", {"run_id": req.run_id, "status": "cancelled"})
            self.run_registry.update_status(req.run_id, status="cancelled", finished=True)
        except Exception as exc:
            err = {"message": str(exc)}
            logging.getLogger(__name__).exception("workflow run failed: run_id=%s", req.run_id)
            self._publish(req.run_id, "run.failed", {"run_id": req.run_id, "status": "failed", "error": err})
            self.run_registry.update_status(req.run_id, status="failed", error=err, finished=True)

    def _default_runtime_runner(self, req: RuntimeRunRequest) -> dict[str, Any]:
        from graph_knowledge_engine.conversation.resolvers import default_resolver
        from graph_knowledge_engine.runtime.runtime import WorkflowRuntime

        def predicate_always(_workflow_info, _state, _last_result):
            return True

        initial_state = dict(req.initial_state or {})
        deps = initial_state.get("_deps")
        if not isinstance(deps, dict):
            deps = {}
        deps.setdefault("conversation_engine", req.conversation_engine)
        deps.setdefault("knowledge_engine", req.knowledge_engine)
        deps.setdefault("ref_knowledge_engine", req.knowledge_engine)
        deps.setdefault("workflow_engine", req.workflow_engine)
        deps.setdefault("agentic_workflow_engine", req.workflow_engine)
        initial_state["_deps"] = deps

        runtime = WorkflowRuntime(
            workflow_engine=req.workflow_engine,
            conversation_engine=req.conversation_engine,
            step_resolver=default_resolver,
            predicate_registry={"always": predicate_always},
            checkpoint_every_n_steps=1,
            max_workers=1,
            cancel_requested=lambda _rid: req.is_cancel_requested(),
        )
        run_result = runtime.run(
            workflow_id=req.workflow_id,
            conversation_id=req.conversation_id,
            turn_node_id=req.turn_node_id,
            initial_state=initial_state,
            run_id=req.run_id,
        )
        final_state = self._json_safe(dict(getattr(run_result, "final_state", {}) or {}))
        final_state.pop("_deps", None)
        return {
            "workflow_status": str(getattr(run_result, "status", "succeeded") or "succeeded"),
            "final_state": final_state,
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.run_registry.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        steps = self.list_steps(run_id)
        if steps:
            run["last_step_seq"] = int(steps[-1]["step_seq"])
            run["step_count"] = len(steps)
        else:
            run["last_step_seq"] = None
            run["step_count"] = 0
        return run

    def list_run_events(self, run_id: str, *, after_seq: int = 0) -> list[dict[str, Any]]:
        if self.run_registry.get_run(run_id) is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self.run_registry.list_events(run_id, after_seq=after_seq)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        run = self.run_registry.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        if run["terminal"]:
            return run
        try:
            self._conversation_service().persist_workflow_cancel_request(
                conversation_id=str(run.get("conversation_id") or ""),
                run_id=str(run_id),
                workflow_id=str(run.get("workflow_id") or ""),
                requested_by="api",
                reason="api_cancel",
            )
        except Exception:
            logging.getLogger(__name__).exception("failed to persist cancel request node: run_id=%s", run_id)
        self._publish(run_id, "run.cancelling", {"run_id": run_id, "status": "cancelling"})
        return self.run_registry.request_cancel(run_id)

    def _workflow_nodes(self, *, entity_type: str, run_id: str) -> list[Any]:
        try:
            return self._conversation_engine().get_nodes(
                where={"$and": [{"entity_type": entity_type}, {"run_id": run_id}]},
                limit=200_000,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Nothing found on disk" in msg or "hnsw segment reader" in msg:
                return []
            raise

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        nodes = self._workflow_nodes(entity_type="workflow_step_exec", run_id=run_id)
        out: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            raw = metadata.get("result_json")
            out.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "step_seq": int(metadata.get("step_seq", 0) or 0),
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "workflow_node_id": str(metadata.get("workflow_node_id") or ""),
                    "op": str(metadata.get("op") or ""),
                    "status": str(metadata.get("status") or ""),
                    "duration_ms": int(metadata.get("duration_ms", 0) or 0),
                    "result": None if not raw else json.loads(str(raw)),
                }
            )
        out.sort(key=lambda item: int(item["step_seq"]))
        return out

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        nodes = self._workflow_nodes(entity_type="workflow_checkpoint", run_id=run_id)
        out: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            out.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "step_seq": int(metadata.get("step_seq", 0) or 0),
                    "workflow_id": str(metadata.get("workflow_id") or ""),
                    "state": json.loads(str(metadata.get("state_json") or "{}")),
                }
            )
        out.sort(key=lambda item: int(item["step_seq"]))
        return out

    def get_checkpoint(self, run_id: str, step_seq: int) -> dict[str, Any]:
        state = load_checkpoint(conversation_engine=self._conversation_engine(), run_id=run_id, step_seq=step_seq)
        return {
            "run_id": run_id,
            "step_seq": int(step_seq),
            "state": state,
        }

    def replay_run(self, run_id: str, target_step_seq: int) -> dict[str, Any]:
        state = replay_to(
            conversation_engine=self._conversation_engine(),
            run_id=run_id,
            target_step_seq=int(target_step_seq),
        )
        return {
            "run_id": run_id,
            "target_step_seq": int(target_step_seq),
            "state": state,
        }
