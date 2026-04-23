from __future__ import annotations

import asyncio
import logging
import pathlib
import threading
from typing import Any

from kogwistar.conversation.agentic_answering import AgenticAnsweringAgent
from kogwistar.conversation.models import (
    FilteringResult,
    MetaFromLastSummary,
)
from kogwistar.id_provider import new_id_str
from kogwistar.cdc.sqlite_sink import _get_shared_sqlite_sink
from kogwistar.runtime.telemetry import EventEmitter
from kogwistar.runtime.replay import load_checkpoint

from .chat_service_shared import (
    AnswerRunRequest,
    RunCancelledError,
    RuntimeRunRequest,
    RuntimeResumeRequest,
    _BaseComponent,
)
from .run_registry import RunRegistryTraceBridge


class _RunExecutionService(_BaseComponent):
    """Owns answer/workflow submission and run lifecycle operations."""

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

        resolved_user_id = str(
            user_id or self._conversation_owner(conversation_id) or ""
        )
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
            filtering_callback=lambda *args, **kwargs: (
                FilteringResult(node_ids=[], edge_ids=[]),
                "",
            ),
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
            publish=lambda event_type, payload=None: self._publish(
                run_id, event_type, payload
            ),
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
        self._publish(
            req.run_id, "run.started", {"run_id": req.run_id, "status": "running"}
        )
        self.run_registry.update_status(req.run_id, status="running", started=True)
        try:
            out = self.answer_runner(req) or {}
            workflow_status = str(out.get("workflow_status") or "succeeded")
            if workflow_status == "cancelled":
                self._publish(
                    req.run_id,
                    "run.cancelled",
                    {"run_id": req.run_id, "status": "cancelled"},
                )
                self.run_registry.update_status(
                    req.run_id, status="cancelled", result=out, finished=True
                )
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
                    "assistant_turn_node_id": str(
                        out.get("assistant_turn_node_id") or ""
                    ),
                },
            )
            self._publish(
                req.run_id,
                "run.completed",
                {
                    "run_id": req.run_id,
                    "status": "succeeded",
                    "assistant_turn_node_id": str(
                        out.get("assistant_turn_node_id") or ""
                    ),
                },
            )
            self.run_registry.update_status(
                req.run_id,
                status="succeeded",
                assistant_turn_node_id=str(out.get("assistant_turn_node_id") or "")
                or None,
                result=out,
                finished=True,
            )
        except RunCancelledError:
            self._publish(
                req.run_id,
                "run.cancelled",
                {"run_id": req.run_id, "status": "cancelled"},
            )
            self.run_registry.update_status(
                req.run_id, status="cancelled", finished=True
            )
        except Exception as exc:
            err = {"message": str(exc)}
            logging.getLogger(__name__).exception(
                "chat run failed: run_id=%s", req.run_id
            )
            self._publish(
                req.run_id,
                "run.failed",
                {"run_id": req.run_id, "status": "failed", "error": err},
            )
            self.run_registry.update_status(
                req.run_id, status="failed", error=err, finished=True
            )

    def _default_answer_runner(self, req: AnswerRunRequest) -> dict[str, Any]:
        trace_db_path = (
            pathlib.Path(str(getattr(req.workflow_engine, "persist_directory", ".")))
            / "wf_trace.sqlite"
        )
        shared_sink = _get_shared_sqlite_sink(str(trace_db_path), drop_when_full=True)
        sink = RunRegistryTraceBridge(
            registry=req.registry, run_id=req.run_id, delegate=shared_sink
        )
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
        priority_class: str = "foreground",
        token_budget: int | None = None,
        time_budget_ms: int | None = None,
        runtime_kind: str | None = None,
    ) -> dict[str, Any]:
        workflow_id = str(workflow_id or "").strip()
        if not workflow_id:
            raise ValueError("workflow_id is required")
        self._assert_workflow_projection_not_rebuilding(workflow_id=workflow_id)
        conversation_id = str(conversation_id or "").strip()
        if not conversation_id:
            raise ValueError("conversation_id is required")

        self._conversation_nodes(conversation_id)

        resolved_turn_node_id = str(turn_node_id or "").strip()
        if not resolved_turn_node_id:
            tail = self._conversation_service().get_conversation_tail(
                conversation_id=conversation_id
            )
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
            publish=lambda event_type, payload=None: self._publish(
                run_id, event_type, payload
            ),
            is_cancel_requested=lambda: self.run_registry.is_cancel_requested(run_id),
            priority_class=str(priority_class or "foreground"),
            token_budget=token_budget,
            time_budget_ms=time_budget_ms,
            capabilities=tuple(self._owner._effective_capabilities()),
            capability_subject=self._owner._capability_subject(),
            runtime_kind=str(runtime_kind or self._owner.default_runtime_kind or "sync"),
        )
        sched = self._owner.scheduler.submit(
            run_id=run_id,
            priority_class=str(priority_class or "foreground"),
            start_fn=lambda: self._run_workflow(req),
        )
        return {
            "run_id": run_id,
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "turn_node_id": resolved_turn_node_id,
            "status": "queued",
            "priority_class": str(priority_class or "foreground"),
            "token_budget": token_budget,
            "time_budget_ms": time_budget_ms,
            "admission": sched.get("admission", "accepted"),
            **({"reason": sched["reason"]} if "reason" in sched else {}),
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
            workflow_status = str(
                out.get("workflow_status") or out.get("status") or "succeeded"
            )
            if workflow_status == "cancelled":
                self._publish(
                    req.run_id,
                    "run.cancelled",
                    {"run_id": req.run_id, "status": "cancelled"},
                )
                self.run_registry.update_status(
                    req.run_id, status="cancelled", result=out, finished=True
                )
                return
            if workflow_status in {"failed", "error"}:
                err = out.get("error")
                if not isinstance(err, dict):
                    err = {
                        "message": f"Workflow runtime failed: status={workflow_status}"
                    }
                self._publish(
                    req.run_id,
                    "run.failed",
                    {"run_id": req.run_id, "status": "failed", "error": err},
                )
                self.run_registry.update_status(
                    req.run_id, status="failed", result=out, error=err, finished=True
                )
                return
            self._publish(
                req.run_id,
                "run.completed",
                {"run_id": req.run_id, "status": "succeeded"},
            )
            self.run_registry.update_status(
                req.run_id, status="succeeded", result=out, finished=True
            )
        except RunCancelledError:
            self._publish(
                req.run_id,
                "run.cancelled",
                {"run_id": req.run_id, "status": "cancelled"},
            )
            self.run_registry.update_status(
                req.run_id, status="cancelled", finished=True
            )
        except Exception as exc:
            err = {"message": str(exc)}
            logging.getLogger(__name__).exception(
                "workflow run failed: run_id=%s", req.run_id
            )
            self._publish(
                req.run_id,
                "run.failed",
                {"run_id": req.run_id, "status": "failed", "error": err},
            )
            self.run_registry.update_status(
                req.run_id, status="failed", error=err, finished=True
            )

    def _default_runtime_runner(self, req: RuntimeRunRequest) -> dict[str, Any]:
        from kogwistar.conversation.resolvers import default_resolver
        from kogwistar.runtime.budget import StateBackedBudgetLedger
        from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime
        from kogwistar.runtime.runtime import WorkflowRuntime

        def predicate_always(_workflow_info, _state, _last_result):
            return True

        initial_state = dict(req.initial_state or {})
        budget_state = initial_state.get("budget")
        if not isinstance(budget_state, dict):
            budget_state = {}
        if getattr(req, "token_budget", None) is not None:
            budget_state.setdefault("token_budget", int(req.token_budget or 0))
        if getattr(req, "time_budget_ms", None) is not None:
            budget_state.setdefault("time_budget_ms", int(req.time_budget_ms or 0))
        budget_state.setdefault("token_used", int(budget_state.get("token_used", 0) or 0))
        budget_state.setdefault("time_used_ms", int(budget_state.get("time_used_ms", 0) or 0))
        budget_state.setdefault("cost_budget", float(budget_state.get("cost_budget", 0.0) or 0.0))
        budget_state.setdefault("cost_used", float(budget_state.get("cost_used", 0.0) or 0.0))
        budget_state.setdefault("budget_kind", str(budget_state.get("budget_kind") or "token"))
        budget_state.setdefault("budget_scope", str(budget_state.get("budget_scope") or "run"))
        initial_state["budget"] = budget_state
        deps = initial_state.get("_deps")
        if not isinstance(deps, dict):
            deps = {}
        deps.setdefault("conversation_engine", req.conversation_engine)
        deps.setdefault("knowledge_engine", req.knowledge_engine)
        deps.setdefault("ref_knowledge_engine", req.knowledge_engine)
        deps.setdefault("workflow_engine", req.workflow_engine)
        deps.setdefault("agentic_workflow_engine", req.workflow_engine)
        deps.setdefault("capabilities", list(getattr(req, "capabilities", ()) or ()))
        deps.setdefault(
            "capability_subject",
            getattr(req, "capability_subject", None) or self._owner._capability_subject(),
        )
        if budget_state:
            deps.setdefault("budget_ledger", StateBackedBudgetLedger(budget_state))
        initial_state["_deps"] = deps

        runtime_kind = str(
            getattr(req, "runtime_kind", "") or self._owner.default_runtime_kind or "sync"
        ).strip().lower()
        if runtime_kind == "async":
            runtime = AsyncWorkflowRuntime(
                workflow_engine=req.workflow_engine,
                conversation_engine=req.conversation_engine,
                step_resolver=default_resolver,
                predicate_registry={"always": predicate_always},
                checkpoint_every_n_steps=1,
                max_workers=1,
                cancel_requested=lambda _rid: req.is_cancel_requested(),
                experimental_native_scheduler=True,
            )
            run_result = asyncio.run(
                runtime.run(
                    workflow_id=req.workflow_id,
                    conversation_id=req.conversation_id,
                    turn_node_id=req.turn_node_id,
                    initial_state=initial_state,
                    run_id=req.run_id,
                )
            )
        else:
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
        final_state = self._json_safe(
            dict(getattr(run_result, "final_state", {}) or {})
        )
        final_state.pop("_deps", None)
        return {
            "workflow_status": str(
                getattr(run_result, "status", "succeeded") or "succeeded"
            ),
            "final_state": final_state,
            "budget": final_state.get("budget", budget_state),
        }

    def _default_resume_runner(self, req: RuntimeResumeRequest) -> dict[str, Any]:
        from kogwistar.conversation.resolvers import default_resolver
        from kogwistar.runtime.budget import StateBackedBudgetLedger
        from kogwistar.runtime.models import RunFailure, RunSuccess, RunSuspended
        from kogwistar.runtime.async_runtime import AsyncWorkflowRuntime
        from kogwistar.runtime.runtime import WorkflowRuntime

        def predicate_always(_workflow_info, _state, _last_result):
            return True

        step_seq = int(req.client_result.get("step_seq", 0) or 0)
        initial_state = dict(
            load_checkpoint(
                conversation_engine=req.conversation_engine,
                run_id=req.run_id,
                step_seq=step_seq,
            )
        )
        budget_state = initial_state.get("budget")
        if not isinstance(budget_state, dict):
            budget_state = {}
        if getattr(req, "token_budget", None) is not None:
            budget_state.setdefault("token_budget", int(req.token_budget or 0))
        if getattr(req, "time_budget_ms", None) is not None:
            budget_state.setdefault("time_budget_ms", int(req.time_budget_ms or 0))
        budget_state.setdefault("token_used", int(budget_state.get("token_used", 0) or 0))
        budget_state.setdefault("time_used_ms", int(budget_state.get("time_used_ms", 0) or 0))
        budget_state.setdefault("cost_budget", float(budget_state.get("cost_budget", 0.0) or 0.0))
        budget_state.setdefault("cost_used", float(budget_state.get("cost_used", 0.0) or 0.0))
        budget_state.setdefault("budget_kind", str(budget_state.get("budget_kind") or "token"))
        budget_state.setdefault("budget_scope", str(budget_state.get("budget_scope") or "run"))
        initial_state["budget"] = budget_state
        deps = initial_state.get("_deps")
        if not isinstance(deps, dict):
            deps = {}
        deps.setdefault("conversation_engine", req.conversation_engine)
        deps.setdefault("knowledge_engine", req.knowledge_engine)
        deps.setdefault("ref_knowledge_engine", req.knowledge_engine)
        deps.setdefault("workflow_engine", req.workflow_engine)
        deps.setdefault("agentic_workflow_engine", req.workflow_engine)
        deps.setdefault("capabilities", list(getattr(req, "capabilities", ()) or ()))
        deps.setdefault(
            "capability_subject",
            getattr(req, "capability_subject", None) or self._owner._capability_subject(),
        )
        if budget_state:
            deps.setdefault("budget_ledger", StateBackedBudgetLedger(budget_state))
        initial_state["_deps"] = deps
        runtime_kind = str(
            getattr(req, "runtime_kind", "") or self._owner.default_runtime_kind or "sync"
        ).strip().lower()
        runtime_kwargs = dict(
            workflow_engine=req.workflow_engine,
            conversation_engine=req.conversation_engine,
            step_resolver=default_resolver,
            predicate_registry={"always": predicate_always},
            checkpoint_every_n_steps=1,
            max_workers=1,
            cancel_requested=lambda _rid: req.is_cancel_requested(),
        )
        if runtime_kind == "async":
            runtime = AsyncWorkflowRuntime(
                **runtime_kwargs,
                experimental_native_scheduler=True,
            )
        else:
            runtime = WorkflowRuntime(**runtime_kwargs)
        kind = str(req.client_result.get("status") or "success")
        model_map = {
            "success": RunSuccess,
            "failure": RunFailure,
            "suspended": RunSuspended,
        }
        model = model_map.get(kind, RunSuccess)
        client_result = model.model_validate(req.client_result)
        run_result = runtime.resume_run(
            run_id=req.run_id,
            suspended_node_id=req.suspended_node_id,
            suspended_token_id=req.suspended_token_id,
            client_result=client_result,
            workflow_id=req.workflow_id,
            conversation_id=req.conversation_id,
            turn_node_id=req.turn_node_id,
        )
        final_state = self._json_safe(dict(getattr(run_result, "final_state", {}) or {}))
        final_state.pop("_deps", None)
        return {
            "workflow_status": str(getattr(run_result, "status", "succeeded") or "succeeded"),
            "final_state": final_state,
            "budget": final_state.get("budget", budget_state),
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.run_registry.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return run

    def list_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        if self.run_registry.get_run(run_id) is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self.run_registry.list_events(
            run_id, after_seq=after_seq, limit=limit
        )

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
            logging.getLogger(__name__).exception(
                "failed to persist cancel request node: run_id=%s", run_id
            )
        self._publish(
            run_id, "run.cancelling", {"run_id": run_id, "status": "cancelling"}
        )
        return self.run_registry.request_cancel(run_id)
