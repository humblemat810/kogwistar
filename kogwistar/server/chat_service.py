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
from .capability_kernel import CapabilityKernel, DEFAULT_CAPABILITY_SPECS
from .auth_middleware import (
    can_access_security_scope,
    describe_storage_security_mapping,
    get_current_agent_id,
    get_current_capabilities,
    get_current_role,
    get_execution_namespace,
    get_current_subject,
    get_current_user_id,
    has_explicit_capabilities_claim,
    get_security_scope,
    get_storage_namespace,
)
from .chat_service_workflow_design import _WorkflowDesignService
from .run_scheduler import RunScheduler
from .run_registry import RunRegistry
from .service_daemon import ServiceSupervisor
from kogwistar.runtime.cost_ledger import CostLedger


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
        self.scheduler = RunScheduler(max_active=2)

        self._workflow_design = _WorkflowDesignService(self)
        self._conversation_queries = _ConversationQueryService(self)
        self._run_execution = _RunExecutionService(self)
        self._run_inspection = _RunInspectionService(self)
        self.cost_ledger = CostLedger(workspace_id="chat-service")
        self.capability_kernel = CapabilityKernel()
        for spec in DEFAULT_CAPABILITY_SPECS:
            self.capability_kernel.register(spec)
        self.service_supervisor = ServiceSupervisor(
            get_workflow_engine=self._get_workflow_engine,
            get_conversation_engine=self._get_conversation_engine,
            run_registry=self.run_registry,
            spawn_workflow_run=lambda **kwargs: self._run_execution.submit_workflow_run(
                **kwargs
            ),
            scope_snapshot=self._scope_snapshot,
        )

        self.answer_runner = answer_runner or self._run_execution._default_answer_runner
        self.runtime_runner = (
            runtime_runner or self._run_execution._default_runtime_runner
        )
        self.service_supervisor.bootstrap()

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

    def _capability_subject(self) -> str:
        return (
            get_current_subject()
            or get_current_user_id()
            or get_current_agent_id()
            or "anonymous"
        )

    def _legacy_capability_grants(self) -> set[str]:
        role = str(get_current_role() or "ro").lower()
        if role == "rw":
            return {spec.name for spec in self.capability_kernel.list_specs()}
        return {
            "read_graph",
            "read_security_scope",
            "project_view",
            "workflow.design.inspect",
            "workflow.run.read",
        }

    def _effective_capabilities(self) -> tuple[str, ...]:
        caps = get_current_capabilities()
        if caps or has_explicit_capabilities_claim():
            base_caps: tuple[str, ...] = tuple(sorted(caps))
        else:
            base_caps = tuple(sorted(self._legacy_capability_grants()))
        return self.capability_kernel.materialize_capabilities(
            subject=self._capability_subject(),
            parent_capabilities=base_caps,
        )

    def _require_capability(
        self,
        action: str,
        required: str | list[str] | set[str] | tuple[str, ...],
        *,
        approval_message: str | None = None,
    ) -> dict[str, Any]:
        decision = self.capability_kernel.require(
            subject=self._capability_subject(),
            action=action,
            required=required,
            parent_capabilities=self._effective_capabilities(),
            approval_message=approval_message,
        )
        return {
            "ts_ms": decision.ts_ms,
            "subject": decision.subject,
            "action": decision.action,
            "required": list(decision.required),
            "granted": list(decision.granted),
            "outcome": decision.outcome,
            "reason": decision.reason,
            "parent_capabilities": list(decision.parent_capabilities),
        }

    def approve_action(
        self,
        *,
        subject: str | None = None,
        action: str,
        capabilities: str | list[str] | tuple[str, ...],
    ) -> dict[str, Any]:
        owner = str(subject or self._capability_subject()).strip().lower()
        self.capability_kernel.grant(
            subject=owner, action=action, capabilities=capabilities
        )
        return self.capability_snapshot()

    def revoke_capability(
        self,
        *,
        subject: str | None = None,
        capability: str,
    ) -> dict[str, Any]:
        owner = str(subject or self._capability_subject()).strip().lower()
        self.capability_kernel.revoke(subject=owner, capability=capability)
        return self.capability_snapshot()

    def capability_snapshot(self) -> dict[str, Any]:
        snap = self.capability_kernel.snapshot()
        snap["current_subject"] = self._capability_subject()
        snap["effective_capabilities"] = list(self._effective_capabilities())
        return snap

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
        self._require_capability(
            "workflow.design.inspect",
            ["workflow.design.inspect"],
            approval_message="Inspecting workflow design requires workflow.design.inspect capability",
        )
        return self._workflow_design.workflow_design_history(workflow_id=workflow_id)

    def refresh_workflow_design_projection(self, *, workflow_id: str) -> dict[str, Any]:
        self._require_capability(
            "workflow.design.inspect",
            ["workflow.design.inspect"],
            approval_message="Refreshing workflow design projection requires workflow.design.inspect capability",
        )
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
        self._require_capability(
            "workflow.design.write",
            ["workflow.design.write", "write_graph"],
            approval_message="Upserting workflow nodes requires workflow.design.write capability",
        )
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
        self._require_capability(
            "workflow.design.write",
            ["workflow.design.write", "write_graph"],
            approval_message="Upserting workflow edges requires workflow.design.write capability",
        )
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
        self._require_capability(
            "workflow.design.write",
            ["workflow.design.write", "write_graph"],
            approval_message="Deleting workflow nodes requires workflow.design.write capability",
        )
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
        self._require_capability(
            "workflow.design.write",
            ["workflow.design.write", "write_graph"],
            approval_message="Deleting workflow edges requires workflow.design.write capability",
        )
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
        self._require_capability(
            "workflow.design.write",
            ["workflow.design.write", "approve_action"],
            approval_message="Undoing workflow design requires workflow.design.write capability",
        )
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
        self._require_capability(
            "workflow.design.write",
            ["workflow.design.write", "approve_action"],
            approval_message="Redoing workflow design requires workflow.design.write capability",
        )
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
        self._require_capability(
            "create_conversation",
            ["spawn_process"],
            approval_message="Creating conversation requires spawn_process capability",
        )
        return self._conversation_queries.create_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            start_node_id=start_node_id,
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        self._require_capability(
            "read_conversation",
            ["read_graph"],
            approval_message="Reading conversation requires read_graph capability",
        )
        return self._conversation_queries.get_conversation(conversation_id)

    def list_transcript(self, conversation_id: str) -> list[dict[str, Any]]:
        self._require_capability(
            "read_conversation",
            ["read_graph"],
            approval_message="Reading transcript requires read_graph capability",
        )
        return self._conversation_queries.list_transcript(conversation_id)

    def latest_snapshot(
        self,
        conversation_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading snapshot requires project_view capability",
        )
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
        self._require_capability(
            "send_message",
            ["send_message"],
            approval_message="Sending message requires send_message capability",
        )
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
        priority_class: str = "foreground",
        token_budget: int | None = None,
        time_budget_ms: int | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "spawn_process",
            ["spawn_process", "workflow.run.write"],
            approval_message="Submitting workflow run requires spawn_process and workflow.run.write capabilities",
        )
        return self._run_execution.submit_workflow_run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            initial_state=initial_state,
            turn_node_id=turn_node_id,
            user_id=user_id,
            priority_class=priority_class,
            token_budget=token_budget,
            time_budget_ms=time_budget_ms,
        )

    def get_run(self, run_id: str) -> dict[str, Any]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading run requires workflow.run.read capability",
        )
        return self._run_execution.get_run(run_id)

    async def aget_run(self, run_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.get_run, run_id)

    def list_run_events(
        self, run_id: str, *, after_seq: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading run events requires workflow.run.read capability",
        )
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
        self._require_capability(
            "workflow.run.write",
            ["workflow.run.write"],
            approval_message="Cancelling run requires workflow.run.write capability",
        )
        return self._run_execution.cancel_run(run_id)

    def resource_snapshot(self) -> dict[str, Any]:
        self._require_capability(
            "read_security_scope",
            ["read_security_scope", "project_view"],
            approval_message="Reading resource snapshot requires read_security_scope capability",
        )
        def _dir_size(path: str | None) -> int:
            import os

            if not path:
                return 0
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
            return total

        engines = [
            self._knowledge_engine(),
            self._conversation_engine(),
            self._workflow_engine(),
        ]
        storage_usage_bytes = 0
        for eng in engines:
            storage_usage_bytes += _dir_size(str(getattr(eng, "persist_directory", "") or ""))
        return {
            "scheduler": self.scheduler.snapshot(),
            "runs": self.run_registry.snapshot(),
            "services": {
                "total_services": len(self.service_supervisor.list_services(limit=10_000)),
                "by_health": self._service_counts_by_health(),
            },
            "storage_usage_bytes": storage_usage_bytes,
            "cost_ledger": self.cost_ledger.snapshot(),
            "budget_model": {
                "token_kind": "token",
                "time_kind": "ms",
                "storage_kind": "bytes",
            },
            "policy_infra": {
                "cpu_millicores": None,
                "memory_mb": None,
            },
        }

    def visibility_snapshot(self) -> dict[str, Any]:
        self._require_capability(
            "read_security_scope",
            ["read_security_scope", "project_view"],
            approval_message="Reading visibility snapshot requires read_security_scope capability",
        )
        scope = self._scope_snapshot()
        mapping = describe_storage_security_mapping()
        return {
            "current_subject": self._capability_subject(),
            "current_role": get_current_role(),
            "current_capabilities": list(self._effective_capabilities()),
            "namespaces": {
                "storage_namespace": scope["storage_namespace"],
                "execution_namespace": scope["execution_namespace"],
            },
            "security_scope": scope["security_scope"],
            "storage_security_mapping": mapping,
            "can_access_public": can_access_security_scope("public", shared=True),
        }

    def capability_snapshot(self) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Inspecting capabilities requires project_view capability",
        )
        return self.capability_kernel.snapshot() | {
            "current_subject": self._capability_subject(),
            "effective_capabilities": list(self._effective_capabilities()),
        }

    def capability_approve(
        self,
        *,
        action: str,
        capabilities: str | list[str] | tuple[str, ...],
        subject: str | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "approve_action",
            ["approve_action"],
            approval_message="Approving capability requires approve_action capability",
        )
        return self.approve_action(
            subject=subject or self._capability_subject(),
            action=action,
            capabilities=capabilities,
        )

    def capability_revoke(
        self,
        *,
        capability: str,
        subject: str | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "approve_action",
            ["approve_action"],
            approval_message="Revoking capability requires approve_action capability",
        )
        return self.revoke_capability(subject=subject, capability=capability)

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading workflow steps requires workflow.run.read capability",
        )
        return self._run_inspection.list_steps(run_id)

    def list_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading workflow checkpoints requires workflow.run.read capability",
        )
        return self._run_inspection.list_checkpoints(run_id)

    def get_checkpoint(self, run_id: str, step_seq: int) -> dict[str, Any]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading workflow checkpoint requires workflow.run.read capability",
        )
        return self._run_inspection.get_checkpoint(run_id, step_seq)

    def replay_run(self, run_id: str, target_step_seq: int) -> dict[str, Any]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Replaying workflow run requires workflow.run.read capability",
        )
        return self._run_inspection.replay_run(run_id, target_step_seq)

    def resume_contract(self, run_id: str) -> dict[str, Any]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading resume contract requires workflow.run.read capability",
        )
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
        self._require_capability(
            "workflow.run.write",
            ["workflow.run.write", "approve_action"],
            approval_message="Resuming workflow run requires workflow.run.write capability",
        )
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
                capabilities=tuple(self._effective_capabilities()),
                capability_subject=self._capability_subject(),
            )
        )

    def workflow_design_graph(
        self, workflow_id: str, refresh: bool = False
    ) -> dict[str, object]:
        self._require_capability(
            "read_graph",
            ["read_graph", "workflow.design.inspect"],
            approval_message="Reading workflow graph requires read_graph capability",
        )
        return self._workflow_design.workflow_design_graph(workflow_id, refresh)

    def workflow_catalog_ops(self) -> list[dict[str, object]]:
        self._require_capability(
            "workflow.design.inspect",
            ["workflow.design.inspect"],
            approval_message="Inspecting workflow catalog requires workflow.design.inspect capability",
        )
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

    def _service_counts_by_health(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.service_supervisor.list_services(limit=10_000):
            health = str(row.get("health_status") or "unknown")
            counts[health] = counts.get(health, 0) + 1
        return counts

    def declare_service(
        self,
        *,
        service_id: str,
        service_kind: str,
        target_kind: str,
        target_ref: str,
        target_config: dict[str, Any] | None = None,
        enabled: bool = True,
        autostart: bool = False,
        restart_policy: dict[str, Any] | None = None,
        heartbeat_ttl_ms: int = 60_000,
        trigger_specs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Declaring service requires service.manage capability",
        )
        return self.service_supervisor.declare_service(
            service_id=service_id,
            service_kind=service_kind,
            target_kind=target_kind,
            target_ref=target_ref,
            target_config=target_config,
            enabled=enabled,
            autostart=autostart,
            restart_policy=restart_policy,
            heartbeat_ttl_ms=heartbeat_ttl_ms,
            trigger_specs=trigger_specs,
        )

    def get_service(self, service_id: str) -> dict[str, Any]:
        self._require_capability(
            "service.inspect",
            ["service.inspect", "project_view"],
            approval_message="Inspecting service requires service.inspect capability",
        )
        return self.service_supervisor.get_service(service_id)

    def list_services(self, *, limit: int = 200) -> list[dict[str, Any]]:
        self._require_capability(
            "service.inspect",
            ["service.inspect", "project_view"],
            approval_message="Listing services requires service.inspect capability",
        )
        return self.service_supervisor.list_services(limit=limit)

    def enable_service(self, service_id: str) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Enabling service requires service.manage capability",
        )
        return self.service_supervisor.enable_service(service_id)

    def disable_service(self, service_id: str) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Disabling service requires service.manage capability",
        )
        return self.service_supervisor.disable_service(service_id)

    def record_service_heartbeat(
        self,
        service_id: str,
        *,
        instance_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "service.heartbeat",
            ["service.heartbeat", "service.manage"],
            approval_message="Recording service heartbeat requires service.heartbeat capability",
        )
        return self.service_supervisor.record_service_heartbeat(
            service_id,
            instance_id=instance_id,
            payload=payload,
        )

    def trigger_service(
        self,
        service_id: str,
        *,
        trigger_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Triggering service requires service.manage capability",
        )
        return self.service_supervisor.trigger_service(
            service_id,
            trigger_type=trigger_type,
            payload=payload,
        )

    def list_service_events(
        self, service_id: str, *, limit: int = 500
    ) -> list[dict[str, Any]]:
        self._require_capability(
            "service.inspect",
            ["service.inspect", "project_view"],
            approval_message="Reading service events requires service.inspect capability",
        )
        return self.service_supervisor.list_service_events(service_id, limit=limit)

    def list_process_table(
        self,
        *,
        status: str | None = None,
        workflow_id: str | None = None,
        conversation_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._require_capability(
            "project_view",
            ["project_view", "workflow.run.read"],
            approval_message="Reading process table requires project_view capability",
        )
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
        for service in self.service_supervisor.list_services(limit=limit):
            child_run_id = str(service.get("current_child_run_id") or "")
            if workflow_id and str(service.get("target_ref") or "") != str(workflow_id):
                continue
            if conversation_id:
                target_cfg = dict(service.get("target_config") or {})
                if str(target_cfg.get("conversation_id") or "") != str(conversation_id):
                    continue
            service_status = str(service.get("lifecycle_status") or "")
            if status and service_status != str(status):
                continue
            out.append(
                {
                    "process_id": str(service.get("service_id") or ""),
                    "process_kind": "service",
                    "status": service_status,
                    "workflow_id": service.get("target_ref")
                    if str(service.get("target_kind") or "") == "workflow"
                    else None,
                    "conversation_id": dict(service.get("target_config") or {}).get(
                        "conversation_id"
                    ),
                    "user_id": None,
                    "user_turn_node_id": None,
                    "assistant_turn_node_id": None,
                    "created_at_ms": service.get("created_at_ms"),
                    "updated_at_ms": service.get("updated_at_ms"),
                    "started_at_ms": service.get("current_child_started_at_ms"),
                    "finished_at_ms": None,
                    "terminal": not bool(service.get("enabled")),
                    "event_count": len(
                        self.service_supervisor.list_service_events(
                            str(service.get("service_id") or ""), limit=200
                        )
                    ),
                    "last_event_type": None,
                    "last_event_seq": None,
                    "owner": None,
                    "namespace": service.get("execution_namespace"),
                    "storage_namespace": service.get("storage_namespace"),
                    "security_scope": service.get("security_scope"),
                    "service_kind": service.get("service_kind"),
                    "health_status": service.get("health_status"),
                    "last_heartbeat_ms": service.get("last_heartbeat_ms"),
                    "restart_count": service.get("restart_count"),
                    "last_trigger_type": service.get("last_trigger_type"),
                    "current_child_run_id": child_run_id or None,
                    "current_child_status": service.get("current_child_status"),
                }
            )
        out.sort(
            key=lambda row: (
                int(row.get("updated_at_ms") or 0),
                str(row.get("process_kind") or ""),
                str(row.get("process_id") or ""),
            ),
            reverse=True,
        )
        return out

    def list_operator_inbox(
        self,
        *,
        inbox_id: str | None = None,
        status: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading operator inbox requires project_view capability",
        )
        meta_store = self._execution_meta_store()
        list_fn = getattr(meta_store, "list_projected_lane_messages", None)
        if not callable(list_fn):
            return []
        rows = list_fn(
            namespace=self._scope_snapshot()["storage_namespace"],
            inbox_id=inbox_id,
            status=status,
        )
        visible_rows = []
        for row in rows:
            nodes = self._conversation_engine().read.get_nodes(ids=[row.message_id])
            if not nodes:
                continue
            md = dict(getattr(nodes[0], "metadata", {}) or {})
            if not can_access_security_scope(
                str(md.get("security_scope") or ""),
                shared=bool(md.get("shared_scope") or md.get("shared_inbox")),
            ):
                continue
            visible_rows.append((row, md))
        visible_rows = visible_rows[: int(limit)]
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
                "message_security_scope": str(md.get("security_scope") or ""),
                "visibility": str(md.get("visibility") or "private"),
                "storage_namespace": scope["storage_namespace"],
                "execution_namespace": scope["execution_namespace"],
                "security_scope": scope["security_scope"],
            }
            for row, md in visible_rows
        ]

    def list_blocked_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        self._require_capability(
            "project_view",
            ["project_view", "workflow.run.read"],
            approval_message="Reading blocked runs requires project_view capability",
        )
        rows = self.list_process_table(limit=limit)
        blocked_statuses = {"blocked", "waiting", "suspended", "cancelling"}
        out = [row for row in rows if str(row.get("status") or "") in blocked_statuses]
        return out[: int(limit)]

    def repair_service_projection(self, service_id: str) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Repairing service projection requires service.manage capability",
        )
        self.service_supervisor._rebuild_projection(service_id)
        return {"service_id": service_id, "projection": self.get_service(service_id)}

    def repair_service_projections(self, *, limit: int = 10_000) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Repairing service projections requires service.manage capability",
        )
        repaired = []
        for row in self.service_supervisor.list_services(limit=limit):
            service_id = str(row.get("service_id") or "")
            if not service_id:
                continue
            self.service_supervisor._rebuild_projection(service_id)
            repaired.append(service_id)
        return {"repaired_service_ids": repaired}

    def repair_orphaned_claimed_messages(
        self,
        *,
        inbox_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Repairing claimed messages requires project_view capability",
        )
        meta_store = self._execution_meta_store()
        list_fn = getattr(meta_store, "list_projected_lane_messages", None)
        update_fn = getattr(meta_store, "update_projected_lane_message_status", None)
        if not callable(list_fn) or not callable(update_fn):
            return {"repaired_message_ids": []}
        scope = self._scope_snapshot()["storage_namespace"]
        rows = list_fn(namespace=scope, inbox_id=inbox_id, status="claimed")
        repaired: list[str] = []
        now = now_ms()
        for row in rows:
            lease_until = int(getattr(row, "lease_until", 0) or 0)
            if lease_until and lease_until > now:
                continue
            update_fn(message_id=row.message_id, status="pending")
            repaired.append(str(row.message_id))
            if len(repaired) >= int(limit):
                break
        return {"repaired_message_ids": repaired}

    def replay_run_history(
        self, run_id: str, *, after_seq: int = 0, limit: int = 200
    ) -> dict[str, Any]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Replaying run history requires workflow.run.read capability",
        )
        events = self.list_process_timeline(run_id=run_id, after_seq=after_seq, limit=limit)
        return {"run_id": run_id, "events": events, "event_count": len(events)}

    def dead_letter_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading dead letters requires project_view capability",
        )
        return {"runs": self.scheduler.dead_letter()[: int(limit)], "limit": int(limit)}

    def replay_dead_letter(self, run_id: str) -> dict[str, Any]:
        self._require_capability(
            "service.manage",
            ["service.manage"],
            approval_message="Replaying dead letters requires service.manage capability",
        )
        dead = [row for row in self.scheduler.dead_letter() if str(row.get("run_id") or "") == str(run_id)]
        if not dead:
            return {"run_id": run_id, "replayed": False}
        self.scheduler.resume(run_id)
        return {"run_id": run_id, "replayed": True, "dead_letter": dead[0]}

    def message_queue_snapshot(self, *, inbox_id: str | None = None) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading queue snapshot requires project_view capability",
        )
        meta_store = self._execution_meta_store()
        list_fn = getattr(meta_store, "list_projected_lane_messages", None)
        if not callable(list_fn):
            return {"total": 0, "by_status": {}, "by_inbox": {}, "failed": []}
        scope = self._scope_snapshot()["storage_namespace"]
        rows = list_fn(namespace=scope, inbox_id=inbox_id, status=None)
        by_status: dict[str, int] = {}
        by_inbox: dict[str, int] = {}
        failed: list[dict[str, Any]] = []
        for row in rows:
            status_val = str(getattr(row, "status", "") or "")
            inbox_val = str(getattr(row, "inbox_id", "") or "")
            by_status[status_val] = by_status.get(status_val, 0) + 1
            by_inbox[inbox_val] = by_inbox.get(inbox_val, 0) + 1
            if status_val == "failed":
                failed.append(
                    {
                        "message_id": getattr(row, "message_id", None),
                        "inbox_id": inbox_val,
                        "conversation_id": getattr(row, "conversation_id", None),
                        "correlation_id": getattr(row, "correlation_id", None),
                    }
                )
        return {
            "total": len(rows),
            "by_status": by_status,
            "by_inbox": by_inbox,
            "failed": failed[:20],
        }

    def blocked_flow_graph(self, *, limit: int = 100) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading blocked flow graph requires project_view capability",
        )
        blocked = self.list_blocked_runs(limit=limit)
        nodes = [
            {
                "id": row.get("process_id"),
                "kind": row.get("process_kind"),
                "status": row.get("status"),
                "workflow_id": row.get("workflow_id"),
                "conversation_id": row.get("conversation_id"),
            }
            for row in blocked
        ]
        return {
            "nodes": nodes,
            "edges": [],
            "blocked_count": len(nodes),
        }

    def operator_dashboard(self, *, limit: int = 100) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading operator dashboard requires project_view capability",
        )
        def _dir_size(path: str | None) -> int:
            import os

            if not path:
                return 0
            total = 0
            for root, _dirs, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
            return total

        engines = [
            self._knowledge_engine(),
            self._conversation_engine(),
            self._workflow_engine(),
        ]
        storage_usage_bytes = 0
        for eng in engines:
            storage_usage_bytes += _dir_size(str(getattr(eng, "persist_directory", "") or ""))
        return {
            "process_table": self.list_process_table(limit=limit),
            "operator_inbox": self.list_operator_inbox(limit=limit),
            "blocked_runs": self.list_blocked_runs(limit=limit),
            "blocked_flow_graph": self.blocked_flow_graph(limit=limit),
            "message_queue": self.message_queue_snapshot(),
            "dead_letters": self.dead_letter_snapshot(limit=limit),
            "capabilities": self.capability_snapshot(),
            "resources": {
                "scheduler": self.scheduler.snapshot(),
                "runs": self.run_registry.snapshot(),
                "services": {
                    "total_services": len(self.service_supervisor.list_services(limit=10_000)),
                    "by_health": self._service_counts_by_health(),
                },
                "storage_usage_bytes": storage_usage_bytes,
                "cost_ledger": self.cost_ledger.snapshot(),
            },
        }

    def list_process_timeline(
        self, *, run_id: str, after_seq: int = 0, limit: int = 200
    ) -> list[dict[str, Any]]:
        self._require_capability(
            "workflow.run.read",
            ["workflow.run.read"],
            approval_message="Reading process timeline requires workflow.run.read capability",
        )
        meta_store = self._execution_meta_store()
        list_events = getattr(meta_store, "list_server_run_events", None)
        if not callable(list_events):
            return []
        return list_events(str(run_id), after_seq=int(after_seq), limit=int(limit))

    def list_scheduler_timeline(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        self._require_capability(
            "project_view",
            ["project_view"],
            approval_message="Reading scheduler timeline requires project_view capability",
        )
        return self.scheduler.timeline(run_id=run_id, limit=limit)

    def budget_snapshot(self) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view", "read_security_scope"],
            approval_message="Reading budget snapshot requires project_view capability",
        )
        return {
            "cost_ledger": self.cost_ledger.snapshot(),
            "budget_model": {
                "token_kind": "token",
                "time_kind": "ms",
                "storage_kind": "bytes",
            },
        }

    def budget_history(self, *, limit: int = 200) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view", "read_security_scope"],
            approval_message="Reading budget history requires project_view capability",
        )
        return {
            "cost_ledger": self.cost_ledger.snapshot(),
            "events": self.cost_ledger.history(limit=limit),
        }

    def list_tool_audit(
        self, *, conversation_id: str | None = None, limit: int = 200
    ) -> dict[str, Any]:
        self._require_capability(
            "project_view",
            ["project_view", "read_graph"],
            approval_message="Reading tool audit trail requires project_view capability",
        )
        engine = self._conversation_engine()
        nodes: list[Any] = []
        for entity_type in ("tool_call", "tool_result"):
            where: dict[str, Any] = {"entity_type": entity_type}
            if conversation_id:
                where["conversation_id"] = str(conversation_id)
            nodes.extend(engine.read.get_nodes(where=where, limit=int(limit)))
        items: list[dict[str, Any]] = []
        for node in nodes[: int(limit)]:
            md = dict(getattr(node, "metadata", {}) or {})
            if not can_access_security_scope(
                str(md.get("security_scope") or ""),
                shared=bool(md.get("shared_scope") or md.get("shared_inbox")),
            ):
                continue
            items.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "entity_type": str(md.get("entity_type") or ""),
                    "tool_name": str(md.get("tool_name") or ""),
                    "conversation_id": str(getattr(node, "conversation_id", "") or ""),
                    "turn_index": getattr(node, "turn_index", None),
                    "summary": getattr(node, "summary", ""),
                    "properties": dict(getattr(node, "properties", {}) or {}),
                }
            )
        return {
            "conversation_id": conversation_id,
            "items": items[: int(limit)],
        }


__all__ = [
    "AnswerRunRequest",
    "ChatRunService",
    "RunCancelledError",
    "RuntimeRunRequest",
    "WorkflowProjectionRebuildingError",
]
