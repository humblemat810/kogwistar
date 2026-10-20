"""Durable workflow-run and step trace-node persistence."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol

from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.runtime.models import (
    StepRunResult,
    WorkflowRunNode,
    WorkflowState,
    WorkflowStepExecNode,
)
from kogwistar.runtime.serialize import try_serialize_with_ref
from kogwistar.runtime.telemetry import TraceContext
from kogwistar.typing_interfaces import ReadLike, WriteLike

from .runtime_artifacts import _make_runtime_edge, _make_trace_span


class RuntimeTraceEngine(Protocol):
    read: ReadLike
    write: WriteLike


class RuntimeTracePersistenceHost(Protocol):
    @property
    def conversation_engine(self) -> RuntimeTraceEngine: ...

    def _trace_write_mode(self) -> AbstractContextManager[None]: ...


class RuntimeTracePersistenceMixin:
    """Persist stable run and step trace artifacts through the host engine."""

    def _persist_workflow_run(
        self: RuntimeTracePersistenceHost,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        turn_node_id: str,
        status: str,
        run_metadata: Mapping[str, JsonValue] | None = None,
        trace_context: TraceContext | None = None,
    ) -> WorkflowRunNode:
        excerpt = f"workflow_run {workflow_id} {run_id} status={status}"
        span = Span(
            **_make_trace_span(
                conversation_id=conversation_id,
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )
        metadata: JsonObject = {
            "entity_type": "workflow_run",
            "doc_id": f"wf_run|{run_id}",
            "workflow_id": workflow_id,
            "workflow_version": "v1",
            "run_id": run_id,
            "conversation_id": conversation_id,
            "turn_node_id": turn_node_id,
            "status": status,
            "level_from_root": 0,
        }
        try:
            existing = self.conversation_engine.read.get_nodes(
                ids=[f"wf_run|{run_id}"], limit=1
            )
            existing_metadata: JsonObject = (
                dict(existing[0].metadata) if existing else {}
            )
        except Exception:
            existing_metadata = {}
        for key, value in existing_metadata.items():
            metadata.setdefault(str(key), value)
        for key, value in (run_metadata or {}).items():
            if key not in metadata and value is not None:
                metadata[str(key)] = value
        if metadata.get("parent_run_id"):
            metadata.setdefault(
                "wf_invoked",
                {
                    "parent_run_id": str(metadata["parent_run_id"]),
                    "workflow_id": str(workflow_id),
                },
            )
        if trace_context is not None and trace_context.has_valid_w3c_ids:
            metadata["trace_id"] = trace_context.trace_id
            metadata["run_execution_span_id"] = trace_context.span_id
            if trace_context.parent_span_id is not None:
                metadata["parent_span_id"] = trace_context.parent_span_id
        node = WorkflowRunNode(
            id=f"wf_run|{run_id}",
            label=f"Workflow run {workflow_id}",
            type="entity",
            doc_id=f"wf_run|{run_id}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            metadata=metadata,
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        with self._trace_write_mode():
            self.conversation_engine.write.add_node(node)
        return node

    def _persist_step_exec(
        self: RuntimeTracePersistenceHost,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        workflow_node_id: str,
        op: str,
        status: str,
        duration_ms: int,
        result: StepRunResult,
        state: WorkflowState,
        token_id: str | None = None,
        parent_token_id: str | None = None,
        join_mask: int | None = None,
        last_exec_node: WorkflowStepExecNode | WorkflowRunNode | None = None,
    ) -> WorkflowStepExecNode:
        result_json = try_serialize_with_ref(result)
        workspace_id = str(state.get("workspace_id") or "").strip()
        excerpt = f"{dict(step=step_seq, op=op, status=status, duration_ms=duration_ms)}"
        span = Span(
            **_make_trace_span(
                conversation_id=conversation_id,
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )
        metadata: JsonObject = {
            "entity_type": "workflow_step_exec",
            "run_id": run_id,
            "workflow_id": workflow_id,
            "workflow_node_id": workflow_node_id,
            "step_seq": step_seq,
            "op": op,
            "status": status,
            "duration_ms": duration_ms,
            "result_json": result_json,
            "token_id": token_id,
            "parent_token_id": parent_token_id,
            "join_mask": join_mask,
            "conversation_id": conversation_id,
            "char_distance_from_last_summary": 0,
            "turn_distance_from_last_summary": 0,
        }
        node = WorkflowStepExecNode(
            id=f"wf_step|{run_id}|{step_seq}",
            label=f"WF step {step_seq}: {op}",
            type="entity",
            doc_id=f"wf_step|{run_id}|{step_seq}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            metadata=metadata,
        )
        if workspace_id:
            node.metadata["workspace_id"] = workspace_id
        with self._trace_write_mode():
            self.conversation_engine.write.add_node(node)
            if last_exec_node:
                content = f"{last_exec_node.safe_get_id()} next is {node.safe_get_id()}"
                edge_span = Span(
                    collection_page_url=f"conversation/{conversation_id}",
                    document_page_url=f"conversation/{conversation_id}#{node.id}",
                    doc_id=f"conv:{conversation_id}",
                    insertion_method="step_exec",
                    page_number=1,
                    start_char=0,
                    end_char=len(content),
                    excerpt=content,
                    context_before="",
                    context_after="",
                    chunk_id=None,
                    source_cluster_id=None,
                    verification=MentionVerification(
                        method="system",
                        is_verified=True,
                        score=1.0,
                        notes="step run execution",
                    ),
                )
                previous_id = last_exec_node.safe_get_id()
                node_id = node.safe_get_id()
                edge = _make_runtime_edge(
                    edge_id=(
                        f"wf_next_step_exec|{run_id}|{step_seq}|"
                        f"last::{previous_id}|to::{node_id}"
                    ),
                    relation="wf_next_step_exec",
                    label=f"wf_next_step_exec {step_seq=}",
                    summary=f"wf_next_step_exec {step_seq=}",
                    doc_id=f"wf_next_step_exec|{run_id}|{step_seq}",
                    conversation_id=conversation_id,
                    source_ids=[previous_id],
                    target_ids=[node_id],
                    mentions=[Grounding(spans=[edge_span])],
                    run_id=run_id,
                    metadata={
                        "source_id": [previous_id],
                        "target_id": [node_id],
                        "char_distance_from_last_summary": 0,
                        "turn_distance_from_last_summary": 0,
                    },
                )
                self.conversation_engine.write.add_edge(edge)
            if result.conversation_node_id:
                content = f"{node.safe_get_id()} created {result.conversation_node_id} durign execution"
                edge_span = Span(
                    collection_page_url=f"conversation/{conversation_id}",
                    document_page_url=f"conversation/{conversation_id}#{node.id}",
                    doc_id=f"conv:{conversation_id}",
                    insertion_method="step_exec",
                    page_number=1,
                    start_char=0,
                    end_char=len(content),
                    excerpt=content,
                    context_before="",
                    context_after="",
                    chunk_id=None,
                    source_cluster_id=None,
                    verification=MentionVerification(
                        method="system",
                        is_verified=True,
                        score=1.0,
                        notes="step execution created node",
                    ),
                )
                edge = _make_runtime_edge(
                    edge_id=(
                        f"conv:{conversation_id}|wfexe:{node.safe_get_id()}|"
                        f"created:{result.conversation_node_id}"
                    ),
                    relation="created_child",
                    label=f"created node during {step_seq=}",
                    summary=f"created node during {step_seq=}",
                    doc_id=f"wf_next_step_exec|{run_id}|{step_seq}",
                    conversation_id=conversation_id,
                    source_ids=[node.safe_get_id()],
                    target_ids=[result.conversation_node_id],
                    mentions=[Grounding(spans=[edge_span])],
                    run_id=run_id,
                    metadata={
                        "source_id": [node.safe_get_id()],
                        "target_id": [result.conversation_node_id],
                        "char_distance_from_last_summary": 0,
                        "turn_distance_from_last_summary": 0,
                    },
                )
                self.conversation_engine.write.add_edge(edge)
        return node
