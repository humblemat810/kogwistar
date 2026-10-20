"""Checkpoint lookup and durable workflow-run ownership operations."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from os import PathLike
from typing import Any, Callable, Protocol, cast

from kogwistar.engine_core.models import Grounding, MentionVerification, Span
from kogwistar.json_types import JsonObject, JsonValue
from kogwistar.runtime.models import (
    WorkflowRunNode,
    WorkflowState,
    WorkflowStepExecNode,
    WorkflowCheckpointMetadata,
    WorkflowCheckpointNode,
)
from kogwistar.runtime.serialize import try_serialize_with_ref
from kogwistar.runtime.telemetry import TraceContext
from kogwistar.runtime.step_context import RunResult
from kogwistar.typing_interfaces import ReadLike, WriteLike
from kogwistar.runtime.run_ownership import (
    RunOwnership,
    RunOwnershipConflict,
    RunOwnershipStore,
    claim_run_ownership,
    release_run_ownership_for_resume,
    renew_run_ownership,
)

from .projections import (
    WORKFLOW_RUNTIME_PROJECTION_SCHEMA_VERSION,
    workflow_checkpoint_latest_projection_namespace,
    workflow_run_ownership_projection_namespace,
    workflow_run_status_projection_namespace,
)
from .runtime_artifacts import _make_trace_span
from .runtime_artifacts import _make_runtime_edge
from .checkpoint_state import _json_int, decode_checkpoint_state


class RuntimeMetadataStore(RunOwnershipStore, Protocol):
    """Typed metadata surface used by workflow recovery and projections."""

    def replace_named_projection(
        self,
        namespace: str,
        key: str,
        payload: JsonObject,
        *,
        last_authoritative_seq: int,
        last_materialized_seq: int,
        projection_schema_version: int,
        materialization_status: str,
    ) -> None: ...

    def get_latest_entity_event_seq(self, *, namespace: str = "default") -> int: ...


class CheckpointEngine(Protocol):
    read: ReadLike
    write: WriteLike


class CheckpointRuntimeHost(Protocol):
    @property
    def CHECKPOINT_SCHEMA_VERSION(self) -> int: ...

    @property
    def conversation_engine(self) -> CheckpointEngine: ...

    @property
    def run_ownership_lease_ms(self) -> int: ...

    def _runtime_metadata_store(self) -> RuntimeMetadataStore | None: ...

    def _conversation_namespace(self) -> str: ...

    def _replace_runtime_projection(
        self,
        *,
        namespace: str,
        key: str,
        payload: JsonObject,
        materialization_status: str = "ready",
    ) -> None: ...

    def _checkpoint_node_from_projection(
        self,
        *,
        conversation_id: str,
        run_id: str,
        node_id: str,
        metadata: Mapping[str, JsonValue],
    ) -> WorkflowCheckpointNode: ...

    def _replace_checkpoint_latest_projection(
        self,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        node_id: str,
        checkpoint_metadata: Mapping[str, JsonValue],
    ) -> None: ...

    def _trace_write_mode(self) -> AbstractContextManager[None]: ...

    def checkpointable_state_copy(self, state: WorkflowState) -> WorkflowState: ...

    def _latest_checkpoint_for_run(
        self, *, conversation_id: str, run_id: str
    ) -> WorkflowCheckpointNode | None: ...

    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None,
        initial_state: WorkflowState,
        run_id: str | None = None,
        cache_dir: str | PathLike[str] | None = None,
        _resume_step_seq: int | None = None,
        _trace_context: TraceContext | None = None,
        _authority_context: Mapping[str, Any] | None = None,
        _parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult: ...


class CheckpointRuntimeMixin:
    """Methods mixed into ``WorkflowRuntime`` without changing its API."""

    def _replace_runtime_projection(
        self: CheckpointRuntimeHost,
        *,
        namespace: str,
        key: str,
        payload: JsonObject,
        materialization_status: str = "ready",
    ) -> None:
        metadata = self._runtime_metadata_store()
        if metadata is None:
            return
        latest_seq = 0
        try:
            latest_seq = metadata.get_latest_entity_event_seq(
                namespace=self._conversation_namespace()
            )
        except (TypeError, ValueError, RuntimeError):
            latest_seq = 0
        metadata.replace_named_projection(
            namespace=namespace,
            key=str(key),
            payload=payload,
            last_authoritative_seq=latest_seq,
            last_materialized_seq=latest_seq,
            projection_schema_version=WORKFLOW_RUNTIME_PROJECTION_SCHEMA_VERSION,
            materialization_status=materialization_status,
        )

    def resume_from_latest_checkpoint(
        self: CheckpointRuntimeHost,
        *,
        run_id: str,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str,
        cache_dir: str | PathLike[str] | None = None,
        _parent_trace_context: TraceContext | None = None,
        _parent_authority_context: Mapping[str, Any] | None = None,
        _parent_lease_renew: Callable[[], None] | None = None,
    ) -> RunResult:
        """Continue a run from its latest ordinary workflow checkpoint."""
        latest_ckpt = self._latest_checkpoint_for_run(
            conversation_id=conversation_id,
            run_id=str(run_id),
        )
        if latest_ckpt is None:
            raise ValueError(f"Cannot resume run {run_id}: no checkpoints found.")
        metadata = latest_ckpt.metadata
        checkpoint_workflow_id = str(metadata.get("workflow_id") or workflow_id)
        if checkpoint_workflow_id != str(workflow_id):
            raise ValueError(
                f"Checkpoint workflow mismatch for {run_id}: "
                f"stored={checkpoint_workflow_id!r} requested={workflow_id!r}"
            )
        state = decode_checkpoint_state(metadata.get("state_json"), run_id=str(run_id))
        step_seq = _json_int(metadata.get("step_seq"), default=0)
        continuation_context = None
        persisted_trace_id = metadata.get("trace_id")
        persisted_span_id = metadata.get("run_execution_span_id")
        if persisted_trace_id and persisted_span_id:
            candidate = TraceContext(
                run_id=str(run_id),
                token_id=str(run_id),
                step_seq=step_seq,
                node_id="resume",
                trace_id=str(persisted_trace_id),
                span_id=str(persisted_span_id),
            )
            if candidate.has_valid_w3c_ids:
                continuation_context = candidate
        if continuation_context is None and _parent_trace_context is not None:
            continuation_context = _parent_trace_context.child_run(
                run_id=str(run_id),
                token_id=str(run_id),
                step_seq=step_seq,
                node_id="resume",
            )
        return self.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=state,
            run_id=str(run_id),
            cache_dir=cache_dir,
            _resume_step_seq=step_seq + 1,
            _trace_context=continuation_context,
            _authority_context=_parent_authority_context,
            _parent_lease_renew=_parent_lease_renew,
        )

    def _replace_checkpoint_latest_projection(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        node_id: str,
        checkpoint_metadata: Mapping[str, JsonValue],
    ) -> None:
        self._replace_runtime_projection(
            namespace=workflow_checkpoint_latest_projection_namespace(conversation_id),
            key=str(run_id),
            payload={
                "entity_type": "workflow_checkpoint",
                "run_id": str(run_id),
                "workflow_id": str(workflow_id),
                "conversation_id": str(conversation_id),
                "step_seq": int(step_seq),
                "node_id": str(node_id),
                "checkpoint_metadata": dict(checkpoint_metadata),
            },
        )

    def _replace_run_status_projection(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        status: str,
        node_id: str,
        accepted_step_seq: int,
    ) -> None:
        self._replace_runtime_projection(
            namespace=workflow_run_status_projection_namespace(conversation_id),
            key=str(run_id),
            payload={
                "run_id": str(run_id),
                "workflow_id": str(workflow_id),
                "conversation_id": str(conversation_id),
                "status": str(status),
                "terminal": str(status) in {"completed", "failed", "cancelled"},
                "terminal_node_id": str(node_id),
                "accepted_step_seq": int(accepted_step_seq),
            },
            materialization_status=str(status),
        )

    def _latest_checkpoint_for_run(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        run_id: str,
    ) -> WorkflowCheckpointNode | None:
        metadata = self._runtime_metadata_store()
        if metadata is not None:
            row = metadata.get_named_projection(
                workflow_checkpoint_latest_projection_namespace(conversation_id),
                str(run_id),
            )
            row_map: dict[str, JsonValue] = row if row is not None else {}
            payload_value = row_map.get("payload")
            payload: JsonObject = (
                dict(payload_value) if isinstance(payload_value, dict) else {}
            )
            node_id = str(payload.get("node_id") or "")
            if node_id:
                nodes = self.conversation_engine.read.get_nodes(ids=[node_id], limit=1)
                if nodes:
                    return cast(WorkflowCheckpointNode, nodes[0])
            checkpoint_metadata = payload.get("checkpoint_metadata")
            if isinstance(checkpoint_metadata, dict):
                return self._checkpoint_node_from_projection(
                    conversation_id=conversation_id,
                    run_id=str(run_id),
                    node_id=node_id,
                    metadata=checkpoint_metadata,
                )

        ckpts = self.conversation_engine.read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_checkpoint"},
                    {"run_id": str(run_id)},
                ]
            },
        )
        if not ckpts:
            return None

        def checkpoint_step_seq(checkpoint: WorkflowCheckpointNode) -> int:
            value = checkpoint.metadata.get("step_seq")
            if isinstance(value, bool):
                return -1
            if isinstance(value, int):
                return value
            if isinstance(value, (float, str)):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return -1
            return -1

        return cast(
            WorkflowCheckpointNode,
            max(
                ckpts,
                key=lambda checkpoint: checkpoint_step_seq(
                    cast(WorkflowCheckpointNode, checkpoint)
                ),
            ),
        )

    def _checkpoint_node_from_projection(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        run_id: str,
        node_id: str,
        metadata: Mapping[str, JsonValue],
    ) -> WorkflowCheckpointNode:
        """Rebuild a checkpoint when its graph node is not durable."""
        checkpoint_metadata = dict(metadata)
        checkpoint_metadata.setdefault("conversation_id", str(conversation_id))
        checkpoint_metadata.setdefault("level_from_root", 0)
        checkpoint_metadata.setdefault(
            "checkpoint_schema_version", self.CHECKPOINT_SCHEMA_VERSION
        )
        validated = WorkflowCheckpointMetadata.model_validate(checkpoint_metadata)
        if validated.run_id != str(run_id):
            raise ValueError("durable checkpoint projection run_id does not match lookup")
        projected_conversation_id = checkpoint_metadata.get("conversation_id")
        if projected_conversation_id is not None and str(projected_conversation_id) != str(
            conversation_id
        ):
            raise ValueError(
                "durable checkpoint projection conversation_id does not match lookup"
            )
        checkpoint_id = str(node_id or "")
        if not checkpoint_id:
            checkpoint_id = f"wf_ckpt|{validated.run_id}|{validated.step_seq}"
        excerpt = f"checkpoint step_seq={validated.step_seq}"
        span = Span(
            **_make_trace_span(
                conversation_id=str(conversation_id),
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )
        return WorkflowCheckpointNode(
            id=checkpoint_id,
            label=f"WF checkpoint {validated.step_seq}",
            type="entity",
            doc_id=checkpoint_id,
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            metadata=checkpoint_metadata,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )

    def _claim_workflow_run(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        allow_resume_takeover: bool = False,
    ) -> RunOwnership | None:
        metadata = self._runtime_metadata_store()
        if metadata is None:
            return None
        try:
            return claim_run_ownership(
                metadata,
                namespace=workflow_run_ownership_projection_namespace(conversation_id),
                run_id=run_id,
                workflow_id=workflow_id,
                conversation_id=conversation_id,
                allow_resume_takeover=allow_resume_takeover,
                lease_ms=self.run_ownership_lease_ms,
            )
        except RunOwnershipConflict as exc:
            raise WorkflowRunOwnershipError(str(exc)) from exc

    def _renew_workflow_run(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        run_id: str,
        ownership: RunOwnership | None,
    ) -> RunOwnership | None:
        if ownership is None:
            return None
        metadata = self._runtime_metadata_store()
        if metadata is None:
            return ownership
        try:
            return renew_run_ownership(
                metadata,
                namespace=workflow_run_ownership_projection_namespace(conversation_id),
                run_id=run_id,
                ownership=ownership,
                lease_ms=self.run_ownership_lease_ms,
            )
        except RunOwnershipConflict as exc:
            raise WorkflowRunOwnershipError(str(exc)) from exc

    def _release_workflow_run_for_resume(
        self: CheckpointRuntimeHost,
        *,
        conversation_id: str,
        run_id: str,
        ownership: RunOwnership | None,
    ) -> None:
        if ownership is None:
            return
        metadata = self._runtime_metadata_store()
        if metadata is None:
            return
        if not release_run_ownership_for_resume(
            metadata,
            namespace=workflow_run_ownership_projection_namespace(conversation_id),
            run_id=run_id,
            ownership=ownership,
        ):
            raise WorkflowRunOwnershipError(
                f"workflow run {run_id!r} ownership was lost while suspending"
            )

    def _persist_checkpoint(
        self: CheckpointRuntimeHost,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        step_seq: int,
        state: WorkflowState,
        last_exec_node: WorkflowStepExecNode | WorkflowRunNode | None = None,
        trace_context: TraceContext | None = None,
    ) -> None:
        state_copy = self.checkpointable_state_copy(state)
        state_json = try_serialize_with_ref(state_copy)
        excerpt = f"checkpoint step_seq={step_seq}"
        span = Span(
            **_make_trace_span(
                conversation_id=conversation_id,
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )

        node = WorkflowCheckpointNode(
            id=f"wf_ckpt|{run_id}|{step_seq}",
            label=f"WF checkpoint {step_seq}",
            type="entity",
            doc_id=f"wf_ckpt|{run_id}|{step_seq}",
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={},
            metadata={
                "entity_type": "workflow_checkpoint",
                "run_id": run_id,
                "workflow_id": workflow_id,
                "step_seq": step_seq,
                "checkpoint_schema_version": self.CHECKPOINT_SCHEMA_VERSION,
                "state_json": state_json,
                "level_from_root": 0,
                "conversation_id": conversation_id,
            },
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
            level_from_root=0,
        )
        if trace_context is not None:
            node.metadata["trace_id"] = trace_context.trace_id
            node.metadata["run_execution_span_id"] = trace_context.span_id
            if trace_context.parent_span_id is not None:
                node.metadata["parent_span_id"] = trace_context.parent_span_id

        with self._trace_write_mode():
            self.conversation_engine.write.add_node(node)
            if last_exec_node is not None:
                edge_span = Span(
                    collection_page_url=f"conversation/{conversation_id}",
                    document_page_url=f"conversation/{conversation_id}#{node.id}",
                    doc_id=f"conv:{conversation_id}",
                    insertion_method="persist_checkpoint",
                    page_number=1,
                    start_char=0,
                    end_char=len(node.summary),
                    excerpt=node.summary,
                    context_before="",
                    context_after="",
                    chunk_id=None,
                    source_cluster_id=None,
                    verification=MentionVerification(
                        method="system",
                        is_verified=True,
                        score=1.0,
                        notes="persist_checkpoint",
                    ),
                )
                last_exec_id = last_exec_node.safe_get_id()
                checkpoint_id = node.safe_get_id()
                edge = _make_runtime_edge(
                    edge_id=(
                        f"persist_checkpoint|{run_id}|{step_seq}|"
                        f"last::{last_exec_id}|to::{checkpoint_id}"
                    ),
                    relation="persist_checkpoint during",
                    label=f"persist_checkpoint during {step_seq=}",
                    summary=f"persist_checkpoint during {step_seq=}",
                    doc_id=f"wf_next_step_exec|{run_id}|{step_seq}",
                    conversation_id=conversation_id,
                    source_ids=[checkpoint_id],
                    target_ids=[last_exec_id],
                    mentions=[Grounding(spans=[edge_span])],
                    run_id=run_id,
                    metadata={
                        "source_id": [last_exec_id],
                        "target_id": [checkpoint_id],
                        "char_distance_from_last_summary": 0,
                        "turn_distance_from_last_summary": 0,
                    },
                )
                self.conversation_engine.write.add_edge(edge)
            self._replace_checkpoint_latest_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                step_seq=step_seq,
                node_id=str(node.id),
                checkpoint_metadata=dict(node.metadata),
            )


class WorkflowRunOwnershipError(RuntimeError):
    """Raised when a durable run cannot be safely resumed by this worker."""
