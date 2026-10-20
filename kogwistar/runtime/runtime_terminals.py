"""Idempotent terminal workflow records and graph relationships."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Literal, Protocol
from typing_extensions import TypedDict

from kogwistar.engine_core.models import Grounding, Span
from kogwistar.id_provider import stable_id
from kogwistar.runtime.models import (
    WorkflowCancelledNode,
    WorkflowCompletedNode,
    WorkflowFailedNode,
)
from kogwistar.typing_interfaces import WriteLike

from .runtime_artifacts import _make_runtime_edge, _make_trace_span


class CancelInfo(TypedDict):
    """Typed cancellation request selected from callback or graph evidence."""

    source: Literal["callback", "graph"]
    node_id: str | None
    seq: int | None
    watermark: int | None


class TerminalPersistenceEngine(Protocol):
    write: WriteLike


class TerminalPersistenceHost(Protocol):
    @property
    def conversation_engine(self) -> TerminalPersistenceEngine: ...

    def _trace_write_mode(self) -> AbstractContextManager[None]: ...

    def _conversation_node_exists(self, node_id: str) -> bool: ...

    def _conversation_edge_exists(self, edge_id: str) -> bool: ...

    def _replace_run_status_projection(
        self,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        status: str,
        node_id: str,
        accepted_step_seq: int,
    ) -> None: ...


class TerminalPersistenceMixin:
    """Persist terminal records without changing ``WorkflowRuntime`` methods."""

    def _persist_cancelled_terminal(
        self: TerminalPersistenceHost,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        accepted_step_seq: int,
        cancel_info: CancelInfo | None,
        last_processed_node_id: str | None = None,
    ) -> str:
        node_id = f"wf_cancelled|{run_id}"
        if self._conversation_node_exists(node_id):
            self._replace_run_status_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                status="cancelled",
                node_id=node_id,
                accepted_step_seq=accepted_step_seq,
            )
            return node_id

        req_node_id = None if not cancel_info else str(cancel_info.get("node_id") or "")
        req_seq = None if not cancel_info else cancel_info.get("seq")
        watermark = None if not cancel_info else cancel_info.get("watermark")
        excerpt = (
            f"workflow cancelled run_id={run_id} accepted_step_seq={accepted_step_seq}"
        )
        span = Span(
            **_make_trace_span(
                conversation_id=conversation_id,
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )
        node = WorkflowCancelledNode(
            id=node_id,
            label="Workflow cancelled",
            type="entity",
            doc_id=node_id,
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={"entity_type": "workflow_cancelled"},
            metadata={
                "entity_type": "workflow_cancelled",
                "workflow_id": workflow_id,
                "run_id": run_id,
                "conversation_id": conversation_id,
                "accepted_step_seq": int(accepted_step_seq),
                "cancel_request_node_id": req_node_id or None,
                "cancel_request_seq": req_seq,
                "accepted_watermark": watermark,
                "last_processed_node_id": (
                    str(last_processed_node_id) if last_processed_node_id else None
                ),
                "level_from_root": 0,
            },
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        with self._trace_write_mode():
            self.conversation_engine.write.add_node(node)

            run_node_id = f"wf_run|{run_id}"
            if self._conversation_node_exists(run_node_id):
                edge_id = str(stable_id("workflow.edge", "cancelled", run_node_id, node_id))
                if not self._conversation_edge_exists(edge_id):
                    edge = _make_runtime_edge(
                        edge_id=edge_id,
                        relation="wf_cancelled",
                        label="wf_cancelled",
                        summary="workflow cancelled",
                        doc_id=f"wf_cancelled|{run_id}",
                        conversation_id=conversation_id,
                        source_ids=[run_node_id],
                        target_ids=[node_id],
                        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                        run_id=run_id,
                    )
                    self.conversation_engine.write.add_edge(edge)

            if req_node_id:
                req_edge_id = str(
                    stable_id("workflow.edge", "cancel_reconciled", req_node_id, node_id)
                )
                if not self._conversation_edge_exists(req_edge_id):
                    req_edge = _make_runtime_edge(
                        edge_id=req_edge_id,
                        relation="wf_cancel_reconciled",
                        label="wf_cancel_reconciled",
                        summary="cancel request reconciled",
                        doc_id=f"wf_cancelled|{run_id}",
                        conversation_id=conversation_id,
                        source_ids=[req_node_id],
                        target_ids=[node_id],
                        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                        run_id=run_id,
                    )
                    self.conversation_engine.write.add_edge(req_edge)

            if last_processed_node_id:
                cancelled_at_edge_id = str(
                    stable_id(
                        "workflow.edge",
                        "cancelled_at",
                        node_id,
                        str(last_processed_node_id),
                    )
                )
                if not self._conversation_edge_exists(cancelled_at_edge_id):
                    cancelled_at_edge = _make_runtime_edge(
                        edge_id=cancelled_at_edge_id,
                        relation="wf_cancelled_at",
                        label="wf_cancelled_at",
                        summary="workflow cancelled at node",
                        doc_id=f"wf_cancelled|{run_id}",
                        conversation_id=conversation_id,
                        source_ids=[node_id],
                        target_ids=[str(last_processed_node_id)],
                        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                        run_id=run_id,
                    )
                    self.conversation_engine.write.add_edge(cancelled_at_edge)
            self._replace_run_status_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                status="cancelled",
                node_id=node_id,
                accepted_step_seq=accepted_step_seq,
            )
        return node_id

    def _persist_completed_terminal(
        self: TerminalPersistenceHost,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        accepted_step_seq: int,
        last_processed_node_id: str | None = None,
    ) -> str:
        node_id = f"wf_completed|{run_id}"
        if self._conversation_node_exists(node_id):
            self._replace_run_status_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                status="completed",
                node_id=node_id,
                accepted_step_seq=accepted_step_seq,
            )
            return node_id

        excerpt = (
            f"workflow completed run_id={run_id} accepted_step_seq={accepted_step_seq}"
        )
        span = Span(
            **_make_trace_span(
                conversation_id=conversation_id,
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )
        node = WorkflowCompletedNode(
            id=node_id,
            label="Workflow completed",
            type="entity",
            doc_id=node_id,
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={"entity_type": "workflow_completed"},
            metadata={
                "entity_type": "workflow_completed",
                "workflow_id": workflow_id,
                "run_id": run_id,
                "conversation_id": conversation_id,
                "accepted_step_seq": int(accepted_step_seq),
                "last_processed_node_id": (
                    str(last_processed_node_id) if last_processed_node_id else None
                ),
                "level_from_root": 0,
            },
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        with self._trace_write_mode():
            self.conversation_engine.write.add_node(node)

            run_node_id = f"wf_run|{run_id}"
            if self._conversation_node_exists(run_node_id):
                edge_id = str(stable_id("workflow.edge", "completed", run_node_id, node_id))
                if not self._conversation_edge_exists(edge_id):
                    edge = _make_runtime_edge(
                        edge_id=edge_id,
                        relation="wf_completed",
                        label="wf_completed",
                        summary="workflow completed",
                        doc_id=f"wf_completed|{run_id}",
                        conversation_id=conversation_id,
                        source_ids=[run_node_id],
                        target_ids=[node_id],
                        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                        run_id=run_id,
                    )
                    self.conversation_engine.write.add_edge(edge)
            self._replace_run_status_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                status="completed",
                node_id=node_id,
                accepted_step_seq=accepted_step_seq,
            )
        return node_id

    def _persist_failed_terminal(
        self: TerminalPersistenceHost,
        *,
        conversation_id: str,
        workflow_id: str,
        run_id: str,
        accepted_step_seq: int,
        errors: list[str] | None,
        last_processed_node_id: str | None = None,
    ) -> str:
        node_id = f"wf_failed|{run_id}"
        if self._conversation_node_exists(node_id):
            self._replace_run_status_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                status="failed",
                node_id=node_id,
                accepted_step_seq=accepted_step_seq,
            )
            return node_id

        safe_errors = [str(err) for err in (errors or [])]
        excerpt = f"workflow failed run_id={run_id} accepted_step_seq={accepted_step_seq}"
        if safe_errors:
            excerpt += f" errors={len(safe_errors)}"
        span = Span(
            **_make_trace_span(
                conversation_id=conversation_id,
                excerpt=excerpt,
                doc_id=f"conv:{conversation_id}",
            )
        )
        metadata = {
            "entity_type": "workflow_failed",
            "workflow_id": workflow_id,
            "run_id": run_id,
            "conversation_id": conversation_id,
            "accepted_step_seq": int(accepted_step_seq),
            "last_processed_node_id": (
                str(last_processed_node_id) if last_processed_node_id else None
            ),
            "level_from_root": 0,
        }
        if safe_errors:
            metadata["errors"] = safe_errors

        node = WorkflowFailedNode(
            id=node_id,
            label="Workflow failed",
            type="entity",
            doc_id=node_id,
            summary=excerpt,
            mentions=[Grounding(spans=[span])],
            properties={"entity_type": "workflow_failed"},
            metadata=metadata,
            level_from_root=0,
            domain_id=None,
            canonical_entity_id=None,
            embedding=None,
        )
        with self._trace_write_mode():
            self.conversation_engine.write.add_node(node)

            run_node_id = f"wf_run|{run_id}"
            if self._conversation_node_exists(run_node_id):
                edge_id = str(stable_id("workflow.edge", "failed", run_node_id, node_id))
                if not self._conversation_edge_exists(edge_id):
                    edge = _make_runtime_edge(
                        edge_id=edge_id,
                        relation="wf_failed",
                        label="wf_failed",
                        summary="workflow failed",
                        doc_id=f"wf_failed|{run_id}",
                        conversation_id=conversation_id,
                        source_ids=[run_node_id],
                        target_ids=[node_id],
                        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                        run_id=run_id,
                    )
                    self.conversation_engine.write.add_edge(edge)

            if last_processed_node_id:
                failed_at_edge_id = str(
                    stable_id(
                        "workflow.edge",
                        "failed_at",
                        node_id,
                        str(last_processed_node_id),
                    )
                )
                if not self._conversation_edge_exists(failed_at_edge_id):
                    failed_at_edge = _make_runtime_edge(
                        edge_id=failed_at_edge_id,
                        relation="wf_failed_at",
                        label="wf_failed_at",
                        summary="workflow failed at node",
                        doc_id=f"wf_failed|{run_id}",
                        conversation_id=conversation_id,
                        source_ids=[node_id],
                        target_ids=[str(last_processed_node_id)],
                        mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
                        run_id=run_id,
                    )
                    self.conversation_engine.write.add_edge(failed_at_edge)
            self._replace_run_status_projection(
                conversation_id=conversation_id,
                workflow_id=workflow_id,
                run_id=run_id,
                status="failed",
                node_id=node_id,
                accepted_step_seq=accepted_step_seq,
            )
        return node_id
