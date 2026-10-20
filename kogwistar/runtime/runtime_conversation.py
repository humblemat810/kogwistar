"""Conversation-graph lookups used by workflow cancellation and terminal recovery."""

from __future__ import annotations

import queue
from collections.abc import Mapping
from typing import Protocol, cast

from kogwistar.engine_core.models import Node as GraphNode
from kogwistar.engine_core.async_compat import run_awaitable_blocking
from kogwistar.json_types import JsonObject
from kogwistar.runtime.models import WorkflowCheckpointNode, WorkflowState
from kogwistar.typing_interfaces import ReadLike

from .checkpoint_state import decode_checkpoint_state
from .runtime_checkpoints import RuntimeMetadataStore
from .checkpoint_state import _json_int as _json_int
from .runtime_terminals import CancelInfo
from .step_context import RunResult


class ConversationBackendLike(Protocol):
    """Minimal async backend lookup surface used by existence checks."""

    def node_get(self, **kwargs: object) -> Mapping[str, object]: ...

    def edge_get(self, **kwargs: object) -> Mapping[str, object]: ...


class CurrentUserSequenceStore(Protocol):
    """Optional metadata capability used to fence cancellation requests."""

    def current_user_seq(self, conversation_id: str) -> int: ...


class ConversationReadEngine(Protocol):
    read: ReadLike


class ConversationRuntimeHost(Protocol):
    @property
    def conversation_engine(self) -> ConversationReadEngine: ...

    def _runtime_metadata_store(self) -> RuntimeMetadataStore | None: ...

    def _safe_get_conversation_nodes(
        self, *, where: JsonObject, limit: int = 1000
    ) -> list[GraphNode]: ...

    def _conversation_backend(self) -> ConversationBackendLike | None: ...

    def _latest_checkpoint_for_run(
        self, *, conversation_id: str, run_id: str
    ) -> WorkflowCheckpointNode | None: ...


def _json_string_list(value: object) -> list[str]:
    """Decode a persisted string-list field without trusting its shape."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


class ConversationRuntimeMixin:
    """Typed conversation graph queries for cancellation and run recovery."""

    def _safe_get_conversation_nodes(
        self: ConversationRuntimeHost, *, where: JsonObject, limit: int = 1000
    ) -> list[GraphNode]:
        try:
            return list(self.conversation_engine.read.get_nodes(where=where, limit=limit))
        except Exception as exc:
            message = str(exc)
            if "Nothing found on disk" in message or "hnsw segment reader" in message:
                return []
            raise

    def _find_eligible_cancel_request(
        self: ConversationRuntimeHost,
        *,
        conversation_id: str,
        run_id: str,
    ) -> CancelInfo | None:
        cancelled_nodes = self._safe_get_conversation_nodes(
            where={"$and": [{"entity_type": "workflow_cancelled"}, {"run_id": run_id}]},
            limit=1,
        )
        if cancelled_nodes:
            return None

        request_nodes = self._safe_get_conversation_nodes(
            where={
                "$and": [
                    {"entity_type": "workflow_cancel_request"},
                    {"run_id": run_id},
                    {"conversation_id": conversation_id},
                ]
            },
            limit=10_000,
        )
        if not request_nodes:
            return None

        watermark: int | None = None
        metadata = self._runtime_metadata_store()
        current_user_seq = (
            getattr(metadata, "current_user_seq", None) if metadata is not None else None
        )
        if callable(current_user_seq):
            try:
                watermark = int(
                    cast(CurrentUserSequenceStore, metadata).current_user_seq(
                        conversation_id
                    )
                )
            except (TypeError, ValueError, RuntimeError):
                watermark = None

        best: CancelInfo | None = None
        for node in request_nodes:
            seq = _json_int(node.metadata.get("seq"), default=-1)
            if seq < 0 or (watermark is not None and seq > watermark):
                continue
            if not node.id:
                continue
            best_seq = best["seq"] if best is not None else None
            if best is None or best_seq is None or seq < best_seq:
                best = {
                    "source": "graph",
                    "node_id": node.id,
                    "seq": seq,
                    "watermark": watermark,
                }
        return best

    def _conversation_backend(self: ConversationRuntimeHost) -> ConversationBackendLike | None:
        backend = getattr(self.conversation_engine, "backend", None)
        return cast(ConversationBackendLike, backend) if backend is not None else None

    def _runtime_metadata_store(
        self: ConversationRuntimeHost,
    ) -> RuntimeMetadataStore | None:
        """Narrow optional legacy metadata adapters at one compatibility boundary."""
        metadata = getattr(self.conversation_engine, "meta_sqlite", None)
        if metadata is None:
            return None
        required = (
            "get_named_projection",
            "compare_and_swap_named_projection",
            "replace_named_projection",
            "get_latest_entity_event_seq",
        )
        if not all(callable(getattr(metadata, name, None)) for name in required):
            return None
        return cast(RuntimeMetadataStore, metadata)

    def _conversation_namespace(self: ConversationRuntimeHost) -> str:
        """Read engine namespace across legacy test and adapter doubles."""
        namespace = getattr(self.conversation_engine, "namespace", "default")
        return namespace if isinstance(namespace, str) and namespace else "default"

    def _terminal_run_result(
        self: ConversationRuntimeHost, *, conversation_id: str, run_id: str
    ) -> RunResult | None:
        """Rehydrate terminal state instead of invoking a child workflow again."""
        terminal_types = (
            ("workflow_completed", "succeeded"),
            ("workflow_failed", "failure"),
            ("workflow_cancelled", "cancelled"),
        )
        terminal_metadata: JsonObject | None = None
        status: str | None = None
        for entity_type, candidate_status in terminal_types:
            nodes = self.conversation_engine.read.get_nodes(
                where={
                    "$and": [
                        {"entity_type": entity_type},
                        {"run_id": str(run_id)},
                    ]
                },
                limit=1,
            )
            if nodes:
                terminal_metadata = dict(nodes[0].metadata)
                status = candidate_status
                break
        if status is None:
            return None

        checkpoint = self._latest_checkpoint_for_run(
            conversation_id=conversation_id, run_id=str(run_id)
        )
        state: WorkflowState = (
            decode_checkpoint_state(
                checkpoint.metadata.get("state_json"), run_id=str(run_id)
            )
            if checkpoint is not None
            else {}
        )
        errors = (
            _json_string_list(terminal_metadata.get("errors"))
            if terminal_metadata is not None
            else []
        )
        return RunResult(
            run_id=str(run_id),
            final_state=state,
            mq=queue.Queue(),
            status=status,
            errors=errors,
        )

    def _conversation_node_exists(
        self: ConversationRuntimeHost, node_id: str
    ) -> bool:
        backend = self._conversation_backend()
        if backend is not None:
            existing = run_awaitable_blocking(
                backend.node_get(ids=[node_id], include=[])
            )
            return bool(existing.get("ids"))
        try:
            return bool(self.conversation_engine.read.node_exists(ids=[node_id]))
        except Exception:
            nodes = getattr(self.conversation_engine, "nodes", [])
            return any(
                str(getattr(node, "id", "") or "") == node_id
                for node in (nodes or [])
            )

    def _conversation_edge_exists(
        self: ConversationRuntimeHost, edge_id: str
    ) -> bool:
        backend = self._conversation_backend()
        if backend is not None:
            existing = run_awaitable_blocking(
                backend.edge_get(ids=[edge_id], include=[])
            )
            return bool(existing.get("ids"))
        try:
            return bool(self.conversation_engine.read.edge_exists(ids=[edge_id]))
        except Exception:
            edges = getattr(self.conversation_engine, "edges", [])
            return any(
                str(getattr(edge, "id", "") or "") == edge_id
                for edge in (edges or [])
            )
