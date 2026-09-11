from __future__ import annotations

"""
Storage backend abstraction.

Goal: remove direct Chroma collection usage from engine.py so we can later plug in
a Postgres+pgvector backend with the same surface.

This backend is intentionally "thin": it mostly forwards calls to underlying
stores. Semantics live in engine/runtime, not in the backend.

Key design points:
- Methods accept **kwargs and forward to the underlying backend implementation.
- Chroma backend supports the same collection set the engine uses today:
  node_index, nodes, edges, edge_endpoints, documents, domains, node_docs,
  node_refs, edge_refs.
- UnitOfWork exists so engine/runtime can write `with engine.uow(): ...`.
  In Chroma mode it's a no-op for the vector index; transactions are handled by
  the meta store (SQLite today, Postgres later).

Update contract for implementers:
- Omitted `documents` and `embeddings` mean preserve the stored values; they do
  not mean clear or recompute either value.
- A metadata-only `*_update(ids=..., metadatas=...)` must not invoke an
  embedding provider and must preserve the existing embedding exactly.
- A caller that changes a document must explicitly choose a vector policy:
  provide an embedding, deliberately recompute one, or reject the change.

This is observable API semantics, not a Chroma detail.  Third-party backends
must pass `tests/core/test_lifecycle_read_contract.py`; otherwise lifecycle
patches can silently introduce model cost, latency, and vector drift.
"""

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import inspect
from typing import Any, AsyncIterator, Dict, Iterator, Literal, Protocol

JSONDict = Dict[str, Any]


@dataclass(frozen=True)
class TwoStageProjectionCapability:
    """Backend arrangement contract for ADR-018 deferred semantic projection."""

    supports_two_stage: bool = False
    canonical_event_replay: bool = False
    canonical_read: bool = False
    stage1_strategy: Literal["none", "transient_projection"] = "none"
    stage1_metadata_query: bool = False
    stage1_cleanup: bool = False
    stage2_semantic_projection: bool = False
    revision_gated_promotion: bool = False
    semantic_readiness_gate: bool = False
    delete_reconciliation: bool = False
    atomic_promotion: Literal["same_store", "eventual_reconcile"] = "eventual_reconcile"
    reason: str = "two-stage projection is not implemented for this backend arrangement"

    def missing_contracts(self) -> tuple[str, ...]:
        """Return unfulfilled ADR-018 guarantees for fail-closed configuration."""
        missing: list[str] = []
        for field in (
            "supports_two_stage",
            "canonical_event_replay",
            "canonical_read",
            "stage2_semantic_projection",
            "revision_gated_promotion",
            "semantic_readiness_gate",
            "delete_reconciliation",
        ):
            if not getattr(self, field):
                missing.append(field)
        if self.stage1_strategy == "transient_projection":
            for field in ("stage1_metadata_query", "stage1_cleanup"):
                if not getattr(self, field):
                    missing.append(field)
        return tuple(missing)

    def is_complete(self) -> bool:
        return not self.missing_contracts()


class TwoStageProjectionAdapter(Protocol):
    """Executable ADR-018 write arrangement supplied by a capable backend.

    The adapter owns canonical Stage-1 admission and later Stage-2 promotion.
    ``GraphKnowledgeEngine`` only dispatches to it after the immutable capability
    descriptor passes; it never silently falls back to synchronous embedding.
    """

    def add_node(self, node: Any, *, doc_id: str | None = None) -> None: ...

    def add_edge(self, edge: Any, *, doc_id: str | None = None) -> None: ...

    def apply_embedding_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str | None,
    ) -> None: ...


class AsyncTwoStageProjectionAdapter(Protocol):
    """Async counterpart for arrangements used from an async engine.

    This protocol is intentionally separate from the synchronous adapter:
    async callers must not satisfy the contract by hiding blocking bridge
    calls inside an ``async`` method.
    """

    async def add_node(self, node: Any, *, doc_id: str | None = None) -> None: ...

    async def add_edge(self, edge: Any, *, doc_id: str | None = None) -> None: ...

    async def apply_embedding_job(
        self,
        *,
        entity_kind: str,
        entity_id: str,
        op: str,
        payload_json: str | None,
    ) -> None: ...


def get_async_two_stage_projection_adapter(
    backend: Any,
) -> AsyncTwoStageProjectionAdapter | None:
    """Return only an executable async arrangement; never bridge sync calls."""

    adapter = getattr(backend, "async_two_stage_projection_adapter", None)
    if callable(adapter) and not hasattr(adapter, "add_node"):
        adapter = adapter()
    if adapter is None:
        return None
    required = ("add_node", "add_edge", "apply_embedding_job")
    capability = get_two_stage_projection_capability(backend)
    if capability.stage1_strategy == "transient_projection":
        required += (
            "stage1_query",
            "remove_stage1",
            "promote_stage2",
            "remove_stage2_or_invalidate",
            "reconcile_projection",
        )
    if all(
        callable(getattr(adapter, name, None))
        and inspect.iscoroutinefunction(getattr(adapter, name))
        for name in required
    ):
        return adapter
    return None


def get_two_stage_projection_capability(backend: Any) -> TwoStageProjectionCapability:
    """Read an optional backend declaration without widening StorageBackend."""

    declared = getattr(backend, "two_stage_projection_capability", None)
    if callable(declared):
        declared = declared()
    if isinstance(declared, TwoStageProjectionCapability):
        return declared
    return TwoStageProjectionCapability()


def get_two_stage_projection_adapter(backend: Any) -> TwoStageProjectionAdapter | None:
    """Read an optional executable arrangement without widening StorageBackend."""
    adapter = getattr(backend, "two_stage_projection_adapter", None)
    if callable(adapter) and not hasattr(adapter, "add_node"):
        adapter = adapter()
    if adapter is None:
        return None
    required = (
        "add_node",
        "add_edge",
        "apply_embedding_job",
    )
    if get_two_stage_projection_capability(backend).stage1_strategy == "transient_projection":
        # Descriptor flags are not executable capability. A transient
        # arrangement must expose its query/cleanup/reconciliation seam too.
        required += (
            "stage1_query",
            "remove_stage1",
            "promote_stage2",
            "remove_stage2_or_invalidate",
            "reconcile_projection",
        )
    if all(
        callable(getattr(adapter, name, None))
        for name in required
    ):
        return adapter
    return None


class UnitOfWork(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[None]: ...


class AsyncUnitOfWork(Protocol):
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]: ...


@dataclass
class NoopUnitOfWork(UnitOfWork):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        yield


@dataclass
class AsyncNoopUnitOfWork(AsyncUnitOfWork):
    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        yield


class StorageBackend(Protocol):
    # Generic dispatch (optional to use directly)
    def call(self, collection_key: str, method: str, **kwargs) -> Any: ...

    # Node index (optional)
    def node_index_get(self, **kwargs) -> Any: ...
    def node_index_query(self, **kwargs) -> Any: ...
    def node_index_add(self, **kwargs) -> Any: ...
    def node_index_upsert(self, **kwargs) -> Any: ...
    def node_index_update(self, **kwargs) -> Any: ...
    def node_index_delete(self, **kwargs) -> Any: ...

    # Nodes
    def node_get(self, **kwargs) -> Any: ...
    def node_query(self, **kwargs) -> Any: ...
    def node_add(self, **kwargs) -> Any: ...
    def node_upsert(self, **kwargs) -> Any: ...
    def node_update(self, **kwargs) -> Any: ...
    def node_delete(self, **kwargs) -> Any: ...

    # Edges
    def edge_get(self, **kwargs) -> Any: ...
    def edge_query(self, **kwargs) -> Any: ...
    def edge_add(self, **kwargs) -> Any: ...
    def edge_upsert(self, **kwargs) -> Any: ...
    def edge_update(self, **kwargs) -> Any: ...
    def edge_delete(self, **kwargs) -> Any: ...

    # Edge endpoints (hypergraph incidence/materialization)
    def edge_endpoints_get(self, **kwargs) -> Any: ...
    def edge_endpoints_query(self, **kwargs) -> Any: ...
    def edge_endpoints_add(self, **kwargs) -> Any: ...
    def edge_endpoints_upsert(self, **kwargs) -> Any: ...
    def edge_endpoints_update(self, **kwargs) -> Any: ...
    def edge_endpoints_delete(self, **kwargs) -> Any: ...

    # Documents
    def document_get(self, **kwargs) -> Any: ...
    def document_query(self, **kwargs) -> Any: ...
    def document_add(self, **kwargs) -> Any: ...
    def document_upsert(self, **kwargs) -> Any: ...
    def document_update(self, **kwargs) -> Any: ...
    def document_delete(self, **kwargs) -> Any: ...

    # Domains
    def domain_get(self, **kwargs) -> Any: ...
    def domain_query(self, **kwargs) -> Any: ...
    def domain_add(self, **kwargs) -> Any: ...
    def domain_upsert(self, **kwargs) -> Any: ...
    def domain_update(self, **kwargs) -> Any: ...
    def domain_delete(self, **kwargs) -> Any: ...

    # Node docs
    def node_docs_get(self, **kwargs) -> Any: ...
    def node_docs_query(self, **kwargs) -> Any: ...
    def node_docs_add(self, **kwargs) -> Any: ...
    def node_docs_upsert(self, **kwargs) -> Any: ...
    def node_docs_update(self, **kwargs) -> Any: ...
    def node_docs_delete(self, **kwargs) -> Any: ...

    # Node refs
    def node_refs_get(self, **kwargs) -> Any: ...
    def node_refs_query(self, **kwargs) -> Any: ...
    def node_refs_add(self, **kwargs) -> Any: ...
    def node_refs_upsert(self, **kwargs) -> Any: ...
    def node_refs_update(self, **kwargs) -> Any: ...
    def node_refs_delete(self, **kwargs) -> Any: ...

    # Edge refs
    def edge_refs_get(self, **kwargs) -> Any: ...
    def edge_refs_query(self, **kwargs) -> Any: ...
    def edge_refs_add(self, **kwargs) -> Any: ...
    def edge_refs_upsert(self, **kwargs) -> Any: ...
    def edge_refs_update(self, **kwargs) -> Any: ...
    def edge_refs_delete(self, **kwargs) -> Any: ...
