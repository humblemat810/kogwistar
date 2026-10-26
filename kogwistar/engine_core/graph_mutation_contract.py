"""Python validation for the versioned Rust graph-mutation wire contract.

This is a DTO seam only. It does not perform graph writes or enqueue jobs; the
selected transaction owner remains responsible for applying the validated
command atomically.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

_MAX_POSTGRES_INTEGER = 2_147_483_647


class GraphMutationContractError(ValueError):
    """Raised when a graph mutation payload cannot cross the language seam."""


class GraphScopePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    namespace: str
    workspace_id: str | None = None
    graph_space: str | None = None


class GraphRecordPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    document: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    embedding: list[StrictFloat] | None = None


class IndexJobIntentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    index_kind: str
    op: Literal["UPSERT", "DELETE"]
    payload_json: str | None = None
    max_retries: StrictInt = 3

    @model_validator(mode="after")
    def validate_fields(self) -> "IndexJobIntentPayload":
        if any(
            not value
            for value in (
                self.job_id,
                self.namespace,
                self.entity_kind,
                self.entity_id,
                self.index_kind,
            )
        ):
            raise GraphMutationContractError("index job identifiers must not be empty")
        if not 0 <= self.max_retries <= _MAX_POSTGRES_INTEGER:
            raise GraphMutationContractError(
                "index job max_retries must fit a non-negative PostgreSQL INTEGER"
            )
        return self


class GraphMutationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: GraphScopePayload
    table: Literal["gke_nodes", "gke_edges"]
    entity_kind: Literal["node", "edge"]
    event_id: str
    op: Literal["ADD", "REPLACE", "TOMBSTONE"]
    payload: dict[str, object]
    record: GraphRecordPayload
    embedding_dim: StrictInt
    model_envelope: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "GraphMutationPayload":
        expected_table = "gke_nodes" if self.entity_kind == "node" else "gke_edges"
        if self.table != expected_table:
            raise GraphMutationContractError(
                f"graph entity kind {self.entity_kind!r} must use table {expected_table!r}"
            )
        if not self.scope.namespace:
            raise GraphMutationContractError("graph mutation namespace must not be empty")
        if not self.event_id or not self.record.id:
            raise GraphMutationContractError("graph mutation identifiers must not be empty")
        if self.payload.get("id") != self.record.id:
            raise GraphMutationContractError(
                "graph mutation payload id must match the projection record id"
            )
        if self.embedding_dim < 0:
            raise GraphMutationContractError("embedding dimension must not be negative")
        if self.record.embedding is not None and len(self.record.embedding) != self.embedding_dim:
            raise GraphMutationContractError(
                "embedding dimension does not match vector length"
            )
        if self.record.embedding is not None and any(
            not math.isfinite(value) for value in self.record.embedding
        ):
            raise GraphMutationContractError("embedding values must be finite")
        return self


class GraphMutationBatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal[1]
    atomic: Literal[True]
    mutations: list[GraphMutationPayload] = Field(min_length=1)
    index_jobs: list[IndexJobIntentPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_batch(self) -> "GraphMutationBatchPayload":
        first_scope = self.mutations[0].scope
        if any(mutation.scope != first_scope for mutation in self.mutations[1:]):
            raise GraphMutationContractError(
                "all atomic graph mutations must share one graph scope"
            )
        if any(job.namespace != first_scope.namespace for job in self.index_jobs):
            raise GraphMutationContractError(
                "index job namespace must match graph mutation namespace"
            )
        return self


def validate_graph_mutation_batch(payload: object) -> GraphMutationBatchPayload:
    """Validate and return an immutable, typed batch for either implementation."""

    try:
        return GraphMutationBatchPayload.model_validate(payload)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, GraphMutationContractError):
            raise
        raise GraphMutationContractError(str(exc)) from exc

