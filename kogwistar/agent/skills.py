"""Deterministic, provider-neutral skill-to-hypergraph ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import copy
import asyncio
from collections.abc import Mapping
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import CatalogEntry, DurableCatalogStore, scoped_projection_namespace
from kogwistar.engine_core.embedding_profile import (
    AsyncNamedProjectionStore,
    NamedProjectionStore,
)

SkillStepKind = Literal[
    "instruction",
    "capability_call",
    "mcp_call",
    "script_call",
    "command_template",
    "nested_workflow",
    "check",
]
SkillProjectionAcl = Callable[["SkillGraphArtifact", str], bool]

_SAFE_COMMAND_CHARS = re.compile(r"[;&|<>`$]|\n")
_SHELL_WRAPPERS = {"sh", "bash", "zsh", "fish", "cmd", "cmd.exe", "powershell", "pwsh"}
_PATH_PARTS = {"", ".", ".."}


def validate_package_relative_path(path: str) -> str:
    value = str(path).replace("\\", "/").strip()
    parts = value.split("/")
    if not value or value.startswith("/") or ":" in parts[0] or any(part in _PATH_PARTS for part in parts):
        raise ValueError("skill script path must be package-relative and traversal-free")
    return value


def validate_command_argv(command: str) -> list[str]:
    value = str(command).strip()
    if not value or _SAFE_COMMAND_CHARS.search(value):
        raise ValueError("command template must be argv-only; shell operators are forbidden")
    argv = shlex.split(value, posix=True)
    if not argv:
        raise ValueError("command template must contain an executable")
    if argv[0].lower() in _SHELL_WRAPPERS:
        raise ValueError("shell wrapper commands are forbidden")
    return argv


class SkillGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: SkillStepKind | Literal["skill", "capability", "mcp"]
    name: str
    summary: str = ""
    source_ref: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    binding_status: Literal["unbound", "validated"] = "unbound"
    invocable: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _safe_invocation(self) -> "SkillGraphNode":
        if self.invocable and self.binding_status != "validated":
            raise ValueError("only validated skill bindings may be invocable")
        if self.kind in {"script_call", "command_template"} and not self.required_capabilities:
            raise ValueError("effectful skill steps require explicit capabilities")
        return self


class SkillGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edge_id: str
    kind: str
    source_ids: list[str]
    target_ids: list[str]
    source_ref: str | None = None
    status: Literal["active", "candidate", "rejected"] = "active"
    metadata: dict[str, object] = Field(default_factory=dict)


class SkillGraphArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    provider_local_id: str
    skill_version: str = "v1"
    source_fingerprint: str
    parser_id: str = "kogwistar.core.markdown"
    parser_version: str = "v1"
    projection_revision: int = 1
    tenant_id: str | None = None
    project_id: str | None = None
    namespace: str | None = None
    nodes: list[SkillGraphNode] = Field(default_factory=list)
    edges: list[SkillGraphEdge] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    provenance: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bounded_graph(self) -> "SkillGraphArtifact":
        if len(self.nodes) > 256 or len(self.edges) > 512:
            raise ValueError("skill graph exceeds deterministic ingestion bounds")
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("skill graph node IDs must be unique")
        edge_ids = {edge.edge_id for edge in self.edges}
        if len(edge_ids) != len(self.edges):
            raise ValueError("skill graph edge IDs must be unique")
        for edge in self.edges:
            if not edge.source_ids or not edge.target_ids:
                raise ValueError("skill graph edges require source and target IDs")
            if not set(edge.source_ids + edge.target_ids) <= node_ids:
                raise ValueError(f"skill graph edge references unknown node: {edge.edge_id}")
        if self.projection_revision < 1:
            raise ValueError("projection_revision must be positive")
        for node in self.nodes:
            if not str(node.source_ref or "").strip():
                raise ValueError(f"skill node lacks source reference: {node.node_id}")
        return self

    def artifact_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class SkillProjectionRequest(BaseModel):
    """Typed projection work item; providers cannot choose arbitrary lanes."""

    provider_id: str
    provider_local_id: str
    source_fingerprint: str
    project_id: str
    tenant_id: str
    projection_revision: int = Field(1, ge=1)
    lane_id: str = ""

    @model_validator(mode="after")
    def _derive_lane(self) -> "SkillProjectionRequest":
        self.lane_id = f"ws:{self.tenant_id}:{self.project_id}:g:projection:lane:skills"
        return self


class SkillExecutionPlan(BaseModel):
    """Bounded graph-guided plan; executing workflow still owns effects."""

    skill_id: str
    node_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    mcp_schema_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source_fingerprint: str
    projection_revision: int
    execution_policy: "SkillExecutionPolicy | None" = None


class SkillExecutionPolicy(BaseModel):
    """Explicit limits for effectful skill steps.

    This is a plan contract, not an executor. The ordinary workflow/tool
    binding remains responsible for enforcing the declared capabilities.
    """

    model_config = ConfigDict(extra="forbid")

    max_output_bytes: int = Field(default=64 * 1024, ge=1, le=16 * 1024 * 1024)
    max_time_ms: int = Field(default=30_000, ge=1, le=10 * 60 * 1000)
    cwd: str | None = None
    environment_keys: list[str] = Field(default_factory=list)
    sandbox: str = "default"
    allow_network: bool = False

    @model_validator(mode="after")
    def _safe_policy(self) -> "SkillExecutionPolicy":
        if self.cwd is not None:
            validate_package_relative_path(self.cwd)
        if any(not str(key).strip() or "=" in str(key) for key in self.environment_keys):
            raise ValueError("environment_keys must contain names, not assignments")
        if not self.sandbox.strip():
            raise ValueError("sandbox must be non-empty")
        return self


SkillExecutionPlan.model_rebuild()


class SkillExecutionEvidence(BaseModel):
    """Auditable references emitted by an ordinary skill workflow step."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    workflow_step_ref: str
    skill_id: str
    node_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    mcp_schema_refs: list[str] = Field(default_factory=list)
    source_fingerprint: str
    projection_revision: int
    result_ref: str | None = None
    outcome: Literal["success", "failure", "cancelled"]

    def state_patch(self) -> dict[str, object]:
        """Return JSON-compatible state for ordinary checkpoint/evidence writes."""
        return {"skill_execution_evidence": self.model_dump(mode="json")}


@runtime_checkable
class SemanticSkillIngestionProvider(Protocol):
    """Optional bounded parser; it proposes data, never materializes it."""

    provider_id: str
    parser_id: str
    parser_version: str

    def parse(self, source: Mapping[str, Any]) -> SkillGraphArtifact: ...


def validate_skill_artifact(
    artifact: SkillGraphArtifact,
    *,
    expected_fingerprint: str | None = None,
    tenant_id: str | None = None,
    project_id: str | None = None,
    require_exact_scope: bool = False,
    allowed_node_kinds: set[str] | frozenset[str] | None = None,
) -> SkillGraphArtifact:
    """Validate source/scope/bindings before any projection write."""

    if expected_fingerprint is not None and artifact.source_fingerprint != expected_fingerprint:
        raise ValueError("skill source fingerprint mismatch")
    if tenant_id is not None and (
        artifact.tenant_id != tenant_id if require_exact_scope else artifact.tenant_id not in (None, tenant_id)
    ):
        raise PermissionError("skill artifact tenant scope mismatch")
    if project_id is not None and (
        artifact.project_id != project_id if require_exact_scope else artifact.project_id not in (None, project_id)
    ):
        raise PermissionError("skill artifact project scope mismatch")
    if allowed_node_kinds is not None:
        unknown = {str(node.kind) for node in artifact.nodes} - set(allowed_node_kinds)
        if unknown:
            raise ValueError(f"unsupported skill node kinds: {sorted(unknown)}")
    return artifact


class SkillProjectionStore:
    """Rebuildable current projection with stale-revision rejection."""

    def __init__(
        self,
        *,
        acl_enabled: bool = False,
        acl_checker: SkillProjectionAcl | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self.acl_enabled = acl_enabled
        self.acl_checker = acl_checker
        self.tenant_id = tenant_id
        self.project_id = project_id
        self._current: dict[str, SkillGraphArtifact] = {}
        self._history: dict[str, list[SkillGraphArtifact]] = {}

    def upsert(self, artifact: SkillGraphArtifact) -> SkillGraphArtifact:
        validate_skill_artifact(
            artifact,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            require_exact_scope=self.tenant_id is not None or self.project_id is not None,
        )
        key = f"{artifact.provider_id}:{artifact.provider_local_id}"
        current = self._current.get(key)
        if current is not None:
            if artifact.projection_revision < current.projection_revision:
                raise ValueError("stale skill projection revision")
            if (
                artifact.projection_revision == current.projection_revision
                and artifact.source_fingerprint != current.source_fingerprint
            ):
                raise ValueError("skill projection revision fingerprint collision")
            if artifact.projection_revision == current.projection_revision:
                return current.model_copy(deep=True)
        self._current[key] = artifact
        self._history.setdefault(key, []).append(artifact)
        return artifact

    def _visible(
        self,
        artifact: SkillGraphArtifact,
        *,
        principal: str,
        tenant_id: str | None,
        project_id: str | None,
    ) -> bool:
        exact_scope = self.tenant_id is not None or self.project_id is not None
        if self.tenant_id is not None and tenant_id != self.tenant_id:
            return False
        if self.project_id is not None and project_id != self.project_id:
            return False
        expected_tenant = self.tenant_id if exact_scope else tenant_id
        expected_project = self.project_id if exact_scope else project_id
        if expected_tenant is not None and artifact.tenant_id != expected_tenant:
            return False
        if expected_project is not None and artifact.project_id != expected_project:
            return False
        if not self.acl_enabled:
            return True
        return self.acl_checker is not None and bool(self.acl_checker(artifact, principal))

    def get(
        self,
        provider_id: str,
        provider_local_id: str,
        *,
        principal: str = "system",
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> SkillGraphArtifact | None:
        artifact = self._current.get(f"{provider_id}:{provider_local_id}")
        if artifact is None or not self._visible(
            artifact,
            principal=principal,
            tenant_id=tenant_id,
            project_id=project_id,
        ):
            return None
        return artifact.model_copy(deep=True)

    def remove(self, provider_id: str, provider_local_id: str) -> None:
        self._current.pop(f"{provider_id}:{provider_local_id}", None)

    def remove_provider(
        self,
        provider_id: str,
        *,
        provider_version: str | None = None,
        lifecycle_token: str | None = None,
    ) -> tuple[str, ...]:
        removed = tuple(
            key
            for key, artifact in self._current.items()
            if key.startswith(f"{provider_id}:")
            and (
                provider_version is None
                or str(artifact.provenance.get("provider_version") or "v1")
                == provider_version
            )
            and (
                lifecycle_token is None
                or artifact.provenance.get("provider_lifecycle_token") == lifecycle_token
            )
        )
        for key in removed:
            self._current.pop(key, None)
        return removed

    def history(
        self,
        provider_id: str,
        provider_local_id: str,
        *,
        principal: str = "system",
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[SkillGraphArtifact, ...]:
        return tuple(
            artifact.model_copy(deep=True)
            for artifact in self._history.get(f"{provider_id}:{provider_local_id}", ())
            if self._visible(
                artifact,
                principal=principal,
                tenant_id=tenant_id,
                project_id=project_id,
            )
        )


class DurableSkillProjectionStore(SkillProjectionStore):
    """Skill projection backed by Kogwistar's existing named-projection CAS."""

    namespace = "agent_skill_graph"

    def __init__(
        self,
        metadata: NamedProjectionStore,
        *,
        namespace: str = namespace,
        tenant_id: str | None = None,
        project_id: str | None = None,
        acl_enabled: bool = True,
        acl_checker: SkillProjectionAcl | None = None,
    ) -> None:
        super().__init__(
            acl_enabled=acl_enabled,
            acl_checker=acl_checker,
            tenant_id=tenant_id,
            project_id=project_id,
        )
        self._metadata = metadata
        self.namespace = scoped_projection_namespace(
            str(namespace), tenant_id=tenant_id, project_id=project_id
        )
        self.tenant_id = tenant_id
        self.project_id = project_id
        if not callable(getattr(metadata, "list_named_projections", None)):
            raise TypeError("metadata must implement named projection operations")
        self._refresh()

    def _matches_store_scope(
        self,
        *,
        tenant_id: str | None,
        project_id: str | None,
    ) -> bool:
        return tenant_id == self.tenant_id and project_id == self.project_id

    @staticmethod
    def _key(provider_id: str, provider_local_id: str) -> str:
        return f"{provider_id}:{provider_local_id}"

    @staticmethod
    def _storage_id(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @classmethod
    def _pointer_key(cls, key: str) -> str:
        return f"skill-current:{cls._storage_id(key)}"

    @classmethod
    def _revision_key(cls, key: str, revision: int) -> str:
        return f"skill-revision:{cls._storage_id(key)}:{int(revision)}"

    @classmethod
    def _graph_pointer_key(cls, key: str) -> str:
        return f"skill-graph-current:{cls._storage_id(key)}"

    @classmethod
    def _graph_item_key(cls, key: str, kind: str, item_id: str) -> str:
        item_hash = hashlib.sha256(str(item_id).encode("utf-8")).hexdigest()[:32]
        return f"skill-graph-{kind}:{cls._storage_id(key)}:{item_hash}"

    @staticmethod
    def _cas_values(row: dict[str, Any] | None) -> tuple[int | None, int | None]:
        if row is None:
            return None, None
        return int(row.get("last_authoritative_seq", 0)), int(row.get("last_materialized_seq", 0))

    @staticmethod
    def _update(
        *,
        namespace: str,
        key: str,
        payload: dict[str, Any],
        row: dict[str, Any] | None,
        revision: int,
        status: str = "ready",
    ) -> dict[str, Any]:
        expected_a, expected_m = DurableSkillProjectionStore._cas_values(row)
        return {
            "namespace": namespace,
            "key": key,
            "payload": payload,
            "expected_last_authoritative_seq": expected_a,
            "expected_last_materialized_seq": expected_m,
            "last_authoritative_seq": int(revision),
            "last_materialized_seq": int(revision),
            "projection_schema_version": 1,
            "materialization_status": status,
        }

    def _refresh(self) -> None:
        self._current.clear()
        self._history.clear()
        revisions: dict[str, list[SkillGraphArtifact]] = {}
        pointers: dict[str, int] = {}
        legacy_current: dict[str, SkillGraphArtifact] = {}
        for row in self._metadata.list_named_projections(self.namespace):
            payload = row.get("payload") or {}
            artifact_data = payload.get("artifact")
            key = str(row.get("key"))
            if isinstance(artifact_data, dict):
                artifact = SkillGraphArtifact.model_validate(artifact_data)
                if not self._matches_store_scope(
                    tenant_id=artifact.tenant_id, project_id=artifact.project_id
                ):
                    raise ValueError("skill projection row scope does not match namespace")
                logical_key = str(payload.get("logical_key") or key)
                if payload.get("record_kind") == "skill_revision":
                    revisions.setdefault(logical_key, []).append(artifact)
                else:  # Legacy current/history blob; retain read compatibility.
                    legacy_current[logical_key] = artifact
            history_data = payload.get("history") or []
            if history_data:
                self._history[key] = [SkillGraphArtifact.model_validate(item) for item in history_data]
            if payload.get("record_kind") == "skill_current" and row.get(
                "materialization_status"
            ) != "retired":
                pointers[str(payload["logical_key"])] = int(payload["current_revision"])
        for key, values in revisions.items():
            self._history[key] = sorted(
                values, key=lambda artifact: artifact.projection_revision
            )
        self._current.update(legacy_current)
        for key, revision in pointers.items():
            artifact = next(
                (
                    item
                    for item in self._history.get(key, ())
                    if item.projection_revision == revision
                ),
                None,
            )
            if artifact is not None:
                self._current[key] = artifact

    def prepare_upsert_update(
        self, artifact: SkillGraphArtifact
    ) -> tuple[SkillGraphArtifact, list[dict[str, Any]]]:
        """Build immutable revision/current-pointer CAS updates without applying."""

        validate_skill_artifact(
            artifact,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            require_exact_scope=True,
        )
        self._refresh()
        key = self._key(artifact.provider_id, artifact.provider_local_id)
        current = self._current.get(key)
        pointer_key = self._pointer_key(key)
        pointer_row = self._metadata.get_named_projection(self.namespace, pointer_key)
        history = self._history.get(key, [])
        latest = history[-1] if history else current
        if latest is not None:
            if artifact.projection_revision < latest.projection_revision:
                raise ValueError("stale skill projection revision")
            if artifact.projection_revision == latest.projection_revision:
                if artifact.artifact_fingerprint() != latest.artifact_fingerprint():
                    raise ValueError("skill projection revision content collision")
                if current is not None:
                    return latest.model_copy(deep=True), []
        revision_key = self._revision_key(key, artifact.projection_revision)
        revision_row = self._metadata.get_named_projection(self.namespace, revision_key)
        updates: list[dict[str, Any]] = []
        if revision_row is None:
            updates.append(
                self._update(
                    namespace=self.namespace,
                    key=revision_key,
                    payload={
                        "record_kind": "skill_revision",
                        "logical_key": key,
                        "artifact": artifact.model_dump(mode="json"),
                        "artifact_fingerprint": artifact.artifact_fingerprint(),
                    },
                    row=None,
                    revision=artifact.projection_revision,
                )
            )
        updates.append(
            self._update(
                namespace=self.namespace,
                key=pointer_key,
                payload={
                    "record_kind": "skill_current",
                    "logical_key": key,
                    "current_revision": artifact.projection_revision,
                    "revision_key": revision_key,
                },
                row=pointer_row,
                revision=artifact.projection_revision,
            )
        )
        return artifact.model_copy(deep=True), updates

    def apply_prepared_updates(self, updates: list[dict[str, Any]]) -> None:
        """Apply same-store updates atomically through existing metadata CAS."""

        if not updates:
            return
        batch = getattr(self._metadata, "compare_and_swap_named_projections", None)
        if not callable(batch):
            raise TypeError("metadata must support atomic named projection batch CAS")
        if not batch(updates):
            raise ValueError("skill projection changed concurrently")
        self._refresh()

    def prepare_graph_updates(self, artifact: SkillGraphArtifact) -> list[dict[str, Any]]:
        """Prepare current graph-node/edge projections for one artifact revision."""

        validate_skill_artifact(
            artifact,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
            require_exact_scope=True,
        )
        logical_key = self._key(artifact.provider_id, artifact.provider_local_id)
        pointer_key = self._graph_pointer_key(logical_key)
        pointer_row = self._metadata.get_named_projection(self.namespace, pointer_key)
        previous = (pointer_row or {}).get("payload") or {}
        previous_keys = set(previous.get("node_keys", ())) | set(previous.get("edge_keys", ()))
        node_keys = [
            self._graph_item_key(logical_key, "node", node.node_id)
            for node in artifact.nodes
        ]
        edge_keys = [
            self._graph_item_key(logical_key, "edge", edge.edge_id)
            for edge in artifact.edges
        ]
        updates: list[dict[str, Any]] = []
        for node, row_key in zip(artifact.nodes, node_keys):
            row = self._metadata.get_named_projection(self.namespace, row_key)
            updates.append(
                self._update(
                    namespace=self.namespace,
                    key=row_key,
                    payload={
                        "record_kind": "skill_graph_node",
                        "logical_key": logical_key,
                        "node_id": node.node_id,
                        "revision": artifact.projection_revision,
                        "node": node.model_dump(mode="json"),
                    },
                    row=row,
                    revision=artifact.projection_revision,
                )
            )
        for edge, row_key in zip(artifact.edges, edge_keys):
            row = self._metadata.get_named_projection(self.namespace, row_key)
            updates.append(
                self._update(
                    namespace=self.namespace,
                    key=row_key,
                    payload={
                        "record_kind": "skill_graph_edge",
                        "logical_key": logical_key,
                        "edge_id": edge.edge_id,
                        "revision": artifact.projection_revision,
                        "edge": edge.model_dump(mode="json"),
                    },
                    row=row,
                    revision=artifact.projection_revision,
                )
            )
        current_keys = set(node_keys) | set(edge_keys)
        for row_key in sorted(previous_keys - current_keys):
            row = self._metadata.get_named_projection(self.namespace, row_key)
            if row is not None:
                updates.append(
                    self._update(
                        namespace=self.namespace,
                        key=row_key,
                        payload=dict(row.get("payload") or {}),
                        row=row,
                        revision=artifact.projection_revision,
                        status="retired",
                    )
                )
        updates.append(
            self._update(
                namespace=self.namespace,
                key=pointer_key,
                payload={
                    "record_kind": "skill_graph_current",
                    "logical_key": logical_key,
                    "revision": artifact.projection_revision,
                    "node_keys": node_keys,
                    "edge_keys": edge_keys,
                },
                row=pointer_row,
                revision=artifact.projection_revision,
            )
        )
        return updates

    def graph(
        self,
        provider_id: str,
        provider_local_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """Read current graph-native rows, excluding retired projections."""

        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return None
        self._refresh()
        logical_key = self._key(provider_id, provider_local_id)
        artifact = super().get(
            provider_id,
            provider_local_id,
            principal=str(kwargs.get("principal", "system")),
            tenant_id=kwargs.get("tenant_id"),
            project_id=kwargs.get("project_id"),
        )
        if artifact is None:
            return None
        pointer = self._metadata.get_named_projection(
            self.namespace, self._graph_pointer_key(logical_key)
        )
        payload = (pointer or {}).get("payload") or {}
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for row_key in payload.get("node_keys", ()):
            row = self._metadata.get_named_projection(self.namespace, str(row_key))
            if row and row.get("materialization_status") != "retired":
                item = (row.get("payload") or {}).get("node")
                if isinstance(item, dict):
                    nodes.append(item)
        for row_key in payload.get("edge_keys", ()):
            row = self._metadata.get_named_projection(self.namespace, str(row_key))
            if row and row.get("materialization_status") != "retired":
                item = (row.get("payload") or {}).get("edge")
                if isinstance(item, dict):
                    edges.append(item)
        return {
            "logical_key": logical_key,
            "revision": int(payload.get("revision", 0) or 0),
            "nodes": nodes,
            "edges": edges,
        }

    def upsert(self, artifact: SkillGraphArtifact) -> SkillGraphArtifact:
        prepared, updates = self.prepare_upsert_update(artifact)
        self.apply_prepared_updates([*updates, *self.prepare_graph_updates(prepared)])
        return prepared

    def get(
        self,
        provider_id: str,
        provider_local_id: str,
        **kwargs: Any,
    ) -> SkillGraphArtifact | None:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return None
        self._refresh()
        return super().get(provider_id, provider_local_id, **kwargs)

    def history(
        self,
        provider_id: str,
        provider_local_id: str,
        **kwargs: Any,
    ) -> tuple[SkillGraphArtifact, ...]:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return ()
        self._refresh()
        return super().history(provider_id, provider_local_id, **kwargs)

    def remove(self, provider_id: str, provider_local_id: str) -> None:
        self._refresh()
        key = self._key(provider_id, provider_local_id)
        pointer_key = self._pointer_key(key)
        row = self._metadata.get_named_projection(self.namespace, pointer_key)
        if row is not None:
            payload = dict(row.get("payload") or {})
            payload.update(
                {
                    "record_kind": "skill_current",
                    "logical_key": key,
                    "current_revision": None,
                }
            )
            expected_a, expected_m = self._cas_values(row)
            if not self._metadata.compare_and_swap_named_projection(
                self.namespace,
                pointer_key,
                payload,
                expected_last_authoritative_seq=expected_a,
                expected_last_materialized_seq=expected_m,
                last_authoritative_seq=expected_a or 1,
                last_materialized_seq=expected_m or 1,
                projection_schema_version=1,
                materialization_status="retired",
            ):
                raise ValueError("skill projection changed concurrently")
        else:
            legacy_row = self._metadata.get_named_projection(self.namespace, key)
            if legacy_row is not None:
                payload = dict(legacy_row.get("payload") or {})
                payload["artifact"] = None
                expected_a, expected_m = self._cas_values(legacy_row)
                if not self._metadata.compare_and_swap_named_projection(
                    self.namespace,
                    key,
                    payload,
                    expected_last_authoritative_seq=expected_a,
                    expected_last_materialized_seq=expected_m,
                    last_authoritative_seq=expected_a or 1,
                    last_materialized_seq=expected_m or 1,
                    projection_schema_version=1,
                    materialization_status="retired",
                ):
                    raise ValueError("skill projection changed concurrently")
        self._current.pop(key, None)

    def remove_provider(
        self,
        provider_id: str,
        *,
        provider_version: str | None = None,
        lifecycle_token: str | None = None,
    ) -> tuple[str, ...]:
        self._refresh()
        removed = tuple(
            key
            for key, artifact in self._current.items()
            if key.startswith(f"{provider_id}:")
            and (
                provider_version is None
                or str(artifact.provenance.get("provider_version") or "v1")
                == provider_version
            )
            and (
                lifecycle_token is None
                or artifact.provenance.get("provider_lifecycle_token") == lifecycle_token
            )
        )
        for key in removed:
            local_id = key[len(provider_id) + 1 :]
            self.remove(provider_id, local_id)
        return removed


class DurableSkillCatalogMaterializer:
    """Atomically materialize one scoped artifact and its catalog descriptors.

    It composes existing named-projection batch CAS; it introduces neither a
    second queue nor a second authority store. The artifact remains a serving
    projection of its provider-owned source.
    """

    def __init__(
        self,
        *,
        projections: DurableSkillProjectionStore,
        catalog: DurableCatalogStore,
    ) -> None:
        if projections._metadata is not catalog._metadata:
            raise ValueError("durable skill and catalog stores must share one metadata store")
        if (
            projections.tenant_id != catalog.tenant_id
            or projections.project_id != catalog.project_id
        ):
            raise ValueError("durable skill and catalog stores must share one scope")
        self.projections = projections
        self.catalog = catalog

    def materialize(
        self,
        artifact: SkillGraphArtifact,
        *,
        guard_updates: list[dict[str, Any]] | None = None,
    ) -> SkillGraphArtifact:
        projected, artifact_updates = self.projections.prepare_upsert_update(artifact)
        graph_updates = self.projections.prepare_graph_updates(projected)
        catalog_updates: list[dict[str, Any]] = []
        for entry in catalog_entries_from_artifact(projected):
            _entry, updates = self.catalog.prepare_upsert_update(entry)
            catalog_updates.extend(updates)
        updates = [*artifact_updates, *graph_updates, *catalog_updates, *(guard_updates or [])]
        if updates:
            batch = getattr(self.projections._metadata, "compare_and_swap_named_projections", None)
            if not callable(batch):
                raise TypeError("metadata must support atomic named projection batch CAS")
            if not batch(updates):
                raise ValueError("skill/catalog projection changed concurrently")
            self.projections._refresh()
            self.catalog._refresh()
        return projected

    def remove_provider(
        self,
        provider_id: str,
        *,
        provider_version: str | None = None,
        lifecycle_token: str | None = None,
    ) -> tuple[str, ...]:
        """Retire skill, graph, and catalog current pointers in one CAS batch."""

        self.projections._refresh()
        self.catalog._refresh()
        skill_keys = tuple(
            key
            for key, artifact in self.projections._current.items()
            if key.startswith(f"{provider_id}:")
            and (
                provider_version is None
                or str(artifact.provenance.get("provider_version") or "v1")
                == provider_version
            )
            and (
                lifecycle_token is None
                or artifact.provenance.get("provider_lifecycle_token") == lifecycle_token
            )
        )
        catalog_keys = tuple(
            key
            for key, entry in self.catalog._entries.items()
            if entry.provider_id == provider_id
            and (provider_version is None or entry.provider_version == provider_version)
            and (
                lifecycle_token is None
                or entry.metadata.get("provider_lifecycle_token") == lifecycle_token
            )
        )
        updates: list[dict[str, Any]] = []
        for logical_key in skill_keys:
            pointer_key = self.projections._pointer_key(logical_key)
            pointer = self.projections._metadata.get_named_projection(
                self.projections.namespace, pointer_key
            )
            if pointer is not None:
                payload = dict(pointer.get("payload") or {})
                payload.update({"current_revision": None})
                updates.append(
                    self.projections._update(
                        namespace=self.projections.namespace,
                        key=pointer_key,
                        payload=payload,
                        row=pointer,
                        revision=int(pointer.get("last_authoritative_seq", 1) or 1),
                        status="retired",
                    )
                )
            graph_pointer_key = self.projections._graph_pointer_key(logical_key)
            graph_pointer = self.projections._metadata.get_named_projection(
                self.projections.namespace, graph_pointer_key
            )
            if graph_pointer is not None:
                graph_payload = dict(graph_pointer.get("payload") or {})
                graph_keys = set(graph_payload.get("node_keys", ())) | set(
                    graph_payload.get("edge_keys", ())
                )
                revision = int(graph_pointer.get("last_authoritative_seq", 1) or 1)
                for graph_key in graph_keys:
                    graph_row = self.projections._metadata.get_named_projection(
                        self.projections.namespace, str(graph_key)
                    )
                    if graph_row is not None:
                        updates.append(
                            self.projections._update(
                                namespace=self.projections.namespace,
                                key=str(graph_key),
                                payload=dict(graph_row.get("payload") or {}),
                                row=graph_row,
                                revision=revision,
                                status="retired",
                            )
                        )
                graph_payload.update({"revision": None, "node_keys": [], "edge_keys": []})
                updates.append(
                    self.projections._update(
                        namespace=self.projections.namespace,
                        key=graph_pointer_key,
                        payload=graph_payload,
                        row=graph_pointer,
                        revision=revision,
                        status="retired",
                    )
                )
        for logical_id in catalog_keys:
            pointer_key = self.catalog._pointer_key(logical_id)
            pointer = self.catalog._metadata.get_named_projection(
                self.catalog._projection_namespace, pointer_key
            )
            if pointer is not None:
                payload = dict(pointer.get("payload") or {})
                payload.update({"current_revision": None})
                updates.append(
                    self.catalog._update(
                        namespace=self.catalog._projection_namespace,
                        key=pointer_key,
                        payload=payload,
                        row=pointer,
                        revision=int(pointer.get("last_authoritative_seq", 1) or 1),
                        status="retired",
                    )
                )
        if updates:
            batch = getattr(self.projections._metadata, "compare_and_swap_named_projections", None)
            if not callable(batch):
                raise TypeError("metadata must support atomic named projection batch CAS")
            if not batch(updates):
                raise ValueError("skill/provider cleanup changed concurrently")
        self.projections._refresh()
        self.catalog._refresh()
        return tuple(sorted({*skill_keys, *catalog_keys}))

    def reconcile(self) -> int:
        """Repair incomplete current/graph/catalog promotion from revisions."""

        self.projections._refresh()
        self.catalog._refresh()
        updates: list[dict[str, Any]] = []
        repaired: set[str] = set()
        for logical_key, history in self.projections._history.items():
            pointer = self.projections._metadata.get_named_projection(
                self.projections.namespace,
                self.projections._pointer_key(logical_key),
            )
            if pointer is not None and pointer.get("materialization_status") == "retired":
                continue
            if not history:
                continue
            artifact = max(history, key=lambda item: item.projection_revision)
            projected, artifact_updates = self.projections.prepare_upsert_update(artifact)
            updates.extend(artifact_updates)
            updates.extend(self.projections.prepare_graph_updates(projected))
            for entry in catalog_entries_from_artifact(projected):
                _entry, catalog_updates = self.catalog.prepare_upsert_update(entry)
                updates.extend(catalog_updates)
            repaired.add(logical_key)
        if updates:
            batch = getattr(
                self.projections._metadata,
                "compare_and_swap_named_projections",
                None,
            )
            if not callable(batch):
                raise TypeError("metadata must support atomic named projection batch CAS")
            if not batch(updates):
                raise ValueError("skill projection reconciliation changed concurrently")
        self.projections._refresh()
        self.catalog._refresh()
        return len(repaired)


class _AsyncProjectionSnapshot:
    """Sync preparation view backed by one async-store snapshot.

    Skill/catalog update construction is pure once rows are loaded. This
    adapter keeps that existing logic while applying its final CAS batch with
    native async I/O; it never bridges backend calls through a worker thread.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = {
            (str(row["namespace"]), str(row["key"])): copy.deepcopy(row)
            for row in rows
        }
        self.pending: list[dict[str, Any]] = []

    def get_named_projection(self, namespace: str, key: str) -> dict[str, Any] | None:
        row = self._rows.get((str(namespace), str(key)))
        return copy.deepcopy(row) if row is not None else None

    def list_named_projections(self, namespace: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(row)
            for (row_namespace, _), row in sorted(self._rows.items())
            if row_namespace == str(namespace)
        ]

    def compare_and_swap_named_projection(
        self, namespace: str, key: str, payload: dict[str, Any], **values: Any
    ) -> bool:
        return self.compare_and_swap_named_projections(
            [{"namespace": namespace, "key": key, "payload": payload, **values}]
        )

    def compare_and_swap_named_projections(self, updates: list[dict[str, Any]]) -> bool:
        for item in updates:
            identity = (str(item["namespace"]), str(item["key"]))
            current = self._rows.get(identity)
            expected_a = item.get("expected_last_authoritative_seq")
            expected_m = item.get("expected_last_materialized_seq")
            if expected_a is None and expected_m is None:
                if current is not None:
                    return False
            elif (
                current is None
                or int(current.get("last_authoritative_seq", 0)) != int(expected_a)
                or int(current.get("last_materialized_seq", 0)) != int(expected_m)
            ):
                return False
        for item in updates:
            row = {
                "namespace": str(item["namespace"]),
                "key": str(item["key"]),
                "payload": copy.deepcopy(item["payload"]),
                "last_authoritative_seq": int(item.get("last_authoritative_seq", 0)),
                "last_materialized_seq": int(item.get("last_materialized_seq", 0)),
                "projection_schema_version": int(item.get("projection_schema_version", 1)),
                "materialization_status": str(item.get("materialization_status", "ready")),
                "updated_at_ms": 0,
            }
            self._rows[(row["namespace"], row["key"])] = row
        self.pending.extend(copy.deepcopy(updates))
        return True


class AsyncDurableSkillCatalogMaterializer:
    """Async durable skill/catalog materialization over one async CAS store.

    Existing synchronous preparation and validation are reused against a
    point-in-time metadata snapshot. Only backend reads and the final atomic
    batch are awaited, so PostgreSQL adapters remain genuinely async and no
    broad ``AsyncEngineFacade`` is introduced.
    """

    def __init__(
        self,
        *,
        metadata: AsyncNamedProjectionStore,
        tenant_id: str | None,
        project_id: str | None,
    ) -> None:
        self.metadata = metadata
        self.tenant_id = tenant_id
        self.project_id = project_id

    async def _prepare(self) -> tuple[_AsyncProjectionSnapshot, DurableSkillCatalogMaterializer]:
        projection_namespace = scoped_projection_namespace(
            DurableSkillProjectionStore.namespace,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        )
        catalog_namespace = scoped_projection_namespace(
            "agent_catalog",
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        )
        projection_rows, catalog_rows = await asyncio.gather(
            self.metadata.list_named_projections(projection_namespace),
            self.metadata.list_named_projections(catalog_namespace),
        )
        snapshot = _AsyncProjectionSnapshot([*projection_rows, *catalog_rows])
        projections = DurableSkillProjectionStore(
            snapshot,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        )
        catalog = DurableCatalogStore(
            snapshot,
            tenant_id=self.tenant_id,
            project_id=self.project_id,
        )
        materializer = DurableSkillCatalogMaterializer(
            projections=projections,
            catalog=catalog,
        )
        return snapshot, materializer

    async def _apply(self, snapshot: _AsyncProjectionSnapshot) -> None:
        if snapshot.pending and not await self.metadata.compare_and_swap_named_projections(
            snapshot.pending
        ):
            raise ValueError("skill/catalog projection changed concurrently")

    async def materialize(
        self,
        artifact: SkillGraphArtifact,
        *,
        guard_updates: list[dict[str, Any]] | None = None,
    ) -> SkillGraphArtifact:
        snapshot, materializer = await self._prepare()
        result = materializer.materialize(artifact, guard_updates=guard_updates)
        await self._apply(snapshot)
        return result

    async def remove_provider(
        self,
        provider_id: str,
        *,
        provider_version: str | None = None,
        lifecycle_token: str | None = None,
    ) -> tuple[str, ...]:
        snapshot, materializer = await self._prepare()
        result = materializer.remove_provider(
            provider_id,
            provider_version=provider_version,
            lifecycle_token=lifecycle_token,
        )
        await self._apply(snapshot)
        return result

    async def reconcile(self) -> int:
        snapshot, materializer = await self._prepare()
        result = materializer.reconcile()
        await self._apply(snapshot)
        return result


def select_skill_subgraph(
    artifact: SkillGraphArtifact, *, node_ids: list[str], max_nodes: int = 32
) -> SkillGraphArtifact:
    """Return bounded selected nodes/edges, preserving source attribution."""

    selected = set(node_ids)
    if len(selected) > max_nodes:
        raise ValueError("selected skill subgraph exceeds bound")
    nodes = [node for node in artifact.nodes if node.node_id in selected]
    node_set = {node.node_id for node in nodes}
    edges = [
        edge
        for edge in artifact.edges
        if set(edge.source_ids + edge.target_ids) <= node_set
    ]
    return artifact.model_copy(update={"nodes": nodes, "edges": edges})


def prepare_skill_execution(
    artifact: SkillGraphArtifact,
    *,
    node_ids: list[str],
    effective_capabilities: set[str] | frozenset[str],
    execution_policy: SkillExecutionPolicy | None = None,
) -> SkillExecutionPlan:
    """Prepare evidence/capability references; never runs a skill itself."""

    selected = select_skill_subgraph(artifact, node_ids=node_ids)
    caps = sorted(
        {
            cap
            for node in selected.nodes
            for cap in (
                list(node.required_capabilities)
                + ([node.name] if node.kind == "capability" else [])
            )
        }
    )
    missing = set(caps) - set(effective_capabilities)
    if missing:
        raise PermissionError(
            "skill execution requires absent capabilities: " + ", ".join(sorted(missing))
        )
    effectful = {"script_call", "command_template", "capability_call", "mcp_call", "nested_workflow"}
    for node in selected.nodes:
        if node.kind in effectful and (
            not node.invocable or node.binding_status != "validated"
        ):
            raise PermissionError(f"skill node is not authorized for invocation: {node.node_id}")
    if any(node.kind in effectful for node in selected.nodes) and execution_policy is None:
        raise PermissionError("effectful skill execution requires an explicit execution policy")
    return SkillExecutionPlan(
        skill_id=f"{artifact.provider_id}:{artifact.provider_local_id}",
        node_ids=[node.node_id for node in selected.nodes],
        source_refs=[str(node.source_ref) for node in selected.nodes if node.source_ref],
        required_capabilities=caps,
        mcp_schema_refs=sorted(
            {node.name for node in selected.nodes if node.kind == "mcp"}
        ),
        evidence_refs=[str(node.source_ref) for node in selected.nodes if node.source_ref],
        source_fingerprint=artifact.source_fingerprint,
        projection_revision=artifact.projection_revision,
        execution_policy=execution_policy,
    )


def attach_glossary_references(
    artifact: SkillGraphArtifact,
    *,
    glossary_entity_ids: Mapping[str, str],
) -> SkillGraphArtifact:
    """Attach project-knowledge references without copying glossary authority."""

    nodes = []
    for node in artifact.nodes:
        reference = glossary_entity_ids.get(node.node_id)
        if reference is None:
            nodes.append(node)
            continue
        metadata = dict(node.metadata)
        metadata["glossary_entity_ids"] = [str(reference)]
        nodes.append(node.model_copy(update={"metadata": metadata}))
    return artifact.model_copy(update={"nodes": nodes})


def build_skill_execution_evidence(
    plan: SkillExecutionPlan,
    *,
    run_id: str,
    workflow_step_ref: str,
    outcome: Literal["success", "failure", "cancelled"],
    result_ref: str | None = None,
) -> SkillExecutionEvidence:
    """Build evidence payload; caller persists it through normal workflow state."""

    return SkillExecutionEvidence(
        run_id=run_id,
        workflow_step_ref=workflow_step_ref,
        skill_id=plan.skill_id,
        node_ids=list(plan.node_ids),
        source_refs=list(plan.source_refs),
        required_capabilities=list(plan.required_capabilities),
        mcp_schema_refs=list(plan.mcp_schema_refs),
        source_fingerprint=plan.source_fingerprint,
        projection_revision=plan.projection_revision,
        result_ref=result_ref,
        outcome=outcome,
    )


def _frontmatter(text: str) -> tuple[dict[str, str], list[str], int]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, lines, 1
    metadata: dict[str, str] = {}
    end = None
    for index, line in enumerate(lines[1:], start=2):
        if line.strip() == "---":
            end = index
            break
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip().strip("\"'")
    if end is None:
        return {}, lines, 1
    return metadata, lines[end:], end + 1


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def parse_skill_text(
    text: str,
    *,
    provider_id: str,
    provider_local_id: str,
    skill_version: str = "v1",
    source_fingerprint: str | None = None,
    max_nodes: int = 256,
    max_edges: int = 512,
) -> SkillGraphArtifact:
    """Parse Markdown/frontmatter without network, LLM, or code execution."""

    raw = str(text)
    fingerprint = source_fingerprint or hashlib.sha256(raw.encode("utf-8")).hexdigest()
    metadata, lines, line_offset = _frontmatter(raw)
    root_id = f"skill:{provider_id}:{provider_local_id}"
    nodes = [
        SkillGraphNode(
            node_id=root_id,
            kind="skill",
            name=metadata.get("name", provider_local_id),
            summary=metadata.get("description", ""),
            source_ref=f"{provider_id}:{provider_local_id}#frontmatter",
        )
    ]
    edges: list[SkillGraphEdge] = []
    warnings: list[str] = []
    unsupported: list[str] = []
    previous_id = root_id
    in_fence = False
    fence_lang = ""
    fence_lines: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=line_offset):
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = stripped[3:].strip().lower()
                fence_lines = []
            else:
                if fence_lang in {"bash", "sh", "shell", "powershell", "pwsh"}:
                    for command_line, command in fence_lines:
                        if not command.strip() or command.lstrip().startswith("#"):
                            continue
                        node_id = _stable_id("step", f"{provider_id}:{provider_local_id}:{command}")
                        try:
                            argv = validate_command_argv(command)
                        except ValueError as exc:
                            unsupported.append(f"line {command_line}: {exc}")
                            warnings.append(f"unsupported command at line {command_line}")
                            continue
                        nodes.append(
                            SkillGraphNode(
                                node_id=node_id,
                                kind="command_template",
                                name=argv[0],
                                summary=command,
                                source_ref=f"{provider_id}:{provider_local_id}#L{command_line}",
                                required_capabilities=["process.execute"],
                                metadata={"argv": argv},
                            )
                        )
                        edges.append(
                            SkillGraphEdge(
                                edge_id=f"edge:{previous_id}:{node_id}",
                                kind="sequence",
                                source_ids=[previous_id],
                                target_ids=[node_id],
                            )
                        )
                        previous_id = node_id
                elif fence_lang:
                    unsupported.append(f"fenced language not executable: {fence_lang}")
                in_fence = False
                fence_lang = ""
                fence_lines = []
            continue
        if in_fence:
            fence_lines.append((index, line))
            continue
        if not stripped or stripped.startswith("#"):
            continue
        lower = stripped.lower()
        if lower.startswith("capability:"):
            capability = stripped.split(":", 1)[1].strip()
            node_id = _stable_id("capability", capability)
            nodes.append(SkillGraphNode(node_id=node_id, kind="capability", name=capability, source_ref=f"{provider_id}:{provider_local_id}#L{index}"))
            edges.append(SkillGraphEdge(edge_id=f"edge:{root_id}:{node_id}", kind="requires", source_ids=[root_id], target_ids=[node_id]))
            continue
        if lower.startswith("mcp:"):
            tool = stripped.split(":", 1)[1].strip()
            node_id = _stable_id("mcp", tool)
            nodes.append(SkillGraphNode(node_id=node_id, kind="mcp", name=tool, source_ref=f"{provider_id}:{provider_local_id}#L{index}"))
            edges.append(SkillGraphEdge(edge_id=f"edge:{root_id}:{node_id}", kind="may_invoke", source_ids=[root_id], target_ids=[node_id]))
            continue
        if lower.startswith("script:"):
            script_path = validate_package_relative_path(stripped.split(":", 1)[1].strip())
            node_id = _stable_id("script", f"{provider_id}:{provider_local_id}:{script_path}")
            nodes.append(
                SkillGraphNode(
                    node_id=node_id,
                    kind="script_call",
                    name=script_path.rsplit("/", 1)[-1],
                    summary=script_path,
                    source_ref=f"{provider_id}:{provider_local_id}#L{index}",
                    required_capabilities=["process.execute"],
                    metadata={"package_relative_path": script_path},
                )
            )
            edges.append(SkillGraphEdge(edge_id=f"edge:{previous_id}:{node_id}", kind="sequence", source_ids=[previous_id], target_ids=[node_id]))
            previous_id = node_id
            continue
        if re.match(r"^(?:[-*]|\d+[.)])\s+", stripped):
            instruction = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", stripped)
            node_id = _stable_id("step", f"{provider_id}:{provider_local_id}:{instruction}")
            nodes.append(SkillGraphNode(node_id=node_id, kind="instruction", name=instruction[:80], summary=instruction, source_ref=f"{provider_id}:{provider_local_id}#L{index}"))
            edges.append(SkillGraphEdge(edge_id=f"edge:{previous_id}:{node_id}", kind="sequence", source_ids=[previous_id], target_ids=[node_id]))
            previous_id = node_id
    if len(nodes) > max_nodes or len(edges) > max_edges:
        raise ValueError("skill input exceeds parser bounds")
    return SkillGraphArtifact(
        provider_id=provider_id,
        provider_local_id=provider_local_id,
        skill_version=skill_version,
        source_fingerprint=fingerprint,
        nodes=nodes,
        edges=edges,
        warnings=warnings,
        unsupported=unsupported,
        provenance={"source_kind": "provider_native_skill", "frontmatter": metadata},
    )


def catalog_entries_from_artifact(artifact: SkillGraphArtifact) -> list[CatalogEntry]:
    """Project artifact nodes into catalog descriptors without executing them."""

    root_id = f"{artifact.provider_id}:{artifact.provider_local_id}"
    entries: list[CatalogEntry] = []
    for node in artifact.nodes:
        logical_id = root_id if node.node_id == f"skill:{artifact.provider_id}:{artifact.provider_local_id}" else f"{root_id}#{node.node_id}"
        entries.append(
            CatalogEntry(
                logical_id=logical_id,
                provider_id=artifact.provider_id,
                # Every graph node is still sourced from the same provider
                # artifact.  Keep provider-local identity stable so raw
                # progressive disclosure can load the source document.
                provider_local_id=artifact.provider_local_id,
                provider_version=str(artifact.provenance.get("provider_version") or "v1"),
                kind=str(node.kind),
                name=node.name,
                summary=node.summary,
                version=artifact.skill_version,
                source_fingerprint=artifact.source_fingerprint,
                revision=artifact.projection_revision,
                group_ids=[root_id],
                required_capabilities=list(node.required_capabilities),
                semantic_ready=False,
                metadata={
                    "source_ref": node.source_ref,
                    "binding_status": node.binding_status,
                    "invocable": node.invocable,
                    "artifact_provider_local_id": artifact.provider_local_id,
                    "artifact_node_id": node.node_id,
                    "artifact_fingerprint": artifact.artifact_fingerprint(),
                    **node.metadata,
                    **(
                        {
                            "provider_lifecycle_token": artifact.provenance[
                                "provider_lifecycle_token"
                            ]
                        }
                        if artifact.provenance.get("provider_lifecycle_token") is not None
                        else {}
                    ),
                },
                tenant_id=artifact.tenant_id,
                project_id=artifact.project_id,
            )
        )
    return entries
