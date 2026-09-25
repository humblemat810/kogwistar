"""Deterministic, provider-neutral skill-to-hypergraph ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .catalog import CatalogEntry

SkillStepKind = Literal[
    "instruction",
    "capability_call",
    "mcp_call",
    "script_call",
    "command_template",
    "nested_workflow",
    "check",
]

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
        self.lane_id = f"ws:{self.project_id}:g:projection:lane:skills"
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
    allowed_node_kinds: set[str] | frozenset[str] | None = None,
) -> SkillGraphArtifact:
    """Validate source/scope/bindings before any projection write."""

    if expected_fingerprint is not None and artifact.source_fingerprint != expected_fingerprint:
        raise ValueError("skill source fingerprint mismatch")
    if tenant_id is not None and artifact.tenant_id not in (None, tenant_id):
        raise PermissionError("skill artifact tenant scope mismatch")
    if project_id is not None and artifact.project_id not in (None, project_id):
        raise PermissionError("skill artifact project scope mismatch")
    if allowed_node_kinds is not None:
        unknown = {str(node.kind) for node in artifact.nodes} - set(allowed_node_kinds)
        if unknown:
            raise ValueError(f"unsupported skill node kinds: {sorted(unknown)}")
    return artifact


class SkillProjectionStore:
    """Rebuildable current projection with stale-revision rejection."""

    def __init__(self) -> None:
        self._current: dict[str, SkillGraphArtifact] = {}
        self._history: dict[str, list[SkillGraphArtifact]] = {}

    def upsert(self, artifact: SkillGraphArtifact) -> SkillGraphArtifact:
        validate_skill_artifact(artifact)
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
                return current
        self._current[key] = artifact
        self._history.setdefault(key, []).append(artifact)
        return artifact

    def get(self, provider_id: str, provider_local_id: str) -> SkillGraphArtifact | None:
        return self._current.get(f"{provider_id}:{provider_local_id}")

    def remove(self, provider_id: str, provider_local_id: str) -> None:
        self._current.pop(f"{provider_id}:{provider_local_id}", None)

    def remove_provider(self, provider_id: str) -> tuple[str, ...]:
        removed = tuple(
            key for key in self._current if key.startswith(f"{provider_id}:")
        )
        for key in removed:
            self._current.pop(key, None)
        return removed

    def history(self, provider_id: str, provider_local_id: str) -> tuple[SkillGraphArtifact, ...]:
        return tuple(self._history.get(f"{provider_id}:{provider_local_id}", ()))


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
    for node in selected.nodes:
        if node.invocable and node.binding_status != "validated":
            raise ValueError(f"skill node is not validated: {node.node_id}")
    effectful = {"script_call", "command_template", "capability_call", "mcp_call", "nested_workflow"}
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
                provider_local_id=node.node_id,
                provider_version=str(artifact.provenance.get("provider_version") or "v1"),
                kind=str(node.kind),
                name=node.name,
                summary=node.summary,
                version=artifact.skill_version,
                source_fingerprint=artifact.source_fingerprint,
                group_ids=[root_id],
                required_capabilities=list(node.required_capabilities),
                semantic_ready=False,
                metadata={
                    "source_ref": node.source_ref,
                    "binding_status": node.binding_status,
                    "invocable": node.invocable,
                    **node.metadata,
                },
                tenant_id=artifact.tenant_id,
                project_id=artifact.project_id,
            )
        )
    return entries
