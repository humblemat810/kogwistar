"""Deterministic, provider-neutral skill-to-hypergraph ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from typing import Literal

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
        return self

    def artifact_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


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
            )
        )
    return entries
