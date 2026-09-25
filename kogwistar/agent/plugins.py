"""Optional discovery plugins; providers remain outside execution authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .catalog import CatalogStore
from .skills import (
    SkillGraphArtifact,
    SkillGraphEdge,
    SkillProjectionStore,
    catalog_entries_from_artifact,
    parse_skill_text,
    validate_skill_artifact,
)


class FilesystemSkillProvider:
    """Read-only package descriptor provider for local Markdown skills."""

    provider_version = "v1"

    def __init__(self, root: str | Path, *, provider_id: str = "filesystem.skills") -> None:
        self.provider_id = provider_id
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("skill provider root must be a directory")

    def descriptors(self) -> list[Mapping[str, Any]]:
        values: list[Mapping[str, Any]] = []
        for path in sorted(self.root.rglob("*.md")):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            local_id = path.relative_to(self.root).as_posix()
            values.append(
                {
                    "provider_local_id": local_id,
                    "kind": "skill",
                    "name": path.stem,
                    "summary": raw.decode("utf-8", errors="replace").splitlines()[0][:240] if raw else "",
                    "source_fingerprint": hashlib.sha256(raw).hexdigest(),
                    "metadata": {"source_path": local_id, "source_kind": "filesystem"},
                }
            )
        return values

    def load(self, provider_local_id: str) -> str:
        candidate = (self.root / str(provider_local_id)).resolve()
        if self.root != candidate and self.root not in candidate.parents:
            raise PermissionError("skill path escapes provider root")
        if candidate.suffix.lower() != ".md" or not candidate.is_file():
            raise FileNotFoundError(str(provider_local_id))
        return candidate.read_text(encoding="utf-8")

    def close(self) -> None:
        return None


class McpDiscoveryProvider:
    """Discovery/schema facade; invocation is an explicitly separate callback."""

    provider_version = "v1"

    def __init__(
        self,
        descriptors: Callable[[], list[Mapping[str, Any]]],
        *,
        provider_id: str = "mcp.discovery",
        describe: Callable[[str], Mapping[str, Any]] | None = None,
        invoke: Callable[..., Any] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._descriptors = descriptors
        self._describe = describe
        self._invoke = invoke

    def descriptors(self) -> list[Mapping[str, Any]]:
        return [dict(item) for item in self._descriptors()]

    def describe(self, provider_local_id: str) -> Mapping[str, Any]:
        if self._describe is None:
            raise LookupError("MCP schema loading is unavailable")
        return dict(self._describe(provider_local_id))

    def invoke(self, provider_local_id: str, **kwargs: Any) -> Any:
        if self._invoke is None:
            raise PermissionError("MCP invocation is not configured")
        return self._invoke(provider_local_id, **kwargs)

    def close(self) -> None:
        return None


def select_mcp_schemas(
    provider: McpDiscoveryProvider,
    provider_local_ids: list[str] | tuple[str, ...],
    *,
    authorize: Callable[[str], bool] | None = None,
    max_schemas: int = 16,
    max_bytes: int = 64 * 1024,
) -> dict[str, Mapping[str, Any]]:
    """Load only selected, authorized, bounded schemas for one model call."""

    if len(provider_local_ids) > max_schemas:
        raise ValueError("selected MCP schema count exceeds bound")
    selected: dict[str, Mapping[str, Any]] = {}
    total_bytes = 0
    for local_id in provider_local_ids:
        key = str(local_id)
        if authorize is not None and not authorize(key):
            raise PermissionError(f"MCP schema is not authorized: {key}")
        schema = dict(provider.describe(key))
        encoded = len(json.dumps(schema, sort_keys=True, default=str).encode("utf-8"))
        total_bytes += encoded
        if total_bytes > max_bytes:
            raise ValueError("selected MCP schemas exceed byte bound")
        selected[key] = schema
    return selected


def ingest_filesystem_skill(
    provider: FilesystemSkillProvider,
    provider_local_id: str,
    *,
    projection: SkillProjectionStore,
    catalog: CatalogStore | None = None,
    skill_version: str = "v1",
    tenant_id: str | None = None,
    project_id: str | None = None,
) -> SkillGraphArtifact:
    """Parse/project local skill while retaining provider-native source."""

    raw = provider.load(provider_local_id)
    fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    artifact = parse_skill_text(
        raw,
        provider_id=provider.provider_id,
        provider_local_id=provider_local_id,
        skill_version=skill_version,
        source_fingerprint=fingerprint,
    ).model_copy(
        update={
            "tenant_id": tenant_id,
            "project_id": project_id,
            "namespace": f"project:{project_id}" if project_id else None,
            "provenance": {
                "source_kind": "filesystem_skill",
                "provider_version": provider.provider_version,
                "provider_local_id": provider_local_id,
            },
        }
    )
    validate_skill_artifact(
        artifact,
        expected_fingerprint=fingerprint,
        tenant_id=tenant_id,
        project_id=project_id,
    )
    projected = projection.upsert(artifact)
    if catalog is not None:
        for entry in catalog_entries_from_artifact(projected):
            catalog.upsert(entry)
    return projected


class LlmWikiIngestionAdapter:
    """Optional semantic provider adapter; no llm-wiki import is mandatory."""

    provider_id = "kogwistar-llm-wiki"
    parser_id = "llm-wiki.semantic"
    parser_version = "v1"

    def __init__(
        self,
        parse: Callable[[Mapping[str, Any]], SkillGraphArtifact],
        *,
        authorize: Callable[[Mapping[str, Any]], bool] | None = None,
        max_source_bytes: int = 64 * 1024,
    ) -> None:
        self._parse = parse
        self._authorize = authorize
        self._max_source_bytes = max(1, int(max_source_bytes))

    def parse(self, source: Mapping[str, Any]) -> SkillGraphArtifact:
        request = dict(source)
        if self._authorize is None or not self._authorize(request):
            raise PermissionError("LLM-Wiki skill ingestion requires authorization")
        encoded = json.dumps(request, sort_keys=True, default=str).encode("utf-8")
        if len(encoded) > self._max_source_bytes:
            raise ValueError("LLM-Wiki skill source exceeds byte bound")
        artifact = self._parse(request)
        validate_skill_artifact(artifact)
        return artifact

    def close(self) -> None:
        return None


def mark_inferred_edges_as_candidates(
    artifact: SkillGraphArtifact,
    *,
    confidence: float | None = None,
) -> SkillGraphArtifact:
    """Keep semantic-provider links reviewable and non-authoritative."""

    if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("inferred-edge confidence must be between 0 and 1")
    edges: list[SkillGraphEdge] = []
    for edge in artifact.edges:
        metadata = dict(edge.metadata)
        metadata.update(
            {
                "inferred": True,
                "source_provider": artifact.provider_id,
                "parser_id": artifact.parser_id,
                "parser_version": artifact.parser_version,
            }
        )
        if confidence is not None:
            metadata["confidence"] = float(confidence)
        edges.append(edge.model_copy(update={"status": "candidate", "metadata": metadata}))
    return artifact.model_copy(update={"edges": edges})


def deduplicate_inferred_edges(artifact: SkillGraphArtifact) -> SkillGraphArtifact:
    """Collapse semantic-provider duplicate links without approving them."""

    selected: dict[tuple[str, str, str], SkillGraphEdge] = {}
    for edge in artifact.edges:
        key = (edge.kind, ",".join(sorted(edge.source_ids)), ",".join(sorted(edge.target_ids)))
        current = selected.get(key)
        confidence = float(edge.metadata.get("confidence", 0.0) or 0.0)
        current_confidence = float(current.metadata.get("confidence", 0.0) or 0.0) if current else -1.0
        if current is None or confidence > current_confidence:
            selected[key] = edge
    return artifact.model_copy(update={"edges": list(selected.values())})


@dataclass(frozen=True, slots=True)
class ProjectPluginManifest:
    """Declarative project extension; it carries no execution authority."""

    provider_id: str
    project_id: str
    tenant_id: str
    skill_ids: tuple[str, ...] = ()
    glossary_ids: tuple[str, ...] = ()
    policy_ids: tuple[str, ...] = ()
    tool_ids: tuple[str, ...] = ()


class ProjectGlossaryProvider:
    """ACL-filtered exact/alias project terminology source."""

    provider_version = "v1"

    def __init__(self, manifest: ProjectPluginManifest, terms: list[Mapping[str, Any]]) -> None:
        self.provider_id = manifest.provider_id
        self.project_id = manifest.project_id
        self.tenant_id = manifest.tenant_id
        self._terms = tuple(dict(term) for term in terms)

    def descriptors(self) -> list[Mapping[str, Any]]:
        result: list[Mapping[str, Any]] = []
        for term in self._terms:
            item = dict(term)
            item.setdefault("provider_id", self.provider_id)
            item.setdefault("kind", "glossary_term")
            item.setdefault("project_id", self.project_id)
            item.setdefault("tenant_id", self.tenant_id)
            result.append(item)
        return result

    def search(
        self,
        query: str,
        *,
        authorize: Callable[[Mapping[str, Any]], bool] | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[Mapping[str, Any]]:
        if authorize is None:
            raise PermissionError("project glossary search requires ACL callback")
        needle = str(query).strip().casefold()
        if not needle or limit < 1:
            return []
        results: list[Mapping[str, Any]] = []
        for term in self.descriptors():
            if tenant_id is not None and term.get("tenant_id") not in (None, tenant_id):
                continue
            if project_id is not None and term.get("project_id") not in (None, project_id):
                continue
            if not authorize(term):
                continue
            haystack = " ".join(
                [str(term.get("term", "")), *(str(item) for item in term.get("aliases", ())), str(term.get("definition", ""))]
            ).casefold()
            if needle in haystack:
                results.append(term)
            if len(results) >= int(limit):
                break
        return results

    def close(self) -> None:
        return None


def ingest_project_glossary_to_knowledge(
    provider: ProjectGlossaryProvider,
    *,
    write: Callable[[Mapping[str, Any]], str],
) -> tuple[str, ...]:
    """Emit scoped knowledge records; canonical knowledge writer owns storage."""

    refs: list[str] = []
    for term in provider.descriptors():
        record = {
            "entity_id": str(term.get("id") or f"{provider.provider_id}:{term.get('term', '')}"),
            "kind": "project_glossary_term",
            "term": str(term.get("term", "")),
            "aliases": [str(value) for value in term.get("aliases", ())],
            "definition": str(term.get("definition", "")),
            "source": term.get("source"),
            "valid_from": term.get("valid_from"),
            "valid_to": term.get("valid_to"),
            "tenant_id": provider.tenant_id,
            "project_id": provider.project_id,
            "namespace": f"project:{provider.project_id}:knowledge",
            "provider_id": provider.provider_id,
        }
        refs.append(str(write(record)))
    return tuple(refs)


__all__ = [
    "FilesystemSkillProvider",
    "LlmWikiIngestionAdapter",
    "McpDiscoveryProvider",
    "ingest_filesystem_skill",
    "mark_inferred_edges_as_candidates",
    "deduplicate_inferred_edges",
    "ProjectGlossaryProvider",
    "ProjectPluginManifest",
    "select_mcp_schemas",
    "ingest_project_glossary_to_knowledge",
]
