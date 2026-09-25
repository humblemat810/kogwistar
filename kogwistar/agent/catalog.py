"""Backend-neutral searchable catalog for skills, tools, and MCP entries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TOKEN_RE = re.compile(r"[\w.-]+", re.UNICODE)
CatalogAcl = Callable[["CatalogEntry", str], bool]


class CatalogEntry(BaseModel):
    """Stable logical catalog record; revisions replace, not duplicate, current."""

    model_config = ConfigDict(extra="allow")

    logical_id: str
    provider_id: str
    provider_local_id: str
    kind: str
    name: str
    provider_version: str = "v1"
    summary: str = ""
    version: str = "v1"
    source_fingerprint: str
    revision: int = 1
    group_ids: list[str] = Field(default_factory=list)
    group_parent_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    scope: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    semantic_ready: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("logical_id", "provider_id", "provider_local_id", "kind", "name")
    @classmethod
    def _required_text(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("catalog identity fields must be non-empty")
        return value

    @field_validator("revision")
    @classmethod
    def _positive_revision(cls, value: int) -> int:
        if value < 1:
            raise ValueError("revision must be positive")
        return value


@dataclass(frozen=True, slots=True)
class CatalogSearchResult:
    entry: CatalogEntry
    score: float
    match: str


class CatalogGroup(BaseModel):
    """Graph-native group descriptor; tree is only a browsing projection."""

    group_id: str
    name: str
    parent_ids: list[str] = Field(default_factory=list)
    scope: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None


class CatalogStore:
    """Deterministic catalog with ACL/scope filtering before ranking."""

    def __init__(
        self,
        *,
        acl_enabled: bool = True,
        acl_checker: CatalogAcl | None = None,
    ) -> None:
        self.acl_enabled = acl_enabled
        self.acl_checker = acl_checker
        self._entries: dict[str, CatalogEntry] = {}
        self._revisions: dict[tuple[str, int], str] = {}
        self._history: dict[str, list[CatalogEntry]] = {}
        self._groups: dict[str, CatalogGroup] = {}

    def upsert_group(self, group: CatalogGroup) -> CatalogGroup:
        self._groups[group.group_id] = group
        return group

    def group_tree(self, *, root_id: str | None = None) -> tuple[CatalogGroup, ...]:
        values = tuple(self._groups.values())
        if root_id is None:
            return tuple(sorted(values, key=lambda item: item.group_id))
        return tuple(
            sorted(
                (item for item in values if root_id in item.parent_ids or item.group_id == root_id),
                key=lambda item: item.group_id,
            )
        )

    def upsert(self, entry: CatalogEntry) -> CatalogEntry:
        current = self._entries.get(entry.logical_id)
        if current is not None:
            if entry.revision < current.revision:
                raise ValueError(f"stale catalog revision: {entry.logical_id}")
            if entry.revision == current.revision:
                if entry.source_fingerprint != current.source_fingerprint:
                    raise ValueError(f"revision fingerprint collision: {entry.logical_id}")
                return current
        self._entries[entry.logical_id] = entry
        self._revisions[(entry.logical_id, entry.revision)] = entry.source_fingerprint
        self._history.setdefault(entry.logical_id, []).append(entry)
        return entry

    def ingest_descriptors(
        self, entries: list[CatalogEntry] | tuple[CatalogEntry, ...]
    ) -> tuple[CatalogEntry, ...]:
        """Materialize validated read descriptors; provider remains source owner."""

        return tuple(self.upsert(entry) for entry in entries)

    def remove_provider(self, provider_id: str) -> tuple[str, ...]:
        removed = tuple(
            logical_id
            for logical_id, entry in self._entries.items()
            if entry.provider_id == provider_id
        )
        for logical_id in removed:
            self._entries.pop(logical_id, None)
        return removed

    def get(self, logical_id: str) -> CatalogEntry | None:
        return self._entries.get(logical_id)

    def history(self, logical_id: str) -> tuple[CatalogEntry, ...]:
        """Return immutable provider/revision history, including unloaded entries."""

        return tuple(self._history.get(logical_id, ()))

    def browse(
        self,
        group_id: str | None = None,
        *,
        principal: str = "system",
        scope: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[CatalogEntry, ...]:
        entries = (
            entry
            for entry in self._entries.values()
            if self._visible(
                entry,
                principal=principal,
                scope=scope,
                tenant_id=tenant_id,
                project_id=project_id,
            )
        )
        if group_id is not None:
            entries = (entry for entry in entries if group_id in entry.group_ids)
        return tuple(sorted(entries, key=lambda entry: entry.logical_id))

    def _visible(
        self,
        entry: CatalogEntry,
        *,
        principal: str,
        scope: str | None,
        tenant_id: str | None,
        project_id: str | None,
    ) -> bool:
        if scope is not None and entry.scope not in (None, scope):
            return False
        if tenant_id is not None and entry.tenant_id not in (None, tenant_id):
            return False
        if project_id is not None and entry.project_id not in (None, project_id):
            return False
        if not self.acl_enabled:
            return True
        if self.acl_checker is None:
            return False
        return bool(self.acl_checker(entry, principal))

    @staticmethod
    def _tokens(entry: CatalogEntry) -> set[str]:
        text = " ".join((entry.name, entry.summary, *entry.group_ids))
        return {token.lower() for token in _TOKEN_RE.findall(text)}

    def search(
        self,
        query: str,
        *,
        principal: str = "system",
        scope: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
        mode: str = "lexical",
        limit: int = 20,
    ) -> tuple[CatalogSearchResult, ...]:
        if limit < 1:
            return ()
        query = str(query).strip().lower()
        if not query:
            return ()
        if mode not in {"lexical", "bm25", "semantic"}:
            raise ValueError("mode must be lexical, bm25, or semantic")
        results: list[CatalogSearchResult] = []
        for entry in self._entries.values():
            if not self._visible(
                entry,
                principal=principal,
                scope=scope,
                tenant_id=tenant_id,
                project_id=project_id,
            ):
                continue
            if mode == "semantic" and not entry.semantic_ready:
                continue
            names = {entry.logical_id.lower(), entry.name.lower()}
            raw_metadata_aliases = entry.metadata.get("aliases", [])
            if isinstance(raw_metadata_aliases, str):
                raw_metadata_aliases = [raw_metadata_aliases]
            elif not isinstance(raw_metadata_aliases, (list, tuple, set)):
                raw_metadata_aliases = []
            raw_aliases = (*entry.aliases, *raw_metadata_aliases)
            aliases = {str(value).lower() for value in raw_aliases}
            if query in names or query in aliases:
                results.append(CatalogSearchResult(entry, 1.0, "exact"))
                continue
            searchable = (*names, *aliases)
            if any(value.startswith(query) for value in searchable):
                results.append(CatalogSearchResult(entry, 0.8, "prefix"))
                continue
            tokens = self._tokens(entry)
            query_tokens = set(_TOKEN_RE.findall(query))
            overlap = len(tokens & query_tokens)
            if overlap:
                results.append(
                    CatalogSearchResult(
                        entry,
                        0.5 + overlap / max(len(query_tokens), 1) / 10,
                        "bm25" if mode == "bm25" else "lexical",
                    )
                )
                continue
            if any(query in value for value in (*searchable, entry.summary.lower())):
                results.append(CatalogSearchResult(entry, 0.3, "partial"))
        return tuple(
            sorted(results, key=lambda result: (-result.score, result.entry.logical_id))[:limit]
        )
