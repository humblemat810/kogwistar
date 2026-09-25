"""Backend-neutral searchable catalog for skills, tools, and MCP entries."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kogwistar.engine_core.embedding_profile import NamedProjectionStore

_TOKEN_RE = re.compile(r"[\w.-]+", re.UNICODE)
_LOG = logging.getLogger(__name__)
CatalogAcl = Callable[["CatalogEntry", str], bool]
CatalogGroupAcl = Callable[["CatalogGroup", str], bool]
CatalogSemanticRanker = Callable[
    [str, tuple["CatalogEntry", ...]], Mapping[str, float]
]
CATALOG_PROJECTION_NAMESPACE = "agent_catalog"


def scoped_projection_namespace(
    base: str,
    *,
    tenant_id: str | None,
    project_id: str | None,
) -> str:
    """Return an unambiguous durable namespace for one tenant/project scope."""

    def component(value: str | None) -> str:
        return "global" if value is None else f"value:{quote(str(value), safe='')}"

    return f"{base}:tenant={component(tenant_id)}:project={component(project_id)}"


def catalog_entry_fingerprint(entry: "CatalogEntry") -> str:
    """Fingerprint normalized descriptor content, not only its source bytes."""

    payload = entry.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        group_acl_checker: CatalogGroupAcl | None = None,
        metadata: Any | None = None,
        projection_namespace: str = CATALOG_PROJECTION_NAMESPACE,
        semantic_ranker: CatalogSemanticRanker | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self.acl_enabled = acl_enabled
        self.acl_checker = acl_checker
        self.group_acl_checker = group_acl_checker
        self._metadata = metadata
        self._projection_namespace = str(projection_namespace)
        self.semantic_ranker = semantic_ranker
        self.tenant_id = tenant_id
        self.project_id = project_id
        self._entries: dict[str, CatalogEntry] = {}
        self._revisions: dict[tuple[str, int], str] = {}
        self._history: dict[str, list[CatalogEntry]] = {}
        self._groups: dict[str, CatalogGroup] = {}

    def upsert_group(self, group: CatalogGroup) -> CatalogGroup:
        if self.tenant_id is not None and group.tenant_id != self.tenant_id:
            raise PermissionError("catalog group tenant scope mismatch")
        if self.project_id is not None and group.project_id != self.project_id:
            raise PermissionError("catalog group project scope mismatch")
        self._groups[group.group_id] = group.model_copy(deep=True)
        return group.model_copy(deep=True)

    def group_tree(
        self,
        *,
        root_id: str | None = None,
        principal: str = "system",
        scope: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[CatalogGroup, ...]:
        values = tuple(
            group
            for group in self._groups.values()
            if self._group_visible(
                group,
                principal=principal,
                scope=scope,
                tenant_id=tenant_id,
                project_id=project_id,
            )
        )
        if root_id is None:
            return tuple(
                item.model_copy(deep=True)
                for item in sorted(values, key=lambda item: item.group_id)
            )
        return tuple(
            item.model_copy(deep=True)
            for item in sorted(
                (item for item in values if root_id in item.parent_ids or item.group_id == root_id),
                key=lambda item: item.group_id,
            )
        )

    def upsert(self, entry: CatalogEntry) -> CatalogEntry:
        if self.tenant_id is not None and entry.tenant_id != self.tenant_id:
            raise PermissionError("catalog entry tenant scope mismatch")
        if self.project_id is not None and entry.project_id != self.project_id:
            raise PermissionError("catalog entry project scope mismatch")
        current = self._entries.get(entry.logical_id)
        if current is not None:
            if entry.revision < current.revision:
                raise ValueError(f"stale catalog revision: {entry.logical_id}")
            if entry.revision == current.revision:
                if catalog_entry_fingerprint(entry) != catalog_entry_fingerprint(current):
                    raise ValueError(f"revision fingerprint collision: {entry.logical_id}")
                return current.model_copy(deep=True)
        self._entries[entry.logical_id] = entry.model_copy(deep=True)
        self._revisions[(entry.logical_id, entry.revision)] = entry.source_fingerprint
        self._history.setdefault(entry.logical_id, []).append(entry.model_copy(deep=True))
        return entry.model_copy(deep=True)

    def ingest_descriptors(
        self, entries: list[CatalogEntry] | tuple[CatalogEntry, ...]
    ) -> tuple[CatalogEntry, ...]:
        """Materialize validated read descriptors; provider remains source owner."""

        return tuple(self.upsert(entry) for entry in entries)

    def remove_provider(
        self,
        provider_id: str,
        *,
        provider_version: str | None = None,
        lifecycle_token: str | None = None,
    ) -> tuple[str, ...]:
        removed = tuple(
            logical_id
            for logical_id, entry in self._entries.items()
            if entry.provider_id == str(provider_id)
            and (provider_version is None or entry.provider_version == provider_version)
            and (
                lifecycle_token is None
                or entry.metadata.get("provider_lifecycle_token") == lifecycle_token
            )
        )
        for logical_id in removed:
            self._entries.pop(logical_id, None)
        return removed

    def get(
        self,
        logical_id: str,
        *,
        principal: str = "system",
        scope: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> CatalogEntry | None:
        entry = self._entries.get(logical_id)
        if entry is None or not self._visible(
            entry,
            principal=principal,
            scope=scope,
            tenant_id=tenant_id,
            project_id=project_id,
        ):
            return None
        return entry.model_copy(deep=True)

    def history(
        self,
        logical_id: str,
        *,
        principal: str = "system",
        scope: str | None = None,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> tuple[CatalogEntry, ...]:
        """Return immutable provider/revision history, including unloaded entries."""

        return tuple(
            entry.model_copy(deep=True)
            for entry in self._history.get(logical_id, ())
            if self._visible(
                entry,
                principal=principal,
                scope=scope,
                tenant_id=tenant_id,
                project_id=project_id,
            )
        )

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
        return tuple(
            entry.model_copy(deep=True)
            for entry in sorted(entries, key=lambda entry: entry.logical_id)
        )

    def _visible(
        self,
        entry: CatalogEntry,
        *,
        principal: str,
        scope: str | None,
        tenant_id: str | None,
        project_id: str | None,
    ) -> bool:
        explicitly_global = bool(
            getattr(entry, "is_global", False)
            or entry.metadata.get("visibility") == "global"
        )
        if (
            self.acl_enabled
            and scope is None
            and entry.scope is not None
            and not explicitly_global
        ):
            return False
        if scope is not None and entry.scope not in (None, scope):
            return False
        exact_scope = self.tenant_id is not None or self.project_id is not None
        if self.tenant_id is not None and tenant_id != self.tenant_id:
            return False
        if self.project_id is not None and project_id != self.project_id:
            return False
        expected_tenant = self.tenant_id if exact_scope else tenant_id
        expected_project = self.project_id if exact_scope else project_id
        if (
            self.acl_enabled
            and expected_tenant is None
            and entry.tenant_id is not None
            and not explicitly_global
        ):
            return False
        if (
            self.acl_enabled
            and expected_project is None
            and entry.project_id is not None
            and not explicitly_global
        ):
            return False
        if expected_tenant is not None and entry.tenant_id != expected_tenant:
            if not (entry.tenant_id is None and explicitly_global):
                return False
        if expected_project is not None and entry.project_id != expected_project:
            if not (entry.project_id is None and explicitly_global):
                return False
        if not self.acl_enabled:
            return True
        if self.acl_checker is None:
            return False
        return bool(self.acl_checker(entry, principal))

    def _group_visible(
        self,
        group: CatalogGroup,
        *,
        principal: str,
        scope: str | None,
        tenant_id: str | None,
        project_id: str | None,
    ) -> bool:
        if self.acl_enabled and scope is None and group.scope is not None:
            return False
        if scope is not None and group.scope not in (None, scope):
            return False
        if self.tenant_id is not None and tenant_id != self.tenant_id:
            return False
        if self.project_id is not None and project_id != self.project_id:
            return False
        exact_scope = self.tenant_id is not None or self.project_id is not None
        expected_tenant = self.tenant_id if exact_scope else tenant_id
        expected_project = self.project_id if exact_scope else project_id
        if (
            self.acl_enabled
            and expected_tenant is None
            and group.tenant_id is not None
        ):
            return False
        if expected_tenant is not None and group.tenant_id != expected_tenant:
            return False
        if (
            self.acl_enabled
            and expected_project is None
            and group.project_id is not None
        ):
            return False
        if expected_project is not None and group.project_id != expected_project:
            return False
        if not self.acl_enabled:
            return True
        return self.group_acl_checker is not None and bool(self.group_acl_checker(group, principal))

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
        visible_entries = [
            entry
            for entry in self._entries.values()
            if self._visible(
                entry,
                principal=principal,
                scope=scope,
                tenant_id=tenant_id,
                project_id=project_id,
            )
        ]
        if mode == "semantic" and self.semantic_ranker is not None:
            candidates = tuple(
                entry.model_copy(deep=True)
                for entry in visible_entries
                if entry.semantic_ready
            )
            try:
                scores = self.semantic_ranker(query, candidates)
                if not isinstance(scores, Mapping):
                    raise TypeError(
                        "semantic_ranker must return a mapping of logical_id to score"
                    )
                ranked: list[CatalogSearchResult] = []
                for entry in candidates:
                    score = float(scores.get(entry.logical_id, 0.0))
                    if math.isfinite(score) and score > 0.0:
                        ranked.append(CatalogSearchResult(entry, score, "semantic"))
                if ranked:
                    return tuple(
                        sorted(
                            ranked,
                            key=lambda result: (-result.score, result.entry.logical_id),
                        )[:limit]
                    )
            except Exception:
                _LOG.warning(
                    "semantic catalog ranking unavailable; using lexical fallback",
                    exc_info=True,
                )
            # Semantic mode remains useful when the optional ranker is absent,
            # unavailable, or has no score. Lexical fallback searches all ACL
            # visible entries, including pending semantic projections.
        # Semantic mode degrades to the strongest built-in lexical surface;
        # BM25 keeps search useful without pretending it was vector-ranked.
        lexical_mode = "bm25" if mode == "semantic" else mode
        document_frequency: dict[str, int] = {}
        for entry in visible_entries:
            for token in self._tokens(entry):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        average_length = (
            sum(len(self._tokens(entry)) for entry in visible_entries) / len(visible_entries)
            if visible_entries
            else 1.0
        )
        results: list[CatalogSearchResult] = []
        query_tokens = set(_TOKEN_RE.findall(query))
        for entry in visible_entries:
            names = {entry.logical_id.lower(), entry.name.lower()}
            raw_metadata_aliases = entry.metadata.get("aliases", [])
            if isinstance(raw_metadata_aliases, str):
                raw_metadata_aliases = [raw_metadata_aliases]
            elif not isinstance(raw_metadata_aliases, (list, tuple, set)):
                raw_metadata_aliases = []
            raw_aliases = (*entry.aliases, *raw_metadata_aliases)
            aliases = {str(value).lower() for value in raw_aliases}
            if query in names or query in aliases:
                results.append(CatalogSearchResult(entry.model_copy(deep=True), 1.0, "exact"))
                continue
            searchable = (*names, *aliases)
            if any(value.startswith(query) for value in searchable):
                results.append(CatalogSearchResult(entry.model_copy(deep=True), 0.8, "prefix"))
                continue
            tokens = self._tokens(entry)
            overlap = len(tokens & query_tokens)
            if overlap:
                if lexical_mode == "bm25":
                    document_length = max(len(tokens), 1)
                    score = 0.0
                    for term in query_tokens:
                        if term not in tokens:
                            continue
                        document_count = document_frequency.get(term, 0)
                        inverse_document_frequency = math.log(
                            1.0
                            + (len(visible_entries) - document_count + 0.5)
                            / (document_count + 0.5)
                        )
                        term_frequency = 1.0
                        normalization = 1.2 * (
                            1.0
                            - 0.75
                            + 0.75 * document_length / max(average_length, 1.0)
                        )
                        score += inverse_document_frequency * (
                            term_frequency * 2.2 / (term_frequency + normalization)
                        )
                    results.append(
                        CatalogSearchResult(entry.model_copy(deep=True), score, "bm25")
                    )
                    continue
                results.append(
                    CatalogSearchResult(
                        entry.model_copy(deep=True),
                        0.5 + overlap / max(len(query_tokens), 1) / 10,
                        "lexical",
                    )
                )
                continue
            if any(query in value for value in (*searchable, entry.summary.lower())):
                results.append(CatalogSearchResult(entry.model_copy(deep=True), 0.3, "partial"))
        return tuple(
            sorted(results, key=lambda result: (-result.score, result.entry.logical_id))[:limit]
        )


class DurableCatalogStore(CatalogStore):
    """CatalogStore backed by an existing named-projection metadata adapter.

    The catalog remains a serving projection. Source packages, runs, and
    workflow authority stay in their existing stores; this adapter only adds
    restart-safe current/history materialization and CAS revision protection.
    """

    def __init__(
        self,
        metadata: NamedProjectionStore,
        *,
        tenant_id: str | None,
        project_id: str | None,
        **kwargs: Any,
    ) -> None:
        self.tenant_id = tenant_id
        self.project_id = project_id
        super().__init__(
            metadata=metadata,
            projection_namespace=scoped_projection_namespace(
                CATALOG_PROJECTION_NAMESPACE,
                tenant_id=tenant_id,
                project_id=project_id,
            ),
            tenant_id=tenant_id,
            project_id=project_id,
            **kwargs,
        )
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

    def _assert_write_scope(self, *, tenant_id: str | None, project_id: str | None) -> None:
        if not self._matches_store_scope(tenant_id=tenant_id, project_id=project_id):
            raise PermissionError("catalog projection scope mismatch")

    @staticmethod
    def _storage_id(logical_id: str) -> str:
        return hashlib.sha256(logical_id.encode("utf-8")).hexdigest()

    @classmethod
    def _pointer_key(cls, logical_id: str) -> str:
        return f"catalog-current:{cls._storage_id(logical_id)}"

    @classmethod
    def _revision_key(cls, logical_id: str, revision: int) -> str:
        return f"catalog-revision:{cls._storage_id(logical_id)}:{int(revision)}"

    def _refresh(self) -> None:
        self._entries.clear()
        self._history.clear()
        self._groups.clear()
        revisions: dict[str, list[CatalogEntry]] = {}
        pointers: dict[str, int] = {}
        legacy_current: dict[str, CatalogEntry] = {}
        rows = self._metadata.list_named_projections(self._projection_namespace)
        for row in rows:
            payload = row.get("payload") or {}
            entry_data = payload.get("entry")
            if isinstance(entry_data, dict):
                entry = CatalogEntry.model_validate(entry_data)
                logical_id = str(payload.get("logical_id") or entry.logical_id)
                if payload.get("record_kind") == "catalog_revision":
                    revisions.setdefault(logical_id, []).append(entry)
                else:  # Legacy current/history blob; retain read compatibility.
                    legacy_current[logical_id] = entry
            history_data = payload.get("history") or []
            if history_data and not str(row.get("key", "")).startswith("group:"):
                self._history[str(row["key"])] = [CatalogEntry.model_validate(item) for item in history_data]
            if payload.get("record_kind") == "catalog_current" and row.get(
                "materialization_status"
            ) != "retired":
                pointers[str(payload["logical_id"])] = int(payload["current_revision"])
            group_data = payload.get("group")
            if isinstance(group_data, dict):
                group = CatalogGroup.model_validate(group_data)
                self._groups[group.group_id] = group
        for logical_id, values in revisions.items():
            self._history[logical_id] = sorted(values, key=lambda entry: entry.revision)
        self._entries.update(legacy_current)
        for logical_id, revision in pointers.items():
            entry = next(
                (
                    item
                    for item in self._history.get(logical_id, ())
                    if item.revision == revision
                ),
                None,
            )
            if entry is not None:
                self._entries[logical_id] = entry

    @staticmethod
    def _cas_values(row: dict[str, Any] | None) -> tuple[int | None, int | None]:
        if row is None:
            return None, None
        return int(row.get("last_authoritative_seq", 0)), int(row.get("last_materialized_seq", 0))

    def _cas(self, key: str, payload: dict[str, Any], row: dict[str, Any] | None, revision: int) -> None:
        expected_a, expected_m = self._cas_values(row)
        if not self._metadata.compare_and_swap_named_projection(
            self._projection_namespace,
            key,
            payload,
            expected_last_authoritative_seq=expected_a,
            expected_last_materialized_seq=expected_m,
            last_authoritative_seq=int(revision),
            last_materialized_seq=int(revision),
            projection_schema_version=1,
            materialization_status="ready",
        ):
            raise ValueError(f"catalog projection changed concurrently: {key}")

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
        expected_a, expected_m = DurableCatalogStore._cas_values(row)
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

    def prepare_upsert_update(self, entry: CatalogEntry) -> tuple[CatalogEntry, list[dict[str, Any]]]:
        """Build immutable revision/current-pointer CAS updates without applying."""

        self._assert_write_scope(tenant_id=entry.tenant_id, project_id=entry.project_id)
        self._refresh()
        current = self._entries.get(entry.logical_id)
        pointer_key = self._pointer_key(entry.logical_id)
        pointer_row = self._metadata.get_named_projection(self._projection_namespace, pointer_key)
        history = self._history.get(entry.logical_id, [])
        latest = history[-1] if history else current
        if latest is not None:
            if entry.revision < latest.revision:
                raise ValueError(f"stale catalog revision: {entry.logical_id}")
            if entry.revision == latest.revision:
                if catalog_entry_fingerprint(entry) != catalog_entry_fingerprint(latest):
                    raise ValueError(f"revision content collision: {entry.logical_id}")
                if current is not None:
                    return latest.model_copy(deep=True), []
        revision_key = self._revision_key(entry.logical_id, entry.revision)
        revision_row = self._metadata.get_named_projection(
            self._projection_namespace, revision_key
        )
        updates: list[dict[str, Any]] = []
        if revision_row is None:
            updates.append(
                self._update(
                    namespace=self._projection_namespace,
                    key=revision_key,
                    payload={
                        "record_kind": "catalog_revision",
                        "logical_id": entry.logical_id,
                        "entry": entry.model_dump(mode="json"),
                        "entry_fingerprint": catalog_entry_fingerprint(entry),
                    },
                    row=None,
                    revision=entry.revision,
                )
            )
        updates.append(
            self._update(
                namespace=self._projection_namespace,
                key=pointer_key,
                payload={
                    "record_kind": "catalog_current",
                    "logical_id": entry.logical_id,
                    "current_revision": entry.revision,
                    "revision_key": revision_key,
                },
                row=pointer_row,
                revision=entry.revision,
            )
        )
        return entry.model_copy(deep=True), updates

    def apply_prepared_updates(self, updates: list[dict[str, Any]]) -> None:
        """Apply same-store updates atomically through existing metadata CAS."""

        if not updates:
            return
        batch = getattr(self._metadata, "compare_and_swap_named_projections", None)
        if not callable(batch):
            raise TypeError("metadata must support atomic named projection batch CAS")
        if not batch(updates):
            raise ValueError("catalog projection changed concurrently")
        self._refresh()

    def upsert_group(self, group: CatalogGroup) -> CatalogGroup:
        self._assert_write_scope(tenant_id=group.tenant_id, project_id=group.project_id)
        self._refresh()
        key = f"group:{group.group_id}"
        row = self._metadata.get_named_projection(self._projection_namespace, key)
        self._cas(key, {"group": group.model_dump(mode="json")}, row, 1)
        self._groups[group.group_id] = group.model_copy(deep=True)
        return group.model_copy(deep=True)

    def upsert(self, entry: CatalogEntry) -> CatalogEntry:
        prepared, updates = self.prepare_upsert_update(entry)
        self.apply_prepared_updates(updates)
        return prepared

    def remove_provider(
        self,
        provider_id: str,
        *,
        provider_version: str | None = None,
        lifecycle_token: str | None = None,
    ) -> tuple[str, ...]:
        self._refresh()
        removed = tuple(
            logical_id
            for logical_id, entry in self._entries.items()
            if entry.provider_id == str(provider_id)
            and (provider_version is None or entry.provider_version == provider_version)
            and (
                lifecycle_token is None
                or entry.metadata.get("provider_lifecycle_token") == lifecycle_token
            )
        )
        for logical_id in removed:
            pointer_key = self._pointer_key(logical_id)
            row = self._metadata.get_named_projection(self._projection_namespace, pointer_key)
            if row is not None:
                payload = dict(row.get("payload") or {})
                payload.update(
                    {
                        "record_kind": "catalog_current",
                        "logical_id": logical_id,
                        "current_revision": None,
                    }
                )
                expected_a, expected_m = self._cas_values(row)
                if not self._metadata.compare_and_swap_named_projection(
                    self._projection_namespace,
                    pointer_key,
                    payload,
                    expected_last_authoritative_seq=expected_a,
                    expected_last_materialized_seq=expected_m,
                    last_authoritative_seq=expected_a or 1,
                    last_materialized_seq=expected_m or 1,
                    projection_schema_version=1,
                    materialization_status="retired",
                ):
                    raise ValueError(f"catalog projection changed concurrently: {logical_id}")
            else:
                legacy = self._metadata.get_named_projection(self._projection_namespace, logical_id)
                if legacy is not None:
                    payload = dict(legacy.get("payload") or {})
                    payload["entry"] = None
                    revision = int(legacy.get("last_authoritative_seq", 1))
                    self._cas(logical_id, payload, legacy, revision)
            self._entries.pop(logical_id, None)
        return removed

    def get(self, logical_id: str, **kwargs: Any) -> CatalogEntry | None:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return None
        self._refresh()
        return super().get(logical_id, **kwargs)

    def history(self, logical_id: str, **kwargs: Any) -> tuple[CatalogEntry, ...]:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return ()
        self._refresh()
        return super().history(logical_id, **kwargs)

    def browse(self, *args: Any, **kwargs: Any) -> tuple[CatalogEntry, ...]:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return ()
        self._refresh()
        return super().browse(*args, **kwargs)

    def search(self, *args: Any, **kwargs: Any) -> tuple[CatalogSearchResult, ...]:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return ()
        self._refresh()
        return super().search(*args, **kwargs)

    def group_tree(self, *args: Any, **kwargs: Any) -> tuple[CatalogGroup, ...]:
        if not self._matches_store_scope(
            tenant_id=kwargs.get("tenant_id"), project_id=kwargs.get("project_id")
        ):
            return ()
        self._refresh()
        return super().group_tree(*args, **kwargs)
