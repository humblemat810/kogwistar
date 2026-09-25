"""Bounded, read-only agent adapters over existing service seams.

The facade deliberately accepts collaborators instead of engines. Existing
ChatService/runtime API objects can be supplied directly; no backend read path
is introduced here.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

from kogwistar.server.capability_kernel import CapabilityKernel

from .catalog import CatalogEntry, CatalogSearchResult, CatalogStore
from .providers import ProviderRegistry
from .skills import SkillGraphArtifact


class ReadScope(BaseModel):
    """Request identity and namespace limits carried through every read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str
    namespace: str | None = None
    security_scope: str | None = None
    tenant_id: str | None = None
    project_id: str | None = None
    conversation_id: str | None = None
    allow_cross_conversation: bool = False
    approved_wisdom_only: bool = True

    @field_validator("principal_id")
    @classmethod
    def _principal_required(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("principal_id is required")
        return value


@dataclass(frozen=True, slots=True)
class ReadPage:
    items: tuple[dict[str, Any], ...]
    next_cursor: str | None
    limit: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [dict(item) for item in self.items],
            "next_cursor": self.next_cursor,
            "limit": self.limit,
            "truncated": self.truncated,
        }


class VisibilityChecker(Protocol):
    def __call__(self, item: Mapping[str, Any], scope: ReadScope) -> bool: ...


class ReadSource(Protocol):
    """Minimal object protocol; concrete server services already satisfy it."""


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _decode_cursor(cursor: str | None) -> int:
    if cursor is None or cursor == "":
        return 0
    try:
        value = int(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("ascii"))
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError("invalid read cursor") from exc
    if value < 0:
        raise ValueError("invalid read cursor")
    return value


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii")


def _page(
    values: Sequence[Mapping[str, Any]], *, cursor: str | None, limit: int, max_limit: int
) -> ReadPage:
    if limit < 1 or limit > max_limit:
        raise ValueError(f"limit must be between 1 and {max_limit}")
    offset = _decode_cursor(cursor)
    page = values[offset : offset + limit]
    next_offset = offset + len(page)
    next_cursor = _encode_cursor(next_offset) if next_offset < len(values) else None
    return ReadPage(
        items=tuple(_json_safe(dict(item)) for item in page),
        next_cursor=next_cursor,
        limit=limit,
        truncated=next_cursor is not None,
    )


def _effective_limit(limit: int | None, max_limit: int) -> int:
    return min(20, max_limit) if limit is None else limit


def _source_items(source: Any, method: str, **kwargs: Any) -> list[Mapping[str, Any]]:
    if source is None:
        return []
    if isinstance(source, Mapping):
        value = source.get(method, source.get("items", source.get("results", [])))
    elif callable(source) and method == "search":
        value = source(**kwargs)
    else:
        function = getattr(source, method, None)
        if not callable(function):
            raise TypeError(f"read source lacks {method}()")
        value = function(**kwargs)
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("items", value.get("results", [value]))
    return [item for item in value if isinstance(item, Mapping)]


class AgentReadTools:
    """Read-only facade for agent context assembly and progressive disclosure."""

    def __init__(
        self,
        *,
        catalog: CatalogStore,
        providers: ProviderRegistry | None = None,
        skill_artifacts: Mapping[str, SkillGraphArtifact] | None = None,
        mcp_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
        capability_kernel: CapabilityKernel | None = None,
        execution_source: Any | None = None,
        memory_source: Any | None = None,
        knowledge_source: Any | None = None,
        wisdom_source: Any | None = None,
        glossary_source: Any | None = None,
        visibility_checker: VisibilityChecker | None = None,
        acl_required: bool = True,
        max_limit: int = 100,
        max_resource_bytes: int = 64 * 1024,
    ) -> None:
        if max_limit < 1 or max_resource_bytes < 1:
            raise ValueError("read bounds must be positive")
        self.catalog = catalog
        self.providers = providers
        self.skill_artifacts = dict(skill_artifacts or {})
        self.mcp_descriptors = dict(mcp_descriptors or {})
        self.capability_kernel = capability_kernel
        self.execution_source = execution_source
        self.memory_source = memory_source
        self.knowledge_source = knowledge_source
        self.wisdom_source = wisdom_source
        self.glossary_source = glossary_source
        self.visibility_checker = visibility_checker
        self.acl_required = acl_required
        self.max_limit = max_limit
        self.max_resource_bytes = max_resource_bytes

    def _visible(self, item: Mapping[str, Any], scope: ReadScope) -> bool:
        for key, expected in (
            ("tenant_id", scope.tenant_id),
            ("project_id", scope.project_id),
            ("namespace", scope.namespace),
            ("security_scope", scope.security_scope),
        ):
            actual = item.get(key)
            if expected is not None and actual not in (None, expected):
                return False
        if self.visibility_checker is not None:
            return bool(self.visibility_checker(item, scope))
        return not self.acl_required

    def _filter(self, items: list[Mapping[str, Any]], scope: ReadScope) -> list[dict[str, Any]]:
        return [_json_safe(dict(item)) for item in items if self._visible(item, scope)]

    def catalog_search(
        self,
        query: str,
        *,
        scope: ReadScope,
        mode: str = "lexical",
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        results = self.catalog.search(
            query,
            principal=scope.principal_id,
            scope=scope.security_scope,
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            mode=mode,
            limit=self.max_limit,
        )
        items = self._filter(
            [self._catalog_result(result) for result in results], scope
        )
        return _page(items, cursor=cursor, limit=_effective_limit(limit, self.max_limit), max_limit=self.max_limit)

    def catalog_browse(
        self,
        *,
        scope: ReadScope,
        group_id: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        entries = self.catalog.browse(
            group_id,
            principal=scope.principal_id,
            scope=scope.security_scope,
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
        )
        items = self._filter([entry.model_dump(mode="json") for entry in entries], scope)
        return _page(items, cursor=cursor, limit=_effective_limit(limit, self.max_limit), max_limit=self.max_limit)

    def catalog_get(self, logical_id: str, *, scope: ReadScope) -> dict[str, Any] | None:
        entry = self.catalog.get(logical_id)
        if entry is None or not self._catalog_visible(entry, scope):
            return None
        return entry.model_dump(mode="json")

    def _catalog_visible(self, entry: CatalogEntry, scope: ReadScope) -> bool:
        if self.catalog.acl_enabled:
            visible = self.catalog.search(
                entry.logical_id,
                principal=scope.principal_id,
                scope=scope.security_scope,
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
                limit=1,
            )
            return bool(visible and visible[0].entry.logical_id == entry.logical_id)
        return self._visible(entry.model_dump(mode="python"), scope)

    @staticmethod
    def _catalog_result(result: CatalogSearchResult) -> dict[str, Any]:
        item = result.entry.model_dump(mode="json")
        item["match"] = result.match
        item["score"] = result.score
        return item

    def skill_get(
        self,
        logical_id: str,
        *,
        scope: ReadScope,
        representation: str = "descriptor",
        resource: str | None = None,
    ) -> dict[str, Any] | None:
        descriptor = self.catalog_get(logical_id, scope=scope)
        if descriptor is None:
            return None
        if representation == "descriptor":
            return descriptor
        if representation == "graph":
            artifact = self.skill_artifacts.get(logical_id)
            if artifact is None:
                return None
            return {"descriptor": descriptor, "artifact": artifact.model_dump(mode="json")}
        if representation != "raw":
            raise ValueError("representation must be descriptor, graph, or raw")
        if self.providers is None:
            return None
        registration = self.providers.get(
            descriptor["provider_id"], descriptor.get("provider_version", "v1")
        )
        provider = registration.provider
        loader = getattr(provider, "get_resource" if resource else "get_skill", None)
        if not callable(loader):
            return None
        payload = loader(descriptor["provider_local_id"], resource=resource) if resource else loader(descriptor["provider_local_id"])
        payload = _json_safe(payload)
        encoded = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        if len(encoded) > self.max_resource_bytes:
            return {"descriptor": descriptor, "resource": None, "truncated": True}
        return {"descriptor": descriptor, "resource": payload, "truncated": False}

    def skill_resource_get(
        self,
        logical_id: str,
        resource: str,
        *,
        scope: ReadScope,
    ) -> dict[str, Any] | None:
        return self.skill_get(
            logical_id, scope=scope, representation="raw", resource=resource
        )

    def mcp_describe(self, logical_id: str, *, scope: ReadScope) -> dict[str, Any] | None:
        descriptor = self.mcp_descriptors.get(logical_id)
        if descriptor is None or not self._visible(descriptor, scope):
            return None
        return _json_safe(dict(descriptor))

    def capability_describe(self, name: str, *, scope: ReadScope) -> dict[str, Any] | None:
        del scope
        if self.capability_kernel is None:
            return None
        for spec in self.capability_kernel.list_specs():
            if spec.name == name:
                return {
                    "name": spec.name,
                    "description": spec.description,
                    "action_kind": spec.action_kind,
                    "parent": spec.parent,
                }
        return None

    def execution_read(
        self,
        method: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> ReadPage:
        effective_limit = _effective_limit(limit, self.max_limit)
        if self.execution_source is None:
            return ReadPage((), None, effective_limit, False)
        items = self._filter(_source_items(self.execution_source, method, **kwargs), scope)
        return _page(items, cursor=cursor, limit=effective_limit, max_limit=self.max_limit)

    def execution_get(
        self, method: str, *, scope: ReadScope, **kwargs: Any
    ) -> dict[str, Any] | None:
        if self.execution_source is None:
            return None
        items = self._filter(_source_items(self.execution_source, method, **kwargs), scope)
        return items[0] if items else None

    def run_status(self, run_id: str, *, scope: ReadScope) -> dict[str, Any] | None:
        return self.execution_get("get_run", scope=scope, run_id=run_id)

    def run_events(
        self,
        run_id: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        return self.execution_read(
            "list_run_events", scope=scope, cursor=cursor, limit=limit, run_id=run_id
        )

    def run_steps(
        self,
        run_id: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        return self.execution_read(
            "list_steps", scope=scope, cursor=cursor, limit=limit, run_id=run_id
        )

    def run_checkpoints(
        self,
        run_id: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        return self.execution_read(
            "list_checkpoints", scope=scope, cursor=cursor, limit=limit, run_id=run_id
        )

    def run_lineage(
        self,
        run_id: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        return self.execution_read(
            "workflow_run_lineage", scope=scope, cursor=cursor, limit=limit, run_id=run_id
        )

    def run_evidence(
        self,
        run_id: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        return self.execution_read(
            "run_evidence", scope=scope, cursor=cursor, limit=limit, run_id=run_id
        )

    def memory_search(
        self,
        query: str,
        *,
        scope: ReadScope,
        conversation_id: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        target = conversation_id or scope.conversation_id
        if not target:
            raise ValueError("memory search requires conversation_id")
        if target != scope.conversation_id and not scope.allow_cross_conversation:
            raise PermissionError("cross-conversation memory requires explicit opt-in")
        items = self._filter(
            _source_items(self.memory_source, "search", query=query, conversation_id=target),
            scope,
        )
        items = [item for item in items if item.get("conversation_id", target) == target]
        return _page(items, cursor=cursor, limit=_effective_limit(limit, self.max_limit), max_limit=self.max_limit)

    def knowledge_search(
        self,
        query: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        items = self._filter(
            _source_items(
                self.knowledge_source,
                "search",
                query=query,
                namespace=scope.namespace,
                tenant_id=scope.tenant_id,
                project_id=scope.project_id,
            ),
            scope,
        )
        return _page(items, cursor=cursor, limit=_effective_limit(limit, self.max_limit), max_limit=self.max_limit)

    def knowledge_get(
        self, entity_id: str, *, scope: ReadScope
    ) -> dict[str, Any] | None:
        items = self._filter(
            _source_items(self.knowledge_source, "get", entity_id=entity_id), scope
        )
        return items[0] if items else None

    def knowledge_expand(
        self,
        entity_id: str,
        *,
        scope: ReadScope,
        depth: int = 1,
        limit: int | None = None,
    ) -> ReadPage:
        if depth < 0 or depth > 4:
            raise ValueError("knowledge expansion depth must be between 0 and 4")
        items = self._filter(
            _source_items(
                self.knowledge_source,
                "expand",
                entity_id=entity_id,
                depth=depth,
                limit=_effective_limit(limit, self.max_limit),
            ),
            scope,
        )
        return _page(items, cursor=None, limit=_effective_limit(limit, self.max_limit), max_limit=self.max_limit)

    def wisdom_search(
        self,
        query: str,
        *,
        scope: ReadScope,
        status: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        items = self._filter(
            _source_items(self.wisdom_source, "search", query=query, status=status), scope
        )
        if scope.approved_wisdom_only and status not in (None, "approved"):
            raise PermissionError("default wisdom reads allow approved status only")
        if scope.approved_wisdom_only:
            items = [item for item in items if item.get("status") == "approved"]
        return _page(items, cursor=cursor, limit=_effective_limit(limit, self.max_limit), max_limit=self.max_limit)

    def glossary_search(
        self,
        query: str,
        *,
        scope: ReadScope,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ReadPage:
        if self.glossary_source is None:
            return ReadPage((), None, _effective_limit(limit, self.max_limit), False)
        items = _source_items(
            self.glossary_source,
            "search",
            query=query,
            tenant_id=scope.tenant_id,
            project_id=scope.project_id,
            authorize=lambda item: self._visible(item, scope),
            limit=_effective_limit(limit, self.max_limit),
        )
        return _page(
            self._filter(items, scope),
            cursor=cursor,
            limit=_effective_limit(limit, self.max_limit),
            max_limit=self.max_limit,
        )

    def dispatch(
        self,
        operation: str,
        *,
        payload: Mapping[str, Any],
        scope: ReadScope,
    ) -> dict[str, Any] | None:
        """Dispatch only the shared read contract used by REST/MCP adapters."""

        operation = str(operation).strip().lower()
        data = dict(payload)
        if operation == "catalog.search":
            result: Any = self.catalog_search(scope=scope, **data)
        elif operation == "catalog.browse":
            result = self.catalog_browse(scope=scope, **data)
        elif operation == "catalog.get":
            result = self.catalog_get(data.pop("logical_id"), scope=scope)
        elif operation == "skill.get":
            result = self.skill_get(scope=scope, **data)
        elif operation == "skill.resource_get":
            result = self.skill_resource_get(
                data.pop("logical_id"), data.pop("resource"), scope=scope
            )
        elif operation == "mcp.describe":
            result = self.mcp_describe(data.pop("logical_id"), scope=scope)
        elif operation == "capability.describe":
            result = self.capability_describe(data.pop("name"), scope=scope)
        elif operation == "run.status":
            result = self.run_status(data.pop("run_id"), scope=scope)
        elif operation in {
            "run.events",
            "run.steps",
            "run.checkpoints",
            "run.lineage",
            "run.evidence",
        }:
            method = getattr(self, operation.replace(".", "_"))
            result = method(data.pop("run_id"), scope=scope, **data)
        elif operation == "memory.search":
            result = self.memory_search(scope=scope, **data)
        elif operation == "knowledge.search":
            result = self.knowledge_search(scope=scope, **data)
        elif operation == "knowledge.get":
            result = self.knowledge_get(data.pop("entity_id"), scope=scope)
        elif operation == "knowledge.expand":
            result = self.knowledge_expand(data.pop("entity_id"), scope=scope, **data)
        elif operation == "wisdom.search":
            result = self.wisdom_search(scope=scope, **data)
        elif operation == "glossary.search":
            result = self.glossary_search(scope=scope, **data)
        else:
            raise ValueError(f"unsupported read operation: {operation}")
        if isinstance(result, ReadPage):
            return result.to_dict()
        return _json_safe(result) if result is not None else None


__all__ = ["AgentReadTools", "ReadPage", "ReadScope"]
