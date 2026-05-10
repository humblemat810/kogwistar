from __future__ import annotations

"""Durable latest-health registry for long-running operational services."""

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .models import Grounding, Node, Span

if TYPE_CHECKING:
    from .engine import GraphKnowledgeEngine


SERVICE_HEALTH_PROJECTION_NAMESPACE = "service_health"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True, slots=True)
class ServiceDefinition:
    service_id: str
    namespace: str
    service_kind: str
    owner_app: str
    deterministic: bool
    llm_assisted: bool
    version: str | None = None
    config_metadata: dict[str, Any] = field(default_factory=dict)
    operator_tags: tuple[str, ...] = ()
    workspace_id: str | None = None
    heartbeat_ttl_ms: int = 60_000


@dataclass(frozen=True, slots=True)
class ServiceInstanceHealth:
    service_id: str
    instance_id: str
    started_at_ms: int
    last_seen_ms: int
    status: str
    host: str | None = None
    pid: int | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class ServiceHealthRepairResult:
    workspace_id: str | None
    namespace: str | None
    service_id: str | None
    scanned_count: int
    repaired_count: int
    skipped_count: int


class ServiceHealthRegistry:
    """Small registry for durable latest health of operational services.

    This is intentionally not a capability registry, tool registry, scheduler,
    or agent framework. Sparse lifecycle facts are graph nodes; high-frequency
    heartbeat state is a durable named projection.
    """

    def __init__(self, engine: "GraphKnowledgeEngine") -> None:
        self.engine = engine

    def declare_service(
        self,
        *,
        service_id: str,
        service_kind: str,
        owner_app: str,
        deterministic: bool,
        llm_assisted: bool,
        namespace: str | None = None,
        workspace_id: str | None = None,
        version: str | None = None,
        config_metadata: dict[str, Any] | None = None,
        operator_tags: list[str] | tuple[str, ...] | None = None,
        heartbeat_ttl_ms: int = 60_000,
    ) -> dict[str, Any]:
        now = _now_ms()
        ns = str(namespace or getattr(self.engine, "namespace", "default") or "default")
        definition = ServiceDefinition(
            service_id=str(service_id),
            namespace=ns,
            service_kind=str(service_kind),
            owner_app=str(owner_app),
            deterministic=bool(deterministic),
            llm_assisted=bool(llm_assisted),
            version=None if version is None else str(version),
            config_metadata=dict(config_metadata or {}),
            operator_tags=tuple(str(item) for item in (operator_tags or ()) if str(item)),
            workspace_id=None if workspace_id is None else str(workspace_id),
            heartbeat_ttl_ms=max(1, int(heartbeat_ttl_ms or 1)),
        )
        existing = self.get_service(
            definition.service_id,
            workspace_id=definition.workspace_id,
            namespace=definition.namespace,
        )
        payload = self._merge_definition(existing, definition, now_ms=now)

        if existing is None:
            self._append_lifecycle_event(
                service_id=definition.service_id,
                event_type="service.registered",
                payload={"definition": self._definition_payload(definition)},
            )
        elif self._definition_changed(existing, payload):
            self._append_lifecycle_event(
                service_id=definition.service_id,
                event_type="service.config_changed",
                payload={"definition": self._definition_payload(definition)},
            )

        self._store(
            definition.service_id,
            payload,
            workspace_id=definition.workspace_id,
            namespace=definition.namespace,
        )
        return payload

    def start_instance(
        self,
        *,
        service_id: str,
        workspace_id: str | None = None,
        namespace: str | None = None,
        instance_id: str | None = None,
        status: str = "starting",
        host: str | None = None,
        pid: int | None = None,
        started_at_ms: int | None = None,
    ) -> dict[str, Any]:
        payload = self._require_payload(
            service_id,
            workspace_id=workspace_id,
            namespace=namespace,
        )
        now = _now_ms()
        chosen_instance = str(instance_id or uuid.uuid4())
        started = int(started_at_ms or now)
        previous_instance = str(payload.get("instance_id") or "")
        payload.update(
            {
                "instance_id": chosen_instance,
                "started_at_ms": started,
                "last_seen_ms": now,
                "status": str(status or "starting"),
                "host": host or _host_name(),
                "pid": int(pid if pid is not None else os.getpid()),
                "updated_at_ms": now,
            }
        )
        if previous_instance != chosen_instance:
            self._append_lifecycle_event(
                service_id=str(service_id),
                event_type="service.instance_started",
                payload={
                    "instance_id": chosen_instance,
                    "started_at_ms": started,
                    "status": payload["status"],
                },
            )
        self._store(
            service_id,
            payload,
            workspace_id=self._coalesce_scope(workspace_id, payload.get("workspace_id")),
            namespace=self._coalesce_scope(namespace, payload.get("namespace")),
        )
        return payload

    def heartbeat(
        self,
        *,
        service_id: str,
        workspace_id: str | None = None,
        namespace: str | None = None,
        instance_id: str,
        status: str = "healthy",
        last_error: str | None = None,
        host: str | None = None,
        pid: int | None = None,
    ) -> dict[str, Any]:
        payload = self._require_payload(
            service_id,
            workspace_id=workspace_id,
            namespace=namespace,
        )
        now = _now_ms()
        previous_status = str(payload.get("status") or "")
        previous_error = payload.get("last_error")
        status_text = str(status or "healthy")
        payload.update(
            {
                "instance_id": str(instance_id),
                "last_seen_ms": now,
                "status": status_text,
                "last_error": None if last_error is None else str(last_error),
                "host": host or payload.get("host") or _host_name(),
                "pid": int(pid if pid is not None else payload.get("pid") or os.getpid()),
                "updated_at_ms": now,
            }
        )
        if previous_status in {"stale", "degraded", "failed"} and status_text == "healthy":
            self._append_lifecycle_event(
                service_id=str(service_id),
                event_type="service.recovered",
                payload={"instance_id": str(instance_id), "status": status_text},
            )
        elif previous_status and previous_status != status_text and status_text in {
            "stale",
            "degraded",
            "failed",
            "stopped",
        }:
            self._append_lifecycle_event(
                service_id=str(service_id),
                event_type=f"service.{status_text}",
                payload={"instance_id": str(instance_id), "status": status_text},
            )
        if previous_error != payload["last_error"] and payload["last_error"]:
            self._append_lifecycle_event(
                service_id=str(service_id),
                event_type="service.error_changed",
                payload={
                    "instance_id": str(instance_id),
                    "last_error": payload["last_error"],
                },
            )
        self._store(
            service_id,
            payload,
            workspace_id=self._coalesce_scope(workspace_id, payload.get("workspace_id")),
            namespace=self._coalesce_scope(namespace, payload.get("namespace")),
        )
        return payload

    def stop_service(
        self,
        *,
        service_id: str,
        workspace_id: str | None = None,
        namespace: str | None = None,
        instance_id: str | None = None,
        status: str = "stopped",
        last_error: str | None = None,
    ) -> dict[str, Any]:
        payload = self._require_payload(
            service_id,
            workspace_id=workspace_id,
            namespace=namespace,
        )
        now = _now_ms()
        payload.update(
            {
                "instance_id": str(instance_id or payload.get("instance_id") or ""),
                "last_seen_ms": now,
                "status": str(status or "stopped"),
                "last_error": None if last_error is None else str(last_error),
                "updated_at_ms": now,
            }
        )
        self._append_lifecycle_event(
            service_id=str(service_id),
            event_type="service.stopped",
            payload={
                "instance_id": payload["instance_id"],
                "status": payload["status"],
                "last_error": payload["last_error"],
            },
        )
        self._store(
            service_id,
            payload,
            workspace_id=self._coalesce_scope(workspace_id, payload.get("workspace_id")),
            namespace=self._coalesce_scope(namespace, payload.get("namespace")),
        )
        return payload

    def get_service(
        self,
        service_id: str,
        *,
        workspace_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any] | None:
        get_projection = getattr(self.engine.meta_sqlite, "get_named_projection", None)
        if not callable(get_projection):
            return None
        key = self._resolve_projection_key(
            service_id,
            workspace_id=workspace_id,
            namespace=namespace,
        )
        if key is None:
            return None
        row = get_projection(SERVICE_HEALTH_PROJECTION_NAMESPACE, key)
        payload = row.get("payload") if isinstance(row, dict) else None
        return dict(payload) if isinstance(payload, dict) else None

    def list_services(
        self,
        *,
        workspace_id: str | None = None,
        namespace: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        list_projection = getattr(self.engine.meta_sqlite, "list_named_projections", None)
        if not callable(list_projection):
            return []
        rows = list_projection(SERVICE_HEALTH_PROJECTION_NAMESPACE)
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload") if isinstance(row, dict) else None
            if not isinstance(payload, dict):
                continue
            if workspace_id is not None and str(payload.get("workspace_id") or "") != str(workspace_id):
                continue
            if namespace is not None and str(payload.get("namespace") or "") != str(namespace):
                continue
            out.append(dict(payload))
        out.sort(key=lambda item: str(item.get("service_id") or ""))
        return out[: max(1, int(limit))]

    def stale_services(
        self,
        *,
        workspace_id: str | None = None,
        now_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        now = int(now_ms or _now_ms())
        stale: list[dict[str, Any]] = []
        for payload in self.list_services(workspace_id=workspace_id, limit=10_000):
            last_seen = _optional_int(payload.get("last_seen_ms"))
            ttl = int(payload.get("heartbeat_ttl_ms", 60_000) or 60_000)
            if last_seen is not None and now - last_seen > ttl:
                stale.append(payload)
        return stale

    def repair_projection(
        self,
        *,
        workspace_id: str | None = None,
        namespace: str | None = None,
        service_id: str | None = None,
    ) -> ServiceHealthRepairResult:
        events = self._service_event_nodes()
        scanned = 0
        repaired = 0
        skipped = 0
        rebuilt: dict[str, dict[str, Any]] = {}
        for node in events:
            payload = self._event_payload(node)
            if payload is None:
                skipped += 1
                continue
            scanned += 1
            event_service_id = str(node.metadata.get("service_id") or "")
            if service_id is not None and event_service_id != str(service_id):
                continue
            event_type = str(node.metadata.get("service_event_type") or "")
            service_payload = rebuilt.setdefault(event_service_id, {})
            self._apply_event_payload(service_payload, event_type=event_type, payload=payload)
        for repaired_service_id, payload in rebuilt.items():
            event_workspace = self._coalesce_scope(workspace_id, payload.get("workspace_id"))
            event_namespace = self._coalesce_scope(namespace, payload.get("namespace"))
            if workspace_id is not None and event_workspace != str(workspace_id):
                continue
            if namespace is not None and event_namespace != str(namespace):
                continue
            if not payload or payload.get("service_id") is None:
                skipped += 1
                continue
            try:
                key = self._projection_key(
                    repaired_service_id,
                    workspace_id=event_workspace,
                    namespace=event_namespace,
                )
            except KeyError:
                skipped += 1
                continue
            existing = self._get_projection_payload_by_key(key)
            if existing is None:
                repaired += 1
            self._store(
                repaired_service_id,
                payload,
                workspace_id=event_workspace,
                namespace=event_namespace,
            )
        return ServiceHealthRepairResult(
            workspace_id=None if workspace_id is None else str(workspace_id),
            namespace=None if namespace is None else str(namespace),
            service_id=None if service_id is None else str(service_id),
            scanned_count=scanned,
            repaired_count=repaired,
            skipped_count=skipped,
        )

    def _merge_definition(
        self,
        existing: dict[str, Any] | None,
        definition: ServiceDefinition,
        *,
        now_ms: int,
    ) -> dict[str, Any]:
        payload = dict(existing or {})
        payload.update(self._definition_payload(definition))
        payload.setdefault("instance_id", None)
        payload.setdefault("started_at_ms", None)
        payload.setdefault("last_seen_ms", None)
        payload.setdefault("status", "registered")
        payload.setdefault("host", None)
        payload.setdefault("pid", None)
        payload.setdefault("last_error", None)
        payload.setdefault("created_at_ms", now_ms)
        payload["updated_at_ms"] = now_ms
        return payload

    @staticmethod
    def _definition_payload(definition: ServiceDefinition) -> dict[str, Any]:
        return {
            "service_id": definition.service_id,
            "workspace_id": definition.workspace_id,
            "namespace": definition.namespace,
            "service_kind": definition.service_kind,
            "owner_app": definition.owner_app,
            "deterministic": definition.deterministic,
            "llm_assisted": definition.llm_assisted,
            "version": definition.version,
            "config_metadata": dict(definition.config_metadata),
            "operator_tags": list(definition.operator_tags),
            "heartbeat_ttl_ms": definition.heartbeat_ttl_ms,
        }

    @staticmethod
    def _definition_changed(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
        keys = {
            "workspace_id",
            "namespace",
            "service_kind",
            "owner_app",
            "deterministic",
            "llm_assisted",
            "version",
            "config_metadata",
            "operator_tags",
            "heartbeat_ttl_ms",
        }
        return any(existing.get(key) != candidate.get(key) for key in keys)

    def _store(
        self,
        service_id: str,
        payload: dict[str, Any],
        *,
        workspace_id: str | None = None,
        namespace: str | None = None,
    ) -> None:
        latest_seq = 0
        getter = getattr(self.engine.meta_sqlite, "get_latest_entity_event_seq", None)
        if callable(getter):
            latest_seq = int(getter(namespace=getattr(self.engine, "namespace", "default")) or 0)
        projection_key = self._projection_key(
            service_id,
            workspace_id=self._coalesce_scope(workspace_id, payload.get("workspace_id")),
            namespace=self._coalesce_scope(namespace, payload.get("namespace")),
        )
        self.engine.meta_sqlite.replace_named_projection(
            SERVICE_HEALTH_PROJECTION_NAMESPACE,
            projection_key,
            payload,
            last_authoritative_seq=latest_seq,
            last_materialized_seq=latest_seq,
            projection_schema_version=1,
            materialization_status="ready",
        )

    def _append_lifecycle_event(
        self,
        *,
        service_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        now = _now_ms()
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        node = Node(
            id=f"service_health_evt:{service_id}:{now}:{uuid.uuid4().hex}",
            label=f"service_health_event:{event_type}",
            type="entity",
            summary=f"Service health lifecycle event {event_type} for {service_id}",
            mentions=[Grounding(spans=[Span.from_dummy_for_workflow(str(service_id))])],
            properties={"payload_json": text},
            metadata={
                "entity_type": "service_health_event",
                "artifact_kind": "service_health_event",
                "service_id": str(service_id),
                "service_event_type": str(event_type),
                "ts_ms": now,
                "in_conversation_chain": False,
            },
        )
        self.engine.write.add_node(node)

    def _require_payload(
        self,
        service_id: str,
        *,
        workspace_id: str | None = None,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        payload = self.get_service(
            service_id,
            workspace_id=workspace_id,
            namespace=namespace,
        )
        if payload is None:
            raise KeyError(f"Unknown service_id: {service_id}")
        return payload

    def _resolve_projection_key(
        self,
        service_id: str,
        *,
        workspace_id: str | None = None,
        namespace: str | None = None,
    ) -> str | None:
        if workspace_id is not None or namespace is not None:
            return self._projection_key(service_id, workspace_id=workspace_id, namespace=namespace)
        get_projection = getattr(self.engine.meta_sqlite, "get_named_projection", None)
        if callable(get_projection):
            direct = get_projection(SERVICE_HEALTH_PROJECTION_NAMESPACE, str(service_id))
            if isinstance(direct, dict) and isinstance(direct.get("payload"), dict):
                return str(service_id)
        matches = [
            row.get("key")
            for row in self._projection_rows()
            if isinstance(row.get("payload"), dict)
            and str(row["payload"].get("service_id") or "") == str(service_id)
        ]
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            raise KeyError(
                f"Ambiguous service_id without workspace_id/namespace: {service_id}"
            )
        return None

    def _projection_key(
        self,
        service_id: str,
        *,
        workspace_id: str | None,
        namespace: str | None,
    ) -> str:
        scoped_workspace = self._coalesce_scope(workspace_id, None)
        scoped_namespace = self._coalesce_scope(namespace, None)
        if not scoped_workspace and not scoped_namespace:
            return str(service_id)
        if not scoped_namespace:
            raise KeyError(f"namespace required for scoped service health: {service_id}")
        return f"{scoped_workspace}|{scoped_namespace}|{service_id}"

    def _projection_rows(self) -> list[dict[str, Any]]:
        list_projection = getattr(self.engine.meta_sqlite, "list_named_projections", None)
        if not callable(list_projection):
            return []
        return list_projection(SERVICE_HEALTH_PROJECTION_NAMESPACE)

    def _get_projection_payload_by_key(self, key: str) -> dict[str, Any] | None:
        get_projection = getattr(self.engine.meta_sqlite, "get_named_projection", None)
        if not callable(get_projection):
            return None
        row = get_projection(SERVICE_HEALTH_PROJECTION_NAMESPACE, key)
        payload = row.get("payload") if isinstance(row, dict) else None
        return dict(payload) if isinstance(payload, dict) else None

    def _service_event_nodes(self) -> list[Node]:
        nodes = list(
            self.engine.read.get_nodes(
                where={"entity_type": "service_health_event"},
                limit=100_000,
            )
        )
        nodes.sort(
            key=lambda node: (
                int(node.metadata.get("ts_ms") or 0),
                str(node.id),
            )
        )
        return nodes

    @staticmethod
    def _event_payload(node: Node) -> dict[str, Any] | None:
        raw = node.properties.get("payload_json")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _apply_event_payload(
        service_payload: dict[str, Any],
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type in {"service.registered", "service.config_changed"}:
            definition = payload.get("definition")
            if isinstance(definition, dict):
                service_payload.update(definition)
                service_payload.setdefault("instance_id", None)
                service_payload.setdefault("started_at_ms", None)
                service_payload.setdefault("last_seen_ms", None)
                service_payload.setdefault("status", "registered")
                service_payload.setdefault("host", None)
                service_payload.setdefault("pid", None)
                service_payload.setdefault("last_error", None)
            return
        if event_type == "service.instance_started":
            service_payload["instance_id"] = payload.get("instance_id")
            service_payload["started_at_ms"] = payload.get("started_at_ms")
            service_payload["status"] = payload.get("status") or "starting"
            service_payload["last_seen_ms"] = payload.get("started_at_ms")
            return
        if event_type == "service.error_changed":
            service_payload["instance_id"] = payload.get("instance_id", service_payload.get("instance_id"))
            service_payload["last_error"] = payload.get("last_error")
            return
        if event_type == "service.recovered":
            service_payload["instance_id"] = payload.get("instance_id", service_payload.get("instance_id"))
            service_payload["status"] = payload.get("status") or "healthy"
            service_payload["last_error"] = None
            return
        if event_type == "service.stopped":
            service_payload["instance_id"] = payload.get("instance_id", service_payload.get("instance_id"))
            service_payload["status"] = payload.get("status") or "stopped"
            service_payload["last_error"] = payload.get("last_error")
            return
        if event_type in {"service.stale", "service.failed", "service.degraded"}:
            service_payload["instance_id"] = payload.get("instance_id", service_payload.get("instance_id"))
            service_payload["status"] = payload.get("status") or event_type.removeprefix("service.")
            return

    @staticmethod
    def _coalesce_scope(primary: Any, fallback: Any) -> str | None:
        value = primary if primary is not None else fallback
        if value is None:
            return None
        text = str(value)
        return text if text else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _host_name() -> str | None:
    try:
        return socket.gethostname()
    except Exception:
        return None


__all__ = [
    "SERVICE_HEALTH_PROJECTION_NAMESPACE",
    "ServiceDefinition",
    "ServiceHealthRepairResult",
    "ServiceHealthRegistry",
    "ServiceInstanceHealth",
]
