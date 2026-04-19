from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from kogwistar.engine_core.models import Grounding, Node, Span


SERVICE_PROJECTION_NAMESPACE = "service_registry"
SERVICE_TRIGGER_TYPES = {
    "schedule",
    "message arrival",
    "graph change",
    "external event",
    "autostart",
    "restart",
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _json_text(value)


def _now_ms() -> int:
    return int(time.time() * 1000)


class ServiceTriggerSpec(BaseModel):
    type: str = Field(min_length=1)
    enabled: bool = True
    selector: dict[str, Any] = Field(default_factory=dict)
    debounce_ms: int = 0
    cooldown_ms: int = 0


class ServiceDefinition(BaseModel):
    service_id: str = Field(min_length=1)
    service_kind: str = "service"
    target_kind: str = "workflow"
    target_ref: str = Field(min_length=1)
    target_config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    autostart: bool = False
    restart_policy: dict[str, Any] = Field(default_factory=dict)
    heartbeat_ttl_ms: int = 60_000
    trigger_specs: list[ServiceTriggerSpec] = Field(default_factory=list)
    security_scope: str = "workflow"
    storage_namespace: str = "workflow"
    execution_namespace: str = "workflow"
    created_at_ms: int = 0
    updated_at_ms: int = 0


class ServiceHealthSnapshot(BaseModel):
    service_id: str
    lifecycle_status: str
    health_status: str
    enabled: bool
    last_heartbeat_ms: int | None = None
    restart_count: int = 0
    current_child_run_id: str | None = None
    current_child_status: str | None = None
    last_trigger_type: str | None = None
    last_triggered_at_ms: int | None = None


class ServiceProjectionRow(BaseModel):
    service_id: str
    service_kind: str
    target_kind: str
    target_ref: str
    enabled: bool
    autostart: bool
    storage_namespace: str
    execution_namespace: str
    security_scope: str
    lifecycle_status: str
    health_status: str
    last_heartbeat_ms: int | None = None
    restart_count: int = 0
    current_child_run_id: str | None = None
    current_child_status: str | None = None
    current_child_started_at_ms: int | None = None
    last_handled_child_run_id: str | None = None
    last_trigger_type: str | None = None
    last_triggered_at_ms: int | None = None
    heartbeat_ttl_ms: int = 60_000
    restart_policy: dict[str, Any] = Field(default_factory=dict)
    trigger_specs: list[dict[str, Any]] = Field(default_factory=list)
    last_message_seen_ms: int = 0
    last_graph_event_seq: int = 0
    next_due_at_ms: int | None = None
    restart_not_before_ms: int | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0


@dataclass
class ServiceSupervisor:
    get_workflow_engine: Callable[[], Any]
    get_conversation_engine: Callable[[], Any]
    run_registry: Any
    spawn_workflow_run: Callable[..., dict[str, Any]]
    scope_snapshot: Callable[[], dict[str, str]]

    def bootstrap(self) -> None:
        self._ensure_all_projections()
        for row in self.list_services(limit=10_000):
            service_id = str(row.get("service_id") or "")
            if not service_id or not bool(row.get("enabled")) or not bool(row.get("autostart")):
                continue
            if str(row.get("last_trigger_type") or "") == "autostart":
                continue
            current_child_run_id = str(row.get("current_child_run_id") or "")
            if current_child_run_id:
                run = self.run_registry.get_run(current_child_run_id)
                if run is not None and not bool(run.get("terminal")):
                    continue
            self._trigger_service(
                service_id,
                trigger_type="autostart",
                payload={},
                force=False,
            )
        self.tick()

    def declare_service(
        self,
        *,
        service_id: str,
        service_kind: str,
        target_kind: str,
        target_ref: str,
        target_config: dict[str, Any] | None = None,
        enabled: bool = True,
        autostart: bool = False,
        restart_policy: dict[str, Any] | None = None,
        heartbeat_ttl_ms: int = 60_000,
        trigger_specs: list[dict[str, Any]] | list[ServiceTriggerSpec] | None = None,
    ) -> dict[str, Any]:
        now = _now_ms()
        scope = self.scope_snapshot()
        definition = ServiceDefinition(
            service_id=service_id,
            service_kind=service_kind,
            target_kind=target_kind,
            target_ref=target_ref,
            target_config=dict(target_config or {}),
            enabled=bool(enabled),
            autostart=bool(autostart),
            restart_policy=dict(restart_policy or {}),
            heartbeat_ttl_ms=max(1, int(heartbeat_ttl_ms or 1)),
            trigger_specs=[
                spec if isinstance(spec, ServiceTriggerSpec) else ServiceTriggerSpec.model_validate(spec)
                for spec in (trigger_specs or [])
            ],
            security_scope=str(scope["security_scope"] or "workflow"),
            storage_namespace=str(scope["storage_namespace"] or "workflow"),
            execution_namespace=str(scope["execution_namespace"] or "workflow"),
            created_at_ms=now,
            updated_at_ms=now,
        )
        self._append_definition(definition)
        self._append_event(
            service_id=service_id,
            event_type="service.enabled" if enabled else "service.stopped",
            payload={"enabled": enabled, "autostart": autostart},
        )
        self._rebuild_projection(service_id)
        self.tick()
        return self.get_service(service_id)

    def get_service(self, service_id: str) -> dict[str, Any]:
        projection = self._projection(service_id)
        if projection is None:
            self._rebuild_projection(service_id)
            projection = self._projection(service_id)
        if projection is None:
            raise KeyError(f"Unknown service_id: {service_id}")
        return projection["payload"]

    def list_services(self, *, limit: int = 200) -> list[dict[str, Any]]:
        self._ensure_all_projections()
        meta = self._workflow_engine().meta_sqlite
        rows = list(meta.list_named_projections(SERVICE_PROJECTION_NAMESPACE))
        items = [row["payload"] for row in rows][: int(limit)]
        items.sort(key=lambda item: str(item.get("service_id") or ""))
        return items

    def enable_service(self, service_id: str) -> dict[str, Any]:
        return self._set_enabled(service_id, True)

    def disable_service(self, service_id: str) -> dict[str, Any]:
        return self._set_enabled(service_id, False)

    def record_service_heartbeat(
        self,
        service_id: str,
        *,
        instance_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._append_event(
            service_id=service_id,
            event_type="service.heartbeat",
            payload={"instance_id": instance_id, **dict(payload or {})},
        )
        self._rebuild_projection(service_id)
        return self.get_service(service_id)

    def trigger_service(
        self,
        service_id: str,
        *,
        trigger_type: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._trigger_service(
            service_id=service_id,
            trigger_type=trigger_type,
            payload=dict(payload or {}),
            force=True,
        )

    def health_snapshot(self, service_id: str) -> dict[str, Any]:
        row = ServiceProjectionRow.model_validate(self.get_service(service_id))
        return ServiceHealthSnapshot(
            service_id=row.service_id,
            lifecycle_status=row.lifecycle_status,
            health_status=row.health_status,
            enabled=row.enabled,
            last_heartbeat_ms=row.last_heartbeat_ms,
            restart_count=row.restart_count,
            current_child_run_id=row.current_child_run_id,
            current_child_status=row.current_child_status,
            last_trigger_type=row.last_trigger_type,
            last_triggered_at_ms=row.last_triggered_at_ms,
        ).model_dump(mode="python")

    def list_service_events(
        self, service_id: str, *, limit: int = 500
    ) -> list[dict[str, Any]]:
        nodes = self._workflow_engine().read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "service_event"},
                    {"service_id": str(service_id)},
                ]
            },
            limit=max(1, int(limit)),
        )
        out: list[dict[str, Any]] = []
        for node in nodes:
            md = dict(getattr(node, "metadata", {}) or {})
            properties = dict(getattr(node, "properties", {}) or {})
            out.append(
                {
                    "event_id": str(getattr(node, "id", "") or ""),
                    "service_id": str(md.get("service_id") or ""),
                    "event_type": str(md.get("service_event_type") or ""),
                    "ts_ms": int(md.get("ts_ms", 0) or 0),
                    "payload": json.loads(str(properties.get("payload_json") or "{}")),
                }
            )
        out.sort(key=lambda item: int(item["ts_ms"]))
        return out

    def tick(self) -> None:
        self._ensure_all_projections()
        meta = self._workflow_engine().meta_sqlite
        rows = list(meta.list_named_projections(SERVICE_PROJECTION_NAMESPACE))
        now = _now_ms()
        for row in rows:
            payload = dict(row.get("payload") or {})
            service_id = str(payload.get("service_id") or "")
            if not service_id:
                continue
            self._evaluate_child_state(service_id, payload, now_ms=now)
            self._evaluate_health(service_id, payload, now_ms=now)
            self._evaluate_schedule_trigger(service_id, payload, now_ms=now)
            self._evaluate_message_trigger(service_id, payload, now_ms=now)
            self._evaluate_graph_trigger(service_id, payload, now_ms=now)
            self._rebuild_projection(service_id)

    def _set_enabled(self, service_id: str, enabled: bool) -> dict[str, Any]:
        latest = self._latest_definition(service_id)
        if latest is None:
            raise KeyError(f"Unknown service_id: {service_id}")
        updated = latest.model_copy(
            update={
                "enabled": bool(enabled),
                "updated_at_ms": _now_ms(),
            }
        )
        self._append_definition(updated)
        self._append_event(
            service_id=service_id,
            event_type="service.enabled" if enabled else "service.stopped",
            payload={"enabled": enabled},
        )
        self._rebuild_projection(service_id)
        return self.get_service(service_id)

    def _trigger_service(
        self,
        service_id: str,
        *,
        trigger_type: str,
        payload: dict[str, Any],
        force: bool = False,
    ) -> dict[str, Any]:
        definition = self._latest_definition(service_id)
        if definition is None:
            raise KeyError(f"Unknown service_id: {service_id}")
        normalized_trigger_type = str(trigger_type or "").strip().lower()
        if normalized_trigger_type not in SERVICE_TRIGGER_TYPES:
            raise ValueError(f"Unsupported trigger_type: {trigger_type}")
        projection_payload = self._projection_payload(service_id)
        if projection_payload is None:
            self._rebuild_projection(service_id)
            projection_payload = self._projection_payload(service_id)
        if projection_payload is None:
            raise KeyError(f"Unknown service_id: {service_id}")
        projection = ServiceProjectionRow.model_validate(projection_payload)
        if not definition.enabled and not force:
            return projection.model_dump(mode="python")
        spec = self._matching_trigger_spec(
            definition=definition,
            trigger_type=normalized_trigger_type,
        )
        now_ms = _now_ms()
        if spec is not None and not force:
            if not bool(spec.get("enabled", True)):
                return projection.model_dump(mode="python")
            cooldown_ms = int(spec.get("cooldown_ms", 0) or 0)
            debounce_ms = int(spec.get("debounce_ms", 0) or 0)
            not_before = max(cooldown_ms, debounce_ms)
            last_triggered_at_ms = int(projection.last_triggered_at_ms or 0)
            if not_before > 0 and last_triggered_at_ms > 0:
                if now_ms - last_triggered_at_ms < not_before:
                    return projection.model_dump(mode="python")
        if projection.current_child_run_id:
            run = self.run_registry.get_run(projection.current_child_run_id)
            if run is not None and not bool(run.get("terminal")):
                return projection.model_dump(mode="python")
        self._append_event(
            service_id=service_id,
            event_type="service.starting",
            payload={"trigger_type": normalized_trigger_type},
        )
        self._append_event(
            service_id=service_id,
            event_type="service.triggered",
            payload={"trigger_type": normalized_trigger_type, **payload},
        )
        if definition.target_kind == "workflow":
            target_cfg = dict(definition.target_config or {})
            conversation_id = str(
                payload.get("conversation_id")
                or target_cfg.get("conversation_id")
                or f"svc:{service_id}"
            )
            turn_node_id = payload.get("turn_node_id") or target_cfg.get("turn_node_id")
            user_id = payload.get("user_id") or target_cfg.get("user_id")
            initial_state = dict(target_cfg.get("initial_state") or {})
            initial_state.update(dict(payload.get("initial_state") or {}))
            run_payload = self.spawn_workflow_run(
                workflow_id=definition.target_ref,
                conversation_id=conversation_id,
                turn_node_id=turn_node_id,
                user_id=user_id,
                initial_state=initial_state,
                priority_class=str(target_cfg.get("priority_class") or "background"),
                token_budget=target_cfg.get("token_budget"),
                time_budget_ms=target_cfg.get("time_budget_ms"),
            )
            self._append_event(
                service_id=service_id,
                event_type="service.run_spawned",
                payload={
                    "trigger_type": normalized_trigger_type,
                    "run_id": run_payload.get("run_id"),
                    "workflow_id": definition.target_ref,
                },
            )
        self._rebuild_projection(service_id)
        return self.get_service(service_id)

    def _evaluate_child_state(
        self, service_id: str, payload: dict[str, Any], *, now_ms: int
    ) -> None:
        current_child_run_id = str(payload.get("current_child_run_id") or "")
        if not current_child_run_id:
            return
        run = self.run_registry.get_run(current_child_run_id)
        if run is None:
            return
        status = str(run.get("status") or "")
        last_handled = str(payload.get("last_handled_child_run_id") or "")
        if not bool(run.get("terminal")):
            return
        if current_child_run_id == last_handled and payload.get("restart_not_before_ms") is None:
            return
        self._append_event(
            service_id=service_id,
            event_type="service.run_failed" if status == "failed" else "service.stopped",
            payload={"run_id": current_child_run_id, "status": status},
        )
        payload["last_handled_child_run_id"] = current_child_run_id
        definition = self._latest_definition(service_id)
        if definition is None:
            return
        mode = str((definition.restart_policy or {}).get("mode") or "never").strip().lower()
        max_restarts = int((definition.restart_policy or {}).get("max_restarts", 0) or 0)
        backoff_ms = int((definition.restart_policy or {}).get("restart_backoff_ms", 0) or 0)
        restart_count = int(payload.get("restart_count", 0) or 0)
        should_restart = False
        if mode == "always":
            should_restart = True
        elif mode == "on_failure" and status == "failed":
            should_restart = True
        if should_restart:
            restart_not_before_ms = payload.get("restart_not_before_ms")
            if restart_not_before_ms is None:
                if restart_count >= max_restarts:
                    self._store_projection_payload(service_id, payload)
                    return
                payload["restart_count"] = restart_count + 1
                payload["restart_not_before_ms"] = now_ms + max(0, backoff_ms)
                payload["lifecycle_status"] = "restarting"
                payload["updated_at_ms"] = now_ms
                self._append_event(
                    service_id=service_id,
                    event_type="service.restarting",
                    payload={
                        "previous_run_id": current_child_run_id,
                        "status": status,
                        "restart_count": payload["restart_count"],
                        "restart_not_before_ms": payload["restart_not_before_ms"],
                    },
                )
                self._store_projection_payload(service_id, payload)
                return
            if now_ms < int(restart_not_before_ms):
                self._store_projection_payload(service_id, payload)
                return
            payload["restart_not_before_ms"] = None
            payload["updated_at_ms"] = now_ms
            self._store_projection_payload(service_id, payload)
            self._trigger_service(
                service_id,
                trigger_type="restart",
                payload={"initial_state": {}},
                force=True,
            )
            return
        payload["updated_at_ms"] = now_ms
        self._store_projection_payload(service_id, payload)

    def _evaluate_health(
        self, service_id: str, payload: dict[str, Any], *, now_ms: int
    ) -> None:
        enabled = bool(payload.get("enabled"))
        last_heartbeat_ms = payload.get("last_heartbeat_ms")
        ttl = int(payload.get("heartbeat_ttl_ms", 60_000) or 60_000)
        current_child_run_id = str(payload.get("current_child_run_id") or "")
        current_child_status = str(payload.get("current_child_status") or "")
        lifecycle_status = "stopped"
        health_status = "stopped"
        if enabled:
            if payload.get("restart_not_before_ms") is not None or str(
                payload.get("lifecycle_status") or ""
            ) == "restarting":
                lifecycle_status = "restarting"
                health_status = "degraded"
            elif current_child_run_id and current_child_status in {"queued", "running"}:
                lifecycle_status = "starting" if current_child_status == "queued" else "healthy"
                health_status = "healthy"
            elif last_heartbeat_ms is not None and now_ms - int(last_heartbeat_ms) <= ttl:
                lifecycle_status = "healthy"
                health_status = "healthy"
            else:
                lifecycle_status = "degraded"
                health_status = "degraded"
        if payload.get("lifecycle_status") != lifecycle_status:
            event_type = f"service.{lifecycle_status}"
            self._append_event(
                service_id=service_id,
                event_type=event_type,
                payload={"health_status": health_status},
            )
        payload["lifecycle_status"] = lifecycle_status
        payload["health_status"] = health_status

    def _evaluate_schedule_trigger(
        self, service_id: str, payload: dict[str, Any], *, now_ms: int
    ) -> None:
        if not bool(payload.get("enabled")):
            return
        if payload.get("current_child_run_id") and payload.get("current_child_status") in {
            "queued",
            "running",
        }:
            return
        for spec in payload.get("trigger_specs", []):
            if str(spec.get("type") or "") != "schedule" or not bool(spec.get("enabled", True)):
                continue
            selector = dict(spec.get("selector") or {})
            cooldown_ms = int(spec.get("cooldown_ms", 0) or 0)
            interval_ms = int(selector.get("interval_ms", 0) or 0)
            next_due = payload.get("next_due_at_ms")
            if next_due is None:
                next_due = now_ms if bool(payload.get("autostart")) else now_ms + interval_ms
            if now_ms < int(next_due):
                continue
            last_triggered_at_ms = int(payload.get("last_triggered_at_ms", 0) or 0)
            if cooldown_ms > 0 and now_ms - last_triggered_at_ms < cooldown_ms:
                continue
            self._trigger_service(
                service_id,
                trigger_type="schedule",
                payload={"initial_state": selector.get("initial_state") or {}},
                force=False,
            )
            payload["next_due_at_ms"] = now_ms + max(interval_ms, cooldown_ms, 1)
            self._store_projection_payload(service_id, payload)
            return

    def _evaluate_message_trigger(
        self, service_id: str, payload: dict[str, Any], *, now_ms: int
    ) -> None:
        if not bool(payload.get("enabled")):
            return
        if payload.get("current_child_run_id") and payload.get("current_child_status") in {
            "queued",
            "running",
        }:
            return
        list_fn = getattr(
            self._conversation_engine().meta_sqlite, "list_projected_lane_messages", None
        )
        if not callable(list_fn):
            return
        for spec in payload.get("trigger_specs", []):
            if str(spec.get("type") or "") != "message arrival" or not bool(spec.get("enabled", True)):
                continue
            selector = dict(spec.get("selector") or {})
            inbox_id = selector.get("inbox_id")
            if not inbox_id:
                continue
            cooldown_ms = int(spec.get("cooldown_ms", 0) or 0)
            rows = list_fn(
                namespace=str(payload.get("storage_namespace") or "default"),
                inbox_id=str(inbox_id),
                status="pending",
            )
            latest_ms = 0
            for row in rows:
                latest_ms = max(
                    latest_ms,
                    int(getattr(row, "updated_at_ms", 0) or 0),
                    int(getattr(row, "created_at_ms", 0) or 0),
                    int(getattr(row, "created_at", 0) or 0) * 1000,
                )
            if latest_ms <= int(payload.get("last_message_seen_ms", 0) or 0):
                continue
            last_triggered_at_ms = int(payload.get("last_triggered_at_ms", 0) or 0)
            if cooldown_ms > 0 and last_triggered_at_ms > 0 and now_ms - last_triggered_at_ms < cooldown_ms:
                continue
            self._trigger_service(
                service_id,
                trigger_type="message arrival",
                payload={"inbox_id": str(inbox_id)},
                force=False,
            )
            payload["last_message_seen_ms"] = latest_ms
            self._store_projection_payload(service_id, payload)
            return

    def _evaluate_graph_trigger(
        self, service_id: str, payload: dict[str, Any], *, now_ms: int
    ) -> None:
        if not bool(payload.get("enabled")):
            return
        if payload.get("current_child_run_id") and payload.get("current_child_status") in {
            "queued",
            "running",
        }:
            return
        for spec in payload.get("trigger_specs", []):
            if str(spec.get("type") or "") != "graph change" or not bool(spec.get("enabled", True)):
                continue
            selector = dict(spec.get("selector") or {})
            engine_name = str(selector.get("engine") or "workflow").strip().lower()
            cooldown_ms = int(spec.get("cooldown_ms", 0) or 0)
            if engine_name == "conversation":
                engine = self._conversation_engine()
            else:
                engine = self._workflow_engine()
            namespace = str(selector.get("namespace") or getattr(engine, "namespace", "default") or "default")
            getter = getattr(engine.meta_sqlite, "get_latest_entity_event_seq", None)
            if not callable(getter):
                continue
            latest_seq = int(getter(namespace=namespace) or 0)
            if latest_seq <= int(payload.get("last_graph_event_seq", 0) or 0):
                continue
            last_triggered_at_ms = int(payload.get("last_triggered_at_ms", 0) or 0)
            if cooldown_ms > 0 and last_triggered_at_ms > 0 and now_ms - last_triggered_at_ms < cooldown_ms:
                continue
            self._trigger_service(
                service_id,
                trigger_type="graph change",
                payload={"engine": engine_name, "namespace": namespace},
                force=False,
            )
            payload["last_graph_event_seq"] = latest_seq
            self._store_projection_payload(service_id, payload)
            return

    def _append_definition(self, definition: ServiceDefinition) -> None:
        now = _now_ms()
        properties = {
            "service_id": definition.service_id,
            "service_kind": definition.service_kind,
            "target_kind": definition.target_kind,
            "target_ref": definition.target_ref,
            "target_config_json": _json_text(definition.target_config),
            "enabled": definition.enabled,
            "autostart": definition.autostart,
            "restart_policy_json": _json_text(definition.restart_policy),
            "heartbeat_ttl_ms": int(definition.heartbeat_ttl_ms),
            "trigger_specs_json": _json_text(
                [spec.model_dump(mode="python") for spec in definition.trigger_specs]
            ),
            "security_scope": definition.security_scope,
            "storage_namespace": definition.storage_namespace,
            "execution_namespace": definition.execution_namespace,
            "created_at_ms": int(definition.created_at_ms),
            "updated_at_ms": int(definition.updated_at_ms),
        }
        node = Node(
            id=f"service_def:{definition.service_id}:{now}:{uuid.uuid4().hex}",
            label=f"service_definition:{definition.service_id}",
            type="entity",
            summary=f"Service definition {definition.service_id}",
            mentions=[Grounding(spans=[Span.from_dummy_for_workflow(definition.service_id)])],
            properties=properties,
            metadata={
                "entity_type": "service_definition",
                "artifact_kind": "service_definition",
                "service_id": definition.service_id,
                "service_kind": definition.service_kind,
                "target_kind": definition.target_kind,
                "target_ref": definition.target_ref,
                "enabled": definition.enabled,
                "updated_at_ms": definition.updated_at_ms,
                "created_at_ms": definition.created_at_ms,
                "in_conversation_chain": False,
            },
        )
        self._workflow_engine().write.add_node(node)

    def _append_event(
        self, *, service_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        now = _now_ms()
        node = Node(
            id=f"service_evt:{service_id}:{now}:{uuid.uuid4().hex}",
            label=f"service_event:{event_type}",
            type="entity",
            summary=f"Service event {event_type} for {service_id}",
            mentions=[Grounding(spans=[Span.from_dummy_for_workflow(service_id)])],
            properties={
                "payload_json": _json_text(payload),
                **{
                    str(key): _json_value(value)
                    for key, value in payload.items()
                    if isinstance(key, str)
                },
            },
            metadata={
                "entity_type": "service_event",
                "artifact_kind": "service_event",
                "service_id": service_id,
                "service_event_type": event_type,
                "ts_ms": now,
                "in_conversation_chain": False,
            },
        )
        self._workflow_engine().write.add_node(node)

    def _latest_definition(self, service_id: str) -> ServiceDefinition | None:
        nodes = self._workflow_engine().read.get_nodes(
            where={
                "$and": [
                    {"entity_type": "service_definition"},
                    {"service_id": str(service_id)},
                ]
            },
            limit=500,
        )
        if not nodes:
            return None
        latest = max(
            nodes,
            key=lambda node: int((getattr(node, "metadata", {}) or {}).get("updated_at_ms", 0) or 0),
        )
        raw = dict(getattr(latest, "properties", {}) or {})
        payload = {
            "service_id": str(raw.get("service_id") or service_id),
            "service_kind": str(raw.get("service_kind") or "service"),
            "target_kind": str(raw.get("target_kind") or "workflow"),
            "target_ref": str(raw.get("target_ref") or ""),
            "target_config": json.loads(str(raw.get("target_config_json") or "{}")),
            "enabled": bool(raw.get("enabled", True)),
            "autostart": bool(raw.get("autostart", False)),
            "restart_policy": json.loads(str(raw.get("restart_policy_json") or "{}")),
            "heartbeat_ttl_ms": int(raw.get("heartbeat_ttl_ms", 60_000) or 60_000),
            "trigger_specs": json.loads(str(raw.get("trigger_specs_json") or "[]")),
            "security_scope": str(raw.get("security_scope") or "workflow"),
            "storage_namespace": str(raw.get("storage_namespace") or "workflow"),
            "execution_namespace": str(raw.get("execution_namespace") or "workflow"),
            "created_at_ms": int(raw.get("created_at_ms", 0) or 0),
            "updated_at_ms": int(raw.get("updated_at_ms", 0) or 0),
        }
        if not payload:
            return None
        return ServiceDefinition.model_validate(payload)

    def _all_service_ids(self) -> list[str]:
        nodes = self._workflow_engine().read.get_nodes(
            where={"entity_type": "service_definition"},
            limit=10_000,
        )
        service_ids = {
            str((getattr(node, "metadata", {}) or {}).get("service_id") or "")
            for node in nodes
        }
        return sorted(service_id for service_id in service_ids if service_id)

    def _ensure_all_projections(self) -> None:
        meta = self._workflow_engine().meta_sqlite
        existing = {
            str((row.get("payload") or {}).get("service_id") or row.get("key") or "")
            for row in meta.list_named_projections(SERVICE_PROJECTION_NAMESPACE)
        }
        for service_id in self._all_service_ids():
            if service_id not in existing:
                self._rebuild_projection(service_id)

    def _matching_trigger_spec(
        self, *, definition: ServiceDefinition, trigger_type: str
    ) -> dict[str, Any] | None:
        for spec in definition.trigger_specs:
            payload = spec.model_dump(mode="python")
            if str(payload.get("type") or "").strip().lower() == str(trigger_type).strip().lower():
                return payload
        if trigger_type in {"autostart", "restart"}:
            return {
                "type": trigger_type,
                "enabled": True,
                "selector": {},
                "debounce_ms": 0,
                "cooldown_ms": 0,
            }
        return None

    def _rebuild_projection(self, service_id: str) -> None:
        definition = self._latest_definition(service_id)
        if definition is None:
            return
        events = self.list_service_events(service_id, limit=10_000)
        payload = ServiceProjectionRow(
            service_id=definition.service_id,
            service_kind=definition.service_kind,
            target_kind=definition.target_kind,
            target_ref=definition.target_ref,
            enabled=definition.enabled,
            autostart=definition.autostart,
            storage_namespace=definition.storage_namespace,
            execution_namespace=definition.execution_namespace,
            security_scope=definition.security_scope,
            lifecycle_status="stopped" if not definition.enabled else "degraded",
            health_status="stopped" if not definition.enabled else "degraded",
            heartbeat_ttl_ms=int(definition.heartbeat_ttl_ms),
            restart_policy=dict(definition.restart_policy or {}),
            trigger_specs=[spec.model_dump(mode="python") for spec in definition.trigger_specs],
            created_at_ms=int(definition.created_at_ms or _now_ms()),
            updated_at_ms=int(definition.updated_at_ms or _now_ms()),
        ).model_dump(mode="python")
        for event in events:
            event_type = str(event.get("event_type") or "")
            evt_payload = dict(event.get("payload") or {})
            ts_ms = int(event.get("ts_ms", 0) or 0)
            payload["updated_at_ms"] = max(int(payload.get("updated_at_ms", 0) or 0), ts_ms)
            if event_type == "service.heartbeat":
                payload["last_heartbeat_ms"] = ts_ms
            elif event_type == "service.enabled":
                payload["enabled"] = bool(evt_payload.get("enabled", True))
            elif event_type == "service.triggered":
                payload["last_trigger_type"] = str(evt_payload.get("trigger_type") or "")
                payload["last_triggered_at_ms"] = ts_ms
                payload["lifecycle_status"] = "starting"
            elif event_type == "service.starting":
                payload["lifecycle_status"] = "starting"
                payload["health_status"] = "healthy"
            elif event_type == "service.healthy":
                payload["lifecycle_status"] = "healthy"
                payload["health_status"] = str(evt_payload.get("health_status") or "healthy")
            elif event_type == "service.degraded":
                payload["lifecycle_status"] = "degraded"
                payload["health_status"] = str(evt_payload.get("health_status") or "degraded")
            elif event_type == "service.run_spawned":
                payload["current_child_run_id"] = str(evt_payload.get("run_id") or "")
                payload["current_child_started_at_ms"] = ts_ms
                payload["current_child_status"] = "queued"
                payload["lifecycle_status"] = "starting"
            elif event_type == "service.restarting":
                payload["lifecycle_status"] = "restarting"
                payload["restart_not_before_ms"] = evt_payload.get("restart_not_before_ms")
                payload["restart_count"] = int(
                    evt_payload.get("restart_count", payload.get("restart_count", 0))
                    or payload.get("restart_count", 0)
                    or 0
                )
            elif event_type == "service.stopped":
                payload["enabled"] = bool(evt_payload.get("enabled", payload["enabled"]))
                if not payload["enabled"] or str(evt_payload.get("status") or "") in {
                    "succeeded",
                    "cancelled",
                    "failed",
                }:
                    payload["lifecycle_status"] = "stopped" if not payload["enabled"] else payload["lifecycle_status"]
                    payload["health_status"] = "stopped" if not payload["enabled"] else payload["health_status"]
            elif event_type == "service.run_failed":
                payload["current_child_status"] = str(evt_payload.get("status") or "failed")
                payload["last_handled_child_run_id"] = str(evt_payload.get("run_id") or "")
        current_child_run_id = str(payload.get("current_child_run_id") or "")
        if current_child_run_id:
            run = self.run_registry.get_run(current_child_run_id)
            if run is not None:
                payload["current_child_status"] = str(run.get("status") or "")
        self._evaluate_health(service_id, payload, now_ms=_now_ms())
        self._store_projection_payload(service_id, payload)

    def _store_projection_payload(self, service_id: str, payload: dict[str, Any]) -> None:
        payload["updated_at_ms"] = int(payload.get("updated_at_ms") or _now_ms())
        getter = getattr(self._workflow_engine().meta_sqlite, "get_latest_entity_event_seq", None)
        latest_seq = 0
        if callable(getter):
            latest_seq = int(getter(namespace=getattr(self._workflow_engine(), "namespace", "default")) or 0)
        self._workflow_engine().meta_sqlite.replace_named_projection(
            SERVICE_PROJECTION_NAMESPACE,
            str(service_id),
            payload,
            last_authoritative_seq=latest_seq,
            last_materialized_seq=latest_seq,
            projection_schema_version=1,
            materialization_status="ready",
        )

    def _projection(self, service_id: str) -> dict[str, Any] | None:
        return self._workflow_engine().meta_sqlite.get_named_projection(
            SERVICE_PROJECTION_NAMESPACE, str(service_id)
        )

    def _projection_payload(self, service_id: str) -> dict[str, Any] | None:
        projection = self._projection(service_id)
        if projection is None:
            return None
        return dict(projection.get("payload") or {})

    def _workflow_engine(self) -> Any:
        return self.get_workflow_engine()

    def _conversation_engine(self) -> Any:
        return self.get_conversation_engine()
