from __future__ import annotations

"""Core restart-recovery coordination and operator visibility surfaces."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..messaging.models import LaneMessageProjectionRepairResult
from .service_health import ServiceHealthRepairResult


TERMINAL_RUN_STATUSES = {"succeeded", "failed", "cancelled", "completed"}
TERMINAL_CHECKPOINT_STATUSES = {"succeeded", "failed", "cancelled", "completed"}


@dataclass(frozen=True, slots=True)
class ResumePolicy:
    auto_resume: bool = False
    only_restartable: bool = True
    require_resume_marker: bool = True
    resume_runner: Callable[["CheckpointRecoveryState"], Any] | None = None


@dataclass(frozen=True, slots=True)
class RecoverySurface:
    surface_id: str
    surface_kind: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class OutputReconciliationState:
    surface_id: str
    surface_kind: str
    status: str
    expected_version: str | None = None
    observed_version: str | None = None
    drift_detected: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryAction:
    action_kind: str
    surface: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveryFinding:
    severity: str
    surface: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueueRecoveryState:
    job_id: str
    namespace: str
    entity_kind: str
    entity_id: str
    job_kind: str
    status: str
    retry_count: int
    max_retries: int
    lease_until: int | None = None
    next_run_at: int | None = None
    expired_lease: bool = False
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class LaneRecoveryState:
    message_id: str
    namespace: str
    inbox_id: str
    conversation_id: str
    msg_type: str
    status: str
    retry_count: int
    claimed_by: str | None = None
    lease_until: int | None = None
    expired_lease: bool = False
    correlation_id: str | None = None
    run_id: str | None = None
    error_json: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointRecoveryState:
    run_id: str
    namespace: str
    workflow_id: str | None
    conversation_id: str | None
    latest_step_seq: int
    classification: str
    restartable: bool
    resume_marker: bool
    node_id: str
    status: str | None = None


@dataclass(frozen=True, slots=True)
class RunRecoveryState:
    run_id: str
    status: str
    terminal: bool
    namespace: str | None = None
    workflow_id: str | None = None
    conversation_id: str | None = None
    last_event_type: str | None = None
    last_event_seq: int | None = None
    worker_event_count: int = 0


@dataclass(frozen=True, slots=True)
class DeadLetterRecoveryState:
    surface: str
    item_id: str
    namespace: str | None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DaemonHealthState:
    daemon_id: str
    desired_state: str = "running"
    observed_state: str = "unknown"
    last_heartbeat_at: int | None = None
    restart_count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    workspace_id: str
    namespaces: tuple[str, ...]
    repaired_lane_projections: tuple[LaneMessageProjectionRepairResult, ...] = ()
    queues: tuple[QueueRecoveryState, ...] = ()
    lane_rows: tuple[LaneRecoveryState, ...] = ()
    checkpoints: tuple[CheckpointRecoveryState, ...] = ()
    run_history: tuple[RunRecoveryState, ...] = ()
    dead_letters: tuple[DeadLetterRecoveryState, ...] = ()
    daemon_health: tuple[DaemonHealthState, ...] = ()
    app_surfaces: tuple[RecoverySurface | OutputReconciliationState, ...] = ()
    actions: tuple[RecoveryAction, ...] = ()
    findings: tuple[RecoveryFinding, ...] = ()
    resume_policy: ResumePolicy = field(default_factory=ResumePolicy)

    @property
    def repaired_count(self) -> int:
        return sum(int(item.repaired_count) for item in self.repaired_lane_projections)

    @property
    def scanned_count(self) -> int:
        return sum(int(item.scanned_count) for item in self.repaired_lane_projections)


class RecoverySubsystem:
    """Coordinate bounded startup recovery using core durable surfaces.

    The subsystem is deliberately report-first: safe lane projection repair is
    automatic, lease-based redelivery remains owned by queues/lanes, and workflow
    resume needs an explicit policy plus a caller-provided resume hook.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def recover_startup(
        self,
        *,
        workspace_id: str,
        namespaces: list[str],
        app_surfaces: list[RecoverySurface | OutputReconciliationState] | None = None,
        resume_policy: ResumePolicy | None = None,
    ) -> RecoveryReport:
        policy = resume_policy or ResumePolicy()
        normalized = self._dedupe_namespaces(namespaces)
        service_health_repairs = self._repair_service_health(
            workspace_id=workspace_id,
            namespaces=normalized,
        )
        registry = getattr(self.engine, "service_health", None)
        repaired = self.repair_lane_projections(list(normalized))
        report = self._build_report(
            workspace_id=workspace_id,
            namespaces=normalized,
            app_surfaces=app_surfaces,
            resume_policy=policy,
            repaired_lane_projections=tuple(repaired),
            repaired_service_health=service_health_repairs,
        )
        actions = list(report.actions)
        findings = list(report.findings)
        resume_actions, resume_findings = self._maybe_resume(report.checkpoints, policy)
        actions.extend(resume_actions)
        findings.extend(resume_findings)
        return self._replace(report, actions=tuple(actions), findings=tuple(findings))

    def inspect(
        self,
        *,
        workspace_id: str,
        namespaces: list[str],
        app_surfaces: list[RecoverySurface | OutputReconciliationState] | None = None,
    ) -> RecoveryReport:
        normalized = self._dedupe_namespaces(namespaces)
        return self._build_report(
            workspace_id=workspace_id,
            namespaces=normalized,
            app_surfaces=app_surfaces,
            resume_policy=ResumePolicy(),
            repaired_lane_projections=(),
        )

    def repair_lane_projections(
        self, namespaces: list[str]
    ) -> list[LaneMessageProjectionRepairResult]:
        results: list[LaneMessageProjectionRepairResult] = []
        repair = getattr(self.engine, "repair_lane_message_projection", None)
        if not callable(repair):
            return results
        for namespace in self._dedupe_namespaces(namespaces):
            results.append(repair(namespace=namespace))
        return results

    def inspect_queues(self, namespaces: list[str]) -> list[QueueRecoveryState]:
        list_jobs = getattr(getattr(self.engine, "meta_sqlite", None), "list_index_jobs", None)
        if not callable(list_jobs):
            return []
        now = int(time.time())
        states: list[QueueRecoveryState] = []
        for namespace in self._dedupe_namespaces(namespaces):
            for row in list_jobs(namespace=namespace, limit=10_000):
                status = str(self._field(row, "status", "UNKNOWN"))
                lease_until = self._optional_int(self._field(row, "lease_until"))
                states.append(
                    QueueRecoveryState(
                        job_id=str(self._field(row, "job_id", "")),
                        namespace=str(self._field(row, "namespace", namespace)),
                        entity_kind=str(self._field(row, "entity_kind", "")),
                        entity_id=str(self._field(row, "entity_id", "")),
                        job_kind=str(self._field(row, "index_kind", "")),
                        status=status,
                        retry_count=int(self._field(row, "retry_count", 0) or 0),
                        max_retries=int(self._field(row, "max_retries", 0) or 0),
                        lease_until=lease_until,
                        next_run_at=self._optional_int(self._field(row, "next_run_at")),
                        expired_lease=status == "DOING"
                        and lease_until is not None
                        and lease_until < now,
                        last_error=self._optional_str(self._field(row, "last_error")),
                    )
                )
        return states

    def inspect_lane_rows(self, namespaces: list[str]) -> list[LaneRecoveryState]:
        list_rows = getattr(
            getattr(self.engine, "meta_sqlite", None),
            "list_projected_lane_messages",
            None,
        )
        if not callable(list_rows):
            return []
        now = int(time.time())
        states: list[LaneRecoveryState] = []
        for namespace in self._dedupe_namespaces(namespaces):
            for row in list_rows(namespace=namespace, limit=10_000):
                status = str(self._field(row, "status", "unknown"))
                lease_until = self._optional_int(self._field(row, "lease_until"))
                states.append(
                    LaneRecoveryState(
                        message_id=str(self._field(row, "message_id", "")),
                        namespace=str(self._field(row, "namespace", namespace)),
                        inbox_id=str(self._field(row, "inbox_id", "")),
                        conversation_id=str(self._field(row, "conversation_id", "")),
                        msg_type=str(self._field(row, "msg_type", "")),
                        status=status,
                        retry_count=int(self._field(row, "retry_count", 0) or 0),
                        claimed_by=self._optional_str(self._field(row, "claimed_by")),
                        lease_until=lease_until,
                        expired_lease=status == "claimed"
                        and lease_until is not None
                        and lease_until < now,
                        correlation_id=self._optional_str(
                            self._field(row, "correlation_id")
                        ),
                        run_id=self._optional_str(self._field(row, "run_id")),
                        error_json=self._optional_str(self._field(row, "error_json")),
                    )
                )
        return states

    def inspect_checkpoints(
        self, *, namespace: str, workspace_id: str
    ) -> list[CheckpointRecoveryState]:
        del workspace_id
        try:
            nodes = self._checkpoint_nodes(namespace)
        except Exception as exc:
            if self._is_recoverable_checkpoint_lookup_error(exc):
                return []
            raise
        latest_by_run: dict[str, Any] = {}
        for node in nodes:
            md = dict(getattr(node, "metadata", {}) or {})
            run_id = str(md.get("run_id") or getattr(node, "id", ""))
            previous = latest_by_run.get(run_id)
            if previous is None or self._step_seq(node) > self._step_seq(previous):
                latest_by_run[run_id] = node
        return [
            self._checkpoint_state(namespace=namespace, node=node)
            for node in sorted(
                latest_by_run.values(),
                key=lambda n: (str((getattr(n, "metadata", {}) or {}).get("run_id")), self._step_seq(n)),
            )
        ]

    def inspect_run_history(
        self, *, namespace: str, workspace_id: str
    ) -> list[RunRecoveryState]:
        meta = getattr(self.engine, "meta_sqlite", None)
        list_runs = getattr(meta, "list_server_runs", None)
        list_events = getattr(meta, "list_server_run_events", None)
        if not callable(list_runs):
            return []
        out: list[RunRecoveryState] = []
        query_kwargs: dict[str, Any] = {"limit": 10_000}
        if namespace:
            query_kwargs["conversation_id"] = str(namespace)
        try:
            runs = list_runs(**query_kwargs)
        except TypeError:
            runs = list_runs(limit=10_000)
        for run in runs:
            conversation_id = self._optional_str(run.get("conversation_id"))
            if conversation_id != str(namespace):
                continue
            events = list_events(str(run.get("run_id")), after_seq=0, limit=10_000) if callable(list_events) else []
            last_event = events[-1] if events else None
            worker_count = sum(
                1
                for event in events
                if str(event.get("event_type") or "").startswith("worker.")
            )
            status = str(run.get("status") or "unknown")
            out.append(
                RunRecoveryState(
                    run_id=str(run.get("run_id") or ""),
                    status=status,
                    terminal=bool(run.get("terminal"))
                    or status in TERMINAL_RUN_STATUSES,
                    namespace=conversation_id,
                    workflow_id=self._optional_str(run.get("workflow_id")),
                    conversation_id=self._optional_str(run.get("conversation_id")),
                    last_event_type=None
                    if last_event is None
                    else self._optional_str(last_event.get("event_type")),
                    last_event_seq=None
                    if last_event is None
                    else self._optional_int(last_event.get("seq")),
                    worker_event_count=worker_count,
                )
            )
        return out

    def _build_report(
        self,
        *,
        workspace_id: str,
        namespaces: tuple[str, ...],
        app_surfaces: list[RecoverySurface | OutputReconciliationState] | None,
        resume_policy: ResumePolicy,
        repaired_lane_projections: tuple[LaneMessageProjectionRepairResult, ...],
        repaired_service_health: tuple[RecoveryAction, ...] = (),
    ) -> RecoveryReport:
        queues = tuple(self.inspect_queues(list(namespaces)))
        lane_rows = tuple(self.inspect_lane_rows(list(namespaces)))
        checkpoint_states: list[CheckpointRecoveryState] = []
        checkpoint_findings: list[RecoveryFinding] = []
        for namespace in namespaces:
            try:
                checkpoint_states.extend(
                    self.inspect_checkpoints(namespace=namespace, workspace_id=workspace_id)
                )
            except Exception as exc:
                checkpoint_findings.append(
                    RecoveryFinding(
                        severity="error",
                        surface="checkpoints",
                        message="checkpoint inspection failed",
                        details={
                            "namespace": namespace,
                            "workspace_id": workspace_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
        checkpoints = tuple(checkpoint_states)
        run_history_map: dict[str, RunRecoveryState] = {}
        for namespace in namespaces:
            for state in self.inspect_run_history(namespace=namespace, workspace_id=workspace_id):
                run_history_map.setdefault(state.run_id, state)
        run_history = tuple(run_history_map.values())
        dead_letters = tuple(self._dead_letters(queues, lane_rows, run_history))
        daemon_health = tuple(
            self._daemon_health(app_surfaces or [])
            + self._service_health(workspace_id=workspace_id)
        )
        findings = tuple(
            self._findings(
                queues=queues,
                lane_rows=lane_rows,
                checkpoints=checkpoints,
                run_history=run_history,
                daemon_health=daemon_health,
                app_surfaces=tuple(app_surfaces or ()),
            )
        )
        findings = tuple(list(findings) + checkpoint_findings)
        actions = tuple(list(self._repair_actions(repaired_lane_projections)) + list(repaired_service_health))
        return RecoveryReport(
            workspace_id=str(workspace_id),
            namespaces=namespaces,
            repaired_lane_projections=repaired_lane_projections,
            queues=queues,
            lane_rows=lane_rows,
            checkpoints=checkpoints,
            run_history=run_history,
            dead_letters=dead_letters,
            daemon_health=daemon_health,
            app_surfaces=tuple(app_surfaces or ()),
            actions=actions,
            findings=findings,
            resume_policy=resume_policy,
        )

    def _maybe_resume(
        self,
        checkpoints: tuple[CheckpointRecoveryState, ...],
        policy: ResumePolicy,
    ) -> tuple[list[RecoveryAction], list[RecoveryFinding]]:
        actions: list[RecoveryAction] = []
        findings: list[RecoveryFinding] = []
        if not policy.auto_resume:
            return actions, findings
        if policy.resume_runner is None:
            findings.append(
                RecoveryFinding(
                    severity="warning",
                    surface="checkpoints",
                    message="auto_resume requested but no resume_runner was provided",
                )
            )
            return actions, findings
        for checkpoint in checkpoints:
            if checkpoint.classification != "interrupted_restartable":
                continue
            if policy.only_restartable and not checkpoint.restartable:
                continue
            if policy.require_resume_marker and not checkpoint.resume_marker:
                continue
            try:
                result = policy.resume_runner(checkpoint)
            except Exception as exc:
                findings.append(
                    RecoveryFinding(
                        severity="error",
                        surface="checkpoints",
                        message="checkpoint auto-resume failed",
                        details={
                            "run_id": checkpoint.run_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
            else:
                actions.append(
                    RecoveryAction(
                        action_kind="resume_run",
                        surface="checkpoints",
                        status="completed",
                        details={"run_id": checkpoint.run_id, "result": result},
                    )
                )
        return actions, findings

    def _checkpoint_nodes(self, namespace: str) -> list[Any]:
        read = getattr(self.engine, "read", None)
        get_nodes = getattr(read, "get_nodes", None)
        if not callable(get_nodes):
            return []
        where = {"entity_type": "workflow_checkpoint"}
        try:
            from .engine import scoped_namespace

            with scoped_namespace(self.engine, str(namespace)):
                return list(get_nodes(where=where, limit=10_000))
        except Exception as exc:
            if self._is_recoverable_checkpoint_lookup_error(exc):
                return []
            raise

    def _checkpoint_state(self, *, namespace: str, node: Any) -> CheckpointRecoveryState:
        md = dict(getattr(node, "metadata", {}) or {})
        status = self._optional_str(md.get("status") or md.get("run_status"))
        restartable = bool(
            md.get("restartable")
            or md.get("auto_resume")
            or md.get("resume_marker")
            or md.get("restartable_after_crash")
        )
        resume_marker = bool(
            md.get("resume_marker")
            or md.get("auto_resume")
            or md.get("restartable_after_crash")
        )
        suspended = bool(md.get("suspended") or md.get("manual_suspend"))
        terminal = bool(md.get("terminal")) or (status in TERMINAL_CHECKPOINT_STATUSES)
        if terminal:
            classification = "terminal"
        elif suspended and not restartable:
            classification = "suspended_manual"
        elif restartable:
            classification = "interrupted_restartable"
        else:
            classification = "interrupted_unknown"
        return CheckpointRecoveryState(
            run_id=str(md.get("run_id") or getattr(node, "id", "")),
            namespace=str(namespace),
            workflow_id=self._optional_str(md.get("workflow_id")),
            conversation_id=self._optional_str(md.get("conversation_id")),
            latest_step_seq=self._step_seq(node),
            classification=classification,
            restartable=restartable,
            resume_marker=resume_marker,
            node_id=str(getattr(node, "id", "")),
            status=status,
        )

    def _dead_letters(
        self,
        queues: tuple[QueueRecoveryState, ...],
        lane_rows: tuple[LaneRecoveryState, ...],
        run_history: tuple[RunRecoveryState, ...],
    ) -> list[DeadLetterRecoveryState]:
        out: list[DeadLetterRecoveryState] = []
        for job in queues:
            if job.status == "FAILED":
                out.append(
                    DeadLetterRecoveryState(
                        surface="queue",
                        item_id=job.job_id,
                        namespace=job.namespace,
                        reason=job.last_error,
                    )
                )
        for row in lane_rows:
            if row.status in {"dead", "dead_letter", "failed"}:
                out.append(
                    DeadLetterRecoveryState(
                        surface="lane_row",
                        item_id=row.message_id,
                        namespace=row.namespace,
                        reason=row.error_json,
                    )
                )
        for run in run_history:
            if run.status == "failed":
                out.append(
                    DeadLetterRecoveryState(
                        surface="run",
                        item_id=run.run_id,
                        namespace=None,
                        reason=run.last_event_type,
                    )
                )
        return out

    def _findings(
        self,
        *,
        queues: tuple[QueueRecoveryState, ...],
        lane_rows: tuple[LaneRecoveryState, ...],
        checkpoints: tuple[CheckpointRecoveryState, ...],
        run_history: tuple[RunRecoveryState, ...],
        daemon_health: tuple[DaemonHealthState, ...],
        app_surfaces: tuple[RecoverySurface | OutputReconciliationState, ...],
    ) -> list[RecoveryFinding]:
        del run_history
        findings: list[RecoveryFinding] = []
        for job in queues:
            if job.expired_lease:
                findings.append(
                    RecoveryFinding(
                        severity="info",
                        surface="queue",
                        message="job lease is expired and eligible for redelivery",
                        details={"job_id": job.job_id, "namespace": job.namespace},
                    )
                )
        for row in lane_rows:
            if row.expired_lease:
                findings.append(
                    RecoveryFinding(
                        severity="info",
                        surface="lane_row",
                        message="lane-message lease is expired and eligible for redelivery",
                        details={"message_id": row.message_id, "namespace": row.namespace},
                    )
                )
        for checkpoint in checkpoints:
            if checkpoint.classification == "interrupted_unknown":
                findings.append(
                    RecoveryFinding(
                        severity="warning",
                        surface="checkpoints",
                        message="interrupted workflow checkpoint needs operator policy",
                        details={"run_id": checkpoint.run_id},
                    )
                )
        for state in daemon_health:
            if state.observed_state in {"stale", "degraded", "failed", "error"}:
                findings.append(
                    RecoveryFinding(
                        severity="warning",
                        surface="service_health",
                        message="service health needs operator attention",
                        details={
                            "service_id": state.daemon_id,
                            "observed_state": state.observed_state,
                            **dict(state.details or {}),
                        },
                    )
                )
        for surface in app_surfaces:
            if getattr(surface, "status", "") in {"missing", "drift", "error"}:
                findings.append(
                    RecoveryFinding(
                        severity="warning",
                        surface=getattr(surface, "surface_kind", "app_surface"),
                        message="app output surface needs reconciliation",
                        details={
                            "surface_id": getattr(surface, "surface_id", ""),
                            "status": getattr(surface, "status", ""),
                        },
                    )
                )
        return findings

    def _service_health(self, *, workspace_id: str) -> list[DaemonHealthState]:
        registry = getattr(self.engine, "service_health", None)
        list_services = getattr(registry, "list_services", None)
        if not callable(list_services):
            return []
        out: list[DaemonHealthState] = []
        now_ms = int(time.time() * 1000)
        for payload in list_services(workspace_id=workspace_id, limit=10_000):
            last_seen = self._optional_int(payload.get("last_seen_ms"))
            ttl = int(payload.get("heartbeat_ttl_ms", 60_000) or 60_000)
            observed = str(payload.get("status") or "unknown")
            if last_seen is not None and now_ms - last_seen > ttl:
                observed = "stale"
            out.append(
                DaemonHealthState(
                    daemon_id=str(payload.get("service_id") or ""),
                    desired_state="running",
                    observed_state=observed,
                    last_heartbeat_at=last_seen,
                    restart_count=None,
                    details=dict(payload),
                )
            )
        return out

    def _daemon_health(
        self, app_surfaces: list[RecoverySurface | OutputReconciliationState]
    ) -> list[DaemonHealthState]:
        health: list[DaemonHealthState] = []
        for surface in app_surfaces:
            if getattr(surface, "surface_kind", "") != "daemon_health":
                continue
            details = dict(getattr(surface, "details", {}) or {})
            health.append(
                DaemonHealthState(
                    daemon_id=str(getattr(surface, "surface_id", "")),
                    desired_state=str(details.get("desired_state") or "running"),
                    observed_state=str(
                        details.get("observed_state")
                        or getattr(surface, "status", "")
                        or "unknown"
                    ),
                    last_heartbeat_at=self._optional_int(
                        details.get("last_heartbeat_at")
                    ),
                    restart_count=self._optional_int(details.get("restart_count")),
                    details=details,
                )
            )
        return health

    @staticmethod
    def _repair_actions(
        repaired: tuple[LaneMessageProjectionRepairResult, ...]
    ) -> list[RecoveryAction]:
        return [
            RecoveryAction(
                action_kind="repair_lane_projection",
                surface="lane_projection",
                status="completed",
                details={
                    "namespace": item.namespace,
                    "scanned_count": item.scanned_count,
                    "repaired_count": item.repaired_count,
                    "skipped_count": item.skipped_count,
                    "rebuilt": item.rebuilt,
                },
            )
            for item in repaired
        ]

    def _repair_service_health(
        self,
        *,
        workspace_id: str,
        namespaces: tuple[str, ...],
    ) -> tuple[RecoveryAction, ...]:
        registry = getattr(self.engine, "service_health", None)
        repair = getattr(registry, "repair_projection", None)
        if not callable(repair):
            return ()
        repaired: list[RecoveryAction] = []
        result = repair(workspace_id=workspace_id)
        repaired_ids = tuple(getattr(result, "repaired_service_ids", ()) or ())
        if not repaired_ids and int(getattr(result, "repaired_count", 0) or 0) <= 0:
            return ()
        services_by_id: dict[str, dict[str, Any]] = {}
        list_services = getattr(registry, "list_services", None)
        if callable(list_services):
            try:
                for payload in list_services(workspace_id=workspace_id, limit=10_000):
                    if isinstance(payload, dict) and str(payload.get("service_id") or ""):
                        services_by_id[str(payload["service_id"])] = dict(payload)
            except Exception:
                services_by_id = {}
        for service_id in repaired_ids:
            payload = services_by_id.get(str(service_id))
            repaired.append(
                RecoveryAction(
                    action_kind="repair_service_health_projection",
                    surface="service_health",
                    status="completed",
                    details={
                        "workspace_id": workspace_id,
                        "namespace": None if not isinstance(payload, dict) else payload.get("namespace"),
                        "service_id": service_id,
                        "status": None if not isinstance(payload, dict) else payload.get("status"),
                        "repaired_count": int(getattr(result, "repaired_count", 0) or 0),
                        "skipped_count": int(getattr(result, "skipped_count", 0) or 0),
                        "rebuilt_from_sparse_lifecycle": True,
                    },
                )
            )
        return tuple(repaired)

    @staticmethod
    def _dedupe_namespaces(namespaces: list[str]) -> tuple[str, ...]:
        out: list[str] = []
        for namespace in namespaces:
            value = str(namespace)
            if value and value not in out:
                out.append(value)
        return tuple(out)

    @staticmethod
    def _field(row: Any, name: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(name, default)
        return getattr(row, name, default)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value)
        return text if text else None

    @staticmethod
    def _step_seq(node: Any) -> int:
        md = dict(getattr(node, "metadata", {}) or {})
        try:
            return int(md.get("step_seq", -1))
        except Exception:
            return -1

    @staticmethod
    def _is_recoverable_checkpoint_lookup_error(exc: Exception) -> bool:
        current: Exception | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if "Missing Embeddings" in str(current):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _replace(report: RecoveryReport, **changes: Any) -> RecoveryReport:
        data = {
            "workspace_id": report.workspace_id,
            "namespaces": report.namespaces,
            "repaired_lane_projections": report.repaired_lane_projections,
            "queues": report.queues,
            "lane_rows": report.lane_rows,
            "checkpoints": report.checkpoints,
            "run_history": report.run_history,
            "dead_letters": report.dead_letters,
            "daemon_health": report.daemon_health,
            "app_surfaces": report.app_surfaces,
            "actions": report.actions,
            "findings": report.findings,
            "resume_policy": report.resume_policy,
        }
        data.update(changes)
        return RecoveryReport(**data)


__all__ = [
    "CheckpointRecoveryState",
    "DaemonHealthState",
    "DeadLetterRecoveryState",
    "LaneRecoveryState",
    "OutputReconciliationState",
    "QueueRecoveryState",
    "RecoveryAction",
    "RecoveryFinding",
    "RecoveryReport",
    "RecoverySubsystem",
    "RecoverySurface",
    "ResumePolicy",
    "RunRecoveryState",
]
