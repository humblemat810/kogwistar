"""Python callback worker for ADR-015 Rust-owned durable runtime lanes.

Worker callbacks run outside Rust/SQLite transactions.  A local journal makes
completed callback results replayable after response loss.  A process that
finds an ``executing`` journal row refuses to run that callback again because
its side-effect outcome is ambiguous; operators may then reconcile it using
the stable lane ``message_id`` idempotency key.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import os
import queue
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import httpx

from kogwistar.id_provider import stable_id
from kogwistar.implementation_selection import (
    ImplementationSelectionSnapshot,
    callback_execution_scope,
    capture_implementation_selection,
    implementation_selection_bound,
)
from kogwistar.json_types import JsonValue
from kogwistar.runtime.models import StepRunResult

from .contract import Predicate
from .strict_json import normalize_json_value, strict_json_dumps

JsonObject = dict[str, JsonValue]

# A missing target is legacy-compatible; when present, Python workers must
# never accept work explicitly addressed to another executor.
PYTHON_EXECUTOR_TARGET = "python-worker"

if TYPE_CHECKING:
    from kogwistar.runtime.runtime import StepContext


class RustStepCallback(Protocol):
    """Callback resolved for one durable runtime operation."""

    def __call__(self, context: StepContext) -> StepRunResult: ...


class RustStepResolver(Protocol):
    def __call__(self, op: str) -> RustStepCallback: ...


class RustDependencyProvider(Protocol):
    def __call__(self, work: Mapping[str, JsonValue]) -> Mapping[str, object]: ...


RustPredicate = Predicate


class RustWorkerExecutor(Protocol):
    def __call__(self, work: JsonObject) -> Mapping[str, JsonValue]: ...


class AsyncRustWorkerExecutor(Protocol):
    async def __call__(self, work: JsonObject) -> Mapping[str, JsonValue]: ...


class RustWorkerError(RuntimeError):
    """Rust worker protocol or transport failure."""


class AmbiguousWorkerExecution(RustWorkerError):
    """Callback began before restart, but no durable result was recorded."""


@dataclass(frozen=True)
class _FrozenWorkerRoute:
    edge_id: str
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    predicate: str | None
    priority: int
    is_default: bool
    multiplicity: str
    source_fanout: bool
    join_mask: int
    label: str
    aliases: tuple[str, ...]
    metadata: Mapping[str, JsonValue]

    @property
    def id(self) -> str:
        return self.edge_id


def _json_copy(value: object, *, field: str) -> JsonValue:
    try:
        return normalize_json_value(value, field=field)
    except (TypeError, ValueError) as error:
        raise RustWorkerError(f"{field} must be JSON-only: {error}") from error


def _json_object_copy(value: object, *, field: str) -> JsonObject:
    copied = _json_copy(value, field=field)
    if not isinstance(copied, dict):
        raise RustWorkerError(f"{field} must be a JSON object")
    return copied


def _with_worker_execution_trace(
    effect: Mapping[str, JsonValue], work: Mapping[str, JsonValue], worker_id: str
) -> JsonObject:
    """Attach operational worker attribution without treating it as attestation."""
    traced = dict(effect)
    raw_traces = traced.get("trace_events", [])
    if not isinstance(raw_traces, list):
        raise RustWorkerError("worker effect trace_events must be an array")
    traces = _json_copy(raw_traces, field="worker effect trace_events")
    if not isinstance(traces, list):
        raise RustWorkerError("worker effect trace_events must be an array")

    payload = work.get("payload")
    if not isinstance(payload, dict):
        raise RustWorkerError("claimed work payload must be an object")
    event: JsonObject = {
        "schema_version": 1,
        "event": "runtime.python_worker.result_prepared",
        "executor_target": PYTHON_EXECUTOR_TARGET,
        "worker_id": worker_id,
        "identity_source": "worker_configuration",
        "lease_claim_id": str(work["claimed_by"]),
        "message_id": str(work["message_id"]),
        "run_id": str(work["run_id"]),
        "step_id": str(work["step_id"]),
        "correlation_id": str(work["correlation_id"]),
        "operation": str(payload["op"]),
        "op_version": str(payload.get("op_version") or "v1"),
    }
    environment_id = payload.get("op_environment_id")
    if isinstance(environment_id, str):
        event["required_environment_id"] = environment_id
    traces.append(event)
    traced["trace_events"] = traces
    return cast(JsonObject, traced)


def _with_worker_submission_trace(
    envelope: Mapping[str, JsonValue], work: Mapping[str, JsonValue], worker_id: str
) -> JsonObject:
    """Record the lease used for this delivery, including replay deliveries."""
    traced = _json_copy(envelope, field="worker result envelope")
    if not isinstance(traced, dict):
        raise RustWorkerError("worker result envelope must be an object")
    effect = traced.get("effect")
    if not isinstance(effect, dict):
        raise RustWorkerError("worker result envelope effect must be an object")
    raw_traces = effect.get("trace_events", [])
    if not isinstance(raw_traces, list):
        raise RustWorkerError("worker effect trace_events must be an array")
    traces = _json_copy(raw_traces, field="worker effect trace_events")
    if not isinstance(traces, list):
        raise RustWorkerError("worker effect trace_events must be an array")

    payload = work.get("payload")
    if not isinstance(payload, dict):
        raise RustWorkerError("claimed work payload must be an object")
    traces.append(
        {
            "schema_version": 1,
            "event": "runtime.python_worker.result_submitted",
            "executor_target": PYTHON_EXECUTOR_TARGET,
            "worker_id": worker_id,
            "identity_source": "worker_configuration",
            "lease_claim_id": str(work["claimed_by"]),
            "message_id": str(work["message_id"]),
            "run_id": str(work["run_id"]),
            "step_id": str(work["step_id"]),
            "correlation_id": str(work["correlation_id"]),
            "operation": str(payload["op"]),
            "op_version": str(payload.get("op_version") or "v1"),
        }
    )
    effect["trace_events"] = traces
    return cast(JsonObject, traced)


def _json_int(value: JsonValue | None, default: int = 0) -> int:
    """Decode an integer field from a JSON payload without widening its type."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _json_object(value: JsonValue | None) -> JsonObject:
    if isinstance(value, dict):
        return value
    return {}


def _json_string_list(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _durable_state(value: Mapping[str, object], *, field: str) -> JsonObject:
    durable = {
        key: item
        for key, item in value.items()
        if key not in {"_deps", "dream_deps"}
    }
    copied = _json_copy(durable, field=field)
    if not isinstance(copied, dict):
        raise RustWorkerError(f"{field} must be a JSON object")
    return copied


def _environment_recipient_id(environment_id: str) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(environment_id.encode()).digest())
    return f"python-worker-env-{digest.decode().rstrip('=')}"


class RustStepResolverAdapter:
    """Map one frozen Rust lane request onto an existing sync Python resolver."""

    def __init__(
        self,
        step_resolver: RustStepResolver,
        *,
        predicate_registry: Mapping[str, RustPredicate] | None = None,
        dependency_provider: RustDependencyProvider | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        implementation_selection_snapshot: ImplementationSelectionSnapshot | None = None,
    ) -> None:
        if not callable(step_resolver):
            raise TypeError("step_resolver must be callable")
        self.step_resolver = step_resolver
        self.predicate_registry = dict(predicate_registry or {})
        self.dependency_provider = dependency_provider
        self.cache_dir = os.fspath(cache_dir) if cache_dir is not None else None
        self._implementation_selection_snapshot = (
            implementation_selection_snapshot or capture_implementation_selection()
        )

    def _runtime_state(self, work: JsonObject, durable_state: JsonObject) -> dict[str, object]:
        """Attach process-local dependencies without treating them as wire JSON."""
        state: dict[str, object] = dict(durable_state)
        if self.dependency_provider is not None:
            dependencies = self.dependency_provider(work)
            if not isinstance(dependencies, Mapping):
                raise RustWorkerError("dependency_provider must return a mapping")
            state["_deps"] = dependencies
        return state

    @property
    def claim_recipient_ids(self) -> tuple[str, ...]:
        environments = getattr(self.step_resolver, "op_environment_ids", {})
        versioned_environments = getattr(
            self.step_resolver, "versioned_environment_ids", {}
        )
        environment_values = [
            *environments.values(),
            *versioned_environments.values(),
        ] if isinstance(environments, Mapping) and isinstance(
            versioned_environments, Mapping
        ) else []
        environment_ids = (
            {str(value) for value in environment_values if isinstance(value, str)}
        )
        return tuple(
            sorted(_environment_recipient_id(value) for value in environment_ids)
            + ["python-worker"]
        )

    def _routes(self, payload: Mapping[str, JsonValue], node_id: str) -> list[_FrozenWorkerRoute]:
        raw_routes = payload.get("runtime_routes", [])
        if not isinstance(raw_routes, list):
            raise RustWorkerError("claimed work runtime_routes must be an array")
        routes: list[_FrozenWorkerRoute] = []
        missing_predicates: set[str] = set()
        for index, raw in enumerate(raw_routes):
            if not isinstance(raw, dict):
                raise RustWorkerError(f"runtime_routes[{index}] must be an object")
            source = str(raw.get("source_node_id") or "")
            target = str(raw.get("target_node_id") or "")
            if not source or not target:
                raise RustWorkerError(
                    f"runtime_routes[{index}] requires source_node_id and target_node_id"
                )
            if source != node_id:
                continue
            predicate_value = raw.get("predicate")
            predicate = (
                str(predicate_value)
                if predicate_value is not None and str(predicate_value)
                else None
            )
            if predicate is not None and predicate not in self.predicate_registry:
                missing_predicates.add(predicate)
            edge_id = str(raw.get("edge_id") or f"{source}->{target}")
            aliases = _json_string_list(raw.get("aliases"))
            label = aliases[0] if aliases else target.rsplit("|", 1)[-1]
            priority = _json_int(raw.get("priority"), 100)
            is_default = bool(raw.get("is_default", False))
            multiplicity = str(raw.get("multiplicity") or "one")
            routes.append(
                _FrozenWorkerRoute(
                    edge_id=edge_id,
                    source_ids=(source,),
                    target_ids=(target,),
                    predicate=predicate,
                    priority=priority,
                    is_default=is_default,
                    multiplicity=multiplicity,
                    source_fanout=bool(raw.get("source_fanout", False)),
                    join_mask=_json_int(raw.get("join_mask")),
                    label=label,
                    aliases=tuple(aliases),
                    metadata={
                        "wf_predicate": predicate,
                        "wf_priority": priority,
                        "wf_is_default": is_default,
                        "wf_multiplicity": multiplicity,
                    },
                )
            )
        if missing_predicates:
            raise RustWorkerError(
                f"frozen workflow uses unregistered predicates: {sorted(missing_predicates)!r}"
            )
        return routes

    def _reject_unsupported_op(self, op: str) -> None:
        for attribute, capability in [
            ("nested_ops", "nested workflow"),
            ("sandboxed_ops", "sandbox"),
        ]:
            values = getattr(self.step_resolver, attribute, ())
            if op in values:
                raise RustWorkerError(
                    f"Rust worker protocol does not yet represent {capability} op {op!r}"
                )

    def _validate_op_version(self, op: str, payload: Mapping[str, JsonValue]) -> str:
        requested = payload.get("op_version", "v1")
        if not isinstance(requested, str) or not requested.strip():
            raise RustWorkerError("claimed work op_version must be a non-empty string")
        supports_version = getattr(self.step_resolver, "supports_version", None)
        if callable(supports_version):
            if not supports_version(op, requested):
                registered_ops = getattr(self.step_resolver, "ops", set())
                if op not in registered_ops and getattr(
                    self.step_resolver, "default", None
                ) is None:
                    return requested
                registered = getattr(self.step_resolver, "op_versions", {})
                current = (
                    registered.get(op, "v1")
                    if isinstance(registered, Mapping)
                    else "v1"
                )
                raise RustWorkerError(
                    f"workflow op {op!r} requires version {requested!r}, "
                    f"but this worker registers {current!r}"
                )
            return requested
        registered = getattr(self.step_resolver, "op_versions", {})
        current = (
            registered.get(op, "v1") if isinstance(registered, Mapping) else "v1"
        )
        if requested != current:
            raise RustWorkerError(
                f"workflow op {op!r} requires version {requested!r}, "
                f"but this worker registers {current!r}"
            )
        return requested

    def _validate_op_environment(
        self, op: str, payload: Mapping[str, JsonValue]
    ) -> str | None:
        requested = payload.get("op_environment_id")
        if requested is not None and (
            not isinstance(requested, str) or not requested.strip()
        ):
            raise RustWorkerError(
                "claimed work op_environment_id must be a non-empty string"
            )
        registered = getattr(self.step_resolver, "op_environment_ids", {})
        environment_for_version = getattr(
            self.step_resolver, "environment_for_version", None
        )
        if callable(environment_for_version):
            current = environment_for_version(op, str(payload.get("op_version", "v1")))
        else:
            current = registered.get(op) if isinstance(registered, Mapping) else None
        if requested != current:
            raise RustWorkerError(
                f"workflow op {op!r} requires environment {requested!r}, "
                f"but this worker registers {current!r}"
            )
        return requested

    @staticmethod
    def _validate_executor_target(
        payload: Mapping[str, JsonValue], op: str
    ) -> None:
        requested = payload.get("executor_target")
        if requested is not None and requested != PYTHON_EXECUTOR_TARGET:
            raise RustWorkerError(
                "Python worker cannot execute payload for executor target "
                f"{requested!r}"
            )
        if op.startswith("rust-native/"):
            raise RustWorkerError(
                "Python worker cannot execute a rust-native operation"
            )

    @staticmethod
    def _validate_work_identity(work: Mapping[str, object]) -> None:
        """Reject work that cannot be correlated before invoking user code."""
        required = ("message_id", "run_id", "step_id", "claimed_by", "correlation_id")
        missing: list[str] = []
        for field in required:
            value = work.get(field)
            if not isinstance(value, str) or not value.strip():
                missing.append(field)
        if missing:
            raise RustWorkerError(
                f"claimed work missing identity fields: {missing!r}"
            )

    def validate_work(self, work: JsonObject) -> None:
        """Check static compatibility before the journal marks work executing."""
        self._validate_work_identity(work)
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        protocol = str(payload.get("worker_protocol") or "sync-v1")
        if protocol != "sync-v1":
            raise RustWorkerError(
                f"sync Python worker cannot execute worker protocol {protocol!r}"
            )
        if payload.get("resume_effect") is not None:
            return
        op = str(payload.get("op") or "")
        if not op:
            raise RustWorkerError("claimed work requires a frozen workflow op")
        self._validate_executor_target(payload, op)
        self._validate_op_version(op, payload)
        self._validate_op_environment(op, payload)

    @implementation_selection_bound
    def __call__(self, work: JsonObject) -> JsonObject:
        # Keep these imports local: this module deliberately avoids importing
        # the public runtime while its protocol helpers are imported.
        from kogwistar.runtime.models import RunFailure
        from kogwistar.runtime.runtime import StepContext

        self.validate_work(work)

        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        protocol = str(payload.get("worker_protocol") or "sync-v1")
        if protocol != "sync-v1":
            raise RustWorkerError(
                f"sync Python worker cannot execute worker protocol {protocol!r}"
            )
        resume_effect = payload.get("resume_effect")
        if resume_effect is not None:
            copied_effect = _json_copy(resume_effect, field="claimed resume effect")
            if not isinstance(copied_effect, dict):
                raise RustWorkerError("claimed resume effect must be an object")
            if str(copied_effect.get("status") or "") not in {
                "success",
                "failed",
            }:
                raise RustWorkerError(
                    "claimed resume effect status must be success or failed"
                )
            return copied_effect
        op = str(payload.get("op") or "")
        node_id = str(payload.get("node_id") or work.get("step_id") or "")
        if not op or not node_id:
            raise RustWorkerError("claimed work requires frozen op and node_id")
        op_version = self._validate_op_version(op, payload)
        op_environment_id = self._validate_op_environment(op, payload)
        self._reject_unsupported_op(op)
        routes = self._routes(payload, node_id)

        raw_state = payload.get("state", payload.get("initial_state", {}))
        if not isinstance(raw_state, dict):
            raise RustWorkerError("claimed work state must be an object")
        before = _json_object_copy(raw_state, field="claimed work state")
        state = self._runtime_state(work, before)

        try:
            resolve_version = getattr(self.step_resolver, "resolve_version", None)
            resolver = (
                resolve_version(op, op_version)
                if callable(resolve_version)
                else self.step_resolver(op)
            )
        except Exception as error:
            raise RustWorkerError(f"cannot resolve frozen op {op!r}: {error}") from error
        if not callable(resolver):
            raise RustWorkerError(f"step_resolver({op!r}) returned a non-callable")
        if inspect.iscoroutinefunction(resolver):
            raise RustWorkerError("async resolver callbacks are not in worker contract v1")
        message_queue: queue.Queue[JsonObject] = queue.Queue()
        lane_message_attempts: list[JsonObject] = []

        def record_lane_message(**kwargs: JsonValue) -> dict[str, str]:
            lane_message_attempts.append(dict(kwargs))
            return {"message_id": ""}

        ctx = StepContext(
            run_id=str(payload.get("run_id") or work.get("run_id") or ""),
            workflow_id=str(payload.get("workflow_id") or ""),
            workflow_node_id=node_id,
            op=op,
            op_version=op_version,
            op_environment_id=op_environment_id,
            idempotency_key=str(work.get("message_id") or "") or None,
            token_id=str(payload.get("token_id") or work.get("run_id") or ""),
            attempt=_json_int(payload.get("attempt"), 1),
            step_seq=_json_int(payload.get("step_seq")),
            cache_dir=self.cache_dir,
            conversation_id=str(payload.get("conversation_id") or "") or None,
            turn_node_id=str(payload.get("turn_node_id") or "") or None,
            message_queue=message_queue,
            lane_message_sender=record_lane_message,
            state=state,
            authority_context={"effective_capabilities": ()},
        )
        try:
            with callback_execution_scope():
                result = resolver(ctx)
        except RustWorkerError:
            raise
        except Exception as error:
            import traceback

            result = RunFailure(
                conversation_node_id=node_id,
                state_update=[],
                errors=[str(error), traceback.format_exc()],
            )
        if inspect.isawaitable(result):
            raise RustWorkerError("async resolver callbacks are not in worker contract v1")
        result = cast(StepRunResult, result)
        return self._effect_from_result(
            result=result,
            state=state,
            before=before,
            routes=routes,
            node_id=node_id,
            message_queue=message_queue,
            lane_message_attempts=lane_message_attempts,
        )

    def _effect_from_result(
        self,
        *,
        result: StepRunResult,
        state: Mapping[str, object],
        before: JsonObject,
        routes: list[_FrozenWorkerRoute],
        node_id: str,
        message_queue: queue.Queue[JsonObject],
        lane_message_attempts: list[JsonObject],
    ) -> JsonObject:
        """Apply v1 result restrictions shared by sync-v1 and async-v2 workers."""
        from kogwistar.runtime.base_runtime import (
            RuntimeStateUpdate,
            apply_state_update_inplace,
        )
        from kogwistar.runtime.models import (
            RunFailure,
            RunSuccess,
            RunSuspended,
            get_route_next_names,
        )
        from kogwistar.runtime.routing import compute_route_next

        if lane_message_attempts:
            raise RustWorkerError("ctx.send_lane_message is not in worker contract v1")
        if not isinstance(result, (RunSuccess, RunFailure, RunSuspended)):
            raise RustWorkerError("resolver must return RunSuccess, RunFailure, or RunSuspended")
        if not message_queue.empty():
            raise RustWorkerError("ctx.publish is not in worker contract v1")
        if isinstance(result, RunSuccess) and result.workflow_invocations:
            raise RustWorkerError("nested workflow invocations are not in worker contract v1")
        after = _durable_state(state, field="resolver state")
        for key in set(before).union(after):
            if key.startswith("_rt_") and before.get(key) != after.get(key):
                raise RustWorkerError(f"resolver may not mutate scheduler state key {key!r}")
        changed = {
            key: value
            for key, value in after.items()
            if key not in before or before[key] != value
        }
        deleted = before.keys() - after.keys()
        if deleted:
            raise RustWorkerError(
                "direct resolver state deletion is not in the runtime state-update contract: "
                f"{sorted(deleted)!r}"
            )
        state_update: list[JsonValue] = []
        if changed:
            state_update.append(["u", changed])
        state_schema_value = _json_copy(
            getattr(self.step_resolver, "_state_schema", {}) or {},
            field="resolver state schema",
        )
        if not isinstance(state_schema_value, dict):
            raise RustWorkerError("resolver state schema must be an object")
        state_schema = {str(key): str(value) for key, value in state_schema_value.items()}
        result_state_update_value = _json_copy(
            result.state_update, field="result.state_update"
        )
        if not isinstance(result_state_update_value, list):
            raise RustWorkerError("result.state_update must be an array")
        result_state_update = cast(list[list[JsonValue]], result_state_update_value)
        result_update_value = _json_copy(result.update, field="result.update")
        result_update = (
            result_update_value if isinstance(result_update_value, dict) else {}
        )
        result_keys = {
            str(key)
            for item in result_state_update
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict)
            for key in item[1]
        }
        if isinstance(result_update, dict):
            result_keys.update(str(key) for key in result_update)
        forbidden_result_keys = sorted(
            key
            for key in result_keys
            if key in {"_deps", "dream_deps"} or key.startswith("_rt_")
        )
        if forbidden_result_keys:
            raise RustWorkerError(
                "resolver result may not persist runtime plumbing keys: "
                f"{forbidden_result_keys!r}"
            )
        if result_update and result_state_update:
            raise RustWorkerError("result may use either update or state_update, not both")
        if result_update:
            normalized: dict[str, JsonObject] = {"u": {}, "e": {}}
            for key, value in result_update.items():
                mode = "e" if str(state_schema.get(key) or "u") == "a" else "u"
                if mode not in normalized:
                    raise RustWorkerError(
                        f"resolver state schema has unsupported mode {mode!r} for {key!r}"
                    )
                normalized[mode][key] = value
            result_state_update.extend(
                [mode, values] for mode, values in normalized.items() if values
            )
        state_update.extend(cast(list[JsonValue], result_state_update))
        predicted_state = _json_copy(before, field="claimed work state")
        if not isinstance(predicted_state, dict):
            raise RustWorkerError("claimed work state must be an object")
        apply_state_update_inplace(
            cast(dict[str, object], predicted_state),
            cast(RuntimeStateUpdate, state_update),
            None,
            state_schema=state_schema,
        )
        computed = compute_route_next(
            edges=cast(list[object], routes),
            state=cast(Mapping[str, object], predicted_state),
            last_result=result,
            fanout=any(route.source_fanout for route in routes),
            predicate_registry=self.predicate_registry,
            _native_disabled=True,
        )
        route_by_target = {route.target_ids[0]: route for route in routes}
        successors = [
            {
                "node_id": target,
                "join_mask": route_by_target[target].join_mask,
            }
            for target in computed.next_node_ids
        ]
        status = {
            "success": "success",
            "suspended": "suspended",
            "failure": "failed",
        }[result.status]
        workflow_status = {
            "success": "succeeded",
            "suspended": "suspended",
            "failure": "failed",
        }[result.status]
        effect = cast(JsonObject, {
            "status": status,
            "state_update": cast(JsonValue, state_update),
            "state_schema": cast(JsonValue, state_schema),
            "successors": cast(JsonValue, successors),
            "route_next": get_route_next_names(result),
            "result": {
                "workflow_status": workflow_status,
                "final_state": predicted_state,
            },
        })
        if isinstance(result, RunFailure):
            effect["errors"] = list(result.errors)
        if isinstance(result, RunSuspended):
            effect["wait_reason"] = result.wait_reason
            effect["resume_payload"] = _json_copy(
                result.resume_payload, field="result.resume_payload"
            )
        for field in ("usage", "trace_events"):
            value = getattr(result, field, None)
            if value is not None:
                effect[field] = _json_copy(value, field=f"result.{field}")
        return effect


class AsyncRustStepResolverAdapter(RustStepResolverAdapter):
    """Await `async-v2` callbacks while preserving the v1 durable effect DTO.

    The protocol changes callback execution ownership only. Result mapping,
    frontier validation, journal semantics, and Rust reducer input remain the
    exact restricted worker-effect contract shared with sync-v1.
    """

    def validate_work(self, work: JsonObject) -> None:
        self._validate_work_identity(work)
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        if str(payload.get("worker_protocol") or "") != "async-v2":
            raise RustWorkerError(
                "async Python worker requires worker_protocol='async-v2'"
            )
        if payload.get("resume_effect") is not None:
            return
        op = str(payload.get("op") or "")
        if not op:
            raise RustWorkerError("claimed work requires a frozen workflow op")
        self._validate_executor_target(payload, op)
        self._validate_op_version(op, payload)
        self._validate_op_environment(op, payload)

    @implementation_selection_bound
    async def execute(self, work: JsonObject) -> JsonObject:
        from kogwistar.runtime.models import RunFailure
        from kogwistar.runtime.runtime import StepContext

        self.validate_work(work)

        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        if str(payload.get("worker_protocol") or "") != "async-v2":
            raise RustWorkerError(
                "async Python worker requires worker_protocol='async-v2'"
            )
        resume_effect = payload.get("resume_effect")
        if resume_effect is not None:
            copied_effect = _json_copy(resume_effect, field="claimed resume effect")
            if not isinstance(copied_effect, dict):
                raise RustWorkerError("claimed resume effect must be an object")
            if str(copied_effect.get("status") or "") not in {"success", "failed"}:
                raise RustWorkerError(
                    "claimed resume effect status must be success or failed"
                )
            return copied_effect
        op = str(payload.get("op") or "")
        node_id = str(payload.get("node_id") or work.get("step_id") or "")
        if not op or not node_id:
            raise RustWorkerError("claimed work requires frozen op and node_id")
        op_version = self._validate_op_version(op, payload)
        op_environment_id = self._validate_op_environment(op, payload)
        self._reject_unsupported_op(op)
        routes = self._routes(payload, node_id)
        raw_state = payload.get("state", payload.get("initial_state", {}))
        if not isinstance(raw_state, dict):
            raise RustWorkerError("claimed work state must be an object")
        before = _json_object_copy(raw_state, field="claimed work state")
        state = self._runtime_state(work, before)
        try:
            resolve_async = getattr(self.step_resolver, "resolve_async", None)
            resolve_async_version = getattr(
                self.step_resolver, "resolve_async_version", None
            )
            if callable(resolve_async_version):
                resolver = resolve_async_version(op, op_version)
            else:
                resolver = (
                    resolve_async(op)
                    if callable(resolve_async)
                    else self.step_resolver(op)
                )
        except Exception as error:
            raise RustWorkerError(f"cannot resolve frozen op {op!r}: {error}") from error
        if not callable(resolver):
            raise RustWorkerError(f"step_resolver({op!r}) returned a non-callable")
        message_queue: queue.Queue[JsonObject] = queue.Queue()
        lane_message_attempts: list[JsonObject] = []

        def record_lane_message(**kwargs: JsonValue) -> dict[str, str]:
            lane_message_attempts.append(dict(kwargs))
            return {"message_id": ""}

        ctx = StepContext(
            run_id=str(payload.get("run_id") or work.get("run_id") or ""),
            workflow_id=str(payload.get("workflow_id") or ""),
            workflow_node_id=node_id,
            op=op,
            op_version=op_version,
            op_environment_id=op_environment_id,
            idempotency_key=str(work.get("message_id") or "") or None,
            token_id=str(payload.get("token_id") or work.get("run_id") or ""),
            attempt=_json_int(payload.get("attempt"), 1),
            step_seq=_json_int(payload.get("step_seq")),
            cache_dir=self.cache_dir,
            conversation_id=str(payload.get("conversation_id") or "") or None,
            turn_node_id=str(payload.get("turn_node_id") or "") or None,
            message_queue=message_queue,
            lane_message_sender=record_lane_message,
            state=state,
            authority_context={"effective_capabilities": ()},
        )
        try:
            with callback_execution_scope():
                result = resolver(ctx)
                if inspect.isawaitable(result):
                    result = await result
        except RustWorkerError:
            raise
        except Exception as error:
            import traceback

            result = RunFailure(
                conversation_node_id=node_id,
                state_update=[],
                errors=[str(error), traceback.format_exc()],
            )
        result = cast(StepRunResult, result)
        return self._effect_from_result(
            result=result,
            state=state,
            before=before,
            routes=routes,
            node_id=node_id,
            message_queue=message_queue,
            lane_message_attempts=lane_message_attempts,
        )


def _canonical(value: object) -> str:
    return strict_json_dumps(value)


def _work_digest(work: Mapping[str, JsonValue]) -> str:
    immutable = {
        key: work.get(key)
        for key in (
            "message_id",
            "run_id",
            "step_id",
            "correlation_id",
            "expected_event_seq",
            "payload",
        )
    }
    return hashlib.sha256(_canonical(immutable).encode("utf-8")).hexdigest()


class WorkerResultJournal:
    """Small SQLite journal shared across worker process restarts."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rust_worker_results(
                        message_id TEXT PRIMARY KEY,
                        work_digest TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('executing', 'completed')),
                        result_json TEXT
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        with closing(connection.execute("PRAGMA busy_timeout=30000")):
            pass
        return connection

    def begin(self, *, message_id: str, work_digest: str) -> JsonObject | None:
        with closing(self._connect()) as connection:
            with connection:
                row = connection.execute(
                    "SELECT work_digest, status, result_json FROM rust_worker_results "
                    "WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO rust_worker_results(message_id, work_digest, status) "
                        "VALUES (?, ?, 'executing')",
                        (message_id, work_digest),
                    )
                    return None
                stored_digest, status, result_json = row
                if stored_digest != work_digest:
                    raise RustWorkerError(
                        f"lane message {message_id!r} changed across retry"
                    )
                if status == "executing":
                    raise AmbiguousWorkerExecution(
                        f"lane message {message_id!r} may already have run; "
                        "reconcile using message_id before retry"
                    )
                if not isinstance(result_json, str):
                    raise RustWorkerError(
                        f"lane message {message_id!r} has completed journal row without result"
                    )
                result = json.loads(result_json)
                if not isinstance(result, dict):
                    raise RustWorkerError(
                        f"lane message {message_id!r} journal result is not an object"
                    )
                return result

    def complete(self, *, message_id: str, result: Mapping[str, JsonValue]) -> None:
        result_json = _canonical(dict(result))
        with closing(self._connect()) as connection:
            with connection:
                changed = connection.execute(
                    "UPDATE rust_worker_results SET status = 'completed', result_json = ? "
                    "WHERE message_id = ? AND status = 'executing'",
                    (result_json, message_id),
                ).rowcount
                if changed != 1:
                    raise RustWorkerError(
                        f"lane message {message_id!r} journal ownership changed"
                    )


class RustRuntimeWorker:
    """Claim Rust runtime work, execute Python callback, submit durable result."""

    def __init__(
        self,
        *,
        base_url: str,
        worker_id: str,
        journal_path: str | os.PathLike[str],
        execute: RustWorkerExecutor,
        recipient_ids: Sequence[str] = ("python-worker",),
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        normalized_recipient_ids = tuple(dict.fromkeys(recipient_ids))
        if not normalized_recipient_ids or any(
            not value.strip()
            or value != value.strip()
            or len(value.encode("utf-8")) > 256
            for value in normalized_recipient_ids
        ):
            raise ValueError(
                "recipient_ids must be canonical non-empty values up to 256 bytes"
            )
        self.worker_id = worker_id
        self.recipient_ids = normalized_recipient_ids
        self.execute = execute
        self.journal = WorkerResultJournal(journal_path)
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"), headers=dict(headers or {}), timeout=timeout
        )

    @classmethod
    def from_step_resolver(
        cls,
        *,
        base_url: str,
        worker_id: str,
        journal_path: str | os.PathLike[str],
        step_resolver: RustStepResolver,
        predicate_registry: Mapping[str, RustPredicate] | None = None,
        dependency_provider: RustDependencyProvider | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        implementation_selection_snapshot: ImplementationSelectionSnapshot | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> RustRuntimeWorker:
        adapter = RustStepResolverAdapter(
            step_resolver,
            predicate_registry=predicate_registry,
            dependency_provider=dependency_provider,
            cache_dir=cache_dir,
            implementation_selection_snapshot=implementation_selection_snapshot,
        )
        return cls(
            base_url=base_url,
            worker_id=worker_id,
            journal_path=journal_path,
            execute=adapter,
            recipient_ids=adapter.claim_recipient_ids,
            headers=headers,
            timeout=timeout,
            client=client,
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> RustRuntimeWorker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _post(self, path: str, payload: Mapping[str, JsonValue]) -> JsonObject:
        try:
            response = self.client.post(path, json=dict(payload))
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RustWorkerError(str(error)) from error
        value = response.json()
        if not isinstance(value, dict):
            raise RustWorkerError(f"{path} returned non-object JSON")
        return value

    def claim(
        self,
        *,
        limit: int = 1,
        lease_seconds: int = 60,
        run_id: str | None = None,
    ) -> list[JsonObject]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("limit and lease_seconds must be positive")
        if run_id is not None and not run_id.strip():
            raise ValueError("run_id must not be empty")
        payload: JsonObject = {
            "claimed_by": self.worker_id,
            "limit": limit,
            "lease_seconds": lease_seconds,
        }
        if self.recipient_ids != ("python-worker",):
            payload["recipient_ids"] = list(self.recipient_ids)
        if run_id is not None:
            payload["run_id"] = run_id
        value = self._post(
            "/internal/runtime/claim",
            payload,
        )
        work = value.get("work")
        if not isinstance(work, list) or not all(isinstance(item, dict) for item in work):
            raise RustWorkerError("runtime claim response has invalid work list")
        return [cast(JsonObject, item) for item in work]

    @staticmethod
    def _result_envelope(
        work: Mapping[str, JsonValue], effect: Mapping[str, JsonValue]
    ) -> JsonObject:
        RustStepResolverAdapter._validate_work_identity(work)
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        message_id = str(work.get("message_id") or "")
        run_id = str(work.get("run_id") or "")
        step_id = str(work.get("step_id") or "")
        claimed_by = str(work.get("claimed_by") or "")
        correlation_id = str(work.get("correlation_id") or "")
        required = {
            "message_id": message_id,
            "run_id": run_id,
            "step_id": step_id,
            "claimed_by": claimed_by,
            "correlation_id": correlation_id,
        }
        restricted = {
            "contract_version": 1,
            "effect_id": str(
                stable_id("runtime.worker.result", message_id, run_id, step_id)
            ),
            **dict(effect),
        }
        forbidden = {
            "run_id",
            "workflow_id",
            "conversation_id",
            "step_seq",
            "node_id",
            "token_id",
            "parent_token_id",
            "expected_event_seq",
            "kind",
            "frontier",
            "transition_id",
        }.intersection(restricted)
        if forbidden:
            raise RustWorkerError(
                f"worker effect may not override scheduler fields: {sorted(forbidden)!r}"
            )
        allowed = {
            "contract_version",
            "effect_id",
            "status",
            "state_update",
            "update",
            "state_schema",
            "successors",
            "route_next",
            "result",
            "errors",
            "wait_reason",
            "resume_payload",
            "usage",
            "trace_events",
        }
        unknown = set(restricted).difference(allowed)
        if unknown:
            raise RustWorkerError(f"worker effect has unknown fields: {sorted(unknown)!r}")
        fixed = {
            "contract_version": 1,
            "effect_id": str(
                stable_id("runtime.worker.result", message_id, run_id, step_id)
            ),
        }
        for key, value in fixed.items():
            if key in restricted and restricted[key] != value:
                raise RustWorkerError(
                    f"worker effect may not override scheduler field {key!r}"
                )
            restricted[key] = value
        successors = restricted.get("successors", [])
        if not isinstance(successors, list):
            raise RustWorkerError("worker effect successors must be an array")
        return cast(JsonObject, {
            "handoff": required,
            "effect": cast(JsonValue, restricted),
            "transition": None,
        })

    def process(self, work: JsonObject) -> JsonObject:
        message_id = str(work.get("message_id") or "")
        if not message_id:
            raise RustWorkerError("claimed work missing message_id")
        RustStepResolverAdapter._validate_work_identity(work)
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        op = payload.get("op")
        if not isinstance(op, str) or not op.strip():
            raise RustWorkerError(
                "claimed work lacks frozen workflow op; refusing node-id resolver guess"
            )
        validator = getattr(self.execute, "validate_work", None)
        if not callable(validator):
            validator = getattr(getattr(self.execute, "__self__", None), "validate_work", None)
        if callable(validator):
            validator(work)
        digest = _work_digest(work)
        envelope = self.journal.begin(message_id=message_id, work_digest=digest)
        if envelope is None:
            effect = self.execute(dict(work))
            if not isinstance(effect, Mapping):
                raise RustWorkerError("worker callback must return an object")
            traced_effect = _with_worker_execution_trace(effect, work, self.worker_id)
            envelope = self._result_envelope(work, traced_effect)
            self.journal.complete(message_id=message_id, result=envelope)
        else:
            handoff = envelope.get("handoff")
            if not isinstance(handoff, dict):
                raise RustWorkerError("journaled result lacks handoff object")
            handoff["claimed_by"] = work.get("claimed_by")
        submitted_envelope = _with_worker_submission_trace(
            envelope, work, self.worker_id
        )
        return self._post("/internal/runtime/results", submitted_envelope)

    def poll_once(
        self,
        *,
        limit: int = 1,
        lease_seconds: int = 60,
        run_id: str | None = None,
    ) -> int:
        work = self.claim(
            limit=limit,
            lease_seconds=lease_seconds,
            run_id=run_id,
        )
        for item in work:
            self.process(item)
        return len(work)


class AsyncRustRuntimeWorker:
    """Async counterpart of :class:`RustRuntimeWorker` for ``async-v2``.

    Callback execution may await, but journal ownership remains a short,
    synchronous SQLite transaction.  Cancellation or process loss between
    ``begin`` and ``complete`` therefore leaves ``executing`` and a later
    worker fails closed rather than replaying an unknown side effect.
    """

    def __init__(
        self,
        *,
        base_url: str,
        worker_id: str,
        journal_path: str | os.PathLike[str],
        execute: AsyncRustWorkerExecutor,
        recipient_ids: Sequence[str] = ("python-worker",),
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        normalized_recipient_ids = tuple(dict.fromkeys(recipient_ids))
        if not normalized_recipient_ids or any(
            not value.strip()
            or value != value.strip()
            or len(value.encode("utf-8")) > 256
            for value in normalized_recipient_ids
        ):
            raise ValueError(
                "recipient_ids must be canonical non-empty values up to 256 bytes"
            )
        self.worker_id = worker_id
        self.recipient_ids = normalized_recipient_ids
        self.execute = execute
        self.journal = WorkerResultJournal(journal_path)
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=dict(headers or {}), timeout=timeout
        )

    @classmethod
    def from_step_resolver(
        cls,
        *,
        base_url: str,
        worker_id: str,
        journal_path: str | os.PathLike[str],
        step_resolver: RustStepResolver,
        predicate_registry: Mapping[str, RustPredicate] | None = None,
        dependency_provider: RustDependencyProvider | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        implementation_selection_snapshot: ImplementationSelectionSnapshot | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> AsyncRustRuntimeWorker:
        adapter = AsyncRustStepResolverAdapter(
            step_resolver,
            predicate_registry=predicate_registry,
            dependency_provider=dependency_provider,
            cache_dir=cache_dir,
            implementation_selection_snapshot=implementation_selection_snapshot,
        )
        return cls(
            base_url=base_url,
            worker_id=worker_id,
            journal_path=journal_path,
            execute=adapter.execute,
            recipient_ids=adapter.claim_recipient_ids,
            headers=headers,
            timeout=timeout,
            client=client,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _post(self, path: str, payload: Mapping[str, JsonValue]) -> JsonObject:
        try:
            response = await self.client.post(path, json=dict(payload))
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RustWorkerError(str(error)) from error
        value = response.json()
        if not isinstance(value, dict):
            raise RustWorkerError(f"{path} returned non-object JSON")
        return value

    async def claim(
        self,
        *,
        limit: int = 1,
        lease_seconds: int = 60,
        run_id: str | None = None,
    ) -> list[JsonObject]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("limit and lease_seconds must be positive")
        if run_id is not None and not run_id.strip():
            raise ValueError("run_id must not be empty")
        payload: JsonObject = {
            "claimed_by": self.worker_id,
            "limit": limit,
            "lease_seconds": lease_seconds,
        }
        if self.recipient_ids != ("python-worker",):
            payload["recipient_ids"] = list(self.recipient_ids)
        if run_id is not None:
            payload["run_id"] = run_id
        value = await self._post("/internal/runtime/claim", payload)
        work = value.get("work")
        if not isinstance(work, list) or not all(isinstance(item, dict) for item in work):
            raise RustWorkerError("runtime claim response has invalid work list")
        return [cast(JsonObject, item) for item in work]

    async def process(self, work: JsonObject) -> JsonObject:
        message_id = str(work.get("message_id") or "")
        if not message_id:
            raise RustWorkerError("claimed work missing message_id")
        RustStepResolverAdapter._validate_work_identity(work)
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        op = payload.get("op")
        if not isinstance(op, str) or not op.strip():
            raise RustWorkerError(
                "claimed work lacks frozen workflow op; refusing node-id resolver guess"
            )
        validator = getattr(self.execute, "validate_work", None)
        if not callable(validator):
            validator = getattr(getattr(self.execute, "__self__", None), "validate_work", None)
        if callable(validator):
            validator(work)
        digest = _work_digest(work)
        envelope = self.journal.begin(message_id=message_id, work_digest=digest)
        if envelope is None:
            effect = self.execute(dict(work))
            if inspect.isawaitable(effect):
                effect = await effect
            if not isinstance(effect, Mapping):
                raise RustWorkerError("worker callback must return an object")
            traced_effect = _with_worker_execution_trace(effect, work, self.worker_id)
            envelope = RustRuntimeWorker._result_envelope(work, traced_effect)
            self.journal.complete(message_id=message_id, result=envelope)
        else:
            handoff = envelope.get("handoff")
            if not isinstance(handoff, dict):
                raise RustWorkerError("journaled result lacks handoff object")
            handoff["claimed_by"] = work.get("claimed_by")
        submitted_envelope = _with_worker_submission_trace(
            envelope, work, self.worker_id
        )
        return await self._post("/internal/runtime/results", submitted_envelope)

    async def poll_once(
        self,
        *,
        limit: int = 1,
        lease_seconds: int = 60,
        run_id: str | None = None,
    ) -> int:
        work = await self.claim(
            limit=limit,
            lease_seconds=lease_seconds,
            run_id=run_id,
        )
        for item in work:
            await self.process(item)
        return len(work)


__all__ = [
    "AmbiguousWorkerExecution",
    "AsyncRustWorkerExecutor",
    "AsyncRustRuntimeWorker",
    "AsyncRustStepResolverAdapter",
    "RustRuntimeWorker",
    "RustStepResolverAdapter",
    "RustWorkerError",
    "WorkerResultJournal",
]
