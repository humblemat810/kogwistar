"""Python callback worker for ADR-015 Rust-owned durable runtime lanes.

Worker callbacks run outside Rust/SQLite transactions.  A local journal makes
completed callback results replayable after response loss.  A process that
finds an ``executing`` journal row refuses to run that callback again because
its side-effect outcome is ambiguous; operators may then reconcile it using
the stable lane ``message_id`` idempotency key.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import queue
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from kogwistar.id_provider import stable_id

if TYPE_CHECKING:
    from kogwistar.runtime.runtime import StepContext


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
    metadata: Mapping[str, Any]

    @property
    def id(self) -> str:
        return self.edge_id


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(_canonical(value))
    except (TypeError, ValueError) as error:
        raise RustWorkerError(f"{field} must be JSON-only: {error}") from error


def _durable_state(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    durable = {
        key: item
        for key, item in value.items()
        if key not in {"_deps", "dream_deps"}
    }
    copied = _json_copy(durable, field=field)
    if not isinstance(copied, dict):
        raise RustWorkerError(f"{field} must be a JSON object")
    return copied


class RustStepResolverAdapter:
    """Map one frozen Rust lane request onto an existing sync Python resolver."""

    def __init__(
        self,
        step_resolver: Callable[[str], Callable[[StepContext], Any]],
        *,
        predicate_registry: Mapping[str, Callable[..., Any]] | None = None,
        dependency_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if not callable(step_resolver):
            raise TypeError("step_resolver must be callable")
        self.step_resolver = step_resolver
        self.predicate_registry = dict(predicate_registry or {})
        self.dependency_provider = dependency_provider
        self.cache_dir = os.fspath(cache_dir) if cache_dir is not None else None

    def _routes(self, payload: Mapping[str, Any], node_id: str) -> list[_FrozenWorkerRoute]:
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
            aliases = [str(value) for value in (raw.get("aliases") or [])]
            label = aliases[0] if aliases else target.rsplit("|", 1)[-1]
            priority = int(raw.get("priority", 100))
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
                    join_mask=int(raw.get("join_mask", 0)),
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

    def __call__(self, work: dict[str, Any]) -> dict[str, Any]:
        # Keep these imports local: this module deliberately avoids importing
        # the public runtime while its protocol helpers are imported.
        from kogwistar.runtime.models import RunFailure
        from kogwistar.runtime.runtime import StepContext

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
        self._reject_unsupported_op(op)
        routes = self._routes(payload, node_id)

        raw_state = payload.get("state", payload.get("initial_state", {}))
        if not isinstance(raw_state, dict):
            raise RustWorkerError("claimed work state must be an object")
        state = _durable_state(raw_state, field="claimed work state")
        before = _json_copy(state, field="claimed work state")
        if self.dependency_provider is not None:
            dependencies = self.dependency_provider(work)
            if not isinstance(dependencies, Mapping):
                raise RustWorkerError("dependency_provider must return a mapping")
            state["_deps"] = dependencies

        try:
            resolver = self.step_resolver(op)
        except Exception as error:
            raise RustWorkerError(f"cannot resolve frozen op {op!r}: {error}") from error
        if not callable(resolver):
            raise RustWorkerError(f"step_resolver({op!r}) returned a non-callable")
        if inspect.iscoroutinefunction(resolver):
            raise RustWorkerError("async resolver callbacks are not in worker contract v1")
        message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        lane_message_attempts: list[dict[str, Any]] = []

        def record_lane_message(**kwargs: Any) -> dict[str, str]:
            lane_message_attempts.append(dict(kwargs))
            return {"message_id": ""}

        ctx = StepContext(
            run_id=str(payload.get("run_id") or work.get("run_id") or ""),
            workflow_id=str(payload.get("workflow_id") or ""),
            workflow_node_id=node_id,
            op=op,
            token_id=str(payload.get("token_id") or work.get("run_id") or ""),
            attempt=int(payload.get("attempt", 1)),
            step_seq=int(payload.get("step_seq", 0)),
            cache_dir=self.cache_dir,
            conversation_id=str(payload.get("conversation_id") or "") or None,
            turn_node_id=str(payload.get("turn_node_id") or "") or None,
            message_queue=message_queue,
            lane_message_sender=record_lane_message,
            state=state,
        )
        try:
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
        result: Any,
        state: dict[str, Any],
        before: Any,
        routes: list[_FrozenWorkerRoute],
        node_id: str,
        message_queue: queue.Queue[dict[str, Any]],
        lane_message_attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply v1 result restrictions shared by sync-v1 and async-v2 workers."""
        from kogwistar.runtime.base_runtime import apply_state_update_inplace
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
        state_update: list[Any] = []
        if changed:
            state_update.append(["u", changed])
        state_schema = _json_copy(
            getattr(self.step_resolver, "_state_schema", {}) or {},
            field="resolver state schema",
        )
        result_state_update = _json_copy(
            result.state_update, field="result.state_update"
        )
        result_update = _json_copy(result.update, field="result.update")
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
            normalized: dict[str, dict[str, Any]] = {"u": {}, "e": {}}
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
        state_update.extend(result_state_update)
        predicted_state = _json_copy(before, field="claimed work state")
        apply_state_update_inplace(
            predicted_state,
            state_update,
            None,
            state_schema=state_schema,
        )
        computed = compute_route_next(
            edges=routes,
            state=predicted_state,
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
        effect: dict[str, Any] = {
            "status": status,
            "state_update": state_update,
            "state_schema": state_schema,
            "successors": successors,
            "route_next": get_route_next_names(result),
            "result": {
                "workflow_status": workflow_status,
                "final_state": predicted_state,
            },
        }
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

    async def execute(self, work: dict[str, Any]) -> dict[str, Any]:
        from kogwistar.runtime.models import RunFailure
        from kogwistar.runtime.runtime import StepContext

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
        self._reject_unsupported_op(op)
        routes = self._routes(payload, node_id)
        raw_state = payload.get("state", payload.get("initial_state", {}))
        if not isinstance(raw_state, dict):
            raise RustWorkerError("claimed work state must be an object")
        state = _durable_state(raw_state, field="claimed work state")
        before = _json_copy(state, field="claimed work state")
        if self.dependency_provider is not None:
            dependencies = self.dependency_provider(work)
            if not isinstance(dependencies, Mapping):
                raise RustWorkerError("dependency_provider must return a mapping")
            state["_deps"] = dependencies
        try:
            resolve_async = getattr(self.step_resolver, "resolve_async", None)
            resolver = (
                resolve_async(op) if callable(resolve_async) else self.step_resolver(op)
            )
        except Exception as error:
            raise RustWorkerError(f"cannot resolve frozen op {op!r}: {error}") from error
        if not callable(resolver):
            raise RustWorkerError(f"step_resolver({op!r}) returned a non-callable")
        message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        lane_message_attempts: list[dict[str, Any]] = []

        def record_lane_message(**kwargs: Any) -> dict[str, str]:
            lane_message_attempts.append(dict(kwargs))
            return {"message_id": ""}

        ctx = StepContext(
            run_id=str(payload.get("run_id") or work.get("run_id") or ""),
            workflow_id=str(payload.get("workflow_id") or ""),
            workflow_node_id=node_id,
            op=op,
            token_id=str(payload.get("token_id") or work.get("run_id") or ""),
            attempt=int(payload.get("attempt", 1)),
            step_seq=int(payload.get("step_seq", 0)),
            cache_dir=self.cache_dir,
            conversation_id=str(payload.get("conversation_id") or "") or None,
            turn_node_id=str(payload.get("turn_node_id") or "") or None,
            message_queue=message_queue,
            lane_message_sender=record_lane_message,
            state=state,
        )
        try:
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
        return self._effect_from_result(
            result=result,
            state=state,
            before=before,
            routes=routes,
            node_id=node_id,
            message_queue=message_queue,
            lane_message_attempts=lane_message_attempts,
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _work_digest(work: Mapping[str, Any]) -> str:
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
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def begin(self, *, message_id: str, work_digest: str) -> dict[str, Any] | None:
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

    def complete(self, *, message_id: str, result: Mapping[str, Any]) -> None:
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
        execute: Callable[[dict[str, Any]], Mapping[str, Any]],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        self.worker_id = worker_id
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
        step_resolver: Callable[[str], Callable[[Any], Any]],
        predicate_registry: Mapping[str, Callable[..., Any]] | None = None,
        dependency_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> RustRuntimeWorker:
        return cls(
            base_url=base_url,
            worker_id=worker_id,
            journal_path=journal_path,
            execute=RustStepResolverAdapter(
                step_resolver,
                predicate_registry=predicate_registry,
                dependency_provider=dependency_provider,
                cache_dir=cache_dir,
            ),
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

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
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
    ) -> list[dict[str, Any]]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("limit and lease_seconds must be positive")
        if run_id is not None and not run_id.strip():
            raise ValueError("run_id must not be empty")
        payload: dict[str, Any] = {
            "claimed_by": self.worker_id,
            "limit": limit,
            "lease_seconds": lease_seconds,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        value = self._post(
            "/internal/runtime/claim",
            payload,
        )
        work = value.get("work")
        if not isinstance(work, list) or not all(isinstance(item, dict) for item in work):
            raise RustWorkerError("runtime claim response has invalid work list")
        return work

    @staticmethod
    def _result_envelope(
        work: Mapping[str, Any], effect: Mapping[str, Any]
    ) -> dict[str, Any]:
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
        empty = sorted(key for key, value in required.items() if not value)
        if empty:
            raise RustWorkerError(f"claimed work missing identity fields: {empty!r}")

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
        return {
            "handoff": required,
            "effect": restricted,
        }

    def process(self, work: dict[str, Any]) -> dict[str, Any]:
        message_id = str(work.get("message_id") or "")
        if not message_id:
            raise RustWorkerError("claimed work missing message_id")
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        op = payload.get("op")
        if not isinstance(op, str) or not op.strip():
            raise RustWorkerError(
                "claimed work lacks frozen workflow op; refusing node-id resolver guess"
            )
        digest = _work_digest(work)
        envelope = self.journal.begin(message_id=message_id, work_digest=digest)
        if envelope is None:
            effect = self.execute(dict(work))
            if not isinstance(effect, Mapping):
                raise RustWorkerError("worker callback must return an object")
            envelope = self._result_envelope(work, effect)
            self.journal.complete(message_id=message_id, result=envelope)
        else:
            handoff = envelope.get("handoff")
            if not isinstance(handoff, dict):
                raise RustWorkerError("journaled result lacks handoff object")
            handoff["claimed_by"] = work.get("claimed_by")
        return self._post("/internal/runtime/results", envelope)

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
        execute: Callable[[dict[str, Any]], Any],
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        self.worker_id = worker_id
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
        step_resolver: Callable[[str], Callable[[Any], Any]],
        predicate_registry: Mapping[str, Callable[..., Any]] | None = None,
        dependency_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> "AsyncRustRuntimeWorker":
        return cls(
            base_url=base_url,
            worker_id=worker_id,
            journal_path=journal_path,
            execute=AsyncRustStepResolverAdapter(
                step_resolver,
                predicate_registry=predicate_registry,
                dependency_provider=dependency_provider,
                cache_dir=cache_dir,
            ).execute,
            headers=headers,
            timeout=timeout,
            client=client,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
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
    ) -> list[dict[str, Any]]:
        if limit <= 0 or lease_seconds <= 0:
            raise ValueError("limit and lease_seconds must be positive")
        if run_id is not None and not run_id.strip():
            raise ValueError("run_id must not be empty")
        payload: dict[str, Any] = {
            "claimed_by": self.worker_id,
            "limit": limit,
            "lease_seconds": lease_seconds,
        }
        if run_id is not None:
            payload["run_id"] = run_id
        value = await self._post("/internal/runtime/claim", payload)
        work = value.get("work")
        if not isinstance(work, list) or not all(isinstance(item, dict) for item in work):
            raise RustWorkerError("runtime claim response has invalid work list")
        return work

    async def process(self, work: dict[str, Any]) -> dict[str, Any]:
        message_id = str(work.get("message_id") or "")
        if not message_id:
            raise RustWorkerError("claimed work missing message_id")
        payload = work.get("payload")
        if not isinstance(payload, dict):
            raise RustWorkerError("claimed work payload must be an object")
        op = payload.get("op")
        if not isinstance(op, str) or not op.strip():
            raise RustWorkerError(
                "claimed work lacks frozen workflow op; refusing node-id resolver guess"
            )
        digest = _work_digest(work)
        envelope = self.journal.begin(message_id=message_id, work_digest=digest)
        if envelope is None:
            effect = self.execute(dict(work))
            if inspect.isawaitable(effect):
                effect = await effect
            if not isinstance(effect, Mapping):
                raise RustWorkerError("worker callback must return an object")
            envelope = RustRuntimeWorker._result_envelope(work, effect)
            self.journal.complete(message_id=message_id, result=envelope)
        else:
            handoff = envelope.get("handoff")
            if not isinstance(handoff, dict):
                raise RustWorkerError("journaled result lacks handoff object")
            handoff["claimed_by"] = work.get("claimed_by")
        return await self._post("/internal/runtime/results", envelope)

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
    "AsyncRustRuntimeWorker",
    "AsyncRustStepResolverAdapter",
    "RustRuntimeWorker",
    "RustStepResolverAdapter",
    "RustWorkerError",
    "WorkerResultJournal",
]
