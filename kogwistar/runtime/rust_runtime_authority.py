"""Public Python facade for the Rust-owned durable workflow scheduler.

Python continues to execute user resolvers through the versioned worker
boundary.  Rust alone admits work, advances the durable frontier, and writes
checkpoints while this facade waits for the public synchronous result shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .design import validate_workflow_design
from .rust_worker import AsyncRustRuntimeWorker, RustRuntimeWorker

if TYPE_CHECKING:
    from .runtime import RunResult, WorkflowRuntime


class RustRuntimeAuthorityError(RuntimeError):
    """Rust runtime authority is unavailable or violates its contract."""


def rust_runtime_authority_url() -> str | None:
    """Return configured authority URL only for explicit Rust runtime mode."""
    if not rust_runtime_authority_selected():
        return None
    value = os.getenv("KOGWISTAR_RUST_RUNTIME_URL", "").strip()
    return value.rstrip("/") or None


def rust_runtime_authority_selected() -> bool:
    """Whether public runtime routing explicitly selected Rust authority."""
    from kogwistar._rust_bridge import runtime_implementation_mode

    return runtime_implementation_mode() == "rust"


def _json_state(initial_state: Mapping[str, Any]) -> dict[str, Any]:
    durable = {
        key: value
        for key, value in initial_state.items()
        if key not in {"_deps", "dream_deps"}
    }
    try:
        value = json.loads(
            json.dumps(durable, ensure_ascii=False, separators=(",", ":"))
        )
    except (TypeError, ValueError) as error:
        raise RustRuntimeAuthorityError(
            f"Rust runtime durable state must be JSON-only: {error}"
        ) from error
    if not isinstance(value, dict):
        raise RustRuntimeAuthorityError("Rust runtime initial state must be an object")
    return value


def _aliases(edge: Any, target: Any, target_id: str) -> list[str]:
    values = [
        getattr(edge, "label", None),
        target_id,
        target_id.rsplit("|", 1)[-1],
        getattr(target, "label", None),
        getattr(target, "op", None),
    ]
    aliases: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in aliases:
            aliases.append(text)
    return aliases


def freeze_runtime_plan(runtime: Any, workflow_id: str) -> dict[str, Any]:
    """Freeze the validated Python graph into the Rust worker contract."""
    from .runtime import _compute_may_reach_join_bitsets

    start, nodes, adjacency = validate_workflow_design(
        workflow_engine=runtime.workflow_engine,
        workflow_id=workflow_id,
        predicate_registry=runtime.predicate_registry,
        resolver=runtime.step_resolver,
    )
    node_ids = [str(node_id) for node_id in nodes]
    join_node_ids = sorted(
        str(node_id)
        for node_id, node in nodes.items()
        if bool((getattr(node, "metadata", None) or {}).get("wf_join", False))
        or str(getattr(node, "op", "")) == "join"
    )
    if len(join_node_ids) >= 63:
        raise RustRuntimeAuthorityError("worker contract v1 supports at most 62 joins")
    may_reach = _compute_may_reach_join_bitsets(
        node_ids=node_ids,
        adj=adjacency,
        join_ids=join_node_ids,
    )
    routes: list[dict[str, Any]] = []
    for source_id, edges in adjacency.items():
        source = nodes[str(source_id)]
        for edge in edges:
            targets = list(getattr(edge, "target_ids", ()) or ())
            if len(targets) != 1:
                raise RustRuntimeAuthorityError(
                    f"workflow edge {getattr(edge, 'id', '')!r} must have one target"
                )
            target_id = str(targets[0])
            target = nodes[target_id]
            routes.append(
                {
                    "edge_id": str(edge.safe_get_id()),
                    "source_node_id": str(source_id),
                    "target_node_id": target_id,
                    "aliases": _aliases(edge, target, target_id),
                    "join_mask": int(may_reach.get(target_id, 0)),
                    "predicate": getattr(edge, "predicate", None),
                    "multiplicity": str(getattr(edge, "multiplicity", None) or "one"),
                    "is_default": bool(getattr(edge, "is_default", False)),
                    "priority": int(getattr(edge, "priority", 100)),
                    "source_fanout": bool(getattr(source, "fanout", False)),
                }
            )
    return {
        "start_node_id": str(start.id),
        "start_join_mask": int(may_reach.get(str(start.id), 0)),
        "join_node_ids": join_node_ids,
        "node_ops": {
            str(node_id): str(getattr(node, "op", "") or "noop")
            for node_id, node in nodes.items()
        },
        "runtime_routes": routes,
    }


class RustRuntimeAuthority:
    """Submit one public run and synchronously pump its Python callbacks."""

    def __init__(
        self,
        *,
        runtime: Any,
        base_url: str,
        cache_dir: str | os.PathLike[str] | None,
        client: httpx.Client | None = None,
    ) -> None:
        self.runtime = runtime
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        headers: dict[str, str] = {}
        token = os.getenv("KOGWISTAR_RUST_RUNTIME_TOKEN", "").strip()
        if token:
            headers["authorization"] = f"Bearer {token}"
        request_timeout = float(
            os.getenv("KOGWISTAR_RUST_RUNTIME_REQUEST_TIMEOUT_SECONDS", "30")
        )
        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=request_timeout,
        )
        journal_root = Path(
            os.getenv("KOGWISTAR_RUST_RUNTIME_JOURNAL_DIR", "").strip()
            or cache_dir
            or Path(tempfile.gettempdir()) / "kogwistar-runtime-worker"
        )
        journal_root.mkdir(parents=True, exist_ok=True)
        worker_id = f"python-facade-{os.getpid()}-{id(runtime):x}"
        self.worker = RustRuntimeWorker.from_step_resolver(
            base_url=self.base_url,
            worker_id=worker_id,
            journal_path=journal_root / "results.sqlite3",
            step_resolver=runtime.step_resolver,
            predicate_registry=runtime.predicate_registry,
            dependency_provider=self._dependencies,
            cache_dir=cache_dir,
            headers=headers,
            client=self.client,
        )
        self._live_dependencies: Mapping[str, Any] = {}

    def _dependencies(self, _work: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._live_dependencies

    def close(self) -> None:
        self.worker.close()
        if self._owns_client:
            self.client.close()

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RustRuntimeAuthorityError(str(error)) from error
        value = response.json()
        if not isinstance(value, dict):
            raise RustRuntimeAuthorityError(f"{path} returned non-object JSON")
        return value

    def _submit_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Retry once only when the response may have been lost after commit."""
        last_error: httpx.TransportError | None = None
        for _attempt in range(2):
            try:
                response = self.client.post("/api/workflow/runs", json=dict(payload))
                response.raise_for_status()
            except httpx.TransportError as error:
                last_error = error
                continue
            except httpx.HTTPError as error:
                raise RustRuntimeAuthorityError(str(error)) from error
            value = response.json()
            if not isinstance(value, dict):
                raise RustRuntimeAuthorityError(
                    "/api/workflow/runs returned non-object JSON"
                )
            return value
        assert last_error is not None
        raise RustRuntimeAuthorityError(str(last_error)) from last_error

    def _final_state(
        self,
        *,
        run_id: str,
        durable_initial: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = self._request_json(
            "GET", f"/api/workflow/runs/{run_id}/checkpoints"
        )
        raw = value.get("checkpoints")
        checkpoints = raw if isinstance(raw, list) else []
        candidates = [item for item in checkpoints if isinstance(item, dict)]
        latest = max(
            candidates,
            key=lambda item: (
                int(item.get("step_seq", -1)),
                int(item.get("event_seq", -1)),
            ),
            default=None,
        )
        state = dict(durable_initial)
        if latest is not None:
            latest_state = latest.get("state")
            if not isinstance(latest_state, dict):
                raise RustRuntimeAuthorityError("Rust checkpoint state is not an object")
            state = dict(latest_state)
        state.pop("_rt_routes", None)
        state.pop("_rt_node_ops", None)
        if dependencies:
            state["_deps"] = dependencies
        return state

    def _client_result_effect(
        self,
        *,
        workflow_id: str,
        suspended_node_id: str,
        client_result: Any,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        from .base_runtime import apply_state_update_inplace
        from .models import RunFailure, RunSuccess, get_route_next_names
        from .routing import compute_route_next

        if not isinstance(client_result, (RunSuccess, RunFailure)):
            raise RustRuntimeAuthorityError(
                "Rust resume contract v1 accepts RunSuccess or RunFailure"
            )
        result_keys = {
            str(key)
            for item in client_result.state_update
            if isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[1], Mapping)
            for key in item[1]
        }
        update = getattr(client_result, "update", None)
        if isinstance(update, Mapping):
            result_keys.update(str(key) for key in update)
        forbidden_result_keys = sorted(
            key
            for key in result_keys
            if key in {"_deps", "dream_deps"} or key.startswith("_rt_")
        )
        if forbidden_result_keys:
            raise RustRuntimeAuthorityError(
                "resume result may not persist runtime plumbing keys: "
                f"{forbidden_result_keys!r}"
            )
        _start, nodes, adjacency = validate_workflow_design(
            workflow_engine=self.runtime.workflow_engine,
            workflow_id=workflow_id,
            predicate_registry=self.runtime.predicate_registry,
            resolver=self.runtime.step_resolver,
        )
        if suspended_node_id not in nodes:
            raise RustRuntimeAuthorityError(
                f"suspended node {suspended_node_id!r} is not in workflow"
            )
        predicted_state = dict(state)
        state_schema = dict(
            getattr(self.runtime.step_resolver, "_state_schema", {}) or {}
        )
        state_update = json.loads(
            json.dumps(client_result.state_update, ensure_ascii=False)
        )
        normalized_update = dict(update) if isinstance(update, Mapping) else None
        apply_state_update_inplace(
            predicted_state,
            state_update,
            normalized_update,
            state_schema=state_schema,
        )
        edges = list(adjacency.get(suspended_node_id, []))
        computed = compute_route_next(
            edges=edges,
            state=predicted_state,
            last_result=client_result,
            fanout=bool(getattr(nodes[suspended_node_id], "fanout", False)),
            predicate_registry=self.runtime.predicate_registry,
            nodes=nodes,
            _native_disabled=True,
        )
        plan = freeze_runtime_plan(self.runtime, workflow_id)
        routes = {
            (route["source_node_id"], route["target_node_id"]): route
            for route in plan["runtime_routes"]
        }
        successors = [
            {
                "node_id": target,
                "join_mask": int(
                    routes[(suspended_node_id, target)]["join_mask"]
                ),
            }
            for target in computed.next_node_ids
        ]
        status = "failed" if isinstance(client_result, RunFailure) else "success"
        effect: dict[str, Any] = {
            "status": status,
            "state_update": state_update,
            "update": update,
            "state_schema": state_schema,
            "successors": successors,
            "route_next": get_route_next_names(client_result),
            "result": {
                "workflow_status": "failed" if status == "failed" else "succeeded",
                "final_state": predicted_state,
            },
        }
        if isinstance(client_result, RunFailure):
            effect["errors"] = list(client_result.errors)
        return _json_state(effect)

    def _pump_existing_run(
        self,
        *,
        run_id: str,
        durable_initial: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> RunResult:
        from .runtime import RunResult

        deadline = time.monotonic() + float(
            os.getenv("KOGWISTAR_RUST_RUNTIME_RUN_TIMEOUT_SECONDS", "300")
        )
        cancelled = False
        status_value: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status_value = self._request_json(
                "GET", f"/api/workflow/runs/{run_id}"
            )
            status = str(status_value.get("status") or "")
            if status in {"succeeded", "failed", "cancelled", "suspended"}:
                break
            if (
                not cancelled
                and self.runtime.cancel_requested is not None
                and self.runtime.cancel_requested(run_id)
            ):
                self._request_json(
                    "POST", f"/api/workflow/runs/{run_id}/cancel"
                )
                cancelled = True
            processed = self.worker.poll_once(
                limit=max(1, int(self.runtime.max_workers)),
                lease_seconds=60,
                run_id=run_id,
            )
            if processed == 0:
                time.sleep(0.01)
        else:
            raise RustRuntimeAuthorityError(
                f"Rust runtime run {run_id!r} exceeded configured timeout"
            )

        final_state = self._final_state(
            run_id=run_id,
            durable_initial=durable_initial,
            dependencies=dependencies,
        )
        server_status = str(status_value.get("status") or "failed")
        status = "failure" if server_status == "failed" else server_status
        raw_error = status_value.get("error")
        if isinstance(raw_error, list):
            errors = [str(value) for value in raw_error]
        elif raw_error in (None, {}, ""):
            errors = []
        else:
            errors = [str(raw_error)]
        return RunResult(
            run_id=run_id,
            final_state=final_state,
            mq=queue.Queue(maxsize=10_000),
            status=status,
            errors=errors,
        )

    def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None,
        initial_state: Mapping[str, Any],
        run_id: str,
    ) -> RunResult:

        if initial_state.get("dream_deps"):
            raise RustRuntimeAuthorityError(
                "dream_deps is not represented by Rust worker contract v1"
            )
        dependencies = initial_state.get("_deps") or {}
        if not isinstance(dependencies, Mapping):
            raise RustRuntimeAuthorityError("initial_state['_deps'] must be a mapping")
        self._live_dependencies = dependencies
        durable_initial = _json_state(initial_state)
        plan = freeze_runtime_plan(self.runtime, workflow_id)
        payload = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "turn_node_id": turn_node_id,
            "initial_state": durable_initial,
            "runtime_kind": "sync",
            **plan,
        }
        submitted = self._submit_run(payload)
        if submitted.get("run_id") != run_id or submitted.get("admission") != "accepted":
            raise RustRuntimeAuthorityError(
                f"Rust scheduler returned invalid admission: {submitted!r}"
            )

        return self._pump_existing_run(
            run_id=run_id,
            durable_initial=durable_initial,
            dependencies=dependencies,
        )

    def resume(
        self,
        *,
        run_id: str,
        suspended_node_id: str,
        suspended_token_id: str,
        client_result: Any,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None,
    ) -> RunResult:
        contract = self._request_json(
            "GET", f"/api/workflow/runs/{run_id}/resume-contract"
        )
        plan = freeze_runtime_plan(self.runtime, workflow_id)
        if (
            contract.get("runtime_routes") != plan["runtime_routes"]
            or contract.get("node_ops") != plan["node_ops"]
        ):
            raise RustRuntimeAuthorityError(
                "current Python workflow differs from the run's frozen resume contract"
            )
        checkpoints = self._request_json(
            "GET", f"/api/workflow/runs/{run_id}/checkpoints"
        ).get("checkpoints")
        entries = checkpoints if isinstance(checkpoints, list) else []
        latest = max(
            (item for item in entries if isinstance(item, dict)),
            key=lambda item: int(item.get("event_seq", -1)),
            default=None,
        )
        if latest is None or not isinstance(latest.get("state"), dict):
            raise RustRuntimeAuthorityError("Rust suspended run has no checkpoint state")
        state = dict(latest["state"])
        effect = self._client_result_effect(
            workflow_id=workflow_id,
            suspended_node_id=suspended_node_id,
            client_result=client_result,
            state=state,
        )
        suspended = contract.get("suspended")
        expected = [suspended_node_id, suspended_token_id]
        if not isinstance(suspended, list) or not any(
            isinstance(item, list)
            and len(item) >= 3
            and [str(item[0]), str(item[2])] == expected
            for item in suspended
        ):
            raise RustRuntimeAuthorityError("requested suspended token is not resumable")
        self._request_json(
            "POST",
            f"/api/workflow/runs/{run_id}/resume",
            json={
                "suspended_node_id": suspended_node_id,
                "suspended_token_id": suspended_token_id,
                "client_result": effect,
                "workflow_id": workflow_id,
                "conversation_id": conversation_id,
                "turn_node_id": turn_node_id,
            },
        )
        dependencies = self._live_dependencies
        return self._pump_existing_run(
            run_id=run_id,
            durable_initial=state,
            dependencies=dependencies,
        )


def run_with_rust_authority(
    runtime: WorkflowRuntime,
    *,
    workflow_id: str,
    conversation_id: str,
    turn_node_id: str | None,
    initial_state: Mapping[str, Any],
    run_id: str,
    cache_dir: str | os.PathLike[str] | None,
) -> RunResult:
    url = rust_runtime_authority_url()
    if url is None:
        raise RustRuntimeAuthorityError("Rust runtime authority URL is not configured")
    authority = RustRuntimeAuthority(
        runtime=runtime,
        base_url=url,
        cache_dir=cache_dir,
    )
    try:
        return authority.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=initial_state,
            run_id=run_id,
        )
    finally:
        authority.close()


class AsyncRustRuntimeAuthority:
    """Async facade over same Rust durable scheduler and worker-effect DTO.

    This deliberately admits only ``runtime_kind=async``.  It does not adapt
    callbacks through a nested event loop; ``async-v2`` is an explicit lane
    protocol and Python remains the callback executor at that boundary.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        base_url: str,
        cache_dir: str | os.PathLike[str] | None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.runtime = runtime
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        headers: dict[str, str] = {}
        token = os.getenv("KOGWISTAR_RUST_RUNTIME_TOKEN", "").strip()
        if token:
            headers["authorization"] = f"Bearer {token}"
        request_timeout = float(
            os.getenv("KOGWISTAR_RUST_RUNTIME_REQUEST_TIMEOUT_SECONDS", "30")
        )
        self.client = client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=request_timeout,
        )
        journal_root = Path(
            os.getenv("KOGWISTAR_RUST_RUNTIME_JOURNAL_DIR", "").strip()
            or cache_dir
            or Path(tempfile.gettempdir()) / "kogwistar-runtime-worker"
        )
        journal_root.mkdir(parents=True, exist_ok=True)
        worker_id = f"python-async-facade-{os.getpid()}-{id(runtime):x}"
        self.worker = AsyncRustRuntimeWorker.from_step_resolver(
            base_url=self.base_url,
            worker_id=worker_id,
            journal_path=journal_root / "async-results.sqlite3",
            step_resolver=runtime.step_resolver,
            predicate_registry=runtime.predicate_registry,
            dependency_provider=self._dependencies,
            cache_dir=cache_dir,
            headers=headers,
            client=self.client,
        )
        self._live_dependencies: Mapping[str, Any] = {}

    def _dependencies(self, _work: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._live_dependencies

    async def aclose(self) -> None:
        await self.worker.aclose()
        if self._owns_client:
            await self.client.aclose()

    async def _request_json(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        try:
            response = await self.client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RustRuntimeAuthorityError(str(error)) from error
        value = response.json()
        if not isinstance(value, dict):
            raise RustRuntimeAuthorityError(f"{path} returned non-object JSON")
        return value

    async def _submit_run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        last_error: httpx.TransportError | None = None
        for _attempt in range(2):
            try:
                response = await self.client.post(
                    "/api/workflow/runs", json=dict(payload)
                )
                response.raise_for_status()
            except httpx.TransportError as error:
                last_error = error
                continue
            except httpx.HTTPError as error:
                raise RustRuntimeAuthorityError(str(error)) from error
            value = response.json()
            if not isinstance(value, dict):
                raise RustRuntimeAuthorityError(
                    "/api/workflow/runs returned non-object JSON"
                )
            return value
        assert last_error is not None
        raise RustRuntimeAuthorityError(str(last_error)) from last_error

    async def _final_state(
        self,
        *,
        run_id: str,
        durable_initial: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = await self._request_json(
            "GET", f"/api/workflow/runs/{run_id}/checkpoints"
        )
        raw = value.get("checkpoints")
        checkpoints = raw if isinstance(raw, list) else []
        candidates = [item for item in checkpoints if isinstance(item, dict)]
        latest = max(
            candidates,
            key=lambda item: (
                int(item.get("step_seq", -1)),
                int(item.get("event_seq", -1)),
            ),
            default=None,
        )
        state = dict(durable_initial)
        if latest is not None:
            latest_state = latest.get("state")
            if not isinstance(latest_state, dict):
                raise RustRuntimeAuthorityError("Rust checkpoint state is not an object")
            state = dict(latest_state)
        state.pop("_rt_routes", None)
        state.pop("_rt_node_ops", None)
        if dependencies:
            state["_deps"] = dependencies
        return state

    async def _pump_existing_run(
        self,
        *,
        run_id: str,
        durable_initial: Mapping[str, Any],
        dependencies: Mapping[str, Any],
    ) -> "RunResult":
        from .runtime import RunResult

        deadline = time.monotonic() + float(
            os.getenv("KOGWISTAR_RUST_RUNTIME_RUN_TIMEOUT_SECONDS", "300")
        )
        cancelled = False
        status_value: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status_value = await self._request_json("GET", f"/api/workflow/runs/{run_id}")
            status = str(status_value.get("status") or "")
            if status in {"succeeded", "failed", "cancelled", "suspended"}:
                break
            if (
                not cancelled
                and self.runtime.cancel_requested is not None
                and self.runtime.cancel_requested(run_id)
            ):
                await self._request_json("POST", f"/api/workflow/runs/{run_id}/cancel")
                cancelled = True
            processed = await self.worker.poll_once(
                limit=max(1, int(self.runtime.max_workers)),
                lease_seconds=60,
                run_id=run_id,
            )
            if processed == 0:
                await asyncio.sleep(0.01)
        else:
            raise RustRuntimeAuthorityError(
                f"Rust runtime run {run_id!r} exceeded configured timeout"
            )

        final_state = await self._final_state(
            run_id=run_id,
            durable_initial=durable_initial,
            dependencies=dependencies,
        )
        server_status = str(status_value.get("status") or "failed")
        status = "failure" if server_status == "failed" else server_status
        raw_error = status_value.get("error")
        if isinstance(raw_error, list):
            errors = [str(value) for value in raw_error]
        elif raw_error in (None, {}, ""):
            errors = []
        else:
            errors = [str(raw_error)]
        return RunResult(
            run_id=run_id,
            final_state=final_state,
            mq=queue.Queue(maxsize=10_000),
            status=status,
            errors=errors,
        )

    async def run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        turn_node_id: str | None,
        initial_state: Mapping[str, Any],
        run_id: str,
    ) -> "RunResult":
        if initial_state.get("dream_deps"):
            raise RustRuntimeAuthorityError(
                "dream_deps is not represented by Rust worker contract v1"
            )
        dependencies = initial_state.get("_deps") or {}
        if not isinstance(dependencies, Mapping):
            raise RustRuntimeAuthorityError("initial_state['_deps'] must be a mapping")
        self._live_dependencies = dependencies
        durable_initial = _json_state(initial_state)
        plan = freeze_runtime_plan(self.runtime, workflow_id)
        payload = {
            "run_id": run_id,
            "workflow_id": workflow_id,
            "conversation_id": conversation_id,
            "turn_node_id": turn_node_id,
            "initial_state": durable_initial,
            "runtime_kind": "async",
            **plan,
        }
        submitted = await self._submit_run(payload)
        if submitted.get("run_id") != run_id or submitted.get("admission") != "accepted":
            raise RustRuntimeAuthorityError(
                f"Rust scheduler returned invalid admission: {submitted!r}"
            )
        return await self._pump_existing_run(
            run_id=run_id,
            durable_initial=durable_initial,
            dependencies=dependencies,
        )


async def run_with_rust_authority_async(
    runtime: Any,
    *,
    workflow_id: str,
    conversation_id: str,
    turn_node_id: str | None,
    initial_state: Mapping[str, Any],
    run_id: str,
    cache_dir: str | os.PathLike[str] | None,
) -> "RunResult":
    url = rust_runtime_authority_url()
    if url is None:
        raise RustRuntimeAuthorityError("Rust runtime authority URL is not configured")
    authority = AsyncRustRuntimeAuthority(
        runtime=runtime,
        base_url=url,
        cache_dir=cache_dir,
    )
    try:
        return await authority.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=initial_state,
            run_id=run_id,
        )
    finally:
        await authority.aclose()


__all__ = [
    "AsyncRustRuntimeAuthority",
    "RustRuntimeAuthority",
    "RustRuntimeAuthorityError",
    "freeze_runtime_plan",
    "run_with_rust_authority",
    "run_with_rust_authority_async",
    "rust_runtime_authority_selected",
    "rust_runtime_authority_url",
]
