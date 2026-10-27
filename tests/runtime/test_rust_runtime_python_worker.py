from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from kogwistar.runtime.models import RunFailure, RunSuccess, RunSuspended
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.rust_runtime_authority import (
    RustRuntimeAuthorityError,
    _json_state,
)
from kogwistar.runtime.rust_worker import (
    AmbiguousWorkerExecution,
    AsyncRustRuntimeWorker,
    AsyncRustStepResolverAdapter,
    RustRuntimeWorker,
    RustStepResolverAdapter,
    RustWorkerError,
    WorkerResultJournal,
)
from kogwistar.implementation_selection import (
    capture_implementation_selection,
    transaction_authority_scope,
)
from kogwistar._rust_bridge import runtime_implementation_mode

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


def _work(*, owner: str = "worker-1") -> dict[str, Any]:
    return {
        "message_id": "lane-1",
        "claimed_by": owner,
        "run_id": "run-1",
        "step_id": "start",
        "correlation_id": "run-1",
        "expected_event_seq": 2,
        "payload": {
            "workflow_id": "wf-1",
            "conversation_id": "conv-1",
            "op": "start",
            "step_seq": 0,
            "token_id": "run-1",
            "parent_token_id": None,
            "state": {"seed": 1},
        },
    }


def _effect() -> dict[str, Any]:
    return {
        "state_update": [["u", {"answer": "worker"}]],
        "successors": [],
        "result": {"answer": "worker"},
    }


def test_python_worker_noop_matches_shared_native_effect_golden() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "runtime-native-noop-v1.json"
        ).read_text(encoding="utf-8")
    )
    resolver = MappingStepResolver()

    @resolver.register("noop")
    def execute(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    effect = RustStepResolverAdapter(resolver)(
        _adapter_work(
            state={"seed": 1},
            op="noop",
            routes=[_route("finish")],
        )
    )
    assert {
        "contract_version": 1,
        "status": effect["status"],
        "state_update": effect["state_update"],
        "update": effect.get("update"),
        "state_schema": effect["state_schema"],
        "successors": effect["successors"],
        "route_next": effect["route_next"],
        "result": effect["result"],
    } == fixture


def test_python_worker_increment_counter_matches_shared_native_effect_golden() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "runtime-native-increment-counter-v1.json"
        ).read_text(encoding="utf-8")
    )
    resolver = MappingStepResolver()

    @resolver.register("increment-counter")
    def execute(ctx):
        counter = int(ctx.state_view["counter"])
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"counter": counter + 1})],
        )

    effect = RustStepResolverAdapter(resolver)(
        _adapter_work(state={"counter": 1}, op="increment-counter")
    )
    assert {
        "contract_version": 1,
        "status": effect["status"],
        "state_update": effect["state_update"],
        "update": effect.get("update"),
        "state_schema": effect["state_schema"],
        "successors": effect["successors"],
        "route_next": effect["route_next"],
        "result": effect["result"],
    } == fixture


def test_python_worker_application_operation_matches_shared_native_effect_golden() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "runtime-native-app-append-v1.json"
        ).read_text(encoding="utf-8")
    )
    resolver = MappingStepResolver()

    @resolver.register("app/append-item", version="v1")
    def append_item(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"items": "appended"})],
        )

    effect = RustStepResolverAdapter(resolver)(
        _adapter_work(state=fixture["initial_state"], op="app/append-item")
    )
    assert {
        "status": effect["status"],
        "state_update": effect["state_update"],
        "state_schema": effect["state_schema"],
        "successors": effect["successors"],
        "route_next": effect["route_next"],
        "result": effect["result"],
    } == fixture["effect"]


def test_python_worker_journals_declarative_graph_batch_only_when_server_advertises_it() -> None:
    resolver = MappingStepResolver()
    batch = {
        "contract_version": 1,
        "atomic": True,
        "mutations": [
            {
                "scope": {"namespace": "ns"},
                "table": "gke_nodes",
                "entity_kind": "node",
                "event_id": "effect-1",
                "op": "ADD",
                "payload": {"id": "node-1", "text": "evidence"},
                "record": {"id": "node-1", "document": "evidence"},
                "embedding_dim": 3,
            }
        ],
        "index_jobs": [
            {
                "job_id": "index-1",
                "namespace": "ns",
                "entity_kind": "node",
                "entity_id": "node-1",
                "index_kind": "node_docs",
                "op": "UPSERT",
            }
        ],
    }

    @resolver.register("start")
    def execute(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[],
            graph_mutation_batch=batch,
        )

    work = _adapter_work()
    work["worker_effect_capabilities"] = ["graph_mutation_batch_v1"]
    effect = RustStepResolverAdapter(resolver)(work)
    assert effect["graph_mutation_batch"]["contract_version"] == 1
    assert effect["graph_mutation_batch"]["mutations"][0]["event_id"] == "effect-1"
    assert effect["graph_mutation_batch"]["index_jobs"][0]["job_id"] == "index-1"
    envelope = RustRuntimeWorker._result_envelope(work, effect)
    assert envelope["effect"]["graph_mutation_batch"] == effect["graph_mutation_batch"]

    unsupported_work = _adapter_work()
    with pytest.raises(RustWorkerError, match="does not advertise graph_mutation_batch_v1"):
        RustRuntimeWorker._result_envelope(unsupported_work, effect)


def test_python_worker_omitted_operation_version_preserves_v1_effect() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "runtime-operation-legacy-version-v1.json"
        ).read_text(encoding="utf-8")
    )
    resolver = MappingStepResolver()

    @resolver.register(fixture["operation"])
    def legacy_label(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"implementation": "legacy-v1"})],
        )

    work = _adapter_work(
        op=fixture["operation"],
        state=fixture["initial_state"],
    )
    del work["payload"]["op_version"]
    effect = RustStepResolverAdapter(resolver)(work)
    assert {
        "status": effect["status"],
        "state_update": effect["state_update"],
        "state_schema": effect["state_schema"],
        "successors": effect["successors"],
        "route_next": effect["route_next"],
        "result": effect["result"],
    } == fixture["effect"]


def _adapter_work(
    *,
    state: dict[str, Any] | None = None,
    routes: list[dict[str, Any]] | None = None,
    op: str = "start",
    op_version: str = "v1",
    op_environment_id: str | None = None,
) -> dict[str, Any]:
    work = _work()
    work["payload"].update(
        {
            "node_id": "start",
            "turn_node_id": "turn-1",
            "state": dict(state or {"seed": 1}),
            "runtime_routes": list(routes or []),
            "op": op,
            "op_version": op_version,
            **(
                {"op_environment_id": op_environment_id}
                if op_environment_id is not None
                else {}
            ),
        }
    )
    return work


def _route(
    target: str,
    *,
    predicate: str | None = None,
    aliases: list[str] | None = None,
    is_default: bool = False,
    source_fanout: bool = False,
    join_mask: int = 0,
) -> dict[str, Any]:
    return {
        "edge_id": f"start->{target}",
        "source_node_id": "start",
        "target_node_id": target,
        "aliases": list(aliases or [target]),
        "predicate": predicate,
        "priority": 100,
        "is_default": is_default,
        "multiplicity": "one",
        "source_fanout": source_fanout,
        "join_mask": join_mask,
    }


def test_step_resolver_adapter_maps_context_mutation_update_and_explicit_route() -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(ctx):
        assert (ctx.run_id, ctx.workflow_id, ctx.workflow_node_id, ctx.op) == (
            "run-1",
            "wf-1",
            "start",
            "start",
        )
        assert ctx.turn_node_id == "turn-1"
        assert ctx.idempotency_key == "lane-1"
        assert ctx.state_view["_deps"]["injected"] == 7
        with ctx.state_write as state:
            state["direct"] = "kept"
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("a", {"items": "result"})],
            _route_next=["go"],
        )

    effect = RustStepResolverAdapter(
        resolver,
        dependency_provider=lambda _work: {"injected": 7},
    )(
        _adapter_work(
            state={"seed": 1, "items": []},
            routes=[_route("next", aliases=["go", "next"])],
        )
    )

    assert effect["status"] == "success"
    assert effect["state_update"] == [
        ["u", {"direct": "kept"}],
        ["a", {"items": "result"}],
    ]
    assert effect["successors"] == [{"node_id": "next", "join_mask": 0}]
    assert effect["route_next"] == ["go"]
    assert effect["result"] == {
        "workflow_status": "succeeded",
        "final_state": {"seed": 1, "items": ["result"], "direct": "kept"},
    }


def test_sync_rust_worker_binds_runtime_selection_through_callback_and_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.runtime import base_runtime

    observed: list[str] = []
    reducer = base_runtime.apply_state_update_inplace

    def observe_reducer(*args: Any, **kwargs: Any) -> None:
        observed.append(runtime_implementation_mode())
        reducer(*args, **kwargs)

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    selection_snapshot = capture_implementation_selection()
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        observed.append(runtime_implementation_mode())
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"answer": "ok"})],
        )

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    adapter = RustStepResolverAdapter(
        resolver,
        implementation_selection_snapshot=selection_snapshot,
    )
    monkeypatch.setattr(base_runtime, "apply_state_update_inplace", observe_reducer)

    effect = adapter(_adapter_work())

    assert observed == ["python", "python"]
    assert effect["result"]["final_state"] == {"seed": 1, "answer": "ok"}


@pytest.mark.asyncio
async def test_async_rust_worker_binds_runtime_selection_through_callback_and_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar.runtime import base_runtime
    from kogwistar.runtime.resolvers import AsyncMappingStepResolver

    observed: list[str] = []
    reducer = base_runtime.apply_state_update_inplace

    def observe_reducer(*args: Any, **kwargs: Any) -> None:
        observed.append(runtime_implementation_mode())
        reducer(*args, **kwargs)

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "python")
    selection_snapshot = capture_implementation_selection()
    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def execute(_ctx):
        observed.append(runtime_implementation_mode())
        monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"answer": "ok"})],
        )

    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", "rust")
    adapter = AsyncRustStepResolverAdapter(
        resolver,
        implementation_selection_snapshot=selection_snapshot,
    )
    monkeypatch.setattr(base_runtime, "apply_state_update_inplace", observe_reducer)
    work = _adapter_work()
    work["payload"]["worker_protocol"] = "async-v2"

    effect = await adapter.execute(work)

    assert observed == ["python", "python"]
    assert effect["result"]["final_state"] == {"seed": 1, "answer": "ok"}


def test_step_idempotency_key_stays_stable_when_another_worker_reclaims() -> None:
    resolver = MappingStepResolver()
    seen: list[str | None] = []

    @resolver.register("start")
    def execute(ctx):
        seen.append(ctx.idempotency_key)
        return RunSuccess(conversation_node_id=None, state_update=[])

    adapter = RustStepResolverAdapter(resolver)
    for worker_id in ("worker-a", "worker-b"):
        work = _adapter_work()
        work["claimed_by"] = worker_id
        assert adapter(work)["status"] == "success"

    assert seen == ["lane-1", "lane-1"]


def test_python_runtime_dependencies_stay_process_local_and_accept_library_objects() -> None:
    class NumericalService:
        def mean(self, values: list[float]) -> float:
            return sum(values) / len(values)

    service = NumericalService()
    resolver = MappingStepResolver()

    @resolver.register("numerical", version="v2")
    def execute(ctx):
        assert ctx.op_version == "v2"
        injected = ctx.state_view["_deps"]["numerical_service"]
        assert isinstance(injected, NumericalService)
        assert injected is service
        mean = injected.mean([2.0, 4.0, 6.0])
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"mean": mean})],
        )

    work = _adapter_work(op="numerical", op_version="v2")
    effect = RustStepResolverAdapter(
        resolver,
        dependency_provider=lambda _work: {"numerical_service": service},
    )(work)

    assert effect["status"] == "success"
    result = effect.get("result")
    assert isinstance(result, dict)
    assert result.get("final_state") == {"seed": 1, "mean": 4.0}
    assert "_deps" not in effect


def test_numpy_worker_result_must_be_explicitly_projected_to_json() -> None:
    np = pytest.importorskip("numpy")
    resolver = MappingStepResolver()

    @resolver.register("numerical", version="v1", environment_id="numpy-test-v1")
    def execute(ctx):
        values = np.asarray(ctx.state_view["values"], dtype=np.float64)
        mean = np.mean(values)
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"mean": mean.item()})],
        )

    effect = RustStepResolverAdapter(
        resolver,
        dependency_provider=lambda _work: {"numpy": np},
    )(
        _adapter_work(
            state={"values": [2.0, 4.0, 6.0]},
            op="numerical",
            op_version="v1",
            op_environment_id="numpy-test-v1",
        )
    )
    assert effect["state_update"] == [["u", {"mean": 4.0}]]
    assert effect["result"]["final_state"] == {"values": [2.0, 4.0, 6.0], "mean": 4.0}
    assert "_deps" not in effect

    array_resolver = MappingStepResolver()

    @array_resolver.register("numerical", version="v1", environment_id="numpy-test-v1")
    def return_array(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"vector": np.asarray([1.0, 2.0])})],
        )

    with pytest.raises(RustWorkerError, match="must be JSON-only"):
        RustStepResolverAdapter(array_resolver)(_adapter_work(
            op="numerical",
            op_version="v1",
            op_environment_id="numpy-test-v1",
        ))


def test_python_worker_rejects_operation_version_mismatch_before_execution() -> None:
    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "runtime-operation-version-admission-v1.json"
        ).read_text(encoding="utf-8")
    )
    resolver = MappingStepResolver()
    calls: list[str] = []

    @resolver.register(fixture["operation"], version=fixture["registered_version"])
    def execute(_ctx):
        calls.append("executed")
        return RunSuccess(conversation_node_id=None, state_update=[])

    with pytest.raises(
        RustWorkerError,
        match=f"registers '{fixture['registered_version']}'",
    ):
        RustStepResolverAdapter(resolver)(
            _adapter_work(
                op=fixture["operation"],
                op_version=fixture["requested_version"],
            )
        )
    assert len(calls) == fixture["expected"]["handler_invocations"]


@pytest.mark.parametrize(
    "identity_field",
    ["message_id", "run_id", "step_id", "claimed_by", "correlation_id"],
)
def test_worker_rejects_missing_correlation_identity_before_callback(
    tmp_path: Path, identity_field: str
) -> None:
    resolver = MappingStepResolver()
    calls: list[str] = []

    @resolver.register("start")
    def execute(_ctx):
        calls.append("executed")
        return RunSuccess(conversation_node_id=None, state_update=[])

    work = _adapter_work()
    del work[identity_field]
    journal_path = tmp_path / f"missing-{identity_field}.sqlite"
    worker = RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=journal_path,
        execute=RustStepResolverAdapter(resolver),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("invalid work must not be submitted")
            )
        ),
    )

    expected_error = "missing message_id" if identity_field == "message_id" else "missing identity fields"
    with worker, pytest.raises(RustWorkerError, match=expected_error):
        worker.process(work)

    assert calls == []
    with sqlite3.connect(journal_path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id = ?",
            ("lane-1",),
        ).fetchone() is None


def test_async_worker_preflight_rejects_missing_correlation_identity() -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        pytest.fail("invalid work must be rejected before callback execution")

    work = _adapter_work()
    work.pop("message_id")

    with pytest.raises(RustWorkerError, match="missing identity fields"):
        AsyncRustStepResolverAdapter(resolver).validate_work(work)


def test_sync_python_worker_rejects_foreign_executor_target() -> None:
    resolver = MappingStepResolver()
    work = _adapter_work()
    work["payload"]["executor_target"] = "rust-native"

    with pytest.raises(RustWorkerError, match="cannot execute payload"):
        RustStepResolverAdapter(resolver).validate_work(work)


def test_worker_rejects_foreign_executor_before_callback_or_journal(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        calls.append("executed")
        return RunSuccess(conversation_node_id=None, state_update=[])

    work = _adapter_work()
    work["payload"]["executor_target"] = "rust-native"
    journal_path = tmp_path / "foreign-executor.sqlite"
    worker = RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=journal_path,
        execute=RustStepResolverAdapter(resolver),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("foreign work must not be submitted")
            )
        ),
    )

    with worker, pytest.raises(RustWorkerError, match="cannot execute payload"):
        worker.process(work)

    assert calls == []
    with sqlite3.connect(journal_path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id = ?",
            ("lane-1",),
        ).fetchone() is None


def test_async_python_worker_rejects_foreign_executor_target() -> None:
    resolver = MappingStepResolver()
    work = _adapter_work()
    work["payload"]["worker_protocol"] = "async-v2"
    work["payload"]["executor_target"] = "rust-native"

    with pytest.raises(RustWorkerError, match="cannot execute payload"):
        AsyncRustStepResolverAdapter(resolver).validate_work(work)


def test_sync_python_worker_rejects_native_operation_without_target() -> None:
    resolver = MappingStepResolver()
    work = _adapter_work(op="rust-native/noop")

    with pytest.raises(RustWorkerError, match="rust-native operation"):
        RustStepResolverAdapter(resolver).validate_work(work)


def test_async_python_worker_rejects_native_operation_without_target() -> None:
    resolver = MappingStepResolver()
    work = _adapter_work(op="rust-native/noop")
    work["payload"]["worker_protocol"] = "async-v2"

    with pytest.raises(RustWorkerError, match="rust-native operation"):
        AsyncRustStepResolverAdapter(resolver).validate_work(work)


def test_worker_process_validates_identity_for_generic_executor(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def execute(_work):
        calls.append("executed")
        return _effect()

    work = _work()
    work.pop("correlation_id")
    journal_path = tmp_path / "generic-executor-invalid-identity.sqlite"
    worker = RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=journal_path,
        execute=execute,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("invalid work must not be submitted")
            )
        ),
    )

    with worker, pytest.raises(RustWorkerError, match="missing identity fields"):
        worker.process(work)

    assert calls == []
    with sqlite3.connect(journal_path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id = ?",
            ("lane-1",),
        ).fetchone() is None


@pytest.mark.asyncio
async def test_async_worker_process_validates_identity_for_generic_executor(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def execute(_work):
        calls.append("executed")
        return _effect()

    work = _work()
    work.pop("correlation_id")
    journal_path = tmp_path / "async-generic-executor-invalid-identity.sqlite"

    async def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid work must not be submitted")

    worker = AsyncRustRuntimeWorker(
        base_url="http://rust",
        worker_id="async-worker-1",
        journal_path=journal_path,
        execute=execute,
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_request),
            base_url="http://rust",
        ),
    )

    try:
        with pytest.raises(RustWorkerError, match="missing identity fields"):
            await worker.process(work)
    finally:
        await worker.aclose()

    assert calls == []
    with sqlite3.connect(journal_path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id = ?",
            ("lane-1",),
        ).fetchone() is None


def test_python_worker_checks_environment_before_numerical_callback() -> None:
    resolver = MappingStepResolver()
    calls: list[str] = []

    @resolver.register("numerical", version="v2", environment_id="numpy-cpu-2")
    def execute(ctx):
        calls.append(ctx.op_environment_id or "")
        return RunSuccess(conversation_node_id=None, state_update=[])

    adapter = RustStepResolverAdapter(resolver)
    accepted = adapter(
        _adapter_work(
            op="numerical", op_version="v2", op_environment_id="numpy-cpu-2"
        )
    )
    assert accepted["status"] == "success"
    assert calls == ["numpy-cpu-2"]
    with pytest.raises(RustWorkerError, match="registers 'numpy-cpu-2'"):
        adapter(
            _adapter_work(
                op="numerical", op_version="v2", op_environment_id="numpy-cpu-1"
            )
        )
    assert calls == ["numpy-cpu-2"]

    with pytest.raises(RustWorkerError, match="requires environment None"):
        adapter(_adapter_work(op="numerical", op_version="v2"))
    assert calls == ["numpy-cpu-2"]


def test_python_worker_dispatches_legacy_and_versioned_handlers_in_one_resolver() -> None:
    resolver = MappingStepResolver()

    @resolver.register("numerical", version="v2", environment_id="numpy-cpu-v2")
    def current(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"implementation": f"new-{ctx.op_version}"})],
        )

    @resolver.register_versioned(
        "numerical", version="v1", environment_id="numpy-cpu-v1"
    )
    def legacy(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"implementation": f"old-{ctx.op_version}"})],
        )

    adapter = RustStepResolverAdapter(resolver)
    legacy_effect = adapter(
        _adapter_work(
            op="numerical", op_version="v1", op_environment_id="numpy-cpu-v1"
        )
    )
    current_effect = adapter(
        _adapter_work(
            op="numerical", op_version="v2", op_environment_id="numpy-cpu-v2"
        )
    )

    assert legacy_effect["result"]["final_state"]["implementation"] == "old-v1"
    assert current_effect["result"]["final_state"]["implementation"] == "new-v2"
    assert resolver.op_versions == {"numerical": "v2"}
    assert resolver.supports_version("numerical", "v1")
    assert resolver.supports_version("numerical", "v2")
    assert len(adapter.claim_recipient_ids) == 3
    assert adapter.claim_recipient_ids[-1] == "python-worker"
    assert all(
        recipient.startswith("python-worker-env-")
        for recipient in adapter.claim_recipient_ids[:-1]
    )


@pytest.mark.asyncio
async def test_async_python_worker_dispatches_multiple_versions() -> None:
    from kogwistar.runtime.resolvers import AsyncMappingStepResolver

    resolver = AsyncMappingStepResolver()

    @resolver.register("summary", version="v2")
    async def current(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"implementation": f"new-{ctx.op_version}"})],
        )

    @resolver.register_versioned("summary", version="v1")
    async def legacy(ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"implementation": f"old-{ctx.op_version}"})],
        )

    adapter = AsyncRustStepResolverAdapter(resolver)
    legacy_work = _adapter_work(op="summary", op_version="v1")
    current_work = _adapter_work(op="summary", op_version="v2")
    legacy_work["payload"]["worker_protocol"] = "async-v2"
    current_work["payload"]["worker_protocol"] = "async-v2"
    legacy_effect = await adapter.execute(legacy_work)
    current_effect = await adapter.execute(current_work)

    assert legacy_effect["result"]["final_state"]["implementation"] == "old-v1"
    assert current_effect["result"]["final_state"]["implementation"] == "new-v2"


@pytest.mark.parametrize("environment_id", ["", " numpy-cpu-2", "numpy-cpu-2 "])
def test_resolver_rejects_noncanonical_environment_ids(environment_id: str) -> None:
    resolver = MappingStepResolver()
    with pytest.raises(ValueError, match="canonical"):
        resolver.register("numerical", environment_id=environment_id)


def test_worker_environment_mismatch_does_not_poison_execution_journal(
    tmp_path: Path,
) -> None:
    resolver = MappingStepResolver()

    @resolver.register("numerical", version="v2", environment_id="numpy-cpu-2")
    def execute(_ctx):
        pytest.fail("incompatible worker must not invoke the callback")

    journal_path = tmp_path / "environment-mismatch.sqlite"
    worker = RustRuntimeWorker(
        base_url="http://rust",
        worker_id="wrong-env-worker",
        journal_path=journal_path,
        execute=RustStepResolverAdapter(resolver),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("preflight must fail before result IO")
            )
        ),
    )
    with worker, pytest.raises(RustWorkerError, match="registers 'numpy-cpu-2'"):
        worker.process(
            _adapter_work(
                op="numerical", op_version="v2", op_environment_id="numpy-cpu-1"
            )
        )

    with sqlite3.connect(journal_path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id='lane-1'"
        ).fetchone() is None


def test_step_resolver_adapter_maps_suspend() -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        return RunSuspended(
            conversation_node_id=None,
            state_update=[("u", {"parked": True})],
            wait_reason="approval",
            resume_payload={"request_id": "r1"},
        )

    effect = RustStepResolverAdapter(resolver)(_adapter_work())
    assert effect["status"] == "suspended"
    assert effect["successors"] == []
    assert effect["wait_reason"] == "approval"
    assert effect["resume_payload"] == {"request_id": "r1"}
    assert effect["result"]["workflow_status"] == "suspended"


@pytest.mark.parametrize("handled", [False, True], ids=["unhandled", "handled"])
def test_step_resolver_adapter_maps_failure_routing(handled: bool) -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        return RunFailure(
            conversation_node_id=None,
            state_update=[("u", {"attempted": True})],
            errors=["boom"],
        )

    route = _route("recover", predicate="on_failure", join_mask=4)
    adapter = RustStepResolverAdapter(
        resolver,
        predicate_registry={"on_failure": lambda _edge, _state, _result: handled},
    )
    effect = adapter(_adapter_work(routes=[route]))
    assert effect["status"] == "failed"
    assert effect["errors"] == ["boom"]
    assert effect["result"]["workflow_status"] == "failed"
    assert effect["successors"] == (
        [{"node_id": "recover", "join_mask": 4}] if handled else []
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: state.pop("seed"), "direct resolver state deletion"),
        (lambda state: state.__setitem__("_rt_join", {"pending": []}), "scheduler state key"),
    ],
)
def test_step_resolver_adapter_rejects_state_semantic_drift(mutate, message: str) -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(ctx):
        with ctx.state_write as state:
            mutate(state)
        return RunSuccess(conversation_node_id=None, state_update=[])

    with pytest.raises(RustWorkerError, match=message):
        RustStepResolverAdapter(resolver)(
            _adapter_work(state={"seed": 1, "_rt_join": {"pending": ["original"]}})
        )


@pytest.mark.parametrize("key", ["_deps", "dream_deps", "_rt_join"])
def test_step_resolver_adapter_rejects_result_runtime_plumbing(key: str) -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {key: {"must": "not persist"}})],
        )

    with pytest.raises(RustWorkerError, match="runtime plumbing keys"):
        RustStepResolverAdapter(resolver)(_adapter_work())


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_python_worker_rejects_non_finite_numbers_at_rust_json_seam(
    non_finite: float,
) -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"numeric_result": non_finite})],
        )

    with pytest.raises(RustWorkerError, match="must be JSON-only"):
        RustStepResolverAdapter(resolver)(_adapter_work())


def test_python_worker_rejects_non_string_json_keys_without_coercion() -> None:
    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        return RunSuccess(
            conversation_node_id=None,
            state_update=[("u", {"numeric_result": {1: "must not become string key"}})],
        )

    with pytest.raises(RustWorkerError, match="non-string object key"):
        RustStepResolverAdapter(resolver)(_adapter_work())


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_rust_authority_rejects_non_finite_initial_state(non_finite: float) -> None:
    with pytest.raises(RustRuntimeAuthorityError, match="durable state must be JSON-only"):
        _json_state({"numeric_value": non_finite})


def test_rust_authority_rejects_non_string_keys_without_coercion() -> None:
    with pytest.raises(RustRuntimeAuthorityError, match="non-string object key"):
        _json_state({"nested": {1: "would be coerced by json.dumps"}})


def test_rust_authority_preserves_legacy_tuple_to_json_array_conversion() -> None:
    assert _json_state({"items": (1, "two")}) == {"items": [1, "two"]}


def test_step_resolver_adapter_rejects_unrepresented_callbacks() -> None:
    resolver = MappingStepResolver()

    @resolver.register("nested", is_nested=True)
    def nested(_ctx):
        pytest.fail("nested callback must not execute")

    with pytest.raises(RustWorkerError, match="nested workflow"):
        RustStepResolverAdapter(resolver)(_adapter_work(op="nested"))

    @resolver.register("publish")
    def publish(ctx):
        ctx.publish({"type": "not-durable"})
        return RunSuccess(conversation_node_id=None, state_update=[])

    with pytest.raises(RustWorkerError, match="ctx.publish"):
        RustStepResolverAdapter(resolver)(_adapter_work(op="publish"))

    @resolver.register("lane")
    def lane(ctx):
        ctx.send_lane_message(msg_type="unsupported")
        return RunSuccess(conversation_node_id=None, state_update=[])

    with pytest.raises(RustWorkerError, match="ctx.send_lane_message"):
        RustStepResolverAdapter(resolver)(_adapter_work(op="lane"))


def test_step_resolver_adapter_rejects_missing_op_and_predicate() -> None:
    resolver = MappingStepResolver()
    with pytest.raises(RustWorkerError, match="cannot resolve frozen op 'missing'"):
        RustStepResolverAdapter(resolver)(_adapter_work(op="missing"))

    @resolver.register("start")
    def execute(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    with pytest.raises(RustWorkerError, match="unregistered predicates"):
        RustStepResolverAdapter(resolver)(
            _adapter_work(routes=[_route("next", predicate="unknown")])
        )


def test_worker_journals_callback_before_send_and_replays_after_restart(
    tmp_path: Path,
) -> None:
    calls = {"callback": 0, "result": 0}
    accepted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/internal/runtime/results"
        body = json.loads(request.content)
        calls["result"] += 1
        if calls["result"] == 1:
            raise httpx.ReadError("response lost", request=request)
        accepted.append(body)
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    def execute(_work: dict[str, Any]) -> dict[str, Any]:
        calls["callback"] += 1
        return _effect()

    journal = tmp_path / "worker.sqlite"
    original_work = _work()
    original_work["payload"]["op_environment_id"] = "numpy-cpu-v2"
    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=journal,
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match="response lost"):
            worker.process(original_work)

    replayed_work = _work(owner="worker-2")
    replayed_work["payload"]["op_environment_id"] = "numpy-cpu-v2"
    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-2",
        journal_path=journal,
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as restarted:
        assert restarted.process(replayed_work)["event_seq"] == 3

    assert calls == {"callback": 1, "result": 2}
    assert accepted[0]["handoff"]["claimed_by"] == "worker-2"
    assert "run_id" not in accepted[0]["effect"]
    assert "frontier" not in accepted[0]["effect"]
    assert accepted[0]["effect"]["state_update"] == [
        ["u", {"answer": "worker"}]
    ]
    prepared_trace, submitted_trace = accepted[0]["effect"]["trace_events"]
    assert prepared_trace == {
        "schema_version": 1,
        "event": "runtime.python_worker.result_prepared",
        "executor_target": "python-worker",
        "worker_id": "worker-1",
        "identity_source": "worker_configuration",
        "lease_claim_id": "worker-1",
        "message_id": "lane-1",
        "run_id": "run-1",
        "step_id": "start",
        "correlation_id": "run-1",
        "operation": "start",
        "op_version": "v1",
        "required_environment_id": "numpy-cpu-v2",
    }
    assert submitted_trace["event"] == "runtime.python_worker.result_submitted"
    assert submitted_trace["worker_id"] == "worker-2"
    assert submitted_trace["lease_claim_id"] == "worker-2"


def test_step_resolver_worker_rejects_callback_inside_transaction(
    tmp_path: Path,
) -> None:
    calls = 0
    accepted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        accepted.append(json.loads(request.content))
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    resolver = MappingStepResolver()

    @resolver.register("start")
    def execute(_ctx):
        nonlocal calls
        calls += 1
        return RunSuccess(conversation_node_id=None, state_update=[])

    worker = RustRuntimeWorker.from_step_resolver(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "transaction-guard.sqlite",
        step_resolver=resolver,
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://rust"
        ),
    )
    with worker:
        with transaction_authority_scope("rust"):
            worker.process(_work())

    assert calls == 0
    assert accepted[0]["effect"]["status"] == "failed"
    assert "callback execution is not allowed" in accepted[0]["effect"]["errors"][0]


def test_executing_journal_row_fails_closed_after_crash(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.sqlite"
    journal = WorkerResultJournal(path)
    work = _work()
    journal.begin(message_id="lane-1", work_digest="digest")
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT status FROM rust_worker_results WHERE message_id='lane-1'"
        ).fetchone() == ("executing",)

    with pytest.raises(AmbiguousWorkerExecution, match="may already have run"):
        journal.begin(message_id="lane-1", work_digest="digest")


def test_worker_journal_releases_database_file_handles(tmp_path: Path) -> None:
    path = tmp_path / "released.sqlite"
    journal = WorkerResultJournal(path)
    journal.begin(message_id="lane-1", work_digest="digest")
    path.unlink()
    assert not path.exists()


def test_worker_claim_and_scheduler_identity_are_validated(tmp_path: Path) -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("claim"):
            return httpx.Response(200, json={"work": [_work()]}, request=request)
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "worker.sqlite",
        execute=lambda _item: _effect(),
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        assert worker.poll_once(lease_seconds=45, run_id="run-1") == 1

    assert requests[0] == (
        "/internal/runtime/claim",
        {
            "claimed_by": "worker-1",
            "limit": 1,
            "lease_seconds": 45,
            "run_id": "run-1",
        },
    )
    assert requests[1][0] == "/internal/runtime/results"
    assert requests[1][1]["effect"]["successors"] == []

    bad_effect = _effect()
    bad_effect["run_id"] = "forged"
    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "forged.sqlite",
        execute=lambda _item: bad_effect,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match=r"may not override scheduler fields: \['run_id'\]"):
            worker.process(_work())


def test_python_worker_matches_shared_rust_runtime_wire_fixture(
    tmp_path: Path,
) -> None:
    from kogwistar.id_provider import stable_id

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "adr015-runtime-wire-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"work": []}, request=request)

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "golden-worker.sqlite",
        execute=lambda _work: _effect(),
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://rust"
        ),
    ) as worker:
        assert worker.claim(limit=2, run_id="run-1") == []

    assert requests == [fixture["claim"]]
    expected = copy.deepcopy(fixture["result_effect"])
    expected["effect"]["effect_id"] = str(
        stable_id("runtime.worker.result", "lane-1", "run-1", "entry")
    )
    assert RustRuntimeWorker._result_envelope(
        fixture["claimed_work_sqlite"], expected["effect"]
    ) == expected


def test_environment_pinned_resolver_advertises_compatible_claim_recipients(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"work": []}, request=request)

    resolver = MappingStepResolver()

    @resolver.register("numerical", environment_id="numpy-cpu-2")
    def execute(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    worker = RustRuntimeWorker.from_step_resolver(
        base_url="http://rust",
        worker_id="numpy-worker",
        journal_path=tmp_path / "numpy-worker.sqlite",
        step_resolver=resolver,
        client=httpx.Client(
            transport=httpx.MockTransport(handler), base_url="http://rust"
        ),
    )
    with worker:
        assert worker.claim() == []

    assert requests == [
        {
            "claimed_by": "numpy-worker",
            "limit": 1,
            "lease_seconds": 60,
            "recipient_ids": [
                    "python-worker-env-"
                    + base64.urlsafe_b64encode(
                        hashlib.sha256(b"numpy-cpu-2").digest()
                    )
                    .decode()
                    .rstrip("="),
                    "python-worker",
            ],
        }
    ]


def test_async_environment_pinned_resolver_advertises_compatible_claim_recipients(
    tmp_path: Path,
) -> None:
    from kogwistar.runtime.resolvers import AsyncMappingStepResolver

    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"work": []}, request=request)

    resolver = AsyncMappingStepResolver()

    @resolver.register("numerical", environment_id="numpy-cpu-2")
    async def execute(_ctx):
        return RunSuccess(conversation_node_id=None, state_update=[])

    async def invoke() -> None:
        worker = AsyncRustRuntimeWorker.from_step_resolver(
            base_url="http://rust",
            worker_id="numpy-async-worker",
            journal_path=tmp_path / "numpy-async-worker.sqlite",
            step_resolver=resolver,
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://rust"
            ),
        )
        try:
            assert await worker.claim() == []
        finally:
            await worker.aclose()

    asyncio.run(invoke())
    assert requests[0]["recipient_ids"] == [
        "python-worker-env-"
        + base64.urlsafe_b64encode(hashlib.sha256(b"numpy-cpu-2").digest())
        .decode()
        .rstrip("="),
        "python-worker",
    ]


def test_worker_rejects_claim_without_frozen_workflow_op(tmp_path: Path) -> None:
    work = _work()
    work["payload"].pop("op")
    worker = RustRuntimeWorker(
        base_url="http://test",
        worker_id="worker-1",
        journal_path=tmp_path / "journal.sqlite",
        execute=lambda _work: _effect(),
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("no request expected")
            )
        ),
    )
    with worker, pytest.raises(RustWorkerError, match="lacks frozen workflow op"):
        worker.process(work)


def test_resume_effect_bypasses_suspended_resolver_callback() -> None:
    calls = 0

    def resolve(_op: str):
        def execute(_ctx):
            nonlocal calls
            calls += 1
            raise AssertionError("suspended resolver must not run again")

        return execute

    effect = RustStepResolverAdapter(resolve)(
        _adapter_work()
        | {
            "payload": _adapter_work()["payload"]
            | {
                "resume_effect": {
                    "status": "success",
                    "state_update": [["u", {"approved": True}]],
                    "successors": [],
                    "route_next": [],
                    "result": {"workflow_status": "succeeded"},
                }
            }
        }
    )

    assert calls == 0
    assert effect["state_update"] == [["u", {"approved": True}]]


def test_out_of_order_result_retries_without_reexecuting_callback(tmp_path: Path) -> None:
    calls = {"callback": 0, "result": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["result"] += 1
        if calls["result"] == 1:
            return httpx.Response(
                409,
                json={"detail": "not next in canonical frontier order"},
                request=request,
            )
        return httpx.Response(200, json={"event_seq": 4}, request=request)

    def execute(_work: dict[str, Any]) -> dict[str, Any]:
        calls["callback"] += 1
        return _effect()

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=tmp_path / "ordered-worker.sqlite",
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match="409 Conflict"):
            worker.process(_work())
        assert worker.process(_work())["event_seq"] == 4

    assert calls == {"callback": 1, "result": 2}


def test_async_step_resolver_adapter_awaits_callback_and_rejects_sync_protocol() -> None:
    from kogwistar.runtime.resolvers import AsyncMappingStepResolver

    resolver = AsyncMappingStepResolver()

    @resolver.register("start")
    async def execute(ctx):
        assert ctx.idempotency_key == "lane-1"
        await asyncio.sleep(0)
        with ctx.state_write as state:
            state["async"] = True
        return RunSuccess(conversation_node_id=None, state_update=[])

    work = _adapter_work()
    work["payload"]["worker_protocol"] = "async-v2"
    effect = asyncio.run(AsyncRustStepResolverAdapter(resolver).execute(work))
    assert effect["result"]["final_state"] == {"seed": 1, "async": True}

    with pytest.raises(RustWorkerError, match="cannot execute"):
        RustStepResolverAdapter(resolver)(work)

    work["payload"]["worker_protocol"] = "sync-v1"
    with pytest.raises(RustWorkerError, match="requires worker_protocol"):
        asyncio.run(AsyncRustStepResolverAdapter(resolver).execute(work))


def test_async_worker_journals_awaited_callback_before_response_retry(tmp_path: Path) -> None:
    calls = {"callback": 0, "result": 0}
    accepted: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["result"] += 1
        if calls["result"] == 1:
            raise httpx.ReadError("response lost", request=request)
        accepted.append(json.loads(request.content))
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    async def execute(_work: dict[str, Any]) -> dict[str, Any]:
        calls["callback"] += 1
        await asyncio.sleep(0)
        return _effect()

    async def invoke() -> None:
        journal = tmp_path / "async-worker.sqlite"
        original_work = _work()
        original_work["payload"]["op_environment_id"] = "numpy-cpu-v2"
        first = AsyncRustRuntimeWorker(
            base_url="http://rust",
            worker_id="worker-1",
            journal_path=journal,
            execute=execute,
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://rust"
            ),
        )
        with pytest.raises(RustWorkerError, match="response lost"):
            await first.process(original_work)
        await first.aclose()
        second = AsyncRustRuntimeWorker(
            base_url="http://rust",
            worker_id="worker-2",
            journal_path=journal,
            execute=execute,
            client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://rust"
            ),
        )
        replayed_work = _work(owner="worker-2")
        replayed_work["payload"]["op_environment_id"] = "numpy-cpu-v2"
        assert (await second.process(replayed_work))["event_seq"] == 3
        await second.aclose()

    asyncio.run(invoke())
    assert calls == {"callback": 1, "result": 2}
    prepared_trace, submitted_trace = accepted[0]["effect"]["trace_events"]
    assert prepared_trace["worker_id"] == "worker-1"
    assert prepared_trace["lease_claim_id"] == "worker-1"
    assert prepared_trace["required_environment_id"] == "numpy-cpu-v2"
    assert submitted_trace["event"] == "runtime.python_worker.result_submitted"
    assert submitted_trace["worker_id"] == "worker-2"
    assert submitted_trace["lease_claim_id"] == "worker-2"


def test_async_worker_cancellation_leaves_ambiguous_journal_row(tmp_path: Path) -> None:
    async def cancelled(_work: dict[str, Any]) -> dict[str, Any]:
        raise asyncio.CancelledError()

    async def invoke() -> None:
        journal = tmp_path / "async-cancelled.sqlite"
        no_post = httpx.MockTransport(
            lambda _request: pytest.fail("cancelled callback must not post an effect")
        )
        first = AsyncRustRuntimeWorker(
            base_url="http://rust",
            worker_id="worker-1",
            journal_path=journal,
            execute=cancelled,
            client=httpx.AsyncClient(transport=no_post, base_url="http://rust"),
        )
        with pytest.raises(asyncio.CancelledError):
            await first.process(_work())
        await first.aclose()
        restarted = AsyncRustRuntimeWorker(
            base_url="http://rust",
            worker_id="worker-2",
            journal_path=journal,
            execute=cancelled,
            client=httpx.AsyncClient(transport=no_post, base_url="http://rust"),
        )
        with pytest.raises(AmbiguousWorkerExecution, match="may already have run"):
            await restarted.process(_work(owner="worker-2"))
        await restarted.aclose()

    asyncio.run(invoke())
