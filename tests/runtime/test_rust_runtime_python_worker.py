from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest

from kogwistar.runtime.rust_worker import (
    AmbiguousWorkerExecution,
    AsyncRustRuntimeWorker,
    AsyncRustStepResolverAdapter,
    RustRuntimeWorker,
    RustStepResolverAdapter,
    RustWorkerError,
    WorkerResultJournal,
)
from kogwistar.runtime.models import RunFailure, RunSuccess, RunSuspended
from kogwistar.runtime.resolvers import MappingStepResolver


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


def _adapter_work(
    *,
    state: dict[str, Any] | None = None,
    routes: list[dict[str, Any]] | None = None,
    op: str = "start",
) -> dict[str, Any]:
    work = _work()
    work["payload"].update(
        {
            "node_id": "start",
            "turn_node_id": "turn-1",
            "state": dict(state or {"seed": 1}),
            "runtime_routes": list(routes or []),
            "op": op,
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
    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-1",
        journal_path=journal,
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as worker:
        with pytest.raises(Exception, match="response lost"):
            worker.process(_work())

    with RustRuntimeWorker(
        base_url="http://rust",
        worker_id="worker-2",
        journal_path=journal,
        execute=execute,
        client=httpx.Client(transport=httpx.MockTransport(handler), base_url="http://rust"),
    ) as restarted:
        assert restarted.process(_work(owner="worker-2"))["event_seq"] == 3

    assert calls == {"callback": 1, "result": 2}
    assert accepted[0]["handoff"]["claimed_by"] == "worker-2"
    assert "run_id" not in accepted[0]["effect"]
    assert "frontier" not in accepted[0]["effect"]
    assert accepted[0]["effect"]["state_update"] == [
        ["u", {"answer": "worker"}]
    ]


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

    def handler(request: httpx.Request) -> httpx.Response:
        calls["result"] += 1
        if calls["result"] == 1:
            raise httpx.ReadError("response lost", request=request)
        return httpx.Response(200, json={"event_seq": 3}, request=request)

    async def execute(_work: dict[str, Any]) -> dict[str, Any]:
        calls["callback"] += 1
        await asyncio.sleep(0)
        return _effect()

    async def invoke() -> None:
        journal = tmp_path / "async-worker.sqlite"
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
            await first.process(_work())
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
        assert (await second.process(_work(owner="worker-2")))["event_seq"] == 3
        await second.aclose()

    asyncio.run(invoke())
    assert calls == {"callback": 1, "result": 2}


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
