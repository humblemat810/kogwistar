from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kogwistar.agent.event_control import EventControlMode, EventControlSubmission
from kogwistar.agent.streaming import ProviderCapabilities
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore
from kogwistar.server.agent_event_api import (
    EventAgentControlCapabilities,
    create_agent_event_router,
)
from kogwistar.server.agent_event_service import (
    EventAgentPublishCallbacks,
    EventAgentRunContext,
    RunExecutor,
    ScheduledEventAgentRunService,
)
from kogwistar.server.run_registry import RunRegistry

pytestmark = [pytest.mark.ci, pytest.mark.server]


class FakeScheduler:
    def __init__(self, *, admission: str = "accepted") -> None:
        self.admission = admission
        self.pending: list[Callable[[], None]] = []
        self.max_retries: list[int] = []

    def submit(
        self,
        *,
        run_id: str,
        priority_class: str,
        start_fn: Callable[[], None],
        max_retries: int = 0,
    ) -> Mapping[str, object]:
        assert run_id
        assert priority_class == "foreground"
        self.max_retries.append(max_retries)
        if self.admission == "accepted":
            self.pending.append(start_fn)
        return {"admission": self.admission}


class FakeControlJournal:
    def __init__(self) -> None:
        self.receipts: dict[tuple[str, str, str], EventControlSubmission] = {}

    def submit(
        self,
        *,
        run_id: str,
        conversation_id: str,
        user_id: str,
        recipient_id: str,
        control_id: str,
        text: str,
        effective_mode: EventControlMode,
        security_scope: str,
    ) -> EventControlSubmission:
        assert conversation_id and recipient_id and text and security_scope
        receipt = EventControlSubmission(
            control_id=control_id,
            message_id="message-1",
            status="pending",
            effective_mode=effective_mode,
        )
        self.receipts[(run_id, user_id, control_id)] = receipt
        return receipt

    def get_submission(
        self, *, run_id: str, user_id: str, control_id: str
    ) -> EventControlSubmission | None:
        return self.receipts.get((run_id, user_id, control_id))


def _service(
    *,
    scheduler: FakeScheduler | None = None,
    execute_run: RunExecutor | None = None,
    authorize_start: Callable[[str, str, str], None] | None = None,
    publish_transient_delta: Callable[[EventAgentRunContext, str, int], object] | None = None,
) -> tuple[ScheduledEventAgentRunService, FakeScheduler, list[EventAgentRunContext]]:
    run_registry = RunRegistry(InMemoryMetaStore())
    work_scheduler = scheduler or FakeScheduler()
    captured_contexts: list[EventAgentRunContext] = []

    async def default_execute(
        context: EventAgentRunContext,
        publish: EventAgentPublishCallbacks,
    ) -> Mapping[str, object]:
        captured_contexts.append(context)
        publish.publish_semantic_event("response_completed", {"response_id": "resp-1"})
        await publish.publish_transient_delta("hello", 1)
        return {"status": "succeeded"}

    service = ScheduledEventAgentRunService(
        registry=run_registry,
        scheduler=work_scheduler,
        control_journal=FakeControlJournal(),
        authorize_start=authorize_start or (lambda _conversation_id, _workflow_id, _user_id: None),
        create_user_turn=lambda run_id, _conversation_id, _user_id, _text: f"turn:{run_id}",
        execute_run=execute_run or default_execute,
        publish_transient_delta=publish_transient_delta,
        provider_capabilities=lambda _run_id, _user_id: ProviderCapabilities(
            steering=True, cancellation=True
        ),
        control_recipient=lambda _run_id, _user_id: "agent:main",
        security_scope=lambda _run_id, _user_id: "tenant-a",
    )
    return service, work_scheduler, captured_contexts


def test_scheduled_event_run_binds_registry_execution_and_sse_projection() -> None:
    service, scheduler, contexts = _service()
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    assert started["status"] == "queued"
    assert started["admission"] == "accepted"
    assert scheduler.max_retries == [0]
    assert len(scheduler.pending) == 1

    scheduler.pending.pop()()

    run = service.get_run(run_id=str(started["run_id"]), user_id="user-1")
    assert run["status"] == "succeeded"
    assert contexts[0].input == "hello"
    assert contexts[0].user_turn_node_id == started["user_turn_node_id"]
    events = service.list_events(
        run_id=str(started["run_id"]), user_id="user-1", after_seq=0, limit=100
    )
    assert [event["sse_sequence"] for event in events] == [1, 2, 3, 4]
    assert [event["event_type"] for event in events] == [
        "run.created",
        "run.started",
        "response_completed",
        "run.completed",
    ]
    # Token deltas use a separate, host-owned transient sink and never enter
    # the durable run registry event log.
    assert all("hello" not in str(event) for event in events)


def test_transient_deltas_are_sent_only_to_transient_sink() -> None:
    seen: list[tuple[str, str, int]] = []

    async def sink(context: EventAgentRunContext, delta: str, sequence: int) -> None:
        seen.append((context.run_id, delta, sequence))

    async def execute(
        context: EventAgentRunContext,
        publish: EventAgentPublishCallbacks,
    ) -> Mapping[str, object]:
        await publish.publish_transient_delta("token", 7)
        publish.publish_semantic_event("response_completed", {"response_id": "r"})
        return {"status": "succeeded"}

    service, scheduler, _ = _service(
        execute_run=execute,
        publish_transient_delta=sink,
    )
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    scheduler.pending.pop()()
    durable = service.list_events(
        run_id=str(started["run_id"]), user_id="user-1", after_seq=0, limit=20
    )
    assert seen == [(str(started["run_id"]), "token", 7)]
    assert "token" not in str(durable)
    assert "response_completed" in [event["event_type"] for event in durable]


def test_failed_optional_delta_sink_does_not_fail_run_or_drop_sse_buffer() -> None:
    async def failing_sink(
        _context: EventAgentRunContext, _delta: str, _sequence: int
    ) -> None:
        raise RuntimeError("client transport unavailable")

    service, scheduler, _ = _service(publish_transient_delta=failing_sink)
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    run_id = str(started["run_id"])
    scheduler.pending.pop()()

    assert service.get_run(run_id=run_id, user_id="user-1")["status"] == "succeeded"
    assert service.list_transient_deltas(
        run_id=run_id, user_id="user-1", after_sequence=0, limit=10
    )[0]["delta"] == "hello"


def test_transient_delta_buffer_is_bounded_and_reports_evicted_prefix() -> None:
    async def execute(
        _context: EventAgentRunContext,
        publish: EventAgentPublishCallbacks,
    ) -> Mapping[str, object]:
        for sequence in range(1, 71):
            await publish.publish_transient_delta(f"delta-{sequence}", sequence)
        return {"status": "succeeded"}

    service, scheduler, _ = _service(execute_run=execute)
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    run_id = str(started["run_id"])
    scheduler.pending.pop()()

    buffered = service.list_transient_deltas(
        run_id=run_id, user_id="user-1", after_sequence=0, limit=100
    )
    assert buffered[0] == {
        "event_type": "response.text.delta_gap",
        "client_sse_sequence": 6,
        "dropped_through_sequence": 6,
    }
    assert len(buffered) == 65
    assert buffered[1]["client_sse_sequence"] == 7


def test_durable_event_type_rejects_sse_control_characters() -> None:
    service, _scheduler, _ = _service()
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    with pytest.raises(ValueError, match="event type"):
        service._append_event(
            str(started["run_id"]), "response.completed\ndata: forged", {}
        )


def test_event_run_service_enforces_owner_and_queue_only_controls() -> None:
    service, scheduler, _ = _service()
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    run_id = str(started["run_id"])
    with pytest.raises(PermissionError):
        service.get_run(run_id=run_id, user_id="user-2")

    capabilities = service.get_control_capabilities(run_id=run_id, user_id="user-1")
    assert isinstance(capabilities, EventAgentControlCapabilities)
    assert capabilities.provider.steering is True
    assert capabilities.live_steering is False
    assert capabilities.cancel_and_continue is False
    receipt = service.submit_user_input(
        run_id=run_id,
        user_id="user-1",
        text="continue",
        requested_mode="steer",
    )
    assert receipt.status == "pending"
    assert receipt.effective_mode == "queue"
    assert len(scheduler.pending) == 1


def test_rejected_schedule_is_recorded_terminally_without_retry() -> None:
    service, scheduler, _ = _service(scheduler=FakeScheduler(admission="rejected"))
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    assert started["status"] == "failed"
    assert started["terminal"] is True
    assert scheduler.pending == []
    events = service.list_events(
        run_id=str(started["run_id"]), user_id="user-1", after_seq=0, limit=10
    )
    assert [event["event_type"] for event in events] == ["run.created", "run.failed"]


def test_scheduler_exception_terminalizes_run_without_exposing_exception_text() -> None:
    class BrokenScheduler(FakeScheduler):
        def submit(self, **_: object) -> Mapping[str, object]:
            raise RuntimeError("scheduler credential must not leak")

    service, _scheduler, _ = _service(scheduler=BrokenScheduler())
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    assert started["status"] == "failed"
    assert started["terminal"] is True
    run = service._registry.get_run(str(started["run_id"]))
    assert run is not None
    assert run["error"] == {
        "code": "scheduler_submit_failed",
        "detail": "RuntimeError",
    }


def test_async_event_workflow_executes_on_scheduler_thread_with_safe_failure_record() -> None:
    async def execute(
        _context: EventAgentRunContext,
        _publish: EventAgentPublishCallbacks,
    ) -> Mapping[str, object]:
        raise RuntimeError("provider output may contain sensitive details")

    service, scheduler, _ = _service(execute_run=execute)
    started = service.start_run(
        conversation_id="conversation-1",
        workflow_id="agent.event.v1",
        user_id="user-1",
        input="hello",
    )
    scheduler.pending.pop()()
    run = service.get_run(run_id=str(started["run_id"]), user_id="user-1")
    assert run["status"] == "failed"
    registry_run = service._registry.get_run(str(started["run_id"]))
    assert registry_run is not None
    assert registry_run["error"] == {
        "code": "event_workflow_exception",
        "detail": "RuntimeError",
    }


def test_opt_in_api_uses_scheduled_event_run_service_end_to_end() -> None:
    service, scheduler, _ = _service()
    app = FastAPI()
    app.include_router(
        create_agent_event_router(
            get_service=lambda: service,
            require_role=lambda _role: None,
            require_namespace=lambda _namespace: None,
            conversation_namespace="tenant-a",
            get_user_id=lambda: "user-1",
        )
    )
    client = TestClient(app)
    start = client.post(
        "/api/agent-events/runs",
        json={
            "conversation_id": "conversation-1",
            "workflow_id": "agent.event.v1",
            "input": "hello",
        },
    )
    assert start.status_code == 202
    run_id = start.json()["run_id"]

    control = client.post(
        f"/api/agent-events/runs/{run_id}/input",
        json={"text": "add this", "requested_mode": "steer"},
    )
    assert control.status_code == 202, control.text
    assert control.json()["effective_mode"] == "queue"
    receipt = client.get(
        f"/api/agent-events/runs/{run_id}/input/{control.json()['control_id']}"
    )
    assert receipt.status_code == 200
    assert receipt.json()["status"] == "pending"

    scheduler.pending.pop()()
    inspect = client.get(f"/api/agent-events/runs/{run_id}")
    assert inspect.status_code == 200
    assert inspect.json()["status"] == "succeeded"
    stream = client.get(f"/api/agent-events/runs/{run_id}/events")
    assert stream.status_code == 200
    assert "event: response_completed" in stream.text
    assert "id: delta-1\nevent: response.text.delta" in stream.text
    assert '"delta":"hello"' in stream.text
    assert "id: 4\nevent: run.completed" in stream.text
