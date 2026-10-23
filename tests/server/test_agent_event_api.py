from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from kogwistar.agent.streaming import ProviderCapabilities
from kogwistar.server.agent_event_api import (
    EventAgentControlCapabilities,
    SubmitEventAgentInputIn,
    create_agent_event_router,
)

pytestmark = [pytest.mark.ci, pytest.mark.server]


class FakeRunService:
    requested_modes: list[str] = []
    events_drained = False

    def start_run(self, **kwargs: object) -> object:
        return {"run_id": "run-1", "input": kwargs["input"]}

    def submit_user_input(self, **kwargs: object) -> object:
        self.requested_modes.append(str(kwargs["requested_mode"]))
        return {
            "status": "pending",
            "effective_mode": "queue",
            "control_id": "control-1",
            "message_id": "message-1",
        }

    def get_control_receipt(self, **kwargs: object) -> object:
        assert kwargs == {
            "run_id": "run-1",
            "control_id": "control-1",
            "user_id": "user-1",
        }
        return {
            "status": "committed",
            "effective_mode": "queue",
            "control_id": str(kwargs["control_id"]),
            "message_id": "message-1",
        }

    def get_control_capabilities(self, **_: object) -> EventAgentControlCapabilities:
        return EventAgentControlCapabilities(
            provider=ProviderCapabilities(steering=False, cancellation=False)
        )

    def get_run(self, *, run_id: str, user_id: str) -> Mapping[str, object]:
        assert run_id == "run-1"
        assert user_id == "user-1"
        return {"run_id": run_id, "terminal": self.events_drained}

    def list_events(self, **kwargs: object) -> list[Mapping[str, object]]:
        after_seq = kwargs.get("after_seq")
        if isinstance(after_seq, int) and not isinstance(after_seq, bool) and after_seq > 0:
            self.events_drained = True
            return []
        return [
            {
                "sse_sequence": 1,
                "provider_sequence": 5,
                "event_type": "response_completed",
                "payload": {"status": "completed", "run_id": "forged"},
            }
        ]

    def list_transient_deltas(self, **kwargs: object) -> list[Mapping[str, object]]:
        after = kwargs.get("after_sequence")
        if isinstance(after, int) and not isinstance(after, bool) and after >= 1:
            return []
        return [{"client_sse_sequence": 1, "delta": "hello"}]


def test_event_agent_routes_authenticate_reuse_namespace_and_stream_client_sequence() -> None:
    FakeRunService.requested_modes.clear()
    FakeRunService.events_drained = False
    roles: list[str] = []
    namespaces: list[object] = []
    namespace = object()
    app = FastAPI()
    app.include_router(
        create_agent_event_router(
            get_service=FakeRunService,
            require_role=roles.append,
            require_namespace=namespaces.append,
            conversation_namespace=namespace,
            get_user_id=lambda: "user-1",
        )
    )
    client = TestClient(app)

    started = client.post(
        "/api/agent-events/runs",
        json={"conversation_id": "conv-1", "workflow_id": "agent.event.v1", "input": "hello"},
    )
    assert started.status_code == 202
    assert started.json()["run_id"] == "run-1"
    control = client.post(
        "/api/agent-events/runs/run-1/input",
        json={"text": "also check this", "requested_mode": "steer"},
    )
    assert control.status_code == 202
    assert control.json()["status"] == "pending"
    assert control.json()["effective_mode"] == "queue"
    assert "unsupported" in control.json()["note"]
    assert FakeRunService.requested_modes == ["queue"]
    control_status = client.get("/api/agent-events/runs/run-1/input/control-1")
    assert control_status.status_code == 200
    assert control_status.json()["status"] == "committed"
    inspection = client.get("/api/agent-events/runs/run-1")
    assert inspection.status_code == 200
    stream = client.get("/api/agent-events/runs/run-1/events")
    assert stream.status_code == 200
    assert "id: 1\nevent: response_completed" in stream.text
    assert "id: delta-1\nevent: response.text.delta" in stream.text
    assert '"delta":"hello"' in stream.text
    assert '"run_id":"run-1"' in stream.text
    assert '"run_id":"forged"' not in stream.text
    assert '"provider_sequence":5' in stream.text
    assert roles == ["rw", "rw", "ro", "ro", "ro"]
    assert namespaces == [namespace] * 5


def test_responses_provider_capabilities_bound_control_modes() -> None:
    from kogwistar.server.agent_event_api import resolve_control_mode

    capabilities = EventAgentControlCapabilities(
        provider=ProviderCapabilities(steering=False, cancellation=True),
        cancel_and_continue=True,
    )
    assert resolve_control_mode("steer", capabilities) == "queue"
    assert resolve_control_mode("cancel_and_continue", capabilities) == "cancel_and_continue"
    assert resolve_control_mode("queue", capabilities) == "queue"
    assert resolve_control_mode(
        "cancel_and_continue",
        EventAgentControlCapabilities(
            provider=ProviderCapabilities(steering=False, cancellation=False),
            cancel_and_continue=True,
        ),
    ) == "queue"
    assert resolve_control_mode(
        "steer",
        EventAgentControlCapabilities(
            provider=ProviderCapabilities(steering=True, cancellation=True),
            live_steering=True,
        ),
    ) == "steer"
    assert resolve_control_mode(
        "steer",
        EventAgentControlCapabilities(
            provider=ProviderCapabilities(steering=True), live_steering=False
        ),
    ) == "queue"


def test_control_api_rejects_service_receipt_that_widens_negotiated_mode() -> None:
    FakeRunService.events_drained = False

    class UnsafeRunService(FakeRunService):
        def submit_user_input(self, **_: object) -> object:
            return {
                "status": "accepted",
                "effective_mode": "steer",
                "control_id": "control-unsafe",
                "message_id": "message-unsafe",
            }

    router = create_agent_event_router(
        get_service=UnsafeRunService,
        require_role=lambda _: None,
        require_namespace=lambda _: None,
        conversation_namespace="conversation",
        get_user_id=lambda: "user-1",
    )
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path.endswith("/{run_id}/input")
    )
    assert route.endpoint is not None
    try:
        route.endpoint(
            "run-1",
            SubmitEventAgentInputIn(text="continue", requested_mode="steer"),
        )
    except HTTPException as exc:
        assert exc.status_code == 502
    else:
        raise AssertionError("service widened the negotiated control mode")


def test_control_api_rejects_input_for_terminal_run() -> None:
    class TerminalRunService(FakeRunService):
        def get_run(self, *, run_id: str, user_id: str) -> Mapping[str, object]:
            del run_id, user_id
            return {"terminal": True}

    router = create_agent_event_router(
        get_service=TerminalRunService,
        require_role=lambda _: None,
        require_namespace=lambda _: None,
        conversation_namespace="conversation",
        get_user_id=lambda: "user-1",
    )
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path.endswith("/{run_id}/input")
    )
    assert route.endpoint is not None
    try:
        route.endpoint(
            "run-1",
            SubmitEventAgentInputIn(text="late input", requested_mode="queue"),
        )
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("terminal run accepted a new control message")


def test_control_receipt_api_returns_not_found_for_unknown_control() -> None:
    class MissingReceiptService(FakeRunService):
        def get_control_receipt(self, **_: object) -> object:
            return None

    app = FastAPI()
    app.include_router(
        create_agent_event_router(
            get_service=MissingReceiptService,
            require_role=lambda _: None,
            require_namespace=lambda _: None,
            conversation_namespace="conversation",
            get_user_id=lambda: "user-1",
        )
    )
    response = TestClient(app).get(
        "/api/agent-events/runs/run-1/input/not-a-control"
    )
    assert response.status_code == 404


def test_terminal_sse_stream_stops_when_service_only_returns_stale_events() -> None:
    FakeRunService.events_drained = True

    class StaleEventService(FakeRunService):
        def list_events(self, **_: object) -> list[Mapping[str, object]]:
            return [{"event_type": "bad", "payload": {}}]

        def list_transient_deltas(self, **_: object) -> list[Mapping[str, object]]:
            return []

    router = create_agent_event_router(
        get_service=StaleEventService,
        require_role=lambda _: None,
        require_namespace=lambda _: None,
        conversation_namespace="conversation",
        get_user_id=lambda: "user-1",
    )
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and route.path.endswith("/{run_id}/events")
    )
    assert route.endpoint is not None
    response = asyncio.run(route.endpoint("run-1", after_seq=1))

    async def collect() -> list[bytes]:
        chunks: list[bytes] = []
        async for chunk in response.body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
        return chunks

    assert asyncio.run(collect()) == []
