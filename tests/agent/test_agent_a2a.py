"""A2A adapter maps external identifiers without conflating runtime identity."""

from __future__ import annotations

import pytest

from kogwistar.server.a2a import A2AAdapter, A2AAgentCard, A2AMessage


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


def test_a2a_submit_poll_stream_cancel_and_duplicate_idempotency() -> None:
    state = {"run-1": {"run_id": "run-1", "status": "running", "evidence_refs": ["e1"]}}
    adapter = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1", "status": "queued"},
        inspect=lambda run_id: state[run_id],
        cancel=lambda run_id: {"run_id": run_id, "status": "cancelled"},
        resume=lambda run_id, _message: {"run_id": run_id, "status": "running"},
        events=lambda run_id, after: [{"seq": 1, "run_id": run_id}] if after < 1 else [],
        authorize=lambda _context, role: role == "user",
        card=A2AAgentCard("agent", "v1", capabilities=("poll",), security=("acl",)),
    )
    msg = A2AMessage(message_id="m1", role="user", parts=({"text": "hi"},))
    task = adapter.submit_task(task_id="task-1", context_id="ctx-1", message=msg)
    assert task.run_id == "run-1"
    assert task.task_id != task.run_id
    assert adapter.submit_task(task_id="task-1", context_id="ctx-1", message=msg).run_id == "run-1"
    assert adapter.stream_events(task_id="task-1", context_id="ctx-1")
    assert adapter.disconnect_task(task_id="task-1", context_id="ctx-1").status == "running"
    assert adapter.cancel_task(task_id="task-1", context_id="ctx-1").status == "cancelled"
    assert adapter.resume_task(
        task_id="task-1", context_id="ctx-1", message=A2AMessage("m2", "user")
    ).status == "running"
    assert adapter.agent_card().capabilities == ("poll",)
    with pytest.raises(KeyError):
        adapter.inspect_task(task_id="task-1", context_id="other")


def test_a2a_authorization_fails_closed() -> None:
    adapter = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1"},
        inspect=lambda _run_id: {"status": "queued"},
        cancel=lambda _run_id: {"status": "cancelled"},
        authorize=lambda _context, _role: False,
    )
    with pytest.raises(PermissionError):
        adapter.submit_task(
            task_id="task-1",
            context_id="ctx-1",
            message=A2AMessage(message_id="m1", role="user"),
        )


def test_a2a_external_mapping_survives_adapter_recreation() -> None:
    mappings: dict[str, tuple[str, str]] = {}

    class Store:
        def get(self, task_id: str) -> tuple[str, str] | None:
            return mappings.get(task_id)

        def put(self, task_id: str, context_id: str, run_id: str) -> None:
            mappings[task_id] = (context_id, run_id)

    store = Store()
    first = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1", "status": "queued"},
        inspect=lambda run_id: {"run_id": run_id, "status": "running"},
        cancel=lambda run_id: {"run_id": run_id, "status": "cancelled"},
        task_mapping_store=store,
    )
    first.submit_task(
        task_id="task-1",
        context_id="ctx-1",
        message=A2AMessage(message_id="m1", role="user"),
    )
    recreated = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "unexpected"},
        inspect=lambda run_id: {"run_id": run_id, "status": "running"},
        cancel=lambda run_id: {"run_id": run_id, "status": "cancelled"},
        task_mapping_store=store,
    )
    task = recreated.inspect_task(task_id="task-1", context_id="ctx-1")
    assert task.run_id == "run-1"
    with pytest.raises(KeyError):
        recreated.inspect_task(task_id="task-1", context_id="other")


def test_a2a_push_uses_signed_durable_delivery_with_audit_and_idempotency_key() -> None:
    queued: list[dict[str, object]] = []
    audited: list[dict[str, object]] = []
    adapter = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1", "status": "queued"},
        inspect=lambda _run_id: {"run_id": "run-1", "status": "running"},
        cancel=lambda _run_id: {"run_id": "run-1", "status": "cancelled"},
        enqueue_delivery=lambda item: (
            queued.append(dict(item)) or {"status": "queued", "durable": True}
        ),
        sign_callback=lambda delivery_id, _body: {
            "authorization": f"Signature {delivery_id}"
        },
        audit_delivery=lambda item: audited.append(dict(item)),
        allowed_callback_hosts=("callbacks.example.test",),
    )
    adapter.submit_task(
        task_id="task-1",
        context_id="ctx-1",
        message=A2AMessage(message_id="m1", role="user"),
    )
    subscription = adapter.register_push_callback(
        task_id="task-1",
        context_id="ctx-1",
        callback_url="https://callbacks.example.test/a2a",
        max_attempts=4,
        retry_backoff_seconds=7,
    )
    assert subscription.callback_url.startswith("https://")
    delivery = adapter.push_task_update(
        task_id="task-1",
        context_id="ctx-1",
        event_seq=3,
        payload={"status": "running", "result": {"answer": "bounded"}},
    )
    duplicate = adapter.push_task_update(
        task_id="task-1",
        context_id="ctx-1",
        event_seq=3,
        payload={"status": "running", "result": {"answer": "bounded"}},
    )
    assert delivery.delivery_id == duplicate.delivery_id == "a2a:task-1:3"
    assert queued[0]["delivery_id"] == "a2a:task-1:3"
    assert queued[0]["max_attempts"] == 4
    assert queued[0]["retry_backoff_seconds"] == 7
    assert audited == [
        {
            "delivery_id": "a2a:task-1:3",
            "task_id": "task-1",
            "context_id": "ctx-1",
            "event_seq": 3,
            "status": "queued",
        },
        {
            "delivery_id": "a2a:task-1:3",
            "task_id": "task-1",
            "context_id": "ctx-1",
            "event_seq": 3,
            "status": "queued",
        }
    ]


def test_a2a_push_rejects_untrusted_or_unauthenticated_callbacks() -> None:
    adapter = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1"},
        inspect=lambda _run_id: {"run_id": "run-1", "status": "queued"},
        cancel=lambda _run_id: {"run_id": "run-1", "status": "cancelled"},
        enqueue_delivery=lambda _item: {"status": "queued"},
        sign_callback=lambda _delivery_id, _body: {"authorization": "sig"},
        allowed_callback_hosts=("callbacks.example.test",),
    )
    adapter.submit_task(
        task_id="task-1",
        context_id="ctx-1",
        message=A2AMessage(message_id="m1", role="user"),
    )
    with pytest.raises(ValueError, match="HTTPS"):
        adapter.register_push_callback(
            task_id="task-1",
            context_id="ctx-1",
            callback_url="http://callbacks.example.test/a2a",
        )
    with pytest.raises(ValueError, match="allowlisted"):
        adapter.register_push_callback(
            task_id="task-1",
            context_id="ctx-1",
            callback_url="https://other.example.test/a2a",
        )

    no_delivery = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1"},
        inspect=lambda _run_id: {"run_id": "run-1", "status": "queued"},
        cancel=lambda _run_id: {"run_id": "run-1", "status": "cancelled"},
        allowed_callback_hosts=("callbacks.example.test",),
    )
    no_delivery.submit_task(
        task_id="task-1",
        context_id="ctx-1",
        message=A2AMessage(message_id="m1", role="user"),
    )
    with pytest.raises(RuntimeError, match="durable enqueue"):
        no_delivery.register_push_callback(
            task_id="task-1",
            context_id="ctx-1",
            callback_url="https://callbacks.example.test/a2a",
        )


def test_a2a_push_rejects_unconfirmed_queue_and_audits_failure() -> None:
    audited: list[dict[str, object]] = []
    adapter = A2AAdapter(
        submit=lambda **_kwargs: {"run_id": "run-1"},
        inspect=lambda _run_id: {"run_id": "run-1", "status": "queued"},
        cancel=lambda _run_id: {"run_id": "run-1", "status": "cancelled"},
        enqueue_delivery=lambda _item: {"status": "queued", "durable": False},
        sign_callback=lambda _delivery_id, _body: {"authorization": "sig"},
        audit_delivery=lambda item: audited.append(dict(item)),
        allowed_callback_hosts=("callbacks.example.test",),
    )
    adapter.submit_task(
        task_id="task-1",
        context_id="ctx-1",
        message=A2AMessage(message_id="m1", role="user"),
    )
    adapter.register_push_callback(
        task_id="task-1",
        context_id="ctx-1",
        callback_url="https://callbacks.example.test/a2a",
    )
    with pytest.raises(RuntimeError, match="durable enqueue"):
        adapter.push_task_update(
            task_id="task-1",
            context_id="ctx-1",
            event_seq=1,
            payload={"status": "running"},
        )
    assert audited[-1]["status"] == "enqueue_unconfirmed"
