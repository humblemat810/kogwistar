"""Small polling-first A2A mapping; no A2A dependency or new run authority."""

from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass
from urllib.parse import urlsplit
from typing import Any, Callable, Iterable, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class A2AAgentCard:
    name: str
    version: str
    capabilities: tuple[str, ...] = ()
    security: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class A2AMessage:
    message_id: str
    role: str
    parts: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class A2ATask:
    task_id: str
    context_id: str
    run_id: str
    status: str
    result: Mapping[str, Any] | None = None
    input_required: bool = False
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class A2APushSubscription:
    task_id: str
    context_id: str
    callback_url: str
    max_attempts: int = 5
    retry_backoff_seconds: int = 5


@dataclass(frozen=True, slots=True)
class A2APushDelivery:
    delivery_id: str
    task_id: str
    context_id: str
    callback_url: str
    event_seq: int
    body: str
    headers: Mapping[str, str]
    max_attempts: int
    retry_backoff_seconds: int


class A2ATaskMappingStore(Protocol):
    """Durable host-owned mapping for external A2A IDs."""

    def get(self, task_id: str) -> tuple[str, str] | None: ...

    def put(self, task_id: str, context_id: str, run_id: str) -> None: ...


class A2AAdapter:
    """Map external task/context IDs to ordinary Kogwistar run APIs."""

    def __init__(
        self,
        *,
        submit: Callable[..., Mapping[str, Any]],
        inspect: Callable[[str], Mapping[str, Any]],
        cancel: Callable[[str], Mapping[str, Any]],
        resume: Callable[[str, A2AMessage], Mapping[str, Any]] | None = None,
        events: Callable[[str, int], Iterable[Mapping[str, Any]]] | None = None,
        authorize: Callable[[str, str], bool] | None = None,
        card: A2AAgentCard | None = None,
        enqueue_delivery: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        sign_callback: Callable[[str, bytes], Mapping[str, str]] | None = None,
        audit_delivery: Callable[[Mapping[str, Any]], None] | None = None,
        allowed_callback_hosts: Iterable[str] = (),
        max_push_body_bytes: int = 256 * 1024,
        task_mapping_store: A2ATaskMappingStore | None = None,
    ) -> None:
        self._submit = submit
        self._inspect = inspect
        self._cancel = cancel
        self._resume = resume
        self._events = events
        self._authorize = authorize
        self._card = card
        self._enqueue_delivery = enqueue_delivery
        self._sign_callback = sign_callback
        self._audit_delivery = audit_delivery
        self._allowed_callback_hosts = frozenset(
            host.strip().lower().rstrip(".")
            for host in allowed_callback_hosts
            if host and host.strip()
        )
        if max_push_body_bytes <= 0:
            raise ValueError("max_push_body_bytes must be positive")
        self._max_push_body_bytes = int(max_push_body_bytes)
        self._task_mapping_store = task_mapping_store
        self._by_task: dict[str, tuple[str, str]] = {}
        self._push: dict[str, A2APushSubscription] = {}

    def agent_card(self) -> A2AAgentCard | None:
        """Return declarative capability/security metadata, never credentials."""

        return self._card

    def submit_task(self, *, task_id: str, context_id: str, message: A2AMessage, **kwargs: Any) -> A2ATask:
        if self._authorize is not None and not self._authorize(context_id, message.role):
            raise PermissionError("A2A task is not authorized")
        existing = self._by_task.get(task_id)
        if existing is not None:
            if existing[0] != context_id:
                raise ValueError("A2A task ID is bound to another context")
            return self.inspect_task(task_id=task_id, context_id=context_id)
        payload = dict(self._submit(context_id=context_id, message=message, **kwargs))
        run_id = str(payload["run_id"])
        self._by_task[task_id] = (context_id, run_id)
        if self._task_mapping_store is not None:
            self._task_mapping_store.put(task_id, context_id, run_id)
        return self._task(task_id, context_id, {**payload, "run_id": run_id})

    def inspect_task(self, *, task_id: str, context_id: str) -> A2ATask:
        bound = self._bound(task_id, context_id)
        return self._task(task_id, context_id, {**dict(self._inspect(bound[1])), "run_id": bound[1]})

    def cancel_task(self, *, task_id: str, context_id: str) -> A2ATask:
        bound = self._bound(task_id, context_id)
        return self._task(task_id, context_id, {**dict(self._cancel(bound[1])), "run_id": bound[1]})

    def resume_task(
        self, *, task_id: str, context_id: str, message: A2AMessage
    ) -> A2ATask:
        if self._resume is None:
            raise NotImplementedError("A2A resume is not configured")
        if self._authorize is not None and not self._authorize(context_id, message.role):
            raise PermissionError("A2A resume is not authorized")
        bound = self._bound(task_id, context_id)
        payload = dict(self._resume(bound[1], message))
        return self._task(task_id, context_id, {**payload, "run_id": bound[1]})

    def stream_events(self, *, task_id: str, context_id: str, after: int = 0) -> tuple[Mapping[str, Any], ...]:
        if self._events is None:
            raise NotImplementedError("A2A event streaming is not configured")
        bound = self._bound(task_id, context_id)
        return tuple(self._events(bound[1], int(after)))

    def disconnect_task(self, *, task_id: str, context_id: str) -> A2ATask:
        """Observe client disconnect without cancelling the ordinary run."""

        return self.inspect_task(task_id=task_id, context_id=context_id)

    def register_push_callback(
        self,
        *,
        task_id: str,
        context_id: str,
        callback_url: str,
        actor_role: str = "user",
        max_attempts: int = 5,
        retry_backoff_seconds: int = 5,
    ) -> A2APushSubscription:
        """Register a signed callback delivered by an injected durable queue.

        The adapter never performs network I/O.  Queue durability, retry, and
        delivery workers remain owned by the host's existing infrastructure.
        """

        if self._authorize is not None and not self._authorize(context_id, actor_role):
            raise PermissionError("A2A push callback is not authorized")
        self._bound(task_id, context_id)
        if self._enqueue_delivery is None or self._sign_callback is None:
            raise RuntimeError("A2A push requires durable enqueue and callback signing")
        if max_attempts < 1 or max_attempts > 100:
            raise ValueError("max_attempts must be between 1 and 100")
        if retry_backoff_seconds < 0 or retry_backoff_seconds > 86400:
            raise ValueError("retry_backoff_seconds is out of bounds")
        normalized_url = _validate_callback_url(
            callback_url,
            allowed_hosts=self._allowed_callback_hosts,
        )
        subscription = A2APushSubscription(
            task_id=task_id,
            context_id=context_id,
            callback_url=normalized_url,
            max_attempts=int(max_attempts),
            retry_backoff_seconds=int(retry_backoff_seconds),
        )
        self._push[task_id] = subscription
        return subscription

    def remove_push_callback(self, *, task_id: str, context_id: str) -> None:
        self._bound(task_id, context_id)
        self._push.pop(task_id, None)

    def push_task_update(
        self,
        *,
        task_id: str,
        context_id: str,
        event_seq: int,
        payload: Mapping[str, Any],
    ) -> A2APushDelivery:
        """Enqueue one signed, bounded, idempotent task update."""

        self._bound(task_id, context_id)
        subscription = self._push.get(task_id)
        if subscription is None:
            raise KeyError("A2A push callback is not registered")
        if event_seq < 0:
            raise ValueError("event_seq must be non-negative")
        body = json.dumps(
            {
                "task_id": task_id,
                "context_id": context_id,
                "event_seq": int(event_seq),
                "payload": dict(payload),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        body_bytes = body.encode("utf-8")
        if len(body_bytes) > self._max_push_body_bytes:
            raise ValueError("A2A push payload exceeds configured bound")
        delivery_id = f"a2a:{task_id}:{int(event_seq)}"
        headers = dict(self._sign_callback(delivery_id, body_bytes))
        if not headers:
            raise PermissionError("A2A callback signer returned no authentication headers")
        delivery = A2APushDelivery(
            delivery_id=delivery_id,
            task_id=task_id,
            context_id=context_id,
            callback_url=subscription.callback_url,
            event_seq=int(event_seq),
            body=body,
            headers=headers,
            max_attempts=subscription.max_attempts,
            retry_backoff_seconds=subscription.retry_backoff_seconds,
        )
        queue_payload = {
            "delivery_id": delivery.delivery_id,
            "kind": "a2a_task_update",
            "task_id": delivery.task_id,
            "context_id": delivery.context_id,
            "callback_url": delivery.callback_url,
            "body": delivery.body,
            "headers": dict(delivery.headers),
            "max_attempts": delivery.max_attempts,
            "retry_backoff_seconds": delivery.retry_backoff_seconds,
        }
        try:
            result = dict(self._enqueue_delivery(queue_payload))
        except Exception as exc:
            if self._audit_delivery is not None:
                self._audit_delivery(
                    {
                        "delivery_id": delivery.delivery_id,
                        "task_id": delivery.task_id,
                        "context_id": delivery.context_id,
                        "event_seq": delivery.event_seq,
                        "status": "enqueue_failed",
                        "error_type": type(exc).__name__,
                    }
                )
            raise
        if result.get("durable") is not True:
            if self._audit_delivery is not None:
                self._audit_delivery(
                    {
                        "delivery_id": delivery.delivery_id,
                        "task_id": delivery.task_id,
                        "context_id": delivery.context_id,
                        "event_seq": delivery.event_seq,
                        "status": "enqueue_unconfirmed",
                    }
                )
            raise RuntimeError("A2A delivery queue did not confirm durable enqueue")
        if self._audit_delivery is not None:
            self._audit_delivery(
                {
                    "delivery_id": delivery.delivery_id,
                    "task_id": delivery.task_id,
                    "context_id": delivery.context_id,
                    "event_seq": delivery.event_seq,
                    "status": str(result.get("status", "queued")),
                }
            )
        return delivery

    def _bound(self, task_id: str, context_id: str) -> tuple[str, str]:
        bound = self._by_task.get(task_id)
        if bound is None and self._task_mapping_store is not None:
            bound = self._task_mapping_store.get(task_id)
            if bound is not None:
                bound = (str(bound[0]), str(bound[1]))
                self._by_task[task_id] = bound
        if bound is None or bound[0] != context_id:
            raise KeyError("unknown A2A task or context")
        return bound

    @staticmethod
    def _task(task_id: str, context_id: str, payload: Mapping[str, Any]) -> A2ATask:
        return A2ATask(
            task_id=task_id,
            context_id=context_id,
            run_id=str(payload["run_id"]),
            status=str(payload.get("status", "queued")),
            result=payload.get("result"),
            input_required=bool(payload.get("input_required", False)),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", ())),
        )


def _validate_callback_url(callback_url: str, *, allowed_hosts: frozenset[str]) -> str:
    parsed = urlsplit(str(callback_url))
    if parsed.scheme != "https":
        raise ValueError("A2A callback URL must use HTTPS")
    if parsed.username or parsed.password or not parsed.hostname:
        raise ValueError("A2A callback URL must not contain credentials")
    host = parsed.hostname.lower().rstrip(".")
    if host not in allowed_hosts:
        raise ValueError("A2A callback host is not allowlisted")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ValueError("A2A callback host is not publicly routable")
    if parsed.port not in (None, 443):
        raise ValueError("A2A callback URL must use HTTPS port 443")
    return parsed.geturl()
