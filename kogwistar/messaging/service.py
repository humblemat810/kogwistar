from __future__ import annotations

from datetime import datetime, timezone
import json
import time
import uuid
from contextlib import nullcontext
from typing import Any

from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.acl import current_acl_context
from kogwistar.id_provider import stable_id
from kogwistar.server.auth_middleware import (
    can_access_security_scope,
    claims_ctx,
    get_security_scope,
    require_security_scope_access,
)

from .models import (
    LaneMessageLookup,
    LaneMessageProjectionRepairResult,
    LaneMessageSendResult,
    ProjectedLaneMessageRow,
)


def _now_epoch() -> int:
    return int(time.time())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _message_span(conversation_id: str, *, insertion_method: str, excerpt: str) -> Grounding:
    return Grounding(
        spans=[
            Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method=insertion_method,
                page_number=1,
                start_char=0,
                end_char=1,
                excerpt=excerpt,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
            )
        ]
    )


def _infer_message_purpose(msg_type: str, purpose: str | None) -> str:
    if purpose:
        return str(purpose)
    lowered = str(msg_type or "").lower()
    if "maintenance" in lowered or "repair" in lowered or "rebuild" in lowered:
        return "maintenance"
    return "user_visible"


def _decode_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except Exception:
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


def _compact_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _coerce_lane_datetime(value: datetime | int | float | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        return datetime.fromisoformat(text)
    except Exception:
        try:
            return datetime.fromtimestamp(float(text), tz=timezone.utc)
        except Exception:
            return None


def _coerce_lane_epoch(value: datetime | int | float | str | None) -> int | None:
    dt = _coerce_lane_datetime(value)
    if dt is not None:
        return int(dt.timestamp())
    return None


def _lane_record_from_payload(
    payload: dict[str, Any],
    *,
    namespace: str,
    entity_id: str | None,
    order: int,
) -> dict[str, Any] | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if str(metadata.get("artifact_kind") or "") != "lane_message":
        return None
    metadata_namespace = str(metadata.get("namespace") or namespace)
    if metadata_namespace != str(namespace):
        return None
    message_id = str(payload.get("id") or entity_id or "").strip()
    inbox_id = str(metadata.get("inbox_id") or "").strip()
    conversation_id = str(metadata.get("conversation_id") or "").strip()
    recipient_id = str(metadata.get("recipient_id") or "").strip()
    sender_id = str(metadata.get("sender_id") or "").strip()
    msg_type = str(metadata.get("msg_type") or "").strip()
    if not all([message_id, inbox_id, conversation_id, recipient_id, sender_id, msg_type]):
        return None
    payload_json = _compact_json(metadata.get("payload_json"))
    if payload_json is None:
        payload_json = _compact_json(metadata.get("payload"))
    error_json = _compact_json(metadata.get("error_json"))
    if error_json is None:
        error_json = _compact_json(metadata.get("error"))
    return {
        "order": int(order),
        "message_id": message_id,
        "namespace": metadata_namespace,
        "purpose": str(metadata.get("purpose") or "user_visible"),
        "inbox_id": inbox_id,
        "conversation_id": conversation_id,
        "recipient_id": recipient_id,
        "sender_id": sender_id,
        "msg_type": msg_type,
        "status": str(metadata.get("status") or "pending"),
        "created_at": int(order),
        "available_at": int(order),
        "run_id": metadata.get("run_id"),
        "step_id": metadata.get("step_id"),
        "correlation_id": metadata.get("correlation_id"),
        "payload_json": payload_json,
        "error_json": error_json,
    }


class LaneMessagingService:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def send_message(
        self,
        *,
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        payload: dict[str, Any],
        run_id: str | None = None,
        step_id: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        priority: int = 0,
        purpose: str | None = None,
        security_scope: str | None = None,
        shared_scope: bool = False,
        shared_inbox: bool = False,
        idempotency_key: str | None = None,
        idempotency_lookup: LaneMessageLookup | None = None,
    ) -> LaneMessageSendResult:
        claims = claims_ctx.get() or {}
        namespace = str(
            claims.get("storage_ns")
            or getattr(self.engine, "namespace", "default")
            or "default"
        )
        effective_scope = str(security_scope or get_security_scope()).strip().lower()
        shared_flag = bool(shared_scope or shared_inbox)
        effective_purpose = _infer_message_purpose(msg_type, purpose)
        require_security_scope_access(
            effective_scope,
            shared=shared_flag,
            action="send message into",
        )
        correlation = (
            correlation_id
            or (str(idempotency_key) if idempotency_key is not None else None)
            or f"corr:{uuid.uuid4()}"
        )
        now_epoch = _now_epoch()
        created_at = _now_iso()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        lookup = self._resolve_idempotency_lookup(
            namespace=namespace,
            conversation_id=conversation_id,
            inbox_id=inbox_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            msg_type=msg_type,
            correlation_id=correlation,
            reply_to_message_id=reply_to,
            idempotency_key=idempotency_key,
            provided=idempotency_lookup,
        )
        if idempotency_key:
            existing = self.find_messages(
                namespace=lookup.namespace,
                inbox_id=lookup.inbox_id,
                conversation_id=lookup.conversation_id,
                status=lookup.status,
                purpose=lookup.purpose,
                msg_type=lookup.msg_type,
                sender_id=lookup.sender_id,
                recipient_id=lookup.recipient_id,
                correlation_id=lookup.correlation_id,
                reply_to_message_id=lookup.reply_to_message_id,
                idempotency_key=lookup.idempotency_key,
                created_at_gte=lookup.created_at_gte,
                created_at_lte=lookup.created_at_lte,
                available_at_gte=lookup.available_at_gte,
                available_at_lte=lookup.available_at_lte,
                limit=max(int(lookup.limit or 1), 1),
                newest_first=True,
            )
            if existing:
                existing_node = existing[0]
                existing_metadata = dict(getattr(existing_node, "metadata", {}) or {})
                self._validate_idempotent_message(
                    metadata=existing_metadata,
                    conversation_id=conversation_id,
                    inbox_id=inbox_id,
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    msg_type=msg_type,
                    reply_to=reply_to,
                    correlation_id=correlation,
                    payload_json=payload_json,
                    run_id=run_id,
                    step_id=step_id,
                    purpose=effective_purpose,
                )
                self._ensure_projected_row_for_node(existing_node, namespace=namespace)
                return self._result_for_existing_message(
                    conversation_id=conversation_id,
                    inbox_id=inbox_id,
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    message_id=str(existing_node.id),
                )

        message_id = f"msg:{uuid.uuid4()}"

        unit_of_work = getattr(self.engine, "unit_of_work", None) or getattr(
            self.engine, "uow", None
        )
        uow_context = unit_of_work() if callable(unit_of_work) else nullcontext()
        with uow_context, scoped_namespace(self.engine, namespace):
            anchor_nodes = self._ensure_anchor_nodes(
                conversation_id=conversation_id,
                inbox_id=inbox_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
            )
            created_by = str(sender_id).strip()
            acl_context = current_acl_context(
                acl_enabled=bool(getattr(self.engine, "acl_enabled", False)),
                purpose="lane_message",
                source_graph="conversation",
                source_entity_id=message_id,
                visibility="shared" if shared_flag else "private",
                owner_id=sender_id,
            )
            acl_context_json = json.dumps(
                acl_context.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            message_node = Node(
                id=message_id,
                label=f"lane_message:{msg_type}",
                type="entity",
                summary=f"Lane message {msg_type} from {sender_id} to {recipient_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_message", excerpt=msg_type)],
                metadata={
                    "artifact_kind": "lane_message",
                    "conversation_id": conversation_id,
                    "inbox_id": inbox_id,
                    "sender_id": sender_id,
                    "recipient_id": recipient_id,
                    "msg_type": msg_type,
                    "status": "pending",
                    "priority": int(priority),
                    "correlation_id": correlation,
                    "reply_to_message_id": reply_to,
                    "run_id": run_id,
                    "step_id": step_id,
                    "security_scope": effective_scope,
                    "shared_scope": shared_flag,
                    "shared_inbox": bool(shared_inbox),
                    "visibility": "shared" if shared_flag else "private",
                    "purpose": effective_purpose,
                    "acl_context_json": acl_context_json,
                    "payload_json": payload_json,
                    "idempotency_key": idempotency_key,
                    "created_at": created_at,
                    "updated_at": created_at,
                    "completed_at": None,
                    "namespace": namespace,
                    "kind": "lane_message",
                },
            )
            self.engine.write.add_node(message_node)
            record_acl = getattr(self.engine, "record_acl", None)
            if callable(record_acl):
                record_acl(
                    truth_graph="conversation",
                    entity_id=message_id,
                    version=1,
                    mode="shared" if shared_flag else "private",
                    created_by=created_by,
                    owner_id=sender_id,
                    security_scope=effective_scope,
                )

            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "in_conversation", anchor_nodes["conversation"].id)),
                source_id=message_id,
                target_id=str(anchor_nodes["conversation"].id),
                relation="in_conversation",
                conversation_id=conversation_id,
            )
            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "in_inbox", anchor_nodes["inbox"].id)),
                source_id=message_id,
                target_id=str(anchor_nodes["inbox"].id),
                relation="in_inbox",
                conversation_id=conversation_id,
            )
            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "sent_by", anchor_nodes["sender"].id)),
                source_id=message_id,
                target_id=str(anchor_nodes["sender"].id),
                relation="sent_by",
                conversation_id=conversation_id,
            )
            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "sent_to", anchor_nodes["recipient"].id)),
                source_id=message_id,
                target_id=str(anchor_nodes["recipient"].id),
                relation="sent_to",
                conversation_id=conversation_id,
            )
            if reply_to:
                self._add_semantic_edge(
                    edge_id=str(stable_id("lane_message_edge", message_id, "reply_to", reply_to)),
                    source_id=message_id,
                    target_id=reply_to,
                    relation="reply_to",
                    conversation_id=conversation_id,
                )
            if run_id:
                run_target_id = self._first_existing_node_id(
                    [str(run_id), f"wf_run|{run_id}"]
                )
                if run_target_id:
                    self._add_semantic_edge(
                        edge_id=str(stable_id("lane_message_edge", message_id, "about_run", run_target_id)),
                        source_id=message_id,
                        target_id=run_target_id,
                        relation="about_run",
                        conversation_id=conversation_id,
                    )
            if step_id:
                step_candidates = [str(step_id)]
                if run_id is not None:
                    step_candidates.append(f"wf_step|{run_id}|{step_id}")
                step_target_id = self._first_existing_node_id(step_candidates)
                if step_target_id:
                    self._add_semantic_edge(
                        edge_id=str(stable_id("lane_message_edge", message_id, "about_step", step_target_id)),
                        source_id=message_id,
                        target_id=step_target_id,
                        relation="about_step",
                        conversation_id=conversation_id,
                    )

            project = getattr(self.engine.meta_sqlite, "project_lane_message", None)
            if callable(project):
                project(
                    message_id=message_id,
                    namespace=namespace,
                    purpose=effective_purpose,
                    inbox_id=inbox_id,
                    conversation_id=conversation_id,
                    recipient_id=recipient_id,
                    sender_id=sender_id,
                    msg_type=msg_type,
                    status="pending",
                    created_at=now_epoch,
                    available_at=now_epoch,
                    run_id=run_id,
                    step_id=step_id,
                    correlation_id=correlation,
                    payload_json=payload_json,
                    error_json=None,
                )

        return LaneMessageSendResult(
            message_id=message_id,
            conversation_anchor_id=str(anchor_nodes["conversation"].id),
            inbox_anchor_id=str(anchor_nodes["inbox"].id),
            sender_anchor_id=str(anchor_nodes["sender"].id),
            recipient_anchor_id=str(anchor_nodes["recipient"].id),
        )

    def update_message_status(
        self,
        *,
        message_id: str,
        status: str,
        error: dict[str, Any] | None = None,
        completed: bool | None = None,
    ) -> None:
        namespace = str(getattr(self.engine, "namespace", "default") or "default")
        now_iso = _now_iso()
        unit_of_work = getattr(self.engine, "unit_of_work", None) or getattr(
            self.engine, "uow", None
        )
        uow_context = unit_of_work() if callable(unit_of_work) else nullcontext()
        with uow_context, scoped_namespace(self.engine, namespace):
            try:
                current = self.engine.backend.node_get(
                    ids=[message_id],
                    include=["documents", "metadatas", "embeddings"],
                )
            except Exception:
                current = self.engine.backend.node_get(
                    ids=[message_id],
                    include=["documents", "metadatas"],
                )
            docs = current.get("documents") or []
            if not docs or not docs[0]:
                return
            node = Node.model_validate_json(docs[0])
            metadatas = current.get("metadatas") or []
            existing_metadata = metadatas[0] if metadatas else {}
            node.metadata = dict(existing_metadata or node.metadata or {})
            node.metadata["status"] = str(status)
            node.metadata["updated_at"] = now_iso
            if error is not None:
                node.metadata["error_json"] = json.dumps(
                    error,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            if completed or str(status) in {"completed", "failed", "cancelled"}:
                node.metadata["completed_at"] = now_iso

            doc, meta = self.engine.write.node_doc_and_meta(node)
            embeddings = current.get("embeddings")
            embedding = None
            if embeddings is not None and len(embeddings) >= 1:
                embedding = embeddings[0]
            update_kwargs: dict[str, Any] = {
                "ids": [message_id],
                "documents": [doc],
                "metadatas": [meta],
            }
            if embedding is not None:
                update_kwargs["embeddings"] = [embedding]
            self.engine.backend.node_update(**update_kwargs)
            payload = node.model_dump(field_mode="backend", exclude=["embedding"])
            self.engine._append_event_for_entity(
                namespace=namespace,
                entity_kind="node",
                entity_id=message_id,
                op="REPLACE",
                payload=payload if isinstance(payload, dict) else {},
            )

        update = getattr(self.engine.meta_sqlite, "update_projected_lane_message_status", None)
        if callable(update):
            update(
                message_id=message_id,
                status=str(status),
                error_json=(
                    json.dumps(error, sort_keys=True, separators=(",", ":"))
                    if error is not None
                    else None
                ),
            )

    def claim_pending(
        self,
        *,
        inbox_id: str,
        claimed_by: str,
        limit: int,
        lease_seconds: int,
    ) -> list[ProjectedLaneMessageRow]:
        claim = getattr(self.engine.meta_sqlite, "claim_projected_lane_messages", None)
        if not callable(claim):
            return []
        namespace = str(getattr(self.engine, "namespace", "default") or "default")
        return claim(
            namespace=namespace,
            inbox_id=inbox_id,
            claimed_by=claimed_by,
            limit=int(limit),
            lease_seconds=int(lease_seconds),
        )

    def ack(self, *, message_id: str, claimed_by: str) -> None:
        ack = getattr(self.engine.meta_sqlite, "ack_projected_lane_message", None)
        if callable(ack):
            ack(message_id=message_id, claimed_by=claimed_by)

    def requeue(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error: dict[str, Any] | None = None,
        delay_seconds: int = 0,
    ) -> None:
        requeue = getattr(self.engine.meta_sqlite, "requeue_projected_lane_message", None)
        if callable(requeue):
            requeue(
                message_id=message_id,
                claimed_by=claimed_by,
                error_json=(
                    json.dumps(error, sort_keys=True, separators=(",", ":"))
                    if error is not None
                    else None
                ),
                delay_seconds=int(delay_seconds),
            )

    def dead_letter(
        self,
        *,
        message_id: str,
        claimed_by: str,
        error: dict[str, Any] | None = None,
    ) -> None:
        dead_letter = getattr(self.engine.meta_sqlite, "dead_letter_projected_lane_message", None)
        if callable(dead_letter):
            dead_letter(
                message_id=message_id,
                claimed_by=claimed_by,
                error_json=(
                    json.dumps(error, sort_keys=True, separators=(",", ":"))
                    if error is not None
                    else None
                ),
            )

    def list_projected(
        self,
        *,
        inbox_id: str | None = None,
        status: str | None = None,
        purpose: str | None = None,
        conversation_id: str | None = None,
        msg_type: str | None = None,
        sender_id: str | None = None,
        recipient_id: str | None = None,
        correlation_id: str | None = None,
        reply_to_message_id: str | None = None,
        created_at_gte: datetime | int | float | str | None = None,
        created_at_lte: datetime | int | float | str | None = None,
        available_at_gte: datetime | int | float | str | None = None,
        available_at_lte: datetime | int | float | str | None = None,
        limit: int = 1000,
        newest_first: bool = False,
    ) -> list[ProjectedLaneMessageRow]:
        list_fn = getattr(self.engine.meta_sqlite, "list_projected_lane_messages", None)
        if not callable(list_fn):
            return []
        claims = claims_ctx.get() or {}
        namespace = str(
            claims.get("storage_ns")
            or getattr(self.engine, "namespace", "default")
            or "default"
        )
        rows = list_fn(
            namespace=namespace,
            inbox_id=inbox_id,
            status=status,
            purpose=purpose,
            conversation_id=conversation_id,
            msg_type=msg_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            correlation_id=correlation_id,
            created_at_gte=_coerce_lane_epoch(created_at_gte),
            created_at_lte=_coerce_lane_epoch(created_at_lte),
            available_at_gte=_coerce_lane_epoch(available_at_gte),
            available_at_lte=_coerce_lane_epoch(available_at_lte),
            limit=int(limit),
            newest_first=bool(newest_first),
        )
        if reply_to_message_id is not None:
            matching_ids = {
                str(node.id)
                for node in self.find_messages(
                    namespace=namespace,
                    reply_to_message_id=reply_to_message_id,
                    limit=max(int(limit), 100),
                    newest_first=bool(newest_first),
                )
            }
            rows = [row for row in rows if row.message_id in matching_ids]
        return [row for row in rows if self._row_visible(row)]

    def find_messages(
        self,
        *,
        namespace: str | None = None,
        inbox_id: str | None = None,
        conversation_id: str | None = None,
        status: str | None = None,
        purpose: str | None = None,
        msg_type: str | None = None,
        sender_id: str | None = None,
        recipient_id: str | None = None,
        correlation_id: str | None = None,
        reply_to_message_id: str | None = None,
        available_at_gte: datetime | int | float | str | None = None,
        available_at_lte: datetime | int | float | str | None = None,
        idempotency_key: str | None = None,
        created_at_gte: datetime | int | float | str | None = None,
        created_at_lte: datetime | int | float | str | None = None,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[Node]:
        target_namespace = str(namespace or getattr(self.engine, "namespace", "default") or "default")
        lower = _coerce_lane_datetime(created_at_gte)
        upper = _coerce_lane_datetime(created_at_lte)
        lower_available = _coerce_lane_datetime(available_at_gte)
        upper_available = _coerce_lane_datetime(available_at_lte)
        with scoped_namespace(self.engine, target_namespace):
            nodes = self.engine.read.get_nodes(where={"artifact_kind": "lane_message"}, limit=100_000)
        matched: list[Node] = []
        for node in nodes:
            metadata = dict(getattr(node, "metadata", {}) or {})
            if str(metadata.get("namespace") or target_namespace) != target_namespace:
                continue
            if inbox_id is not None and str(metadata.get("inbox_id") or "") != str(inbox_id):
                continue
            if conversation_id is not None and str(metadata.get("conversation_id") or "") != str(conversation_id):
                continue
            if status is not None and str(metadata.get("status") or "") != str(status):
                continue
            if purpose is not None and str(metadata.get("purpose") or "") != str(purpose):
                continue
            if msg_type is not None and str(metadata.get("msg_type") or "") != str(msg_type):
                continue
            if sender_id is not None and str(metadata.get("sender_id") or "") != str(sender_id):
                continue
            if recipient_id is not None and str(metadata.get("recipient_id") or "") != str(recipient_id):
                continue
            if correlation_id is not None and str(metadata.get("correlation_id") or "") != str(correlation_id):
                continue
            if reply_to_message_id is not None and str(metadata.get("reply_to_message_id") or "") != str(reply_to_message_id):
                continue
            if idempotency_key is not None and str(metadata.get("idempotency_key") or "") != str(idempotency_key):
                continue
            created_at = _coerce_lane_datetime(metadata.get("created_at"))
            if lower is not None and (created_at is None or created_at < lower):
                continue
            if upper is not None and (created_at is None or created_at > upper):
                continue
            available_at = _coerce_lane_datetime(metadata.get("available_at") or metadata.get("created_at"))
            if lower_available is not None and (available_at is None or available_at < lower_available):
                continue
            if upper_available is not None and (available_at is None or available_at > upper_available):
                continue
            matched.append(node)
        matched.sort(
            key=lambda node: (
                _coerce_lane_datetime((getattr(node, "metadata", {}) or {}).get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
                str(getattr(node, "id", "")),
            ),
            reverse=bool(newest_first),
        )
        return matched[: int(limit)]

    def repair_projection(
        self,
        *,
        namespace: str | None = None,
        rebuild: bool = False,
    ) -> LaneMessageProjectionRepairResult:
        target_namespace = str(namespace or getattr(self.engine, "namespace", "default") or "default")
        records = self._lane_projection_records_from_events(target_namespace)
        if not records:
            records = self._lane_projection_records_from_graph(target_namespace)
        meta = self.engine.meta_sqlite
        if rebuild:
            clear = getattr(meta, "clear_projected_lane_messages", None)
            if not callable(clear):
                raise AttributeError("metastore missing clear_projected_lane_messages")
            clear(target_namespace)
            existing_ids: set[str] = set()
        else:
            existing_ids = {
                row.message_id for row in self._list_projected_rows_for_repair(target_namespace)
            }
        repaired_count = 0
        skipped_count = 0
        project = getattr(meta, "project_lane_message", None)
        if not callable(project):
            raise AttributeError("metastore missing project_lane_message")
        for record in records:
            message_id = str(record["message_id"])
            if message_id in existing_ids:
                skipped_count += 1
                continue
            project(
                message_id=message_id,
                namespace=target_namespace,
                purpose=str(record["purpose"]),
                inbox_id=str(record["inbox_id"]),
                conversation_id=str(record["conversation_id"]),
                recipient_id=str(record["recipient_id"]),
                sender_id=str(record["sender_id"]),
                msg_type=str(record["msg_type"]),
                status=str(record["status"]),
                created_at=int(record["created_at"]),
                available_at=int(record["available_at"]),
                run_id=record["run_id"],
                step_id=record["step_id"],
                correlation_id=record["correlation_id"],
                payload_json=record["payload_json"],
                error_json=record["error_json"],
            )
            existing_ids.add(message_id)
            repaired_count += 1
        return LaneMessageProjectionRepairResult(
            namespace=target_namespace,
            scanned_count=len(records),
            repaired_count=repaired_count,
            skipped_count=skipped_count,
            rebuilt=bool(rebuild),
        )

    def _list_projected_rows_for_repair(self, namespace: str) -> list[ProjectedLaneMessageRow]:
        list_fn = getattr(self.engine.meta_sqlite, "list_projected_lane_messages", None)
        if not callable(list_fn):
            return []
        try:
            return list_fn(namespace=str(namespace), limit=100_000)
        except TypeError:
            return list_fn(namespace=str(namespace))

    def _lane_projection_records_from_events(self, namespace: str) -> list[dict[str, Any]]:
        iter_events = getattr(self.engine.meta_sqlite, "iter_entity_events", None)
        if not callable(iter_events):
            return []
        records_by_id: dict[str, dict[str, Any]] = {}
        try:
            events = iter_events(namespace=str(namespace), from_seq=1)
            for event in events:
                seq, entity_kind, entity_id, op, payload_json = event[:5]
                if str(entity_kind) != "node" or str(op) not in {"ADD", "REPLACE"}:
                    continue
                payload = _decode_json_dict(payload_json)
                record = _lane_record_from_payload(
                    payload,
                    namespace=str(namespace),
                    entity_id=str(entity_id),
                    order=int(seq),
                )
                if record is None:
                    continue
                existing = records_by_id.get(str(record["message_id"]))
                if existing is not None:
                    record["order"] = existing["order"]
                    record["created_at"] = existing["created_at"]
                    record["available_at"] = existing["available_at"]
                records_by_id[str(record["message_id"])] = record
        except Exception:
            return []
        refreshed: list[dict[str, Any]] = []
        for record in records_by_id.values():
            current = self._lane_projection_record_from_node(
                message_id=str(record["message_id"]),
                namespace=str(namespace),
                order=int(record["order"]),
            )
            refreshed.append(current or record)
        return sorted(refreshed, key=lambda item: (int(item["order"]), str(item["message_id"])))

    def _lane_projection_records_from_graph(self, namespace: str) -> list[dict[str, Any]]:
        with scoped_namespace(self.engine, namespace):
            nodes = self.engine.read.get_nodes(where={"artifact_kind": "lane_message"}, limit=100_000)
        records: list[dict[str, Any]] = []
        for index, node in enumerate(nodes, start=1):
            payload = node.model_dump(field_mode="backend", exclude={"embedding"})
            metadata = dict(getattr(node, "metadata", {}) or {})
            payload["metadata"] = metadata
            record = _lane_record_from_payload(
                payload,
                namespace=str(namespace),
                entity_id=str(getattr(node, "id", "") or ""),
                order=index,
            )
            if record is not None:
                record["_fallback_sort"] = (
                    str(metadata.get("created_at") or ""),
                    str(getattr(node, "id", "") or ""),
                )
                records.append(record)
        records.sort(key=lambda item: item.get("_fallback_sort") or ("", ""))
        for index, record in enumerate(records, start=1):
            record["order"] = index
            record["created_at"] = index
            record["available_at"] = index
            record.pop("_fallback_sort", None)
        return records

    def _lane_projection_record_from_node(
        self,
        *,
        message_id: str,
        namespace: str,
        order: int,
    ) -> dict[str, Any] | None:
        nodes = self.engine.read.get_nodes(ids=[str(message_id)])
        if not nodes:
            return None
        node = nodes[0]
        payload = node.model_dump(field_mode="backend", exclude={"embedding"})
        payload["metadata"] = dict(getattr(node, "metadata", {}) or {})
        return _lane_record_from_payload(
            payload,
            namespace=str(namespace),
            entity_id=str(message_id),
            order=int(order),
        )

    def _resolve_idempotency_lookup(
        self,
        *,
        namespace: str,
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        correlation_id: str,
        reply_to_message_id: str | None,
        idempotency_key: str | None,
        provided: LaneMessageLookup | None,
    ) -> LaneMessageLookup:
        base = {
            "namespace": str(namespace),
            "conversation_id": str(conversation_id),
            "inbox_id": str(inbox_id),
            "sender_id": str(sender_id),
            "recipient_id": str(recipient_id),
            "msg_type": str(msg_type),
            "correlation_id": str(correlation_id),
            "reply_to_message_id": None if reply_to_message_id is None else str(reply_to_message_id),
            "idempotency_key": None if idempotency_key is None else str(idempotency_key),
            "limit": 1,
            "newest_first": True,
        }
        if provided is None:
            return LaneMessageLookup(**base)
        merged = {
            "namespace": provided.namespace if provided.namespace is not None else base["namespace"],
            "inbox_id": provided.inbox_id if provided.inbox_id is not None else base["inbox_id"],
            "conversation_id": provided.conversation_id if provided.conversation_id is not None else base["conversation_id"],
            "status": provided.status,
            "purpose": provided.purpose,
            "msg_type": provided.msg_type if provided.msg_type is not None else base["msg_type"],
            "sender_id": provided.sender_id if provided.sender_id is not None else base["sender_id"],
            "recipient_id": provided.recipient_id if provided.recipient_id is not None else base["recipient_id"],
            "correlation_id": provided.correlation_id if provided.correlation_id is not None else base["correlation_id"],
            "reply_to_message_id": (
                provided.reply_to_message_id
                if provided.reply_to_message_id is not None
                else base["reply_to_message_id"]
            ),
            "idempotency_key": provided.idempotency_key if provided.idempotency_key is not None else base["idempotency_key"],
            "created_at_gte": provided.created_at_gte,
            "created_at_lte": provided.created_at_lte,
            "available_at_gte": provided.available_at_gte,
            "available_at_lte": provided.available_at_lte,
            "limit": max(int(provided.limit or 1), 1),
            "newest_first": True,
        }
        return LaneMessageLookup(**merged)

    def _validate_idempotent_message(
        self,
        *,
        metadata: dict[str, Any],
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
        msg_type: str,
        reply_to: str | None,
        correlation_id: str,
        payload_json: str,
        run_id: str | None,
        step_id: str | None,
        purpose: str,
    ) -> None:
        expected = {
            "conversation_id": str(conversation_id),
            "inbox_id": str(inbox_id),
            "sender_id": str(sender_id),
            "recipient_id": str(recipient_id),
            "msg_type": str(msg_type),
            "reply_to_message_id": None if reply_to is None else str(reply_to),
            "correlation_id": str(correlation_id),
            "payload_json": str(payload_json),
            "run_id": None if run_id is None else str(run_id),
            "step_id": None if step_id is None else str(step_id),
            "purpose": str(purpose),
        }
        for key, expected_value in expected.items():
            actual = metadata.get(key)
            actual_value = None if actual is None else str(actual)
            comparison_value = None if expected_value is None else str(expected_value)
            if actual_value != comparison_value:
                raise ValueError(
                    f"lane message idempotency conflict for {key}: expected {comparison_value!r}, got {actual_value!r}"
                )

    def _ensure_projected_row_for_node(self, node: Node, *, namespace: str) -> None:
        message_id = str(getattr(node, "id", "") or "")
        if not message_id:
            return
        get_row = getattr(self.engine.meta_sqlite, "_lane_message_get_row", None)
        if callable(get_row) and get_row(message_id=message_id) is not None:
            return
        record = self._lane_projection_record_from_node(
            message_id=message_id,
            namespace=str(namespace),
            order=1,
        )
        if record is None:
            return
        project = getattr(self.engine.meta_sqlite, "project_lane_message", None)
        if not callable(project):
            return
        project(
            message_id=message_id,
            namespace=str(namespace),
            purpose=str(record["purpose"]),
            inbox_id=str(record["inbox_id"]),
            conversation_id=str(record["conversation_id"]),
            recipient_id=str(record["recipient_id"]),
            sender_id=str(record["sender_id"]),
            msg_type=str(record["msg_type"]),
            status=str(record["status"]),
            created_at=int(record["created_at"]),
            available_at=int(record["available_at"]),
            run_id=record["run_id"],
            step_id=record["step_id"],
            correlation_id=record["correlation_id"],
            payload_json=record["payload_json"],
            error_json=record["error_json"],
        )

    def _result_for_existing_message(
        self,
        *,
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
        message_id: str,
    ) -> LaneMessageSendResult:
        return LaneMessageSendResult(
            message_id=str(message_id),
            conversation_anchor_id=str(stable_id("lane_message_conversation", conversation_id)),
            inbox_anchor_id=str(stable_id("lane_message_inbox", inbox_id)),
            sender_anchor_id=self._anchor_node_id(sender_id),
            recipient_anchor_id=self._anchor_node_id(recipient_id),
        )

    def _row_visible(self, row: ProjectedLaneMessageRow) -> bool:
        nodes = self.engine.read.get_nodes(ids=[row.message_id])
        if not nodes:
            return False
        md = dict(getattr(nodes[0], "metadata", {}) or {})
        return can_access_security_scope(
            str(md.get("security_scope") or ""),
            shared=bool(md.get("shared_scope") or md.get("shared_inbox")),
        )

    def _ensure_anchor_nodes(
        self,
        *,
        conversation_id: str,
        inbox_id: str,
        sender_id: str,
        recipient_id: str,
    ) -> dict[str, Node]:
        anchor_nodes = {
            "conversation": Node(
                id=str(stable_id("lane_message_conversation", conversation_id)),
                label=f"lane_conversation:{conversation_id}",
                type="entity",
                summary=f"Lane-messaging conversation anchor for {conversation_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_anchor", excerpt=conversation_id)],
                metadata={
                    "artifact_kind": "lane_conversation",
                    "conversation_id": conversation_id,
                    "kind": "lane_conversation",
                    "in_conversation_chain": False,
                },
            ),
            "inbox": Node(
                id=str(stable_id("lane_message_inbox", inbox_id)),
                label=f"lane_inbox:{inbox_id}",
                type="entity",
                summary=f"Lane inbox anchor for {inbox_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_anchor", excerpt=inbox_id)],
                metadata={
                    "artifact_kind": "lane_inbox",
                    "inbox_id": inbox_id,
                    "kind": "lane_inbox",
                    "in_conversation_chain": False,
                },
            ),
            "sender": Node(
                id=self._anchor_node_id(sender_id),
                label=f"lane_anchor:{sender_id}",
                type="entity",
                summary=f"Lane sender anchor for {sender_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_anchor", excerpt=sender_id)],
                metadata={
                    "artifact_kind": "lane_anchor",
                    "anchor_id": sender_id,
                    "kind": "lane_anchor",
                    "in_conversation_chain": False,
                },
            ),
            "recipient": Node(
                id=self._anchor_node_id(recipient_id),
                label=f"lane_anchor:{recipient_id}",
                type="entity",
                summary=f"Lane recipient anchor for {recipient_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_anchor", excerpt=recipient_id)],
                metadata={
                    "artifact_kind": "lane_anchor",
                    "anchor_id": recipient_id,
                    "kind": "lane_anchor",
                    "in_conversation_chain": False,
                },
            ),
        }
        for node in anchor_nodes.values():
            existing = self.engine.read.get_nodes(ids=[str(node.id)])
            if existing:
                continue
            self.engine.write.add_node(node)
        return anchor_nodes

    def _add_semantic_edge(
        self,
        *,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation: str,
        conversation_id: str,
    ) -> None:
        edge = Edge(
            id=edge_id,
            source_ids=[source_id],
            target_ids=[target_id],
            relation=relation,
            source_edge_ids=[],
            target_edge_ids=[],
            label=f"lane_message_edge:{relation}",
            type="relationship",
            summary=f"Lane message semantic edge {relation}",
            mentions=[_message_span(conversation_id, insertion_method="lane_message_edge", excerpt=relation)],
            metadata={
                "artifact_kind": "lane_message_edge",
                "relation_kind": relation,
                "conversation_id": conversation_id,
            },
        )
        self.engine.write.add_edge(edge)

    def _first_existing_node_id(self, candidate_ids: list[str]) -> str | None:
        for candidate_id in candidate_ids:
            if not candidate_id:
                continue
            try:
                nodes = self.engine.read.get_nodes(ids=[str(candidate_id)])
            except Exception:
                nodes = []
            if nodes:
                return str(candidate_id)
        return None

    def _anchor_node_id(self, entity_id: str) -> str:
        # Prefer the new anchor prefix, but reuse legacy ids if older graphs already have them.
        new_id = str(stable_id("lane_message_anchor", entity_id))
        legacy_id = str(stable_id("lane_message_actor", entity_id))
        return self._first_existing_node_id([new_id, legacy_id]) or new_id


__all__ = ["LaneMessagingService"]
