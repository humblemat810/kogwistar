from __future__ import annotations

import json
import time
import uuid
from typing import Any

from kogwistar.engine_core.engine import scoped_namespace
from kogwistar.engine_core.models import Edge, Grounding, Node, Span
from kogwistar.id_provider import stable_id
from kogwistar.server.auth_middleware import (
    can_access_security_scope,
    claims_ctx,
    get_security_scope,
    require_security_scope_access,
)

from .models import LaneMessageSendResult, ProjectedLaneMessageRow


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
        security_scope: str | None = None,
        shared_scope: bool = False,
        shared_inbox: bool = False,
    ) -> LaneMessageSendResult:
        claims = claims_ctx.get() or {}
        namespace = str(
            claims.get("storage_ns")
            or getattr(self.engine, "namespace", "default")
            or "default"
        )
        effective_scope = str(security_scope or get_security_scope()).strip().lower()
        shared_flag = bool(shared_scope or shared_inbox)
        require_security_scope_access(
            effective_scope,
            shared=shared_flag,
            action="send message into",
        )
        message_id = f"msg:{uuid.uuid4()}"
        correlation = correlation_id or f"corr:{uuid.uuid4()}"
        now_epoch = _now_epoch()
        created_at = _now_iso()

        with scoped_namespace(self.engine, namespace):
            anchors = self._ensure_anchor_nodes(
                conversation_id=conversation_id,
                inbox_id=inbox_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
            )
            created_by = str(sender_id).strip()

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
                    "payload": payload,
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
                edge_id=str(stable_id("lane_message_edge", message_id, "in_conversation", anchors["conversation"].id)),
                source_id=message_id,
                target_id=str(anchors["conversation"].id),
                relation="in_conversation",
                conversation_id=conversation_id,
            )
            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "in_inbox", anchors["inbox"].id)),
                source_id=message_id,
                target_id=str(anchors["inbox"].id),
                relation="in_inbox",
                conversation_id=conversation_id,
            )
            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "sent_by", anchors["sender"].id)),
                source_id=message_id,
                target_id=str(anchors["sender"].id),
                relation="sent_by",
                conversation_id=conversation_id,
            )
            self._add_semantic_edge(
                edge_id=str(stable_id("lane_message_edge", message_id, "sent_to", anchors["recipient"].id)),
                source_id=message_id,
                target_id=str(anchors["recipient"].id),
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
                self._add_semantic_edge(
                    edge_id=str(stable_id("lane_message_edge", message_id, "about_run", run_id)),
                    source_id=message_id,
                    target_id=run_id,
                    relation="about_run",
                    conversation_id=conversation_id,
                )
            if step_id:
                self._add_semantic_edge(
                    edge_id=str(stable_id("lane_message_edge", message_id, "about_step", step_id)),
                    source_id=message_id,
                    target_id=step_id,
                    relation="about_step",
                    conversation_id=conversation_id,
                )

        project = getattr(self.engine.meta_sqlite, "project_lane_message", None)
        if callable(project):
            project(
                message_id=message_id,
                namespace=namespace,
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
                payload_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                error_json=None,
            )

        return LaneMessageSendResult(
            message_id=message_id,
            conversation_anchor_id=str(anchors["conversation"].id),
            inbox_anchor_id=str(anchors["inbox"].id),
            sender_anchor_id=str(anchors["sender"].id),
            recipient_anchor_id=str(anchors["recipient"].id),
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
        with scoped_namespace(self.engine, namespace):
            current = self.engine.backend.node_get(
                ids=[message_id],
                include=["documents", "metadatas", "embeddings"],
            )
            docs = current.get("documents") or []
            if not docs or not docs[0]:
                return
            node = Node.model_validate_json(docs[0])
            node.metadata = dict(node.metadata or {})
            node.metadata["status"] = str(status)
            node.metadata["updated_at"] = now_iso
            if error is not None:
                node.metadata["error"] = error
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
        rows = list_fn(namespace=namespace, inbox_id=inbox_id, status=status)
        return [row for row in rows if self._row_visible(row)]

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
        anchors = {
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
                id=str(stable_id("lane_message_actor", sender_id)),
                label=f"lane_actor:{sender_id}",
                type="entity",
                summary=f"Lane actor anchor for {sender_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_anchor", excerpt=sender_id)],
                metadata={
                    "artifact_kind": "lane_actor",
                    "actor_id": sender_id,
                    "kind": "lane_actor",
                    "in_conversation_chain": False,
                },
            ),
            "recipient": Node(
                id=str(stable_id("lane_message_actor", recipient_id)),
                label=f"lane_actor:{recipient_id}",
                type="entity",
                summary=f"Lane actor anchor for {recipient_id}",
                mentions=[_message_span(conversation_id, insertion_method="lane_anchor", excerpt=recipient_id)],
                metadata={
                    "artifact_kind": "lane_actor",
                    "actor_id": recipient_id,
                    "kind": "lane_actor",
                    "in_conversation_chain": False,
                },
            ),
        }
        for node in anchors.values():
            self.engine.write.add_node(node)
        return anchors

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


__all__ = ["LaneMessagingService"]
