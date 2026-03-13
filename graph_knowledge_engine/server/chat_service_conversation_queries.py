from __future__ import annotations

from typing import Any, cast

from graph_knowledge_engine.conversation.models import ConversationNode

from .chat_service_shared import _BaseComponent


class _ConversationQueryService(_BaseComponent):
    """Owns conversation creation and transcript/query helpers."""

    def create_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        start_node_id: str | None = None,
    ) -> dict[str, Any]:
        svc = self._conversation_service()
        conv_id, start_id = svc.create_conversation(user_id=user_id, conv_id=conversation_id, node_id=start_node_id)
        return self.get_conversation(conv_id) | {"start_node_id": start_id}

    def _conversation_nodes(self, conversation_id: str) -> list[ConversationNode]:
        nodes: list[ConversationNode] = cast(
            list[ConversationNode],
            self._conversation_engine().get_nodes(
                where={"conversation_id": conversation_id},
                node_type=ConversationNode,
                limit=10_000,
            ),
        )
        if not nodes:
            raise KeyError(f"Unknown conversation_id: {conversation_id}")
        return nodes

    def _conversation_owner(self, conversation_id: str) -> str | None:
        starts = [
            node
            for node in self._conversation_nodes(conversation_id)
            if str((getattr(node, "metadata", {}) or {}).get("entity_type") or "") == "conversation_start"
        ]
        if not starts:
            return None
        starts.sort(key=lambda node: int(getattr(node, "turn_index", -1) or -1))
        return str(getattr(starts[0], "user_id", None) or "")

    def list_conversations_for_user(self, user_id: str) -> list[dict[str, Any]]:
        starts = cast(
            list[ConversationNode],
            self._conversation_engine().get_nodes(
                where={"$and": [{"entity_type": "conversation_start"}, {"user_id": user_id}]},
                node_type=ConversationNode,
                limit=10_000,
            ),
        )

        results = []
        for start in starts:
            conv_id = getattr(start, "conversation_id", None)
            if conv_id:
                results.append(
                    {
                        "id": str(conv_id),
                        "start_node_id": str(getattr(start, "id", "")),
                        "status": str((getattr(start, "properties", {}) or {}).get("status") or "active"),
                        "turn_count": len(self.list_transcript(str(conv_id))),
                    }
                )
        results.sort(key=lambda x: x["turn_count"], reverse=True)
        return results

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        nodes = self._conversation_nodes(conversation_id)
        svc = self._conversation_service()
        tail = svc.get_conversation_tail(conversation_id=conversation_id)
        starts = [
            node
            for node in nodes
            if str((getattr(node, "metadata", {}) or {}).get("entity_type") or "") == "conversation_start"
        ]
        turns = self.list_transcript(conversation_id)
        start_node = starts[0] if starts else None
        return {
            "conversation_id": conversation_id,
            "user_id": str(getattr(start_node, "user_id", None) or ""),
            "status": str((getattr(start_node, "properties", {}) or {}).get("status") or "active"),
            "start_node_id": str(getattr(start_node, "id", None) or ""),
            "tail_node_id": str(getattr(tail, "id", None) or ""),
            "turn_count": len(turns),
        }

    def list_transcript(self, conversation_id: str) -> list[dict[str, Any]]:
        nodes = self._conversation_nodes(conversation_id)
        turns: list[dict[str, Any]] = []
        for node in nodes:
            metadata = getattr(node, "metadata", {}) or {}
            entity_type = str(metadata.get("entity_type") or "")
            if entity_type not in {"conversation_turn", "assistant_turn"}:
                continue
            turn_index = getattr(node, "turn_index", None)
            if turn_index is None:
                continue
            turns.append(
                {
                    "node_id": str(getattr(node, "id", "") or ""),
                    "turn_index": int(turn_index),
                    "role": str(getattr(node, "role", "") or ""),
                    "content": str(getattr(node, "summary", "") or ""),
                    "entity_type": entity_type,
                }
            )
        turns.sort(key=lambda item: (int(item["turn_index"]), str(item["node_id"])))
        return turns

    def latest_snapshot(
        self,
        conversation_id: str,
        *,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> dict[str, Any]:
        svc = self._conversation_service()
        snap = svc.latest_context_snapshot_node(conversation_id=conversation_id, run_id=run_id, stage=stage)
        if snap is None:
            raise KeyError(f"No context snapshot found for conversation_id={conversation_id!r}")
        payload = svc.get_context_snapshot_payload(snapshot_node_id=str(snap.id))
        return {
            "snapshot_node_id": str(snap.id),
            "conversation_id": conversation_id,
            "metadata": payload.get("metadata") or {},
            "properties": payload.get("properties") or {},
        }
