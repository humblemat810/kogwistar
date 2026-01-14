"""ToolRunner: record tool_call/tool_result as conversation graph nodes.

Even if a "tool" is internal (memory/KG retrieval), recording it as events makes
the conversation graph auditable and future-proofs routing.

Storage rule:
- full payload in node.properties
- compact, LLM-safe rendering in node.summary (budget-friendly)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from .models import ConversationNode, Grounding, MentionVerification, Span
from graph_knowledge_engine.models import MetaFromLastSummary

T = TypeVar("T")


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"unserializable": str(obj)}, ensure_ascii=False)


@dataclass
class ToolEventIds:
    tool_call_node_id: str
    tool_result_node_id: str


class ToolRunner:
    def __init__(self, *, tool_call_id_factory, conversation_engine: Any) -> None:
        self.engine = conversation_engine
        self.tool_call_id_factory :Callable[[], str]= tool_call_id_factory
    def run_tool(
        self,
        *,
        conversation_id: str,
        user_id: str,
        turn_node_id: str,
        turn_index: int,
        tool_name: str,
        args: dict[str, Any],
        handler: Callable[[], T],
        prev_turn_meta_summary: MetaFromLastSummary,
        render_result: Optional[Callable[[T], str]] = None,
        
    ) -> T:
        """Execute a tool handler and record tool_call/tool_result nodes."""

        # Tool call node (assistant role)
        call_id = str(self.tool_call_id_factory())
        span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}#{call_id}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="tool_call",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt="",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="tool_call event"),
        )
        call_node_content = f"Calling tool {tool_name}"
        call_node = ConversationNode(
            user_id=user_id,
            id=call_id,
            label=f"tool_call:{tool_name}",
            type="entity",
            doc_id=call_id,
            summary=call_node_content,
            role="assistant",  # type: ignore
            turn_index=turn_index,
            conversation_id=conversation_id,
            mentions=[Grounding(spans=[span])],
            properties={
                "tool_name": tool_name,
                "args_json": json.dumps(args),
                "for_turn_node_id": turn_node_id,
                "ts_ms": int(time.time() * 1000),
            },
            metadata={"entity_type": "tool_call", "tool_name": tool_name,
                      "level_from_root": 0, "char_distance_from_last_summary": 0, "turn_distance_from_last_summary": 0},
            domain_id=None,
            canonical_entity_id=None,
        )
        self.engine.add_node(call_node, None)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary+= len(call_node_content)
        prev_turn_meta_summary.prev_node_turn_distance_from_last_summary+= 1
        # Execute
        result = handler()

        # Tool result node (tool role)
        res_id = str(self.tool_call_id_factory())
        res_span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}#{res_id}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="tool_result",
            page_number=1,
            start_char=0,
            end_char=1,
            excerpt="",
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="tool_result event"),
        )

        compact = ""
        if render_result is not None:
            try:
                compact = str(render_result(result) or "")
            except Exception:
                compact = ""
        if not compact:
            compact = _safe_json(getattr(result, "__dict__", result))[:800]
        res_node_content = compact
        res_node = ConversationNode(
            user_id=user_id,
            id=res_id,
            label=f"tool_result:{tool_name}",
            type="entity",
            doc_id=res_id,
            summary=compact,
            role="tool",  # type: ignore
            turn_index=turn_index,
            conversation_id=conversation_id,
            mentions=[Grounding(spans=[res_span])],
            properties={
                "tool_name": tool_name,
                # "result": getattr(result, "__dict__", result),
                "result_json": _safe_json(getattr(result, "__dict__", result)),
                "for_turn_node_id": turn_node_id,
                "call_node_id": call_id,
                "ts_ms": int(time.time() * 1000),
            },
            metadata={"entity_type": "tool_result", "tool_name": tool_name,
                      "level_from_root": 0, "char_distance_from_last_summary": 0, "turn_distance_from_last_summary": 0},
            domain_id=None,
            canonical_entity_id=None,
        )
        self.engine.add_node(res_node, None)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary+= len(res_node_content)
        prev_turn_meta_summary.prev_node_turn_distance_from_last_summary+= 1
        return result
