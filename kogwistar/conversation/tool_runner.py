"""ToolRunner: record tool_call/tool_result as conversation graph nodes.

Even if a "tool" is internal (memory/KG retrieval), recording it as events makes
the conversation graph auditable and future-proofs routing.

Storage rule:
- full payload in node.properties
- compact, LLM-safe rendering in node.summary (budget-friendly)
"""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Tuple
from fastapi import HTTPException


from .models import ConversationEdge
from .tool_registry import ToolReceipt


from .models import BaseToolResult, ConversationNode
from ..engine_core.models import Grounding, MentionVerification, Span
from ..engine_core.async_compat import run_awaitable_blocking
from .policy import get_chat_tail
from ..server.auth_middleware import get_current_capabilities, has_explicit_capabilities_claim

if TYPE_CHECKING:
    from .models import MetaFromLastSummary
    from kogwistar.engine_core.engine import GraphKnowledgeEngine
    from kogwistar.conversation.conversation_orchestrator import (
        ConversationOrchestrator,
    )

T = TypeVar("T", bound=BaseToolResult)


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
    def __init__(
        self, *, tool_call_id_factory, conversation_engine: GraphKnowledgeEngine
    ) -> None:
        self.engine = conversation_engine
        self.tool_call_id_factory: Callable[..., str] = tool_call_id_factory
        self.last_receipt: ToolReceipt | None = None

    def join_tool_node_to_turn(
        self,
        orchestrator: ConversationOrchestrator,
        conversation_id,
        call_node,
        turn_node_id,
        prev_turn_meta_summary,
    ):
        orchestrator.join_tool_node_to_turn(
            conversation_id,
            call_node.safe_get_id(),
            turn_node_id,
            prev_turn_meta_summary,
        )

    @staticmethod
    def _normalize_kind(kind: str | None, *, fallback: str) -> str:
        value = str(kind or "").strip().lower()
        return value or fallback

    def _record_tool_call(
        self,
        *,
        conversation_id: str,
        user_id: str,
        turn_node_id: str,
        turn_index: int,
        tool_name: str,
        args: list,
        kwargs: dict[str, Any],
        prev_turn_meta_summary: MetaFromLastSummary,
    ) -> tuple[ConversationNode, ConversationNode | None, str]:
        try:
            last_node = get_chat_tail(self.engine, conversation_id=conversation_id)
        except Exception:
            last_node = None
        call_node_content = f"Calling tool {tool_name}"
        tail_id = (
            last_node.safe_get_id()
            if last_node is not None
            else str(turn_node_id)
        )
        call_id = str(
            self.tool_call_id_factory(
                tool_name,
                user_id,
                conversation_id,
                tail_id,
                call_node_content,
                str(args),
                str(kwargs),
            )
        )
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
            verification=MentionVerification(
                method="system", is_verified=True, score=1.0, notes="tool_call event"
            ),
        )
        prev_turn_meta_summary.tail_turn_index += 1
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
            metadata={
                "entity_type": "tool_call",
                "tool_name": tool_name,
                "level_from_root": 0,
                "in_conversation_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        self.engine.write.add_node(call_node, None)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
            call_node_content
        )
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        return call_node, last_node, call_id

    def _record_tool_result(
        self,
        *,
        conversation_id: str,
        user_id: str,
        turn_node_id: str,
        turn_index: int,
        tool_name: str,
        input_args: list,
        input_kwargs: dict[str, Any],
        call_node: ConversationNode,
        result_summary: str,
        result_payload: dict[str, Any],
        prev_turn_meta_summary: MetaFromLastSummary,
        execution_mode: str,
        kind: str,
        side_effects: list[str] | None = None,
    ) -> tuple[ConversationNode, str]:
        if side_effects is None:
            side_effects = []
        res_id = str(
            self.tool_call_id_factory(
                user_id, conversation_id, call_node.safe_get_id(), "tool_result"
            )
        )
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
            verification=MentionVerification(
                method="system",
                is_verified=True,
                score=1.0,
                notes="tool_result event",
            ),
        )
        prev_turn_meta_summary.tail_turn_index += 1
        res_node = ConversationNode(
            user_id=user_id,
            id=res_id,
            label=f"tool_result:{tool_name}",
            type="entity",
            doc_id=res_id,
            summary=result_summary,
            role="tool",  # type: ignore
            turn_index=turn_index,
            conversation_id=conversation_id,
            mentions=[Grounding(spans=[res_span])],
            properties={
                "tool_name": tool_name,
                "result_json": _safe_json(result_payload),
                "for_turn_node_id": turn_node_id,
                "call_node_id": call_node.safe_get_id(),
                "ts_ms": int(time.time() * 1000),
                "execution_mode": execution_mode,
            },
            metadata={
                "entity_type": "tool_result",
                "tool_name": tool_name,
                "level_from_root": 0,
                "in_conversation_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        self.engine.write.add_node(res_node, None)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
            result_summary
        )
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        self_span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}#{res_node.safe_get_id()}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="tool_call",
            page_number=1,
            start_char=0,
            end_char=len(res_node.summary),
            excerpt=res_node.summary,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="system",
                is_verified=True,
                score=1.0,
                notes="tool call created entrypoint to node/nodeset",
            ),
        )
        eid = (
            f"tool_call_result|caller:{call_node.safe_get_id()}|created:{res_node.safe_get_id()}"
        )
        e = ConversationEdge(
            id=eid,
            type="relationship",
            summary=f"call node {call_node.safe_get_id()} result node:{res_node.safe_get_id()}",
            domain_id=None,
            label="tool result",
            properties={},
            mentions=[Grounding(spans=[self_span])],
            canonical_entity_id=None,
            source_ids=[call_node.safe_get_id()],
            target_ids=[res_node.safe_get_id()],
            relation="run_result",
            source_edge_ids=[],
            target_edge_ids=[],
            embedding=None,
            doc_id=f"conv:{conversation_id}",
            metadata={
                "relation": "tool_call create node",
                "source_id": [call_node.safe_get_id()],
                "target_id": [res_node.safe_get_id()],
                "char_distance_from_last_summary": 0,
                "turn_distance_from_last_summary": 0,
                "entity_type": "tool-call->tool-result",
                "in_conversation": False,
            },
        )
        self.engine.write.add_edge(e)
        self.last_receipt = ToolReceipt(
            tool_id=call_node.safe_get_id(),
            tool_name=tool_name,
            kind=kind,
            execution_mode=execution_mode,
            capability=str(result_payload.get("capability") or ""),
            status="completed",
            input={"args": input_args, "kwargs": input_kwargs},
            output={"result_json": _safe_json(result_payload)},
            side_effects=side_effects,
        )
        return res_node, res_id

    def run_tool(
        self,
        *,
        conversation_id: str,
        user_id: str,
        turn_node_id: str,
        turn_index: int,
        tool_name: str,
        args: list,
        kwargs: dict[str, Any],
        handler: Callable[..., T],
        prev_turn_meta_summary: MetaFromLastSummary,
        render_result: Optional[Callable[[T], str]] = None,
        prev_node: ConversationNode | None = None,
        orchestrator: ConversationOrchestrator | None = None,
        tool_kind: str | None = None,
        execution_mode: str = "inline",
        supports_async: bool | None = None,
    ) -> Tuple[T, str]:
        """Execute a tool handler and record tool_call/tool_result nodes."""
        required_capability = str(kwargs.get("capability") or "").strip().lower()
        if required_capability:
            current_caps = get_current_capabilities()
            if not current_caps and not has_explicit_capabilities_claim():
                current_caps = {
                    "invoke_tool",
                    required_capability,
                }
            if required_capability not in current_caps and "invoke_tool" not in current_caps:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Forbidden: tool '{tool_name}' requires capability "
                        f"'{required_capability}'"
                    ),
                )
        call_node, last_node, call_id = self._record_tool_call(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            tool_name=tool_name,
            args=args,
            kwargs=kwargs,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )
        if orchestrator:
            self.join_tool_node_to_turn(
                orchestrator,
                conversation_id,
                call_node,
                turn_node_id,
                prev_turn_meta_summary,
            )
        # Execute
        raw_result = handler(**kwargs)
        if supports_async is False and inspect.isawaitable(raw_result):
            raise ValueError(f"Tool '{tool_name}' declared sync-only but returned awaitable")
        result_any = (
            run_awaitable_blocking(raw_result)
            if inspect.isawaitable(raw_result)
            else raw_result
        )
        result: BaseToolResult = result_any
        if result:
            if (
                tn := result.node_id_entry
            ):  # only tool that craeted nodes can be wrapped with node and edge linkages, some simple tools would not create and not use this code path
                created_nodes = self.engine.read.get_nodes(
                    [tn], node_type=ConversationNode
                )
                if created_nodes:
                    n: ConversationNode = created_nodes[0]
                    self_span = Span(
                        collection_page_url=f"conversation/{conversation_id}",
                        document_page_url=f"conversation/{conversation_id}#{n.safe_get_id()}",
                        doc_id=f"conv:{conversation_id}",
                        insertion_method="tool_call",
                        page_number=1,
                        start_char=0,
                        end_char=len(n.summary),
                        excerpt=n.summary,
                        context_before="",
                        context_after="",
                        chunk_id=None,
                        source_cluster_id=None,
                        verification=MentionVerification(
                            method="system",
                            is_verified=True,
                            score=1.0,
                            notes="tool call created entrypoint to node/nodeset",
                        ),
                    )
                    eid = f"tool_call_created|caller:{call_node.safe_get_id()}|created:{n.safe_get_id()}"
                    e = ConversationEdge(
                        id=eid,
                        type="relationship",
                        summary=f"call node {call_node.safe_get_id()} created entry point {n.safe_get_id()}",
                        domain_id=None,
                        label="craeted entry point",
                        properties={},
                        mentions=[Grounding(spans=[self_span])],
                        canonical_entity_id=None,
                        source_ids=[call_node.safe_get_id()],
                        target_ids=[n.safe_get_id()],
                        relation="run_result",
                        source_edge_ids=[],
                        target_edge_ids=[],
                        embedding=None,
                        doc_id=f"conv:{conversation_id}",
                        metadata={
                            "relation": "tool_call create node",
                            "source_id": [call_node.safe_get_id()],
                            "target_id": [n.safe_get_id()],
                            "char_distance_from_last_summary": 0,
                            "turn_distance_from_last_summary": 0,
                            "entity_type": "tool-call->details",
                            "in_conversation": False,
                        },
                    )
                    self.engine.write.add_edge(e)
        compact = ""
        if render_result is not None:
            try:
                compact = str(render_result(result) or "")
            except Exception:
                compact = ""
        if not compact:
            compact = _safe_json(getattr(result, "__dict__", result))[:800]
        res_node_content = compact
        res_node, res_id = self._record_tool_result(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            tool_name=tool_name,
            call_node=call_node,
            input_args=args,
            input_kwargs=kwargs,
            result_summary=compact,
            result_payload=getattr(result, "__dict__", result),
            prev_turn_meta_summary=prev_turn_meta_summary,
            execution_mode=execution_mode,
            kind=self._normalize_kind(
                tool_kind,
                fallback=(
                    "side_effecting"
                    if bool(getattr(result, "node_id_entry", None))
                    else "pure/query"
                ),
            ),
            side_effects=(
                [str(getattr(result, "node_id_entry"))]
                if getattr(result, "node_id_entry", None)
                else []
            ),
        )
        return result, call_node.safe_get_id()

    def run_subworkflow_tool(
        self,
        *,
        conversation_id: str,
        user_id: str,
        turn_node_id: str,
        turn_index: int,
        tool_name: str,
        args: list,
        kwargs: dict[str, Any],
        subworkflow_runner: Callable[..., Any],
        prev_turn_meta_summary: MetaFromLastSummary,
        render_result: Optional[Callable[[Any], str]] = None,
        orchestrator: ConversationOrchestrator | None = None,
        tool_kind: str = "workflow/subworkflow",
        execution_mode: str = "child-process",
    ) -> Tuple[Any, str]:
        """Run nested workflow as tool-shaped child process."""
        call_node, _, _ = self._record_tool_call(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            tool_name=tool_name,
            args=args,
            kwargs=kwargs,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )
        if orchestrator:
            self.join_tool_node_to_turn(
                orchestrator,
                conversation_id,
                call_node,
                turn_node_id,
                prev_turn_meta_summary,
            )
        result_any = subworkflow_runner(**kwargs)
        if inspect.isawaitable(result_any):
            result_any = run_awaitable_blocking(result_any)

        compact = ""
        if render_result is not None:
            try:
                compact = str(render_result(result_any) or "")
            except Exception:
                compact = ""
        if not compact:
            compact = _safe_json(
                {
                    "run_id": getattr(result_any, "run_id", None),
                    "status": getattr(result_any, "status", None),
                    "final_state": getattr(result_any, "final_state", None),
                }
            )[:800]
        self._record_tool_result(
            conversation_id=conversation_id,
            user_id=user_id,
            turn_node_id=turn_node_id,
            turn_index=turn_index,
            tool_name=tool_name,
            call_node=call_node,
            input_args=args,
            input_kwargs=kwargs,
            result_summary=compact,
            result_payload={
                "run_id": getattr(result_any, "run_id", None),
                "status": getattr(result_any, "status", None),
                "final_state": getattr(result_any, "final_state", None),
                "tool_kind": tool_kind,
            },
            prev_turn_meta_summary=prev_turn_meta_summary,
            execution_mode=execution_mode,
            kind=self._normalize_kind(tool_kind, fallback="workflow/subworkflow"),
            side_effects=[],
        )
        return result_any, call_node.safe_get_id()
