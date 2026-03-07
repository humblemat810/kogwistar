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
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Iterable, Tuple

from pydantic import Json

from .models import ConversationEdge


from .models import BaseToolResult, ConversationNode
from ..engine_core.models import Grounding, MentionVerification, Span
if TYPE_CHECKING:
    from .models import MetaFromLastSummary
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
    from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator

T = TypeVar("T", bound = BaseToolResult)


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
    def __init__(self, *, tool_call_id_factory, conversation_engine: GraphKnowledgeEngine) -> None:
        self.engine = conversation_engine
        self.tool_call_id_factory :Callable[..., str]= tool_call_id_factory
    def join_tool_node_to_turn(self, orchestrator: ConversationOrchestrator, conversation_id, call_node, turn_node_id, prev_turn_meta_summary):
        orchestrator.join_tool_node_to_turn(conversation_id, call_node.safe_get_id(), turn_node_id, prev_turn_meta_summary)
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
        orchestrator: ConversationOrchestrator | None = None
    ) -> Tuple[T, str]:
        """Execute a tool handler and record tool_call/tool_result nodes."""
        if hasattr(self.engine, "conversation") and hasattr(self.engine.conversation, "get_conversation_tail"):
            last_node = self.engine.conversation.get_conversation_tail(conversation_id)
        elif hasattr(self.engine, "_get_conversation_tail"):
            last_node = self.engine._get_conversation_tail(conversation_id)
        else:
            last_node = None
        if last_node is None:
            raise Exception('unreachable')
        # Tool call node (assistant role)
        call_node_content = f"Calling tool {tool_name}"
        call_id = str(self.tool_call_id_factory(tool_name, user_id, conversation_id, last_node.safe_get_id(), call_node_content,  str(args), str(kwargs)))
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
            metadata={"entity_type": "tool_call", "tool_name": tool_name,
                      "level_from_root": 0, 
                      "in_conversation_chain": True},
            domain_id=None,
            canonical_entity_id=None,
        )
        self.engine.add_node(call_node, None)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary+= len(call_node_content)
        prev_turn_meta_summary.prev_node_distance_from_last_summary+= 1
        if orchestrator:
            self.join_tool_node_to_turn(orchestrator, conversation_id, call_node, turn_node_id, prev_turn_meta_summary)
        # Execute
        try:
            result: BaseToolResult = handler(**kwargs)
        except Exception as _e:
            result: BaseToolResult = handler(**kwargs)
        if result:
          if tn := result.node_id_entry: # only tool that craeted nodes can be wrapped with node and edge linkages, some simple tools would not create and not use this code path
            n: ConversationNode = self.engine.get_nodes([tn], node_type = ConversationNode)[0]
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
            e = ConversationEdge(id = eid, type = 'relationship', summary = f"call node {call_node.safe_get_id()} created entry point {n.safe_get_id()}",
                                 domain_id=None, label='craeted entry point', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[call_node.safe_get_id()], 
                                 target_ids=[n.safe_get_id()],
                                 relation="run_result", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"conv:{conversation_id}",
                                 metadata={"relation":"tool_call create node",
                                           "source_id":[call_node.safe_get_id()],
                                           "target_id":[n.safe_get_id()],
                                           "char_distance_from_last_summary": 0,
                                           "turn_distance_from_last_summary": 0,
                                           "entity_type": "tool-call->details",
                                           "in_conversation": False,
                                        
                                           },
                                 )
            self.engine.add_edge(e)
        # Tool result node (tool role)
        # res_id = str(self.tool_call_id_factory())
        # last_node = self.engine.conversation.get_conversation_tail(conversation_id)
        if last_node is None:
            raise Exception('unreachable')
        res_id = str(self.tool_call_id_factory(user_id, conversation_id, call_node.safe_get_id(), "tool_result"))
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
        
        prev_turn_meta_summary.tail_turn_index += 1
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
                      "level_from_root": 0, 
                      "in_conversation_chain": True},
            domain_id=None,
            canonical_entity_id=None,
        )
        self.engine.add_node(res_node, None)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary+= len(res_node_content)
        prev_turn_meta_summary.prev_node_distance_from_last_summary+= 1
        
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
        eid = f"tool_call_result|caller:{call_node.safe_get_id()}|created:{res_node.safe_get_id()}"
        e = ConversationEdge(id = eid, type = 'relationship', summary = f"call node {call_node.safe_get_id()} result node:{res_node.safe_get_id()}",
                                 domain_id=None, label='tool result', 
                                 properties={}, 
                                 mentions=[Grounding(spans=[self_span])], canonical_entity_id=None, 
                                 source_ids=[call_node.safe_get_id()], 
                                 target_ids=[res_node.safe_get_id()],
                                 relation="run_result", source_edge_ids = [], target_edge_ids = [], embedding = None,
                                 doc_id = f"conv:{conversation_id}",
                                 metadata={"relation":"tool_call create node",
                                           "source_id":[call_node.safe_get_id()],
                                           "target_id":[res_node.safe_get_id()],
                                           "char_distance_from_last_summary": 0,
                                           "turn_distance_from_last_summary": 0,
                                           "entity_type": "tool-call->tool-result",
                                           "in_conversation": False,
                                        
                                           },
                                 )
        self.engine.add_edge(e)
        return result, call_node.safe_get_id()

