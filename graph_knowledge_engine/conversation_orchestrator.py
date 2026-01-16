"""Conversation orchestrator (KGE-native, LangGraph-free).

This module owns *policy + control flow* for conversation turns, while delegating
all persistence/mutation to the conversation engine.

It is intentionally lightweight and uses your existing retrievers/agents.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from server_mcp import GraphKnowledgeEngine

from .models import (
    ConversationAIResponse,
    ConversationEdge,
    ConversationNode,
    Grounding,
    KnowledgeRetrievalResult,
    MentionVerification,
    Span,
    Role,
    AddTurnResult
)
from .tool_runner import ToolRunner
from .memory_retriever import MemoryRetriever, MemoryPinResult, MemoryRetrievalResult
from .knowledge_retriever import KnowledgeRetriever


@dataclass
class OrchestratorState:
    conversation_id: str
    user_id: str
    mem_id: str
    user_text: str
    role: str
    ref_knowledge_engine: Any

    # created turn
    current_turn_node_id: str | None = None
    turn_index: int | None = None
    self_span: Span | None = None
    embedding: List[float] | None = None

    # retrieval artifacts
    memory: MemoryRetrievalResult | None = None
    knowledge: KnowledgeRetrievalResult | None = None
    memory_pin: MemoryPinResult | None = None
    pinned_kg_pointer_node_ids: List[str] = field(default_factory=list)
    pinned_kg_edge_ids: List[str] = field(default_factory=list)

    # answer
    answer: ConversationAIResponse | None = None
from graph_knowledge_engine.models import MetaFromLastSummary, RetrievalResult
from graph_knowledge_engine.workflow.executor import WorkflowExecutor
from graph_knowledge_engine.workflow.contract import build_workflow_from_engine, WorkflowSpec
class ConversationOrchestrator:
    """KGE-native orchestrator.

    Minimal, surgical refactor:
    - engine.add_conversation_turn(...) stays stable
    - orchestration steps live here
    - tool-call/tool-result events are recorded in the conversation graph
    """
    # def add_conversation_turn_workflow(
    #     self,
    #     *,
    #     user_id: str,
    #     conversation_id: str,
    #     turn_id: str,
    #     mem_id: str,
    #     role: Role,
    #     content: str,
    #     filtering_callback: Callable[..., tuple[RetrievalResult, str]],
    #     workflow_id: str,
    #     max_retrieval_level: int = 2,
    #     summary_char_threshold: int = 12000,
    #     in_conv: bool = True,
    #     prev_turn_meta_summary: MetaFromLastSummary | None = None,
    # ) -> AddTurnResult:
    #     """
    #     Workflow-driven orchestration path.

    #     - STATIC graph shape: workflow nodes/edges are already registered in engine store.
    #     - DYNAMIC routing: by predicate registry and step results.
    #     - Parallelism + shared message queue: handled by WorkflowExecutor.
    #     - Step implementations: resolved by op name -> callable (no monolithic workflow fn).

    #     Legacy add_conversation_turn remains unchanged and continues to be the default path.
    #     """
    #     if prev_turn_meta_summary is None:
    #         prev_turn_meta_summary = MetaFromLastSummary()

    #     # Compute embedding exactly like legacy path
    #     emb_text0 = f"{role}: {content}"
    #     embedding = self.conversation_engine._iterative_defensive_emb(emb_text0)
    #     if embedding is None:
    #         raise RuntimeError("uncalculatable embeddings")

    #     # Create/persist the turn node (mirror of legacy behavior, minimal)
    #     prev_node = self.conversation_engine._get_conversation_tail(conversation_id) if in_conv else None
    #     new_index = (prev_node.turn_index or 0) + 1 if prev_node is not None else 0

    #     turn_node_id = turn_id or str(uuid.uuid4())
    #     self_span = Span(
    #         collection_page_url=f"conversation/{conversation_id}",
    #         document_page_url=f"conversation/{conversation_id}#{turn_node_id}",
    #         doc_id=f"conv:{conversation_id}",
    #         insertion_method="conversation_turn",
    #         page_number=1,
    #         start_char=0,
    #         end_char=len(content),
    #         excerpt=content,
    #         context_before="",
    #         context_after="",
    #         chunk_id=None,
    #         source_cluster_id=None,
    #         verification=MentionVerification(
    #             method="human",
    #             is_verified=True,
    #             score=1.0,
    #             notes=f"verbatim {role} input",
    #         ),
    #     )

    #     turn_node = ConversationNode(
    #         user_id=user_id,
    #         id=turn_node_id,
    #         label=f"Turn {new_index} ({role})",
    #         type="entity",
    #         doc_id=turn_node_id,
    #         summary=content,
    #         role=role,  # type: ignore
    #         turn_index=new_index,
    #         conversation_id=conversation_id,
    #         mentions=[Grounding(spans=[self_span])],
    #         properties={},
    #         metadata={
    #             "entity_type": "conversation_turn",
    #             "level_from_root": 0,
    #             "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
    #             "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
    #         },
    #         domain_id=None,
    #         canonical_entity_id=None,
    #     )
    #     turn_node.embedding = embedding
    #     self.conversation_engine.add_node(turn_node)
    #     self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span)

    #     # Load workflow spec from engine via convention (wf_start=True)
    #     wf_spec: WorkflowSpec = build_workflow_from_engine(engine=self.conversation_engine, workflow_id=workflow_id)

    #     # Predicate registry for routing
    #     # (you can expand this set, but keep them symbolic)
    #     predicate_registry = {
    #         "has_memory": lambda st, r: st.get("memory") is not None,
    #         "has_kg": lambda st, r: st.get("kg") is not None,
    #     }

    #     # Step resolver: op-name -> callable(ctx)->result
    #     def resolve_step(op_name: str):
    #         if op_name == "memory_retrieve":
    #             def _run(ctx):
    #                 mem_retriever = MemoryRetriever(
    #                     conversation_engine=self.conversation_engine,
    #                     llm=self.llm,
    #                     filtering_callback=filtering_callback,
    #                 )
    #                 mem = self.tool_runner.run_tool(
    #                     conversation_id=conversation_id,
    #                     user_id=user_id,
    #                     turn_node_id=turn_node_id,
    #                     turn_index=new_index,
    #                     tool_name="memory_retrieve",
    #                     args={"n_results": mem_retriever.n_results},
    #                     handler=lambda: mem_retriever.retrieve(
    #                         user_id=user_id,
    #                         current_conversation_id=conversation_id,
    #                         query_embedding=embedding,
    #                         user_text=content,
    #                         context_text="",
    #                     ),
    #                     render_result=lambda r: getattr(r, "reasoning", "")[:800],
    #                     prev_turn_meta_summary=prev_turn_meta_summary,
    #                 )
    #                 # also publish to message queue for parallel observers if needed
    #                 ctx.publish({"type": "memory_done"})
    #                 return mem
    #             return _run

    #         if op_name == "kg_retrieve":
    #             def _run(ctx):
    #                 kg_retriever = KnowledgeRetriever(
    #                     conversation_engine=self.conversation_engine,
    #                     ref_knowledge_engine=self.ref_knowledge_engine,
    #                     llm=self.llm,
    #                     filtering_callback=filtering_callback,
    #                     max_retrieval_level=max_retrieval_level,
    #                 )
    #                 mem = ctx.state.get("memory")
    #                 kg = self.tool_runner.run_tool(
    #                     conversation_id=conversation_id,
    #                     user_id=user_id,
    #                     turn_node_id=turn_node_id,
    #                     turn_index=new_index,
    #                     tool_name="kg_retrieve",
    #                     args={
    #                         "max_retrieval_level": max_retrieval_level,
    #                         "seed_kg_node_ids": list(getattr(mem, "seed_kg_node_ids", []) or []),
    #                     },
    #                     handler=lambda: kg_retriever.retrieve(
    #                         user_id=user_id,
    #                         current_conversation_id=conversation_id,
    #                         query_embedding=embedding,
    #                         user_text=content,
    #                         context_text="",
    #                         seed_kg_node_ids=list(getattr(mem, "seed_kg_node_ids", []) or []),
    #                         self_span=self_span,
    #                         selected_memory=getattr(mem, "selected", None),
    #                         memory_context_text=getattr(mem, "memory_context_text", ""),
    #                         prev_turn_meta_summary=prev_turn_meta_summary,
    #                     ),
    #                     render_result=lambda r: getattr(r, "reasoning", "")[:800],
    #                     prev_turn_meta_summary=prev_turn_meta_summary,
    #                 )
    #                 ctx.publish({"type": "kg_done"})
    #                 return kg
    #             return _run

    #         if op_name == "answer_only":
    #             def _run(ctx):
    #                 # allow think/retry to observe crawl/tool updates via message queue
    #                 _msgs = ctx.drain_messages()
    #                 ans = self.answer_only(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)
    #                 return ans
    #             return _run

    #         raise KeyError(f"Unknown workflow op: {op_name}")

    #     executor = WorkflowExecutor(
    #         engine=self.conversation_engine,
    #         workflow=wf_spec,
    #         step_resolver=resolve_step,
    #         predicate_registry=predicate_registry,
    #         cache_root=".workflow_cache",
    #         max_workers=4,
    #     )

    #     init_state = {
    #         "conversation_id": conversation_id,
    #         "user_id": user_id,
    #         "turn_node_id": turn_node_id,
    #         "turn_index": new_index,
    #         "role": role,
    #         "user_text": content,
    #         "embedding": embedding,
    #         "memory": None,
    #         "kg": None,
    #         "answer": None,
    #     }

    #     for _ev in executor.run(run_id=turn_node_id, initial_state=init_state):
    #         # If you want streaming, yield these from a streaming API variant.
    #         pass

    #     response = executor.state.get("answer")
    #     response_turn_node_id = getattr(response, "response_node_id", None) if response is not None else None

    #     return AddTurnResult(
    #         user_turn_node_id=turn_node_id,
    #         response_turn_node_id=response_turn_node_id,
    #         turn_index=new_index,
    #         prev_turn_meta_summary=prev_turn_meta_summary,
    #     )
    def add_conversation_turn_workflow_v2(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: Role,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        workflow_id: str,
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        in_conv: bool = True,
        prev_turn_meta_summary: MetaFromLastSummary | None = None,
    ) -> AddTurnResult:
        """
        V2 workflow-driven orchestration.

        Design engine: workflow_engine (kg_graph_type="workflow")
        Trace/checkpoints: conversation_engine (kg_graph_type="conversation")
        """
        if prev_turn_meta_summary is None:
            prev_turn_meta_summary = MetaFromLastSummary(0, 0)

        # ---- 1) Compute embedding exactly like legacy ----
        emb_text0 = f"{role}: {content}"
        embedding = self.conversation_engine._iterative_defensive_emb(emb_text0)
        if embedding is None:
            raise RuntimeError("uncalculatable embeddings")

        # ---- 2) Create/persist the turn node (mirror legacy) ----
        prev_node = self.conversation_engine._get_conversation_tail(conversation_id) if in_conv else None
        new_index = (prev_node.turn_index or 0) + 1 if prev_node is not None else 0

        turn_node_id = turn_id or str(uuid.uuid4())
        self_span = Span(
            collection_page_url=f"conversation/{conversation_id}",
            document_page_url=f"conversation/{conversation_id}#{turn_node_id}",
            doc_id=f"conv:{conversation_id}",
            insertion_method="conversation_turn",
            page_number=1,
            start_char=0,
            end_char=len(content),
            excerpt=content,
            context_before="",
            context_after="",
            chunk_id=None,
            source_cluster_id=None,
            verification=MentionVerification(
                method="human",
                is_verified=True,
                score=1.0,
                notes=f"verbatim {role} input",
            ),
        )

        turn_node = ConversationNode(
            user_id=user_id,
            id=turn_node_id,
            label=f"Turn {new_index} ({role})",
            type="entity",
            doc_id=turn_node_id,
            summary=content,
            role=role,  # type: ignore
            turn_index=new_index,
            conversation_id=conversation_id,
            mentions=[Grounding(spans=[self_span])],
            properties={},
            metadata={
                "entity_type": "conversation_turn",
                "level_from_root": 0,
                "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                "in_conversation_chain": in_conv,
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        turn_node.embedding = embedding
        self.conversation_engine.add_node(turn_node)
        if prev_node is not None:
            self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span)

        # ---- 3) Load static workflow design from workflow_engine ----
        from .workflow.design import build_workflow_from_engine
        from .workflow.executor import WorkflowExecutor

        wf_spec = build_workflow_from_engine(engine=self.workflow_engine, workflow_id=workflow_id)

        # ---- 4) Predicate registry for dynamic routing ----
        # Keep symbolic, stable names. Expand as needed.
        predicate_registry = {
            "has_memory": lambda st, r: st.get("memory") is not None,
            "has_kg": lambda st, r: st.get("kg") is not None,
            "always": lambda st, r: True,
        }

        # ---- 5) Step resolver: op-name -> callable(ctx)->serializable result ----
        def resolve_step(op_name: str):
            if op_name == "memory_retrieve":
                def _run(ctx):
                    mem_retriever = MemoryRetriever(
                        conversation_engine=self.conversation_engine,
                        llm=self.llm,
                        filtering_callback=filtering_callback,
                    )
                    mem = self.tool_runner.run_tool(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        turn_node_id=turn_node_id,
                        turn_index=new_index,
                        tool_name="memory_retrieve",
                        args={"n_results": mem_retriever.n_results},
                        handler=lambda: mem_retriever.retrieve(
                            user_id=user_id,
                            current_conversation_id=conversation_id,
                            query_embedding=embedding,
                            user_text=content,
                            context_text="",
                        ),
                        render_result=lambda r: getattr(r, "reasoning", "")[:800],
                        prev_turn_meta_summary=prev_turn_meta_summary,
                    )
                    ctx.publish({"type": "memory_done"})
                    # IMPORTANT: return raw object; executor will convert to JSONable
                    return mem
                return _run

            if op_name == "kg_retrieve":
                def _run(ctx):
                    kg_retriever = KnowledgeRetriever(
                        conversation_engine=self.conversation_engine,
                        ref_knowledge_engine=self.ref_knowledge_engine,
                        llm=self.llm,
                        filtering_callback=filtering_callback,
                        max_retrieval_level=max_retrieval_level,
                    )
                    mem_obj = ctx.state.get("memory")
                    seed_ids = list(getattr(mem_obj, "seed_kg_node_ids", []) or [])
                    kg = self.tool_runner.run_tool(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        turn_node_id=turn_node_id,
                        turn_index=new_index,
                        tool_name="kg_retrieve",
                        args={"max_retrieval_level": max_retrieval_level, "seed_kg_node_ids": seed_ids},
                        handler=lambda: kg_retriever.retrieve(
                            user_id=user_id,
                            current_conversation_id=conversation_id,
                            query_embedding=embedding,
                            user_text=content,
                            context_text="",
                            seed_kg_node_ids=seed_ids,
                            self_span=self_span,
                            selected_memory=getattr(mem_obj, "selected", None),
                            memory_context_text=getattr(mem_obj, "memory_context_text", ""),
                            prev_turn_meta_summary=prev_turn_meta_summary,
                        ),
                        render_result=lambda r: getattr(r, "reasoning", "")[:800],
                        prev_turn_meta_summary=prev_turn_meta_summary,
                    )
                    ctx.publish({"type": "kg_done"})
                    return kg
                return _run

            if op_name == "answer_only":
                def _run(ctx):
                    _msgs = ctx.drain_messages()
                    ans = self.answer_only(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)
                    # include msg count for debugging deterministically
                    return {"answer": ans, "observed_msgs": len(_msgs)}
                return _run

            raise KeyError(f"Unknown workflow op: {op_name}")

        # ---- 6) Trace sink: persist workflow step events/checkpoints into conversation_engine ----
        # Keep minimal: one node per event if you want it in D3.
        def trace_sink(ev):
            # You can upgrade this to real WorkflowRun/StepExec/Checkpoint node types.
            # For now, persist as lightweight conversation nodes to ensure visibility.
            try:
                et = ev.type
                payload = ev.payload
                txt = f"[wf:{workflow_id}] {et} {payload.get('op','')}"
                nid = f"wftrace|{turn_node_id}|{payload.get('step_seq', et)}|{uuid.uuid4()}"
                span = self_span
                n = ConversationNode(
                    user_id=user_id,
                    id=nid,
                    label=f"WF {et}",
                    type="entity",
                    doc_id=nid,
                    summary=txt,
                    role="system",  # type: ignore
                    turn_index=new_index,
                    conversation_id=conversation_id,
                    mentions=[Grounding(spans=[span])],
                    properties={},
                    metadata={
                        "entity_type": "workflow_trace",
                        "workflow_id": workflow_id,
                        "wf_event_type": et,
                        "payload": payload,  # already JSONable
                        "level_from_root": 0,
                    },
                    domain_id=None,
                    canonical_entity_id=None,
                )
                self.conversation_engine.add_node(n)
            except Exception:
                # tracing must never break the workflow
                return

        executor = WorkflowExecutor(
            engine=self.workflow_engine,
            workflow=wf_spec,
            step_resolver=resolve_step,
            predicate_registry=predicate_registry,
            max_workers=4,
            checkpoint_every_n_steps=1,
            trace_sink=trace_sink,
        )

        # ---- 7) Initial state: keep both raw objects + JSONable mirrors ----
        init_state = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "turn_node_id": turn_node_id,
            "turn_index": new_index,
            "role": str(role),
            "user_text": content,
            "embedding": embedding,
            "memory": None,
            "kg": None,
            "answer": None,
        }

        # Run executor; it updates executor.state with JSONable results
        for _ev in executor.run(run_id=turn_node_id, initial_state=init_state):
            # optional: stream events via another API
            pass

        # ---- 8) Map final result into legacy AddTurnResult ----
        # If your answer op returns {"answer": ConversationAIResponse, ...}
        ans_obj = executor.state.get("result.answer_only")
        response_turn_node_id = None

        # If you want to persist a real assistant turn node here, do it like legacy:
        # (depends on how your agent already persists answer nodes)
        if isinstance(ans_obj, dict):
            maybe = ans_obj.get("answer")
            if hasattr(maybe, "response_node_id"):
                response_turn_node_id = getattr(maybe, "response_node_id")

        return AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=response_turn_node_id,
            turn_index=new_index,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )        
    def __init__(
        self,
        *,
        conversation_engine: Any,
        ref_knowledge_engine: Any,
        workflow_engine: Any | None = None,
        tool_call_id_factory = uuid.uuid4,
        llm: BaseChatModel,
    ) -> None:
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.ref_knowledge_engine: GraphKnowledgeEngine = ref_knowledge_engine
        self.workflow_engine: GraphKnowledgeEngine | None = workflow_engine
        self.llm = llm
        self.tool_runner = ToolRunner(conversation_engine=conversation_engine,
                                      tool_call_id_factory=tool_call_id_factory)
    def add_link_to_new_turn(self, turn_node, prev_node, conversation_id, span):
        
            seq_edge = ConversationEdge(
                id=str(uuid.uuid4()),
                source_ids=[prev_node.id],
                target_ids=[turn_node.id],
                relation="next_turn",
                label="next_turn",
                type="relationship",
                summary="Sequential flow",
                doc_id=f"conv:{conversation_id}",
                mentions=[Grounding(spans=[span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity_type": "conversation_edge"},
                embedding=None,
                metadata={},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(seq_edge)     
    # ----------------------------
    # Public entry points
    # ----------------------------
    def add_conversation_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: Role,
        content: str,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        max_retrieval_level: int = 2,
        summary_char_threshold: int = 12000,
        in_conv: bool = True,
        prev_turn_meta_summary : MetaFromLastSummary | None = None
    ) -> AddTurnResult:
        """Ingest a user/assistant turn and run retrieval+answering policy."""

        if self.conversation_engine.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        if self.ref_knowledge_engine is None:
            raise Exception("GraphKnowledgeEngine must be specified")

        st = OrchestratorState(
            conversation_id=conversation_id,
            user_id=user_id,
            mem_id=mem_id,
            user_text=content,
            role=role,
            ref_knowledge_engine=self.ref_knowledge_engine,
        )
        if prev_turn_meta_summary is None:
            prev_turn_meta_summary = MetaFromLastSummary(0,0)
        prev_node = self.conversation_engine._get_conversation_tail(conversation_id)
        if prev_node is not None:
            new_index = (prev_node.turn_index + 1) if prev_node.turn_index is not None else 0
            # prev_turn_meta_summary.prev_node_char_distance_from_last_summary = prev_node.metadata.get("char_distance_from_last_summary") or 0
            node_last_char_dist = prev_node.metadata.get("char_distance_from_last_summary")
            if node_last_char_dist is not None:
                node_last_char_dist += len(prev_node.summary)
            else:
                node_last_char_dist = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = node_last_char_dist
            #======================
            node_last_turn_dist = prev_node.metadata.get("turn_distance_from_last_summary")
            if node_last_turn_dist is not None:
                node_last_turn_dist += 1
            else:
                node_last_turn_dist = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = node_last_turn_dist #prev_node.metadata.get("turn_distance_from_last_summary") or 0
        else:
            new_index = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = -1
        # workflow_to_run = get_workflow(states)
        # for step_indication in workflow_to_run:
        #     step_result = self.execute_step(step_indication)
        #     workflow_to_run.send(step_result)
        # result = workflow_to_run.get_result()
        response = None
        if role in ["assistent", "system"]:
            
            turn_node_id = turn_id or str(uuid.uuid4())
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{turn_node_id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="conversation_turn",
                page_number=1,
                start_char=0,
                end_char=len(content),
                excerpt=content,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="human",
                    is_verified=True,
                    score=1.0,
                    notes=f"verbatim {role} input",
                ),
            )            
            turn_node = ConversationNode(
                user_id=user_id,
                id=turn_node_id,
                label=f"Turn {new_index} ({role})",
                type="entity",
                doc_id=turn_node_id,
                summary=content,
                role=role,  # type: ignore
                turn_index=new_index,
                conversation_id=conversation_id,
                mentions=[Grounding(spans=[self_span])],
                properties={},
                metadata={
                    "entity_type": "conversation_turn",
                    "level_from_root": 0,
                    "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    "in_conversation_chain": in_conv
                },
                domain_id=None,
                canonical_entity_id=None,
            )
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(content)
            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
            self.conversation_engine.add_node(turn_node)
            self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span)
            prev_node = turn_node
            
                      
            add_turn_result = AddTurnResult(
                user_turn_node_id= turn_node_id,
                response_turn_node_id=None,# response_turn_node_id,
                turn_index=new_index,
                relevant_kg_node_ids=[],
                relevant_kg_edge_ids=[],
                pinned_kg_pointer_node_ids=[],
                pinned_kg_edge_ids=[],
                memory_context_node_id=None,
                memory_context_edge_ids=[],
                prev_turn_meta_summary=prev_turn_meta_summary
            )
            new_index += 1
            # return add_turn_result
        else:
            # 1) Append the conversation turn node
            turn_node_id = turn_id or str(uuid.uuid4())
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{turn_node_id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="conversation_turn",
                page_number=1,
                start_char=0,
                end_char=len(content),
                excerpt=content,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="human",
                    is_verified=True,
                    score=1.0,
                    notes="verbatim user/system input",
                ),
            )
            
            turn_node = ConversationNode(
                user_id=user_id,
                id=turn_node_id,
                label=f"Turn {new_index} ({role})",
                type="entity",
                doc_id=turn_node_id,
                summary=content,
                role=role,  # type: ignore
                turn_index=new_index,
                conversation_id=conversation_id,
                mentions=[Grounding(spans=[self_span])],
                properties={},
                metadata={
                    "entity_type": "conversation_turn",
                    "level_from_root": 0,
                    "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    "in_conversation_chain": in_conv,
                },
                domain_id=None,
                canonical_entity_id=None,
            )
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(content)
            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
            new_index += 1
            emb_text0 = f"{role}: {content}"
            embedding = self.conversation_engine._iterative_defensive_emb(emb_text0)
            if embedding is None:
                raise Exception("uncalculatable embeddings")
            turn_node.embedding = embedding
            self.conversation_engine.add_node(turn_node, None)

            st.current_turn_node_id = turn_node_id
            st.turn_index = new_index
            st.self_span = self_span
            st.embedding = embedding

            # 2) Link sequentially
            if prev_node:
                seq_edge = ConversationEdge(
                    id=str(uuid.uuid4()),
                    source_ids=[prev_node.id],
                    target_ids=[turn_node_id],
                    relation="next_turn",
                    label="next_turn",
                    type="relationship",
                    summary="Sequential flow",
                    doc_id=f"conv:{conversation_id}",
                    mentions=[Grounding(spans=[self_span])],
                    domain_id=None,
                    canonical_entity_id=None,
                    properties={"entity_type": "conversation_edge"},
                    embedding=None,
                    metadata={},
                    source_edge_ids=[],
                    target_edge_ids=[],
                )
                self.conversation_engine.add_edge(seq_edge)
                prev_node = turn_node
            # 3) Retrieval + pinning (recorded as tool events)
            mem_retriever = MemoryRetriever(
                conversation_engine=self.conversation_engine,
                llm=self.llm,
                filtering_callback=filtering_callback,
            )
            kg_retriever = KnowledgeRetriever(
                conversation_engine=self.conversation_engine,
                ref_knowledge_engine=self.ref_knowledge_engine,
                llm=self.llm,
                filtering_callback=filtering_callback,
                max_retrieval_level=max_retrieval_level,
            )

            # memory retrieve tool
            mem: MemoryRetrievalResult = self.tool_runner.run_tool(
                conversation_id=conversation_id,
                user_id=user_id,
                turn_node_id=turn_node_id,
                turn_index=new_index,
                tool_name="memory_retrieve",
                args={
                    "n_results": mem_retriever.n_results,
                },
                handler=lambda: mem_retriever.retrieve(
                    user_id=user_id,
                    current_conversation_id=conversation_id,
                    query_embedding=embedding,
                    user_text=content,
                    context_text="",
                ),
                render_result=lambda r: getattr(r, "reasoning", "")[:800],
                prev_turn_meta_summary=prev_turn_meta_summary
            )
            st.memory = mem

            # KG retrieve tool
            kg: KnowledgeRetrievalResult = self.tool_runner.run_tool(
                conversation_id=conversation_id,
                user_id=user_id,
                turn_node_id=turn_node_id,
                turn_index=new_index,
                tool_name="kg_retrieve",
                args={
                    "max_retrieval_level": max_retrieval_level,
                    "seed_kg_node_ids": list(getattr(mem, "seed_kg_node_ids", []) or []),
                },
                handler=lambda: kg_retriever.retrieve(
                    user_text=content,
                    context_text="",
                    query_embedding=embedding,
                    seed_kg_node_ids=list(getattr(mem, "seed_kg_node_ids", []) or []),
                ),
                render_result=lambda r: getattr(r, "reasoning", "")[:800],
                prev_turn_meta_summary=prev_turn_meta_summary
            )
            st.knowledge = kg

            # pin memory (not a tool call; it's a graph mutation derived from the tool outputs)
            memory_pin: Optional[MemoryPinResult] = None
            if mem.selected and mem.memory_context_text:
                memory_pin = mem_retriever.pin_selected(
                    user_id=user_id,
                    current_conversation_id=conversation_id,
                    turn_node_id=turn_node_id,
                    mem_id=mem_id,
                    turn_index=new_index,
                    self_span=self_span,
                    selected_memory=mem.selected,
                    memory_context_text=mem.memory_context_text,
                    prev_turn_meta_summary=prev_turn_meta_summary
                )
            pinned_ptrs = []
            pinned_edges = []
            st.memory_pin = memory_pin
            if kg.selected:
                pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn_node_id=turn_node_id,
                    turn_index=new_index,
                    self_span=self_span,
                    selected_knowledge=kg.selected,
                    prev_turn_meta_summary=prev_turn_meta_summary,
                )
                st.pinned_kg_pointer_node_ids = pinned_ptrs
                st.pinned_kg_edge_ids = pinned_edges

            # 4) Answer (answer-only facade); response persistence happens inside the agent today.
            response = self.answer_only(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)
            st.answer = response
            response_turn_node_id = None
            if response.response_node_id is not None:
                response_turn_node_id = response.response_node_id
                turn_node = self.conversation_engine.get_nodes([response.response_node_id])[0]
                if prev_node:

                    self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span)
                    prev_node = turn_node
                    
                else:
                    raise Exception("An AI turn must at least have a predecessor turn.")
            add_turn_result = AddTurnResult(
                        user_turn_node_id=turn_node_id,
                        response_turn_node_id=response_turn_node_id,
                        turn_index=new_index,
                        relevant_kg_node_ids=[i for i in (kg.selected.node_ids if kg.selected else [])],
                        relevant_kg_edge_ids=[i for i in (kg.selected.edge_ids if kg.selected else [])],
                        pinned_kg_pointer_node_ids=pinned_ptrs,
                        pinned_kg_edge_ids=pinned_edges,
                        memory_context_node_id=memory_pin.memory_context_node.id if memory_pin else None,
                        memory_context_edge_ids=[i.id for i in memory_pin.pinned_edges] if memory_pin else [],
                        prev_turn_meta_summary = prev_turn_meta_summary
                    )
        
        # 5) Summarization trigger policy remains here; implementation stays in engine.
        
        if new_index > 0 and (
            prev_turn_meta_summary.prev_node_distance_from_last_summary - 5 >= 0
            or prev_turn_meta_summary.prev_node_char_distance_from_last_summary > summary_char_threshold
            or (response and bool(getattr(response, "llm_decision_need_summary", False)))
        ):
            added_id = self._summarize_conversation_batch(conversation_id, new_index)
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary=0
        return add_turn_result

    # @conversation_only
    def _summarize_conversation_batch(self, conversation_id: str, current_index: int, batch_size: int = 5, in_conv=True ):
        #in_conversation  = False if side car
        if not in_conv:
            current_index -=1
        if in_conv:
            prev_node = self.conversation_engine._get_conversation_tail(conversation_id)
        else:
            prev_node = None
        """Summarize the last N turns into a Summary/ Memory Node."""
        if self.conversation_engine.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        start_index = max(0, current_index - batch_size + 1)
        
        # Fetch the nodes
        # In a real impl, better querying needed.
        # We assume we can get them by index logic or linear scan
        # For prototype, scan and filter
        all_nodes = self.conversation_engine.get_nodes(where={"conversation_id": conversation_id}, node_type=ConversationNode)
        batch_ids = []
        batch_text = []
        provenance_spans = []
        if all_nodes is None:
            raise Exception("Unreacheable")
        # all_nodes_doc : list[str] | None= all_nodes['documents']
        # all_nodes_meta = all_nodes['metadatas']
        # all_nodes_id : list[str] | None= all_nodes['ids']
        # if all_nodes_doc is None or all_nodes_meta is None or all_nodes_id is None:
        #     raise Exception("documents metadatas and ids all needed")
        nodes: list[ConversationNode] = all_nodes #self.conversation_engine.get_nodes(all_nodes["ids"], ConversationNode)
        # for doc, meta, nid in zip(all_nodes_doc, all_nodes_meta, all_nodes_id):
            # turn_index = meta.get('turn_index')
            # self.nodes_from_single_or_id_query_result(all_nodes)
            # self.nodes_from_query_result
        for n in nodes:
            turn_index = n.turn_index
            if turn_index is not None and start_index <= int(turn_index) <= current_index:
                # Check type
                # d = json.loads(doc)
                # d.update(meta)
                # n = ConversationNode.model_validate(d)
                if n.metadata.get("entity_type") == "conversation_turn":
                    batch_ids.append(n.id)
                    batch_text.append(f"{n.role}: {n.summary}")
                    # Collect provenance: The turn itself is the source
                    if n.mentions:
                        provenance_spans.extend(n.mentions[0].spans)
                elif n.type == "reference_pointer":
                    batch_ids.append(n.id)
                    batch_text.append(f"{n.role}: {n.summary}")
                    # Include referenced knowledge node in provenance too (as requested by user)
                    if n.mentions:
                         provenance_spans.extend(n.mentions[0].spans)

        if not batch_ids:
            return
        
        full_text = "\n".join(batch_text)

        # LLM Summarize
        @self.conversation_engine.memory.cache
        def get_summary(full_text):
            from langchain_core.prompts import ChatPromptTemplate
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize this conversation segment into a concise memory."),
                ("human", full_text)
            ])
            
            summary_res = (summary_prompt | self.llm).invoke({})
            summary_content = summary_res.content
            return summary_content
        summary_content = get_summary(full_text)
        # Dedupe spans for provenance
        unique_spans = []
        seen = set()
        for s in provenance_spans:
            # key based on loc
            k = (s.doc_id, s.start_char, s.end_char)
            if k not in seen:
                seen.add(k)
                unique_spans.append(s)
        import json
        # Create Summary Node
        summary_node = ConversationNode(
            id=str(uuid.uuid4()),
            label=f"Summary {start_index}-{current_index}",
            type="entity",
            summary='\n'.join(i['text'] for i in get_summary(full_text)), # type: ignore
            role="system", # type: ignore
            conversation_id=conversation_id,
            turn_index=current_index, # Anchored at end of batch
            # Provenance: We link back to the source turns and knowledge refs
            mentions=[Grounding(spans=unique_spans)] if unique_spans else [],
            
            properties={"content": json.dumps(summary_content)},
            metadata={"level_from_root": 1, 
                      "entity_type": "conversation_summary", 
                      "char_distance_from_last_summary": 0, 
                      "turn_distance_from_last_summary" : 0 ,
                      "in_conversation_chain": in_conv}, # Summary is higher level?
            domain_id=None,
            canonical_entity_id=None
        )
        if self.conversation_engine.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        # Persist
        
        self.conversation_engine.add_node(summary_node)


        # Edges: Summary -> Turns
        sum_edge = ConversationEdge(
            id=str(uuid.uuid4()),
            source_ids=[summary_node.id],
            target_ids=batch_ids,
            relation="summarizes",
            label="summarizes",
            type="relationship",
            summary="Memory summarization",
            doc_id=f"conv:{conversation_id}",
            domain_id=None,
            canonical_entity_id=None,
            properties=None,
            embedding=None,
            mentions=[Grounding(spans=[
                Span(
                    collection_page_url=f"conversation/{conversation_id}",
                    document_page_url=f"conversation/{conversation_id}",
                    doc_id=f"conv:{conversation_id}",
                    chunk_id = None,
                    source_cluster_id = None,
                    insertion_method="summarization_link",
                    page_number=1, start_char=0, end_char=1,
                    excerpt="summary link", context_before="", context_after="",
                    verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="link")
                )
            ])],
            metadata={},
            source_edge_ids=[],
            target_edge_ids=[]
        )
        self.conversation_engine.add_edge(sum_edge)
        
        if in_conv:
            if prev_node is None:
                raise Exception("unreacheable")
            # span = Span(
            #         collection_page_url=f"conversation/{conversation_id}",
            #         document_page_url=f"conversation/{conversation_id}",
            #         doc_id=f"conv:{conversation_id}",
            #         chunk_id = None,
            #         source_cluster_id = None,
            #         insertion_method="summarization_link",
            #         page_number=1, start_char=0, end_char=1,
            #         excerpt="summary link", context_before="", context_after="",
            #         verification=MentionVerification(method="system", is_verified=True, score=1.0, notes="link")
            #     )
            self_span = Span(
                collection_page_url=f"conversation/{conversation_id}",
                document_page_url=f"conversation/{conversation_id}#{summary_node.id}",
                doc_id=f"conv:{conversation_id}",
                insertion_method="summary_turn",
                page_number=1,
                start_char=0,
                end_char=len(summary_node.summary),
                excerpt=summary_node.summary,
                context_before="",
                context_after="",
                chunk_id=None,
                source_cluster_id=None,
                verification=MentionVerification(
                    method="human",
                    is_verified=True,
                    score=1.0,
                    notes=f"system_summary_turn",
                ),
            )    
            self.add_link_to_new_turn(summary_node, prev_node, conversation_id, self_span)
        return summary_node.id
    def answer_only(self, *, conversation_id: str, model_names: Optional[list[str]] = None, prev_turn_meta_summary: MetaFromLastSummary) -> ConversationAIResponse:
        """Generate an answer for an existing conversation (no new user turn ingestion)."""

        from .agentic_answering import AgenticAnsweringAgent, AgentConfig

        model_names = model_names or [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ]

        last_err: Exception | None = None
        for model_name in model_names:
            try:
                llm = self.conversation_engine.get_llm(model_name) or self.llm
                agent = AgenticAnsweringAgent(
                    conversation_engine=self.conversation_engine,
                    knowledge_engine=self.ref_knowledge_engine,
                    llm=llm,
                    config=AgentConfig(),
                )
                out = agent.answer(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)

                if isinstance(out, ConversationAIResponse):
                    return out
                if isinstance(out, dict):
                    return ConversationAIResponse(
                        text=str(out.get("assistant_text") or out.get("text") or ""),
                        llm_decision_need_summary=bool(out.get("llm_decision_need_summary", False)),
                        used_kg_node_ids=list(out.get("used_kg_node_ids") or out.get("used_node_ids") or []),
                        projected_conversation_node_ids=list(out.get("projected_node_ids") or out.get("projected_pointer_ids") or []),
                        meta={"raw": out, "model_name": model_name},
                        response_node_id = out.get("assistant_turn_node_id")
                    )
                return ConversationAIResponse(text=str(out))
            except Exception as e:
                last_err = e
                continue
        raise Exception(f"tried all models; last error: {last_err}")
