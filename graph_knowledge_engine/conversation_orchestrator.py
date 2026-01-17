"""Conversation orchestrator (KGE-native, LangGraph-free).

This module owns *policy + control flow* for conversation turns, while delegating
all persistence/mutation to the conversation engine.

It is intentionally lightweight and uses your existing retrievers/agents.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from graph_knowledge_engine.engine import GraphKnowledgeEngine

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
    def add_conversation_turn_workflow(
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
        deprecated legacy path for logic reference, DO NOT USE IT
        Workflow-driven orchestration path.

        - STATIC graph shape: workflow nodes/edges are already registered in engine store.
        - DYNAMIC routing: by predicate registry and step results.
        - Parallelism + shared message queue: handled by WorkflowExecutor.
        - Step implementations: resolved by op name -> callable (no monolithic workflow fn).

        Legacy add_conversation_turn remains unchanged and continues to be the default path.
        """
        raise Exception("Deprecated code path only for logic reference")
        if prev_turn_meta_summary is None:
            prev_turn_meta_summary = MetaFromLastSummary()

        # Compute embedding exactly like legacy path
        emb_text0 = f"{role}: {content}"
        embedding = self.conversation_engine._iterative_defensive_emb(emb_text0)
        if embedding is None:
            raise RuntimeError("uncalculatable embeddings")

        # Create/persist the turn node (mirror of legacy behavior, minimal)
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
            },
            domain_id=None,
            canonical_entity_id=None,
        )
        turn_node.embedding = embedding
        self.conversation_engine.add_node(turn_node)
        prev_turn_meta_summary.tail_turn_index = new_index
        self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span, 
                                  prev_turn_meta_summary=prev_turn_meta_summary)

        # Load workflow spec from engine via convention (wf_start=True)
        wf_spec: WorkflowSpec = build_workflow_from_engine(engine=self.conversation_engine, workflow_id=workflow_id)

        # Predicate registry for routing
        # (you can expand this set, but keep them symbolic)
        predicate_registry = {
            "has_memory": lambda st, r: st.get("memory") is not None,
            "has_kg": lambda st, r: st.get("kg") is not None,
        }

        # Step resolver: op-name -> callable(ctx)->result
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
                    # also publish to message queue for parallel observers if needed
                    ctx.publish({"type": "memory_done"})
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
                    mem = ctx.state.get("memory")
                    kg = self.tool_runner.run_tool(
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
                            user_id=user_id,
                            current_conversation_id=conversation_id,
                            query_embedding=embedding,
                            user_text=content,
                            context_text="",
                            seed_kg_node_ids=list(getattr(mem, "seed_kg_node_ids", []) or []),
                            self_span=self_span,
                            selected_memory=getattr(mem, "selected", None),
                            memory_context_text=getattr(mem, "memory_context_text", ""),
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
                    # allow think/retry to observe crawl/tool updates via message queue
                    _msgs = ctx.drain_messages()
                    ans = self.answer_only(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)
                    return ans
                return _run

            raise KeyError(f"Unknown workflow op: {op_name}")

        executor = WorkflowExecutor(
            engine=self.conversation_engine,
            workflow=wf_spec,
            step_resolver=resolve_step,
            predicate_registry=predicate_registry,
            cache_root=".workflow_cache",
            max_workers=4,
        )

        init_state = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "turn_node_id": turn_node_id,
            "turn_index": new_index,
            "role": role,
            "user_text": content,
            "embedding": embedding,
            "memory": None,
            "kg": None,
            "answer": None,
        }

        for _ev in executor.run(run_id=turn_node_id, initial_state=init_state):
            # If you want streaming, yield these from a streaming API variant.
            pass

        response = executor.state.get("answer")
        response_turn_node_id = getattr(response, "response_node_id", None) if response is not None else None

        return AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=response_turn_node_id,
            turn_index=new_index,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

    # ----------------------------
    # Workflow-v2: design + step resolver
    # ----------------------------

    def _ensure_add_turn_workflow_design(self, workflow_id: str) -> None:
        """Ensure a workflow design exists in workflow_engine for legacy add_turn behavior.

        This writes *design nodes/edges* into workflow_engine (kg_graph_type="workflow").
        The design mirrors ConversationOrchestrator.add_conversation_turn:
          - memory_retrieve
          - kg_retrieve
          - pin memory (conditional)
          - pin kg (conditional)
          - answer
          - decide summarize
          - summarize (conditional)
          - end

        If the design is already present, this is a no-op.
        """
        if self.workflow_engine is None:
            raise RuntimeError("workflow_engine must be provided")

        from .workflow.design import validate_workflow_design
        from .models import WorkflowNode, WorkflowEdge

        # If it validates, we assume it exists + is usable.
        try:
            validate_workflow_design(
                workflow_engine=self.workflow_engine,
                workflow_id=workflow_id,
                predicate_registry={
                    "always": lambda st, r: True,
                    "should_pin_memory": lambda st, r: True,
                    "should_pin_kg": lambda st, r: True,
                    "should_summarize": lambda st, r: True,
                },
            )
            return
        except Exception:
            pass

        # ----------------------------
        # Create design nodes
        # ----------------------------
        def n(node_id: str, *, label: str, op: str | None, start: bool = False, terminal: bool = False, fanout: bool = False) -> WorkflowNode:
            return WorkflowNode(
                id=node_id,
                label=label,
                type="entity",
                doc_id=node_id,
                summary=label,
                properties={},
                metadata={
                    "entity_type": "workflow_node",
                    "workflow_id": workflow_id,
                    "wf_op": op,
                    "wf_start": start,
                    "wf_terminal": terminal,
                    "wf_fanout": fanout,
                    "wf_version": "v1",
                },
            )

        # Use workflow_id namespacing to avoid node id collisions across workflows.
        wid = lambda suffix: f"wf:{workflow_id}:{suffix}"

        nodes = [
            n(wid("start"), label="Start (memory_retrieve)", op="memory_retrieve", start=True),
            n(wid("kg"), label="KG retrieve", op="kg_retrieve", fanout=True),
            n(wid("pin_mem"), label="Pin memory", op="memory_pin"),
            n(wid("pin_kg"), label="Pin knowledge", op="kg_pin"),
            n(wid("answer"), label="Answer", op="answer"),
            n(wid("decide_sum"), label="Decide summarize", op="decide_summarize"),
            n(wid("summarize"), label="Summarize", op="summarize"),
            n(wid("end"), label="End", op=None, terminal=True),
        ]

        for node in nodes:
            self.workflow_engine.add_node(node)

        # ----------------------------
        # Create design edges
        # ----------------------------
        def e(edge_id: str, *, src: str, dst: str, pred: str | None, priority: int = 100, is_default: bool = False, multiplicity: str = "one") -> WorkflowEdge:
            return WorkflowEdge(
                id=edge_id,
                source_ids=[src],
                target_ids=[dst],
                relation="wf_next",
                label="wf_next",
                type="relationship",
                summary="wf_next",
                doc_id=edge_id,
                properties={},
                metadata={
                    "entity_type": "workflow_edge",
                    "workflow_id": workflow_id,
                    "wf_predicate": pred,
                    "wf_priority": priority,
                    "wf_is_default": is_default,
                    "wf_multiplicity": multiplicity,
                },
            )

        edges = [
            e(wid("e1"), src=wid("start"), dst=wid("kg"), pred=None, is_default=True),

            # From KG retrieve, optionally pin both memory and KG (fanout node).
            e(wid("e2"), src=wid("kg"), dst=wid("pin_mem"), pred="should_pin_memory", priority=0),
            e(wid("e3"), src=wid("kg"), dst=wid("pin_kg"), pred="should_pin_kg", priority=1),

            # If no pins happen, go straight to answer.
            e(wid("e4"), src=wid("kg"), dst=wid("answer"), pred=None, is_default=True),
            # After pins, continue to answer.
            e(wid("e5"), src=wid("pin_mem"), dst=wid("answer"), pred=None, is_default=True),
            e(wid("e6"), src=wid("pin_kg"), dst=wid("answer"), pred=None, is_default=True),

            # Summarization decision after answer (needs llm_decision_need_summary).
            e(wid("e7"), src=wid("answer"), dst=wid("decide_sum"), pred=None, is_default=True),
            e(wid("e8"), src=wid("decide_sum"), dst=wid("summarize"), pred="should_summarize", priority=0),
            e(wid("e9"), src=wid("decide_sum"), dst=wid("end"), pred=None, is_default=True),
            e(wid("e10"), src=wid("summarize"), dst=wid("end"), pred=None, is_default=True),
        ]

        for edge in edges:
            self.workflow_engine.add_edge(edge)

    def _make_add_turn_step_resolver(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_node_id: str,
        turn_index: int,
        role: Role,
        content: str,
        mem_id: str,
        embedding: Any,
        self_span: Span,
        filtering_callback: Callable[..., tuple[RetrievalResult, str]],
        max_retrieval_level: int,
        summary_char_threshold: int,
        prev_turn_meta_summary: MetaFromLastSummary,
    ):
        """Factory returning a resolver(op_name)->StepFn.

        Keeping this as a method (rather than a giant nested function) makes it:
        - easier to unit test in isolation
        - easier to share helper logic across workflows
        - less likely to capture the wrong variables accidentally
        """

        from .workflow.runtime import StepContext
        from .workflow.serialize import to_jsonable

        def resolve_step(op_name: str):
            if op_name == "memory_retrieve":
                def _run(ctx: StepContext):
                    mem_retriever = MemoryRetriever(
                        conversation_engine=self.conversation_engine,
                        llm=self.llm,
                        filtering_callback=filtering_callback,
                    )
                    mem = self.tool_runner.run_tool(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        turn_node_id=turn_node_id,
                        turn_index=turn_index,
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
                    # store both raw (for later pinning) and json mirror
                    ctx.state["memory_raw"] = mem
                    memj = to_jsonable(mem)
                    ctx.state["memory"] = memj
                    return memj
                return _run

            if op_name == "kg_retrieve":
                def _run(ctx: StepContext):
                    kg_retriever = KnowledgeRetriever(
                        conversation_engine=self.conversation_engine,
                        ref_knowledge_engine=self.ref_knowledge_engine,
                        llm=self.llm,
                        filtering_callback=filtering_callback,
                        max_retrieval_level=max_retrieval_level,
                    )
                    mem_raw = ctx.state.get("memory_raw")
                    seed_ids = list(getattr(mem_raw, "seed_kg_node_ids", []) or []) if mem_raw is not None else []
                    kg = self.tool_runner.run_tool(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        turn_node_id=turn_node_id,
                        turn_index=turn_index,
                        tool_name="kg_retrieve",
                        args={"max_retrieval_level": max_retrieval_level, "seed_kg_node_ids": seed_ids},
                        handler=lambda: kg_retriever.retrieve(
                            user_text=content,
                            context_text="",
                            query_embedding=embedding,
                            seed_kg_node_ids=seed_ids,
                        ),
                        render_result=lambda r: getattr(r, "reasoning", "")[:800],
                        prev_turn_meta_summary=prev_turn_meta_summary,
                    )
                    ctx.state["kg_raw"] = kg
                    kgj = to_jsonable(kg)
                    ctx.state["kg"] = kgj
                    return kgj
                return _run

            if op_name == "memory_pin":
                def _run(ctx: StepContext):
                    mem_raw = ctx.state.get("memory_raw")
                    mem_retriever = MemoryRetriever(
                        conversation_engine=self.conversation_engine,
                        llm=self.llm,
                        filtering_callback=filtering_callback,
                    )
                    out = None
                    if mem_raw is not None and getattr(mem_raw, "selected", None) and getattr(mem_raw, "memory_context_text", None):
                        out = mem_retriever.pin_selected(
                            user_id=user_id,
                            current_conversation_id=conversation_id,
                            turn_node_id=turn_node_id,
                            mem_id=mem_id,
                            turn_index=turn_index,
                            self_span=self_span,
                            selected_memory=getattr(mem_raw, "selected"),
                            memory_context_text=getattr(mem_raw, "memory_context_text"),
                            prev_turn_meta_summary=prev_turn_meta_summary,
                        )

                    ctx.state["memory_pin_raw"] = out
                    outj = {
                        "memory_context_node_id": getattr(getattr(out, "memory_context_node", None), "id", None) if out else None,
                        "pinned_edge_ids": [e.id for e in getattr(out, "pinned_edges", [])] if out else [],
                    }
                    ctx.state["memory_pin"] = outj
                    return outj
                return _run

            if op_name == "kg_pin":
                def _run(ctx: StepContext):
                    kg_raw = ctx.state.get("kg_raw")
                    kg_retriever = KnowledgeRetriever(
                        conversation_engine=self.conversation_engine,
                        ref_knowledge_engine=self.ref_knowledge_engine,
                        llm=self.llm,
                        filtering_callback=filtering_callback,
                        max_retrieval_level=max_retrieval_level,
                    )
                    pinned_ptrs: list[str] = []
                    pinned_edges: list[str] = []
                    if kg_raw is not None and getattr(kg_raw, "selected", None):
                        pinned_ptrs, pinned_edges = kg_retriever.pin_selected(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            turn_node_id=turn_node_id,
                            turn_index=turn_index,
                            self_span=self_span,
                            selected_knowledge=getattr(kg_raw, "selected"),
                            prev_turn_meta_summary=prev_turn_meta_summary,
                        )
                    outj = {"pinned_pointer_node_ids": list(pinned_ptrs), "pinned_edge_ids": list(pinned_edges)}
                    ctx.state["kg_pin"] = outj
                    return outj
                return _run

            if op_name == "answer":
                def _run(ctx: StepContext):
                    resp = self.answer_only(conversation_id=conversation_id, prev_turn_meta_summary=prev_turn_meta_summary)
                    ctx.state["answer_raw"] = resp

                    # Link assistant node to the user turn for conversation chain continuity.
                    response_node_id = getattr(resp, "response_node_id", None)
                    if response_node_id:
                        try:
                            resp_node = self.conversation_engine.get_nodes([response_node_id])[0]
                            self.add_link_to_new_turn(resp_node, self.conversation_engine.get_nodes([turn_node_id])[0], conversation_id, span=self_span)
                        except Exception:
                            pass

                    # mirror legacy: advance distances after adding assistant turn, if available
                    try:
                        if response_node_id:
                            txt = getattr(resp, "answer", None)
                            if isinstance(txt, str):
                                prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(txt)
                            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
                    except Exception:
                        pass

                    outj = {
                        "response_node_id": response_node_id,
                        "llm_decision_need_summary": bool(getattr(resp, "llm_decision_need_summary", False)),
                    }
                    ctx.state["answer"] = outj
                    return outj
                return _run

            if op_name == "decide_summarize":
                def _run(ctx: StepContext):
                    ans = ctx.state.get("answer") or {}
                    # Same policy as legacy: distance>=5 OR char_threshold exceeded OR model says so.
                    should = False
                    if prev_turn_meta_summary.prev_node_distance_from_last_summary - 5 >= 0:
                        should = True
                    if prev_turn_meta_summary.prev_node_char_distance_from_last_summary > summary_char_threshold:
                        should = True
                    if bool(ans.get("llm_decision_need_summary", False)):
                        should = True

                    ctx.state["summary"] = {"should_summarize": should, "did_summarize": False}
                    ctx.state["prev_turn_meta_summary"] = {
                        "prev_node_char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                        "prev_node_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    }
                    return ctx.state["summary"]
                return _run

            if op_name == "summarize":
                def _run(ctx: StepContext):
                    added_id = self._summarize_conversation_batch(conversation_id, turn_index + 1, prev_turn_meta_summary=prev_turn_meta_summary)
                    # legacy resets after summarization
                    prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
                    prev_turn_meta_summary.prev_node_distance_from_last_summary = 0
                    ctx.state["summary"] = {"should_summarize": True, "did_summarize": True, "summary_node_id": added_id}
                    ctx.state["prev_turn_meta_summary"] = {
                        "prev_node_char_distance_from_last_summary": 0,
                        "prev_node_distance_from_last_summary": 0,
                    }
                    return ctx.state["summary"]
                return _run

            raise KeyError(f"Unknown workflow op: {op_name}")

        return resolve_step
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
        if self.workflow_engine is None:
            raise RuntimeError("workflow_engine must be provided for workflow_v2")

        # ----------------------------
        # 0) Compute prior-summary distances (mirror legacy)
        # ----------------------------
        if prev_turn_meta_summary is None:
            prev_turn_meta_summary = MetaFromLastSummary(0, 0)

        prev_node = self.conversation_engine._get_conversation_tail(conversation_id) if in_conv else None
        if prev_node is not None:
            new_index = (prev_node.turn_index + 1) if prev_node.turn_index is not None else 0

            node_last_char_dist = prev_node.metadata.get("char_distance_from_last_summary")
            if node_last_char_dist is not None:
                node_last_char_dist += len(prev_node.summary)
            else:
                node_last_char_dist = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = node_last_char_dist

            node_last_turn_dist = prev_node.metadata.get("turn_distance_from_last_summary")
            if node_last_turn_dist is not None:
                node_last_turn_dist += 1
            else:
                node_last_turn_dist = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = node_last_turn_dist
        else:
            new_index = 0
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary = -1

        # ----------------------------
        # 1) Create/persist the user turn node (mirror legacy)
        # ----------------------------
        emb_text0 = f"{role}: {content}"
        embedding = self.conversation_engine._iterative_defensive_emb(emb_text0)
        if embedding is None:
            raise RuntimeError("uncalculatable embeddings")

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
            verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="verbatim input"),
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
        turn_node.embedding = embedding
        self.conversation_engine.add_node(turn_node)
        prev_turn_meta_summary.tail_turn_index = new_index
        if prev_node is not None:
            self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span, 
                                      prev_turn_meta_summary=prev_turn_meta_summary)

        # mirror legacy: advance distances after adding this user turn
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(content)
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1

        # ----------------------------
        # 2) Ensure the workflow design exists
        # ----------------------------
        self._ensure_add_turn_workflow_design(workflow_id)

        # ----------------------------
        # 3) Predicates (dynamic routing)
        # ----------------------------
        predicate_registry = {
            "always": lambda st, r: True,
            "should_pin_memory": lambda st, r: bool((st.get("memory") or {}).get("selected")) and bool(
                (st.get("memory") or {}).get("memory_context_text")
            ),
            "should_pin_kg": lambda st, r: bool((st.get("kg") or {}).get("selected")),
            "should_summarize": lambda st, r: bool(st.get("summary", {}).get("should_summarize", False)),
        }

        # ----------------------------
        # 4) Step resolver
        # ----------------------------
        resolve_step = self._make_add_turn_step_resolver(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            turn_index=new_index,
            role=role,
            content=content,
            mem_id=mem_id,
            embedding=embedding,
            self_span=self_span,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
            summary_char_threshold=summary_char_threshold,
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

        # ----------------------------
        # 5) Run with WorkflowRuntime (real persisted checkpoints)
        # ----------------------------
        from .workflow.runtime import WorkflowRuntime

        runtime = WorkflowRuntime(
            workflow_engine=self.workflow_engine,
            conversation_engine=self.conversation_engine,
            step_resolver=resolve_step,
            predicate_registry=predicate_registry,
            checkpoint_every_n_steps=1,
            max_workers=4,
        )

        init_state = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "turn_node_id": turn_node_id,
            "turn_index": new_index,
            "role": str(role),
            "user_text": content,
            "embedding": embedding,
            "memory": None,
            "memory_raw": None,
            "kg": None,
            "kg_raw": None,
            "memory_pin": None,
            "memory_pin_raw": None,
            "kg_pin": None,
            "answer": None,
            "answer_raw": None,
            "summary": {"should_summarize": False, "did_summarize": False, "summary_node_id": None},
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                "prev_node_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
            },
        }

        final_state, run_id = runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=init_state,
            run_id=f"add_turn|{turn_node_id}",
        )

        # ----------------------------
        # 6) Map final state to legacy AddTurnResult
        # ----------------------------
        mem = final_state.get("memory") or {}
        kg = final_state.get("kg") or {}
        mem_pin = final_state.get("memory_pin") or {}
        kg_pin = final_state.get("kg_pin") or {}
        answer = final_state.get("answer") or {}

        # update prev_turn_meta_summary in-place so callers can chain it
        mts = final_state.get("prev_turn_meta_summary") or {}
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = int(
            mts.get("prev_node_char_distance_from_last_summary", prev_turn_meta_summary.prev_node_char_distance_from_last_summary)
        )
        prev_turn_meta_summary.prev_node_distance_from_last_summary = int(
            mts.get("prev_node_distance_from_last_summary", prev_turn_meta_summary.prev_node_distance_from_last_summary)
        )

        return AddTurnResult(
            user_turn_node_id=turn_node_id,
            response_turn_node_id=answer.get("response_node_id"),
            turn_index=new_index,
            relevant_kg_node_ids=list((kg.get("selected") or {}).get("node_ids") or []),
            relevant_kg_edge_ids=list((kg.get("selected") or {}).get("edge_ids") or []),
            pinned_kg_pointer_node_ids=list(kg_pin.get("pinned_pointer_node_ids") or []),
            pinned_kg_edge_ids=list(kg_pin.get("pinned_edge_ids") or []),
            memory_context_node_id=mem_pin.get("memory_context_node_id"),
            memory_context_edge_ids=list(mem_pin.get("pinned_edge_ids") or []),
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
        self.tail_search_includes = ["conversation_turn","conversation_summary"]
    def add_link_to_new_turn(self, turn_node, prev_node, conversation_id, span,prev_turn_meta_summary:MetaFromLastSummary):
        
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
                metadata={"relation":"next_turn", "target_id":turn_node.id,
                          "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                            "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(seq_edge)
            return seq_edge
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
                    "turn_index": new_index,
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
            prev_turn_meta_summary.tail_turn_index = new_index
            self.conversation_engine.add_node(turn_node)
            self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span, 
                                      prev_turn_meta_summary=prev_turn_meta_summary)
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
                    "turn_index": new_index,
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
            prev_turn_meta_summary.tail_turn_index = new_index
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
                seq_edge = self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span, prev_turn_meta_summary=prev_turn_meta_summary)
                # seq_edge = ConversationEdge(
                #     id=str(uuid.uuid4()),
                #     source_ids=[prev_node.id],
                #     target_ids=[turn_node_id],
                #     relation="next_turn",
                #     label="next_turn",
                #     type="relationship",
                #     summary="Sequential flow",
                #     doc_id=f"conv:{conversation_id}",
                #     mentions=[Grounding(spans=[self_span])],
                #     domain_id=None,
                #     canonical_entity_id=None,
                #     properties={"entity_type": "conversation_edge"},
                #     embedding=None,
                #     metadata={"char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                #     "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,},
                #     source_edge_ids=[],
                #     target_edge_ids=[],
                # )
                # self.conversation_engine.add_edge(seq_edge)
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

                    self.add_link_to_new_turn(turn_node, prev_node, conversation_id, span=self_span, 
                                              prev_turn_meta_summary=prev_turn_meta_summary)
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
            added_id = self._summarize_conversation_batch(conversation_id, new_index,prev_turn_meta_summary=prev_turn_meta_summary)
            new_index += 1
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
            prev_turn_meta_summary.prev_node_distance_from_last_summary=0
            add_turn_result = replace(add_turn_result, turn_index = new_index)
        return add_turn_result

    # @conversation_only
    def _summarize_conversation_batch(self, conversation_id: str, current_index: int, 
                                      batch_size: int = 5, in_conv=True , 
                                      prev_turn_meta_summary : MetaFromLastSummary= None):
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
        new_index = current_index + 1
        summary_node = ConversationNode(
            id=str(uuid.uuid4()),
            label=f"Summary {start_index}-{current_index}",
            type="entity",
            summary='\n'.join(i['text'] for i in get_summary(full_text)), # type: ignore
            role="system", # type: ignore
            conversation_id=conversation_id,
            turn_index=new_index, # Anchored at end of batch
            # Provenance: We link back to the source turns and knowledge refs
            mentions=[Grounding(spans=unique_spans)] if unique_spans else [],
            
            properties={"content": json.dumps(summary_content)},
            metadata={"level_from_root": 1, 
                      "entity_type": "conversation_summary",
                      "turn_index": new_index, 
                      "char_distance_from_last_summary": 0, 
                      "turn_distance_from_last_summary" : 0 ,
                      "in_conversation_chain": in_conv}, # Summary is higher level?
            domain_id=None,
            canonical_entity_id=None
        )
        prev_turn_meta_summary.tail_turn_index = new_index
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = 0
        prev_turn_meta_summary.prev_node_distance_from_last_summary = 0
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
            metadata={"char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                            "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary},
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
            self.add_link_to_new_turn(summary_node, prev_node, conversation_id, self_span, prev_turn_meta_summary)
            
            # add next turn relationship
            
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
