from __future__ import annotations

import uuid
from typing import Callable, List, Optional, cast

from kogwistar.llm_tasks import LLMTaskSet

from .models import ConversationEdge
from .models import RetrievalResult
from .models import ConversationNode, MetaFromLastSummary, Node, Edge, FilteringResult
from ..engine_core.engine import GraphKnowledgeEngine
from ..engine_core.models import Grounding, Span


from .models import MemoryRetrievalResult, MemoryPinResult


def is_node_memory_context(node: ConversationNode):
    # example node:
    # ConversationNode(
    #         id=mem_node_id,
    #         label=f"Memory context (turn {turn_index})",
    #         type="entity",
    #         doc_id=mem_node_id,
    #         summary=memory_context_text,
    #         role="system",  # type: ignore
    #         turn_index=turn_index,
    #         conversation_id=current_conversation_id,
    #         user_id=user_id,
    #         mentions=[Grounding(spans=[self_span])],
    #         properties={
    #             "user_id": user_id,
    #             "source_memory_nodes_ids": [i.id for i in selected_memory.nodes],
    #             "source_memory_edges_ids": [i.id for i in selected_memory.edges],
    #         },
    #         metadata={
    #             "entity_type": "memory_context",
    #             "type": "entity",
    #             "level_from_root": 0,
    #             "char_distance_from_last_summary": 0,  # memory itself is a summary of other nodes
    #             "turn_distance_from_last_summary": 0,
    #             "user_id": user_id,
    #             "conversation_id": current_conversation_id,
    #             "turn_index": turn_index,
    #         },
    #         domain_id=None,
    #         canonical_entity_id=None,
    #     )
    """
    Return True if the node represents a memory-context ConversationNode.
    This is a structural + semantic check, not an identity check.
    """

    if node is None:
        return False

    # 1. Fast semantic markers (cheap, decisive)
    metadata = getattr(node, "metadata", None) or {}
    if metadata.get("entity_type") == "memory_context":
        return True

    # 2. Fallback: properties-based signal
    props = getattr(node, "properties", None) or {}
    if "source_memory_nodes_ids" in props and "source_memory_edges_ids" in props:
        return True

    # 3. Last-resort heuristic (avoid relying on label unless necessary)
    label = getattr(node, "label", "") or ""
    if label.lower().startswith("memory context"):
        return True

    return False


class MemoryRetriever:
    """Long-memory retriever over the conversation store, keyed by user_id.

    This retriever supports:
    - selecting prior summary/turn nodes across conversations for the same user
    - synthesizing a compact `memory_context` node for the *current* turn
    - extracting KG seeds from selected nodes when they contain pinned KG pointers
      (type == 'reference_pointer' and properties['refers_to_id'] exists)

    Naming conventions (do not mix):
    - conversation_engine: the conversation canvas engine (current engine instance)
    - user_id: identity used for cross-conversation retrieval
    """

    result_model = MemoryPinResult

    def __init__(
        self,
        *,
        conversation_engine,
        llm_tasks: LLMTaskSet,
        filtering_callback: Callable[
            ..., tuple[FilteringResult | RetrievalResult, str]
        ],
        summarize_callback: Optional[
            Callable[..., str]
        ] = None,  # can be a callback with context closured
        prefer_types: Optional[List[str]] = None,
    ) -> None:
        self.conversation_engine: GraphKnowledgeEngine = conversation_engine
        self.llm_tasks = llm_tasks
        self.filtering_callback = filtering_callback
        self.summarize_callback = summarize_callback
        self.prefer_types = prefer_types or [
            "conversation_summary",
            "conversation_turn",
            "reference_pointer",
        ]

    def retrieve(
        self,
        *,
        user_id: str,
        current_conversation_id: str,
        query_embedding: List[float],
        user_text: str,
        context_text: str,
        n_results: int,
    ) -> MemoryRetrievalResult:
        # Broad memory retrieval across same user
        where = {"user_id": user_id}
        memory_nodes = self.conversation_engine.read.query_nodes(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "embeddings"],
            node_type=ConversationNode,
        )[0]
        where = {"user_id": user_id}
        memory_edges = self.conversation_engine.read.query_edges(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "embeddings"],
            edge_type=ConversationEdge,
        )[
            0
        ]  # supposed not work, conversation edge has no role mixin and not retrievable no matter how given current state

        def _rank(m: Node | Edge):
            t = m.type or m.metadata.get("entity_type") or ""
            return 0 if t in self.prefer_types else 1

        memory_nodes.sort(key=lambda x: _rank(x))
        memory_edges.sort(key=lambda x: _rank(x))
        candidates = RetrievalResult(memory_nodes, memory_edges)
        # # Optional type preference reorder (soft heuristic, no filtering)
        # if candidate_ids and candidate_metas:
        #     zipped = list(zip(candidate_ids, candidate_docs, candidate_metas))
        # def _rank(m):
        #     t = (m or {}).get("type") or (m or {}).get("entity_type") or ""
        #     return 0 if t in self.prefer_types else 1
        #     zipped.sort(key=lambda x: _rank(x[2]))
        #     candidate_ids = [z[0] for z in zipped]
        #     candidate_docs = [z[1] for z in zipped]
        #     candidate_metas = [z[2] for z in zipped]
        selected_ids: List[str] = []
        reasoning = ""
        if candidates.nodes or candidates.edges:
            # Keep prompt compact: show meta summaries only
            cand_node_list_str = "\n".join(
                [
                    f"-Node ID: {node.id} | Label: {node.metadata.get('label')} | Summary: {node.metadata.get('summary')}"
                    for node in candidates.nodes
                ]
            )
            cand_edge_list_str = "\n".join(
                [
                    f"-Edge ID: {edge.id} | Label: {edge.metadata.get('label')} | Summary: {edge.metadata.get('summary')}"
                    for edge in candidates.edges
                ]
            )
            filtered, reasoning = self.filtering_callback(
                self.llm_tasks,
                user_text,
                cand_node_list_str,
                cand_edge_list_str,
                [i.id for i in candidates.nodes],
                [i.id for i in candidates.edges],
                context_text,
            )

            filtered = FilteringResult.model_validate(filtered)
            selected = RetrievalResult(
                nodes=[n for n in candidates.nodes if n.id in filtered.node_ids],
                edges=[n for n in candidates.edges if n.id in filtered.edge_ids],
            )
        else:
            filtered = None
            selected = None
        #     return MemoryRetrievalResult(candidate_node_ids=ShallowQueryResult(candidate_kg_node_ids = [], candidate_kg_edge_ids = []),
        #                                     selected_kg_ids=ShallowQueryResult(candidate_kg_node_ids = [], candidate_kg_edge_ids = []),
        #                                     reasoning=reasoning,memory_context_text="", seed_kg_node_ids=[])
        # Extract KG seeds from selected nodes when they are reference pointers

        seed_kg_ids: List[str] = []
        non_kg_node_ids: List[str] = []
        non_kg_edge_ids: List[str] = []
        conv_nodes: List[ConversationNode] = []
        conv_edges: List[ConversationNode] = []
        if selected:
            for n in selected.nodes:
                if n.type != "reference_pointer":
                    non_kg_node_ids.append(n.safe_get_id())
                    conv_nodes.append(n)
                    continue
                rid = (n.properties or {}).get("refers_to_id")
                selected_ids.append(n.safe_get_id())
                if isinstance(rid, str) and rid:
                    seed_kg_ids.append(rid)
            for n in selected.edges:
                if n.type != "reference_pointer":
                    non_kg_edge_ids.append(n.safe_get_id())
                    conv_edges.append(n)
                    continue
                rid = (n.properties or {}).get("refers_to_id")
                selected_ids.append(n.id)
                if isinstance(rid, str) and rid:
                    seed_kg_ids.append(rid)

        # De-dupe seeds in order
        seen = set()
        dedup_seeds: List[str] = []
        for x in seed_kg_ids:
            if x not in seen:
                seen.add(x)
                dedup_seeds.append(x)

        # Build memory context text (LLM summarize or fallback join)
        memory_context_text: None | str = None
        # for ref nodes and ref edges only
        if selected:
            if self.summarize_callback is not None:
                memory_context_text = self.summarize_callback(
                    self.llm_tasks, user_text, selected
                )
            else:
                # Fallback: concatenate short snippets from metadata/documents
                # Keep it bounded: use first 5 items
                parts = []
                selected_conv_nodes = cast(list[ConversationNode], selected.nodes)
                for candidate in sorted(
                    selected_conv_nodes, key=lambda x: x.turn_index or -2
                )[-5:]:
                    # for rid, meta in list(zip(selected_ids, candidate_metas))[:5]:
                    parts.append(
                        f"[{candidate.id}] {candidate.metadata.get('summary') or ''}"
                    )

                selected_conv_edges = cast(list[ConversationNode], selected.edges)
                for candidate in sorted(
                    selected_conv_edges, key=lambda x: x.turn_index or -2
                )[-5:]:
                    parts.append(
                        f"[{candidate.id}] {candidate.metadata.get('summary') or ''}"
                    )
                memory_context_text = "\n".join(parts).strip()

        return MemoryRetrievalResult(
            candidate=candidates,
            selected=selected,
            reasoning=reasoning,
            memory_context_text=memory_context_text,
            seed_kg_node_ids=dedup_seeds,
            node_id_entry=None,
        )

    def pin_selected(
        self,
        *,
        user_id: str,
        current_conversation_id: str,
        mem_id: str | None = None,  # for both node and edge
        turn_node_id: str,
        turn_index: int,
        self_span: Span,
        selected_memory: RetrievalResult,
        memory_context_text: str,
        prev_turn_meta_summary: MetaFromLastSummary,
    ) -> Optional[MemoryPinResult]:
        """Materialize a `memory_context` node into the current conversation canvas.

        Edges:
        - Turn --has_memory_context--> MemoryContext
        - MemoryContext --summarizes--> SourceNode (by id)

        If nothing selected, returns None.
        """
        if not selected_memory or not memory_context_text:
            return None

        mem_node_id = mem_id  # or str(uuid.uuid4())
        mem_node = ConversationNode(
            id=mem_node_id,
            label=f"Memory context (turn {turn_index})",
            type="entity",
            doc_id=mem_node_id,
            summary=memory_context_text,
            role="system",  # type: ignore
            turn_index=turn_index,
            conversation_id=current_conversation_id,
            user_id=user_id,
            mentions=[Grounding(spans=[self_span])],
            properties={
                "user_id": user_id,
                "source_memory_nodes_ids": [i.id for i in selected_memory.nodes],
                "source_memory_edges_ids": [i.id for i in selected_memory.edges],
            },
            metadata={
                "entity_type": "memory_context",
                "type": "entity",
                "level_from_root": 0,
                "user_id": user_id,
                "conversation_id": current_conversation_id,
                "turn_index": turn_index,
                "in_conversation_chain": False,
            },
            domain_id=None,
            canonical_entity_id=None,
        )

        self.conversation_engine.write.add_node(mem_node)
        edge_ids: List[str] = []
        edges: List[ConversationEdge] = []
        # Turn -> MemoryContext
        e1_id = f"{turn_node_id}::hm::{mem_node.safe_get_id()}"
        e1 = ConversationEdge(
            id=e1_id,
            source_ids=[turn_node_id],
            target_ids=[mem_node.safe_get_id()],
            relation="has_memory_context",
            label="has_memory_context",
            type="relationship",
            summary="This turn uses long-memory context",
            doc_id=f"conv:{current_conversation_id}",
            mentions=[Grounding(spans=[self_span])],
            domain_id=None,
            canonical_entity_id=None,
            properties={"entity_type": "conversation_edge"},
            embedding=None,
            metadata={
                "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
            },
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.write.add_edge(e1)
        edge_ids.append(e1_id)
        edges.append(e1)
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
            memory_context_text
        )
        # MemoryContext -> Sources nodes
        for node in selected_memory.nodes:
            e_id = f"{mem_id}::n::{node.id}" if mem_id else str(uuid.uuid4())

            e = ConversationEdge(
                id=e_id,
                source_ids=[mem_node.safe_get_id()],
                target_ids=[node.safe_get_id()] if node.id is not None else [],
                relation="summarizes",
                label="summarizes",
                type="relationship",
                summary="Memory context summarizes prior conversation artifact",
                doc_id=f"conv:{current_conversation_id}",
                mentions=[Grounding(spans=[self_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity_type": "conversation_edge"},
                embedding=None,
                metadata={
                    "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
                },
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.write.add_edge(e)
            edge_ids.append(e_id)
        # MemoryContext -> Sources edges
        for edge in selected_memory.edges:
            e_id = f"{mem_id}::e::{edge.id}" if mem_id else str(uuid.uuid4())

            e = ConversationEdge(
                id=e_id,
                source_ids=[mem_node.safe_get_id()],
                target_ids=[edge.id] if edge.id is not None else [],
                relation="summarizes",
                label="summarizes",
                type="relationship",
                summary="Memory context summarizes prior conversation artifact",
                doc_id=f"conv:{current_conversation_id}",
                mentions=[Grounding(spans=[self_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity_type": "conversation_edge"},
                embedding=None,
                metadata={
                    "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
                    "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                    "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
                },
                source_edge_ids=[],
                target_edge_ids=[edge.safe_get_id()],
            )
            self.conversation_engine.write.add_edge(e)
            edge_ids.append(e_id)
            edges.append(e)
        return MemoryPinResult(
            memory_context_node=mem_node,
            pinned_edges=edges,
            node_id_entry=mem_node.safe_get_id(),
        )
