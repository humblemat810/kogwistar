"""Conversation service facade following service-pattern composition."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Type

from pydantic import BaseModel

from graph_knowledge_engine.conversation.conversation_context import (
    ContextItem,
    ContextRenderer,
    ContextSources,
    ConversationContextView,
    DroppedItem,
    PromptContext,
    apply_ordering,
)
from graph_knowledge_engine.conversation.conversation_orchestrator import ConversationOrchestrator
from graph_knowledge_engine.conversation.models import (
    AddTurnResult,
    ContextSnapshotMetadata,
    ConversationAIResponse,
    ConversationEdge,
    ConversationNode,
    FilteringResult,
    MetaFromLastSummary,
    RetrievalResult,
    infer_conversation_edge_causal_type,
)
from graph_knowledge_engine.engine_core.models import ContextCost, Grounding, Span
from graph_knowledge_engine.id_provider import stable_id
from graph_knowledge_engine.runtime import WorkflowRuntime

if TYPE_CHECKING:
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine


class _ApproxTokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class ConversationService:
    """High-level conversation behavior service."""

    def __init__(
        self,
        *,
        conversation_engine: "GraphKnowledgeEngine",
        knowledge_engine: "GraphKnowledgeEngine",
        workflow_engine: Optional["GraphKnowledgeEngine"] = None,
        llm: Any | None = None,
        runtime_cls: type[WorkflowRuntime] = WorkflowRuntime,
    ) -> None:
        self.conversation_engine = conversation_engine
        self.knowledge_engine = knowledge_engine
        self.workflow_engine = workflow_engine
        self.llm = llm or conversation_engine.llm
        self.runtime_cls = runtime_cls

        self.orchestrator = ConversationOrchestrator(
            conversation_engine=conversation_engine,
            ref_knowledge_engine=knowledge_engine,
            workflow_engine=workflow_engine,
            llm=self.llm,
        )

    @classmethod
    def from_engine(
        cls,
        conversation_engine: "GraphKnowledgeEngine",
        *,
        knowledge_engine: "GraphKnowledgeEngine | None" = None,
        workflow_engine: "GraphKnowledgeEngine | None" = None,
    ) -> "ConversationService":
        cache = getattr(conversation_engine, "_conversation_service_cache", None)
        if cache is None:
            cache = {}
            conversation_engine._conversation_service_cache = cache

        ke = knowledge_engine or conversation_engine
        we = workflow_engine
        key = (id(ke), id(we), id(conversation_engine.llm))
        svc = cache.get(key)
        if svc is not None:
            return svc

        svc = cls(
            conversation_engine=conversation_engine,
            knowledge_engine=ke,
            workflow_engine=we,
            llm=conversation_engine.llm,
        )
        cache[key] = svc
        return svc

    @classmethod
    def orchestrator_for_engine(
        cls,
        conversation_engine: "GraphKnowledgeEngine",
        *,
        ref_knowledge_engine: "GraphKnowledgeEngine",
    ) -> ConversationOrchestrator:
        svc = cls.from_engine(
            conversation_engine,
            knowledge_engine=ref_knowledge_engine,
            workflow_engine=getattr(conversation_engine, "workflow_engine", None),
        )
        return svc.orchestrator

    def max_node_seq_present(self, conversation_id):
        return self.conversation_engine.meta_sqlite.next_user_seq(conversation_id)

    def get_last_seq_node(self, conversation_id, buffer=5):
        _ = buffer
        return self._get_last_seq_node(conversation_id)

    def _normalize_conversation_edge_metadata(self, edge: ConversationEdge) -> None:
        md = dict(edge.metadata or {})
        if md.get("causal_type") is None:
            md["causal_type"] = infer_conversation_edge_causal_type(edge.relation)
        edge.metadata = md

    def _validate_conversation_edge_add(self, edge: ConversationEdge) -> None:
        eng = self.conversation_engine
        if eng.kg_graph_type != "conversation":
            return

        self._normalize_conversation_edge_metadata(edge)
        md = edge.metadata or {}
        causal_type = md.get("causal_type") or infer_conversation_edge_causal_type(edge.relation)
        doc_id = eng._conversation_doc_id_for_edge(edge)

        src = (edge.source_ids or [None])[0]
        tgt = (edge.target_ids or [None])[0]

        if edge.relation == "next_turn" or causal_type == "chain":
            if src is None or tgt is None:
                raise ValueError("next_turn requires single source_id and single target_id")
            if (getattr(edge, "source_edge_ids", []) or []) or (getattr(edge, "target_edge_ids", []) or []):
                raise ValueError("next_turn must be node-to-node only (no edge endpoints)")

            w_out = eng._where_and(
                {"relation": "next_turn"},
                {"role": "src"},
                {"endpoint_type": "node"},
                {"endpoint_id": src},
                ({"doc_id": doc_id} if doc_id else {}),
            )
            if eng._edge_endpoints_exists(where=w_out):
                raise ValueError(f"next_turn outgoing already exists for source_id={src}")

            w_in = eng._where_and(
                {"relation": "next_turn"},
                {"role": "tgt"},
                {"endpoint_type": "node"},
                {"endpoint_id": tgt},
                ({"doc_id": doc_id} if doc_id else {}),
            )
            if eng._edge_endpoints_exists(where=w_in):
                raise ValueError(f"next_turn incoming already exists for target_id={tgt}")

        if causal_type == "dependency":
            if tgt is None:
                raise ValueError("dependency edge requires single target_id")

            w_used_chain = eng._where_and(
                {"role": "src"},
                {"endpoint_type": "node"},
                {"endpoint_id": tgt},
                {"causal_type": "chain"},
                ({"doc_id": doc_id} if doc_id else {}),
            )
            w_used_dep = eng._where_and(
                {"role": "src"},
                {"endpoint_type": "node"},
                {"endpoint_id": tgt},
                {"causal_type": "dependency"},
                ({"doc_id": doc_id} if doc_id else {}),
            )
            if eng._edge_endpoints_exists(where=w_used_chain) or eng._edge_endpoints_exists(where=w_used_dep):
                raise ValueError(f"Cannot add dependency incoming edge into already-used node {tgt}")

    def _create_conversation_primitive(
        self,
        user_id,
        conv_id=None,
        node_id: str | None | uuid.UUID = None,
    ) -> tuple[str, str]:
        from graph_knowledge_engine.conversation.conversation_orchestrator import get_id_for_conversation_turn

        eng = self.conversation_engine
        if eng.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        new_index = -1
        conv_id = conv_id or str(uuid.uuid4())
        node_id = node_id or get_id_for_conversation_turn(
            ConversationNode.id_kind,
            user_id,
            conv_id,
            "Start of conversation",
            str(new_index),
            "system",
            "conversation_summary",
            in_conv=True,
        )
        start_node = ConversationNode(
            id=str(node_id),
            user_id=user_id,
            label="conversation start",
            type="entity",
            summary="Start of conversation",
            role="system",
            turn_index=-1,
            conversation_id=conv_id,
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"status": "active"},
            metadata={
                "level_from_root": 0,
                "entity_type": "conversation_start",
                "turn_index": -1,
                "in_conversation_chain": True,
                "in_ui_chain": True,
            },
            domain_id=None,
            canonical_entity_id=None,
            level_from_root=0,
            doc_id=None,
            embedding=None,
        )
        dummy_span = Span.from_dummy_for_conversation()
        start_node.mentions = [Grounding(spans=[dummy_span])]
        eng.add_node(start_node)
        return conv_id, str(node_id)

    def create_conversation(self, user_id, conv_id=None, node_id: str | None | uuid.UUID = None) -> tuple[str, str]:
        conv_out, node_out = self._create_conversation_primitive(user_id, conv_id, node_id)
        return str(conv_out), str(node_out)

    def _get_last_seq_node(self, conversation_id, min_seq=None):
        eng = self.conversation_engine
        if min_seq is None:
            min_seq = self.max_node_seq_present(conversation_id)
        if eng.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        got = eng.backend.node_get(
            where={
                "$and": [
                    {"conversation_id": conversation_id},
                ]
                + [{"seq": {"$gte": min_seq or 0}}]
            },
            include=["documents", "metadatas", "embeddings"],
        )
        if not got["ids"]:
            return None
        nodes: list[ConversationNode] = eng.nodes_from_single_or_id_query_result(got, node_type=ConversationNode)
        nodes.sort(key=lambda n: n.metadata.get("seq") or -1)
        return nodes[-1]

    def _get_conversation_tail(
        self,
        conversation_id: str,
        min_turn_index: int | None = None,
        tail_search_includes: list[str] = ["conversation_start", "conversation_turn", "conversation_summary", "assistant_turn"],
    ) -> Optional[ConversationNode]:
        eng = self.conversation_engine
        if eng.kg_graph_type != "conversation":
            raise Exception("conversation only allowed to be on canva engine")
        got = eng.backend.node_get(
            where={
                "$and": [{"conversation_id": conversation_id}, {"in_conversation_chain": True}]
                + ([{"turn_index": {"$gte": min_turn_index}}] if min_turn_index is not None else [])
            },
            include=["documents", "metadatas", "embeddings"],
        )
        if not got["ids"]:
            return None
        nodes: list[ConversationNode] = eng.nodes_from_single_or_id_query_result(got, node_type=ConversationNode)
        nodes2 = [x for x in nodes if x.metadata.get("entity_type") in tail_search_includes]
        if not nodes2:
            return None
        nodes2.sort(key=lambda n: n.turn_index or -1)
        return nodes2[-1]

    def last_summary_of_node(self, node: ConversationNode):
        eng = self.conversation_engine
        summaries = eng.get_nodes(
            where=eng._where_and(
                {"conversation_id": node.conversation_id},
                {"entity_type": "conversation_summary"},
            ),
            node_type=ConversationNode,
            limit=20000,
        )
        best = None
        best_idx = -1
        for s in summaries or []:
            ti = getattr(s, "turn_index", None)
            if ti is None:
                continue
            if node.turn_index is not None and ti <= node.turn_index and ti > best_idx:
                best = s
                best_idx = ti
        return [best] if best is not None else []

    def add_turn(self, *args, **kwargs):
        return self.orchestrator.add_conversation_turn(*args, **kwargs)

    def add_conversation_turn(
        self,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        mem_id: str,
        role: str,
        content: str,
        ref_knowledge_engine: "GraphKnowledgeEngine",
        filtering_callback: Callable[..., tuple[FilteringResult | RetrievalResult, str]],
        max_retrieval_level: int = 2,
        summary_char_threshold=12000,
        prev_turn_meta_summary: MetaFromLastSummary = MetaFromLastSummary(0, 0),
        add_turn_only=None,
    ) -> AddTurnResult:
        if ref_knowledge_engine is not self.knowledge_engine:
            self.knowledge_engine = ref_knowledge_engine
            self.orchestrator = ConversationOrchestrator(
                conversation_engine=self.conversation_engine,
                ref_knowledge_engine=ref_knowledge_engine,
                workflow_engine=self.workflow_engine,
                llm=self.llm,
            )
        return self.orchestrator.add_conversation_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            mem_id=mem_id,
            role=role,
            content=content,
            filtering_callback=filtering_callback,
            max_retrieval_level=max_retrieval_level,
            summary_char_threshold=summary_char_threshold,
            prev_turn_meta_summary=prev_turn_meta_summary,
            add_turn_only=add_turn_only,
        )

    def add_turn_workflow_v2(self, *args, **kwargs):
        return self.orchestrator.add_conversation_turn_workflow_v2(*args, **kwargs)

    def answer_only(self, *args, **kwargs):
        return self.orchestrator.answer_only(*args, **kwargs)

    def get_conversation(self, conversation_id):
        _ = conversation_id
        return None

    def get_system_prompt(self, conversation_id: str) -> str:
        _ = conversation_id
        return "You are a helpful assistant. Answer the user using the conversation and any provided evidence."

    def get_response_model(self, conversation_id) -> Type[BaseModel]:
        _ = conversation_id
        return ConversationAIResponse

    def get_conversation_view(
        self,
        *,
        conversation_id: str,
        user_id: str | None = None,
        purpose: str = "answer",
        budget_tokens: int = 6000,
        tail_turns: int = 8,
        include_summaries: bool = True,
        include_memory_context: bool = True,
        include_pinned_kg_refs: bool = True,
        ordering_strategy: str | None = None,
    ):
        _ = user_id
        tokenizer = _ApproxTokenizer()
        eng = self.conversation_engine

        sources = ContextSources(
            conversation_engine=eng,
            tail_turns=tail_turns,
            include_summaries=include_summaries,
            include_memory_context=include_memory_context,
            include_pinned_kg_refs=include_pinned_kg_refs,
        )
        items: list[ContextItem] = sources.gather(conversation_id=conversation_id, purpose=purpose)

        sys = self.get_system_prompt(conversation_id)
        items.insert(
            0,
            ContextItem(
                role="system",
                kind="system_prompt",
                text=str(sys or ""),
                node_id=None,
                priority=0,
                pinned=True,
                max_tokens=900,
                source="system",
            ),
        )

        priced: list[ContextItem] = []
        for it in items:
            cost = tokenizer.count_tokens(it.text or "")
            priced.append(ContextItem(**{**it.__dict__, "token_cost": cost}))

        if ordering_strategy is None or ordering_strategy == "default":
            pinned_non_turn = [i for i in priced if i.pinned and i.kind != "tail_turn"]
            tail_turn_items = [i for i in priced if i.kind == "tail_turn"]

            pinned_non_turn.sort(key=lambda x: x.priority)
            tail_turn_items.sort(key=lambda x: x.priority)

            kept: list[ContextItem] = []
            dropped: list[DroppedItem] = []
            used = 0

            def _try_add(it: ContextItem) -> bool:
                nonlocal used
                if used + it.token_cost <= budget_tokens:
                    kept.append(it)
                    used += it.token_cost
                    return True

                if it.max_tokens is not None and it.max_tokens < it.token_cost:
                    new_text = it.text[: max(1, it.max_tokens * 4)]
                    new_cost = tokenizer.count_tokens(new_text)
                    if used + new_cost <= budget_tokens:
                        kept.append(ContextItem(**{**it.__dict__, "text": new_text, "token_cost": new_cost}))
                        used += new_cost
                        dropped.append(
                            DroppedItem(kind=it.kind, node_id=it.node_id, reason="compressed", token_cost=it.token_cost)
                        )
                        return True

                dropped.append(DroppedItem(kind=it.kind, node_id=it.node_id, reason="over_budget", token_cost=it.token_cost))
                return False

            for it in pinned_non_turn:
                _try_add(it)

            for it in tail_turn_items:
                _try_add(it)

            non_turn_kept = [i for i in kept if i.kind != "tail_turn"]
            turn_kept = [i for i in kept if i.kind == "tail_turn"]
            turn_kept.sort(key=lambda x: int((x.extra or {}).get("turn_index", 10**9)))
            kept = non_turn_kept + turn_kept
        else:
            iter_items = apply_ordering(items=list(priced), ordering=ordering_strategy, phase="pre_pack")

            kept = []
            dropped = []
            used = 0

            def _try_add(it: ContextItem) -> bool:
                nonlocal used
                if used + it.token_cost <= budget_tokens:
                    kept.append(it)
                    used += it.token_cost
                    return True

                if it.max_tokens is not None and it.max_tokens < it.token_cost:
                    new_text = it.text[: max(1, it.max_tokens * 4)]
                    new_cost = tokenizer.count_tokens(new_text)
                    if used + new_cost <= budget_tokens:
                        kept.append(ContextItem(**{**it.__dict__, "text": new_text, "token_cost": new_cost}))
                        used += new_cost
                        dropped.append(
                            DroppedItem(kind=it.kind, node_id=it.node_id, reason="compressed", token_cost=it.token_cost)
                        )
                        return True

                if it.kind == "system_prompt":
                    raise ValueError("System prompt alone exceeds budget")
                dropped.append(DroppedItem(kind=it.kind, node_id=it.node_id, reason="over_budget", token_cost=it.token_cost))
                return False

            for it in iter_items:
                _try_add(it)

            kept = apply_ordering(items=list(kept), ordering=ordering_strategy, phase="post_pack")

        non_turn_kept = [i for i in kept if i.kind != "tail_turn"]
        turn_kept = [i for i in kept if i.kind == "tail_turn"]
        turn_kept.sort(key=lambda x: int((x.extra or {}).get("turn_index", 10**9)))
        kept = non_turn_kept + turn_kept

        renderer = ContextRenderer()
        messages = renderer.render(kept, purpose=purpose)

        included_node_ids = tuple(sorted({i.node_id for i in kept if i.node_id}))
        included_edge_ids = tuple(sorted({e for i in kept for e in (i.edge_ids or ())}))
        included_pointer_ids = tuple(sorted({p for i in kept for p in (i.pointer_ids or ()) if p}))

        head_summary_ids = tuple(i.node_id for i in kept if i.kind == "head_summary" and i.node_id)
        tail_turn_ids_out = tuple(i.node_id for i in kept if i.kind == "tail_turn" and i.node_id)
        active_memory_context_ids = tuple(i.node_id for i in kept if i.kind == "memory_context" and i.node_id)
        pinned_kg_ref_ids = tuple(i.node_id for i in kept if i.kind == "pinned_kg_ref" and i.node_id)

        return ConversationContextView(
            conversation_id=conversation_id,
            purpose=purpose,
            messages=tuple(messages),
            token_budget=budget_tokens,
            tokens_used=used,
            items=tuple(kept),
            dropped=tuple(dropped),
            included_node_ids=included_node_ids,
            included_edge_ids=included_edge_ids,
            included_pointer_ids=included_pointer_ids,
            head_summary_ids=head_summary_ids,
            tail_turn_ids=tail_turn_ids_out,
            active_memory_context_ids=active_memory_context_ids,
            pinned_kg_ref_ids=pinned_kg_ref_ids,
        )

    def make_conversation_span(self, conversation_id):
        return Span.from_dummy_for_conversation(conversation_id)

    def persist_context_snapshot(
        self,
        *,
        conversation_id: str,
        run_id: str,
        run_step_seq: int,
        attempt_seq: int = 0,
        stage: str,
        view: PromptContext,
        model_name: str = "",
        budget_tokens: int = 0,
        tail_turn_index: int = 0,
        extra_hash_payload=None,
        llm_input_payload: dict[str, Any] | None = None,
        evidence_pack_digest: dict[str, Any] | None = None,
    ) -> str:
        eng = self.conversation_engine

        def _stable_json(obj: Any) -> str:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        def _snapshot_hash(payload: Any) -> str:
            h = hashlib.sha256()
            h.update(_stable_json(payload).encode("utf-8"))
            return h.hexdigest()

        msgs = list(getattr(view, "messages", None) or [])
        norm_msgs: list[dict[str, str]] = []
        for m in msgs:
            role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            norm_msgs.append({"role": str(role or ""), "content": str(content or "")})
        rendered_hash = _snapshot_hash(
            {
                "messages": norm_msgs,
                "llm_input_payload": llm_input_payload,
                "evidence_pack_digest": evidence_pack_digest,
                "extra_hash_payload": extra_hash_payload,
            }
        )

        used_node_ids: list[str] = []
        for it in view.items:
            nid = getattr(it, "node_id", None)
            if nid:
                used_node_ids.append(str(nid))

        char_count = sum(len(m.get("content", "")) for m in norm_msgs)
        token_count = getattr(getattr(view, "cost", None), "token_count", None)
        if token_count is None:
            token_count = getattr(view, "tokens_used", None)
        cost = ContextCost(char_count=int(char_count), token_count=(None if token_count is None else int(token_count)))

        meta_model = ContextSnapshotMetadata(
            run_id=run_id,
            run_step_seq=int(run_step_seq),
            attempt_seq=int(attempt_seq),
            stage=str(stage),
            model_name=str(model_name or ""),
            budget_tokens=int(budget_tokens or 0),
            tail_turn_index=int(tail_turn_index or 0),
            used_node_ids=list(used_node_ids),
            rendered_context_hash=str(rendered_hash),
            cost=cost,
        )

        sid = str(
            stable_id(
                "conversation.context_snapshot",
                conversation_id,
                run_id,
                stage,
                str(int(run_step_seq)),
                str(int(attempt_seq)),
            )
        )

        existing = eng.backend.node_get(ids=[sid], include=[])
        if not existing.get("ids"):
            node = ConversationNode(
                id=sid,
                label="Context Snapshot",
                type="entity",
                summary="",
                conversation_id=conversation_id,
                role="system",  # type: ignore
                turn_index=None,
                level_from_root=0,
                properties={
                    "entity_type": "context_snapshot",
                    "prompt_messages": json.dumps(norm_msgs),
                    "llm_input_payload": json.dumps(llm_input_payload or {}),
                    "evidence_pack_digest": json.dumps(evidence_pack_digest or {}),
                },
                mentions=[Grounding(spans=[self.make_conversation_span(conversation_id)])],
                metadata={
                    "entity_type": "context_snapshot",
                    "level_from_root": 0,
                    "in_conversation_chain": False,
                    "in_ui_chain": False,
                    **meta_model.to_chroma_metadata(),
                },
                domain_id=None,
                canonical_entity_id=None,
            )
            eng.add_node(node)

        scope = f"conv:{conversation_id}"
        for ordinal, nid in enumerate(used_node_ids):
            eid = str(stable_id("conversation.edge", scope, "depends_on", sid, nid, str(ordinal)))
            ex = eng.backend.edge_get(ids=[eid], include=[])
            if ex.get("ids"):
                continue
            doc_id = scope
            sp = Span.from_dummy_for_conversation(doc_id=doc_id)
            edge = ConversationEdge(
                id=eid,
                source_ids=[sid],
                target_ids=[nid],
                relation="depends_on",
                label=f"depends_on:{sid}->{nid}",
                type="relationship",
                summary=f"Context snapshot {sid} depends on node {nid}",
                doc_id=doc_id,
                mentions=[Grounding(spans=[sp])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity_type": "conversation_edge", "ordinal": ordinal},
                embedding=None,
                metadata={
                    "entity_type": "conversation_edge",
                    "ordinal": ordinal,
                    "run_id": run_id,
                    "run_step_seq": int(run_step_seq),
                    "attempt_seq": int(attempt_seq),
                    "tail_turn_index": int(tail_turn_index or 0),
                },
                source_edge_ids=[],
                target_edge_ids=[],
            )
            eng.add_edge(edge)

        return sid

    def latest_context_snapshot_node(
        self,
        *,
        conversation_id: str,
        run_id: str | None = None,
        stage: str | None = None,
    ) -> ConversationNode | None:
        where: dict[str, str] = {"entity_type": "context_snapshot"}
        if run_id is not None:
            where["run_id"] = run_id
        if stage is not None:
            where["stage"] = stage
        snaps = self.conversation_engine.query_nodes(where=where)
        if not snaps:
            return None

        def _k(n: ConversationNode):
            try:
                return int((n.metadata or {}).get("run_step_seq", 0))
            except Exception:
                return 0

        return sorted(snaps, key=_k)[-1]

    def get_context_snapshot_payload(
        self,
        *,
        snapshot_node_id: str,
    ) -> dict[str, Any]:
        got = self.conversation_engine.backend.node_get(ids=[snapshot_node_id], include=["documents", "metadatas"])
        ids = got.get("ids") or []
        if not ids:
            raise KeyError(f"context snapshot node not found: {snapshot_node_id!r}")
        doc = (got.get("documents") or [None])[0]
        if isinstance(doc, str):
            try:
                payload = json.loads(doc)
                if isinstance(payload, dict):
                    props = payload.get("properties") or {}
                    return {
                        "properties": props,
                        "metadata": (payload.get("metadata") or {}),
                    }
            except Exception:
                pass
        return {
            "properties": {},
            "metadata": (got.get("metadatas") or [{}])[0] or {},
        }

    def latest_context_snapshot_cost(
        self,
        *,
        conversation_id: str,
        stage: str | None = None,
    ) -> ContextCost | None:
        n = self.latest_context_snapshot_node(conversation_id=conversation_id, stage=stage)
        if n is None:
            return None
        meta = getattr(n, "metadata", {}) or {}
        try:
            cs = ContextSnapshotMetadata.from_chroma_metadata(meta)
            return cs.cost
        except Exception:
            return None

    def get_ai_conversation_response(
        self, conversation_id, ref_knowledge_engine=None, model_names=None
    ) -> ConversationAIResponse:
        if ref_knowledge_engine is not None and ref_knowledge_engine is not self.knowledge_engine:
            self.knowledge_engine = ref_knowledge_engine
            self.orchestrator = ConversationOrchestrator(
                conversation_engine=self.conversation_engine,
                ref_knowledge_engine=ref_knowledge_engine,
                workflow_engine=self.workflow_engine,
                llm=self.llm,
            )
        return self.answer_only(conversation_id=conversation_id, model_names=model_names)
