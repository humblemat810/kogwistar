"""Agentic answering runtime.

This module is intentionally separate from engine.py so the orchestration logic can evolve
without entangling the core storage engine.

The initial implementation focuses on:
 - creating an AgentRunAnchor node in the conversation canvas
 - performing bounded retrieval from the knowledge graph
 - (optionally) using tools in the future (extension point)
 - selecting *used* evidence via an LLM step
 - projecting *used* evidence into the conversation canvas as pointer nodes
 - generating the final assistant response

Design notes:
 - The knowledge graph is mutable; therefore, projection stores a minimal snapshot hash.
 - Canvas projection is idempotent via deterministic pointer ids.
 - This implementation keeps the trace store minimal; it can be replaced with a richer
   orchestration trace/control engine later.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Iterable, Optional, Sequence, Type

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from .models import ConversationNode, ConversationEdge, Grounding, Span


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_hash(payload: Any) -> str:
    """Compute a stable hash for a snapshot payload."""
    h = hashlib.sha256()
    h.update(_stable_json(payload).encode("utf-8"))
    return h.hexdigest()


def pointer_id(*, scope: str, pointer_kind: str, target_kind: str, target_id: str) -> str:
    """Deterministic pointer id.

    Storage-agnostic: does not mention backend collection/table.
    """
    # Keep it readable; escape is handled by stable json if needed later.
    return f"ptr|scope:{scope}|pk:{pointer_kind}|tk:{target_kind}|id:{target_id}"


def edge_id(*, scope: str, rel: str, src: str, dst: str) -> str:
    """Deterministic edge id for idempotent linking in the canvas."""
    return f"e|scope:{scope}|rel:{rel}|src:{src}|dst:{dst}"


class EvidenceSelection(BaseModel):
    """LLM output selecting what is actually used for answering."""

    used_node_ids: list[str] = Field(default_factory=list, description="Knowledge node ids used")
    used_edge_ids: list[str] = Field(default_factory=list, description="Knowledge edge ids used (optional)")
    reasoning: str = Field("", description="Short reasoning for selection")


class AnswerModel(BaseModel):
    text: str = Field(..., description="Final assistant response")


@dataclass
class AgentConfig:
    max_candidates: int = 20
    max_used: int = 8
    max_retrieval_level: int = 4

from .engine import GraphKnowledgeEngine
class AgenticAnsweringAgent:
    """Agent that answers within a conversation canvas using a separate knowledge engine."""

    def __init__(
        self,
        *,
        conversation_engine: GraphKnowledgeEngine,
        knowledge_engine: GraphKnowledgeEngine,
        llm: BaseChatModel,
        config: Optional[AgentConfig] = None,
    ):
        self.conversation_engine = conversation_engine
        self.knowledge_engine = knowledge_engine
        self.llm = llm
        self.config = config or AgentConfig()

    # ----------------------------
    # Public entrypoint
    # ----------------------------
    def answer(self, *, conversation_id: str) -> dict[str, Any]:
        """Run one agentic answering pass.

        Returns a dict with keys: run_node_id, assistant_turn_node_id, used_node_ids.
        """
        # 1) Fetch conversation state (engine-specific hooks)
        conversation = self.conversation_engine.get_conversation(conversation_id)
        system_prompt = self.conversation_engine.get_system_prompt(conversation_id)
        last_user_turn = self._get_last_user_text(conversation)
        if not last_user_turn:
            raise ValueError("No user message found in conversation")

        # 2) Create run anchor in canvas
        run_id = f"run_{int(time.time()*1000)}"
        run_node_id = self._ensure_run_anchor(conversation_id=conversation_id, run_id=run_id)

        # 3) Retrieve candidate KG nodes (bounded)
        candidates = self._retrieve_candidates(last_user_turn)

        # 4) LLM selects used evidence
        selection = self._select_used_evidence(
            system_prompt=system_prompt,
            question=last_user_turn,
            candidates=candidates,
        )

        # 5) Project used evidence to canvas (idempotent)
        used_node_ids = selection.used_node_ids[: self.config.max_used]
        projected_pointer_ids = []
        for kid in used_node_ids:
            pid = self._project_kg_node(
                conversation_id=conversation_id,
                run_node_id=run_node_id,
                kg_node_id=kid,
                provenance_span=Span.from_dummy_for_conversation(),
            )
            projected_pointer_ids.append(pid)

        # 6) Generate final answer (LLM)
        answer_text = self._generate_answer(
            system_prompt=system_prompt,
            question=last_user_turn,
            used_nodes=used_node_ids,
        )

        # 7) Persist assistant response as conversation node and link to run
        assistant_turn_node_id = self._add_assistant_turn(
            conversation_id=conversation_id,
            content=answer_text,
            provenance_span=Span.from_dummy_for_conversation(),
        )
        self._link_run_to_response(
            conversation_id=conversation_id,
            run_node_id=run_node_id,
            response_node_id=assistant_turn_node_id,
            provenance_span=Span.from_dummy_for_conversation(),
        )

        return {
            "run_node_id": run_node_id,
            "assistant_turn_node_id": assistant_turn_node_id,
            "used_node_ids": used_node_ids,
            "projected_pointer_ids": projected_pointer_ids,
        }

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _get_last_user_text(self, conversation: Any) -> str:
        """Best-effort: conversation may be a list[dict] or a custom object."""
        if conversation is None:
            return ""
        # Common shapes: list of {role, content}
        if isinstance(conversation, list):
            for msg in reversed(conversation):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content") or "")
        # Fallback: if engine returns string
        if isinstance(conversation, str):
            return conversation
        # Fallback: try attribute
        text = getattr(conversation, "last_user_text", None)
        return str(text or "")

    def _retrieve_candidates(self, question: str) -> list[dict[str, Any]]:
        emb = self.knowledge_engine.iterative_defensive_emb(question)
        res = self.knowledge_engine.node_collection.query(
            query_embeddings=[emb],
            n_results=self.config.max_candidates,
            where={"level_from_root": {"$lte": self.config.max_retrieval_level}},
        )
        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        out: list[dict[str, Any]] = []
        for i, mid in enumerate(ids):
            out.append(
                {
                    "id": str(mid),
                    "label": (metas[i] or {}).get("label"),
                    "summary": (metas[i] or {}).get("summary"),
                    "doc": docs[i],
                }
            )
        return out

    def _select_used_evidence(self, *, system_prompt: str, question: str, candidates: Sequence[dict[str, Any]]) -> EvidenceSelection:
        # Compact candidate representation
        cand_lines = []
        for c in candidates:
            cand_lines.append(f"- {c['id']}: {c.get('label','')} | {c.get('summary','')}")
        cand_text = "\n".join(cand_lines)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt or "You are a helpful assistant."),
                (
                    "human",
                    """You are selecting which knowledge items are actually USED to answer the user.

User question:
{question}

Candidate knowledge nodes (id | label | summary):
{candidates}

Return JSON with keys: used_node_ids (list of ids), used_edge_ids (optional), reasoning (short).
Select at most {max_used} node ids.
""",
                ),
            ]
        )
        chain = prompt | self.llm.with_structured_output(EvidenceSelection)
        sel: EvidenceSelection = chain.invoke({"question": question, "candidates": cand_text, "max_used": self.config.max_used})
        # Defensive: ensure subset
        cand_ids = {c["id"] for c in candidates}
        sel.used_node_ids = [i for i in sel.used_node_ids if i in cand_ids][: self.config.max_used]
        return sel

    def _generate_answer(self, *, system_prompt: str, question: str, used_nodes: list[str]) -> str:
        # Pull minimal summaries (lazy, bounded)
        ctx_lines = []
        for nid in used_nodes[: self.config.max_used]:
            got = self.knowledge_engine.node_collection.get(ids=[nid], include=["metadatas"])
            metas = got.get("metadatas") or []
            meta = metas[0] if metas else {}
            ctx_lines.append(f"- {nid}: {meta.get('label','')} | {meta.get('summary','')}")
        ctx = "\n".join(ctx_lines)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt or "You are a helpful assistant."),
                (
                    "human",
                    """Answer the user. Use the provided evidence if helpful.

User question:
{question}

Evidence (id | label | summary):
{ctx}
""",
                ),
            ]
        )
        res = (prompt | self.llm).invoke({"question": question, "ctx": ctx})
        # langchain message
        return str(getattr(res, "content", res))

    def _ensure_run_anchor(self, *, conversation_id: str, run_id: str) -> str:
        scope = f"conv:{conversation_id}"
        rid = pointer_id(scope=scope, pointer_kind="agent_run", target_kind="run", target_id=run_id)
        existing = self.conversation_engine.node_collection.get(ids=[rid])
        if existing.get("ids"):
            return rid

        sp = Span.from_dummy_for_conversation()
        node = ConversationNode(
            id=rid,
            label=f"Agent Run {run_id}",
            type="agent_run",
            summary=f"Agent run anchor {run_id}",
            conversation_id=conversation_id,
            role="system",  # type: ignore
            turn_index=None,
            properties={"run_id": run_id, "entity-type": "agent_run"},
            mentions=[Grounding(spans=[sp])],
            metadata={"level_from_root": 0, "entity_type": "agent_run", "char_distance_from_last_summary": 0, "turn_distance_from_last_summary": 0},
            domain_id=None,
            canonical_entity_id=None,
        )
        self.conversation_engine.add_node(node)
        return rid

    def _project_kg_node(
        self,
        *,
        conversation_id: str,
        run_node_id: str,
        kg_node_id: str,
        provenance_span: Span,
    ) -> str:
        scope = f"conv:{conversation_id}"
        pid = pointer_id(scope=scope, pointer_kind="kg_node", target_kind="node", target_id=kg_node_id)

        # If exists, still ensure run->evidence edge exists (idempotent)
        existing = self.conversation_engine.node_collection.get(ids=[pid])
        if not existing.get("ids"):
            kg = self.knowledge_engine.node_collection.get(ids=[kg_node_id], include=["metadatas"])
            meta = (kg.get("metadatas") or [{}])[0] or {}
            snap = {
                "entity_id": kg_node_id,
                "label": meta.get("label"),
                "summary": meta.get("summary"),
                "type": meta.get("type"),
                "canonical_entity_id": meta.get("canonical_entity_id"),
            }
            sh = snapshot_hash(snap)
            node = ConversationNode(
                id=pid,
                label=f"Ref {meta.get('label') or kg_node_id}",
                type="reference_pointer",
                summary=str(meta.get("summary") or ""),
                conversation_id=conversation_id,
                role="system",  # type: ignore
                turn_index=None,
                properties={
                    "target_namespace": "kg",
                    "target_kind": "node",
                    "target_id": kg_node_id,
                    "snapshot_hash": sh,
                    "entity-type": "knowledge_reference",
                },
                mentions=[Grounding(spans=[provenance_span])],
                metadata={"level_from_root": 0, "entity_type": "knowledge_reference", "char_distance_from_last_summary": 0, "turn_distance_from_last_summary": 0},
                domain_id=None,
                canonical_entity_id=None,
            )
            self.conversation_engine.add_node(node)

        # Link run -> evidence
        eid = edge_id(scope=scope, rel="used_evidence", src=run_node_id, dst=pid)
        ex_edge = self.conversation_engine.edge_collection.get(ids=[eid])
        if not ex_edge.get("ids"):
            edge = ConversationEdge(
                id=eid,
                source_ids=[run_node_id],
                target_ids=[pid],
                relation="used_evidence",
                label="used_evidence",
                type="relationship",
                summary="Agent used this evidence",
                doc_id=f"conv:{conversation_id}",
                mentions=[Grounding(spans=[provenance_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={"entity-type": "conversation_edge"},
                embedding=None,
                metadata={},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(edge)
        return pid

    def _add_assistant_turn(self, *, conversation_id: str, content: str, provenance_span: Span) -> str:
        # Minimal assistant turn node
        nid = pointer_id(scope=f"conv:{conversation_id}", pointer_kind="turn", target_kind="assistant", target_id=str(int(time.time()*1000)))
        node = ConversationNode(
            id=nid,
            label="Assistant",
            type="conversation_turn",
            summary=content[:2000],
            conversation_id=conversation_id,
            role="assistant",  # type: ignore
            turn_index=None,
            properties={"content": content},
            mentions=[Grounding(spans=[provenance_span])],
            metadata={"level_from_root": 0, "entity_type": "assistant_turn", "char_distance_from_last_summary": 0, "turn_distance_from_last_summary": 0},
            domain_id=None,
            canonical_entity_id=None,
        )
        self.conversation_engine.add_node(node)
        return nid

    def _link_run_to_response(self, *, conversation_id: str, run_node_id: str, response_node_id: str, provenance_span: Span) -> None:
        scope = f"conv:{conversation_id}"
        eid = edge_id(scope=scope, rel="generated", src=run_node_id, dst=response_node_id)
        ex = self.conversation_engine.edge_collection.get(ids=[eid])
        if ex.get("ids"):
            return
        edge = ConversationEdge(
            id=eid,
            source_ids=[run_node_id],
            target_ids=[response_node_id],
            relation="generated",
            label="generated",
            type="relationship",
            summary="Agent run generated assistant response",
            doc_id=f"conv:{conversation_id}",
            mentions=[Grounding(spans=[provenance_span])],
            domain_id=None,
            canonical_entity_id=None,
            properties={"entity-type": "conversation_edge"},
            embedding=None,
            metadata={},
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.add_edge(edge)
