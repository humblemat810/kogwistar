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
import pathlib
import re
import time
import base64

from typing import Any, Callable, Iterable, Optional, Self, Sequence, Type, TypeVar

from pydantic import BaseModel, Field
from graph_knowledge_engine.llm_tasks import (
    AnswerWithCitationsTaskRequest,
    LLMTaskSet,
    RepairCitationsTaskRequest,
)
from typing import TYPE_CHECKING

from .models import ContextSnapshotMetadata, ConversationEdge, ConversationNode, MetaFromLastSummary
from .policy import get_chat_tail
if TYPE_CHECKING:
    from ..runtime import WorkflowEdgeInfo

from .conversation_state_contracts import ConversationWorkflowState

from ..engine_core.models import (
    Grounding,
    Span,
    Node,
    ContextCost,
)
from ..runtime.models import StepRunResult
BaseM = TypeVar("BaseM", bound=BaseModel)

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_hash(payload: Any) -> str:
    """Compute a stable hash for a snapshot payload."""
    h = hashlib.sha256()
    h.update(_stable_json(payload).encode("utf-8"))
    return h.hexdigest()


def context_messages_hash(messages: Sequence[Any]) -> str:
    """Stable hash of a list of LLM messages.

    Works for both dict-based messages and ContextMessage objects.
    """
    norm: list[dict[str, Any]] = []
    for m in messages or []:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        norm.append({"role": str(role or ""), "content": str(content or "")})
    return snapshot_hash({"messages": norm})

def deterministic_id(prefix: str, fingerprint: dict, bits: int = 96) -> str:
    s = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(s).digest()
    nbytes = bits // 8
    short = h[:nbytes]
    token = base64.b32encode(short).decode("ascii").rstrip("=")
    return f"{prefix}:{token.lower()}"

def pointer_id(*, scope: str, pointer_kind: str, target_kind: str, target_id: str) -> str:
    fp = {
        "scope": scope,
        "pointer_kind": pointer_kind,
        "target_kind": target_kind,
        "target_id": target_id,  # INCLUDED IN FINGERPRINT
    }
    return deterministic_id("ptr", fp)


def edge_id(*, scope: str, rel: str, src: str, dst: str) -> str:
    fp = {
        "scope": scope,
        "rel": rel,
        "src": src,
        "dst": dst,
    }
    return deterministic_id("e", fp)


class EvidenceSelection(BaseModel):
    """LLM output selecting what is actually used for answering."""

    used_node_ids: list[str] = Field(default_factory=list, description="Knowledge node ids used")
    used_edge_ids: list[str] = Field(default_factory=list, description="Knowledge edge ids used (optional)")
    reasoning: str = Field("", description="Short reasoning for selection")


class AnswerModel(BaseModel):
    text: str = Field(..., description="Final assistant response")


class SpanRef(BaseModel):
    """Reference to a specific span within a node's materialized mentions/spans.

    Indices are ephemeral and only guaranteed valid for the evidence pack used in the same run.
    """
    source_node_id: str = Field(..., description="Knowledge node id that owns the mention/span")
    mention_index: int = Field(..., ge=0, description="Index into mentions list")
    span_index: int = Field(..., ge=0, description="Index into spans list within the mention")


class ClaimWithCitations(BaseModel):
    """A single atomic claim with supporting citations."""
    claim: str = Field(..., description="A single factual claim or assertion")
    citations: list[SpanRef] = Field(default_factory=list, description="Supporting span references")


class AnswerWithCitations(BaseModel):
    """Answer text plus claim-level span citations."""
    text: str = Field(..., description="Final assistant response")
    reasoning: str = Field(..., description="Reasoning")
    claims: list[ClaimWithCitations] = Field(default_factory=list, description="Key claims with citations")
from typing import Callable, TypeVar, ParamSpec, cast
from joblib import Memory

P = ParamSpec("P")
R = TypeVar("R")
def cached(memory: Memory, fn: Callable[P, R], *args, **kwargs) -> Callable[P, R]:
    return cast(Callable[P, R], memory.cache(fn, *args, **kwargs))
class AnswerEvaluation(BaseModel):
    """Post-answer evaluation for sufficiency and whether more info is needed."""
    is_sufficient: bool = Field(..., description="Whether current evidence is sufficient to answer")
    needs_more_info: bool = Field(..., description="Whether we should retrieve/expand and answer again")
    missing_aspects: list[str] = Field(default_factory=list, description="What is missing if more info is needed")
    notes: str = Field("", description="Optional short rationale")

@dataclass
class AgentConfig:
    # Retrieval
    max_candidates: int = 20
    max_used: int = 8
    max_retrieval_level: int = 4

    # Control flow
    max_iter: int = 2  # bounded answering loop

    # Evidence selection strategy
    evidence_selector: str = "llm"  # "llm" or "bm25"

    # Materialization depth for OCR / structured docs (policy may escalate shallow->deep)
    materialize_depth: str = "shallow"  # "shallow" or "deep"

    # Budget knobs for materialization (kept simple; your resolvers can interpret these)
    max_chars_per_item: int = 900
    max_total_chars: int = 12000
from ..engine_core.engine import GraphKnowledgeEngine

class AgenticAnsweringAgent:
    """Agent that answers within a conversation canvas using a separate knowledge engine."""
    @staticmethod
    def _select_used_evidence_entry(
        agent: "AgenticAnsweringAgent",
        *,
        system_prompt: str,
        question: str,
        candidates: list[Any],
    ):
        # delegate to the real method (LLM call)
        return agent._select_used_evidence(
            system_prompt=system_prompt,
            question=question,
            candidates=candidates,
        )

    def select_used_evidence_cached(
        self,
        *,
        system_prompt: str,
        question: str,
        candidates: list[Any],
        cache_dir: str | None,
    ):
        if not cache_dir:
            return EvidenceSelection.model_validate(self._select_used_evidence(system_prompt=system_prompt, question=question, candidates=candidates))
        if not candidates:
            return EvidenceSelection(reasoning="No candidates given for selection.")
        mem = Memory(location=cache_dir, verbose=0)
        cached_fn = cached(mem, self._select_used_evidence_entry, ignore=["agent"])
        payload = cached_fn(self, system_prompt=system_prompt, question=question, candidates=candidates)
        return EvidenceSelection.model_validate(payload)
        # from utils.pydanic_model_consumer_wrapper import cache_pydantic_structured
        # cached_entry = cache_pydantic_structured(
        #     memory=mem,
        #     model=EvidenceSelection,
        #     fn=self._select_used_evidence_entry,
        #     ignore=["agent"],
        #     # dump_exclude={"raw"},  # if you add raw above
        # )
        # return cache_pydantic_structured()
    def __init__(
        self,
        *,
        conversation_engine: GraphKnowledgeEngine,
        knowledge_engine: GraphKnowledgeEngine,
        llm_tasks: LLMTaskSet,
        config: Optional[AgentConfig] = None,
        cache_dir: str | None = str((pathlib.Path()/ ".joblib" / "agentic_answering").absolute())
    ):
        self.conversation_engine = conversation_engine
        self.knowledge_engine = knowledge_engine
        self.llm_tasks = llm_tasks
        self.config = config or AgentConfig()
        self.cache_dir = cache_dir
        self.max_retry = 3

    def _provider_label(self, task_name: str) -> str:
        return str(getattr(self.llm_tasks.provider_hints, f"{task_name}_provider", "") or "")
    # ----------------------------
    # Public entrypoint
    # ----------------------------
    def answer(self, *, conversation_id: str, user_id=None, prev_turn_meta_summary:MetaFromLastSummary) -> dict[str, Any]:
        """Run bounded agentic answering with evidence selection + optional citation picking.
        Non runtime workflow
        Must Enforce Builder pattern in this function
        build from each LLM call cached
        Returns a dict with keys (best-effort):
          - run_node_id
          - assistant_turn_node_id
          - assistant_text
          - used_node_ids
          - claim_citations (optional, claim-level SpanRef lists)
          - evaluation (optional)
          - projected_pointer_ids
        """
        # 1) Fetch conversation state
        from .service import ConversationService

        svc = ConversationService.from_engine(
            self.conversation_engine,
            knowledge_engine=self.knowledge_engine,
            workflow_engine=getattr(self.conversation_engine, "workflow_engine", None),
        )
        view = svc.get_conversation_view(
            conversation_id=conversation_id,
            user_id=user_id,
            purpose="answer",
            budget_tokens=6000,
        )
        messages = view.messages
        question = self._get_last_user_text(messages)
        system_prompt = svc.get_system_prompt(conversation_id)
        if not question:
            raise ValueError("No user message found in conversation")

        # 2) Create run anchor in canvas
        run_id = f"run_{int(time.time()*1000)}"
        run_node_id = self._ensure_run_anchor(conversation_id=conversation_id, run_id=run_id)

        # Phase 2B: deterministic step identity for snapshot nodes (local to this agent run).
        run_step_seq = 0
        attempt_seq = 0

        # Policy knobs that can escalate across iterations
        max_candidates = self.config.max_candidates
        materialize_depth = self.config.materialize_depth

        last_eval: AnswerEvaluation | None = None
        last_answer: AnswerWithCitations | None = None
        last_used: list[str] = []
        last_projected: list[str] = []

        for it in range(max(1, self.config.max_iter)):
            # 3) Retrieve candidate KG nodes (bounded)
            candidates = self._retrieve_candidates(question)[:max_candidates]

            # 4) Select used evidence (LLM or cheap fallback)
            if (self.config.evidence_selector or "llm").lower() == "bm25":
                selection = self._select_used_evidence_bm25(question=question, candidates=candidates)
            else:
                # Context snapshot before evidence-selection LLM call.
                self._persist_context_snapshot(
                    conversation_id=conversation_id,
                    run_id=run_id,
                    run_step_seq=run_step_seq,
                    attempt_seq=attempt_seq,
                    stage="select_used_evidence",
                    view=view,
                    model_name=self._provider_label("filter_candidates"),
                    budget_tokens=int(getattr(view, "token_budget", 0) or 0),
                    tail_turn_index=int(prev_turn_meta_summary.tail_turn_index or 0),
                    extra_hash_payload={
                        "question": question,
                        "candidate_ids": [c.get("node_id") for c in (candidates or []) if isinstance(c, dict) and c.get("node_id")],
                    },
                    llm_input_payload={
                        "system_prompt": system_prompt,
                        "question": question,
                        "candidate_ids": [c.get("node_id") for c in (candidates or []) if isinstance(c, dict) and c.get("node_id")],
                    },
                )
                run_step_seq += 1
                selection = self.select_used_evidence_cached(
                    system_prompt=system_prompt,
                    question=question,
                    candidates=candidates,
                    cache_dir = self.cache_dir
                )

            used_node_ids = selection.used_node_ids[: self.config.max_used]
            used_edge_ids = list(getattr(selection, "used_edge_ids", []) or [])
            last_used = used_node_ids
            from ..utils.pydanic_model_consumer_wrapper import cache_pydantic_structured
            from joblib import Memory
            
            # 5) Materialize evidence pack for answering + citation picking
            mem=Memory(location = str((pathlib.Path(".joblib")/"_materialize_evidence_pack").absolute()))
            cached_call = cache_pydantic_structured(fn = self._materialize_evidence_pack,
                                                    memory = mem,
                                                    model = None,
                                                    ignore = ['agent']
                                                    )
            evidence_pack = cached_call(
                agent = self,
                node_ids=used_node_ids,
                edge_ids=used_edge_ids,
                depth=materialize_depth,
                max_chars_per_item=self.config.max_chars_per_item,
                max_total_chars=self.config.max_total_chars,
            )

            # Digest is persisted inside ContextSnapshot so we can re-materialize later.
            from .models import EvidencePackDigest
            evidence_digest = EvidencePackDigest(
                node_ids=list(used_node_ids),
                edge_ids=list(used_edge_ids or []),
                depth=str(materialize_depth),
                max_chars_per_item=int(self.config.max_chars_per_item),
                max_total_chars=int(self.config.max_total_chars),
                evidence_pack_hash=str(snapshot_hash(evidence_pack)),
            ).model_dump(mode="python")
            # cached_entry = cache_pydantic_structured(
            #     memory=mem,
            #     model=FakeSelection,
            #     fn=FakeAgent.entry,
            #     ignore=["agent", "out_model"],
            #     # dump_exclude={"raw"},  # if you add raw above
            # )
            # evidence_pack = self._materialize_evidence_pack(
            #     self,
            #     node_ids=used_node_ids,
            #     depth=materialize_depth,
            #     max_chars_per_item=self.config.max_chars_per_item,
            #     max_total_chars=self.config.max_total_chars,
            # )

            # 6) Generate answer with claim-level citations (SpanRef indices into evidence_pack)
            mem=Memory(location = str((pathlib.Path(".joblib")/"_generate_answer_with_citations").absolute()))
            cached_call = cache_pydantic_structured(fn = self._generate_answer_with_citations,
                                                    memory = mem,
                                                    model = None,
                                                    ignore = ["agent"]
                                                    )            
            # Context snapshot before answer-with-citations LLM call.
            self._persist_context_snapshot(
                conversation_id=conversation_id,
                run_id=run_id,
                run_step_seq=run_step_seq,
                attempt_seq=attempt_seq,
                stage="generate_answer_with_citations",
                view=view,
                model_name=self._provider_label("answer_with_citations"),
                budget_tokens=int(getattr(view, "token_budget", 0) or 0),
                tail_turn_index=int(prev_turn_meta_summary.tail_turn_index or 0),
                extra_hash_payload={
                    "question": question,
                    "used_node_ids": used_node_ids,
                    "evidence_pack_hash": snapshot_hash(evidence_pack),
                },
                llm_input_payload={
                    "system_prompt": system_prompt,
                    "question": question,
                },
                evidence_pack_digest=evidence_digest,
            )
            run_step_seq += 1
            ans_json = cached_call(
                agent = self,
                system_prompt=system_prompt,
                question=question,
                evidence_pack=evidence_pack,
                used_node_ids=used_node_ids,
                out_model_schema = AnswerWithCitations.model_json_schema(),
                out_model = AnswerWithCitations
            )
            ans = AnswerWithCitations.model_validate(ans_json)
            # 6b) Validate / repair citations (bounded)
            
            cached_call3 = cache_pydantic_structured(fn = self._validate_or_repair_citations,
                                                    memory = mem,
                                                    model = None,
                                                    ignore = ["agent", "answer_in_model"]
                                                    )
            # Context snapshot before citation repair (may call LLM on invalid citations).
            self._persist_context_snapshot(
                conversation_id=conversation_id,
                run_id=run_id,
                run_step_seq=run_step_seq,
                attempt_seq=attempt_seq,
                stage="validate_or_repair_citations",
                view=view,
                model_name=self._provider_label("repair_citations"),
                budget_tokens=int(getattr(view, "token_budget", 0) or 0),
                tail_turn_index=int(prev_turn_meta_summary.tail_turn_index or 0),
                extra_hash_payload={
                    "question": question,
                    "used_node_ids": used_node_ids,
                    "answer_hash": snapshot_hash(ans.model_dump()),
                },
                llm_input_payload={
                    "system_prompt": system_prompt,
                    "question": question,
                    "answer_text": getattr(ans, "text", None),
                },
                evidence_pack_digest=evidence_digest,
            )
            run_step_seq += 1
            ans = cached_call3(
                agent = self,
                system_prompt=system_prompt,
                question=question,
                evidence_pack=evidence_pack,
                used_node_ids=used_node_ids,
                answer=ans.model_dump(),
                answer_in_model = AnswerWithCitations
                # out_model_schema = AnswerWithCitations.model_json_schema(),
                # out_model = AnswerWithCitations
            )
            ans = AnswerWithCitations.model_validate(ans)
            last_answer = ans

            # 7) Evaluate sufficiency / need-more-info
            cached_call3 = cache_pydantic_structured(fn = self._evaluate_answer,
                                                    memory = mem,
                                                    model = None,
                                                    ignore = ["agent", "out_model"]
                                                    )
            # Context snapshot before evaluation LLM call.
            self._persist_context_snapshot(
                conversation_id=conversation_id,
                run_id=run_id,
                run_step_seq=run_step_seq,
                attempt_seq=attempt_seq,
                stage="evaluate_answer",
                view=view,
                model_name=self._provider_label("answer_with_citations"),
                budget_tokens=int(getattr(view, "token_budget", 0) or 0),
                tail_turn_index=int(prev_turn_meta_summary.tail_turn_index or 0),
                extra_hash_payload={
                    "question": question,
                    "used_node_ids": used_node_ids,
                    "answer_text": ans.text,
                },
                llm_input_payload={
                    "system_prompt": system_prompt,
                    "question": question,
                    "answer_text": ans.text,
                },
                evidence_pack_digest=evidence_digest,
            )
            run_step_seq += 1
            last_eval = self._evaluate_answer(
                agent = self,
                system_prompt=system_prompt,
                question=question,
                answer_text=ans.text,
                used_node_ids=used_node_ids,
                evidence_pack=evidence_pack,
                out_model_schema=AnswerEvaluation.model_json_schema(),
                out_model=AnswerEvaluation
            )
            last_eval = AnswerEvaluation.model_validate(last_eval)
            if not last_eval.needs_more_info or last_eval.is_sufficient:
                break

            # Escalation: deepen materialization + widen candidates for next iter
            materialize_depth = "deep"
            max_candidates = min(max_candidates * 2, 200)

        # # 8) DEMO, uncommment to demo
        # 
        # Project used evidence to canvas (idempotent) using final selection
        # projected_pointer_ids: list[str] = []
        # for kid in last_used:
        #     pid = self._project_kg_node(
        #         conversation_id=conversation_id,
        #         run_node_id=run_node_id,
        #         kg_node_id=kid,
        #         provenance_span=Span.from_dummy_for_conversation(),
        #     )
        #     projected_pointer_ids.append(pid)
        # last_projected = projected_pointer_ids

        # 9) Persist assistant response as conversation node and link to run
        assistant_text = (last_answer.text if last_answer else "")
        # tail_turn = self.conversation_engine.conversation.get_conversation_tail(conversation_id, prev_turn_meta_summary.tail_turn_index)
        
        # if tail_turn is None or tail_turn.turn_index is None:
            # raise Exception("no tail turn with index found when answering")
        tail_turn_index = prev_turn_meta_summary.tail_turn_index
        assistant_turn_node_id, assistant_turn_node = self._add_assistant_turn(
            conversation_id=conversation_id,
            content=assistant_text,
            provenance_span=Span.from_dummy_for_conversation(),
            turn_index =  tail_turn_index+1,
            prev_turn_meta_summary=prev_turn_meta_summary
        )
        # duplicated, unnecessary to add below !warning
        # prev_turn_meta_summary.tail_turn_index = tail_turn_index
        # prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len()
        # prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        self._link_run_to_response(
            conversation_id=conversation_id,
            run_node_id=run_node_id,
            response_node_id=assistant_turn_node_id,
            used_node_ids=last_used,
            provenance_span = Span.from_dummy_for_conversation(),
            prev_turn_meta_summary=prev_turn_meta_summary,
        )

        return {
            "run_node_id": run_node_id,
            "assistant_turn_node_id": assistant_turn_node_id,
            "assistant_text": assistant_text,
            "used_node_ids": last_used,
            "projected_pointer_ids": last_projected,
            "claim_citations": (last_answer.model_dump() if last_answer else {}),
            "evaluation": (last_eval.model_dump() if last_eval else {}),
            "turn_node_dump" : assistant_turn_node.model_dump()
        }

    # ----------------------------
    # Workflow-v2 entrypoint
    # ----------------------------
    def answer_workflow_v2(
        self,
        *,
        conversation_id: str,
        user_id: str | None = None,
        prev_turn_meta_summary: MetaFromLastSummary,
        workflow_engine: "GraphKnowledgeEngine | None" = None,
        workflow_id: str = "agentic_answering.v2",
        run_id: str | None = None,
        # quick-fix for nested runs: reuse outer trace emitter when available
        events: Any | None = None,
        trace: bool = True,
        cancel_requested: Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        """Run agentic answering using the workflow runtime.

        This keeps the same object (no separate V2 class), but routes execution
        through `WorkflowRuntime` + resolvers.
        """
        workflow_engine = workflow_engine or self.conversation_engine

        # Ensure design exists.
        from ..conversation.designer import AgenticAnsweringWorkflowDesigner
        def predicate_always(workflow_info: WorkflowEdgeInfo, state: ConversationWorkflowState, last_result: StepRunResult):
            return True
        
        def aa_should_iterate(workflow_info: WorkflowEdgeInfo, state: ConversationWorkflowState, last_result: StepRunResult):
            return bool(state.get("should_iterate"))
        predicate_registry = {
            "always": predicate_always, #lambda st, r: True,
            "aa_should_iterate": aa_should_iterate, #lambda st, r: bool(st.get("should_iterate")),
        }

        designer = AgenticAnsweringWorkflowDesigner(workflow_engine=workflow_engine, predicate_registry=predicate_registry)
        designer.ensure_answer_flow(workflow_id=workflow_id, mode="full")

        # Resolver registry (handlers live in workflow/resolvers.py)
        from .resolvers import MappingStepResolver, default_resolver

        class AgenticStepResolver(MappingStepResolver):
            def __init__(self) -> None:
                super().__init__(handlers=dict(default_resolver.handlers), default=default_resolver.default)

        resolve_step = AgenticStepResolver()

        # Runtime
        from ..runtime.runtime import WorkflowRuntime

        runtime = WorkflowRuntime(
            workflow_engine=workflow_engine,
            conversation_engine=self.conversation_engine,
            step_resolver=resolve_step,
            predicate_registry=predicate_registry,
            checkpoint_every_n_steps=1,
            max_workers=1,
            # nested-safety: share outer emitter when provided, and/or disable trace sink creation
            events=events,
            trace=trace,
            cancel_requested=cancel_requested,
        )

        # Choose a turn_node_id for checkpoint/tracing.
        try:
            tail = get_chat_tail(self.conversation_engine, conversation_id=conversation_id)
        except Exception:
            tail = None
        if tail is None:
            raise ValueError(f"conversation_id={conversation_id!r} has no tail node")
        turn_node_id = str(getattr(tail, "id", None) or getattr(tail, "node_id", None) or "")
        if not turn_node_id:
            raise ValueError("Failed to infer turn_node_id from conversation tail")

        # Initial state: keep schema jsonable, stash heavy objects in deps.
        init_state: dict[str, Any] = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "turn_node_id": turn_node_id,
            "run_id": run_id,
            "iter_idx": 0,
            "max_iter": int(self.config.max_iter),
            "max_candidates": int(self.config.max_candidates),
            "materialize_depth": str(self.config.materialize_depth),
            "should_iterate": False,
            "_deps": {
                "agent": self,
                "conversation_engine": self.conversation_engine,
                "knowledge_engine": self.knowledge_engine,
                "llm_tasks": self.llm_tasks,
                "prev_turn_meta_summary": prev_turn_meta_summary,
            },
        }

        run_result = runtime.run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            turn_node_id=turn_node_id,
            initial_state=init_state,
            run_id=(run_id or f"agentic_answer|{turn_node_id}"),
        )
        final_state, rid = run_result.final_state, run_result.run_id

        # Update caller's prev_turn_meta_summary in-place.
        mts = final_state.get("prev_turn_meta_summary") or {}
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary = int(
            mts.get("prev_node_char_distance_from_last_summary", prev_turn_meta_summary.prev_node_char_distance_from_last_summary)
        )
        prev_turn_meta_summary.prev_node_distance_from_last_summary = int(
            mts.get("prev_node_distance_from_last_summary", prev_turn_meta_summary.prev_node_distance_from_last_summary)
        )
        prev_turn_meta_summary.tail_turn_index = int(mts.get("tail_turn_index", prev_turn_meta_summary.tail_turn_index))

        out = final_state.get("agentic_answering_result") or {}
        out["workflow_run_id"] = rid
        out["workflow_id"] = workflow_id
        out["workflow_status"] = getattr(run_result, "status", "succeeded")
        return out

    def _get_last_user_text(self, conversation: Any) -> str:
        if conversation is None:
            return ""

        # NEW: handle ContextMessage objects
        if isinstance(conversation, (list, tuple)):
            for msg in reversed(conversation):
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", None)
                if role == "user" and content:
                    return str(content)

                # keep old dict support too
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content") or "")

        if isinstance(conversation, str):
            return conversation

        return str(getattr(conversation, "last_user_text", "") or "")

    def _retrieve_candidates(self, question: str) -> list[dict[str, Any]]:
        if hasattr(self.knowledge_engine, "embed") and hasattr(self.knowledge_engine.embed, "iterative_defensive_emb"):
            emb = self.knowledge_engine.embed.iterative_defensive_emb(question)
        elif hasattr(self.knowledge_engine, "iterative_defensive_emb"):
            emb = self.knowledge_engine.iterative_defensive_emb(question)
        elif hasattr(self.knowledge_engine, "_iterative_defensive_emb"):
            emb = self.knowledge_engine._iterative_defensive_emb(question)
        else:
            raise AttributeError("knowledge engine has no defensive embedding API")
        # Always go through the backend interface (so PG/Chroma backends behave identically)
        res = self.knowledge_engine.backend.node_query(
            query_embeddings=[emb],
            n_results=self.config.max_candidates,
            where={"level_from_root": {"$lte": self.config.max_retrieval_level}},
            include=["documents", "metadatas"],
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

    def _select_used_evidence(self, *, system_prompt: str, question: str, candidates: Sequence[dict[str, Any]]) -> dict:
        _ = system_prompt
        sel = self._select_used_evidence_bm25(question=question, candidates=candidates)
        return sel.model_dump()


    def _select_used_evidence_bm25(self, *, question: str, candidates: Sequence[dict[str, Any]]) -> EvidenceSelection:
        """Cheap lexical fallback evidence selector (token overlap scoring).

        Note: This is intentionally simple; it is meant as a cost-saving fallback.
        """
        q_tokens = {t for t in re.findall(r"[A-Za-z0-9_]+", question.lower()) if len(t) >= 3}
        scored: list[tuple[float, dict[str, Any]]] = []
        for c in candidates:
            text = f"{c.get('label','')} {c.get('summary','')}".lower()
            c_tokens = {t for t in re.findall(r"[A-Za-z0-9_]+", text) if len(t) >= 3}
            if not c_tokens:
                score = 0.0
            else:
                overlap = len(q_tokens & c_tokens)
                score = overlap / (len(q_tokens) + 1e-6)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        used = [str(c["id"]) for s, c in scored if s > 0][: self.config.max_used]
        return EvidenceSelection(used_node_ids=used, used_edge_ids=[], reasoning="bm25_fallback_token_overlap")
    @staticmethod
    def _materialize_evidence_pack(
        agent: "AgenticAnsweringAgent",
        *,
        node_ids: list[str],
        edge_ids: list[str] | None = None,
        depth: str,
        max_chars_per_item: int,
        max_total_chars: int,
    ) -> dict[str, Any]:
        """Build an evidence pack that includes mentions/spans candidates for citation picking.

        This does NOT change any schema in storage; it only prepares a compact per-run view.
        """
        pack: dict[str, Any] = {"nodes": [], "edges": []}
        total = 0
        edge_ids = list(edge_ids or [])

        for nid in node_ids:
            got = agent.knowledge_engine.backend.node_get(ids=[nid], include=["documents", "metadatas"])
            docs = (got.get("documents") or [None])[0]
            metas = (got.get("metadatas") or [{}])
            meta = metas[0] if metas else {}

            label = meta.get("label") or meta.get("title") or ""
            summary: str = str(meta.get("summary") or "")

            # Heuristic materialization:
            # - shallow: prefer summary
            # - deep: include doc text if available, else summary
            if (depth or "shallow") == "deep" and isinstance(docs, str) and docs.strip():
                text = docs.strip()
            else:
                text = summary.strip() or (docs.strip() if isinstance(docs, str) else "")

            if not text:
                text = ""

            text = text[:max_chars_per_item]
            if total + len(text) > max_total_chars:
                break
            total += len(text)

            # Mentions/spans candidates:
            # If meta already includes mentions/spans, prefer them. Otherwise create a single mention/span candidate
            mentions = meta.get("mentions")
            if isinstance(mentions, list) and mentions:
                # trust existing structure; keep only excerpt-like fields for LLM
                norm_mentions = []
                for mi, mobj in enumerate(mentions):
                    mobj: Grounding
                    spans = []
                    for si, sp in enumerate((mobj or {}).get("spans") or []):
                        if isinstance(sp, dict):
                            spans.append(
                                {
                                    "excerpt": (sp.get("excerpt") or sp.get("verbatim_text") or sp.get("text") or "")[:max_chars_per_item],
                                    "doc_id": sp.get("doc_id") or meta.get("doc_id"),
                                    "page_number": sp.get("page_number") or sp.get("page"),
                                    "document_page_url": sp.get("document_page_url") or meta.get("document_page_url"),
                                }
                            )
                    if spans:
                        norm_mentions.append({"spans": spans})
                if norm_mentions:
                    mentions = norm_mentions
                else:
                    mentions = None
            else:
                mentions = None

            if not mentions:
                # single synthetic mention/span candidate using the materialized text
                mentions = [{"spans": [{"excerpt": text, "doc_id": meta.get("doc_id"), "page_number": meta.get("page_number"), "document_page_url": meta.get("document_page_url")}]}]

            pack["nodes"].append(
                {
                    "node_id": nid,
                    "label": label,
                    "summary": summary[:300],
                    "text": text,
                    "mentions": mentions,
                }
            )

        # Materialize edges as compact hints. No endpoint expansion here.
        for eid in edge_ids:
            got = agent.knowledge_engine.backend.edge_get(ids=[eid], include=["metadatas"])
            if not got.get("ids"):
                continue
            metas = (got.get("metadatas") or [{}])
            meta = metas[0] if metas else {}
            pack["edges"].append({
                "id": eid,
                "label": meta.get("label"),
                "relation": meta.get("relation") or meta.get("type"),
                "summary": meta.get("summary"),
            })
        return pack

    def rehydrate_evidence_pack_from_digest(
        self,
        *,
        digest: dict[str, Any],
        enforce_hash_match: bool = False,
    ) -> dict[str, Any]:
        """Re-materialize an evidence pack from a persisted digest.

        The digest is expected to match :class:`EvidencePackDigest` from models.py.

        This is **best-effort** rehydration:
        - If the underlying KG has changed, the reconstructed pack may differ.
        - If the digest includes `evidence_pack_hash`, callers can compare.
        """
        from .models import EvidencePackDigest

        d = EvidencePackDigest.model_validate(digest)
        pack = self._materialize_evidence_pack(
            agent=self,
            node_ids=list(d.node_ids),
            edge_ids=list(getattr(d, "edge_ids", []) or []),
            depth=str(d.depth),
            max_chars_per_item=int(d.max_chars_per_item or 0),
            max_total_chars=int(d.max_total_chars or 0),
        )
        try:
            pack_hash = snapshot_hash(pack)
        except Exception:
            pack_hash = None
        out = {
            "evidence_pack": pack,
            "rehydrated_hash": pack_hash,
            "expected_hash": d.evidence_pack_hash,
            "hash_matches": (pack_hash == d.evidence_pack_hash) if (pack_hash and d.evidence_pack_hash) else None,
        }

        if enforce_hash_match and out.get("hash_matches") is False:
            raise ValueError(
                "EvidencePackDigest rehydration hash mismatch; underlying KG likely changed. "
                f"expected={out.get('expected_hash')!r} got={out.get('rehydrated_hash')!r}"
            )
        return out
    @staticmethod
    def _generate_answer_with_citations(
        agent: "AgenticAnsweringAgent",
        *,
        system_prompt: str,
        question: str,
        evidence_pack: dict[str, Any],
        used_node_ids: list[str],
        out_model_schema: dict[str, Any],
        out_model: Type[BaseM] = AnswerWithCitations
    ) :
        """Ask the LLM to answer AND cite exact mention/span indices from the provided evidence pack."""
        # Build a compact, indexable representation for the LLM
        lines: list[str] = []
        for n in evidence_pack.get("nodes", []):
            nid = n["node_id"]
            lines.append(f"NODE {nid} | {n.get('label','')}")
            for mi, m in enumerate(n.get("mentions") or []):
                for si, sp in enumerate((m or {}).get("spans") or []):
                    ex = (sp.get("excerpt") or "").replace("\n", " ").strip()
                    if ex:
                        ex = ex[:240]
                    lines.append(f"  M{mi} S{si}: {ex}")
        # Add compact edge hints (structure preserved by projected endpoints; no citations required for edges)
        for e in evidence_pack.get("edges", []) or []:
            eid = e.get("id")
            rel = e.get("relation") or "related"
            summ = (e.get("summary") or "").replace("\n", r" \ ").strip()
            if summ:
                summ = summ[:240]
            lines.append(f"EDGE {eid} | {rel}: {summ}")
        evidence_text = "\n".join(lines)
        last_err: Exception | None = None
        for _ in range(int(getattr(agent, "max_retry", 3) or 3)):
            res = agent.llm_tasks.answer_with_citations(
                AnswerWithCitationsTaskRequest(
                    system_prompt=system_prompt,
                    question=question,
                    evidence=evidence_text,
                    response_model=out_model,
                )
            )
            if res.parsing_error:
                last_err = Exception(str(res.parsing_error))
                continue
            if res.answer_payload is None:
                last_err = Exception("Missing parsed output from answer_with_citations task")
                continue
            parsed: BaseM = out_model.model_validate(res.answer_payload)
            return parsed.model_dump()
        raise Exception(f"retry too many errors parsing: {last_err}")
                
                

    @staticmethod
    def _validate_or_repair_citations(
        agent: "AgenticAnsweringAgent",
        *,
        system_prompt: str,
        question: str,
        evidence_pack: dict[str, Any],
        used_node_ids: list[str],
        answer: dict | list,
        answer_in_model: Type[AnswerWithCitations],
    ) -> dict:
        """Validate citations; if invalid, ask the LLM to repair once (with retries).

        Returns a JSON dict compatible with AnswerWithCitations.
        """
        answer_validated: AnswerWithCitations = answer_in_model.model_validate(answer)

        def _node_by_id(nid: str) -> dict[str, Any] | None:
            for n in evidence_pack.get("nodes", []) or []:
                if str(n.get("node_id")) == nid:
                    return n
            return None

        def is_valid_ref(r: SpanRef) -> bool:
            if r.source_node_id not in used_node_ids:
                return False
            node = _node_by_id(r.source_node_id)
            if not node:
                return False
            mentions = node.get("mentions") or []
            if r.mention_index < 0 or r.mention_index >= len(mentions):
                return False
            spans = (mentions[r.mention_index] or {}).get("spans") or []
            if r.span_index < 0 or r.span_index >= len(spans):
                return False
            ex = (spans[r.span_index] or {}).get("excerpt") or ""
            return bool(str(ex).strip())

        bad = False
        for c in answer_validated.claims:
            for r in c.citations:
                if not is_valid_ref(r):
                    bad = True
                    break
            if bad:
                break

        if not bad:
            return answer_validated.model_dump()

        # Build evidence text again (same format as generation).
        lines: list[str] = []
        for n in evidence_pack.get("nodes", []) or []:
            nid = str(n.get("node_id") or "")
            if not nid:
                continue
            lines.append(f"NODE {nid} | {n.get('label','')}")
            for mi, m in enumerate(n.get("mentions") or []):
                for si, sp in enumerate((m or {}).get("spans") or []):
                    ex = (sp.get("excerpt") or "").replace("\n", " ").strip()
                    if ex:
                        ex = ex[:240]
                    lines.append(f"  M{mi} S{si}: {ex}")

        for e in evidence_pack.get("edges", []) or []:
            eid = str(e.get("id") or "")
            if not eid:
                continue
            rel = e.get("relation") or "related"
            summ = (e.get("summary") or "").replace("\n", " ").strip()
            if summ:
                summ = summ[:240]
            lines.append(f"EDGE {eid} | {rel}: {summ}")

        evidence_text = "\n".join(lines)

        last_err: Exception | None = None
        for _ in range(int(getattr(agent, "max_retry", 3) or 3)):
            res = agent.llm_tasks.repair_citations(
                RepairCitationsTaskRequest(
                    system_prompt=system_prompt,
                    question=question,
                    evidence=evidence_text,
                    answer_text=answer_validated.text,
                    response_model=AnswerWithCitations,
                )
            )
            if res.parsing_error:
                last_err = Exception(str(res.parsing_error))
                continue
            parsed = res.answer_payload
            if parsed is None:
                last_err = Exception("Missing parsed output from structured_output")
                continue
            repaired = AnswerWithCitations.model_validate(parsed)
            # If still bad, return repaired anyway (best effort) to avoid looping.
            return repaired.model_dump()

        # If repair cannot parse, fall back to the original (but mark citations empty).
        stripped = answer_validated.model_copy(deep=True)
        for c in stripped.claims:
            c.citations = []
        stripped.reasoning = (stripped.reasoning or "") + " | citation_repair_failed"
        return stripped.model_dump()


    @staticmethod
    def _evaluate_answer(
        agent: "AgenticAnsweringAgent",
        *,
        system_prompt: str,
        question: str,
        answer_text: str,
        used_node_ids: list[str],
        evidence_pack: dict[str, Any],
        out_model_schema: dict[str, Any],
        out_model: Type[BaseM],
    ) -> BaseM:
        """Coarse sufficiency check (best-effort structured output)."""
        ev_lines: list[str] = []
        for n in evidence_pack.get("nodes", []) or []:
            ev_lines.append(f"- {n.get('node_id')}: {n.get('label','')} | {n.get('summary','')}")
        _ = (system_prompt, question, out_model_schema)
        has_answer = bool(str(answer_text or "").strip())
        has_evidence = bool(used_node_ids) and bool(ev_lines)
        is_sufficient = bool(has_answer and has_evidence)
        needs_more_info = not is_sufficient
        missing_aspects: list[str] = []
        if not has_evidence:
            missing_aspects.append("insufficient_evidence")
        if not has_answer:
            missing_aspects.append("empty_answer")

        payload = {
            "is_sufficient": is_sufficient,
            "needs_more_info": needs_more_info,
            "missing_aspects": missing_aspects,
            "notes": "heuristic_evaluation",
        }
        return out_model.model_validate(payload).model_dump()

    def _ensure_run_anchor(self, *, conversation_id: str, run_id: str) -> str:
        scope = f"conv:{conversation_id}"
        rid = pointer_id(scope=scope, pointer_kind="agent_run", target_kind="run", target_id=run_id)
        existing = self.conversation_engine.backend.node_get(ids=[rid], include=[])
        if existing.get("ids"):
            return rid

        sp = Span.from_dummy_for_conversation()
        node = ConversationNode(
            id=rid,
            label=f"Agent {self.__class__.__name__} Run {run_id}",
            type="entity",
            summary=f"Agent run anchor {run_id}",
            conversation_id=conversation_id,
            role="system",  # type: ignore
            turn_index=None,
            properties={"run_id": run_id, "entity_type": "agent_run"},
            mentions=[Grounding(spans=[sp])],
            metadata={"level_from_root": 0, "entity_type": "agent_run", 
                      "in_conversation_chain": False},
            domain_id=None,
            canonical_entity_id=None,

        )
        self.conversation_engine.add_node(node)
        return rid


    def _persist_context_snapshot(
        self,
        *,
        conversation_id: str,
        run_id: str,
        run_step_seq: int,
        attempt_seq: int,
        stage: str,
        view: Any,
        model_name: str,
        budget_tokens: int,
        tail_turn_index: int,
        extra_hash_payload: dict[str, Any] | None = None,
        llm_input_payload: dict[str, Any] | None = None,
        evidence_pack_digest: dict[str, Any] | None = None,
    ) -> str:
        """Phase 2B wrapper (kept for API stability)."""
        try:
            from .service import ConversationService

            svc = ConversationService.from_engine(
                self.conversation_engine,
                knowledge_engine=self.knowledge_engine,
                workflow_engine=getattr(self.conversation_engine, "workflow_engine", None),
            )
            return svc.persist_context_snapshot(
                conversation_id=conversation_id,
                run_id=run_id,
                run_step_seq=int(run_step_seq),
                attempt_seq=int(attempt_seq),
                stage=stage,
                view=view,
                model_name=str(model_name or ""),
                budget_tokens=int(budget_tokens or 0),
                tail_turn_index=int(max(0, tail_turn_index or 0)),
                extra_hash_payload=extra_hash_payload,
                llm_input_payload=llm_input_payload,
                evidence_pack_digest=evidence_pack_digest,
            )
        except Exception:
            # Lightweight fallback for unit-test doubles.
            return (
                f"context_snapshot::{conversation_id}::{run_id}::{stage}::"
                f"{int(run_step_seq)}::{int(attempt_seq)}"
            )

    def _project_kg_node(
        self,
        *,
        # target_namespace: str, # kg, conv, wisdom
        conversation_id: str,
        run_node_id: str,
        kg_node_id: str,
        provenance_span: Span,
        prev_turn_meta_summary: MetaFromLastSummary,
    ) -> str:
        scope = f"conv:{conversation_id}"
        pid = pointer_id(scope=scope, pointer_kind="kg_node", target_kind="node", target_id=kg_node_id)

        # If exists, still ensure run->evidence edge exists (idempotent)
        existing = self.conversation_engine.backend.node_get(ids=[pid], include=[])
        if not existing.get("ids"):
            kg = self.knowledge_engine.backend.node_get(ids=[kg_node_id], include=["metadatas"])
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
                    "refers_to_collection": "nodes",
                    # "target_kind": "node",
                    "target_id": kg_node_id,
                    "snapshot_hash": sh,
                    "entity_type": "knowledge_reference",
                },
                mentions=[Grounding(spans=[provenance_span])],
                metadata={"level_from_root": 0, "entity_type": "knowledge_reference", 
                          "in_conversation_chain": False},
                domain_id=None,
                canonical_entity_id=None,
            )
            prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
            prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(str(meta.get("summary")))
            self.conversation_engine.add_node(node)

        # Link run -> evidence
        eid = edge_id(scope=scope, rel="used_evidence", src=run_node_id, dst=pid)
        ex_edge = self.conversation_engine.backend.edge_get(ids=[eid], include=[])
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
                properties={"entity_type": "conversation_edge"},
                embedding=None,
                metadata={"char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary, 
                          "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
                          "tail_turn_index": prev_turn_meta_summary.tail_turn_index},
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(edge)
        return pid

    

    def _project_kg_edge(
        self,
        *,
        conversation_id: str,
        run_node_id: str,
        kg_edge_id: str,
        provenance_span: Span,
        prev_turn_meta_summary: MetaFromLastSummary,
    ) -> str:
        """Project a KG edge into the conversation graph as a **pointer edge**.

        Invariants (per your override):
        - Projection is idempotent via deterministic IDs.
        - Structure is preserved by projecting KG endpoints into conversation endpoint IDs
          using the SAME deterministic node projection function.
        - Endpoint node pointers may be missing (dangling endpoints allowed); we do not
          create incidence links and do not auto-project endpoint nodes.
        - Usage is tracked separately (turn/run-scoped), not on the projection edge.
        """
        scope = f"conv:{conversation_id}"
        peid = pointer_id(scope=scope, pointer_kind="kg_edge", target_kind="edge", target_id=kg_edge_id)

        # Fetch KG edge metadata once (also used for the usage-pointer node summary).
        eg = self.knowledge_engine.backend.edge_get(ids=[kg_edge_id], include=["metadatas"])
        if not eg.get("ids"):
            raise ValueError(f"KG edge not found: {kg_edge_id}")
        meta = (eg.get("metadatas") or [{}])[0] or {}

        existing = self.conversation_engine.backend.edge_get(ids=[peid], include=[])
        if not existing.get("ids"):
            kg_src_ids = list(meta.get("source_ids") or [])
            kg_tgt_ids = list(meta.get("target_ids") or [])

            # Deterministic endpoint projection IDs (nodes may be dangling / not yet created).
            conv_src_ids = [
                pointer_id(scope=scope, pointer_kind="kg_node", target_kind="node", target_id=sid)
                for sid in kg_src_ids
            ]
            conv_tgt_ids = [
                pointer_id(scope=scope, pointer_kind="kg_node", target_kind="node", target_id=tid)
                for tid in kg_tgt_ids
            ]

            edge = ConversationEdge(
                id=peid,
                source_ids=conv_src_ids,
                target_ids=conv_tgt_ids,
                relation=str(meta.get("relation") or meta.get("type") or "kg_edge"),
                label=str(meta.get("label") or "kg_edge"),
                type="relationship",
                summary=str(meta.get("summary") or ""),
                doc_id=f"conv:{conversation_id}",
                mentions=[Grounding(spans=[provenance_span])],
                domain_id=None,
                canonical_entity_id=None,
                properties={
                    "is_pointer": True,
                    "refers_to_collection": "edges",
                    "refers_to_entity_id": kg_edge_id,
                    "target_namespace": "kg",
                    "entity_type": "knowledge_reference_edge",
                },
                embedding=None,
                metadata={
                    "entity_type": "knowledge_reference_edge",
                    "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
                },
                source_edge_ids=[],
                target_edge_ids=[],
            )
            self.conversation_engine.add_edge(edge)

        # # Usage is recorded via a separate pointer-node so used_evidence remains node-targeted.
        # edge_ptr_nid = pointer_id(scope=scope, pointer_kind="kg_edge_ptr", target_kind="edge", target_id=kg_edge_id)
        # exn = self.conversation_engine.backend.node_get(ids=[edge_ptr_nid], include=[])
        # if not exn.get("ids"):
        #     node = ConversationNode(
        #         id=edge_ptr_nid,
        #         label=f"RefEdge {kg_edge_id}",
        #         type="reference_pointer",
        #         summary=str(meta.get("summary") or ""),
        #         conversation_id=conversation_id,
        #         role="system",  # type: ignore
        #         turn_index=None,
        #         properties={
        #             "target_namespace": "kg",
        #             "refers_to_collection": "edges",
        #             "target_kind": "edge",
        #             "target_id": kg_edge_id,
        #             "entity_type": "knowledge_reference_edge",
        #             "projection_edge_id": peid,
        #         },
        #         mentions=[Grounding(spans=[provenance_span])],
        #         metadata={
        #             "level_from_root": 0,
        #             "entity_type": "knowledge_reference_edge",
        #             "in_conversation_chain": False,
        #         },
        #         domain_id=None,
        #         canonical_entity_id=None,
        #     )
        #     self.conversation_engine.add_node(node)

        # ue = edge_id(scope=scope, rel="used_evidence", src=run_node_id, dst=edge_ptr_nid)
        # ex_edge = self.conversation_engine.backend.edge_get(ids=[ue], include=[])
        # if not ex_edge.get("ids"):
        #     used = ConversationEdge(
        #         id=ue,
        #         source_ids=[run_node_id],
        #         target_ids=[edge_ptr_nid],
        #         relation="used_evidence",
        #         label="used_evidence",
        #         type="relationship",
        #         summary="Agent used this evidence edge",
        #         doc_id=f"conv:{conversation_id}",
        #         mentions=[Grounding(spans=[provenance_span])],
        #         domain_id=None,
        #         canonical_entity_id=None,
        #         properties={"entity_type": "conversation_edge"},
        #         embedding=None,
        #         metadata={
        #             "char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary,
        #             "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary,
        #             "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
        #         },
        #         source_edge_ids=[],
        #         target_edge_ids=[],
        #     )
        #     self.conversation_engine.add_edge(used)

        return peid
    def _add_assistant_turn(self, *, conversation_id: str, content: str, provenance_span: Span, turn_index : int, 
                            prev_turn_meta_summary: MetaFromLastSummary) -> tuple[str, ConversationNode]:
        # Minimal assistant turn node
        nid = pointer_id(scope=f"conv:{conversation_id}", pointer_kind="turn", target_kind="assistant", target_id=str(int(time.time()*1000)))
        import numpy as np
        emb = cast(np.ndarray, self.conversation_engine.iterative_defensive_emb(content))
        node = ConversationNode(
            id=nid,
            label="Assistant turn",
            type="entity",
            summary=content,
            conversation_id=conversation_id,
            role="assistant",  # type: ignore
            turn_index=turn_index,
            properties={"content": content, "entity_type": "assistant_turn"},
            mentions=[Grounding(spans=[provenance_span])],
            metadata={"level_from_root": 0, 
                      "entity_type": "assistant_turn", 
                      "in_conversation_chain":True, 
                      "in_ui_chain": True},
            domain_id=None,
            canonical_entity_id=None,
            embedding=emb.tolist()
        )
        self.conversation_engine.add_node(node)
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(content)
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        prev_turn_meta_summary.tail_turn_index += 1
        return nid, node

    def _link_run_to_response(self, *, conversation_id: str, run_node_id: str, response_node_id: str, used_node_ids : list[str], 
                              provenance_span: Span | list[Span],
                              prev_turn_meta_summary) -> None:
        scope = f"conv:{conversation_id}"
        eid = edge_id(scope=scope, rel="generated", src=run_node_id, dst=response_node_id)
        ex = self.conversation_engine.backend.edge_get(ids=[eid], include=[])
        if ex.get("ids"):
            return
        
        if type (provenance_span) is Span:
            provenance_span_coerced = [provenance_span]
        elif type (provenance_span) is list:
            provenance_span_coerced = provenance_span
        else:
            raise Exception("Unreacheable")
        # provenance_span_coerced: list[Span] = list[provenance_span] if type(provenance_span) is not list else provenance_span
        edge = ConversationEdge(
            id=eid,
            source_ids=[run_node_id],
            target_ids=[response_node_id],
            relation="generated",
            label="generated",
            type="relationship",
            summary="Agent run generated assistant response",
            doc_id=f"conv:{conversation_id}",
            mentions=[Grounding(spans=provenance_span_coerced)],
            domain_id=None,
            canonical_entity_id=None,
            properties={"entity_type": "conversation_edge", "used_node_ids" : used_node_ids},
            embedding=None,
            metadata={"char_distance_from_last_summary": prev_turn_meta_summary.prev_node_char_distance_from_last_summary, 
                        "turn_distance_from_last_summary": prev_turn_meta_summary.prev_node_distance_from_last_summary, },
            source_edge_ids=[],
            target_edge_ids=[],
        )
        self.conversation_engine.add_edge(edge)

