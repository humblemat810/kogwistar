from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest

from graph_knowledge_engine.engine_core.models import Span
from graph_knowledge_engine.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)
from graph_knowledge_engine.runtime.models import RunFailure
from graph_knowledge_engine.conversation.resolvers import default_resolver
from graph_knowledge_engine.runtime.models import RunSuccess


@dataclass
class _FakeStepContext:
    """Minimal StepContext shim used by resolver unit tests.

    The real runtime provides richer behavior (checkpoints, provenance, etc).
    For resolver unit tests we only need:
      - ctx.state_view (read-only mapping)
      - ctx.state_write (context manager giving mutable state dict)
    """

    state: Dict[str, Any]

    @property
    def state_view(self) -> Dict[str, Any]:
        return self.state

    @property
    def state_write(self):
        @contextlib.contextmanager
        def _cm() -> Iterator[Dict[str, Any]]:
            yield self.state

        return _cm()


def _noop_task_set() -> LLMTaskSet:
    return LLMTaskSet(
        extract_graph=lambda request: ExtractGraphTaskResult(raw=None, parsed_payload={}, parsing_error=None),
        adjudicate_pair=lambda request: AdjudicatePairTaskResult(verdict_payload={}, raw=None, parsing_error=None),
        adjudicate_batch=lambda request: AdjudicateBatchTaskResult(verdict_payloads=[], raw=None, parsing_error=None),
        filter_candidates=lambda request: FilterCandidatesTaskResult(node_ids=[], edge_ids=[], reasoning=""),
        summarize_context=lambda request: SummarizeContextTaskResult(text=""),
        answer_with_citations=lambda request: AnswerWithCitationsTaskResult(answer_payload={}, raw=None, parsing_error=None),
        repair_citations=lambda request: RepairCitationsTaskResult(answer_payload={}, raw=None, parsing_error=None),
    )


class _StubBackend:
    def __init__(self, *, last_user_text: str) -> None:
        span = Span.from_dummy_for_conversation("c1")
        metadata = {
            "entity_type": "conversation_turn",
            "level_from_root": 0,
            "in_conversation_chain": True,
            "conversation_id": "c1",
            "user_id": "u1",
            "role": "user",
            "turn_index": 0,
            "lifecycle_status": "active",
            "redirect_to_id": None,
        }
        self._node_doc = json.dumps(
            {
                "id": "turn:c1:0",
                "label": "user turn",
                "type": "entity",
                "summary": last_user_text,
                "domain_id": None,
                "canonical_entity_id": None,
                "properties": {},
                "mentions": [
                    {
                        "spans": [
                            {
                                "collection_page_url": span.collection_page_url,
                                "document_page_url": span.document_page_url,
                                "doc_id": span.doc_id,
                                "insertion_method": span.insertion_method,
                                "page_number": span.page_number,
                                "start_char": span.start_char,
                                "end_char": span.end_char,
                                "excerpt": span.excerpt,
                                "context_before": span.context_before,
                                "context_after": span.context_after,
                                "chunk_id": span.chunk_id,
                                "source_cluster_id": span.source_cluster_id,
                                "verification": (
                                    None
                                    if span.verification is None
                                    else {
                                        "method": span.verification.method,
                                        "is_verified": span.verification.is_verified,
                                        "score": span.verification.score,
                                        "notes": span.verification.notes,
                                    }
                                ),
                            }
                        ]
                    }
                ],
                "embedding": None,
                "doc_id": "conv:c1",
                "metadata": metadata,
                "level_from_root": 0,
                "role": "user",
                "turn_index": 0,
                "conversation_id": "c1",
                "user_id": "u1",
            }
        )
        self._node_meta = dict(metadata)

    def node_get(self, **kwargs):
        where = kwargs.get("where") or {}
        if where.get("conversation_id") == "c1":
            return {"ids": ["turn:c1:0"], "documents": [self._node_doc], "metadatas": [self._node_meta]}
        return {"ids": [], "documents": [], "metadatas": []}

    def edge_get(self, **kwargs):
        return {"ids": [], "metadatas": []}


class _StubConversationEngine:
    def __init__(self, *, last_user_text: str = "hello") -> None:
        self.kg_graph_type = "conversation"
        self.backend = _StubBackend(last_user_text=last_user_text)
        self.llm_tasks = _noop_task_set()
        self.tool_call_id_factory = lambda *args, **kwargs: "tool-call-id"
        self.pre_add_node_hooks: List[Any] = []
        self.pre_add_edge_hooks: List[Any] = []
        self.pre_add_pure_edge_hooks: List[Any] = []
        self.allow_missing_doc_id_on_endpoint_rows_hooks: List[Any] = []


class _StubAgentConfig:
    def __init__(self):
        self.max_candidates = 3
        self.materialize_depth = "shallow"
        self.max_chars_per_item = 200
        self.max_total_chars = 1000
        self.max_used = 2
        self.max_iter = 1
        self.evidence_selector = "bm25"


class _StubAgent:
    def __init__(self):
        self.config = _StubAgentConfig()
        self.cache_dir = None
        self._project_calls: List[str] = []

    def _ensure_run_anchor(self, *, conversation_id: str, run_id: str) -> str:
        return f"run:{conversation_id}:{run_id}"

    def _get_last_user_text(self, messages: Any) -> str:
        for m in reversed(messages or []):
            if getattr(m, "role", None) == "user":
                return str(getattr(m, "content", ""))
        return ""

    def _retrieve_candidates(self, question: str):
        return [
            {"id": "n1", "label": "L1", "summary": "S1", "doc": "D1"},
            {"id": "n2", "label": "L2", "summary": "S2", "doc": "D2"},
            {"id": "n3", "label": "L3", "summary": "S3", "doc": "D3"},
        ]

    def _select_used_evidence_bm25(self, *, question: str, candidates: List[dict]):
        return SimpleNamespace(
            used_node_ids=[candidates[0]["id"], candidates[1]["id"]],
            model_dump=lambda: {"used_node_ids": ["n1", "n2"], "used_edge_ids": [], "reasoning": "stub"},
        )
    @staticmethod
    def _materialize_evidence_pack(agent, *, node_ids: List[str], edge_ids: List[str] | None, depth: str, max_chars_per_item: int, max_total_chars: int):
        return {"nodes": [{"node_id": nid, "mentions": [{"spans": [{"excerpt": f"e:{nid}"}]}]} for nid in node_ids], 
                "edges": [{"node_id": nid, "mentions": [{"spans": [{"excerpt": f"e:{nid}"}]}]} for nid in edge_ids]}

    def rehydrate_evidence_pack_from_digest(self, *, digest: dict, enforce_hash_match: bool = False):
        return {
            "evidence_pack": self._materialize_evidence_pack(self,
                node_ids=list(digest.get("node_ids") or []),
                edge_ids=list(digest.get("edge_ids") or []),
                depth=str(digest.get("depth") or "shallow"),
                max_chars_per_item=200,
                max_total_chars=1000,
            )
        }

    def _persist_context_snapshot(self, **kwargs):
        return "snap:1"

    def select_used_evidence_cached(self, **kwargs):
        # should not be called in these tests (bm25 path)
        raise AssertionError("select_used_evidence_cached should not be called")
    @staticmethod
    def _generate_answer_with_citations(agent, **kwargs):
        return {"text": "ok", "reasoning": "", "claims": []}

    def _validate_or_repair_citations(self, **kwargs):
        return {"text": "ok", "reasoning": "", "claims": []}

    def _evaluate_answer(self, **kwargs):
        return {"is_sufficient": True, "needs_more_info": False, "missing_aspects": [], "notes": ""}

    def _project_kg_node(self, *, conversation_id: str, run_node_id: str, kg_node_id: str, provenance_span: Span, prev_turn_meta_summary):
        pid = f"ptr:{conversation_id}:{kg_node_id}"
        self._project_calls.append(pid)
        return pid

    def _add_assistant_turn(self, *, conversation_id: str, content: str, provenance_span: Span, turn_index: int, prev_turn_meta_summary):
        return (
            f"turn:{conversation_id}:{turn_index}",
            SimpleNamespace(model_dump=lambda: {"id": f"turn:{conversation_id}:{turn_index}", "content": content}),
        )

    def _link_run_to_response(self, **kwargs):
        return None


def _mk_state(*, agent: _StubAgent, conv_engine: _StubConversationEngine) -> Dict[str, Any]:
    return {
        "conversation_id": "c1",
        "user_id": "u1",
        "turn_node_id": "t0",
        "turn_index": 0,
        "_deps": {
            "agent": agent,
            "conversation_engine": conv_engine,
            "llm": SimpleNamespace(model_name="dummy"),
            "prev_turn_meta_summary": SimpleNamespace(
                tail_turn_index=0,
                prev_node_char_distance_from_last_summary=0,
                prev_node_distance_from_last_summary=0,
            ),
        },
    }


def _run_op(op: str, state: Dict[str, Any]):
    ctx = _FakeStepContext(state)
    fn = default_resolver.resolve(op)
    res = fn(ctx)
    assert isinstance(res, (RunSuccess, RunFailure))

    # The real WorkflowRuntime applies `state_update` merges.
    for mode, payload in getattr(res, "state_update", []) or []:
        if mode == "u":
            assert isinstance(payload, dict)
            state.update(payload)
        elif mode == "a":
            assert isinstance(payload, dict)
            for k, v in payload.items():
                state.setdefault(k, [])
                state[k].append(v)
        elif mode == "e":
            assert isinstance(payload, dict)
            for k, v in payload.items():
                state.setdefault(k, [])
                state[k].extend(list(v))
    return res


def test_aa_prepare_sets_run_identity():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    res = _run_op("aa_prepare", state)
    assert isinstance(res, RunSuccess)
    assert state.get("run_id")
    assert state.get("run_node_id")


def test_aa_get_view_and_question_populates_question_and_system_prompt():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="What is X?")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_prepare", state)
    res = _run_op("aa_get_view_and_question", state)
    assert isinstance(res, RunSuccess)
    assert state["system_prompt"] == "You are a helpful assistant. Answer the user using the conversation and any provided evidence."
    assert state["question"] == "What is X?"


def test_aa_retrieve_candidates_populates_candidates_list():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_get_view_and_question", state)
    res = _run_op("aa_retrieve_candidates", state)
    assert isinstance(res, RunSuccess)
    assert isinstance(state.get("candidates"), list)
    assert len(state["candidates"]) > 0


def test_aa_select_used_evidence_writes_used_node_ids_subset():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_get_view_and_question", state)
    _run_op("aa_retrieve_candidates", state)
    res = _run_op("aa_select_used_evidence", state)
    assert isinstance(res, RunSuccess)
    assert state.get("used_node_ids") == ["n1", "n2"]


def test_aa_materialize_evidence_pack_writes_digest_and_runtime_pack():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    state["used_node_ids"] = ["n1", "n2"]
    state["used_edge_ids"] = ["e1", "e2"]
    res = _run_op("aa_materialize_evidence_pack", state)
    assert isinstance(res, RunSuccess)
    digest = state.get("evidence_pack_digest")
    assert isinstance(digest, dict)
    assert digest.get("node_ids") == ["n1", "n2"]
    assert digest.get("edge_ids") == ["e1", "e2"]
    assert digest.get("evidence_pack_hash")
    assert state.get("_rt", {}).get("evidence_pack")


def test_aa_generate_answer_with_citations_writes_answer():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_get_view_and_question", state)
    state["used_node_ids"] = ["n1"]
    state["evidence_pack_digest"] = {
        "node_ids": ["n1"],
        "depth": "shallow",
        "max_chars_per_item": 200,
        "max_total_chars": 1000,
        "evidence_pack_hash": "h",
    }
    res = _run_op("aa_generate_answer_with_citations", state)
    assert isinstance(res, RunSuccess)
    assert state.get("answer", {}).get("text") == "ok"


def test_aa_evaluate_answer_writes_evaluation():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_get_view_and_question", state)
    state["used_node_ids"] = ["n1"]
    state["answer"] = {"text": "ok"}
    state["evidence_pack_digest"] = {
        "node_ids": ["n1"],
        "depth": "shallow",
        "max_chars_per_item": 200,
        "max_total_chars": 1000,
        "evidence_pack_hash": "h",
    }
    res = _run_op("aa_evaluate_answer", state)
    assert isinstance(res, RunSuccess)
    assert state.get("evaluation", {}).get("is_sufficient") is True


def test_aa_project_pointers_calls_agent_projection_for_each_used_node():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_prepare", state)
    state["used_node_ids"] = ["n1", "n2"]
    res = _run_op("aa_project_pointers", state)
    assert isinstance(res, RunSuccess)
    assert state.get("projected_pointer_ids") == ["ptr:c1:n1", "ptr:c1:n2"]
    assert agent._project_calls == ["ptr:c1:n1", "ptr:c1:n2"]


def test_aa_persist_response_writes_agentic_answering_result_and_prev_turn_meta_summary():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_prepare", state)
    state["answer"] = {"text": "ok"}
    state["evaluation"] = {"is_sufficient": True, "needs_more_info": False}
    state["used_node_ids"] = ["n1"]
    state["projected_pointer_ids"] = ["ptr:c1:n1"]
    res = _run_op("aa_persist_response", state)
    assert isinstance(res, RunSuccess)
    out = state.get("agentic_answering_result")
    assert isinstance(out, dict)
    assert out.get("assistant_text") == "ok"
    assert state.get("prev_turn_meta_summary", {}).get("tail_turn_index") == 1
