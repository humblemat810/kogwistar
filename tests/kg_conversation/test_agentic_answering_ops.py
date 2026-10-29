from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kogwistar.conversation.resolvers import default_resolver
from kogwistar.engine_core.models import Span
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)
from kogwistar.runtime.models import RunFailure, RunSuccess


@dataclass
class _FakeStepContext:
    """Minimal StepContext shim used by resolver unit tests.

    The real runtime provides richer behavior (checkpoints, provenance, etc).
    For resolver unit tests we only need:
      - ctx.state_view (read-only mapping)
      - ctx.state_write (context manager giving mutable state dict)
    """

    state: dict[str, Any]
    idempotency_key: str | None = None

    @property
    def state_view(self) -> dict[str, Any]:
        return self.state

    @property
    def state_write(self):
        @contextlib.contextmanager
        def _cm() -> Iterator[dict[str, Any]]:
            yield self.state

        return _cm()


def _noop_task_set() -> LLMTaskSet:
    return LLMTaskSet(
        extract_graph=lambda request: ExtractGraphTaskResult(
            raw=None, parsed_payload={}, parsing_error=None
        ),
        adjudicate_pair=lambda request: AdjudicatePairTaskResult(
            verdict_payload={}, raw=None, parsing_error=None
        ),
        adjudicate_batch=lambda request: AdjudicateBatchTaskResult(
            verdict_payloads=[], raw=None, parsing_error=None
        ),
        filter_candidates=lambda request: FilterCandidatesTaskResult(
            node_ids=[], edge_ids=[], reasoning=""
        ),
        summarize_context=lambda request: SummarizeContextTaskResult(text=""),
        answer_with_citations=lambda request: AnswerWithCitationsTaskResult(
            answer_payload={}, raw=None, parsing_error=None
        ),
        repair_citations=lambda request: RepairCitationsTaskResult(
            answer_payload={}, raw=None, parsing_error=None
        ),
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
            return {
                "ids": ["turn:c1:0"],
                "documents": [self._node_doc],
                "metadatas": [self._node_meta],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    def edge_get(self, **kwargs):
        return {"ids": [], "metadatas": []}


class _StubConversationEngine:
    def __init__(self, *, last_user_text: str = "hello") -> None:
        self.kg_graph_type = "conversation"
        self.backend = _StubBackend(last_user_text=last_user_text)
        self.llm_tasks = _noop_task_set()
        self.tool_call_id_factory = lambda *args, **kwargs: "tool-call-id"
        self.pre_add_node_hooks: list[Any] = []
        self.pre_add_edge_hooks: list[Any] = []
        self.pre_add_pure_edge_hooks: list[Any] = []
        self.allow_missing_doc_id_on_endpoint_rows_hooks: list[Any] = []


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
        self.cache_dir = str(Path.cwd() / ".tmp_pytest" / "agentic_answering_ops")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self._project_calls: list[str] = []

    def _prepare_run_anchor(self, *, conversation_id: str, run_id: str):
        from kogwistar.conversation.models import ConversationNode
        from kogwistar.engine_core.models import Grounding

        node_id = f"run:{conversation_id}:{run_id}"
        return node_id, ConversationNode(
            id=node_id,
            label=f"Agent Run {run_id}",
            type="entity",
            summary=f"Agent run anchor {run_id}",
            conversation_id=conversation_id,
            role="system",
            mentions=[Grounding(spans=[Span.from_dummy_for_conversation()])],
            properties={"run_id": run_id, "entity_type": "agent_run"},
            metadata={
                "entity_type": "agent_run",
                "level_from_root": 0,
                "in_conversation_chain": False,
            },
        )

    def _ensure_run_anchor(self, *, conversation_id: str, run_id: str) -> str:
        return f"run:{conversation_id}:{run_id}"

    def _prepare_kg_node_projection(
        self,
        *,
        conversation_id: str,
        run_node_id: str,
        kg_node_id: str,
        provenance_span: Span,
        prev_turn_meta_summary: Any,
    ):
        from kogwistar.conversation.models import ConversationEdge, ConversationNode
        from kogwistar.engine_core.models import Grounding

        pointer_node_id = f"pointer:{conversation_id}:{kg_node_id}"
        pointer_node = ConversationNode(
            id=pointer_node_id,
            label=f"Ref {kg_node_id}",
            type="entity",
            summary="source summary",
            conversation_id=conversation_id,
            role="system",
            mentions=[Grounding(spans=[provenance_span])],
            properties={"entity_type": "knowledge_reference"},
            metadata={
                "entity_type": "knowledge_reference",
                "level_from_root": 0,
            },
        )
        prev_turn_meta_summary.prev_node_distance_from_last_summary += 1
        prev_turn_meta_summary.prev_node_char_distance_from_last_summary += len(
            pointer_node.summary
        )
        usage_edge = ConversationEdge(
            id=f"used:{run_node_id}:{pointer_node_id}",
            source_ids=[run_node_id],
            target_ids=[pointer_node_id],
            relation="used_evidence",
            label="used_evidence",
            type="relationship",
            summary="Agent used this evidence",
            doc_id=f"conv:{conversation_id}",
            mentions=[Grounding(spans=[provenance_span])],
            properties={"entity_type": "conversation_edge"},
            metadata={
                "char_distance_from_last_summary": (
                    prev_turn_meta_summary.prev_node_char_distance_from_last_summary
                ),
                "turn_distance_from_last_summary": (
                    prev_turn_meta_summary.prev_node_distance_from_last_summary
                ),
                "tail_turn_index": prev_turn_meta_summary.tail_turn_index,
            },
            source_edge_ids=[],
            target_edge_ids=[],
        )
        return pointer_node_id, pointer_node, usage_edge

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

    def _select_used_evidence_bm25(self, *, question: str, candidates: list[dict]):
        return SimpleNamespace(
            used_node_ids=[candidates[0]["id"], candidates[1]["id"]],
            model_dump=lambda: {
                "used_node_ids": ["n1", "n2"],
                "used_edge_ids": [],
                "reasoning": "stub",
            },
        )

    @staticmethod
    def _materialize_evidence_pack(
        agent,
        *,
        node_ids: list[str],
        edge_ids: list[str] | None,
        depth: str,
        max_chars_per_item: int,
        max_total_chars: int,
    ):
        return {
            "nodes": [
                {"node_id": nid, "mentions": [{"spans": [{"excerpt": f"e:{nid}"}]}]}
                for nid in node_ids
            ],
            "edges": [
                {"node_id": nid, "mentions": [{"spans": [{"excerpt": f"e:{nid}"}]}]}
                for nid in edge_ids
            ],
        }

    def rehydrate_evidence_pack_from_digest(
        self, *, digest: dict, enforce_hash_match: bool = False
    ):
        return {
            "evidence_pack": self._materialize_evidence_pack(
                self,
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
        return {
            "is_sufficient": True,
            "needs_more_info": False,
            "missing_aspects": [],
            "notes": "",
        }

    def _project_kg_node(
        self,
        *,
        conversation_id: str,
        run_node_id: str,
        kg_node_id: str,
        provenance_span: Span,
        prev_turn_meta_summary,
    ):
        pid = f"ptr:{conversation_id}:{kg_node_id}"
        self._project_calls.append(pid)
        return pid

    def _add_assistant_turn(
        self,
        *,
        conversation_id: str,
        run_node_id: str,
        content: str,
        provenance_span: Span,
        turn_index: int,
        prev_turn_meta_summary,
    ):
        from kogwistar.conversation.agentic_answering import pointer_id

        assistant_id = pointer_id(
            scope=f"conv:{conversation_id}",
            pointer_kind="turn",
            target_kind="assistant",
            target_id=run_node_id,
        )
        return (
            assistant_id,
            SimpleNamespace(
                model_dump=lambda: {
                    "id": assistant_id,
                    "content": content,
                }
            ),
        )

    def _prepare_assistant_turn(
        self,
        *,
        conversation_id: str,
        run_node_id: str,
        content: str,
        provenance_span: Span,
        turn_index: int,
    ):
        from kogwistar.conversation.models import ConversationNode
        from kogwistar.conversation.agentic_answering import pointer_id
        from kogwistar.engine_core.models import Grounding

        assistant_id = pointer_id(
            scope=f"conv:{conversation_id}",
            pointer_kind="turn",
            target_kind="assistant",
            target_id=run_node_id,
        )
        return assistant_id, ConversationNode(
            id=assistant_id,
            label="Assistant turn",
            type="entity",
            summary=content,
            conversation_id=conversation_id,
            role="assistant",
            turn_index=turn_index,
            properties={"content": content, "entity_type": "assistant_turn"},
            mentions=[Grounding(spans=[provenance_span])],
            metadata={
                "level_from_root": 0,
                "entity_type": "assistant_turn",
                "in_conversation_chain": True,
                "in_ui_chain": True,
            },
        )

    def _prepare_run_to_response_edge(self, **kwargs):
        from kogwistar.conversation.models import ConversationEdge
        from kogwistar.engine_core.models import Grounding

        span = kwargs["provenance_span"]
        if isinstance(span, list):
            spans = span
        else:
            spans = [span]
        conversation_id = kwargs["conversation_id"]
        run_node_id = kwargs["run_node_id"]
        response_node_id = kwargs["response_node_id"]
        return ConversationEdge(
            id=f"generated:{run_node_id}:{response_node_id}",
            source_ids=[run_node_id],
            target_ids=[response_node_id],
            relation="generated",
            label="generated",
            type="relationship",
            summary="Agent run generated assistant response",
            doc_id=f"conv:{conversation_id}",
            mentions=[Grounding(spans=spans)],
            properties={
                "entity_type": "conversation_edge",
                "used_node_ids": kwargs["used_node_ids"],
            },
            metadata={
                "char_distance_from_last_summary": kwargs[
                    "prev_turn_meta_summary"
                ].prev_node_char_distance_from_last_summary,
                "turn_distance_from_last_summary": kwargs[
                    "prev_turn_meta_summary"
                ].prev_node_distance_from_last_summary,
            },
            source_edge_ids=[],
            target_edge_ids=[],
        )

    def _link_run_to_response(self, **kwargs):
        return None


def _mk_state(
    *, agent: _StubAgent, conv_engine: _StubConversationEngine
) -> dict[str, Any]:
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


def _run_op(op: str, state: dict[str, Any], *, idempotency_key: str | None = None):
    ctx = _FakeStepContext(state, idempotency_key=idempotency_key)
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


def test_aa_prepare_returns_run_anchor_effect_for_rust_worker_without_python_write():
    from kogwistar.conversation.models import MetaFromLastSummary
    from kogwistar.engine_core.graph_mutation_contract import (
        validate_graph_mutation_batch,
    )

    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    ce.namespace = "conversation"
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.backend.nodes = SimpleNamespace(name="gke_nodes")
    ce.backend.embedding_dim = 3
    ce.embed = SimpleNamespace(iterative_defensive_emb=lambda _text: [1.0, 0.0, 0.0])
    ce.write = SimpleNamespace(
        node_doc_and_meta=lambda node: (
            node.model_dump_json(field_mode="backend", exclude=["embedding"]),
            {"entity_type": "agent_run", "namespace": "conversation"},
        ),
        add_node=lambda *_args, **_kwargs: pytest.fail(
            "Rust worker callback must return graph effects instead of writing directly"
        ),
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: "{}"
    )
    ce.pre_add_node_hooks = [
        lambda node: node.properties.update({"hook_prepared": True})
    ]
    state = _mk_state(agent=agent, conv_engine=ce)
    state["_deps"]["_rust_worker_effects"] = True
    state["_deps"]["prev_turn_meta_summary"] = MetaFromLastSummary(4, 2, 1)

    result = _run_op(
        "aa_prepare", state, idempotency_key="stable-worker-message-id"
    )

    assert isinstance(result, RunSuccess)
    batch = validate_graph_mutation_batch(result.graph_mutation_batch)
    assert len(batch.mutations) == 1
    assert batch.mutations[0].record.id == state["run_node_id"]
    assert batch.mutations[0].op == "ADD"
    assert batch.mutations[0].payload["properties"]["hook_prepared"] is True
    assert len(batch.index_jobs) == 2
    assert {job.index_kind for job in batch.index_jobs} == {"node_docs", "node_refs"}
    update = next(item[1] for item in result.state_update if item[0] == "u")
    assert update["prev_turn_meta_summary"] == {
        "prev_node_char_distance_from_last_summary": 4,
        "prev_node_distance_from_last_summary": 2,
        "tail_turn_index": 1,
    }


def test_aa_project_pointers_batches_pointer_and_usage_edge_once_for_rust_worker():
    from kogwistar.conversation.models import MetaFromLastSummary
    from kogwistar.engine_core.graph_mutation_contract import (
        validate_graph_mutation_batch,
    )

    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    ce.namespace = "conversation"
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.backend.nodes = SimpleNamespace(name="gke_nodes")
    ce.backend.edges = SimpleNamespace(name="gke_edges")
    ce.backend.embedding_dim = 3
    ce.embed = SimpleNamespace(iterative_defensive_emb=lambda _text: [1.0, 0.0, 0.0])
    ce.write = SimpleNamespace(
        node_doc_and_meta=lambda node: (
            node.model_dump_json(field_mode="backend", exclude=["embedding"]),
            {"entity_type": "knowledge_reference", "namespace": "conversation"},
        ),
        edge_doc_and_meta=lambda edge: (
            edge.model_dump_json(field_mode="backend", exclude=["embedding"]),
            {
                **edge.metadata,
                "entity_type": "conversation_edge",
                "namespace": "conversation",
            },
        ),
        add_node=lambda *_args, **_kwargs: pytest.fail(
            "Rust worker pointer projection must return graph effects"
        ),
        add_edge=lambda *_args, **_kwargs: pytest.fail(
            "Rust worker pointer projection must return graph effects"
        ),
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: "{}"
    )
    ce.pre_add_node_hooks = [
        lambda node: node.properties.update({"hook_prepared": True})
    ]
    ce.pre_add_edge_hooks = []
    meta_summary = MetaFromLastSummary(0, 0)
    state = _mk_state(agent=agent, conv_engine=ce)
    state.update(
        {
            "conversation_id": "conversation-1",
            "run_node_id": "run-1",
            "used_node_ids": ["kg-1", "kg-1"],
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": 0,
                "prev_node_distance_from_last_summary": 0,
                "tail_turn_index": 0,
            },
        }
    )
    state["_deps"]["_rust_worker_effects"] = True
    state["_deps"]["prev_turn_meta_summary"] = meta_summary

    result = _run_op(
        "aa_project_pointers", state, idempotency_key="stable-worker-message-id"
    )

    assert isinstance(result, RunSuccess)
    batch = validate_graph_mutation_batch(result.graph_mutation_batch)
    assert [mutation.entity_kind for mutation in batch.mutations] == ["node", "edge"]
    assert len(batch.index_jobs) == 4
    assert batch.mutations[0].payload["properties"]["hook_prepared"] is True
    edge_meta = batch.mutations[1].record.metadata
    assert edge_meta["char_distance_from_last_summary"] == len("source summary")
    summary_updates = [
        update[1]
        for update in result.state_update or []
        if update[0] == "u" and "prev_turn_meta_summary" in update[1]
    ]
    assert summary_updates[-1]["prev_turn_meta_summary"][
        "prev_node_distance_from_last_summary"
    ] == 1
    ce.pre_add_edge_hooks = [lambda _edge: False]
    unsupported = _run_op(
        "aa_project_pointers", state, idempotency_key="stable-worker-message-id"
    )
    assert isinstance(unsupported, RunFailure)
    assert any("cannot preserve edge hooks" in error for error in unsupported.errors)


def test_aa_persist_response_batches_stable_turn_and_generated_edge_for_rust_worker():
    from kogwistar.conversation.models import MetaFromLastSummary
    from kogwistar.engine_core.graph_mutation_contract import (
        validate_graph_mutation_batch,
    )

    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="Q")
    ce.namespace = "conversation"
    ce.persistence_mode = "single_stage"
    ce._phase1_enable_index_jobs = True
    ce._disable_event_log = False
    ce.backend.nodes = SimpleNamespace(name="gke_nodes")
    ce.backend.edges = SimpleNamespace(name="gke_edges")
    ce.backend.embedding_dim = 3
    ce.embed = SimpleNamespace(iterative_defensive_emb=lambda _text: [1.0, 0.0, 0.0])

    def _node_projection(node):
        return (
            node.model_dump_json(field_mode="backend", exclude=["embedding"]),
            {"entity_type": "assistant_turn", "namespace": "conversation"},
        )

    def _edge_projection(edge):
        return (
            edge.model_dump_json(field_mode="backend", exclude=["embedding"]),
            {
                **edge.metadata,
                "entity_type": "conversation_edge",
                "namespace": "conversation",
            },
        )

    ce.write = SimpleNamespace(
        node_doc_and_meta=_node_projection,
        edge_doc_and_meta=_edge_projection,
        add_node=lambda *_args, **_kwargs: pytest.fail(
            "Rust worker response persistence must return graph effects"
        ),
        add_edge=lambda *_args, **_kwargs: pytest.fail(
            "Rust worker response persistence must return graph effects"
        ),
    )
    ce.indexing = SimpleNamespace(
        canonical_source_payload_for_projection=lambda **_kwargs: "{}"
    )
    ce.pre_add_node_hooks = [
        lambda node: node.properties.update({"hook_prepared": True})
    ]
    ce.pre_add_edge_hooks = []
    state = _mk_state(agent=agent, conv_engine=ce)
    state.update(
        {
            "conversation_id": "conversation-1",
            "run_node_id": "run-1",
            "answer": {"text": "answer"},
            "evaluation": {"is_sufficient": True},
            "used_node_ids": ["kg-1"],
            "projected_pointer_ids": ["pointer-1"],
            "prev_turn_meta_summary": {
                "prev_node_char_distance_from_last_summary": 0,
                "prev_node_distance_from_last_summary": 0,
                "tail_turn_index": 0,
            },
        }
    )
    state["_deps"]["_rust_worker_effects"] = True
    state["_deps"]["prev_turn_meta_summary"] = MetaFromLastSummary(0, 0)

    result = _run_op(
        "aa_persist_response", state, idempotency_key="stable-worker-message-id"
    )

    assert isinstance(result, RunSuccess)
    batch = validate_graph_mutation_batch(result.graph_mutation_batch)
    assert [mutation.entity_kind for mutation in batch.mutations] == ["node", "edge"]
    assert batch.mutations[0].record.id == result.conversation_node_id
    from kogwistar.conversation.agentic_answering import pointer_id

    assert result.conversation_node_id == pointer_id(
        scope="conv:conversation-1",
        pointer_kind="turn",
        target_kind="assistant",
        target_id="run-1",
    )
    assert batch.mutations[0].payload["properties"]["hook_prepared"] is True
    assert batch.mutations[1].payload["relation"] == "generated"
    assert [job.index_kind for job in batch.index_jobs] == [
        "node_docs",
        "node_refs",
        "edge_refs",
        "edge_endpoints",
    ]
    assert state["agentic_answering_result"]["assistant_turn_node_id"] == (
        result.conversation_node_id
    )

    ce.pre_add_edge_hooks = [lambda _edge: False]
    state["prev_turn_meta_summary"] = {
        "prev_node_char_distance_from_last_summary": 0,
        "prev_node_distance_from_last_summary": 0,
        "tail_turn_index": 0,
    }
    unsupported = _run_op(
        "aa_persist_response", state, idempotency_key="stable-worker-message-id"
    )
    assert isinstance(unsupported, RunFailure)
    assert any("cannot preserve edge hooks" in error for error in unsupported.errors)


def test_aa_get_view_and_question_populates_question_and_system_prompt():
    agent = _StubAgent()
    ce = _StubConversationEngine(last_user_text="What is X?")
    state = _mk_state(agent=agent, conv_engine=ce)
    _run_op("aa_prepare", state)
    res = _run_op("aa_get_view_and_question", state)
    assert isinstance(res, RunSuccess)
    assert (
        state["system_prompt"]
        == "You are a helpful assistant. Answer the user using the conversation and any provided evidence."
    )
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


def test_bm25_selector_matches_shared_rust_contract_fixture():
    from kogwistar.conversation.agentic_answering import AgentConfig, AgenticAnsweringAgent

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "answer-bm25-selection-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    state = fixture["state"]
    agent = AgenticAnsweringAgent.__new__(AgenticAnsweringAgent)
    agent.config = AgentConfig(max_used=state["max_used"])

    selection = agent._select_used_evidence_bm25(
        question=state["question"], candidates=state["candidates"]
    )

    assert selection.model_dump() == fixture["expected"]["selection"]


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


def test_answer_evaluation_matches_shared_rust_contract_fixture():
    from kogwistar.conversation.agentic_answering import (
        AgenticAnsweringAgent,
        AnswerEvaluation,
    )

    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "answer-evaluation-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["operation"] == "aa_evaluate_answer/v1"
    agent = AgenticAnsweringAgent.__new__(AgenticAnsweringAgent)

    for case in fixture["cases"]:
        state = case["state"]
        evidence_pack = state.get("_rt", {}).get("evidence_pack", {})
        answer_text = state.get("answer", {}).get("text", "")
        used_node_ids = state.get("used_node_ids", [])
        evaluation = AgenticAnsweringAgent._evaluate_answer(
            agent,
            system_prompt="",
            question="",
            answer_text=answer_text,
            used_node_ids=used_node_ids,
            evidence_pack=evidence_pack,
            out_model_schema=AnswerEvaluation.model_json_schema(),
            out_model=AnswerEvaluation,
        )
        assert evaluation.model_dump(mode="python") == case["expected"]["evaluation"]

        base_candidates = int(state.get("max_candidates") or 20)
        base_depth = str(state.get("materialize_depth") or "shallow")
        insufficient = evaluation.needs_more_info and not evaluation.is_sufficient
        assert case["expected"]["run_step_seq"] == int(state.get("run_step_seq") or 0) + 1
        assert case["expected"]["max_candidates"] == (
            min(base_candidates * 2, 200) if insufficient else base_candidates
        )
        assert case["expected"]["materialize_depth"] == (
            "deep" if insufficient else base_depth
        )


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
