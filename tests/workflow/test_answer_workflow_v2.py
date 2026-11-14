import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from kogwistar.conversation.agentic_answering import (
    AgentConfig,
    AgenticAnsweringAgent,
    AnswerEvaluation,
    AnswerWithCitations,
    ClaimWithCitations,
    EvidenceSelection,
    SpanRef,
)
from kogwistar.conversation.models import (
    FilteringResult,
    MetaFromLastSummary,
)
from kogwistar.conversation.service import ConversationService
from kogwistar.id_provider import stable_id
from kogwistar.llm_tasks import (
    AdjudicateBatchTaskResult,
    AdjudicatePairTaskResult,
    AnswerWithCitationsTaskResult,
    ExtractGraphTaskResult,
    FilterCandidatesTaskResult,
    LLMTaskProviderHints,
    LLMTaskSet,
    RepairCitationsTaskResult,
    SummarizeContextTaskResult,
)

pytestmark = pytest.mark.ci_full


@pytest.fixture(autouse=True)
def _require_python_only_path(monkeypatch: pytest.MonkeyPatch) -> None:
    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        monkeypatch.setenv(selector, "python")

    def fail_if_rust_load_is_attempted() -> None:
        raise AssertionError("Python-only answer workflow attempted to load Rust")

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", fail_if_rust_load_is_attempted
    )


def _fixed_llm_tasks(answer: AnswerWithCitations) -> LLMTaskSet:
    payload = answer.model_dump(mode="python")
    return LLMTaskSet(
        extract_graph=lambda _req: ExtractGraphTaskResult(
            raw=None, parsed_payload=None, parsing_error="unused"
        ),
        adjudicate_pair=lambda _req: AdjudicatePairTaskResult(
            verdict_payload=None, raw=None, parsing_error="unused"
        ),
        adjudicate_batch=lambda _req: AdjudicateBatchTaskResult(
            verdict_payloads=(), raw=None, parsing_error="unused"
        ),
        filter_candidates=lambda _req: FilterCandidatesTaskResult(
            node_ids=(),
            edge_ids=(),
            reasoning="",
            raw=None,
            parsing_error=None,
        ),
        summarize_context=lambda req: SummarizeContextTaskResult(text=req.full_text),
        answer_with_citations=lambda _req: AnswerWithCitationsTaskResult(
            answer_payload=payload, raw=None, parsing_error=None
        ),
        repair_citations=lambda _req: RepairCitationsTaskResult(
            answer_payload=payload, raw=None, parsing_error=None
        ),
        provider_hints=LLMTaskProviderHints(answer_with_citations_provider="custom"),
    )


def _ensure_user_turn(
    conversation_engine, *, user_id: str, conversation_id: str, text: str
):
    """Create a tail user turn in the conversation graph.

    Prefer the public engine facade if available.
    """
    # In your repo, engine exposes `create_conversation` and `add_conversation_turn`.
    # This helper uses them to avoid reaching into collections directly.
    turn_id = "turn_0"
    mem_id = "mem_0"
    mts = MetaFromLastSummary(0, 0)
    svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=conversation_engine,
    )
    svc.add_conversation_turn(
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        mem_id=mem_id,
        role="user",
        content=text,
        ref_knowledge_engine=conversation_engine,  # not used if add_turn_only=True
        filtering_callback=lambda *_a, **_k: (
            FilteringResult(node_ids=[], edge_ids=[]),
            "noop",
        ),
        prev_turn_meta_summary=mts,
        add_turn_only=True,
    )


@pytest.mark.parametrize("backend_kind", ["fake", "chroma"], indirect=True)
@pytest.mark.parametrize("question", ["What is chlorophyll?"])
def test_answer_workflow_v2_runs_end_to_end(
    workflow_engine, conversation_engine, engine, question,
    tmp_path
):
    # Create a conversation with at least one user turn (tail must exist).
    conversation_engine.tool_call_id_factory = stable_id
    engine.tool_call_id_factory = stable_id
    workflow_engine.tool_call_id_factory = stable_id
    user_id = "u_test"
    import time

    conv_svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=engine,
        workflow_engine=workflow_engine,
    )
    conversation_id, _start_id = conv_svc.create_conversation(user_id=user_id)
    time.sleep(0.5)
    _ensure_user_turn(
        conversation_engine,
        user_id=user_id,
        conversation_id=conversation_id,
        text=question,
    )
    time.sleep(0.5)
    # Fixed outputs for each structured step.
    _selection = EvidenceSelection(
        used_node_ids=[],
        used_edge_ids=[],
        reasoning="test case is always simple and good.",
    )
    answer = AnswerWithCitations(
        text="dummy", reasoning="testing answer no claim needed", claims=[]
    )
    _evaluation = AnswerEvaluation(
        is_sufficient=True,
        needs_more_info=False,
        missing_aspects=[],
        notes="this test example is simple and good.",
    )

    llm_tasks = _fixed_llm_tasks(answer)

    agent = AgenticAnsweringAgent(
        conversation_engine=conversation_engine,
        knowledge_engine=engine,
        llm_tasks=llm_tasks,
        config=AgentConfig(max_iter=1, max_candidates=5),
    )

    assert hasattr(agent, "answer_workflow_v2"), (
        "AgenticAnsweringAgent must expose answer_workflow_v2()"
    )

    mts = MetaFromLastSummary(0, 0)
    out = agent.answer_workflow_v2(
        workflow_engine=workflow_engine,
        conversation_id=conversation_id,
        user_id=user_id,
        prev_turn_meta_summary=mts,
        cache_dir=tmp_path/ "agent_answer_workflow_v2_cache_dir"
    )

    # Minimal end-to-end assertions
    assert isinstance(out, dict)
    assert out.get("assistant_text") == "dummy"
    assert "run_id" in out or "run_node_id" in out or "agent_run" in out


@pytest.mark.manual
@pytest.mark.parametrize("backend_kind", ["pg"], indirect=True)
@pytest.mark.parametrize("sample_id", ["warmup", "sample-1", "sample-2"])
def test_python_agent_postgres_2k_request_profile(
    engine,
    conversation_engine,
    workflow_engine,
    sample_id: str,
    tmp_path,
) -> None:
    """Python full-workflow counterpart to the Rust-native 2k PG profile.

    The timer intentionally covers the in-process Python workflow call, not HTTP;
    the emitted digest covers language-neutral answer semantics only.
    """
    from tests._helpers.graph_builders import build_entity_node

    question = "Where is workflow runtime state stored?"
    answer_text = "The workflow runtime state is stored in the graph."
    target_id = "answer-evidence-node"
    user_id = "alice"
    conversation_id = f"answer-bench-{sample_id}"
    conversation_engine.tool_call_id_factory = stable_id
    engine.tool_call_id_factory = stable_id
    workflow_engine.tool_call_id_factory = stable_id

    conv_svc = ConversationService.from_engine(
        conversation_engine,
        knowledge_engine=engine,
        workflow_engine=workflow_engine,
    )
    conversation_id, _start_id = conv_svc.create_conversation(
        user_id=user_id, conv_id=conversation_id
    )
    _ensure_user_turn(
        conversation_engine,
        user_id=user_id,
        conversation_id=conversation_id,
        text=question,
    )

    target_summary = "The workflow runtime state is stored in the graph."
    query_embedding = engine.embed.iterative_defensive_emb(question)
    distractor_embedding = [-float(value) for value in query_embedding]
    fixture_size = int(os.getenv("KOGWISTAR_AGENT_BENCH_FIXTURE_SIZE", "2000"))
    if not 0 < fixture_size <= 10_000:
        raise ValueError("benchmark fixture size must be in 1..=10000")
    seed_started = time.perf_counter()
    nodes = [
        build_entity_node(
            node_id=target_id,
            doc_id="answer-e2e-document",
            label="Workflow runtime state",
            summary=target_summary,
            embedding=query_embedding,
            metadata={"visibility": "public"},
        )
    ]
    nodes.extend(
        build_entity_node(
            node_id=f"answer-benchmark-distractor-{index:05}",
            doc_id="answer-benchmark-document",
            label=f"Unrelated record {index}",
            summary=f"Unrelated benchmark knowledge record {index}.",
            embedding=distractor_embedding,
            metadata={"visibility": "public"},
        )
        for index in range(fixture_size)
    )
    projection_rows = [
        (node, *engine.write.node_doc_and_meta(node)) for node in nodes
    ]
    for node, _document, metadata in projection_rows:
        metadata["_class_name"] = type(node).__name__
    with engine.uow():
        for node, _document, _metadata in projection_rows:
            engine._append_event_for_entity(
                namespace=engine.namespace,
                entity_kind="node",
                entity_id=node.safe_get_id(),
                op="ADD",
                payload=node.model_dump(field_mode="backend", exclude=["embedding"]),
                required=True,
            )
        engine.backend.node_add(
            ids=[node.safe_get_id() for node, _document, _metadata in projection_rows],
            documents=[document for _node, document, _metadata in projection_rows],
            metadatas=[metadata for _node, _document, metadata in projection_rows],
            embeddings=[node.embedding for node, _document, _metadata in projection_rows],
        )
    seed_ms = (time.perf_counter() - seed_started) * 1_000

    answer = AnswerWithCitations(
        text=answer_text,
        reasoning="The selected evidence states this directly.",
        claims=[
            ClaimWithCitations(
                claim="Workflow runtime state is stored in the graph.",
                citations=[
                    SpanRef(
                        source_node_id=target_id,
                        mention_index=0,
                        span_index=0,
                    )
                ],
            )
        ],
    )
    agent = AgenticAnsweringAgent(
        conversation_engine=conversation_engine,
        knowledge_engine=engine,
        llm_tasks=_fixed_llm_tasks(answer),
        config=AgentConfig(max_iter=1, max_candidates=10, max_used=4),
    )
    candidates = agent._retrieve_candidates(question)
    candidate_ids = [str(candidate.get("id")) for candidate in candidates]
    assert target_id in candidate_ids
    started = time.perf_counter()
    out = agent.answer_workflow_v2(
        workflow_engine=workflow_engine,
        conversation_id=conversation_id,
        user_id=user_id,
        prev_turn_meta_summary=MetaFromLastSummary(0, 0),
        run_id=f"answer-bench-{sample_id}",
        trace=False,
        cache_dir=str(tmp_path / "answer-bench-cache"),
    )
    request_ms = (time.perf_counter() - started) * 1_000
    assert out["assistant_text"] == answer_text
    assert out["used_node_ids"] == [target_id]
    answer_payload = out["claim_citations"]
    canonical_state = {
        "assistant_text": out["assistant_text"],
        "used_node_ids": out["used_node_ids"],
        "claims": answer_payload["claims"],
    }
    result_contract = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "contracts"
            / "golden"
            / "agentic-answer-result-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert canonical_state == result_contract["canonical_result"]
    canonical_json = json.dumps(
        canonical_state, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    assert digest == result_contract["canonical_json_sha256"]
    print(
        "PYTHON_AGENT_BENCH_SAMPLE="
        + json.dumps(
            {
                "sample_id": sample_id,
                "fixture_nodes": fixture_size + 1,
                "request_boundary": "in_process_answer_workflow_v2",
                "seed_ms": round(seed_ms, 3),
                "request_ms": round(request_ms, 3),
                "canonical_state_sha256": digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
