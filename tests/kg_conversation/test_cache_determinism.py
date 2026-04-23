import pytest
pytestmark = pytest.mark.ci_full
import shutil
import os
import pathlib
import dataclasses
import json
from typing import Any
from joblib import Memory

from kogwistar.conversation.agentic_answering import AnswerWithCitations
from kogwistar.conversation.conversation_orchestrator import ConversationOrchestrator
from kogwistar.llm_tasks import AnswerWithCitationsTaskResult, LLMTaskSet
from tests.conftest import _make_engine_pair, _make_workflow_engine
from tests.conftest import _to_stable_key
from kogwistar.id_provider import stable_id
from kogwistar.engine_core.models import Node, Grounding, Span

@pytest.mark.parametrize("llm_provider_name", ["ollama"], indirect=True)
def test_cross_backend_cache_determinism(
    llm_provider_name,
    llm_tasks: LLMTaskSet,
    llm_cache_tracker,
    sa_engine,
    pg_schema,
    tmp_path,
    # llm_cache_dir,
):
    """
    Verify that LLM results are reused when switching backends (Chroma -> PG).
    Uses direct input parameter comparison to identify leakage sources.
    
    this test case selection is hardcoded without using llm
    """
    # 1. Setup fresh cache for this test
    my_cache_dir = str(tmp_path / "deterministic_cache")
    if os.path.exists(my_cache_dir):
        shutil.rmtree(my_cache_dir)
    os.makedirs(my_cache_dir, exist_ok=True)

    memory = Memory(location=my_cache_dir, verbose=0)

    # os.startfile(str(tmp_path))
    # Store normalized requests per run and export backend-specific traces.
    all_requests = []  # list[dict[str, Any]]
    current_run = "chroma"
    trace_export_dir = tmp_path / "cache_determinism_traces"
    trace_export_dir.mkdir(parents=True, exist_ok=True)

    def _append_trace(run_name: str, task_name: str, norm: Any, key: str) -> None:
        all_requests.append(
            {
                "run": run_name,
                "task": task_name,
                "normalized": norm,
                "normalized_serialized": key,
            }
        )

    def _dump_backend_trace(run_name: str) -> pathlib.Path:
        rows = [r for r in all_requests if r["run"] == run_name]
        path = trace_export_dir / f"{run_name}_normalized_requests.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"[TRACE EXPORT] {run_name} -> {path}")
        return path

    def _build_trace_diff(chroma_rows: list[dict[str, Any]], pg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        diffs: list[dict[str, Any]] = []
        max_calls = max(len(chroma_rows), len(pg_rows))
        for i in range(max_calls):
            c = chroma_rows[i] if i < len(chroma_rows) else None
            p = pg_rows[i] if i < len(pg_rows) else None
            if c is None or p is None:
                diffs.append(
                    {
                        "index": i,
                        "kind": "missing_call",
                        "chroma": c,
                        "pg": p,
                    }
                )
                continue
            if c["task"] != p["task"]:
                diffs.append(
                    {
                        "index": i,
                        "kind": "task_order_mismatch",
                        "chroma_task": c["task"],
                        "pg_task": p["task"],
                    }
                )
                continue
            if c["normalized_serialized"] != p["normalized_serialized"]:
                entry = {
                    "index": i,
                    "kind": "normalized_payload_mismatch",
                    "task": c["task"],
                    "chroma_normalized_serialized": c["normalized_serialized"],
                    "pg_normalized_serialized": p["normalized_serialized"],
                }
                c_norm = c.get("normalized")
                p_norm = p.get("normalized")
                if isinstance(c_norm, dict) and isinstance(p_norm, dict):
                    field_diffs = []
                    all_keys = sorted(set(c_norm.keys()) | set(p_norm.keys()))
                    for k in all_keys:
                        cv = c_norm.get(k)
                        pv = p_norm.get(k)
                        if cv != pv:
                            field_diffs.append({"field": k, "chroma": cv, "pg": pv})
                    entry["field_diffs"] = field_diffs
                diffs.append(entry)
        return diffs

    def _make_cached_task(task_fn):
        @memory.cache(ignore=["request"])
        def _actual_cached_call(stable_key: str, request: Any):
            # Return dummy but structured response
            from pydantic import BaseModel
            class DummyResponse(BaseModel):
                parsing_error: bool = False
                text: str = f"This is a dummy response for {task_fn.__name__}."
                content: str = f"This is a dummy response for {task_fn.__name__}."
                used_node_ids: list[str] = ["ceo_node"]
                reasoning: str = "Dummy reasoning."
            if task_fn.__name__ == "_summarize_context": # get_summary
                class SummaryResponse(BaseModel):
                    text: str = "user searched for CEO and data shows Sundar Pichai is the CEO."
                return SummaryResponse()
            # For answer_with_citations, it might need more fields
            if task_fn.__name__ == "_answer_with_citations":
                
                # @dataclass
                # class FakeAnswerWithCitationsTaskResult:
                #     answer_payload: Any = field(
                #         default_factory=lambda: AnswerWithCitations(
                #             text="Sundar Pichai is the CEO.",
                #             reasoning="hardcoded example",
                #             claims=[],
                #         ).model_dump()
                #     )

                #     citations: list[Any] = field(default_factory=list)

                #     parsing_error: bool = False
                
                # return FakeAnswerWithCitationsTaskResult()
                fake_raw_payload = {
                                "answer_payload": AnswerWithCitations(
                                    text="Sundar Pichai is the CEO.",
                                    reasoning="hardcoded example",
                                    claims=[],
                                ).model_dump(),
                                "citations": [],
                                "parsing_error": False,
                            }
                answer_with_citations_payload = AnswerWithCitations(
                                    text="Sundar Pichai is the CEO.",
                                    reasoning="hardcoded example",
                                    claims=[],
                                ).model_dump()
                return AnswerWithCitationsTaskResult(answer_payload = answer_with_citations_payload, raw=str(fake_raw_payload), parsing_error=None)
                
            return DummyResponse().model_dump()

        def _wrapper(request):
            norm = _to_stable_key(request)
            key, task_fn_name  = json.dumps(norm, sort_keys=True, default=str), task_fn.__name__
            _append_trace(current_run, task_fn_name, norm, key)
            
            is_hit = False
            try:
                is_hit = _actual_cached_call.check_call_in_cache(key, request)
            except Exception:
                raise
            
            if is_hit:
                llm_cache_tracker.hits.append(task_fn.__name__)
                print(f"[LLM CACHE HIT] run={current_run} task={task_fn.__name__}")
            else:
                llm_cache_tracker.misses.append(task_fn.__name__)
                print(f"[LLM CACHE MISS] run={current_run} task={task_fn.__name__}")
                
            return _actual_cached_call(key, request)
        return _wrapper

    my_tasks = dataclasses.replace(
        llm_tasks,
        extract_graph=_make_cached_task(llm_tasks.extract_graph),
        adjudicate_pair=_make_cached_task(llm_tasks.adjudicate_pair),
        adjudicate_batch=_make_cached_task(llm_tasks.adjudicate_batch),
        filter_candidates=_make_cached_task(llm_tasks.filter_candidates),
        summarize_context=_make_cached_task(llm_tasks.summarize_context),
        answer_with_citations=_make_cached_task(llm_tasks.answer_with_citations),
        repair_citations=_make_cached_task(llm_tasks.repair_citations),
    )

    llm_cache_tracker.reset()
    
    conversation_id = "test_cache_determinism"
    user_id = "user1"
    # Use tokens that match the seeded node for BM25 just in case
    user_text = "Who is the CEO Sundar Pichai?"

    # Seed data
    ceo_node = Node(
        id="ceo_node",
        label="CEO",
        type="entity",
        summary="The CEO of Google is Sundar Pichai.",
        mentions=[Grounding(spans=[Span.from_dummy_for_document()])],
        level_from_root=0
    )
    from kogwistar.conversation.models import FilteringResult
    def mock_filtering_callback(*args, **kwargs):
        # Always return the same context
        return FilteringResult(node_ids=[ceo_node.safe_get_id()], edge_ids=[]), "query: CEO"

    common_workflow_kwargs = {
        "run_id": "run1",
        "user_id": user_id,
        "conversation_id": conversation_id,
        "turn_id": "turn1",
        "mem_id": "mem1",
        "role": "user",
        "content": user_text,
        "filtering_callback": mock_filtering_callback,
        "workflow_id": "wf1",
        "cache_dir": my_cache_dir
    }

    # Deterministic ID factory
    _counter = 0
    def tool_call_id_factory(*arg, **kwarg):
        nonlocal _counter
        _counter += 1
        return str(stable_id("tool_call", conversation_id, str(_counter)))

    # 2. Run with Chroma backend
    print("\n--- Running with Chroma backend ---")
    current_run = "chroma"
    
    kg_chroma, ce_chroma = _make_engine_pair(
        backend_kind="chroma",
        tmp_path=tmp_path / "chroma",
        sa_engine=sa_engine,
        pg_schema=pg_schema,
    )
    kg_chroma.add_node(ceo_node)
    
    we_chroma = _make_workflow_engine(
        backend_kind="chroma",
        tmp_path=tmp_path / "we_chroma",
        sa_engine=sa_engine,
        pg_schema=pg_schema,
    )
    # from kogwistar.utils.cache_paths import joblib_cache_path
    orch_chroma = ConversationOrchestrator(
        conversation_engine=ce_chroma,
        ref_knowledge_engine=kg_chroma,
        workflow_engine=we_chroma,
        llm_tasks=my_tasks,
        tool_call_id_factory=tool_call_id_factory,
        agent_cache_dir=os.path.join(my_cache_dir, "agent_chroma")
    )
    # FORCE LLM selector to trigger LLM calls
    if orch_chroma.agentic_answering_agent:
        orch_chroma.agentic_answering_agent.config.evidence_selector = "llm"
        # Ensure retrieval uses our seeded node
        orch_chroma.agentic_answering_agent.config.max_candidates = 10
    
    orch_chroma.add_conversation_turn_workflow_v2(**common_workflow_kwargs)
    
    # Export and assert Chroma traces.
    chroma_calls = [r for r in all_requests if r["run"] == "chroma"]
    chroma_trace_path = _dump_backend_trace("chroma")
    assert len(chroma_calls) > 0, f"No LLM calls detected in Chroma run! Trace: {chroma_trace_path}"
    llm_cache_tracker.reset()

    # Reset counter for PG run
    _counter = 0

    # 3. Run with PG backend
    print("\n--- Running with PG backend ---")
    current_run = "pg"
    
    kg_pg, ce_pg = _make_engine_pair(
        backend_kind="pg",
        tmp_path=tmp_path / "pg",
        sa_engine=sa_engine,
        pg_schema=pg_schema,
    )
    kg_pg.add_node(ceo_node)
    
    we_pg = _make_workflow_engine(
        backend_kind="pg",
        tmp_path=tmp_path / "we_pg",
        sa_engine=sa_engine,
        pg_schema=pg_schema,
    )
    orch_pg = ConversationOrchestrator(
        conversation_engine=ce_pg,
        ref_knowledge_engine=kg_pg,
        workflow_engine=we_pg,
        llm_tasks=my_tasks,
        tool_call_id_factory=tool_call_id_factory,
        agent_cache_dir=os.path.join(my_cache_dir, "agent_pg")
    )
    # FORCE LLM selector
    if orch_pg.agentic_answering_agent:
        orch_pg.agentic_answering_agent.config.evidence_selector = "llm"
        orch_pg.agentic_answering_agent.config.max_candidates = 10
    
    orch_pg.add_conversation_turn_workflow_v2(**common_workflow_kwargs)
    
    # 4. Detailed Comparison + exported diff for offline analysis.
    pg_calls = [r for r in all_requests if r["run"] == "pg"]
    pg_trace_path = _dump_backend_trace("pg")
    print(f"Chroma calls: {len(chroma_calls)}")
    print(f"PG calls: {len(pg_calls)}")

    diffs = _build_trace_diff(chroma_calls, pg_calls)
    diff_path = trace_export_dir / "normalized_request_diff.json"
    diff_path.write_text(json.dumps(diffs, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[TRACE EXPORT] diff -> {diff_path}")

    for diff in diffs:
        idx = diff["index"]
        kind = diff["kind"]
        if kind == "missing_call":
            print(f"!!! Missing call at index {idx}: chroma_present={diff['chroma'] is not None}, pg_present={diff['pg'] is not None}")
        elif kind == "task_order_mismatch":
            print(f"!!! Differing task order at index {idx}: Chroma={diff['chroma_task']}, PG={diff['pg_task']}")
        elif kind == "normalized_payload_mismatch":
            print(f"!!! Differing normalized payload for task {diff['task']} at index {idx}!")
            for field_diff in diff.get("field_diffs", []):
                print(f"  Field '{field_diff['field']}' DIFF:")
                print(f"    Chroma: {str(field_diff['chroma'])[:500]}")
                print(f"    PG:     {str(field_diff['pg'])[:500]}")
        else:
            print(f"!!! Unexpected diff entry at index {idx}: {diff}")

    assert len(llm_cache_tracker.misses) == 0, f"Cache LEAKAGE! Misses: {llm_cache_tracker.misses}. Diff: {diff_path}"
    assert len(pg_calls) == 0, "pg new called occur without using cache from chroma"
