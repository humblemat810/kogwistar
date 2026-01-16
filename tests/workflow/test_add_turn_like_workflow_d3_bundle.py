import json
from pathlib import Path

from graph_knowledge_engine.engine import GraphKnowledgeEngine
from graph_knowledge_engine.models import (
    WorkflowNode,
    WorkflowEdge,
    Grounding,
    Span,
    MentionVerification,
)
from graph_knowledge_engine.workflow.runtime import WorkflowRuntime

# IMPORTANT: use your existing dumper (calls to_d3_force internally)
from graph_knowledge_engine.utils.kge_debug_dump import dump_paired_bundles  # type: ignore


def _span() -> Span:
    return Span(
        collection_page_url="test",
        document_page_url="test",
        doc_id="test",
        insertion_method="test",
        page_number=1,
        start_char=0,
        end_char=4,
        excerpt="test",
        context_before="",
        context_after="",
        chunk_id=None,
        source_cluster_id=None,
        source_cluster_id=None,
        verification=MentionVerification(method="human", is_verified=True, score=1.0, notes="test"),
    )


def _g() -> Grounding:
    return Grounding(spans=[_span()])


def _wf_node(*, workflow_id: str, node_id: str, op: str, start=False, terminal=False) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        label=node_id.split("|")[-1],
        type="entity",
        doc_id=node_id,
        summary=op,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_node",
            "workflow_id": workflow_id,
            "wf_op": op,
            "wf_start": start,
            "wf_terminal": terminal,
            "wf_version": "v1",
        },
        domain_id=None,
        canonical_entity_id=None,
    )


def _wf_edge(
    *,
    workflow_id: str,
    edge_id: str,
    src: str,
    dst: str,
    predicate: str | None,
    priority: int,
    is_default: bool,
) -> WorkflowEdge:
    return WorkflowEdge(
        id=edge_id,
        source_ids=[src],
        target_ids=[dst],
        relation="wf_next",
        label="wf_next",
        type="relationship",
        summary="next",
        doc_id=workflow_id,
        mentions=[_g()],
        properties={},
        metadata={
            "entity_type": "workflow_edge",
            "workflow_id": workflow_id,
            "wf_priority": priority,
            "wf_is_default": is_default,
            "wf_predicate": predicate,
            "wf_multiplicity": "one",
        },
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
    )


def test_add_turn_like_workflow_dump_bundles(tmp_path: Path):
    # Persist dirs for engines
    wf_dir = tmp_path / "wf"
    kg_dir = tmp_path / "kg"
    conv_dir = tmp_path / "conv"

    workflow_id = "wf_add_turn_like_manual_bundle"

    workflow_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    kg_engine = GraphKnowledgeEngine(persist_directory=str(kg_dir), kg_graph_type="knowledge")
    conversation_engine = GraphKnowledgeEngine(persist_directory=str(conv_dir), kg_graph_type="conversation")

    # -----------------------------
    # 1) Producer: build + persist workflow design
    # -----------------------------
    n_mem = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|mem", op="memory_retrieve", start=True)
    n_kg  = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|kg", op="kg_retrieve")
    n_mpin= _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|mpin", op="memory_pin")
    n_kpin= _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|kpin", op="kg_pin")
    n_ans = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|ans", op="answer")
    n_dec = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|dec", op="decide_summarize")
    n_sum = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|sum", op="summarize")
    n_end = _wf_node(workflow_id=workflow_id, node_id=f"wf|{workflow_id}|end", op="end", terminal=True)

    for n in [n_mem, n_kg, n_mpin, n_kpin, n_ans, n_dec, n_sum, n_end]:
        workflow_engine.add_node(n)

    # mem -> kg
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|mem->kg",
        src=n_mem.id, dst=n_kg.id,
        predicate=None, priority=100, is_default=True
    ))

    # kg -> mpin if should_pin_memory
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kg->mpin",
        src=n_kg.id, dst=n_mpin.id,
        predicate="should_pin_memory", priority=0, is_default=False
    ))
    # kg -> kpin if should_pin_kg
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kg->kpin",
        src=n_kg.id, dst=n_kpin.id,
        predicate="should_pin_kg", priority=1, is_default=False
    ))
    # default kg -> answer
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kg->ans|default",
        src=n_kg.id, dst=n_ans.id,
        predicate=None, priority=100, is_default=True
    ))

    # mpin -> ans ; kpin -> ans
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|mpin->ans",
        src=n_mpin.id, dst=n_ans.id,
        predicate=None, priority=100, is_default=True
    ))
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|kpin->ans",
        src=n_kpin.id, dst=n_ans.id,
        predicate=None, priority=100, is_default=True
    ))

    # ans -> dec
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|ans->dec",
        src=n_ans.id, dst=n_dec.id,
        predicate=None, priority=100, is_default=True
    ))
    # dec -> sum if should_summarize else -> end
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|dec->sum",
        src=n_dec.id, dst=n_sum.id,
        predicate="should_summarize", priority=0, is_default=False
    ))
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|dec->end|default",
        src=n_dec.id, dst=n_end.id,
        predicate=None, priority=100, is_default=True
    ))
    # sum -> end
    workflow_engine.add_edge(_wf_edge(
        workflow_id=workflow_id,
        edge_id=f"wf|{workflow_id}|e|sum->end",
        src=n_sum.id, dst=n_end.id,
        predicate=None, priority=100, is_default=True
    ))

    workflow_engine.persist()

    # -----------------------------
    # 2) Consumer: run stored design (runnable proof)
    # -----------------------------
    predicate_registry = {
        "should_pin_memory": lambda st, r: bool(st.get("memory", {}).get("selected_ids")),
        "should_pin_kg": lambda st, r: bool(st.get("kg", {}).get("selected_ids")),
        "should_summarize": lambda st, r: bool(st.get("decide", {}).get("need_summary")),
    }

    def resolve_step(op: str):
        def _fn(ctx):
            ctx.state.setdefault("op_log", [])
            ctx.state["op_log"].append(op)

            # JSONable placeholders (manual inspection only)
            if op == "memory_retrieve":
                ctx.state["memory"] = {"selected_ids": ["m1"], "text": "memory context"}
                return {"ok": True}
            if op == "kg_retrieve":
                ctx.state["kg"] = {"selected_ids": ["k1"], "facts": ["f1"]}
                return {"ok": True}
            if op == "memory_pin":
                ctx.state["memory_pin"] = {"pinned_ids": ["m1"]}
                return {"ok": True}
            if op == "kg_pin":
                ctx.state["kg_pin"] = {"pinned_ids": ["k1"]}
                return {"ok": True}
            if op == "answer":
                ctx.state["answer"] = {"text": "answer text", "llm_decision_need_summary": True}
                return ctx.state["answer"]
            if op == "decide_summarize":
                need = bool(ctx.state.get("answer", {}).get("llm_decision_need_summary"))
                ctx.state["decide"] = {"need_summary": need}
                return ctx.state["decide"]
            if op == "summarize":
                ctx.state["summary"] = {"text": "summary text"}
                return ctx.state["summary"]
            if op == "end":
                return {"done": True}
            raise KeyError(op)
        return _fn

    # IMPORTANT: re-open workflow_engine from disk to prove "saved graph is runnable"
    workflow_engine2 = GraphKnowledgeEngine(persist_directory=str(wf_dir), kg_graph_type="workflow")
    rt = WorkflowRuntime(
        workflow_engine=workflow_engine2,
        conversation_engine=conversation_engine,
        step_resolver=resolve_step,
        predicate_registry=predicate_registry,
        checkpoint_every_n_steps=1,
        max_workers=1,
    )

    final_state, run_id = rt.run(
        workflow_id=workflow_id,
        conversation_id="conv_manual",
        turn_node_id="turn_manual",
        initial_state={},
    )
    assert run_id
    assert final_state["op_log"][0] == "memory_retrieve"

    # -----------------------------
    # 3) D3 bundle dumps (HTML) for manual inspection
    # -----------------------------
    # You need your real template file here:
    # graph_knowledge_engine/templates/d3.html (or wherever it lives)
    template_html = Path("graph_knowledge_engine/templates/d3.html").read_text(encoding="utf-8")

    out_dir = Path("tests/_artifacts/add_turn_like_bundle")
    meta = dump_paired_bundles(
        kg_engine=kg_engine,
        conversation_engine=conversation_engine,
        workflow_engine=workflow_engine2,
        template_html=template_html,
        out_dir=out_dir,
        mode="reify",
        insertion_method=None,
        kg_doc_id=None,
        conversation_doc_id=None,
    )

    # Minimal asserts: artifacts exist
    assert (out_dir / "kg.bundle.html").exists()
    assert (out_dir / "conversation.bundle.html").exists()
    assert (out_dir / "workflow.bundle.html").exists()
    assert (out_dir / "bundle.meta.json").exists()
    assert json.loads((out_dir / "bundle.meta.json").read_text(encoding="utf-8"))["mode"] == meta["mode"]
