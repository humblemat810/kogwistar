from __future__ import annotations

from graph_knowledge_engine.workflow.design import ConversationWorkflowDesigner, load_workflow_design


def test_phase2d_conversation_designer_ensure_backbone(backend_kind, tmp_path, sa_engine, pg_schema):
    # Reuse canonical engine fixture builder.
    from tests.conftest import _make_engine_pair

    _kg, conv = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=3, use_fake=True)

    designer = ConversationWorkflowDesigner(
        workflow_engine=conv,
        predicate_registry={"always": lambda st, r: True},
    )

    workflow_id = "phase2d_backbone_design_unit"
    start, nodes, adj = designer.ensure_backbone(workflow_id=workflow_id)

    assert start.id.endswith(":start")
    assert any(n.metadata.get("wf_terminal") for n in nodes.values()), "expected a terminal node"
    # exactly one outgoing edge from start in skeleton
    outs = adj.get(start.id) or []
    assert len(outs) == 1
    e = outs[0]
    assert e.metadata.get("wf_predicate") == "always"
