from __future__ import annotations

from graph_knowledge_engine.conversation.designer import ConversationWorkflowDesigner
from graph_knowledge_engine.runtime.design import load_workflow_design
import pytest
from typing import Sequence

pytest.importorskip("chromadb")
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings
class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in documents_or_texts]


@pytest.mark.parametrize("backend_kind", ["chroma", "pg"])
def test_phase2d_conversation_designer_ensure_backbone(backend_kind, tmp_path, sa_engine, pg_schema):
    # Reuse canonical engine fixture builder.
    # from tests.conftest import _make_engine_pair
    from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
    # _kg, conv = _make_engine_pair(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, dim=3, use_fake=True)
    
    wf_dir = tmp_path / "wf"
    workflow_engine = GraphKnowledgeEngine(persist_directory=str(wf_dir), 
                                           kg_graph_type="workflow",
                                           embedding_function=FakeEmbeddingFunction(dim=3),)
    
    designer = ConversationWorkflowDesigner(
        workflow_engine=workflow_engine,
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
