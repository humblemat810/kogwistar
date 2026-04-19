from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.server.chat_service import ChatRunService
from kogwistar.server.run_registry import RunRegistry
from tests._helpers.fake_backend import build_fake_backend


class FakeEmbeddingFunction:
    def __call__(self, input):  # noqa: A002
        return [[0.0, 0.0, 0.0] for _ in input]


@contextmanager
def build_demo_service():
    root = Path(".tmp_service_demo") / str(uuid.uuid4())
    root.mkdir(parents=True, exist_ok=True)
    try:
        ef = FakeEmbeddingFunction()
        knowledge = GraphKnowledgeEngine(
            persist_directory=str(root / "kg"),
            kg_graph_type="knowledge",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        conversation = GraphKnowledgeEngine(
            persist_directory=str(root / "conversation"),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        workflow = GraphKnowledgeEngine(
            persist_directory=str(root / "workflow"),
            kg_graph_type="workflow",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        service = ChatRunService(
            get_knowledge_engine=lambda: knowledge,
            get_conversation_engine=lambda: conversation,
            get_workflow_engine=lambda: workflow,
            run_registry=RunRegistry(workflow.meta_sqlite),
        )
        yield service
    finally:
        shutil.rmtree(root, ignore_errors=True)
