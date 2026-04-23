from __future__ import annotations

from pathlib import Path
from typing import Any

from kogwistar.engine_core.engine import GraphKnowledgeEngine


def build_engine_triplet(
    *,
    root: Path,
    embedding_function: Any,
    backend_factory=None,
) -> tuple[GraphKnowledgeEngine, GraphKnowledgeEngine, GraphKnowledgeEngine]:
    kwargs = {"embedding_function": embedding_function}
    if backend_factory is not None:
        kwargs["backend_factory"] = backend_factory
    return (
        GraphKnowledgeEngine(
            persist_directory=str(root / "kg"),
            kg_graph_type="knowledge",
            **kwargs,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(root / "conversation"),
            kg_graph_type="conversation",
            **kwargs,
        ),
        GraphKnowledgeEngine(
            persist_directory=str(root / "workflow"),
            kg_graph_type="workflow",
            **kwargs,
        ),
    )
