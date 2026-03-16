from pathlib import Path
from typing import Sequence
import pytest
from chromadb.utils.embedding_functions import EmbeddingFunction
from chromadb.api.types import Embeddings
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend

class FakeEmbeddingFunction(EmbeddingFunction):
    @staticmethod
    def name() -> str:
        return "default"

    def __init__(self, dim: int = 384):
        self._dim = dim

    def __call__(self, documents_or_texts: Sequence[str]) -> Embeddings:
        return [[0.01] * self._dim for _ in documents_or_texts]

def _make_engine_pair(
    *, backend_kind: str, tmp_path, sa_engine, pg_schema, dim: int = 384, use_fake=False
):
    """
    Build (kg_engine, conv_engine) for either chroma or the pg-backed path.
    """
    if backend_kind == "chroma":
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "kg"),
            kg_graph_type="knowledge",
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "conv"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )
        return kg_engine, conv_engine

    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip(
                "pg backend requested but sa_engine/pg_schema fixtures not available"
            )
        kg_schema = f"{pg_schema}_kg"
        conv_schema = f"{pg_schema}_conv"
        kg_backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=dim, schema=kg_schema
        )
        conv_backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=dim, schema=conv_schema
        )
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "kg_meta"),
            kg_graph_type="knowledge",
            embedding_function=FakeEmbeddingFunction(dim=dim),
            backend=kg_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "conv_meta"),
            kg_graph_type="conversation",
            embedding_function=FakeEmbeddingFunction(dim=dim),
            backend=conv_backend,
        )
        return kg_engine, conv_engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")

def _make_workflow_engine(
    *, backend_kind: str, tmp_path, sa_engine, pg_schema, dim: int = 384
) -> GraphKnowledgeEngine:
    if backend_kind == "chroma":
        return GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "wf"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=dim),
        )
    if backend_kind == "pg":
        if sa_engine is None or pg_schema is None:
            pytest.skip(
                "pg backend requested but sa_engine/pg_schema fixtures not available"
            )
        wf_schema = f"{pg_schema}_wf"
        wf_backend = PgVectorBackend(
            engine=sa_engine, embedding_dim=dim, schema=wf_schema
        )
        return GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "wf_meta"),
            kg_graph_type="workflow",
            embedding_function=FakeEmbeddingFunction(dim=dim),
            backend=wf_backend,
        )
    raise ValueError(f"unknown backend_kind: {backend_kind!r}")
