from pathlib import Path
from typing import Sequence
import pytest
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from tests._helpers.embeddings import (
    ConstantEmbeddingFunction,
    EmbeddingFunction,
    Embeddings,
    build_test_embedding_function,
)
from tests._helpers.fake_backend import build_fake_backend

FakeEmbeddingFunction = ConstantEmbeddingFunction

def _make_engine_pair(
    *,
    backend_kind: str,
    tmp_path,
    sa_engine,
    pg_schema,
    dim: int = 384,
    use_fake: bool = False,
    embedding_kind: str | None = None,
    embedding_function: object | None = None,
):
    """Build `(kg_engine, conv_engine)` for either backend with configurable embeddings."""

    ef = (
        embedding_function
        if embedding_function is not None
        else build_test_embedding_function(embedding_kind or "constant", dim=dim)
    )
    if backend_kind == "fake":
        kg_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "kg"),
            kg_graph_type="knowledge",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "conv"),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
        return kg_engine, conv_engine
    if backend_kind == "chroma":
        try:
            kg_engine = GraphKnowledgeEngine(
                persist_directory=str(Path(tmp_path) / "kg"),
                kg_graph_type="knowledge",
                embedding_function=ef,
            )
            conv_engine = GraphKnowledgeEngine(
                persist_directory=str(Path(tmp_path) / "conv"),
                kg_graph_type="conversation",
                embedding_function=ef,
            )
        except Exception as exc:
            if embedding_function is None and str(embedding_kind or "").lower() in {
                "provider",
                "real",
            }:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
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
            embedding_function=ef,
            backend=kg_backend,
        )
        conv_engine = GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "conv_meta"),
            kg_graph_type="conversation",
            embedding_function=ef,
            backend=conv_backend,
        )
        return kg_engine, conv_engine

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")

def _make_workflow_engine(
    *,
    backend_kind: str,
    tmp_path,
    sa_engine,
    pg_schema,
    dim: int = 384,
    use_fake: bool = False,
    embedding_kind: str | None = None,
    embedding_function: object | None = None,
) -> GraphKnowledgeEngine:
    ef = (
        embedding_function
        if embedding_function is not None
        else build_test_embedding_function(embedding_kind or "constant", dim=dim)
    )
    if backend_kind == "fake":
        return GraphKnowledgeEngine(
            persist_directory=str(Path(tmp_path) / "wf"),
            kg_graph_type="workflow",
            embedding_function=ef,
            backend_factory=build_fake_backend,
        )
    if backend_kind == "chroma":
        try:
            return GraphKnowledgeEngine(
                persist_directory=str(Path(tmp_path) / "wf"),
                kg_graph_type="workflow",
                embedding_function=ef,
            )
        except Exception as exc:
            if embedding_function is None and str(embedding_kind or "").lower() in {
                "provider",
                "real",
            }:
                pytest.skip(f"real embedding provider unavailable: {exc}")
            raise
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
            embedding_function=ef,
            backend=wf_backend,
        )
    raise ValueError(f"unknown backend_kind: {backend_kind!r}")
