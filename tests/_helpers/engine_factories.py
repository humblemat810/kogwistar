from pathlib import Path
from typing import Any, List, Optional
import asyncio
import sys

import pytest
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.postgres_backend import PgVectorBackend
from kogwistar.conversation.policy import install_engine_hooks
from tests._helpers.embeddings import build_test_embedding_function
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.embeddings import (
    ConstantEmbeddingFunction,
)
from kogwistar.runtime.runtime import WorkflowRuntime
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.models import RunSuccess

try:
    from langchain_core.runnables import Runnable
except Exception:  # pragma: no cover - optional for langchain-dependent tests

    class Runnable:  # type: ignore
        pass

FakeEmbeddingFunction = ConstantEmbeddingFunction


def _install_conversation_policy(engine: GraphKnowledgeEngine) -> None:
    if getattr(engine, "kg_graph_type", None) == "conversation":
        install_engine_hooks(engine)

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
        _install_conversation_policy(conv_engine)
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
        _install_conversation_policy(conv_engine)
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
        _install_conversation_policy(conv_engine)
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


def _run_async_windows_safe(coro):
    """Run an async coroutine in a loop compatible with psycopg on Windows."""
    if sys.platform == "win32":
        runner = asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)
        try:
            return runner.run(coro)
        finally:
            runner.close()
    return asyncio.run(coro)


def _is_missing_pgvector_extension(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "extension \"vector\" is not available" in msg
        or "could not open extension control file" in msg
        or "vector.control" in msg
    )


def _make_async_engine(
    *,
    backend_kind: str,
    tmp_path,
    request,
    dim: int = 3,
    graph_kind: str = "knowledge",
    embedding_kind: str | None = None,
    embedding_function: object | None = None,
):
    ef = (
        embedding_function
        if embedding_function is not None
        else build_test_embedding_function(embedding_kind or "constant", dim=dim)
    )
    persist_dir = tmp_path / graph_kind

    if backend_kind == "pg":
        async_sa_engine = request.getfixturevalue("async_sa_engine")
        async_pg_schema = request.getfixturevalue("async_pg_schema")
        if async_sa_engine is None or async_pg_schema is None:
            pytest.skip(
                "async pg backend requested but async_pg fixtures are unavailable"
            )
        try:
            backend = PgVectorBackend(
                engine=async_sa_engine, embedding_dim=dim, schema=async_pg_schema
            )
        except Exception as exc:
            if _is_missing_pgvector_extension(exc):
                pytest.skip(f"async pg backend unavailable: {exc}")
            raise
        return GraphKnowledgeEngine(
            persist_directory=str(persist_dir),
            kg_graph_type=graph_kind,
            embedding_function=ef,
            backend=backend,
        )

    if backend_kind == "chroma":
        try:
            real_chroma_server = request.getfixturevalue("real_chroma_server")
        except Exception as exc:
            pytest.skip(f"real async chroma fixture is unavailable: {exc}")
        from tests.core._async_chroma_real import make_real_async_chroma_backend

        backend_client, backend, _collections = _run_async_windows_safe(
            make_real_async_chroma_backend(
                real_chroma_server,
                collection_prefix=f"{graph_kind}_{__import__('uuid').uuid4().hex}",
            )
        )
        _ = backend_client, _collections
        return GraphKnowledgeEngine(
            persist_directory=str(persist_dir),
            kg_graph_type=graph_kind,
            embedding_function=ef,
            backend_factory=lambda _engine, backend=backend: backend,
        )

    raise ValueError(f"unknown backend_kind: {backend_kind!r}")


class FakeStructuredRunnable(Runnable):
    """A minimal Runnable that returns a fixed structured result."""

    def __init__(self, parsed: Any, include_raw: bool = False):
        self._parsed = parsed
        self._include_raw = include_raw

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        if self._include_raw:
            return {"raw": None, "parsed": self._parsed, "parsing_error": None}
        return self._parsed

    async def ainvoke(
        self, input: Any, config: Optional[dict] = None, **kwargs: Any
    ) -> Any:
        return self.invoke(input, config=config, **kwargs)

    def batch(
        self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any
    ) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]

    async def abatch(
        self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any
    ) -> List[Any]:
        return [self.invoke(i, config=config, **kwargs) for i in inputs]
