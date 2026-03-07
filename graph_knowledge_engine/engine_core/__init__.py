"""Engine-core compatibility entrypoints with safe optional imports."""

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.engine_sqlite import EngineSQLite, IndexJobRow
from graph_knowledge_engine.engine_core.indexing import IndexingSubsystem
from graph_knowledge_engine.engine_core.lifecycle import LifecycleSubsystem
from graph_knowledge_engine.engine_core.subsystems import (
    AdjudicateSubsystem,
    EmbedSubsystem,
    ExtractSubsystem,
    IngestSubsystem,
    PersistSubsystem,
    ReadSubsystem,
    RollbackSubsystem,
    WriteSubsystem,
)
from graph_knowledge_engine.engine_core.storage_backend import NoopUnitOfWork, StorageBackend, UnitOfWork
from graph_knowledge_engine.engine_core.types import (
    EngineType,
    ExtractionSchemaMode,
    OffsetMismatchPolicy,
    OffsetRepairScorer,
    ResolvedExtractionSchemaMode,
)

__all__ = [
    "GraphKnowledgeEngine",
    "EnginePostgresConfig",
    "build_postgres_backend",
    "EnginePostgresMetaStore",
    "IndexJob",
    "EngineSQLite",
    "IndexJobRow",
    "IndexingSubsystem",
    "LifecycleSubsystem",
    "PgVectorBackend",
    "PgVectorConfig",
    "PostgresUnitOfWork",
    "ChromaBackend",
    "NoopUnitOfWork",
    "StorageBackend",
    "UnitOfWork",
    "EngineType",
    "ExtractionSchemaMode",
    "ResolvedExtractionSchemaMode",
    "OffsetMismatchPolicy",
    "OffsetRepairScorer",
    "ReadSubsystem",
    "WriteSubsystem",
    "ExtractSubsystem",
    "PersistSubsystem",
    "RollbackSubsystem",
    "AdjudicateSubsystem",
    "IngestSubsystem",
    "EmbedSubsystem",
]


def __getattr__(name: str):
    if name == "ChromaBackend":
        from graph_knowledge_engine.engine_core.chroma_backend import ChromaBackend

        return ChromaBackend

    if name in {"EnginePostgresConfig", "build_postgres_backend"}:
        try:
            from graph_knowledge_engine.engine_core.engine_postgres import EnginePostgresConfig, build_postgres_backend
        except Exception as e:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Postgres backend support requires optional dependencies. "
                "Install with: pip install 'kogwistar[pgvector]'"
            ) from e
        return {"EnginePostgresConfig": EnginePostgresConfig, "build_postgres_backend": build_postgres_backend}[name]

    if name in {"EnginePostgresMetaStore", "IndexJob"}:
        try:
            from graph_knowledge_engine.engine_core.engine_postgres_meta import EnginePostgresMetaStore, IndexJob
        except Exception as e:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Postgres meta store requires optional dependencies. "
                "Install with: pip install 'kogwistar[pgvector]'"
            ) from e
        return {"EnginePostgresMetaStore": EnginePostgresMetaStore, "IndexJob": IndexJob}[name]

    if name in {"PgVectorBackend", "PgVectorConfig", "PostgresUnitOfWork"}:
        try:
            from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend, PgVectorConfig, PostgresUnitOfWork
        except Exception as e:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "PgVector backend requires optional dependencies. "
                "Install with: pip install 'kogwistar[pgvector]'"
            ) from e
        return {
            "PgVectorBackend": PgVectorBackend,
            "PgVectorConfig": PgVectorConfig,
            "PostgresUnitOfWork": PostgresUnitOfWork,
        }[name]

    raise AttributeError(name)
