"""Engine-core compatibility entrypoints.

This package provides explicit, wildcard-free imports that point to current
legacy modules while we migrate implementation files incrementally.
"""

from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.engine_postgres import EnginePostgresConfig, build_postgres_backend
from graph_knowledge_engine.engine_core.engine_postgres_meta import EnginePostgresMetaStore, IndexJob
from graph_knowledge_engine.engine_core.engine_sqlite import EngineSQLite, IndexJobRow
from graph_knowledge_engine.engine_core.indexing import IndexingSubsystem
from graph_knowledge_engine.engine_core.lifecycle import LifecycleSubsystem
from graph_knowledge_engine.engine_core.postgres_backend import PgVectorBackend, PgVectorConfig, PostgresUnitOfWork
from graph_knowledge_engine.engine_core.storage_backend import ChromaBackend, NoopUnitOfWork, StorageBackend, UnitOfWork

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
]
