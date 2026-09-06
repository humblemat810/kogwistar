"""Engine-core compatibility entrypoints with safe optional imports."""

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.engine_sqlite import EngineSQLite, IndexJobRow
from kogwistar.engine_core.event_envelope import EntityEventEnvelope
from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.engine_core.jobs import JobQueueItem, JobQueueSubsystem
from kogwistar.engine_core.lifecycle import LifecycleSubsystem
from kogwistar.engine_core.recovery import (
    CheckpointRecoveryState,
    DaemonHealthState,
    DeadLetterRecoveryState,
    LaneRecoveryState,
    OutputReconciliationState,
    QueueRecoveryState,
    RecoveryAction,
    RecoveryFinding,
    RecoveryReport,
    RecoverySubsystem,
    RecoverySurface,
    ResumePolicy,
    RunRecoveryState,
)
from kogwistar.engine_core.service_health import (
    SERVICE_HEALTH_PROJECTION_NAMESPACE,
    ServiceHealthDefinition,
    ServiceHealthRepairResult,
    ServiceHealthRegistry,
    ServiceInstanceHealth,
)
from kogwistar.engine_core.subsystems import (
    AdjudicateSubsystem,
    EmbedSubsystem,
    ExtractSubsystem,
    IngestSubsystem,
    PersistSubsystem,
    ReadSubsystem,
    RollbackSubsystem,
    WriteSubsystem,
)
from kogwistar.engine_core.storage_backend import (
    NoopUnitOfWork,
    StorageBackend,
    UnitOfWork,
)
from kogwistar.engine_core.embedding_profile import (
    CorruptEmbeddingProfileError,
    EmbeddingProfile,
    EmbeddingProfileError,
    EmbeddingProfileMismatchError,
    EmbeddingProfileRegistry,
    EmbeddingStorageInspector,
    EmbeddingStorageState,
    LegacyEmbeddingProfileError,
    NamedProjectionStore,
    endpoint_fingerprint,
)
from kogwistar.engine_core.types import (
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
    "build_async_postgres_backend",
    "EnginePostgresMetaStore",
    "IndexJob",
    "EngineSQLite",
    "IndexJobRow",
    "EntityEventEnvelope",
    "IndexingSubsystem",
    "JobQueueItem",
    "JobQueueSubsystem",
    "CheckpointRecoveryState",
    "DaemonHealthState",
    "DeadLetterRecoveryState",
    "LaneRecoveryState",
    "OutputReconciliationState",
    "QueueRecoveryState",
    "RecoveryAction",
    "RecoveryFinding",
    "RecoveryReport",
    "RecoverySubsystem",
    "RecoverySurface",
    "ResumePolicy",
    "RunRecoveryState",
    "SERVICE_HEALTH_PROJECTION_NAMESPACE",
    "ServiceHealthDefinition",
    "ServiceHealthRepairResult",
    "ServiceHealthRegistry",
    "ServiceInstanceHealth",
    "LifecycleSubsystem",
    "PgVectorBackend",
    "PgVectorConfig",
    "PgVectorSchemaMismatchError",
    "PostgresUnitOfWork",
    "AsyncPostgresUnitOfWork",
    "ChromaBackend",
    "ChromaStorageInspector",
    "NoopUnitOfWork",
    "StorageBackend",
    "UnitOfWork",
    "CorruptEmbeddingProfileError",
    "EmbeddingProfile",
    "EmbeddingProfileError",
    "EmbeddingProfileMismatchError",
    "EmbeddingProfileRegistry",
    "EmbeddingStorageInspector",
    "EmbeddingStorageState",
    "LegacyEmbeddingProfileError",
    "NamedProjectionStore",
    "endpoint_fingerprint",
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
    "InMemoryBackend",
    "build_in_memory_backend",
]


def __getattr__(name: str):
    if name in {"ChromaBackend", "ChromaStorageInspector"}:
        from kogwistar.engine_core.chroma_backend import (
            ChromaBackend,
            ChromaStorageInspector,
        )

        return {"ChromaBackend": ChromaBackend, "ChromaStorageInspector": ChromaStorageInspector}[name]

    if name in {"InMemoryBackend", "build_in_memory_backend"}:
        from kogwistar.engine_core.in_memory_backend import (
            InMemoryBackend,
            build_in_memory_backend,
        )
        return {
            "InMemoryBackend": InMemoryBackend,
            "build_in_memory_backend": build_in_memory_backend,
        }[name]

    if name in {"EnginePostgresConfig", "build_postgres_backend", "build_async_postgres_backend"}:
        try:
            from kogwistar.engine_core.engine_postgres import (
                EnginePostgresConfig,
                build_postgres_backend,
                build_async_postgres_backend,
            )
        except Exception as e:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Postgres backend support requires optional dependencies. "
                "Install with: pip install 'kogwistar[pgvector]'"
            ) from e
        return {
            "EnginePostgresConfig": EnginePostgresConfig,
            "build_postgres_backend": build_postgres_backend,
            "build_async_postgres_backend": build_async_postgres_backend,
        }[name]

    if name in {"EnginePostgresMetaStore", "IndexJob"}:
        try:
            from kogwistar.engine_core.engine_postgres_meta import (
                EnginePostgresMetaStore,
                IndexJob,
            )
        except Exception as e:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Postgres meta store requires optional dependencies. "
                "Install with: pip install 'kogwistar[pgvector]'"
            ) from e
        return {
            "EnginePostgresMetaStore": EnginePostgresMetaStore,
            "IndexJob": IndexJob,
        }[name]

    if name in {
        "PgVectorBackend",
        "PgVectorConfig",
        "PgVectorSchemaMismatchError",
        "PostgresUnitOfWork",
        "AsyncPostgresUnitOfWork",
    }:
        try:
            from kogwistar.engine_core.postgres_backend import (
                PgVectorBackend,
                PgVectorConfig,
                PgVectorSchemaMismatchError,
                PostgresUnitOfWork,
                AsyncPostgresUnitOfWork,
            )
        except Exception as e:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "PgVector backend requires optional dependencies. "
                "Install with: pip install 'kogwistar[pgvector]'"
            ) from e
        return {
            "PgVectorBackend": PgVectorBackend,
            "PgVectorConfig": PgVectorConfig,
            "PgVectorSchemaMismatchError": PgVectorSchemaMismatchError,
            "PostgresUnitOfWork": PostgresUnitOfWork,
            "AsyncPostgresUnitOfWork": AsyncPostgresUnitOfWork,
        }[name]

    raise AttributeError(name)
