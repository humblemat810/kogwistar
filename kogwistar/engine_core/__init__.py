"""Engine-core compatibility entrypoints with safe optional imports."""

from typing import TYPE_CHECKING

from kogwistar.engine_core.acl_protocol import (
    ACLAwareReadProtocol,
    ACLAwareWriteProtocol,
    ACLPolicyProtocol,
    require_acl_protocols,
)
from kogwistar.engine_core.async_named_projection import (
    AsyncPostgresNamedProjectionStore,
    AsyncSQLiteNamedProjectionStore,
)
from kogwistar.engine_core.embedding_profile import (
    AsyncNamedProjectionStore,
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
from kogwistar.engine_core.engine import GraphKnowledgeEngine, StorageBackendFactory
from kogwistar.engine_core.engine_sqlite import EngineSQLite, IndexJobRow
from kogwistar.engine_core.event_envelope import EntityEventEnvelope
from kogwistar.engine_core.indexing import IndexingSubsystem
from kogwistar.engine_core.jobs import JobQueueItem, JobQueueSubsystem
from kogwistar.engine_core.lifecycle import LifecycleSubsystem
from kogwistar.engine_core.multimodal import (
    EmbeddingReference,
    LegacyLocator,
    MultimodalSpan,
    PinnedLogicalRef,
    SpatialRegionLocator,
    TemporalIntervalLocator,
    TextRangeLocator,
    VideoRegionTrackLocator,
    VideoTrackFrame,
    VideoTrackManifest,
    VideoTrackRegion,
)
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
    ServiceHealthRegistry,
    ServiceHealthRepairResult,
    ServiceInstanceHealth,
)
from kogwistar.engine_core.storage_backend import (
    AtomicMutationCapability,
    NoopUnitOfWork,
    ProjectionCapabilityBackend,
    StorageBackend,
    UnitOfWork,
    get_atomic_mutation_capability,
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
from kogwistar.engine_core.types import (
    EdgePreAddHook,
    EngineType,
    ExtractionSchemaMode,
    NodePreAddHook,
    OffsetMismatchPolicy,
    OffsetRepairScorer,
    ResolvedExtractionSchemaMode,
    ToolCallIdFactory,
)
from kogwistar.engine_core.utils import (
    AliasBook,
    AliasBookStore,
    AliasKind,
    AliasKindMismatchError,
    UnknownAliasError,
)
from kogwistar.engine_core.vector_search import (
    VectorSearchHit,
    similarity_from_distance,
)
from kogwistar.typing_interfaces import TokenAwareEmbeddingFunction

if TYPE_CHECKING:
    from kogwistar.engine_core.chroma_backend import (
        ChromaBackend,
        ChromaStorageInspector,
    )
    from kogwistar.engine_core.engine_postgres import (
        EnginePostgresConfig,
        build_async_postgres_backend,
        build_postgres_backend,
    )
    from kogwistar.engine_core.engine_postgres_meta import (
        EnginePostgresMetaStore,
        IndexJob,
    )
    from kogwistar.engine_core.in_memory_backend import (
        InMemoryBackend,
        build_in_memory_backend,
    )
    from kogwistar.engine_core.postgres_backend import (
        AsyncPostgresUnitOfWork,
        PgVectorBackend,
        PgVectorConfig,
        PgVectorSchemaMismatchError,
        PostgresUnitOfWork,
    )

__all__ = [
    "SERVICE_HEALTH_PROJECTION_NAMESPACE",
    "ACLAwareReadProtocol",
    "ACLAwareWriteProtocol",
    "ACLPolicyProtocol",
    "AdjudicateSubsystem",
    "AliasBook",
    "AliasBookStore",
    "AliasKind",
    "AliasKindMismatchError",
    "AsyncNamedProjectionStore",
    "AsyncPostgresNamedProjectionStore",
    "AsyncPostgresUnitOfWork",
    "AsyncSQLiteNamedProjectionStore",
    "AtomicMutationCapability",
    "CheckpointRecoveryState",
    "ChromaBackend",
    "ChromaStorageInspector",
    "CorruptEmbeddingProfileError",
    "DaemonHealthState",
    "DeadLetterRecoveryState",
    "EdgePreAddHook",
    "EmbedSubsystem",
    "EmbeddingProfile",
    "EmbeddingProfileError",
    "EmbeddingProfileMismatchError",
    "EmbeddingProfileRegistry",
    "EmbeddingReference",
    "EmbeddingStorageInspector",
    "EmbeddingStorageState",
    "EnginePostgresConfig",
    "EnginePostgresMetaStore",
    "EngineSQLite",
    "EngineType",
    "EntityEventEnvelope",
    "ExtractSubsystem",
    "ExtractionSchemaMode",
    "GraphKnowledgeEngine",
    "InMemoryBackend",
    "IndexJob",
    "IndexJobRow",
    "IndexingSubsystem",
    "IngestSubsystem",
    "JobQueueItem",
    "JobQueueSubsystem",
    "LaneRecoveryState",
    "LegacyEmbeddingProfileError",
    "LegacyLocator",
    "LifecycleSubsystem",
    "MultimodalSpan",
    "NamedProjectionStore",
    "NodePreAddHook",
    "NoopUnitOfWork",
    "OffsetMismatchPolicy",
    "OffsetRepairScorer",
    "OutputReconciliationState",
    "PersistSubsystem",
    "PgVectorBackend",
    "PgVectorConfig",
    "PgVectorSchemaMismatchError",
    "PinnedLogicalRef",
    "PostgresUnitOfWork",
    "ProjectionCapabilityBackend",
    "QueueRecoveryState",
    "ReadSubsystem",
    "RecoveryAction",
    "RecoveryFinding",
    "RecoveryReport",
    "RecoverySubsystem",
    "RecoverySurface",
    "ResolvedExtractionSchemaMode",
    "ResumePolicy",
    "RollbackSubsystem",
    "RunRecoveryState",
    "ServiceHealthDefinition",
    "ServiceHealthRegistry",
    "ServiceHealthRepairResult",
    "ServiceInstanceHealth",
    "SpatialRegionLocator",
    "StorageBackend",
    "StorageBackendFactory",
    "TemporalIntervalLocator",
    "TextRangeLocator",
    "TokenAwareEmbeddingFunction",
    "ToolCallIdFactory",
    "UnitOfWork",
    "UnknownAliasError",
    "VectorSearchHit",
    "VideoRegionTrackLocator",
    "VideoTrackFrame",
    "VideoTrackManifest",
    "VideoTrackRegion",
    "WriteSubsystem",
    "build_async_postgres_backend",
    "build_in_memory_backend",
    "build_postgres_backend",
    "endpoint_fingerprint",
    "get_atomic_mutation_capability",
    "require_acl_protocols",
    "similarity_from_distance",
]


def __getattr__(name: str) -> object:
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
                build_async_postgres_backend,
                build_postgres_backend,
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
                AsyncPostgresUnitOfWork,
                PgVectorBackend,
                PgVectorConfig,
                PgVectorSchemaMismatchError,
                PostgresUnitOfWork,
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
