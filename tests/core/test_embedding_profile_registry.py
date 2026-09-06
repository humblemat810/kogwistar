from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kogwistar.engine_core.embedding_profile import (
    CorruptEmbeddingProfileError,
    EmbeddingProfile,
    EmbeddingProfileMismatchError,
    EmbeddingProfileRegistry,
    EmbeddingStorageState,
    LegacyEmbeddingProfileError,
    endpoint_fingerprint,
)
from kogwistar.engine_core.engine_sqlite import EngineSQLite
from kogwistar.engine_core.in_memory_meta import InMemoryMetaStore


class _Inspector:
    def __init__(self, scope: str, *, persistent: bool = True, vector_count: int = 0):
        self.scope = scope
        self.persistent = persistent
        self.vector_count = vector_count

    def embedding_storage_scope(self) -> str:
        return self.scope

    def inspect_embedding_storage(self) -> EmbeddingStorageState:
        return EmbeddingStorageState(
            backend_kind="test",
            storage_scope=self.scope,
            persistent=self.persistent,
            vector_count=self.vector_count,
        )


class _FakeEmbeddings:
    def __init__(self, model: str = "default", dimension: int = 2) -> None:
        self.model = model
        self.dimension = dimension

    def name(self) -> str:
        return self.model

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [[float(index) for index in range(self.dimension)] for _ in input]


def _profile(model: str = "model", dimension: int = 2) -> EmbeddingProfile:
    return EmbeddingProfile(
        provider="fake",
        model=model,
        dimension=dimension,
        endpoint_fingerprint=endpoint_fingerprint("https://user:secret@example.test/embed?token=hidden"),
    )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_registry_binds_exactly_and_rejects_profile_change(kind: str, tmp_path: Path) -> None:
    metadata = InMemoryMetaStore() if kind == "memory" else EngineSQLite(tmp_path / "meta")
    metadata.ensure_initialized()
    registry = EmbeddingProfileRegistry(metadata)
    inspector = _Inspector("test:shared")

    assert registry.ensure_bound(inspector, _profile()) == _profile()
    assert registry.ensure_bound(inspector, _profile()) == _profile()
    with pytest.raises(EmbeddingProfileMismatchError):
        registry.ensure_bound(inspector, _profile(model="another"))


def test_registry_fails_closed_for_nonempty_legacy_storage() -> None:
    registry = EmbeddingProfileRegistry(InMemoryMetaStore())
    inspector = _Inspector("chroma:legacy", vector_count=1)

    with pytest.raises(LegacyEmbeddingProfileError):
        registry.ensure_bound(inspector, _profile())
    assert registry.ensure_bound(inspector, _profile(), allow_legacy_adoption=True) == _profile()


def test_registry_cas_race_cannot_bind_conflicting_profiles() -> None:
    metadata = InMemoryMetaStore()
    metadata.ensure_initialized()
    registry = EmbeddingProfileRegistry(metadata)
    inspector = _Inspector("chroma:raced")

    def bind(model: str):
        try:
            registry.ensure_bound(inspector, _profile(model=model))
            return "ok"
        except EmbeddingProfileMismatchError:
            return "mismatch"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(bind, ("first", "second")))
    assert outcomes == ["mismatch", "ok"]


def test_registry_rejects_unknown_profile_schema() -> None:
    metadata = InMemoryMetaStore()
    metadata.ensure_initialized()
    inspector = _Inspector("test:corrupt")
    metadata.replace_named_projection(
        "__kogwistar_embedding_profiles_v1__",
        "test:corrupt",
        {
            "profile_schema_version": 999,
            "embedding_profile": _profile().as_dict(),
            "embedding_fingerprint": _profile().fingerprint,
        },
        last_authoritative_seq=0,
        last_materialized_seq=0,
        projection_schema_version=1,
        materialization_status="bound",
    )

    with pytest.raises(CorruptEmbeddingProfileError, match="unsupported embedding profile payload schema"):
        EmbeddingProfileRegistry(metadata).ensure_bound(inspector, _profile())


def test_registry_uses_rust_named_projection_store_when_available(tmp_path: Path) -> None:
    pytest.importorskip("kogwistar._rust")
    from kogwistar.engine_core.rust_meta_sqlite import RustEngineSQLite

    metadata = RustEngineSQLite(tmp_path / "rust-meta")
    metadata.ensure_initialized()
    try:
        registry = EmbeddingProfileRegistry(metadata)
        inspector = _Inspector("rust:profile")
        assert registry.ensure_bound(inspector, _profile()) == _profile()
        assert registry.ensure_bound(inspector, _profile()) == _profile()
        with pytest.raises(EmbeddingProfileMismatchError):
            registry.ensure_bound(inspector, _profile(model="changed"))
    finally:
        metadata.close()


def test_chroma_persistent_profile_reopen_rejects_model_change(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    profile = _profile()
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=_FakeEmbeddings("model"),
        embedding_profile=profile,
    )
    engine.close()

    reopened = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=_FakeEmbeddings("model"),
        embedding_profile=profile,
    )
    reopened.close()

    with pytest.raises(EmbeddingProfileMismatchError):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            embedding_function=_FakeEmbeddings("changed"),
            embedding_profile=_profile(model="changed"),
        )


def test_chroma_nonempty_legacy_store_is_rejected_before_write(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    import chromadb
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection("nodes")
    collection.add(ids=["legacy"], documents=["legacy vector"], embeddings=[[1.0, 2.0]])

    with pytest.raises(LegacyEmbeddingProfileError):
        GraphKnowledgeEngine(
            persist_directory=str(tmp_path),
            embedding_function=_FakeEmbeddings(),
            embedding_profile=_profile(),
        )


def test_chroma_directories_have_independent_profile_bindings(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    first = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "first"),
        embedding_function=_FakeEmbeddings("first"),
        embedding_profile=_profile(model="first", dimension=2),
    )
    second = GraphKnowledgeEngine(
        persist_directory=str(tmp_path / "second"),
        embedding_function=_FakeEmbeddings("second", dimension=3),
        embedding_profile=_profile(model="second", dimension=3),
    )
    first.close()
    second.close()
