from __future__ import annotations

from typing import Any

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Grounding, Node, Span
from kogwistar.engine_core.utils.refs import backend_update_record_lifecycle
from tests._helpers.fake_backend import build_fake_backend


pytestmark = pytest.mark.regression

# This is the portable base contract.  The pgvector parameter below joins
# ci_full when a live PostgreSQL fixture is available.


class _CountingEmbeddingFunction:
    def __init__(self) -> None:
        self.calls = 0
        self.forbid_calls = False

    @staticmethod
    def name() -> str:
        return "lifecycle-no-reembed"

    # Chroma 1.5 validates this protocol signature when the collection is
    # constructed.  Keep it aligned with Chroma's ``EmbeddingFunction`` so the
    # regression exercises the real backend rather than a compatibility shim.
    def __call__(self, input: Any) -> list[list[float]]:
        self.calls += 1
        if self.forbid_calls:
            raise AssertionError(
                "metadata-only lifecycle update tried to recompute an embedding"
            )
        return [[0.1, 0.2, 0.3] for _ in input]


@pytest.mark.ci_full
@pytest.mark.parametrize("backend_kind", ("fake", "chroma", "pgvector"))
def test_lifecycle_patch_is_metadata_only_and_preserves_embedding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    backend_kind: str,
    request: pytest.FixtureRequest,
) -> None:
    """Lifecycle must preserve vectors and fail fast on accidental re-embedding."""
    embedder = _CountingEmbeddingFunction()
    kwargs: dict[str, Any] = {
        "persist_directory": str(tmp_path / backend_kind),
        "embedding_function": embedder,
    }
    if backend_kind == "fake":
        kwargs["backend_factory"] = build_fake_backend
    elif backend_kind == "chroma":
        pytest.importorskip("chromadb")
    else:
        pytest.importorskip("sqlalchemy")
        pytest.importorskip("pgvector")
        from kogwistar.engine_core.postgres_backend import PgVectorBackend

        sa_engine = request.getfixturevalue("sa_engine")
        pg_schema = request.getfixturevalue("pg_schema")
        if sa_engine is None or pg_schema is None:
            pytest.skip("pgvector lifecycle contract needs live PostgreSQL")
        kwargs["backend"] = PgVectorBackend(
            engine=sa_engine, embedding_dim=3, schema=pg_schema
        )
    engine = GraphKnowledgeEngine(**kwargs)
    try:
        node = Node(
            id="lifecycle-node",
            label="Lifecycle node",
            type="entity",
            summary="keeps its vector",
            mentions=[Grounding(spans=[Span.from_dummy_for_document()])],
            doc_id="doc:lifecycle",
            metadata={"level_from_root": 0},
            domain_id=None,
            canonical_entity_id=None,
            properties={},
            embedding=[0.25, 0.5, 0.75],
        )
        engine.write.add_node(node)
        # Base insertion and derived indexes may legitimately invoke the
        # configured embedder.  Lifecycle must add no further invocation.
        calls_before_lifecycle = embedder.calls
        # This is deliberately stronger than an after-the-fact call count.  If
        # an extensible backend receives a document and tries to recompute a
        # vector, it must raise at the offending backend call immediately.
        embedder.forbid_calls = True

        original_get = engine.backend.node_get
        original_update = engine.backend.node_update
        requested_includes: list[list[str] | None] = []
        updates: list[dict[str, Any]] = []

        def spy_node_get(**kwargs: Any) -> Any:
            requested_includes.append(kwargs.get("include"))
            return original_get(**kwargs)

        def spy_node_update(**kwargs: Any) -> Any:
            updates.append(kwargs)
            return original_update(**kwargs)

        monkeypatch.setattr(engine.backend, "node_get", spy_node_get)
        monkeypatch.setattr(engine.backend, "node_update", spy_node_update)

        assert engine.lifecycle.tombstone_node(node.id)

        # Lifecycle patch reads metadata only.  The optional derived-index delete
        # fingerprint may separately inspect documents+metadata, but it must not
        # turn the projection patch itself into a document/vector update.
        assert requested_includes[0] == ["metadatas"]
        assert all(
            include in (["metadatas"], ["documents", "metadatas"])
            for include in requested_includes
        )
        assert len(updates) == 1
        assert updates[0]["ids"] == [node.id]
        assert "documents" not in updates[0]
        assert "embeddings" not in updates[0]
        assert updates[0]["metadatas"][0]["lifecycle_status"] == "tombstoned"
        assert embedder.calls == calls_before_lifecycle
        lifecycle_get_count = len(requested_includes)

        # The legacy exported utility is retained for compatibility.  It must
        # obey the same no-document/no-vector contract or a future caller could
        # reintroduce the Chroma re-embedding bug outside LifecycleSubsystem.
        assert backend_update_record_lifecycle(
            backend=engine.backend,
            kind="node",
            record_id=node.id,
            lifecycle_patch={"legacy_lifecycle_contract": True},
            safe_json_dict_fn=lambda _value: {},
            merge_meta_fn=lambda base, patch: {**base, **patch},
        )
        assert len(requested_includes) == lifecycle_get_count + 1
        assert requested_includes[-1] == ["metadatas"]
        assert len(updates) == 2
        assert "documents" not in updates[1]
        assert "embeddings" not in updates[1]
        assert embedder.calls == calls_before_lifecycle

        stored = original_get(ids=[node.id], include=["embeddings", "metadatas"])
        # Chroma returns an ndarray while the fake backend returns a list.
        # Compare the one stored vector, not its container representation.
        assert list(stored["embeddings"][0]) == pytest.approx([0.25, 0.5, 0.75])
        assert stored["metadatas"][0]["lifecycle_status"] == "tombstoned"
        assert stored["metadatas"][0]["legacy_lifecycle_contract"] is True
    finally:
        engine.close()
