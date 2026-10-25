from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from kogwistar.acl.models import ACLEdge, ACLNode
from kogwistar.conversation.models import ConversationEdge, ConversationNode
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import (
    AssocFlattenedLLMEdge,
    AssocFlattenedLLMNode,
    Edge,
    EdgeMixin,
    FlattenedLLMEdge,
    FlattenedLLMNode,
    Grounding,
    LLMEdge,
    LLMNode,
    Node,
    PureChromaEdge,
    PureChromaNode,
    Span,
)
from kogwistar.engine_core.utils.refs import backend_update_record_lifecycle
from kogwistar.runtime.models import WorkflowEdge, WorkflowNode
from tests._helpers.fake_backend import build_fake_backend
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge

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


def test_edge_model_shares_tombstone_and_level_metadata_contract() -> None:
    edge = build_relationship_edge(
        edge_id="edge-contract",
        src="source",
        tgt="target",
        doc_id="doc:edge-contract",
        metadata={"level_from_root": 2},
    )

    assert edge.metadata["lifecycle_status"] == "active"
    assert edge.metadata["redirect_to_id"] is None
    assert edge.level_from_root == 2

    explicit_level = Edge.model_validate(
        {
            **edge.model_dump(field_mode="backend", dump_format="json"),
            "level_from_root": 3,
        }
    )
    assert explicit_level.level_from_root == 3
    assert explicit_level.metadata["level_from_root"] == 3

    negative_level = Edge.model_validate(
        {
            **edge.model_dump(field_mode="backend", dump_format="json"),
            "level_from_root": -1,
        }
    )
    assert negative_level.level_from_root == -1
    assert negative_level.metadata["level_from_root"] == -1

    explicit_null_level = Edge.model_validate(
        {
            **edge.model_dump(field_mode="backend", dump_format="json"),
            "level_from_root": None,
            "metadata": {**edge.metadata, "level_from_root": 2},
        }
    )
    assert explicit_null_level.level_from_root is None
    assert explicit_null_level.metadata["level_from_root"] == 2

    for raw_level, expected in [("2", 2), (True, 1), (2.0, 2)]:
        payload = edge.model_dump(field_mode="backend", dump_format="json")
        payload.pop("level_from_root")
        payload["metadata"]["level_from_root"] = raw_level
        coerced_level = Edge.model_validate(payload)
        assert coerced_level.level_from_root == expected
        assert coerced_level.metadata["level_from_root"] == expected

    invalid_level = edge.model_dump(field_mode="backend", dump_format="json")
    invalid_level.pop("level_from_root")
    invalid_level["metadata"]["level_from_root"] = 2.5
    with pytest.raises(ValidationError, match="valid integer"):
        Edge.model_validate(invalid_level)

    validated = Edge.model_validate(
        {
            **edge.model_dump(field_mode="backend", dump_format="json"),
            "metadata": {"lifecycle_status": "deleted"},
        }
    )
    assert validated.metadata["lifecycle_status"] == "tombstoned"
    assert validated.metadata["redirect_to_id"] is None
    with pytest.raises(ValidationError, match="lifecycle_status"):
        Edge.model_validate(
            {
                **edge.model_dump(field_mode="backend", dump_format="json"),
                "metadata": {"lifecycle_status": "purged"},
            }
        )


def test_edge_model_extends_node_mixin_contract() -> None:
    node_mixins = {base for base in Node.__mro__ if base.__name__.endswith("Mixin")}
    edge_mixins = {base for base in Edge.__mro__ if base.__name__.endswith("Mixin")}

    assert node_mixins <= edge_mixins
    assert EdgeMixin in edge_mixins
    assert len(edge_mixins) > len(node_mixins)


@pytest.mark.parametrize(
    ("node_type", "edge_type"),
    [
        (Node, Edge),
        (LLMNode, LLMEdge),
        (FlattenedLLMNode, FlattenedLLMEdge),
        (AssocFlattenedLLMNode, AssocFlattenedLLMEdge),
        (PureChromaNode, PureChromaEdge),
        (ACLNode, ACLEdge),
        (ConversationNode, ConversationEdge),
        (WorkflowNode, WorkflowEdge),
    ],
)
def test_each_node_edge_family_preserves_shared_mixin_contract(
    node_type: type[object], edge_type: type[object]
) -> None:
    """Edges retain shared graph capabilities but may omit node-only semantics."""
    node_only_mixins = {"ConversationRoleMixin"}
    node_mixins = {
        base.__name__
        for base in node_type.__mro__
        if base.__name__.endswith("Mixin")
    }
    edge_mixins = {
        base.__name__
        for base in edge_type.__mro__
        if base.__name__.endswith("Mixin")
    }

    assert (node_mixins - node_only_mixins) <= edge_mixins
    assert "EdgeMixin" in edge_mixins
    assert edge_mixins > (node_mixins - node_only_mixins)


def test_semantic_queries_filter_tombstones_from_stale_projections(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        backend_factory=build_fake_backend,
        embedding_function=_CountingEmbeddingFunction(),
    )
    node = Node(
        id="query-node-tombstone",
        label="query node",
        type="entity",
        summary="query node",
        mentions=[Grounding(spans=[Span.from_dummy_for_document()])],
        doc_id="doc:query",
        metadata={},
        embedding=[1.0, 0.0, 0.0],
    )
    edge = build_relationship_edge(
        edge_id="query-edge-tombstone",
        src="query-edge-source",
        tgt="query-edge-target",
        doc_id="doc:query",
        metadata={},
        embedding=[1.0, 0.0, 0.0],
    )

    engine.write.add_node(node)
    engine.write.add_node(
        build_entity_node(
            node_id="query-edge-source",
            doc_id="doc:query",
            embedding=[0.0, 1.0, 0.0],
        )
    )
    engine.write.add_node(
        build_entity_node(
            node_id="query-edge-target",
            doc_id="doc:query",
            embedding=[0.0, 0.0, 1.0],
        )
    )
    engine.write.add_edge(edge)
    assert engine.lifecycle.tombstone_node(node.id)
    assert engine.lifecycle.tombstone_edge(edge.id)
    node = engine.read.get_nodes(ids=[node.id], resolve_mode="include_tombstones")[0]
    edge = engine.read.get_edges(ids=[edge.id], resolve_mode="include_tombstones")[0]
    assert node.metadata["lifecycle_status"] == "tombstoned"
    assert edge.metadata["lifecycle_status"] == "tombstoned"

    def query_result(item: Node | Edge) -> dict[str, object]:
        return {
            "ids": [[item.id]],
            "documents": [
                [json.dumps(item.model_dump(field_mode="backend", dump_format="json"))]
            ],
            "embeddings": [[[1.0, 0.0, 0.0]]],
            "metadatas": [[item.metadata]],
        }

    monkeypatch.setattr(engine.backend, "node_query", lambda **_kwargs: query_result(node))
    monkeypatch.setattr(engine.backend, "edge_query", lambda **_kwargs: query_result(edge))
    try:
        assert engine.read.query_nodes(query_embeddings=[[1.0, 0.0, 0.0]]) == [[]]
        assert engine.read.query_edges(query_embeddings=[[1.0, 0.0, 0.0]]) == [[]]
    finally:
        engine.close()


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
        # Compare the one stored vector, not its container embedding.
        assert list(stored["embeddings"][0]) == pytest.approx([0.25, 0.5, 0.75])
        assert stored["metadatas"][0]["lifecycle_status"] == "tombstoned"
        assert stored["metadatas"][0]["legacy_lifecycle_contract"] is True
    finally:
        engine.close()
