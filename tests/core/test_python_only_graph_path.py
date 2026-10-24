from __future__ import annotations

import pytest

from kogwistar._rust_bridge import RustExtensionUnavailableError
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from tests._helpers.graph_builders import build_entity_node, build_relationship_edge
from tests.conftest import FakeEmbeddingFunction, build_fake_backend


pytestmark = [pytest.mark.ci, pytest.mark.core]


@pytest.mark.parametrize("explicit_python", [True, False], ids=["explicit", "legacy-default"])
def test_python_only_graph_path_does_not_load_rust(
    monkeypatch: pytest.MonkeyPatch, tmp_path, explicit_python: bool
) -> None:
    """Python graph execution remains complete when the native loader is absent."""

    for selector in (
        "KOGWISTAR_IMPL_MODE",
        "KOGWISTAR_IMPL_CONTRACTS",
        "KOGWISTAR_IMPL_GRAPH_STORE",
        "KOGWISTAR_IMPL_META_STORE",
        "KOGWISTAR_IMPL_POSTGRES_AUTHORITY",
        "KOGWISTAR_IMPL_RUNTIME",
        "KOGWISTAR_IMPL_SERVER",
    ):
        if explicit_python:
            monkeypatch.setenv(selector, "python")
        else:
            monkeypatch.delenv(selector, raising=False)

    def unavailable() -> None:
        raise RustExtensionUnavailableError("native loader must not be called")

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", unavailable)
    engine = GraphKnowledgeEngine(
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddingFunction(dim=3),
        backend_factory=build_fake_backend,
    )
    source = build_entity_node(
        node_id="python-source",
        doc_id="python-doc",
        label="source",
        summary="source node",
        embedding=[1.0, 0.0, 0.0],
    )
    target = build_entity_node(
        node_id="python-target",
        doc_id="python-doc",
        label="target",
        summary="target node",
        embedding=[0.0, 1.0, 0.0],
    )
    edge = build_relationship_edge(
        edge_id="python-edge",
        src=source.id,
        tgt=target.id,
        doc_id="python-doc",
        relation="supports",
        label="supports",
        summary="source supports target",
        embedding=[0.0, 0.0, 1.0],
    )

    engine.write.add_node(source)
    engine.write.add_node(target)
    engine.write.add_edge(edge)

    assert [node.id for node in engine.read.get_nodes(ids=[source.id, target.id])] == [
        source.id,
        target.id,
    ]
    assert [item.id for item in engine.read.get_edges(ids=[edge.id])] == [edge.id]
    neighbors = engine.query.neighbors(source.id)
    assert neighbors["edges"] == {edge.id}
    assert neighbors["nodes"] == {target.id}

