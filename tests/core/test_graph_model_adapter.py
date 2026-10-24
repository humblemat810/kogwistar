from __future__ import annotations

import json

import pytest
from pydantic_extension.model_slicing.mixin import ModeSlicingMixin

from kogwistar._rust_bridge import project_graph_model_slice
from kogwistar.acl import models as acl_models
from kogwistar.conversation import models as conversation_models
from kogwistar.engine_core import graph_contracts
from kogwistar.engine_core import models as engine_models
from kogwistar.engine_core.graph_contracts import (
    GRAPH_MODEL_ENVELOPE_VERSION,
    GraphModelAdapterError,
    build_graph_model_slice_envelope,
    build_graph_model_envelope,
    graph_model_class_for,
)
from kogwistar.engine_core.graph_slice_manifest import project_with_manifest
from kogwistar.engine_core.models import Edge, Node
from kogwistar.engine_core.subsystems.write import _validate_graph_model_contract
from kogwistar.runtime import models as runtime_models
from kogwistar.runtime.models import WorkflowRunNode, WorkflowRuntimeEdge


def _mentions(excerpt: str) -> list[dict[str, list[dict[str, object]]]]:
    return [
        {
            "spans": [
                {
                    "collection_page_url": "collection/doc",
                    "document_page_url": "document/doc",
                    "doc_id": "doc",
                    "insertion_method": "test",
                    "start_char": 0,
                    "end_char": len(excerpt),
                    "excerpt": excerpt,
                }
            ]
        }
    ]


def test_base_graph_models_get_explicit_versioned_envelopes() -> None:
    node = Node(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )
    edge = Edge(
        label="edge",
        type="relationship",
        summary="An edge",
        relation="relates_to",
        source_ids=["node-a"],
        target_ids=["node-b"],
        source_edge_ids=None,
        target_edge_ids=None,
        mentions=_mentions("edge"),
    )

    node_envelope = build_graph_model_envelope(node)
    edge_envelope = build_graph_model_envelope(edge)

    assert node_envelope["schema_version"] == GRAPH_MODEL_ENVELOPE_VERSION
    assert node_envelope["model_class"] == "node"
    assert edge_envelope["model_class"] == "edge"
    assert edge_envelope["payload"]["source_edge_ids"] is None
    assert edge_envelope["payload"]["target_edge_ids"] is None
    assert node_envelope["payload"] != node.model_dump(field_mode="llm_in", dump_format="json")


def test_unknown_subclass_fails_closed_instead_of_guessing() -> None:
    class FutureNode(Node):
        pass

    node = FutureNode(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )

    with pytest.raises(GraphModelAdapterError, match="unsupported graph model class"):
        graph_model_class_for(node)


def test_every_mapped_graph_model_preserves_field_slicing_contract() -> None:
    """Mapped Rust seam classes must remain governed by pydantic-extension."""

    model_modules = (
        engine_models,
        acl_models,
        conversation_models,
        runtime_models,
    )
    for class_name in graph_contracts._MODEL_CLASS_BY_NAME:
        model_class = next(
            (
                getattr(module, class_name, None)
                for module in model_modules
                if getattr(module, class_name, None) is not None
            ),
            None,
        )
        assert model_class is not None, class_name
        assert issubclass(model_class, ModeSlicingMixin), class_name


def test_python_slice_envelope_keeps_canonical_payload_and_manifest() -> None:
    node = Node(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )
    envelope = build_graph_model_slice_envelope(node, field_mode="llm_in")
    assert envelope["payload"] == node.model_dump(
        field_mode="backend", dump_format="json"
    )
    assert envelope["slice"]["mode"] == "llm_in"
    assert envelope["slice"]["manifest"]["schema_version"] == 1


def test_rust_slice_bridge_uses_the_declared_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeExtension:
        def project_graph_model_slice(self, payload: str) -> str:
            calls.append(payload)
            return '{"schema_version":1,"model_class":"node","mode":"llm_in","payload":{"summary":"visible"}}'

    monkeypatch.setattr(
        "kogwistar._rust_bridge._load_extension", lambda: FakeExtension()
    )
    projected = project_graph_model_slice(
        {
            "schema_version": 1,
            "model_class": "node",
            "payload": {"summary": "visible", "secret": "backend-only"},
            "slice": {
                "schema_version": 1,
                "mode": "llm_in",
                "manifest": {"schema_version": 1, "fields": {}},
            },
        }
    )

    assert projected["payload"] == {"summary": "visible"}
    assert len(calls) == 1


@pytest.mark.requires_rust
def test_built_rust_extension_projects_python_slice_envelope() -> None:
    native = pytest.importorskip("kogwistar._rust")
    models = [
        Node(
            label="node",
            type="entity",
            summary="A node",
            mentions=_mentions("node"),
        ),
        Edge(
            label="edge",
            type="relationship",
            summary="An edge",
            relation="relates_to",
            source_ids=["node-a"],
            target_ids=["node-b"],
            source_edge_ids=None,
            target_edge_ids=None,
            mentions=_mentions("edge"),
        ),
    ]

    for model in models:
        for mode in ("backend", "frontend", "dto", "llm", "llm_in"):
            envelope = build_graph_model_slice_envelope(model, field_mode=mode)
            projected = native.project_graph_model_slice(json.dumps(envelope))
            result = json.loads(projected)
            assert result["mode"] == mode
            expected = project_with_manifest(type(model), model, mode)
            assert result["payload"] == expected, mode


def test_python_slice_envelope_accepts_all_extension_modes() -> None:
    node = Node(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )

    for mode in ("backend", "frontend", "dto", "llm", "llm_in"):
        envelope = build_graph_model_slice_envelope(node, field_mode=mode)
        assert envelope["slice"]["mode"] == mode
        assert envelope["payload"] == node.model_dump(
            field_mode="backend", dump_format="json"
        )


def test_python_contract_mode_does_not_cross_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Node(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )

    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "python")

    def native_must_not_run(_: object) -> None:
        raise AssertionError("Python contract mode crossed the native seam")

    monkeypatch.setattr(
        "kogwistar._rust_bridge.validate_graph_model_family", native_must_not_run
    )
    assert _validate_graph_model_contract(node) is None


def test_shadow_contract_mode_validates_the_versioned_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Node(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )
    captured: list[dict[str, object]] = []

    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "shadow")
    monkeypatch.setattr(
        "kogwistar._rust_bridge.validate_graph_model_family",
        lambda payload: captured.append(payload),
    )
    envelope = _validate_graph_model_contract(node)

    assert captured[0]["schema_version"] == GRAPH_MODEL_ENVELOPE_VERSION
    assert captured[0]["model_class"] == "node"
    assert envelope == captured[0]


def test_workflow_runtime_subtypes_have_explicit_graph_wire_tags() -> None:
    run_node = WorkflowRunNode(
        label="run",
        type="entity",
        summary="workflow run",
        mentions=_mentions("run"),
        metadata={
            "entity_type": "workflow_run",
            "workflow_id": "wf:demo",
            "run_id": "run:1",
            "conversation_id": "conversation:1",
            "turn_node_id": "turn:1",
        },
    )
    runtime_edge = WorkflowRuntimeEdge(
        label="runtime edge",
        type="relationship",
        summary="runtime transition",
        relation="next",
        source_ids=["run:1"],
        target_ids=["step:1"],
        source_edge_ids=None,
        target_edge_ids=None,
        mentions=_mentions("runtime transition"),
        metadata={"relation": "next"},
    )

    assert build_graph_model_envelope(run_node)["model_class"] == "workflow_run_node"
    assert (
        build_graph_model_envelope(runtime_edge)["model_class"]
        == "workflow_runtime_edge"
    )


def test_hyperedge_keeps_the_edge_wire_identity_with_multiple_incidences() -> None:
    edge = Edge(
        label="hyperedge",
        type="relationship",
        summary="many-way relation",
        relation="joins",
        source_ids=["node-a", "node-b"],
        target_ids=["node-c", "node-d"],
        source_edge_ids=["edge-parent"],
        target_edge_ids=None,
        mentions=_mentions("many-way relation"),
    )

    envelope = build_graph_model_envelope(edge)

    assert envelope["model_class"] == "edge"
    assert envelope["payload"]["id"] == edge.id
    assert envelope["payload"]["source_ids"] == ["node-a", "node-b"]
    assert envelope["payload"]["source_edge_ids"] == ["edge-parent"]

