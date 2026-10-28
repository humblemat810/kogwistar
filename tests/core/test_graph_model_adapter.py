from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic_extension.model_slicing.mixin import ModeSlicingMixin

from kogwistar._rust_bridge import (
    RustParityError,
    project_graph_model_slice,
    project_graph_model_slices,
)
from kogwistar._rust_bridge import graph_model_capability_profiles
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
    GraphModelEnvelopeDict,
    _merge_mode_payloads,
)
from kogwistar.engine_core.graph_slice_manifest import project_with_manifest
from kogwistar.engine_core.models import (
    Edge,
    EdgeMixin,
    LevelAwareMixin,
    Node,
    TombstoneMixin,
)
from kogwistar.engine_core.subsystems.write import _validate_graph_model_contract
from kogwistar.runtime import models as runtime_models
from kogwistar.runtime.models import WorkflowRunNode, WorkflowRuntimeEdge


def _python_graph_models_by_wire_name() -> dict[str, type[ModeSlicingMixin]]:
    seen: set[type[ModeSlicingMixin]] = set()

    def collect_subclasses(base: type[ModeSlicingMixin]) -> None:
        for subclass in base.__subclasses__():
            if subclass in seen:
                continue
            seen.add(subclass)
            collect_subclasses(subclass)

    collect_subclasses(engine_models.GraphEntityBase)
    collect_subclasses(engine_models.GraphEntityRefBase)
    structural_bases = {
        engine_models.GraphEntityBase,
        engine_models.GraphEntityRefBase,
        engine_models.GraphEntityExtractionBase,
    }
    discovered_models = {
        model
        for model in seen
        if model.__module__.startswith("kogwistar.") and model not in structural_bases
    }
    discovered_names = [model.__name__ for model in discovered_models]
    registered = set(graph_contracts._MODEL_CLASS_BY_NAME)
    discovered_name_set = set(discovered_names)
    assert len(discovered_names) == len(discovered_name_set) and discovered_name_set == registered, {
        "duplicate_python_model_names": sorted(
            name for name in discovered_name_set if discovered_names.count(name) > 1
        ),
        "unregistered_python_models": sorted(discovered_name_set - registered),
        "missing_python_models": sorted(registered - discovered_name_set),
    }
    wire_names_by_model = {
        model: graph_contracts.graph_model_class_for(model.model_construct())
        for model in discovered_models
    }
    wire_names = list(wire_names_by_model.values())
    assert len(wire_names) == len(set(wire_names)), {
        "duplicate_wire_names": sorted(
            wire_name for wire_name in set(wire_names) if wire_names.count(wire_name) > 1
        )
    }
    return {wire_name: model for model, wire_name in wire_names_by_model.items()}


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


@pytest.mark.parametrize("model_kind", ["node", "edge"])
def test_persistent_graph_models_reject_non_finite_embeddings(model_kind: str) -> None:
    model_type = Node if model_kind == "node" else Edge
    payload: dict[str, object] = {
        "id": f"{model_kind}-finite-check",
        "label": model_kind,
        "type": "entity" if model_kind == "node" else "relationship",
        "summary": "finite vector contract",
        "mentions": _mentions("vector evidence"),
        "embedding": [0.25, float("nan")],
    }
    if model_kind == "edge":
        payload.update(
            {
                "relation": "relates_to",
                "source_ids": ["source"],
                "target_ids": ["target"],
                "source_edge_ids": None,
                "target_edge_ids": None,
            }
        )

    with pytest.raises(ValueError, match="embedding values must be finite"):
        model_type.model_validate(payload)


@pytest.mark.parametrize(
    "field_name",
    ["source_ids", "target_ids", "source_edge_ids", "target_edge_ids"],
)
def test_python_edge_rejects_empty_endpoint_ids(field_name: str) -> None:
    payload: dict[str, object] = {
        "id": "edge-endpoint-check",
        "label": "edge",
        "type": "relationship",
        "summary": "endpoint contract",
        "mentions": _mentions("edge evidence"),
        "source_ids": ["source"],
        "target_ids": ["target"],
        "source_edge_ids": None,
        "target_edge_ids": None,
        field_name: [""],
    }
    with pytest.raises(ValueError, match="edge endpoint IDs must not be empty"):
        Edge.model_validate(payload)


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
    node = engine_models.LLMNode(
        id="nn:slice-envelope-node",
        local_id="nn:slice-envelope-node",
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )
    envelope = build_graph_model_slice_envelope(node, field_mode="llm")
    backend_payload = node.model_dump(field_mode="backend", dump_format="json")
    target_payload = node.model_dump(field_mode="llm", dump_format="json")
    assert envelope["payload"]["mentions"][0]["spans"][0]["insertion_method"] == (
        backend_payload["mentions"][0]["spans"][0]["insertion_method"]
    )
    assert envelope["payload"]["mentions"][0]["spans"][0]["chunk_id"] == (
        target_payload["mentions"][0]["spans"][0]["chunk_id"]
    )
    assert envelope["slice"]["mode"] == "llm"
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


def test_rust_slice_bridge_batches_projection_without_changing_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeExtension:
        def project_graph_model_slices(self, payload: str) -> str:
            envelopes = json.loads(payload)
            return json.dumps(
                [
                    {
                        "schema_version": 1,
                        "model_class": item["model_class"],
                        "mode": item["slice"]["mode"],
                        "payload": {"summary": item["payload"]["summary"]},
                    }
                    for item in envelopes
                ]
            )

    monkeypatch.setattr("kogwistar._rust_bridge._load_extension", lambda: FakeExtension())
    projected = project_graph_model_slices(
        [
            {
                "schema_version": 1,
                "model_class": "node",
                "payload": {"summary": "one", "secret": "hidden"},
                "slice": {
                    "schema_version": 1,
                    "mode": "llm_in",
                    "manifest": {"schema_version": 1, "fields": {}},
                },
            },
            {
                "schema_version": 1,
                "model_class": "node",
                "payload": {"summary": "two", "secret": "hidden"},
                "slice": {
                    "schema_version": 1,
                    "mode": "llm_in",
                    "manifest": {"schema_version": 1, "fields": {}},
                },
            },
        ]
    )
    assert [item["payload"] for item in projected] == [
        {"summary": "one"},
        {"summary": "two"},
    ]


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


@pytest.mark.requires_rust
def test_built_rust_extension_matches_all_registered_model_slice_modes() -> None:
    """Every mapped model must keep Pydantic-extension field visibility in Rust."""
    native = pytest.importorskip("kogwistar._rust")

    for model_class in _python_graph_models_by_wire_name().values():
        # A null-valued instance exercises field inclusion/exclusion without
        # coupling this contract test to each model's domain validation rules.
        model = model_class.model_construct(
            **{field_name: None for field_name in model_class.model_fields}
        )
        for mode in ("backend", "frontend", "dto", "llm", "llm_in"):
            envelope = build_graph_model_slice_envelope(model, field_mode=mode)
            result = json.loads(native.project_graph_model_slice(json.dumps(envelope)))
            expected = model.model_dump(field_mode=mode, dump_format="json")
            assert result["model_class"] == envelope["model_class"]
            assert result["mode"] == mode
            assert result["payload"] == expected, (type(model).__name__, mode)


@pytest.mark.requires_rust
def test_built_rust_extension_preserves_real_llm_node_fields_in_every_slice() -> None:
    """Mode-only model fields and nested evidence survive the language seam."""
    native = pytest.importorskip("kogwistar._rust")
    model = engine_models.LLMNode(
        id="nn:actual-node",
        local_id="nn:actual-node",
        label="Actual node",
        type="entity",
        summary="A concrete LLM-visible graph node",
        domain_id="domain-hidden-from-llm-in",
        canonical_entity_id="canonical-entity-7",
        properties={"topic": "interoperability"},
        mentions=_mentions("nested evidence survives projection"),
    )

    for mode in ("backend", "frontend", "dto", "llm", "llm_in"):
        envelope = build_graph_model_slice_envelope(model, field_mode=mode)
        result = json.loads(native.project_graph_model_slice(json.dumps(envelope)))
        expected = model.model_dump(field_mode=mode, dump_format="json")
        assert result["payload"] == expected, mode


@pytest.mark.requires_rust
def test_native_capability_report_matches_corrected_doc_node_profile() -> None:
    pytest.importorskip("kogwistar._rust")
    profiles = graph_model_capability_profiles()

    assert len(profiles) == 26
    doc_node = next(profile for profile in profiles if profile["model_class"] == "doc_node")
    assert doc_node == {
        "schema_version": 1,
        "model_class": "doc_node",
        "entity_kind": "node",
        "graph_entity": True,
        "edge_entity": False,
        "metadata": False,
        "lifecycle": False,
        "level": False,
    }


@pytest.mark.requires_rust
def test_native_capability_profiles_match_python_model_inheritance() -> None:
    """Rust's graph capability report must track the Python model MRO."""
    pytest.importorskip("kogwistar._rust")
    python_models = _python_graph_models_by_wire_name()

    profiles = graph_model_capability_profiles()
    assert {profile["model_class"] for profile in profiles} == set(python_models)
    for profile in profiles:
        model = python_models[profile["model_class"]]
        fields = model.model_fields
        expected = {
            "graph_entity": issubclass(model, engine_models.GraphEntityBase),
            "edge_entity": issubclass(model, EdgeMixin),
            "metadata": "metadata" in fields,
            "lifecycle": issubclass(model, TombstoneMixin),
            "level": issubclass(model, LevelAwareMixin),
        }
        actual = {
            key: profile[key]
            for key in expected
        }
        assert actual == expected, profile["model_class"]


def test_graph_model_capability_contract_matches_shared_fixture() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "contracts"
        / "golden"
        / "graph-model-capabilities-v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == GRAPH_MODEL_ENVELOPE_VERSION

    python_models = _python_graph_models_by_wire_name()
    expected_classes = {
        model_class
        for group in fixture["profiles"]
        for model_class in group["classes"]
    }
    assert expected_classes == set(python_models)
    assert expected_classes == set(graph_contracts.SUPPORTED_GRAPH_MODEL_CLASSES)

    for group in fixture["profiles"]:
        expected = group["capabilities"]
        for model_class in group["classes"]:
            model = python_models[model_class]
            assert graph_model_class_for(model.model_construct()) == model_class
            actual = {
                "graph_entity": issubclass(model, engine_models.GraphEntityBase),
                "edge_entity": issubclass(model, EdgeMixin),
                "metadata": "metadata" in model.model_fields,
                "lifecycle": issubclass(model, TombstoneMixin),
                "level": issubclass(model, LevelAwareMixin),
            }
            assert group["entity_kind"] == ("edge" if actual["edge_entity"] else "node")
            assert actual == expected, model_class

    edge_node_pairs = fixture["edge_node_pairs"]
    paired_edges = [edge_name for edge_name, _ in edge_node_pairs]
    paired_nodes = [node_name for _, node_name in edge_node_pairs]
    registered_edges = {
        model_class
        for group in fixture["profiles"]
        if group["entity_kind"] == "edge"
        for model_class in group["classes"]
    }
    registered_nodes = {
        model_class
        for group in fixture["profiles"]
        if group["entity_kind"] == "node"
        for model_class in group["classes"]
    }
    assert len(paired_edges) == len(set(paired_edges))
    assert set(paired_edges) == registered_edges
    assert len(paired_nodes) == len(set(paired_nodes))
    assert registered_nodes - set(paired_nodes) == set(fixture["node_only_classes"])

    for edge_name, node_name in edge_node_pairs:
        edge_model = python_models[edge_name]
        node_model = python_models[node_name]
        edge_mixins = {
            base.__name__
            for base in edge_model.__mro__
            if base.__name__.endswith("Mixin")
        }
        node_mixins = {
            base.__name__
            for base in node_model.__mro__
            if base.__name__.endswith("Mixin")
        }
        node_only_mixins = {"ConversationRoleMixin"} if node_name == "conversation_node" else set()
        assert node_mixins - node_only_mixins <= edge_mixins, edge_name
        assert "EdgeMixin" in edge_mixins, edge_name


def test_python_slice_envelope_accepts_all_extension_modes() -> None:
    node = Node(
        label="node",
        type="entity",
        summary="A node",
        mentions=_mentions("node"),
    )

    def contains_projection(source: object, projected: object) -> bool:
        if isinstance(projected, dict):
            return isinstance(source, dict) and all(
                key in source and contains_projection(source[key], value)
                for key, value in projected.items()
            )
        if isinstance(projected, list):
            return (
                isinstance(source, list)
                and len(source) == len(projected)
                and all(
                    contains_projection(source_value, projected_value)
                    for source_value, projected_value in zip(
                        source, projected, strict=True
                    )
                )
            )
        return source == projected

    for mode in ("backend", "frontend", "dto", "llm", "llm_in"):
        envelope = build_graph_model_slice_envelope(node, field_mode=mode)
        assert envelope["slice"]["mode"] == mode
        backend_payload = node.model_dump(field_mode="backend", dump_format="json")
        target_payload = node.model_dump(field_mode=mode, dump_format="json")
        assert contains_projection(envelope["payload"], backend_payload)
        assert contains_projection(envelope["payload"], target_payload)


def test_graph_mode_payload_merge_is_recursive_and_fails_closed() -> None:
    merged = _merge_mode_payloads(
        {"mentions": [{"spans": [{"backend": "source"}]}]},
        {"mentions": [{"spans": [{"llm": "prompt"}]}]},
    )
    assert merged == {
        "mentions": [{"spans": [{"backend": "source", "llm": "prompt"}]}]
    }
    with pytest.raises(GraphModelAdapterError, match="conflict at a shared value"):
        _merge_mode_payloads({"value": 1}, {"value": 2})
    with pytest.raises(GraphModelAdapterError, match="incompatible lists"):
        _merge_mode_payloads({"items": [1]}, {"items": []})


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


@pytest.mark.parametrize(
    ("model_class", "method_name", "backend_method"),
    [
        ("PureChromaNode", "add_pure_node", "node_add"),
        ("PureChromaEdge", "add_pure_edge", "edge_add"),
    ],
)
@pytest.mark.parametrize("contract_mode", ["python", "shadow"])
def test_pure_graph_writes_honor_contract_mode_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    model_class: str,
    method_name: str,
    backend_method: str,
    contract_mode: str,
) -> None:
    from types import SimpleNamespace

    from kogwistar.engine_core.subsystems.write import WriteSubsystem

    model_type = getattr(engine_models, model_class)
    payload: dict[str, object] = {
        "id": f"{model_class}:test",
        "label": "pure graph fixture",
        "type": "entity" if model_class.endswith("Node") else "relationship",
        "summary": "pure graph fixture summary",
        "embedding": [0.0],
    }
    if model_class.endswith("Edge"):
        payload.update(
            {
                "relation": "related_to",
                "source_ids": ["source"],
                "target_ids": ["target"],
                "source_edge_ids": None,
                "target_edge_ids": None,
            }
        )
    entity = model_type.model_validate(payload)
    native_envelopes: list[GraphModelEnvelopeDict] = []
    mutation_kwargs: list[dict[str, object]] = []
    backend_calls: list[dict[str, object]] = []

    def capture_envelope(envelope: object) -> None:
        if not isinstance(envelope, dict):
            raise AssertionError("Rust validator received a non-object envelope")
        native_envelopes.append(cast(GraphModelEnvelopeDict, envelope))

    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", contract_mode)
    monkeypatch.setattr(
        "kogwistar._rust_bridge.validate_graph_model_family",
        capture_envelope,
    )
    backend = SimpleNamespace(
        **{
            backend_method: lambda **kwargs: backend_calls.append(kwargs),
        }
    )
    engine = SimpleNamespace(
        meta_sqlite=object(),
        namespace="test",
        kg_graph_type="knowledge",
        backend=backend,
        adjudicate=SimpleNamespace(
            split_endpoints=lambda source_ids, target_ids: (
                source_ids,
                [],
                target_ids,
                [],
            )
        ),
        persist=SimpleNamespace(assert_endpoints_exist=lambda _edge: None),
        pre_add_pure_edge_hooks=[],
    )
    writer = WriteSubsystem.__new__(WriteSubsystem)
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    writer._e = cast(GraphKnowledgeEngine, engine)
    monkeypatch.setattr(
        writer, "_rust_postgres_projection_upsert", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        writer,
        "_rust_postgres_add",
        lambda **kwargs: mutation_kwargs.append(kwargs) or False,
    )
    if model_class.endswith("Edge"):
        monkeypatch.setattr(
            writer,
            "enrich_edge_meta",
            lambda edge: {"namespace": "test"},
        )

    getattr(writer, method_name)(entity)

    expected_payload = json.loads(
        json.dumps(entity.model_dump(field_mode="backend", exclude=["embedding"]))
    )
    assert mutation_kwargs[0]["payload"] == expected_payload
    if contract_mode == "python":
        assert native_envelopes == []
        assert mutation_kwargs[0]["model_envelope"] is None
    else:
        assert len(native_envelopes) == 1
        assert native_envelopes[0]["model_class"] == (
            "pure_chroma_node" if model_class.endswith("Node") else "pure_chroma_edge"
        )
        assert mutation_kwargs[0]["model_envelope"] == native_envelopes[0]
        assert mutation_kwargs[0]["payload"] == native_envelopes[0]["payload"]
        assert "embedding" not in native_envelopes[0]["payload"]
        assert native_envelopes[0]["payload"] == expected_payload
    assert len(backend_calls) == 1


@pytest.mark.parametrize(
    ("base_name", "method_name"),
    [
        ("PureChromaNode", "add_pure_node"),
        ("PureChromaEdge", "add_pure_edge"),
    ],
)
def test_unmapped_pure_graph_subtype_fails_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    base_name: str,
    method_name: str,
) -> None:
    from types import SimpleNamespace

    from kogwistar.engine_core.graph_contracts import GraphModelAdapterError
    from kogwistar.engine_core.subsystems.write import WriteSubsystem

    base_type = getattr(engine_models, base_name)
    subtype = type(f"Unmapped{base_name}", (base_type,), {})
    payload: dict[str, object] = {
        "id": f"unmapped:{base_name}",
        "label": "unmapped subtype",
        "type": "entity" if base_name.endswith("Node") else "relationship",
        "summary": "must be rejected before persistence",
        "embedding": [0.0],
    }
    if base_name.endswith("Edge"):
        payload.update(
            {
                "relation": "related_to",
                "source_ids": ["source"],
                "target_ids": ["target"],
                "source_edge_ids": None,
                "target_edge_ids": None,
            }
        )
    entity = subtype.model_validate(payload)
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", "shadow")
    backend = SimpleNamespace(
        node_add=lambda **_kwargs: pytest.fail("unmapped node subtype was persisted"),
        edge_add=lambda **_kwargs: pytest.fail("unmapped edge subtype was persisted"),
    )
    engine = SimpleNamespace(
        meta_sqlite=object(),
        namespace="test",
        kg_graph_type="knowledge",
        backend=backend,
        adjudicate=SimpleNamespace(
            split_endpoints=lambda source_ids, target_ids: (
                source_ids,
                [],
                target_ids,
                [],
            )
        ),
        persist=SimpleNamespace(assert_endpoints_exist=lambda _edge: None),
        pre_add_pure_edge_hooks=[],
    )
    writer = WriteSubsystem.__new__(WriteSubsystem)
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    writer._e = cast(GraphKnowledgeEngine, engine)
    if base_name.endswith("Edge"):
        monkeypatch.setattr(
            writer,
            "enrich_edge_meta",
            lambda edge: {"namespace": "test"},
        )

    with pytest.raises(GraphModelAdapterError, match="unsupported graph model class"):
        getattr(writer, method_name)(entity)


@pytest.mark.requires_rust
@pytest.mark.parametrize("model_class", ["PureChromaNode", "PureChromaEdge"])
@pytest.mark.parametrize("contract_mode", ["shadow", "rust"])
def test_rust_family_rejection_precedes_embedding_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
    model_class: str,
    contract_mode: str,
) -> None:
    """The selected native contract validator must reject before write effects."""
    from types import SimpleNamespace

    from kogwistar.engine_core.subsystems.write import WriteSubsystem

    pytest.importorskip("kogwistar._rust")
    is_edge = model_class.endswith("Edge")
    payload: dict[str, object] = {
        "id": f"invalid:{model_class}",
        "label": " ",
        "type": "relationship" if is_edge else "entity",
        "summary": "a valid summary",
    }
    if is_edge:
        payload.update(
            {
                "relation": "related_to",
                "source_ids": ["source"],
                "target_ids": ["target"],
                "source_edge_ids": None,
                "target_edge_ids": None,
            }
        )
    entity = getattr(engine_models, model_class).model_validate(payload)
    monkeypatch.setenv("KOGWISTAR_IMPL_CONTRACTS", contract_mode)
    backend = SimpleNamespace(
        node_add=lambda **_kwargs: pytest.fail("invalid node reached persistence"),
        edge_add=lambda **_kwargs: pytest.fail("invalid edge reached persistence"),
    )
    engine = SimpleNamespace(
        meta_sqlite=object(),
        namespace="test",
        kg_graph_type="knowledge",
        backend=backend,
        adjudicate=SimpleNamespace(
            split_endpoints=lambda source_ids, target_ids: (
                source_ids,
                [],
                target_ids,
                [],
            )
        ),
        persist=SimpleNamespace(assert_endpoints_exist=lambda _edge: None),
        pre_add_pure_edge_hooks=[],
    )
    writer = WriteSubsystem.__new__(WriteSubsystem)
    from kogwistar.engine_core.engine import GraphKnowledgeEngine

    writer._e = cast(GraphKnowledgeEngine, engine)
    with pytest.raises(RustParityError, match="Rust graph-model contract validation failed"):
        getattr(writer, "add_pure_edge" if is_edge else "add_pure_node")(entity)


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


def test_supported_edge_families_retain_node_lifecycle_capabilities() -> None:
    edge_types = (
        acl_models.ACLEdge,
        conversation_models.ConversationEdge,
        runtime_models.WorkflowEdge,
        runtime_models.WorkflowRuntimeEdge,
        engine_models.PureChromaEdge,
        engine_models.LLMEdge,
        engine_models.LLMEdgeExtraction,
        engine_models.FlattenedLLMEdge,
        engine_models.AssocFlattenedLLMEdge,
    )

    for edge_type in edge_types:
        assert issubclass(edge_type, EdgeMixin), edge_type.__name__

    persistent_edge_types = (
        Edge,
        acl_models.ACLEdge,
        conversation_models.ConversationEdge,
        runtime_models.WorkflowEdge,
        runtime_models.WorkflowRuntimeEdge,
    )
    for edge_type in persistent_edge_types:
        assert issubclass(edge_type, TombstoneMixin), edge_type.__name__
        assert issubclass(edge_type, LevelAwareMixin), edge_type.__name__

    assert not issubclass(engine_models.DocNode, TombstoneMixin)
    assert not issubclass(engine_models.DocNode, LevelAwareMixin)

