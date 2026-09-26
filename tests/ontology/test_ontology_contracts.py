from __future__ import annotations

import json
from pathlib import Path

import pytest

from kogwistar.agent import CatalogStore, OntologyCatalogProvider
from kogwistar.cli import main
from kogwistar.ontology import (
    OntologyClassDescriptor,
    OntologyCompositionError,
    OntologyEdgeRole,
    OntologyEdgeShapeDescriptor,
    OntologyImport,
    OntologyPackage,
    OntologyPropertyDescriptor,
    OntologyRelationDescriptor,
    OntologyValidationError,
    compose_ontology_packages,
    ontology_package_json_schema,
    validate_edge_roles,
    validate_payload,
)


pytestmark = [pytest.mark.ci, pytest.mark.core]


def _package(*, ontology_id: str = "communication", imports=()) -> OntologyPackage:
    descriptors = [
        OntologyPropertyDescriptor(
            descriptor_id="subject",
            name="Subject",
            value_kind="string",
            min_count=1,
        ),
        OntologyPropertyDescriptor(
            descriptor_id="body",
            name="Body",
            value_kind="string",
            min_count=1,
        ),
        OntologyClassDescriptor(
            descriptor_id="Message",
            name="Message",
            property_ids=("subject", "body"),
        ),
        OntologyClassDescriptor(descriptor_id="Person", name="Person"),
        OntologyRelationDescriptor(
            descriptor_id="exchange",
            name="Exchange",
            source_class_ids=("Message",),
            target_class_ids=("Person",),
        ),
        OntologyEdgeShapeDescriptor(
            descriptor_id="message_exchange",
            name="Message exchange",
            relation_id="exchange",
            roles=(
                OntologyEdgeRole(
                    role_id="message",
                    target_kind="node",
                    allowed_class_ids=("Message",),
                ),
                OntologyEdgeRole(
                    role_id="recipient",
                    target_kind="node",
                    allowed_class_ids=("Person",),
                    min_count=1,
                    max_count=None,
                ),
            ),
        ),
    ]
    return OntologyPackage.create(
        ontology_id=ontology_id,
        version="1.0.0",
        title="Communication",
        imports=imports,
        descriptors=descriptors,
    )


def test_package_digest_is_stable_and_json_round_trips() -> None:
    package = _package()
    payload = json.loads(package.to_json(indent=None))
    reordered = {
        "descriptors": list(reversed(payload["descriptors"])),
        "manifest": payload["manifest"],
    }

    loaded = OntologyPackage.model_validate(reordered)

    assert loaded.identity == package.identity
    assert loaded.content_digest() == package.manifest.content_sha256


def test_bad_digest_is_rejected() -> None:
    package = _package()
    payload = package.model_dump(mode="json")
    payload["manifest"]["content_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="content_sha256 mismatch"):
        OntologyPackage.model_validate(payload)


def test_composition_is_order_independent_and_validates_payload_and_roles() -> None:
    package = _package()
    first = compose_ontology_packages([package])
    second = compose_ontology_packages([package], root_ontology_ids=("communication",))

    assert first.composition_sha256 == second.composition_sha256
    assert first.get("communication:Message") is not None
    validate_payload(
        first,
        class_id="communication:Message",
        payload={"subject": "Status", "body": "Ready"},
    )
    validate_edge_roles(
        first,
        shape_id="communication:message_exchange",
        roles={
            "message": [{"target_kind": "node", "class_id": "communication:Message"}],
            "recipient": [
                {"target_kind": "node", "class_id": "communication:Person"},
                {"target_kind": "node", "class_id": "communication:Person"},
            ],
        },
    )


def test_validation_rejects_missing_required_payload_and_bad_role() -> None:
    view = compose_ontology_packages([_package()])

    with pytest.raises(OntologyValidationError, match="required property"):
        validate_payload(view, class_id="communication:Message", payload={"subject": "Only"})
    with pytest.raises(OntologyValidationError, match="invalid target kind"):
        validate_edge_roles(
            view,
            shape_id="communication:message_exchange",
            roles={
                "message": [{"target_kind": "edge", "class_id": "communication:Message"}],
                "recipient": [{"target_kind": "node", "class_id": "communication:Person"}],
            },
        )


def test_edge_roles_can_bind_existing_edges_without_creating_graph_entities() -> None:
    base = _package()
    edge_shape = OntologyEdgeShapeDescriptor(
        descriptor_id="exchange_link",
        name="Exchange link",
        roles=(
            OntologyEdgeRole(
                role_id="related_exchange",
                target_kind="edge",
                allowed_class_ids=("message_exchange",),
            ),
        ),
    )
    package = OntologyPackage.create(
        ontology_id=base.identity.ontology_id,
        version=base.identity.version,
        title=base.manifest.title,
        descriptors=[*base.descriptors, edge_shape],
    )
    view = compose_ontology_packages([package])

    validate_edge_roles(
        view,
        shape_id="communication:exchange_link",
        roles={
            "related_exchange": [
                {
                    "target_kind": "edge",
                    "class_id": "communication:message_exchange",
                }
            ]
        },
    )


def test_exact_imports_are_required_and_shared_dependencies_deduplicate() -> None:
    dependency = _package(ontology_id="base")
    dependent = OntologyPackage.create(
        ontology_id="email",
        version="1.0.0",
        title="Email",
        imports=[
            OntologyImport(
                ontology_id=dependency.identity.ontology_id,
                version=dependency.identity.version,
                content_sha256=dependency.identity.content_sha256,
            )
        ],
        descriptors=[OntologyClassDescriptor(descriptor_id="EmailMessage", name="Email message")],
    )

    view = compose_ontology_packages([dependent, dependency], root_ontology_ids=("email",))

    assert [item.ontology_id for item in view.package_identities] == ["base", "email"]
    with pytest.raises(OntologyCompositionError, match="missing exact ontology import"):
        compose_ontology_packages([dependent], root_ontology_ids=("email",))


def test_catalog_acl_filters_before_semantic_ranking() -> None:
    package = _package()
    seen: list[str] = []

    def ranker(_query: str, entries) -> dict[str, float]:
        seen.extend(item.name for item in entries)
        return {item.logical_id: 1.0 for item in entries}

    catalog = CatalogStore(
        acl_enabled=True,
        acl_checker=lambda entry, _principal: entry.name != "Body",
        semantic_ranker=ranker,
    )
    OntologyCatalogProvider(package).ingest(catalog)
    catalog.search("body", principal="user", mode="semantic")

    assert "Body" not in seen


def test_cli_schema_and_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    package_path = tmp_path / "communication.json"
    package_path.write_text(_package().to_json(), encoding="utf-8")

    assert main(["ontology", "schema"]) == 0
    assert '"$defs"' in capsys.readouterr().out
    assert main(["ontology", "validate", str(package_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["package"]["ontology_id"] == "communication"

    dependency = OntologyPackage.create(
        ontology_id="base",
        version="1.0.0",
        title="Base",
        descriptors=[OntologyClassDescriptor(descriptor_id="Record", name="Record")],
    )
    dependent = OntologyPackage.create(
        ontology_id="email",
        version="1.0.0",
        title="Email",
        imports=[
            OntologyImport(
                ontology_id=dependency.identity.ontology_id,
                version=dependency.identity.version,
                content_sha256=dependency.identity.content_sha256,
            )
        ],
        descriptors=[OntologyClassDescriptor(descriptor_id="Message", name="Message")],
    )
    dependency_path = tmp_path / "base.json"
    dependent_path = tmp_path / "email.json"
    dependency_path.write_text(dependency.to_json(), encoding="utf-8")
    dependent_path.write_text(dependent.to_json(), encoding="utf-8")
    assert (
        main(
            [
                "ontology",
                "validate",
                str(dependent_path),
                "--compose-with",
                str(dependency_path),
                "--json",
            ]
        )
        == 0
    )
    composed = json.loads(capsys.readouterr().out)
    assert composed["composition"]["package_count"] == 2


def test_checked_in_schema_matches_pydantic_source_of_truth() -> None:
    schema_path = Path(__file__).parents[2] / "contracts/ontology/ontology_package.v1.schema.json"
    assert json.loads(schema_path.read_text(encoding="utf-8")) == ontology_package_json_schema()


def test_cli_reports_shape_and_composition_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert main(["ontology", "validate", str(malformed), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False

    invalid_digest = tmp_path / "invalid-digest.json"
    invalid_payload = _package().model_dump(mode="json")
    invalid_payload["manifest"]["content_sha256"] = "f" * 64
    invalid_digest.write_text(json.dumps(invalid_payload), encoding="utf-8")
    assert main(["ontology", "validate", str(invalid_digest), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["valid"] is False
