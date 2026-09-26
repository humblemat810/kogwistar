"""Deterministic composition and structural validation for ontology packages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import (
    OntologyClassDescriptor,
    OntologyDescriptor,
    OntologyEdgeShapeDescriptor,
    OntologyPackage,
    OntologyPackageIdentity,
    OntologyPropertyDescriptor,
    OntologyRelationDescriptor,
    qualify_descriptor_ref,
)


class OntologyCompositionError(ValueError):
    """Raised when exact ontology packages cannot form one safe view."""


class OntologyValidationError(ValueError):
    """Raised when a payload or edge binding violates a composed view."""


class ComposedDescriptor(BaseModel):
    """One descriptor together with its package-qualified identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualified_id: str
    descriptor: OntologyDescriptor


class ComposedOntologyView(BaseModel):
    """Immutable, rebuildable result of exact package composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    package_identities: tuple[OntologyPackageIdentity, ...]
    descriptors: tuple[ComposedDescriptor, ...]
    composition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _unique_qualified_descriptors(self) -> "ComposedOntologyView":
        qualified_ids = [item.qualified_id for item in self.descriptors]
        if len(qualified_ids) != len(set(qualified_ids)):
            raise ValueError("composed descriptors must have unique qualified IDs")
        if self.composition_digest() != self.composition_sha256:
            raise ValueError("composition_sha256 does not match the composed view")
        return self

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "package_identities": [
                item.model_dump(mode="json") for item in self.package_identities
            ],
            "descriptors": [item.model_dump(mode="json") for item in self.descriptors],
        }

    def composition_digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_payload(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def get(self, qualified_id: str) -> OntologyDescriptor | None:
        for item in self.descriptors:
            if item.qualified_id == qualified_id:
                return item.descriptor
        return None


def _canonical_view_payload(
    identities: Sequence[OntologyPackageIdentity],
    descriptors: Sequence[tuple[str, OntologyDescriptor]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "package_identities": [item.model_dump(mode="json") for item in identities],
        "descriptors": [
            {"qualified_id": qualified_id, "descriptor": descriptor.model_dump(mode="json")}
            for qualified_id, descriptor in descriptors
        ],
    }


def _qualify(package_id: str, reference: str) -> str:
    return qualify_descriptor_ref(reference, package_id)


def _validate_descriptor_references(
    *,
    package_id: str,
    descriptor: OntologyDescriptor,
    descriptors: Mapping[str, OntologyDescriptor],
) -> None:
    def require(reference: str, expected: type[BaseModel] | tuple[type[BaseModel], ...]) -> None:
        qualified = _qualify(package_id, reference)
        target = descriptors.get(qualified)
        if target is None:
            raise OntologyCompositionError(
                f"{package_id}:{descriptor.descriptor_id} references missing descriptor {qualified}"
            )
        if not isinstance(target, expected):
            expected_names = expected if isinstance(expected, tuple) else (expected,)
            names = ", ".join(item.__name__ for item in expected_names)
            raise OntologyCompositionError(
                f"{qualified} has incompatible descriptor kind; expected {names}"
            )

    if isinstance(descriptor, OntologyClassDescriptor):
        for property_id in descriptor.property_ids:
            require(property_id, OntologyPropertyDescriptor)
    elif isinstance(descriptor, OntologyRelationDescriptor):
        for class_id in (*descriptor.source_class_ids, *descriptor.target_class_ids):
            require(class_id, OntologyClassDescriptor)
    elif isinstance(descriptor, OntologyEdgeShapeDescriptor):
        if descriptor.relation_id is not None:
            require(descriptor.relation_id, OntologyRelationDescriptor)
        for role in descriptor.roles:
            for class_id in role.allowed_class_ids:
                expected = (
                    OntologyClassDescriptor
                    if role.target_kind == "node"
                    else (OntologyRelationDescriptor, OntologyEdgeShapeDescriptor)
                )
                require(class_id, expected)


def compose_ontology_packages(
    packages: Sequence[OntologyPackage],
    *,
    root_ontology_ids: Sequence[str] | None = None,
) -> ComposedOntologyView:
    """Compose exact package identities without installation-order semantics."""

    by_identity = {package.identity.qualified_id: package for package in packages}
    if len(by_identity) != len(packages):
        raise OntologyCompositionError("duplicate package identities have different payloads")
    by_ontology_id: dict[str, list[OntologyPackage]] = {}
    for package in packages:
        by_ontology_id.setdefault(package.identity.ontology_id, []).append(package)

    roots = set(root_ontology_ids or by_ontology_id)
    selected: dict[str, OntologyPackage] = {}
    visiting: set[str] = set()

    def visit(package: OntologyPackage) -> None:
        identity = package.identity.qualified_id
        if identity in selected:
            return
        if identity in visiting:
            raise OntologyCompositionError(f"ontology import cycle detected at {identity}")
        visiting.add(identity)
        for imported in package.manifest.imports:
            dependency = by_identity.get(imported.qualified_id)
            if dependency is None:
                raise OntologyCompositionError(
                    f"missing exact ontology import {imported.qualified_id}"
                )
            visit(dependency)
        visiting.remove(identity)
        selected[identity] = package

    for root_id in sorted(roots):
        candidates = by_ontology_id.get(root_id, [])
        if not candidates:
            raise OntologyCompositionError(f"missing root ontology {root_id}")
        for package in sorted(candidates, key=lambda item: item.identity.qualified_id):
            visit(package)

    indexed: dict[str, OntologyDescriptor] = {}
    qualified_pairs: list[tuple[str, OntologyDescriptor]] = []
    for identity in sorted(selected):
        package = selected[identity]
        ontology_id = package.identity.ontology_id
        for descriptor in package.descriptors:
            qualified_id = f"{ontology_id}:{descriptor.descriptor_id}"
            if qualified_id in indexed:
                raise OntologyCompositionError(f"duplicate qualified descriptor {qualified_id}")
            indexed[qualified_id] = descriptor
            qualified_pairs.append((qualified_id, descriptor))

    for identity in sorted(selected):
        package = selected[identity]
        for descriptor in package.descriptors:
            _validate_descriptor_references(
                package_id=package.identity.ontology_id,
                descriptor=descriptor,
                descriptors=indexed,
            )

    qualified_pairs.sort(key=lambda item: item[0])
    identities = tuple(selected[key].identity for key in sorted(selected))
    canonical = _canonical_view_payload(identities, qualified_pairs)
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    view = ComposedOntologyView.model_validate(
        {
            "schema_version": 1,
            "package_identities": identities,
            "descriptors": tuple(
                ComposedDescriptor(qualified_id=qualified_id, descriptor=descriptor)
                for qualified_id, descriptor in qualified_pairs
            ),
            "composition_sha256": digest,
        }
    )
    return view


def _view_index(view: ComposedOntologyView) -> dict[str, OntologyDescriptor]:
    return {item.qualified_id: item.descriptor for item in view.descriptors}


def validate_payload(
    view: ComposedOntologyView,
    *,
    class_id: str,
    payload: Mapping[str, object],
) -> None:
    """Validate a neutral payload against one composed ontology class."""

    descriptor = _view_index(view).get(class_id)
    if not isinstance(descriptor, OntologyClassDescriptor):
        raise OntologyValidationError(f"unknown ontology class: {class_id}")
    index = _view_index(view)
    properties: dict[str, OntologyPropertyDescriptor] = {}
    for property_id in descriptor.property_ids:
        qualified = _qualify(class_id.split(":", 1)[0], property_id)
        property_descriptor = index.get(qualified)
        if not isinstance(property_descriptor, OntologyPropertyDescriptor):
            raise OntologyValidationError(f"class property is unavailable: {qualified}")
        properties[qualified] = property_descriptor
    for key, value in payload.items():
        qualified = _qualify(class_id.split(":", 1)[0], str(key))
        property_descriptor = properties.get(qualified)
        if property_descriptor is None:
            if not descriptor.allow_additional_properties:
                raise OntologyValidationError(f"unexpected property for {class_id}: {key}")
            continue
        if property_descriptor.max_count == 1 and isinstance(value, list):
            raise OntologyValidationError(f"property {key} does not accept a list")
        if property_descriptor.max_count != 1 and not isinstance(value, list):
            raise OntologyValidationError(f"property {key} requires a list")
        values = value if isinstance(value, list) else [value]
        if len(values) < property_descriptor.min_count:
            raise OntologyValidationError(f"property {key} has too few values")
        if property_descriptor.max_count is not None and len(values) > property_descriptor.max_count:
            raise OntologyValidationError(f"property {key} has too many values")
        for item in values:
            if not _value_matches(property_descriptor.value_kind, item):
                raise OntologyValidationError(
                    f"property {key} expects {property_descriptor.value_kind}"
                )
    for qualified, property_descriptor in properties.items():
        short_id = qualified.split(":", 1)[1]
        if property_descriptor.min_count > 0 and short_id not in payload:
            raise OntologyValidationError(f"required property is missing: {short_id}")


def _value_matches(value_kind: str, value: object) -> bool:
    if value_kind == "string" or value_kind == "timestamp" or value_kind == "logical_ref":
        return isinstance(value, str)
    if value_kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if value_kind == "boolean":
        return isinstance(value, bool)
    return value_kind == "json"


def validate_edge_roles(
    view: ComposedOntologyView,
    *,
    shape_id: str,
    roles: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Validate neutral role bindings without changing graph edge storage."""

    descriptor = _view_index(view).get(shape_id)
    if not isinstance(descriptor, OntologyEdgeShapeDescriptor):
        raise OntologyValidationError(f"unknown ontology edge shape: {shape_id}")
    index = _view_index(view)
    declared = {role.role_id: role for role in descriptor.roles}
    if set(roles) - set(declared):
        unknown = sorted(set(roles) - set(declared))[0]
        raise OntologyValidationError(f"unknown edge role: {unknown}")
    for role_id, role in declared.items():
        values = list(roles.get(role_id, ()))
        if len(values) < role.min_count:
            raise OntologyValidationError(f"edge role {role_id} has too few endpoints")
        if role.max_count is not None and len(values) > role.max_count:
            raise OntologyValidationError(f"edge role {role_id} has too many endpoints")
        allowed = {_qualify(shape_id.split(":", 1)[0], item) for item in role.allowed_class_ids}
        for endpoint in values:
            if endpoint.get("target_kind") != role.target_kind:
                raise OntologyValidationError(f"edge role {role_id} has an invalid target kind")
            endpoint_id = endpoint.get("class_id")
            endpoint_descriptor = index.get(endpoint_id or "")
            valid_endpoint_type = (
                isinstance(endpoint_descriptor, OntologyClassDescriptor)
                if role.target_kind == "node"
                else isinstance(
                    endpoint_descriptor,
                    (OntologyRelationDescriptor, OntologyEdgeShapeDescriptor),
                )
            )
            if not valid_endpoint_type:
                raise OntologyValidationError(f"edge role {role_id} has an invalid target")
            if allowed and endpoint_id not in allowed:
                raise OntologyValidationError(f"edge role {role_id} has an invalid class")


__all__ = [
    "ComposedDescriptor",
    "ComposedOntologyView",
    "OntologyCompositionError",
    "OntologyValidationError",
    "compose_ontology_packages",
    "validate_edge_roles",
    "validate_payload",
]
