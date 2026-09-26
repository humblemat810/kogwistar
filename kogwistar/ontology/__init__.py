"""Public declarative ontology contracts for Kogwistar."""

from .composition import (
    ComposedOntologyView,
    OntologyCompositionError,
    OntologyValidationError,
    compose_ontology_packages,
    validate_edge_roles,
    validate_payload,
)
from .models import (
    JsonPrimitive,
    JsonValue,
    OntologyClassDescriptor,
    OntologyDescriptor,
    OntologyEdgeRole,
    OntologyEdgeShapeDescriptor,
    OntologyImport,
    OntologyPackage,
    OntologyPackageIdentity,
    OntologyPackageManifest,
    OntologyPropertyDescriptor,
    OntologyRelationDescriptor,
    TargetKind,
    ValueKind,
    qualify_descriptor_ref,
)


def ontology_package_json_schema() -> dict[str, object]:
    """Return the versioned JSON Schema for one ontology package bundle."""

    schema = OntologyPackage.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


__all__ = [
    "ComposedOntologyView",
    "JsonPrimitive",
    "JsonValue",
    "OntologyClassDescriptor",
    "OntologyCompositionError",
    "OntologyDescriptor",
    "OntologyEdgeRole",
    "OntologyEdgeShapeDescriptor",
    "OntologyImport",
    "OntologyPackage",
    "OntologyPackageIdentity",
    "OntologyPackageManifest",
    "OntologyPropertyDescriptor",
    "OntologyRelationDescriptor",
    "OntologyValidationError",
    "TargetKind",
    "ValueKind",
    "compose_ontology_packages",
    "ontology_package_json_schema",
    "qualify_descriptor_ref",
    "validate_edge_roles",
    "validate_payload",
]
