"""Declarative ontology package contracts.

Ontology packages are data-only descriptions.  They do not grant execution,
authorization, or graph mutation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Literal, TypeAlias
from typing_extensions import TypeAliasType

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_QUALIFIED_IDENTIFIER = re.compile(
    r"^[A-Za-z][A-Za-z0-9_.-]*:[A-Za-z][A-Za-z0-9_.-]*$"
)
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"],
)

DescriptorKind = Literal[
    "ontology_class",
    "ontology_property",
    "ontology_relation",
    "ontology_edge_shape",
]
ValueKind = Literal[
    "string",
    "integer",
    "number",
    "boolean",
    "timestamp",
    "logical_ref",
    "json",
]
TargetKind = Literal["node", "edge"]


def _validate_identifier(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{field_name} must match [A-Za-z][A-Za-z0-9_.-]*: {value!r}"
        )
    return normalized


def _validate_reference(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not (_IDENTIFIER.fullmatch(normalized) or _QUALIFIED_IDENTIFIER.fullmatch(normalized)):
        raise ValueError(
            f"{field_name} must be a local or qualified descriptor ID: {value!r}"
        )
    return normalized


def _validate_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, path=f"{path}[]") for item in value]
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return value
    raise ValueError(f"{path} must contain only JSON primitives, lists, and objects")


def qualify_descriptor_ref(reference: str, ontology_id: str) -> str:
    """Resolve a package-local reference without changing qualified refs."""

    value = _validate_reference(reference, field_name="descriptor reference")
    if ":" in value:
        return value
    return f"{ontology_id}:{value}"


class OntologyPackageIdentity(BaseModel):
    """Immutable identity of one exact ontology package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ontology_id: str
    version: str
    content_sha256: str = Field(pattern=_SHA256.pattern)

    @field_validator("ontology_id")
    @classmethod
    def _ontology_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="ontology_id")

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        normalized = str(value).strip()
        if not _SEMVER.fullmatch(normalized):
            raise ValueError("version must be a semantic version such as 1.0.0")
        return normalized

    @property
    def qualified_id(self) -> str:
        return f"{self.ontology_id}@{self.version}#{self.content_sha256}"


class OntologyImport(OntologyPackageIdentity):
    """An exact dependency pin declared by a package."""


class OntologyPackageManifest(BaseModel):
    """Data-only package metadata and exact dependency pins."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    ontology_id: str
    version: str
    title: str = Field(min_length=1, max_length=240)
    summary: str = Field(default="", max_length=4000)
    content_sha256: str = Field(pattern=_SHA256.pattern)
    imports: tuple[OntologyImport, ...] = ()
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("ontology_id")
    @classmethod
    def _ontology_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="ontology_id")

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        normalized = str(value).strip()
        if not _SEMVER.fullmatch(normalized):
            raise ValueError("version must be a semantic version such as 1.0.0")
        return normalized

    @field_validator("extensions")
    @classmethod
    def _extension_namespaces(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        for key in value:
            if "/" not in key or key.startswith("/") or key.endswith("/"):
                raise ValueError("extension keys must be namespaced, for example vendor/key")
            _validate_json_value(value[key], path=f"extensions.{key}")
        return value

    @model_validator(mode="after")
    def _unique_imports(self) -> "OntologyPackageManifest":
        identities = [item.qualified_id for item in self.imports]
        if len(identities) != len(set(identities)):
            raise ValueError("imports must not contain duplicate package identities")
        return self

    @property
    def identity(self) -> OntologyPackageIdentity:
        return OntologyPackageIdentity(
            ontology_id=self.ontology_id,
            version=self.version,
            content_sha256=self.content_sha256,
        )


class OntologyDescriptorBase(BaseModel):
    """Common metadata for one package-local descriptor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor_id: str
    name: str = Field(min_length=1, max_length=240)
    summary: str = Field(default="", max_length=4000)
    aliases: tuple[str, ...] = ()
    extensions: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("descriptor_id")
    @classmethod
    def _descriptor_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="descriptor_id")

    @field_validator("extensions")
    @classmethod
    def _extension_namespaces(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        for key in value:
            if "/" not in key or key.startswith("/") or key.endswith("/"):
                raise ValueError("extension keys must be namespaced, for example vendor/key")
            _validate_json_value(value[key], path=f"extensions.{key}")
        return value


class OntologyPropertyDescriptor(OntologyDescriptorBase):
    kind: Literal["ontology_property"] = "ontology_property"
    value_kind: ValueKind
    min_count: int = Field(default=0, ge=0)
    max_count: int | None = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _valid_cardinality(self) -> "OntologyPropertyDescriptor":
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count")
        return self


class OntologyClassDescriptor(OntologyDescriptorBase):
    kind: Literal["ontology_class"] = "ontology_class"
    property_ids: tuple[str, ...] = ()
    allow_additional_properties: bool = False

    @field_validator("property_ids")
    @classmethod
    def _property_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _validate_reference(value, field_name="property_ids item") for value in values
        )


class OntologyRelationDescriptor(OntologyDescriptorBase):
    kind: Literal["ontology_relation"] = "ontology_relation"
    source_class_ids: tuple[str, ...] = Field(min_length=1)
    target_class_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("source_class_ids", "target_class_ids")
    @classmethod
    def _class_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _validate_reference(value, field_name="class reference") for value in values
        )


class OntologyEdgeRole(BaseModel):
    """One named role in an existing Kogwistar multi-endpoint edge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role_id: str
    target_kind: TargetKind
    allowed_class_ids: tuple[str, ...] = ()
    min_count: int = Field(default=1, ge=0)
    max_count: int | None = Field(default=1, ge=1)

    @field_validator("role_id")
    @classmethod
    def _role_id(cls, value: str) -> str:
        return _validate_identifier(value, field_name="role_id")

    @field_validator("allowed_class_ids")
    @classmethod
    def _allowed_class_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            _validate_reference(value, field_name="allowed_class_ids item")
            for value in values
        )

    @model_validator(mode="after")
    def _valid_cardinality(self) -> "OntologyEdgeRole":
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count")
        return self


class OntologyEdgeShapeDescriptor(OntologyDescriptorBase):
    kind: Literal["ontology_edge_shape"] = "ontology_edge_shape"
    relation_id: str | None = None
    roles: tuple[OntologyEdgeRole, ...] = Field(min_length=1)

    @field_validator("relation_id")
    @classmethod
    def _relation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_reference(value, field_name="relation_id")

    @model_validator(mode="after")
    def _unique_roles(self) -> "OntologyEdgeShapeDescriptor":
        role_ids = [role.role_id for role in self.roles]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("edge shape roles must have unique role_id values")
        return self


OntologyDescriptor: TypeAlias = Annotated[
    OntologyClassDescriptor
    | OntologyPropertyDescriptor
    | OntologyRelationDescriptor
    | OntologyEdgeShapeDescriptor,
    Field(discriminator="kind"),
]


class OntologyPackage(BaseModel):
    """Immutable package payload with a self-verifying content digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: OntologyPackageManifest
    descriptors: tuple[OntologyDescriptor, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_descriptors_and_digest(self) -> "OntologyPackage":
        descriptor_ids = [descriptor.descriptor_id for descriptor in self.descriptors]
        if len(descriptor_ids) != len(set(descriptor_ids)):
            raise ValueError("descriptor_id values must be unique within a package")
        expected = self.content_digest()
        if expected != self.manifest.content_sha256:
            raise ValueError(
                "content_sha256 mismatch: "
                f"expected {expected}, got {self.manifest.content_sha256}"
            )
        return self

    @property
    def identity(self) -> OntologyPackageIdentity:
        return self.manifest.identity

    def canonical_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        manifest = dict(payload["manifest"])
        manifest.pop("content_sha256", None)
        manifest["imports"] = sorted(
            manifest.get("imports", []),
            key=lambda item: (
                item["ontology_id"],
                item["version"],
                item["content_sha256"],
            ),
        )
        payload["manifest"] = manifest
        payload["descriptors"] = sorted(
            payload["descriptors"],
            key=lambda descriptor: descriptor["descriptor_id"],
        )
        return payload

    def content_digest(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent else (",", ":"),
        )

    @classmethod
    def create(
        cls,
        *,
        ontology_id: str,
        version: str,
        title: str,
        descriptors: tuple[OntologyDescriptor, ...] | list[OntologyDescriptor],
        summary: str = "",
        imports: tuple[OntologyImport, ...] | list[OntologyImport] = (),
        extensions: dict[str, JsonValue] | None = None,
    ) -> "OntologyPackage":
        """Construct a package and calculate its canonical digest."""

        manifest = OntologyPackageManifest.model_construct(
            schema_version=1,
            ontology_id=ontology_id,
            version=version,
            title=title,
            summary=summary,
            content_sha256="0" * 64,
            imports=tuple(imports),
            extensions=dict(extensions or {}),
        )
        draft = cls.model_construct(manifest=manifest, descriptors=tuple(descriptors))
        digest = draft.content_digest()
        final_manifest = manifest.model_copy(update={"content_sha256": digest})
        return cls.model_validate(
            {"manifest": final_manifest, "descriptors": tuple(descriptors)}
        )


__all__ = [
    "JsonPrimitive",
    "JsonValue",
    "OntologyClassDescriptor",
    "OntologyDescriptor",
    "OntologyEdgeRole",
    "OntologyEdgeShapeDescriptor",
    "OntologyImport",
    "OntologyPackage",
    "OntologyPackageIdentity",
    "OntologyPackageManifest",
    "OntologyPropertyDescriptor",
    "OntologyRelationDescriptor",
    "TargetKind",
    "ValueKind",
    "qualify_descriptor_ref",
]
