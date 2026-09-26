"""Catalog adapter for declarative ontology packages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kogwistar.ontology import OntologyPackage

from .catalog import CatalogEntry, CatalogStore


class OntologyCatalogProvider:
    """Read-only discovery provider backed by one immutable ontology package."""

    def __init__(
        self,
        package: OntologyPackage,
        *,
        tenant_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self.package = package
        self.provider_id = f"ontology:{package.identity.ontology_id}"
        self.provider_version = package.identity.version
        self.tenant_id = tenant_id
        self.project_id = project_id

    def descriptors(self) -> list[Mapping[str, Any]]:
        values: list[Mapping[str, Any]] = []
        for descriptor in self.package.descriptors:
            values.append(
                {
                    "logical_id": f"{self.package.identity.ontology_id}:{descriptor.descriptor_id}",
                    "provider_local_id": descriptor.descriptor_id,
                    "kind": descriptor.kind,
                    "name": descriptor.name,
                    "summary": descriptor.summary,
                    "version": self.package.identity.version,
                    "source_fingerprint": self.package.identity.content_sha256,
                    "aliases": list(descriptor.aliases),
                    "tenant_id": self.tenant_id,
                    "project_id": self.project_id,
                    "metadata": {
                        "ontology_id": self.package.identity.ontology_id,
                        "ontology_version": self.package.identity.version,
                        "ontology_digest": self.package.identity.content_sha256,
                        "descriptor_kind": descriptor.kind,
                        "descriptor_id": descriptor.descriptor_id,
                    },
                }
            )
        return values

    def catalog_entries(self) -> tuple[CatalogEntry, ...]:
        return tuple(
            CatalogEntry.model_validate(
                {
                    **dict(item),
                    "provider_id": self.provider_id,
                    "provider_version": self.provider_version,
                }
            )
            for item in self.descriptors()
        )

    def ingest(self, catalog: CatalogStore) -> tuple[CatalogEntry, ...]:
        """Materialize ACL-scoped serving records through the existing catalog."""

        entries = self.catalog_entries()
        return catalog.ingest_descriptors(list(entries))

    def close(self) -> None:
        return None


__all__ = ["OntologyCatalogProvider"]
