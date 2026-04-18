"""Maintenance-domain helpers.

Maintenance is an application/system semantic domain layered on top of graph
kinds. This package hosts reusable maintenance primitives without folding them
into workflow runtime internals.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.maintenance.models import (
        GroupedArtifactWriteResult,
        MaintenanceTemplateResult,
        VersionedArtifactWriteResult,
    )
    from kogwistar.maintenance.artifacts import (
        write_versioned_artifact,
    )
    from kogwistar.maintenance.grouped_artifacts import (
        write_grouped_versioned_artifacts,
    )
    from kogwistar.maintenance.template import (
        run_grouped_maintenance_template,
    )

__all__ = [
    "VersionedArtifactWriteResult",
    "GroupedArtifactWriteResult",
    "MaintenanceTemplateResult",
    "write_grouped_versioned_artifacts",
    "write_versioned_artifact",
    "run_grouped_maintenance_template",
]

_EXPORTS = {
    "VersionedArtifactWriteResult": "kogwistar.maintenance.models",
    "write_versioned_artifact": "kogwistar.maintenance.artifacts",
    "GroupedArtifactWriteResult": "kogwistar.maintenance.models",
    "write_grouped_versioned_artifacts": "kogwistar.maintenance.grouped_artifacts",
    "MaintenanceTemplateResult": "kogwistar.maintenance.models",
    "run_grouped_maintenance_template": "kogwistar.maintenance.template",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
