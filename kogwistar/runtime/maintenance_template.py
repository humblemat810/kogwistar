from __future__ import annotations

"""Compatibility re-exports for maintenance templates."""

from kogwistar.maintenance.models import MaintenanceTemplateResult
from kogwistar.maintenance.template import run_grouped_maintenance_template

__all__ = [
    "MaintenanceTemplateResult",
    "run_grouped_maintenance_template",
]
