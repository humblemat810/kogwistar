from __future__ import annotations

"""Compatibility re-exports for wisdom template helpers."""

from kogwistar.wisdom.models import ExecutionWisdomTemplateResult
from kogwistar.wisdom.template import write_execution_wisdom_artifacts

__all__ = [
    "ExecutionWisdomTemplateResult",
    "write_execution_wisdom_artifacts",
]
