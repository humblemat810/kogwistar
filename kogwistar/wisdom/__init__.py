"""Wisdom-domain helpers.

This package hosts reusable helpers for distilled, reusable lessons. It should
remain distinct from workflow runtime execution internals.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.wisdom.models import ExecutionWisdomTemplateResult
    from kogwistar.wisdom.template import write_execution_wisdom_artifacts

__all__ = [
    "ExecutionWisdomTemplateResult",
    "write_execution_wisdom_artifacts",
]

_EXPORTS = {
    "ExecutionWisdomTemplateResult": "kogwistar.wisdom.models",
    "write_execution_wisdom_artifacts": "kogwistar.wisdom.template",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
