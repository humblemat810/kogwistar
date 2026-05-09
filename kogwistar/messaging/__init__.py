from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kogwistar.messaging.models import (
        LaneMessageProjectionRepairResult,
        LaneMessageSendResult,
        ProjectedLaneMessageRow,
    )
    from kogwistar.messaging.service import LaneMessagingService

__all__ = [
    "LaneMessagingService",
    "LaneMessageProjectionRepairResult",
    "LaneMessageSendResult",
    "ProjectedLaneMessageRow",
]

_EXPORTS = {
    "LaneMessagingService": "kogwistar.messaging.service",
    "LaneMessageProjectionRepairResult": "kogwistar.messaging.models",
    "LaneMessageSendResult": "kogwistar.messaging.models",
    "ProjectedLaneMessageRow": "kogwistar.messaging.models",
}


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
