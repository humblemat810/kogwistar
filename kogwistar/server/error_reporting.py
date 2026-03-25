from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import HTTPException


def dev_mode_enabled() -> bool:
    return str(os.getenv("AUTH_MODE", "")).strip().lower() == "dev"


def internal_http_error(exc: Exception) -> HTTPException:
    if dev_mode_enabled():
        return HTTPException(
            status_code=500,
            detail={
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "traceback": traceback.format_exception(
                    type(exc), exc, exc.__traceback__
                ),
            },
        )
    return HTTPException(status_code=500, detail=str(exc))


def detail_payload(exc: Exception) -> dict[str, Any]:
    return {
        "error": str(exc),
        "error_type": exc.__class__.__name__,
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
    }
