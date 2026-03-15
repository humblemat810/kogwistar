from __future__ import annotations

import os
from pathlib import Path


def joblib_cache_root() -> Path:
    return Path(os.getenv("GKE_JOBLIB_CACHE_DIR", ".joblib")).absolute()


def joblib_cache_path(*parts: str) -> Path:
    return joblib_cache_root().joinpath(*parts)
