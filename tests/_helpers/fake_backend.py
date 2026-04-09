from __future__ import annotations

"""Backward-compatible alias for the runtime in-memory backend.

Tests historically imported ``build_fake_backend`` from this module. The
implementation now lives in :mod:`kogwistar.engine_core.in_memory_backend` as a
first-class volatile backend, and this module simply re-exports it.
"""

from kogwistar.engine_core.in_memory_backend import (
    InMemoryBackend,
    build_in_memory_backend,
)

build_fake_backend = build_in_memory_backend

__all__ = ["InMemoryBackend", "build_fake_backend", "build_in_memory_backend"]
