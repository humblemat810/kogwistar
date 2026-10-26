"""Backend-neutral metadata naming compatibility."""

import re
from pathlib import Path

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine

pytestmark = [pytest.mark.core, pytest.mark.ci]


def test_metadata_is_backend_neutral_and_legacy_alias_stays_compatible():
    engine = object.__new__(GraphKnowledgeEngine)
    postgres_metadata = object()
    second = object()

    engine.metadata = postgres_metadata
    assert engine.metadata is postgres_metadata
    assert engine.meta_sqlite is postgres_metadata

    engine.metadata = second
    assert engine.meta_sqlite is second


def test_internal_engine_core_consumers_use_backend_neutral_metadata_name():
    engine_core = Path(__file__).resolve().parents[2] / "kogwistar" / "engine_core"
    legacy_access = re.compile(r"\.\s*meta_sqlite\b")
    allowed_compatibility_files = {"engine.py", "storage_backend.py"}

    offenders = [
        str(path.relative_to(engine_core.parent.parent))
        for path in engine_core.rglob("*.py")
        if path.name not in allowed_compatibility_files
        and legacy_access.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == [], (
        "internal engine code must use backend-neutral `.metadata`; "
        f"legacy `.meta_sqlite` access found in: {offenders}"
    )
