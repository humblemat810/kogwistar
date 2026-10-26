"""Backend-neutral metadata naming compatibility."""

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
