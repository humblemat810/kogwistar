"""Backend-neutral metadata naming compatibility."""

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine

pytestmark = [pytest.mark.core, pytest.mark.ci]


def test_metadata_is_backend_neutral_alias_for_existing_store_slot():
    engine = object.__new__(GraphKnowledgeEngine)
    first = object()
    second = object()

    engine.meta_sqlite = first
    assert engine.metadata is first

    engine.metadata = second
    assert engine.meta_sqlite is second
