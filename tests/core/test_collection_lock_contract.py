from __future__ import annotations

import threading

import pytest

from kogwistar.engine_core.engine import GraphKnowledgeEngine


@pytest.mark.ci
@pytest.mark.core
def test_collection_lock_accessor_returns_lock_and_rejects_unknown_name() -> None:
    engine = object.__new__(GraphKnowledgeEngine)
    lock = threading.Lock()
    engine.collection_lock = {"node": lock}

    assert engine.get_collection_lock("node") is lock
    assert lock.acquire(blocking=False)
    lock.release()

    with pytest.raises(ValueError, match="edge has not implemented lock"):
        engine.get_collection_lock("edge")
