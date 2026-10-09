from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from kogwistar.utils.cache_backend import Memory


def test_diskcache_ignored_positional_dependency_is_not_pickled(
    tmp_path: Path,
) -> None:
    pytest.importorskip("diskcache")
    calls = 0

    def invoke(dependency: Callable[[], str], key: str) -> str:
        nonlocal calls
        calls += 1
        return f"{key}:{dependency()}"

    memory = Memory(location=tmp_path / "cache", backend="diskcache")
    try:
        cached = cast(
            Callable[[Callable[[], str], str], str],
            memory.cache(invoke, ignore=["dependency"]),
        )
        assert cached(lambda: "first", "same") == "same:first"
        assert cached(lambda: "second", "same") == "same:first"
        assert calls == 1
    finally:
        memory.close()
