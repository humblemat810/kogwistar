from __future__ import annotations

import inspect

import pytest

from kogwistar.engine_core import AsyncNamedProjectionStore


pytestmark = [pytest.mark.ci, pytest.mark.core, pytest.mark.unit]


class _AsyncMetadata:
    async def get_named_projection(self, namespace: str, key: str):
        return {"namespace": namespace, "key": key}

    async def compare_and_swap_named_projection(
        self, namespace: str, key: str, payload: dict, **values
    ) -> bool:
        return bool(namespace and key and payload is not None and values is not None)

    async def compare_and_swap_named_projections(self, updates: list[dict]) -> bool:
        return bool(updates)

    async def list_named_projections(self, namespace: str) -> list[dict]:
        return [{"namespace": namespace}]


@pytest.mark.asyncio
async def test_async_named_projection_store_is_native_and_batch_capable() -> None:
    store = _AsyncMetadata()

    assert isinstance(store, AsyncNamedProjectionStore)
    assert all(
        inspect.iscoroutinefunction(getattr(store, name))
        for name in (
            "get_named_projection",
            "compare_and_swap_named_projection",
            "compare_and_swap_named_projections",
            "list_named_projections",
        )
    )
    assert await store.get_named_projection("ns", "key") == {
        "namespace": "ns",
        "key": "key",
    }
    assert await store.compare_and_swap_named_projections([{"key": "key"}])
