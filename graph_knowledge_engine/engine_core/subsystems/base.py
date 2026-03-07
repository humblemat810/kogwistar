from __future__ import annotations

from typing import Any


class NamespaceProxy:
    """Thin namespace proxy to route namespaced API calls to engine implementations.

    Resolution order:
    1) mapped alias in `aliases`
    2) bound `_impl_<method>` if present (after shim installation)
    3) method directly on engine
    """

    def __init__(self, engine: Any, *, aliases: dict[str, str] | None = None) -> None:
        self._e = engine
        self._aliases = aliases or {}

    def __getattr__(self, name: str):
        target_name = self._aliases.get(name, name)
        impl_name = f"_impl_{target_name}"
        impl = getattr(self._e, impl_name, None)
        if impl is not None:
            return impl
        return getattr(self._e, target_name)
