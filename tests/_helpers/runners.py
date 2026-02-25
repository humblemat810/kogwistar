
from __future__ import annotations

from typing import Any, Callable
import inspect


def _call_scenario(scn: Callable[..., tuple[Any, str]], *, mode: str, **kwargs) -> tuple[Any, str]:
    """Call scenario with best-effort compatibility.

    Some scenarios are legacy and do not accept `mode=`. For those, do not inject `mode`.
    """
    try:
        sig = inspect.signature(scn)
    except Exception:
        try:
            return scn(mode=mode, **kwargs)
        except TypeError:
            return scn(**kwargs)

    if "mode" in sig.parameters:
        return scn(mode=mode, **kwargs)
    return scn(**kwargs)


def run_v1_scenario(scn: Callable[..., tuple[Any, str]], **kwargs) -> tuple[Any, str]:
    return _call_scenario(scn, mode="v1", **kwargs)


def run_v2_scenario(scn: Callable[..., tuple[Any, str]], **kwargs) -> tuple[Any, str]:
    return _call_scenario(scn, mode="v2", **kwargs)
