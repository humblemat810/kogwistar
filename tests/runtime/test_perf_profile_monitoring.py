from __future__ import annotations

from collections.abc import Callable

import pytest

from kogwistar.runtime import perf_profile


class _Events:
    PY_START = 1
    PY_RETURN = 2
    PY_UNWIND = 4


class _Monitoring:
    PROFILER_ID = 5
    events = _Events()

    def __init__(self) -> None:
        self.callbacks: dict[int, Callable[..., None] | None] = {}
        self.active_events: int | None = None
        self.tool_name: str | None = None
        self.freed_tool_id: int | None = None

    def use_tool_id(self, tool_id: int, name: str) -> None:
        assert tool_id == self.PROFILER_ID
        self.tool_name = name

    def register_callback(
        self, tool_id: int, event: int, callback: Callable[..., None] | None
    ) -> None:
        assert tool_id == self.PROFILER_ID
        self.callbacks[event] = callback

    def set_events(self, tool_id: int, events: int) -> None:
        assert tool_id == self.PROFILER_ID
        self.active_events = events

    def free_tool_id(self, tool_id: int) -> None:
        self.freed_tool_id = tool_id


def test_profiler_gracefully_skips_unavailable_monitoring_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(perf_profile.sys, "monitoring", raising=False)
    profiler = perf_profile.SysMonitoringWallProfiler()

    assert profiler.start() is False
    profiler.stop()


def test_profiler_registers_and_releases_a_valid_monitoring_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitoring = _Monitoring()
    monkeypatch.setattr(perf_profile.sys, "monitoring", monitoring, raising=False)
    profiler = perf_profile.SysMonitoringWallProfiler()

    assert profiler.start() is True
    assert monitoring.tool_name == "kogwistar.perf_profile"
    assert monitoring.active_events == (
        _Events.PY_START | _Events.PY_RETURN | _Events.PY_UNWIND
    )
    assert all(monitoring.callbacks.values())

    profiler.stop()

    assert monitoring.active_events == 0
    assert all(callback is None for callback in monitoring.callbacks.values())
    assert monitoring.freed_tool_id == _Monitoring.PROFILER_ID
