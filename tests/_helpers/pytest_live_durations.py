"""Print each pytest case's total duration as soon as it finishes."""

from __future__ import annotations

import time
from typing import Any


_starts: dict[str, float] = {}
_outcomes: dict[str, str] = {}


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    _starts[nodeid] = time.perf_counter()
    _outcomes[nodeid] = "ERROR"


def pytest_runtest_logreport(report: Any) -> None:
    if report.failed:
        _outcomes[report.nodeid] = "FAILED" if report.when == "call" else "ERROR"
    elif report.skipped:
        _outcomes[report.nodeid] = "SKIPPED"
    elif report.when == "call":
        _outcomes[report.nodeid] = "PASSED"


def pytest_runtest_logfinish(
    nodeid: str, location: tuple[str, int | None, str]
) -> None:
    started = _starts.pop(nodeid, None)
    outcome = _outcomes.pop(nodeid, "UNKNOWN")
    if started is None:
        return

    config = _config
    reporter = config.pluginmanager.get_plugin("terminalreporter") if config else None
    if reporter is not None:
        elapsed = time.perf_counter() - started
        slow_marker = " SLOW (>20m)" if elapsed >= 20 * 60 else ""
        reporter.write_line(
            f"TEST-DURATION {elapsed:.3f}s {outcome}{slow_marker} {nodeid}"
        )


_config: Any = None


def pytest_configure(config: Any) -> None:
    global _config
    _config = config


def pytest_unconfigure(config: Any) -> None:
    global _config
    _config = None
