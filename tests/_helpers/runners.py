
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import pytest


class TurnRunner(Protocol):
    """Execute a test scenario and return (conversation_engine, conversation_id)."""
    def __call__(self, *, backend_kind: str, tmp_path: Any, sa_engine: Any, pg_schema: Any, monkeypatch: Any) -> tuple[Any, str]: ...


def run_v1_scenario(scn: Callable[..., tuple[Any, str]], *, backend_kind: str, tmp_path: Any, sa_engine: Any, pg_schema: Any, monkeypatch: Any) -> tuple[Any, str]:
    return scn(backend_kind=backend_kind, tmp_path=tmp_path, sa_engine=sa_engine, pg_schema=pg_schema, monkeypatch=monkeypatch)


def run_v2_scenario(_scn: Callable[..., tuple[Any, str]], *, backend_kind: str, tmp_path: Any, sa_engine: Any, pg_schema: Any, monkeypatch: Any) -> tuple[Any, str]:
    pytest.skip("v2 workflow runner not implemented yet (PR#1 only installs parity harness)")
