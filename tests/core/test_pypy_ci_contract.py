from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [pytest.mark.ci, pytest.mark.core]


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def test_ci_keeps_automatic_nonblocking_pypy_native_probe() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "pypy-beta-best-effort:" in workflow
    assert "continue-on-error: true" in workflow
    assert "nightly/py3.12/pypy-c-jit-latest-linux64.tar.gz" in workflow
    assert 'sha256sum "$archive"' in workflow
    assert 'stat --format=' in workflow
    assert 'test -x "$pypy_home/bin/pypy3"' in workflow
    assert 'assert sys.implementation.name == "pypy"' in workflow
    assert 'assert sys.version_info[:2] == (3, 12)' in workflow
    assert '("numpy", "chromadb")' in workflow
    assert "import kogwistar._rust" in workflow
    assert "-p no:cacheprovider" in workflow
    assert "not slow and not manual" in workflow
