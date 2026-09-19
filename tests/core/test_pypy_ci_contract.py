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
    assert 'constraints-pypy-3.12.txt' in workflow
    assert 'cargo update --manifest-path rust/Cargo.toml --package pyo3 --precise 0.28.3' in workflow
    assert 'old = \'pyo3 = { version = "0.29.0"\'' in workflow
    assert 'new = \'pyo3 = { version = "0.28.3"\'' in workflow
    assert 'Production and normal' in workflow
    assert 'python -m maturin build --release --locked' in workflow
    assert '--interpreter "$(command -v python)"' in workflow
    assert '--out "$wheelhouse"' in workflow
    assert 'python -m pip install --no-deps --force-reinstall "$wheelhouse"/*.whl' in workflow
    assert 'Diagnose PyPy native extension import' in workflow
    assert 'traceback.print_exc()' in workflow
    assert 'not slow and not manual and not llm_real and not requires_ollama' in workflow
    assert '("numpy", "chromadb")' in workflow
    assert "import kogwistar._rust" in workflow
    assert "-p no:cacheprovider" in workflow
    assert "not slow and not manual" in workflow
