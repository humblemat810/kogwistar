from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [pytest.mark.ci, pytest.mark.core]


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS = ROOT / "constraints-pypy-3.12.txt"


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
    assert "'pydantic-extension>=0.0.7' 'pydantic>=2.6' 'anyio>=4.0' 'Jinja2>=3.1'" in workflow
    assert "'fastapi>=0.111' 'mcp>=1.27.0' 'httpx>=0.28.1'" in workflow
    assert "'python-jose[cryptography]>=3.3' 'PyJWT>=2.8' 'RapidFuzz>=3.13.0'" in workflow
    assert "'pytest>=8' pytest-asyncio pytest-dotenv pytest-xdist sqlalchemy" in workflow
    assert "Rebuild pydantic-core with the PyPy 3.12 FFI workaround" in workflow
    assert "--no-binary=pydantic-core" in workflow
    assert "pyo3-ffi/0.26.0/download" in workflow
    assert "not(Py_3_12)" in workflow
    assert "pyo3-ffi = { path = \"$pyo3_root\" }" in workflow
    assert 'cargo update --manifest-path "$source_root/Cargo.toml" --package pyo3-ffi' in workflow
    assert 'cargo tree --manifest-path "$source_root/Cargo.toml" --package pyo3-ffi' in workflow
    assert 'python -m maturin build \\' in workflow
    assert '--manifest-path "$source_root/Cargo.toml"' in workflow
    assert 'cargo update --manifest-path rust/Cargo.toml --package pyo3 --precise 0.28.3' in workflow
    assert 'old = \'pyo3 = { version = "0.29.0", features = ["extension-module", "abi3-py312", "generate-import-lib"] }\'' in workflow
    assert 'new = \'pyo3 = { version = "0.28.3", features = ["extension-module"] }\'' in workflow
    assert 'Production and normal' in workflow
    assert 'python -m maturin build --release --locked' in workflow
    assert '--interpreter "$(command -v python)"' in workflow
    assert '--out "$wheelhouse"' in workflow
    assert 'python -m pip install --no-deps --force-reinstall "$wheelhouse"/*.whl' in workflow
    assert 'Diagnose PyPy native extension import' in workflow
    assert 'Diagnose PyPy application dependency imports' in workflow
    assert 'import pydantic_core' in workflow
    assert 'pypy-dependency-verification.txt' in workflow
    assert 'traceback.print_exc()' in workflow
    assert 'GITHUB_STEP_SUMMARY' in workflow
    assert 'pypy-native-tests.txt' in workflow
    assert 'pypy-provider-free-tests.txt' in workflow
    assert 'uses: actions/upload-artifact@v4' in workflow
    assert "id: native_verify" in workflow
    assert workflow.count("working-directory: ${{ runner.temp }}") >= 4
    assert '"$GITHUB_WORKSPACE/tests"' in workflow
    assert "Run provider-free PyPy CI tests with Python authorities" in workflow
    assert 'KOGWISTAR_IMPL_MODE: "python"' in workflow
    assert "Keep native PyPy gate visible" in workflow
    assert 'not slow and not manual and not llm_real and not requires_ollama' in workflow
    assert '("numpy", "chromadb")' in workflow
    assert "import kogwistar._rust" in workflow
    assert "-p no:cacheprovider" in workflow
    assert "not slow and not manual" in workflow
    assert "::error title=PyPy native verification::" in workflow
    assert "::error title=PyPy application dependencies::" in workflow
    assert "::error title=PyPy Python-authority tests::" in workflow


def test_pypy_rpds_constraint_is_ci_only_and_exact() -> None:
    constraints = CONSTRAINTS.read_text(encoding="utf-8").splitlines()

    assert "rpds-py==2026.5.1" in constraints
    assert not any(
        line.strip().startswith("rpds-py") and line.strip() != "rpds-py==2026.5.1"
        for line in constraints
    )
    assert "pydantic==2.12.5" in constraints
    assert "pydantic-core==2.41.5" in constraints
    assert "constraints-pypy-3.12.txt" in WORKFLOW.read_text(encoding="utf-8")
