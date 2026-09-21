from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [pytest.mark.ci, pytest.mark.core]


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CONSTRAINTS = ROOT / "constraints-pypy-3.12.txt"
ABI_AUDIT = ROOT / "scripts" / "pypy_ffi_symbol_audit.py"
DLOPEN_PROBE = ROOT / "scripts" / "pypy_native_extension_dlopen.py"


def test_ci_keeps_automatic_nonblocking_pypy_native_probe() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "pypy-beta-best-effort:" in workflow
    assert "continue-on-error: true" in workflow
    assert "pypy-c-jit-171509-4db12e5e6f4d-linux64.tar.gz" in workflow
    assert 'sha256sum "$archive"' in workflow
    assert 'stat --format=' in workflow
    assert 'test -x "$pypy_home/bin/pypy3"' in workflow
    assert 'assert sys.implementation.name == "pypy"' in workflow
    assert 'assert sys.version_info[:2] == (3, 12)' in workflow
    assert 'constraints-pypy-3.12.txt' in workflow
    assert "'pydantic-extension>=0.0.7' 'pydantic>=2.6' 'anyio>=4.0' 'Jinja2>=3.1'" in workflow
    assert "'fastapi>=0.111' 'mcp>=1.27,<2' 'httpx>=0.28.1'" in workflow
    assert "'python-jose[cryptography]>=3.3' 'PyJWT>=2.8' 'RapidFuzz>=3.13.0'" in workflow
    assert "'pytest>=8' pytest-asyncio pytest-dotenv pytest-xdist sqlalchemy" in workflow
    assert "Build patched pydantic-core wheel" in workflow
    assert "Import patched pydantic-core wheel" in workflow
    assert "--no-binary=pydantic-core" in workflow
    assert "pyo3-ffi/0.26.0/download" in workflow
    assert "not(Py_3_12)" in workflow
    assert 'old = "#[cfg_attr(PyPy, link_name = \\"' in workflow
    assert 'root.rglob("*.rs")' in workflow
    assert "patched >= 100" in workflow
    assert "pyo3-ffi = { path = \"$pyo3_root\" }" in workflow
    assert 'cargo update --manifest-path "$source_root/Cargo.toml" --package pyo3-ffi' in workflow
    assert 'cargo tree --manifest-path "$source_root/Cargo.toml" --package pyo3-ffi' in workflow
    assert 'python -m maturin build \\' in workflow
    assert '--manifest-path "$source_root/Cargo.toml"' in workflow
    assert 'python scripts/pypy_pydantic_core_smoke.py --metadata-version' in workflow
    assert 'python scripts/pypy_pydantic_core_smoke.py \\' in workflow
    assert "Audit built pydantic-core Python ABI" in workflow
    assert "Dlopen built pydantic-core extension" in workflow
    assert "Supplementary PyO3 source symbol audit" in workflow
    assert "Audit built Kogwistar Python ABI" in workflow
    assert "Dlopen built Kogwistar extension" in workflow
    assert "scripts/pypy_ffi_symbol_audit.py" in workflow
    assert "::error title=PyPy pydantic-core ABI::" in workflow
    assert "::error title=Kogwistar PyPy ABI::" in workflow
    assert "PYPY_SHA256" in workflow
    assert "actual_sha256" in workflow
    assert "pypy-c-jit-171509-4db12e5e6f4d-linux64.tar.gz" in workflow
    assert "--extension" in workflow
    assert "scripts/pypy_native_extension_dlopen.py" in workflow
    assert 'cargo update --manifest-path rust/Cargo.toml --package pyo3 --precise 0.28.3' in workflow
    assert 'old = \'pyo3 = { version = "0.29.0", features = ["extension-module", "abi3-py312", "generate-import-lib"] }\'' in workflow
    assert 'new = \'pyo3 = { version = "0.28.3", features = ["extension-module"] }\'' in workflow
    assert 'Production and normal' in workflow
    assert 'python -m maturin build --release --locked' in workflow
    assert '--interpreter "$(command -v python)"' in workflow
    assert '--out "$wheelhouse"' in workflow
    assert 'python -m pip install --no-deps --force-reinstall "$wheelhouse"/*.whl' in workflow
    assert 'Directly probe PyPy native extension import' in workflow
    assert 'scripts/pypy_native_extension_smoke.py' in workflow
    assert 'Diagnose PyPy application dependency imports' in workflow
    assert 'import pydantic_core' in workflow
    assert 'pypy-dependency-verification.txt' in workflow
    assert 'traceback.print_exc()' in workflow
    assert 'GITHUB_STEP_SUMMARY' in workflow
    assert 'pypy-native-tests.txt' in workflow
    assert 'pypy-provider-free-tests.txt' in workflow
    assert 'uses: actions/upload-artifact@v6' in workflow
    assert "id: native_verify" in workflow
    assert "id: native_direct_verify" in workflow
    assert "id: pydantic_binary_abi" in workflow
    assert "id: pydantic_dlopen" in workflow
    assert "id: native_binary_abi" in workflow
    assert "id: native_dlopen" in workflow
    assert workflow.index("Audit built pydantic-core Python ABI") < workflow.index(
        "Import patched pydantic-core wheel"
    )
    assert workflow.index("Audit built Kogwistar Python ABI") < workflow.index(
        "Directly probe PyPy native extension import"
    )
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


def test_pypy_binary_audit_reports_exact_missing_symbols(tmp_path, monkeypatch) -> None:
    from scripts import pypy_ffi_symbol_audit as audit_module

    pyo3_root = tmp_path / "pyo3"
    source = pyo3_root / "src" / "setobject.rs"
    source.parent.mkdir(parents=True)
    source.write_text(
        '#[cfg_attr(PyPy, link_name = "PyPySet_New")]\n'
        'fn new();\n'
        '#[cfg_attr(PyPy, link_name = "PyPySet_Add")]\n'
        'fn add();\n',
        encoding="utf-8",
    )
    pypy_root = tmp_path / "pypy"
    runtime = pypy_root / "bin" / "pypy3"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"not an ELF file")
    extension = tmp_path / "_rust.pypy312-pp80-x86_64-linux-gnu.so"
    extension.write_bytes(b"not an ELF file")

    monkeypatch.setattr(audit_module.shutil, "which", lambda _: "nm")

    def fake_run(command, **kwargs):
        class Result:
            returncode = 0
            stderr = ""
            stdout = ""

        if command[2] == "--undefined-only":
            Result.stdout = "                 U PyList_GET_SIZE\n                 U PyPySet_New@PYTHON\n"
        else:
            Result.stdout = "000000 T PyPySet_New\n000000 T PyObject_Call\n"
        return Result()

    monkeypatch.setattr(audit_module.subprocess, "run", fake_run)
    result = audit_module.audit(
        extensions=[extension],
        pyo3_root=pyo3_root,
        pypy_root=pypy_root,
    )

    assert result["runtime_paths"] == [str(runtime)]
    assert result["extensions"][0]["required_python_symbols"] == [
        "PyList_GET_SIZE",
        "PyPySet_New",
    ]
    assert result["extensions"][0]["missing_python_symbols"] == ["PyList_GET_SIZE"]
    assert result["extensions"][0]["ok"] is False
    assert result["binary_abi_ok"] is False
    assert result["source_audit"]["missing_symbols"] == ["PyPySet_Add"]


def test_pypy_dlopen_preflight_uses_rtld_now(monkeypatch, tmp_path) -> None:
    from scripts import pypy_native_extension_dlopen as dlopen_module

    extension = tmp_path / "native.so"
    seen: dict[str, object] = {}

    def fake_cdll(path, *, mode):
        seen["path"] = path
        seen["mode"] = mode

    monkeypatch.setattr(dlopen_module.ctypes, "CDLL", fake_cdll)

    assert dlopen_module.preflight(extension) is None
    assert seen == {
        "path": str(extension),
        "mode": getattr(dlopen_module.os, "RTLD_NOW", 2),
    }


def test_pypy_binary_audit_uses_undefined_nm_and_python_abi_filter() -> None:
    audit_source = ABI_AUDIT.read_text(encoding="utf-8")
    dlopen_source = DLOPEN_PROBE.read_text(encoding="utf-8")

    assert '"--undefined-only"' in audit_source
    assert "_PYTHON_ABI_SYMBOL" in audit_source
    assert "split(\"@\", 1)" in audit_source
    assert "ctypes.CDLL" in dlopen_source
    assert "RTLD_NOW" in dlopen_source


def test_pypy_311_experimental_workflow_is_pinned_and_nonblocking() -> None:
    workflow = (ROOT / ".github" / "workflows" / "pypy-311-experimental.yml").read_text(
        encoding="utf-8"
    )

    assert "continue-on-error: true" in workflow
    assert "uses: actions/setup-python@v7" in workflow
    assert "python-version: pypy-3.11-v7.3.20" in workflow
    assert "cache: pip" in workflow
    assert "Install official PyPy 3.11 release" not in workflow
    assert "pypy_url" not in workflow
    assert "pypy_sha256" not in workflow
    assert "pypy3.11-v7.3.20-linux64.tar.bz2" not in workflow
    assert "actual_sha256" not in workflow
    assert "--bzip2" not in workflow
    assert "KOGWISTAR_IMPL_MODE: python" in workflow
    assert "no image is published" in workflow


def test_main_ci_matrix_uses_hosted_cpython_and_pypy_runtimes() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "label: cpython312" in workflow
    assert "label: cpython313" in workflow
    assert "label: cpython314" in workflow
    assert "python-version: pypy-3.11-v7.3.20" in workflow
    assert "uses: actions/setup-python@v7" in workflow
    assert "Install PyPy 3.11 Python-authority dependencies" in workflow
    assert "Run PyPy 3.11 CI tests" in workflow
