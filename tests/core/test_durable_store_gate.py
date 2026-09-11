from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "durable_store_gate.py"


def _gate() -> ModuleType:
    spec = importlib.util.spec_from_file_location("durable_store_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_report(path: Path, source_sha256: str) -> None:
    extension = path.parent / "_rust.pyd"
    extension.write_bytes(b"native")
    path.write_text(
        json.dumps(
            {
                "schema": "adr015-host-native/v1",
                "candidate_source_sha256": source_sha256,
                "extension": str(extension),
                "extension_sha256": "a" * 64,
                "smoke": {"sqlite_transaction_id_abi": True},
            }
        ),
        encoding="utf-8",
    )


def test_phase3_gate_is_persisted_and_requires_live_postgres_evidence() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "POSTGRES_TESTS" in source
    assert "live_postgres_required" in source
    assert 'counts["skipped"] == 0' in source
    assert "adr015-host-native/v1" in source
    assert "candidate_source_fingerprint" in source
    assert "--junitxml" in source


def test_junit_counts_accepts_testcase_root_and_testsuite_children(tmp_path: Path) -> None:
    gate = _gate()
    direct = tmp_path / "direct.xml"
    direct.write_text(
        "<testsuite><testcase/><testcase><skipped/></testcase></testsuite>",
        encoding="utf-8",
    )
    nested = tmp_path / "nested.xml"
    nested.write_text(
        '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/></testsuites>',
        encoding="utf-8",
    )

    assert gate._junit_counts(direct) == {
        "tests": 2,
        "failures": 0,
        "errors": 0,
        "skipped": 1,
    }
    assert gate._junit_counts(nested) == {
        "tests": 3,
        "failures": 1,
        "errors": 0,
        "skipped": 1,
    }


def test_phase3_postgres_group_rejects_skip_but_sqlite_allows_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()

    def fake_run(command, **_kwargs):
        junit_arg = next(value for value in command if str(value).startswith("--junitxml="))
        junit = Path(str(junit_arg).split("=", 1)[1])
        junit.write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="1"/>',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    postgres = gate._run_group(
        python=python,
        name="postgres",
        tests=("tests/core/test_rust_store_postgres_differential.py",),
        output_root=tmp_path,
        live_postgres=True,
    )
    sqlite = gate._run_group(
        python=python,
        name="sqlite",
        tests=("tests/core/test_rust_store_sqlite_differential.py",),
        output_root=tmp_path,
        live_postgres=False,
    )

    assert postgres["status"] == "failed"
    assert sqlite["status"] == "passed"


def test_phase3_gate_rejects_stale_native_report(tmp_path: Path) -> None:
    gate = _gate()
    native = tmp_path / "native.json"
    _native_report(native, "f" * 64)

    with pytest.raises(RuntimeError, match="does not match current candidate"):
        gate._load_native_report(native, "a" * 64)
