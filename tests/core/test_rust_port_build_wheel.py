from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rust_port_build_wheel.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rust_port_build_wheel", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wheel_builder_uses_persisted_inspectable_entrypoints() -> None:
    module = _load()
    dockerfile = module.DOCKERFILE.read_text(encoding="utf-8")
    worker = module.WORKER.read_text(encoding="utf-8")
    source = SCRIPT.read_text(encoding="utf-8")

    assert "rust:1.91.1-bookworm" in dockerfile
    assert (
        'ENTRYPOINT ["/bin/sh", "/source/scripts/adr015_build_wheel.sh"]'
        in dockerfile
    )
    assert "maturin build" in worker
    assert "--release" in worker and "--locked" in worker
    assert "bash -c" not in source
    assert "python -c" not in source


def test_wheel_builder_report_fingerprints_persisted_build_logic(tmp_path: Path) -> None:
    module = _load()
    first = module._sha256(module.DOCKERFILE)
    second = module._sha256(module.DOCKERFILE)

    assert first == second
    assert len(first) == 64


def test_wheel_builder_source_fingerprint_tracks_dirty_and_untracked_inputs(
    tmp_path: Path,
) -> None:
    module = _load()
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='candidate'\n")
    rust = tmp_path / "rust"
    rust.mkdir()
    tracked = rust / "Cargo.toml"
    tracked.write_text("[workspace]\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "pyproject.toml", "rust/Cargo.toml"],
        check=True,
    )
    tracked.write_text("[workspace]\nmembers=[]\n")
    package = tmp_path / "kogwistar"
    package.mkdir()
    (package / "new.py").write_text("VALUE = 1\n")
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "ignored.txt").write_text("ignore me\n")

    first, count = module._source_sha256(tmp_path)
    staged = tmp_path.parent / "staged"
    for source in module._source_inputs(tmp_path):
        target = staged / source.relative_to(tmp_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    staged_hash, staged_count = module._source_sha256(staged)
    (package / "new.py").write_text("VALUE = 2\n")
    second, second_count = module._source_sha256(tmp_path)

    assert count == second_count == 3
    assert (staged_hash, staged_count) == (first, count)
    assert first != second
