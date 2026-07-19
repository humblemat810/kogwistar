from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.regression


def _harness():
    path = ROOT / "scripts" / "rust_port_container_compat.py"
    spec = importlib.util.spec_from_file_location("rust_port_container_compat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_container_harness_uses_dual_venv_and_cleans_native_bridge() -> None:
    harness = _harness()
    script = harness._container_script()

    assert harness._WORKER_SCRIPT.is_file()
    assert "pytest" not in script
    assert "mkdir -p /workspace/core /workspace/application/logs" in script
    assert "cp -a /source/core/. /workspace/core/" in script
    assert "cp -a /source/application/. /workspace/application/" in script
    assert "/source/core/scripts/adr015_native_path.py" in script
    assert "python -P -c" not in script
    assert 'cp "$native" /workspace/core/kogwistar/_rust.abi3.so' in script
    assert "exec /opt/core/bin/python" in script
    assert "--consumer-python /opt/consumer/bin/python" in script

    dockerfile = harness._DOCKERFILE.read_text(encoding="utf-8")
    assert harness._DOCKERFILE.is_file()
    assert "python -m venv /opt/core" in dockerfile
    assert "python -m venv /opt/consumer" in dockerfile
    assert "${WHEEL_NAME}[full,test]" in dockerfile
    assert "${WHEEL_NAME}[test,chroma]" in dockerfile
    assert "candidate.whl" not in dockerfile
    assert "pytest-xdist>=3.8,<4" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.10.10" in dockerfile
    assert "UV_HTTP_TIMEOUT=60" in dockerfile
    assert "UV_HTTP_RETRIES=2" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv" in dockerfile
    assert "uv pip install --python /opt/core/bin/python" in dockerfile
    assert "uv pip install --python /opt/consumer/bin/python" in dockerfile
    assert "/opt/core/bin/pip install" not in dockerfile
    assert "/opt/consumer/bin/pip install" not in dockerfile

    native_path_helper = ROOT / "scripts" / "adr015_native_path.py"
    assert native_path_helper.is_file()
    assert "Path(native.__file__).resolve()" in native_path_helper.read_text(
        encoding="utf-8"
    )


def test_container_harness_default_image_is_patch_pinned() -> None:
    assert _harness().DEFAULT_IMAGE == "python:3.13.14-slim-bookworm"


def test_wheel_build_report_must_match_candidate_digest(tmp_path: Path) -> None:
    harness = _harness()
    wheel = tmp_path / "kogwistar-test.whl"
    wheel.write_bytes(b"candidate")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='candidate'\n")
    source_sha256, source_file_count = harness.candidate_source_fingerprint(tmp_path)
    report = {
        "schema": "adr015-wheel-build/v1",
        "wheel_sha256": harness._sha256(wheel),
        "candidate_source_sha256": source_sha256,
        "candidate_source_file_count": source_file_count,
    }
    (tmp_path / "build-report.json").write_text(json.dumps(report))

    assert harness._wheel_build_report(wheel, tmp_path) == report
    wheel.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="digest does not match"):
        harness._wheel_build_report(wheel, tmp_path)


def test_container_harness_defaults_xdist_off_after_controlled_slowdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    monkeypatch.setattr("sys.argv", ["rust_port_container_compat.py"])

    assert harness._args().pytest_workers == 0


def test_container_harness_does_not_forward_argparse_terminator() -> None:
    harness = _harness()

    assert harness._pytest_args(["--", "-k", "focused"]) == ["-k", "focused"]


def test_snapshot_files_copies_indexed_dirty_and_untracked_source_only(
    tmp_path: Path,
) -> None:
    harness = _harness()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    subprocess = __import__("subprocess")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    tracked = source / "tracked.py"
    tracked.write_text("before", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "tracked.py"], check=True)
    tracked.write_text("dirty", encoding="utf-8")
    (source / "untracked.py").write_text("include", encoding="utf-8")
    (source / "ignored.py").write_text("ignore", encoding="utf-8")
    (source / ".gitignore").write_text("ignored.py\n", encoding="utf-8")

    files = harness._snapshot_files(source, target)

    assert files == [".gitignore", "tracked.py", "untracked.py"]
    assert (target / "tracked.py").read_text(encoding="utf-8") == "dirty"
    assert (target / "untracked.py").read_text(encoding="utf-8") == "include"
    assert not (target / "ignored.py").exists()


def test_commit_provenance_and_container_cleanup_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness()
    commits = iter(["core", "application", "parser", "sink", "pin"])
    monkeypatch.setattr(harness, "_git_commit", lambda _path: next(commits))

    assert harness._commit_provenance(ROOT) == {
        "core": "core",
        "application": "application",
        "parser": "parser",
        "sink": "sink",
        "application-core-pin": "pin",
    }

    calls: list[list[str]] = []
    monkeypatch.setattr(
        harness.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )
    harness._cleanup_containers(["worker-0", "worker-1"])
    assert calls == [["docker", "rm", "--force", "worker-0", "worker-1"]]


def test_source_stage_copies_public_env_example_but_never_secret_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness()
    root = tmp_path / "root"
    application = tmp_path / "application"
    stage = tmp_path / "stage"
    root.mkdir()
    application.mkdir()
    (application / ".env.example").write_text("TOKEN=placeholder\n", encoding="utf-8")
    (application / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    monkeypatch.setattr(harness, "_snapshot_files", lambda *_args, **_kwargs: [])

    harness._source_stage(root, application, stage)

    assert (stage / "application" / ".env.example").read_text() == "TOKEN=placeholder\n"
    assert not (stage / "application" / ".env").exists()


def _write_shard(
    path: Path,
    *,
    shard_index: int,
    identity: str = "same",
    selected: list[int],
    assignments: list[int] | None = None,
) -> None:
    assignments = assignments or [0, 1]
    path.write_text(
        json.dumps(
            {
                "candidate": {"identity_sha256": identity},
                "execution": {"shard_index": shard_index, "shard_count": 2},
                "status": "passed",
                "layers": [
                    {
                        "name": "core",
                        "status": "passed",
                        "target_group_count": 2,
                        "group_assignments": assignments,
                        "selected_group_indexes": selected,
                        "runs": [
                            {
                                "group_index": index,
                                "group_key": f"test_{index}.py",
                                "command": ["pytest", f"test_{index}.py"],
                                "returncode": 0,
                                "duration_seconds": float(index + 1),
                            }
                            for index in selected
                        ],
                        "duration_seconds": float(shard_index + 1),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_merge_shards_requires_same_identity_and_exact_group_coverage(
    tmp_path: Path,
) -> None:
    harness = _harness()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    _write_shard(first, shard_index=0, selected=[0])
    _write_shard(second, shard_index=1, selected=[1])

    merged = harness._merge_shard_reports([first, second], output)

    assert merged["status"] == "passed"
    assert [run["group_index"] for run in merged["layers"][0]["runs"]] == [0, 1]
    assert output.is_file()

    _write_shard(second, shard_index=1, identity="different", selected=[1])
    with pytest.raises(ValueError, match="identities differ"):
        harness._merge_shard_reports([first, second], output)

    _write_shard(second, shard_index=1, selected=[0])
    with pytest.raises(ValueError, match="missing or duplicated"):
        harness._merge_shard_reports([first, second], output)
