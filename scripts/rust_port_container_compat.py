"""Run ADR-015 compatibility layers in reproducible parallel Docker isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Final

try:
    # Direct script execution places this file's directory on sys.path.
    # The package fallback keeps importlib-based test loading supported.
    from adr015_source_identity import candidate_source_fingerprint  # type: ignore[import-not-found]
except ModuleNotFoundError:
    from scripts.adr015_source_identity import candidate_source_fingerprint


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE: Final = "python:3.13.14-slim-bookworm"
_LAYERS: Final = ("core", "parser", "sink", "application")
_WORKER_SCRIPT: Final = ROOT / "scripts" / "adr015_container_worker.sh"
_DOCKERFILE: Final = ROOT / "scripts" / "adr015_container.Dockerfile"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_build_report(wheel: Path, root: Path = ROOT) -> dict[str, object] | None:
    path = wheel.parent / "build-report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "adr015-wheel-build/v1":
        raise SystemExit("wheel build report schema is unsupported")
    if report.get("wheel_sha256") != _sha256(wheel):
        raise SystemExit("wheel build report digest does not match candidate wheel")
    source_sha256, source_file_count = candidate_source_fingerprint(root)
    if report.get("candidate_source_sha256") != source_sha256:
        raise SystemExit("wheel build report source does not match current candidate")
    if report.get("candidate_source_file_count") != source_file_count:
        raise SystemExit("wheel build report source file count does not match candidate")
    return report


def _orchestrator_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        _WORKER_SCRIPT,
        _DOCKERFILE,
        ROOT / "scripts" / "adr015_native_path.py",
        ROOT / "scripts" / "adr015_candidate_identity.py",
        ROOT / "scripts" / "adr015_source_identity.py",
        ROOT / "scripts" / "rust_port_test_compare.py",
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--application-root",
        type=Path,
        default=ROOT.parent / "kogwistar-llm-wiki",
    )
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--profile",
        choices=("feature", "regression", "milestone"),
        default="feature",
    )
    parser.add_argument(
        "--implementation-mode", choices=("python", "shadow", "rust"), default="python"
    )
    parser.add_argument(
        "--meta-store-mode", choices=("python", "shadow", "rust"), default="rust"
    )
    parser.add_argument("--graph-store-mode", choices=("python", "shadow", "rust"))
    parser.add_argument("--runtime-mode", choices=("python", "shadow", "rust"))
    parser.add_argument("--server-mode", choices=("python", "rust"))
    parser.add_argument("--contracts-mode", choices=("python", "shadow", "rust"))
    parser.add_argument("--layer", action="append", choices=_LAYERS)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / ".codex" / "rust-port-container-compat.json",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=3,
        help="Parallel ordinary-container workers. Existing pytest process boundaries remain intact.",
    )
    parser.add_argument(
        "--pytest-workers",
        type=int,
        default=0,
        help="Opt-in xdist workers for proven-safe layers. Default 0: controlled core evidence found xdist slower.",
    )
    parser.add_argument(
        "--timing-history",
        action="append",
        type=Path,
        default=[],
        help="Prior merged/shard report used for median LPT balancing; repeatable.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--rebuild-image", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _wheel(raw: Path | None) -> Path:
    if raw is not None:
        wheel = raw.expanduser().resolve()
        if not wheel.is_file():
            raise SystemExit(f"wheel does not exist: {wheel}")
        return wheel
    candidates = sorted(
        (ROOT / ".codex").glob("wheelhouse*/kogwistar-*.whl"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise SystemExit("no wheel found; pass --wheel")
    return candidates[0].resolve()


def _dependency_fingerprint(application: Path, wheel: Path, base_image: str) -> str:
    digest = hashlib.sha256()
    digest.update(base_image.encode())
    digest.update(b"\0")
    digest.update(_sha256(wheel).encode())
    digest.update(b"\0")
    digest.update(_DOCKERFILE.read_bytes())
    digest.update(b"\0")
    digest.update(wheel.name.encode())
    for path in (
        application / "kg-doc-parser" / "pyproject.toml",
        application / "kogwistar-obsidian-sink" / "pyproject.toml",
    ):
        if not path.is_file():
            raise SystemExit(f"consumer dependency manifest does not exist: {path}")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _test_image_tag(application: Path, wheel: Path, base_image: str) -> str:
    fingerprint = _dependency_fingerprint(application, wheel, base_image)
    return f"kogwistar-adr015-compat:{fingerprint[:20]}"


def _copy_build_package(source: Path, target: Path) -> None:
    ignored = shutil.ignore_patterns(
        ".git",
        ".venv",
        ".pytest*",
        "pytest-cache-files-*",
        "tmp*",
        ".tmp*",
        "_tmp*",
        "__pycache__",
        "*.pyc",
        "tests",
        "dist",
        "build",
    )
    shutil.copytree(source, target, ignore=ignored)


def _snapshot_files(
    source: Path, target: Path, pathspecs: tuple[str, ...] = ()
) -> list[str]:
    """Copy Git-indexed files with current dirty contents into an immutable stage."""
    command = [
        "git",
        "-C",
        str(source),
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    if pathspecs:
        command.extend(("--", *pathspecs))
    result = subprocess.run(command, check=True, capture_output=True)
    relative_paths = sorted(
        value.decode("utf-8", errors="surrogateescape")
        for value in result.stdout.split(b"\0")
        if value
    )
    for relative in relative_paths:
        input_path = source / relative
        if not input_path.is_file():
            raise SystemExit(f"indexed source file does not exist: {input_path}")
        output_path = target / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
    return relative_paths


def _source_stage(root: Path, application: Path, target: Path) -> dict[str, int]:
    core = target / "core"
    consumer = target / "application"
    core_files = _snapshot_files(
        root,
        core,
        (
            "kogwistar",
            "docs",
            "keycloak",
            ".github",
            "rust",
            "contracts",
            "scripts",
            "tests",
            "pyproject.toml",
            "pytest.ini",
            "Dockerfile",
            "README.md",
            "LICENSE",
            "server_mcp.py",
        ),
    )
    application_files = _snapshot_files(
        application,
        consumer,
        (
            "src",
            "tests",
            "scripts",
            ".vscode",
            "pyproject.toml",
            "pytest.ini",
            ".env.example",
            "README.md",
            "QUICKSTART.md",
        ),
    )
    # The consumer intentionally ignores `.env.*`, but its CI contract reads
    # `.env.example`. Copy that public placeholder explicitly; never copy `.env`.
    env_example = application / ".env.example"
    if not env_example.is_file():
        raise SystemExit(f"consumer fixture does not exist: {env_example}")
    consumer.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_example, consumer / ".env.example")
    if ".env.example" not in application_files:
        application_files.append(".env.example")
    nested_count = 0
    for name in ("kg-doc-parser", "kogwistar-obsidian-sink"):
        nested_count += len(_snapshot_files(application / name, consumer / name))
    core_pin_files = _snapshot_files(
        application / "kogwistar",
        consumer / "kogwistar",
        ("tests/_helpers",),
    )
    return {
        "core": len(core_files),
        "application": len(application_files),
        "nested_consumers": nested_count,
        "application_core_pin_helpers": len(core_pin_files),
    }


def _image_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", tag],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _commit_provenance(application: Path) -> dict[str, str | None]:
    return {
        "core": _git_commit(ROOT),
        "application": _git_commit(application),
        "parser": _git_commit(application / "kg-doc-parser"),
        "sink": _git_commit(application / "kogwistar-obsidian-sink"),
        "application-core-pin": _git_commit(application / "kogwistar"),
    }


def _cleanup_containers(names: list[str]) -> None:
    if not names:
        return
    subprocess.run(
        ["docker", "rm", "--force", *names],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_test_image(
    *, application: Path, wheel: Path, base_image: str, rebuild: bool
) -> str:
    tag = _test_image_tag(application, wheel, base_image)
    if not rebuild and _image_exists(tag):
        return tag
    with tempfile.TemporaryDirectory(prefix="adr015-image-") as raw_context:
        context = Path(raw_context)
        shutil.copy2(wheel, context / wheel.name)
        _copy_build_package(application / "kg-doc-parser", context / "kg-doc-parser")
        _copy_build_package(
            application / "kogwistar-obsidian-sink",
            context / "kogwistar-obsidian-sink",
        )
        shutil.copy2(_DOCKERFILE, context / "Dockerfile")
        result = subprocess.run(
            [
                "docker",
                "build",
                "--build-arg",
                f"BASE_IMAGE={base_image}",
                "--build-arg",
                f"WHEEL_NAME={wheel.name}",
                "--tag",
                tag,
                str(context),
            ],
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    return tag


def _container_script() -> str:
    """Read persisted worker script for tests and diagnostics."""
    return _WORKER_SCRIPT.read_text(encoding="utf-8")


def _shard_report_path(report: Path, shard_index: int) -> Path:
    return report.with_name(f"{report.stem}.shard-{shard_index}{report.suffix}")


def _pytest_args(values: list[str]) -> list[str]:
    """Remove argparse's option terminator before forwarding to inner runner."""
    return values[1:] if values[:1] == ["--"] else values


def _merge_shard_reports(paths: list[Path], output: Path) -> dict[str, object]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("no shard reports")
    identities = {
        report.get("candidate", {}).get("identity_sha256") for report in reports
    }
    if len(identities) != 1 or None in identities:
        raise ValueError("shard candidate identities differ")
    expected_indexes = set(range(len(reports)))
    actual_indexes = {
        report.get("execution", {}).get("shard_index") for report in reports
    }
    if actual_indexes != expected_indexes:
        raise ValueError("shard indexes are incomplete or duplicated")
    if any(report.get("status") != "passed" for report in reports):
        raise ValueError("cannot merge a failed shard")

    merged_layers: list[dict[str, object]] = []
    layer_names = [
        layer.get("name")
        for layer in reports[0].get("layers", [])
        if isinstance(layer, dict)
    ]
    for name in layer_names:
        parts = [
            next(
                layer
                for layer in report.get("layers", [])
                if isinstance(layer, dict) and layer.get("name") == name
            )
            for report in reports
        ]
        counts = {part.get("target_group_count") for part in parts}
        assignments = {tuple(part.get("group_assignments", [])) for part in parts}
        if len(counts) != 1 or len(assignments) != 1:
            raise ValueError(f"{name} shard plans differ")
        target_count = next(iter(counts))
        if not isinstance(target_count, int):
            raise ValueError(f"{name} target group count is invalid")
        selected = [
            index for part in parts for index in part.get("selected_group_indexes", [])
        ]
        if sorted(selected) != list(range(target_count)):
            raise ValueError(f"{name} groups are missing or duplicated")
        runs = sorted(
            (run for part in parts for run in part.get("runs", [])),
            key=lambda run: run.get("group_index", -1),
        )
        merged_layers.append(
            {
                **parts[0],
                "commands": [run["command"] for run in runs],
                "runs": runs,
                "selected_group_indexes": list(range(target_count)),
                "shard_index": None,
                "shard_count": len(reports),
                "duration_seconds": max(
                    float(part.get("duration_seconds", 0.0)) for part in parts
                ),
            }
        )

    merged: dict[str, object] = {
        **reports[0],
        "status": "passed",
        "execution": {
            **reports[0].get("execution", {}),
            "shard_index": None,
            "shard_count": len(reports),
            "worker_reports": [str(path) for path in paths],
        },
        "layers": merged_layers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merged


def main() -> int:
    orchestrator_started = time.perf_counter()
    args = _args()
    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    pytest_workers = args.pytest_workers
    if pytest_workers < 0:
        raise SystemExit("--pytest-workers cannot be negative")
    application = args.application_root.expanduser().resolve()
    if not application.is_dir():
        raise SystemExit(f"application root does not exist: {application}")
    wheel = _wheel(args.wheel)
    build_report = _wheel_build_report(wheel)
    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    image = _ensure_test_image(
        application=application,
        wheel=wheel,
        base_image=args.image,
        rebuild=args.rebuild_image,
    )

    common_args = [
        "--implementation-mode",
        args.implementation_mode,
        "--meta-store-mode",
        args.meta_store_mode,
        "--profile",
        args.profile,
        "--backend",
        "docker-clean-dual-venv-sharded",
        "--storage-root",
        "container-local-isolated",
        "--shard-count",
        str(args.shards),
        "--pytest-workers",
        str(pytest_workers),
    ]
    provenance = _commit_provenance(application)
    for name, value in provenance.items():
        if value is not None:
            common_args.extend((f"--{name}-commit", value))
    for name, value in (
        ("--graph-store-mode", args.graph_store_mode),
        ("--runtime-mode", args.runtime_mode),
        ("--server-mode", args.server_mode),
        ("--contracts-mode", args.contracts_mode),
    ):
        if value is not None:
            common_args.extend((name, value))
    for layer in args.layer or []:
        common_args.extend(("--layer", layer))
    for history in args.timing_history:
        resolved = history.expanduser().resolve()
        try:
            relative = resolved.relative_to(report.parent)
        except ValueError as error:
            raise SystemExit(
                "--timing-history must be inside report directory"
            ) from error
        common_args.extend(("--timing-history", f"/reports/{relative.as_posix()}"))
    if args.resume:
        common_args.append("--resume")
    pytest_args = _pytest_args(args.pytest_args)

    metadata = {
        "base_image": args.image,
        "test_image": image,
        "wheel": str(wheel),
        "wheel_sha256": _sha256(wheel),
        "wheel_build_report": build_report,
        "application_root": str(application),
        "report": str(report),
        "shards": args.shards,
        "pytest_workers": pytest_workers,
        "commit_provenance": provenance,
        "container_orchestrator_sha256": _orchestrator_fingerprint(),
    }
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)

    processes: list[subprocess.Popen[bytes]] = []
    shard_paths: list[Path] = []
    container_names: list[str] = []
    with tempfile.TemporaryDirectory(prefix="adr015-source-") as raw_stage:
        stage = Path(raw_stage)
        snapshot_counts = _source_stage(ROOT, application, stage)
        metadata["snapshot_files"] = snapshot_counts
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
        try:
            for shard_index in range(args.shards):
                shard_report = _shard_report_path(report, shard_index)
                shard_paths.append(shard_report)
                container_name = f"adr015-compat-{os.getpid()}-{shard_index}"
                container_names.append(container_name)
                command = ["docker", "run", "--name", container_name]
                if not args.keep_container:
                    command.append("--rm")
                command.extend(
                    [
                        "--mount",
                        f"type=bind,src={stage / 'core'},dst=/source/core,readonly",
                        "--mount",
                        f"type=bind,src={stage / 'application'},dst=/source/application,readonly",
                        "--mount",
                        f"type=bind,src={report.parent},dst=/reports",
                        "--tmpfs",
                        "/tmp:exec,size=4g",
                        image,
                        "/bin/sh",
                        "/source/core/scripts/adr015_container_worker.sh",
                        *common_args,
                        "--shard-index",
                        str(shard_index),
                        "--report",
                        f"/reports/{shard_report.name}",
                        "--",
                        *pytest_args,
                    ]
                )
                print(f"[{container_name}] starting", flush=True)
                processes.append(subprocess.Popen(command))
        except FileNotFoundError:
            raise SystemExit("docker executable not found") from None

        try:
            returncodes = [process.wait() for process in processes]
        except BaseException:
            if not args.keep_container:
                _cleanup_containers(container_names)
            raise
    if any(returncode != 0 for returncode in returncodes):
        if not args.keep_container:
            _cleanup_containers(container_names)
        return next(code for code in returncodes if code != 0)
    try:
        merged = _merge_shard_reports(shard_paths, report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"shard merge failed: {error}", flush=True)
        return 1
    execution = merged["execution"]
    if not isinstance(execution, dict):
        raise RuntimeError("merged compatibility report execution is invalid")
    execution["orchestrator_wall_seconds"] = round(
        time.perf_counter() - orchestrator_started, 6
    )
    execution["container_orchestrator_sha256"] = metadata[
        "container_orchestrator_sha256"
    ]
    execution["wheel_sha256"] = metadata["wheel_sha256"]
    execution["wheel_build_report"] = metadata["wheel_build_report"]
    report.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
