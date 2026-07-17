"""Run ADR-015 compatibility layers in reproducible Docker dual-venv isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "python:3.13.14-slim-bookworm"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
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
    parser.add_argument("--profile", choices=("feature", "regression", "milestone"), default="feature")
    parser.add_argument("--implementation-mode", choices=("python", "shadow", "rust"), default="python")
    parser.add_argument("--meta-store-mode", choices=("python", "shadow", "rust"), default="rust")
    parser.add_argument("--graph-store-mode", choices=("python", "shadow", "rust"))
    parser.add_argument("--runtime-mode", choices=("python", "shadow", "rust"))
    parser.add_argument("--server-mode", choices=("python", "rust"))
    parser.add_argument("--contracts-mode", choices=("python", "shadow", "rust"))
    parser.add_argument("--layer", action="append", choices=("core", "parser", "sink", "application"))
    parser.add_argument("--report", type=Path, default=ROOT / ".codex" / "rust-port-container-compat.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-container", action="store_true")
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


def _container_script() -> str:
    return r"""
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git >/dev/null
python -m venv /tmp/core
python -m venv /tmp/consumer
wheel="$(find /wheel -maxdepth 1 -type f -name 'kogwistar-*.whl' -print -quit)"
test -n "$wheel"
/tmp/core/bin/pip install --disable-pip-version-check --quiet "${wheel}[full,test]"
/tmp/consumer/bin/pip install --disable-pip-version-check --quiet "${wheel}[test,chroma]" \
  fastapi langchain-openai \
  /workspace/application/kg-doc-parser \
  /workspace/application/kogwistar-obsidian-sink
native="$(cd /tmp && /tmp/core/bin/python -P -c 'import pathlib, kogwistar._rust as native; print(pathlib.Path(native.__file__).resolve())')"
cp "$native" /workspace/core/kogwistar/_rust.abi3.so
cleanup() { rm -f /workspace/core/kogwistar/_rust.abi3.so; }
trap cleanup EXIT INT TERM
cd /workspace/core
/tmp/core/bin/python scripts/rust_port_compat.py \
  --application-root /workspace/application \
  --python /tmp/core/bin/python \
  --consumer-python /tmp/consumer/bin/python \
  "$@"
"""


def main() -> int:
    args = _args()
    application = args.application_root.expanduser().resolve()
    if not application.is_dir():
        raise SystemExit(f"application root does not exist: {application}")
    wheel = _wheel(args.wheel)
    report = args.report.expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report_in_core = report.relative_to(ROOT)
    except ValueError as error:
        raise SystemExit("--report must be inside candidate workspace") from error

    runner_args = [
        "--implementation-mode",
        args.implementation_mode,
        "--meta-store-mode",
        args.meta_store_mode,
        "--profile",
        args.profile,
        "--backend",
        "docker-clean-dual-venv",
        "--storage-root",
        "container-isolated",
        "--report",
        f"/workspace/core/{report_in_core.as_posix()}",
    ]
    for name, value in (
        ("--graph-store-mode", args.graph_store_mode),
        ("--runtime-mode", args.runtime_mode),
        ("--server-mode", args.server_mode),
        ("--contracts-mode", args.contracts_mode),
    ):
        if value is not None:
            runner_args.extend((name, value))
    for layer in args.layer or []:
        runner_args.extend(("--layer", layer))
    if args.resume:
        runner_args.append("--resume")
    runner_args.extend(args.pytest_args)

    command = [
        "docker",
        "run",
        "--name",
        f"adr015-compat-{os.getpid()}",
    ]
    if not args.keep_container:
        command.append("--rm")
    command.extend(
        [
            "-v",
            f"{ROOT}:/workspace/core",
            "-v",
            f"{application}:/workspace/application",
            "-v",
            f"{wheel}:/wheel/{wheel.name}:ro",
            args.image,
            "bash",
            "-lc",
            _container_script(),
            "adr015-container",
            *runner_args,
        ]
    )
    print(
        json.dumps(
            {
                "image": args.image,
                "wheel": str(wheel),
                "wheel_sha256": _sha256(wheel),
                "application_root": str(application),
                "report": str(report),
                "runner_args": runner_args,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        return subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        raise SystemExit("docker executable not found") from None


if __name__ == "__main__":
    raise SystemExit(main())
