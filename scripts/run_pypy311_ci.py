#!/usr/bin/env python3
"""Run Kogwistar's provider-free PyPy 3.11 compatibility profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MARKERS = "ci and not ci_full and not slow and not manual and not llm_real and not requires_ollama"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    if args.venv is not None and os.environ.get("KOGWISTAR_PYPY_VENV_ACTIVE") != "1":
        venv = args.venv.resolve()
        if not venv.exists():
            subprocess.check_call([sys.executable, "-m", "venv", str(venv)], cwd=root())
        env = os.environ.copy()
        env["KOGWISTAR_PYPY_VENV_ACTIVE"] = "1"
        command = [str(venv_python(venv)), str(Path(__file__).resolve()), "--skip-install"]
        return subprocess.call(command, cwd=root(), env=env)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("KOGWISTAR_IMPL_MODE", "python")
    env.setdefault("KOGWISTAR_LOG_LEVEL", "WARNING")
    env.setdefault("LOG_LEVEL", "WARNING")

    if not args.skip_install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], cwd=root(), env=env)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-pypy-3.11-experimental.txt"], cwd=root(), env=env)

    if sys.implementation.name != "pypy" or sys.version_info[:2] != (3, 11):
        print("This profile requires PyPy 3.11", file=sys.stderr)
        return 2

    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests", "-m", MARKERS, "-q", "--durations=25", "--durations-min=0", "-p", "no:cacheprovider"],
        cwd=root(),
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
