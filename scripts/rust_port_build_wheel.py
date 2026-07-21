"""Build one reproducible ADR-015 Linux wheel from current source in Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

try:
    from adr015_source_identity import (
        candidate_source_files,
        candidate_source_fingerprint,
    )
except ModuleNotFoundError:  # imported as a repository module in unit tests
    from scripts.adr015_source_identity import (
        candidate_source_files,
        candidate_source_fingerprint,
    )


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "scripts" / "adr015_wheel_builder.Dockerfile"
WORKER = ROOT / "scripts" / "adr015_build_wheel.sh"
SOURCE_IDENTITY = ROOT / "scripts" / "adr015_source_identity.py"
DEFAULT_IMAGE = "kogwistar-adr015-wheel-builder:rust-1.91.1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".codex" / "wheelhouse-adr015-current",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--rebuild-image", action="store_true")
    return parser.parse_args()


def _run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, **kwargs)


def _image_exists(image: str) -> bool:
    return (
        _run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _ensure_image(image: str, rebuild: bool) -> None:
    if not rebuild and _image_exists(image):
        return
    with tempfile.TemporaryDirectory(prefix="adr015-wheel-image-") as raw:
        context = Path(raw)
        shutil.copy2(DOCKERFILE, context / "Dockerfile")
        result = _run(["docker", "build", "--tag", image, str(context)])
    if result.returncode:
        raise SystemExit(result.returncode)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(root: Path = ROOT) -> tuple[str, int]:
    return candidate_source_fingerprint(root)


def _source_inputs(root: Path = ROOT) -> list[Path]:
    return candidate_source_files(root)


def main() -> int:
    args = _args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _ensure_image(args.image, args.rebuild_image)
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--mount",
            f"type=bind,src={ROOT},dst=/source,readonly",
            "--mount",
            f"type=bind,src={output},dst=/wheelhouse",
            args.image,
        ]
    )
    if result.returncode:
        return result.returncode
    wheels = sorted(output.glob("kogwistar-*.whl"), key=lambda path: path.stat().st_mtime_ns)
    if not wheels:
        raise SystemExit("wheel builder completed without a wheel")
    wheel = wheels[-1]
    source_sha256, source_file_count = _source_sha256()
    report = {
        "schema": "adr015-wheel-build/v1",
        "image": args.image,
        "wheel": str(wheel),
        "wheel_sha256": _sha256(wheel),
        "dockerfile_sha256": _sha256(DOCKERFILE),
        "worker_sha256": _sha256(WORKER),
        "source_identity_helper_sha256": _sha256(SOURCE_IDENTITY),
        "candidate_source_sha256": source_sha256,
        "candidate_source_file_count": source_file_count,
    }
    (output / "build-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
