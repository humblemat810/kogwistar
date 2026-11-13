"""Fail closed when a release artifact set is missing a supported wheel."""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
from tarfile import open as tar_open
from zipfile import ZipFile

from packaging.utils import parse_sdist_filename, parse_wheel_filename


def audit_release_artifacts(dist: Path, expected_version: str) -> dict[str, int]:
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise ValueError(f"expected exactly one sdist, found {len(sdists)}")

    sdist_name, sdist_version = parse_sdist_filename(sdists[0].name)
    if sdist_name != "kogwistar" or str(sdist_version) != expected_version:
        raise ValueError(f"unexpected sdist: {sdists[0].name}")
    with tar_open(sdists[0]) as archive:
        if not archive.getnames():
            raise ValueError(f"empty sdist: {sdists[0].name}")

    pure_wheels: list[Path] = []
    native_wheels: list[Path] = []
    for wheel in wheels:
        name, version, _, tags = parse_wheel_filename(wheel.name)
        if name != "kogwistar" or str(version) != expected_version:
            raise ValueError(f"unexpected wheel: {wheel.name}")
        if any(str(tag) == "py3-none-any" for tag in tags):
            pure_wheels.append(wheel)
        else:
            native_wheels.append(wheel)

    if len(native_wheels) != 4:
        raise ValueError(f"expected 4 platform-native wheels, found {len(native_wheels)}")
    if len(pure_wheels) != 1:
        raise ValueError(f"expected exactly one PyPy-compatible pure wheel, found {len(pure_wheels)}")

    pure_wheel = pure_wheels[0]
    with ZipFile(pure_wheel) as archive:
        names = archive.namelist()
        metadata_path = next(
            (name for name in names if name.endswith(".dist-info/METADATA")), None
        )
        if metadata_path is None:
            raise ValueError(f"wheel has no METADATA: {pure_wheel.name}")
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        native_files = [
            name
            for name in names
            if name.startswith("kogwistar/_rust.")
            or name.endswith((".so", ".pyd", ".dll", ".dylib"))
        ]
        if native_files:
            raise ValueError(f"pure wheel contains native modules: {native_files}")
        if metadata["Version"] != expected_version:
            raise ValueError("pure wheel METADATA version does not match release")
        if metadata["Requires-Python"] != ">=3.11":
            raise ValueError(
                "pure wheel must declare Requires-Python >=3.11, got "
                f"{metadata['Requires-Python']!r}"
            )

    return {"sdists": len(sdists), "native_wheels": len(native_wheels), "pure_wheels": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    counts = audit_release_artifacts(args.dist, args.expected_version)
    print(
        f"release artifacts valid: version={args.expected_version} "
        f"sdists={counts['sdists']} native_wheels={counts['native_wheels']} "
        f"pure_wheels={counts['pure_wheels']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
