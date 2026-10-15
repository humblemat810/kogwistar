from pathlib import Path
import tomllib

import kogwistar


def test_public_version_matches_project_metadata() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert kogwistar.__version__ == project["project"]["version"]
