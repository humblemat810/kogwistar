import math
import tomllib
from pathlib import Path

import pytest

from kogwistar.engine_core.embedding_factory import _l2_normalize


@pytest.mark.ci
def test_numpy_is_not_a_base_dependency() -> None:
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]

    assert not any(str(requirement).lower().startswith("numpy") for requirement in metadata["dependencies"])
    assert any(str(requirement).lower().startswith("numpy") for requirement in metadata["optional-dependencies"]["test"])


@pytest.mark.ci
def test_l2_normalize_preserves_zero_and_unit_vectors() -> None:
    result = _l2_normalize([[0.0, 0.0], [3.0, 4.0]])

    assert result[0] == [0.0, 0.0]
    assert result[1] == pytest.approx([0.6, 0.8])


@pytest.mark.ci
def test_l2_normalize_scales_extreme_finite_values() -> None:
    result = _l2_normalize([[1.0e308, -1.0e308, 1.0e-308]])

    assert result[0] == pytest.approx(
        [math.sqrt(0.5), -math.sqrt(0.5), 0.0], abs=1.0e-12
    )


@pytest.mark.ci
def test_l2_normalize_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _l2_normalize([[1.0, math.inf]])
