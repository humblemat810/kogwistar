from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _harness():
    path = ROOT / "scripts" / "rust_port_container_compat.py"
    spec = importlib.util.spec_from_file_location("rust_port_container_compat", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_container_harness_uses_dual_venv_and_cleans_native_bridge() -> None:
    script = _harness()._container_script()

    assert "python -m venv /tmp/core" in script
    assert "python -m venv /tmp/consumer" in script
    assert '"${wheel}[full,test]"' in script
    assert '"${wheel}[test,chroma]"' in script
    assert "fastapi langchain-openai" in script
    assert "trap cleanup EXIT INT TERM" in script
    assert "exec /tmp/core/bin/python" not in script
    assert "--consumer-python /tmp/consumer/bin/python" in script


def test_container_harness_default_image_is_patch_pinned() -> None:
    assert _harness().DEFAULT_IMAGE == "python:3.13.14-slim-bookworm"
