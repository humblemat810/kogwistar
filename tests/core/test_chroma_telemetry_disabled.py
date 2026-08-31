from __future__ import annotations

import os

import pytest


pytestmark = pytest.mark.unit


def test_test_harness_forces_chroma_product_telemetry_off() -> None:
    """Guard against PostHog being re-enabled by dotenv or a fixture."""
    assert os.environ.get("ANONYMIZED_TELEMETRY", "").upper() == "FALSE"


def test_chroma_settings_are_explicitly_non_telemetry() -> None:
    chromadb = pytest.importorskip("chromadb")
    settings = chromadb.config.Settings(anonymized_telemetry=False)
    assert settings.anonymized_telemetry is False


def test_posthog_is_disabled_and_network_api_is_mocked() -> None:
    posthog = pytest.importorskip("posthog")
    from unittest.mock import MagicMock

    assert posthog.disabled is True
    assert isinstance(posthog.capture, MagicMock)
    assert posthog.capture("test-event") is None
