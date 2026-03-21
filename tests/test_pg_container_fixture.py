from __future__ import annotations

import builtins
import os
import types
from unittest.mock import MagicMock

import pytest
import tests.conftest as test_conf

pytestmark = pytest.mark.ci


def test_configure_testcontainers_ryuk_env_defaults_enabled(monkeypatch):
    monkeypatch.delenv("GKE_TEST_PG_DISABLE_RYUK", raising=False)
    monkeypatch.delenv("TESTCONTAINERS_RYUK_DISABLED", raising=False)

    disabled = test_conf._configure_testcontainers_ryuk_env()

    assert disabled is False
    assert "TESTCONTAINERS_RYUK_DISABLED" not in os.environ


def test_configure_testcontainers_ryuk_env_honors_guard(monkeypatch):
    monkeypatch.setenv("GKE_TEST_PG_DISABLE_RYUK", "1")
    monkeypatch.delenv("TESTCONTAINERS_RYUK_DISABLED", raising=False)

    disabled = test_conf._configure_testcontainers_ryuk_env()

    assert disabled is True
    assert os.environ["TESTCONTAINERS_RYUK_DISABLED"] == "true"


def test_is_ryuk_port_mapping_failure_matches_expected_error():
    exc = RuntimeError("Port mapping for container abc and port 8080 is not available")
    assert test_conf._is_ryuk_port_mapping_failure(exc) is True


def test_is_ryuk_port_mapping_failure_ignores_other_errors():
    exc = RuntimeError("docker unavailable")
    assert test_conf._is_ryuk_port_mapping_failure(exc) is False


def test_load_postgres_container_cls_applies_env_before_import(monkeypatch):
    monkeypatch.setenv("GKE_TEST_PG_DISABLE_RYUK", "1")
    monkeypatch.delenv("TESTCONTAINERS_RYUK_DISABLED", raising=False)
    seen: dict[str, str | None] = {}

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "testcontainers.postgres":
            seen["ryuk_disabled"] = os.getenv("TESTCONTAINERS_RYUK_DISABLED")
            module = types.ModuleType("testcontainers.postgres")

            class _FakePostgresContainer:
                pass

            module.PostgresContainer = _FakePostgresContainer
            return module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    cls = test_conf._load_postgres_container_cls()

    assert cls is not None
    assert seen["ryuk_disabled"] == "true"


def test_pg_container_fixture_starts_and_stops_container(monkeypatch):
    fake_pg = MagicMock()
    start_mock = MagicMock(return_value=fake_pg)
    monkeypatch.setenv("GKE_TEST_PG_IMAGE", "postgres:16")
    monkeypatch.setattr(test_conf, "_start_postgres_container", start_mock)
    monkeypatch.setattr(test_conf.logger, "info", MagicMock())
    gen = test_conf.pg_container.__wrapped__()

    yielded = next(gen)
    assert yielded is fake_pg
    start_mock.assert_called_once()
    assert start_mock.call_args.args[0] == "postgres:16"

    gen.close()
    fake_pg.stop.assert_called_once_with()


def test_pg_container_fixture_yields_none_on_start_failure(monkeypatch):
    monkeypatch.setattr(
        test_conf,
        "_start_postgres_container",
        MagicMock(side_effect=RuntimeError("docker unavailable")),
    )
    monkeypatch.setattr(test_conf.logger, "warning", MagicMock())
    gen = test_conf.pg_container.__wrapped__()

    assert next(gen) is None
    try:
        next(gen)
        assert False, "fixture generator should stop after yielding None once"
    except StopIteration:
        pass


def test_pg_container_fixture_retries_without_ryuk_on_ryuk_port_failure(monkeypatch):
    fake_pg = MagicMock()
    start_mock = MagicMock(
        side_effect=[
            RuntimeError(
                "Port mapping for container abc and port 8080 is not available"
            ),
            fake_pg,
        ]
    )

    monkeypatch.delenv("GKE_TEST_PG_DISABLE_RYUK", raising=False)
    monkeypatch.delenv("TESTCONTAINERS_RYUK_DISABLED", raising=False)
    monkeypatch.setattr(test_conf, "_start_postgres_container", start_mock)
    monkeypatch.setattr(test_conf, "_purge_testcontainers_modules", MagicMock())
    monkeypatch.setattr(test_conf.logger, "warning", MagicMock())
    monkeypatch.setattr(test_conf.logger, "info", MagicMock())

    gen = test_conf.pg_container.__wrapped__()
    yielded = next(gen)

    assert yielded is fake_pg
    assert start_mock.call_count == 2
    assert os.environ["TESTCONTAINERS_RYUK_DISABLED"] == "true"

    gen.close()
    fake_pg.stop.assert_called_once_with()


def test_sa_engine_fixture_disposes_engine(monkeypatch):
    fake_engine = MagicMock()
    fake_sa = types.SimpleNamespace(create_engine=MagicMock(return_value=fake_engine))

    monkeypatch.setattr(test_conf, "has_sa", True)
    monkeypatch.setattr(test_conf, "sa", fake_sa)
    gen = test_conf.sa_engine.__wrapped__("postgresql+psycopg://example/test")

    yielded = next(gen)
    assert yielded is fake_engine
    fake_sa.create_engine.assert_called_once_with(
        "postgresql+psycopg://example/test", future=True
    )

    gen.close()
    fake_engine.dispose.assert_called_once_with()
