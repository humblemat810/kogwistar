from __future__ import annotations

import pytest

from tests.server.oidc_test_support import (
    OIDC_TEST_IDENTITY,
    docker_available,
    oidc_seed_json,
    start_keycloak_container,
    stop_keycloak_container,
)


@pytest.fixture(scope="module")
def oidc_test_identity() -> dict[str, str]:
    return {
        **OIDC_TEST_IDENTITY,
        "seed_json": oidc_seed_json(),
    }


@pytest.fixture(scope="module")
def keycloak_container() -> dict[str, str]:
    if not docker_available():
        pytest.skip("docker is not available; skipping Keycloak OIDC test")

    config = start_keycloak_container()
    try:
        yield config
    finally:
        stop_keycloak_container(config["container_id"])
