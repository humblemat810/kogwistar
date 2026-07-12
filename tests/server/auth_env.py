from __future__ import annotations

# Compatibility facade retained for tests that import tests.server.auth_env.
from ..auth_env import (
    TEST_JWT_ALG as TEST_JWT_ALG,  # noqa: F401
    TEST_JWT_SECRET as TEST_JWT_SECRET,  # noqa: F401
    ensure_test_jwt_env as ensure_test_jwt_env,  # noqa: F401
)
