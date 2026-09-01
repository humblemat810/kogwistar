"""Meta-tests preventing backend two-stage coverage drift."""

from __future__ import annotations

import importlib
import inspect

from tests.core.two_stage_case_catalog import TWO_STAGE_COMMON_CASES


BACKEND_TEST_MODULES = {
    "in_memory": "tests.core.test_two_stage_projection_capability",
    "chroma": "tests.core.test_two_stage_chroma",
    "postgres": "tests.pg_sql.test_two_stage_postgres_projection",
    "async": "tests.outbox.test_async_index_job_worker",
}

NATIVE_PARITY_TESTS = {
    "rust_sqlite_meta": (
        "tests.core.test_two_stage_projection_capability",
        "test_rust_sqlite_meta_authority_runs_two_stage_in_memory_profile",
    ),
    "rust_postgres": (
        "tests.pg_sql.test_two_stage_postgres_projection",
        "test_postgres_two_stage_rust_authority_uses_native_two_stage_adapter",
    ),
}


def _cases(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    cases: set[str] = set()
    for name, test in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            cases.update(
                str(case_id)
                for case_id in getattr(test, "__two_stage_cases__", ())
            )
    return cases


def _function_cases(module_name: str, function_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    test = getattr(module, function_name)
    return set(str(case_id) for case_id in getattr(test, "__two_stage_cases__", ()))


def test_declared_durable_backends_have_common_two_stage_case_coverage() -> None:
    missing = {
        backend: sorted(TWO_STAGE_COMMON_CASES - _cases(module))
        for backend, module in BACKEND_TEST_MODULES.items()
    }
    missing = {backend: cases for backend, cases in missing.items() if cases}
    assert not missing, f"two-stage parity cases missing: {missing}"


def test_native_parity_tests_declare_every_common_case() -> None:
    missing = {
        backend: sorted(TWO_STAGE_COMMON_CASES - _function_cases(*target))
        for backend, target in NATIVE_PARITY_TESTS.items()
    }
    missing = {backend: cases for backend, cases in missing.items() if cases}
    assert not missing, f"native two-stage parity cases missing: {missing}"
