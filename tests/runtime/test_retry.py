from __future__ import annotations

import pytest

from kogwistar.runtime.retry import RetryExhaustedError, retry_with_context

pytestmark = [pytest.mark.ci, pytest.mark.runtime]


def test_retry_with_context_carries_previous_error_into_next_attempt() -> None:
    seen_requests: list[tuple[int, str | None]] = []
    retry_events: list[tuple[int, str]] = []

    def build_request(attempt_number: int, previous_error: str | None) -> dict[str, str | None]:
        seen_requests.append((attempt_number, previous_error))
        return {"attempt": str(attempt_number), "previous_error": previous_error}

    def invoke(request: dict[str, str | None]) -> str:
        if request["attempt"] == "1":
            raise RuntimeError("boom")
        return "ok"

    def validate(response: str) -> str | None:
        return None if response == "ok" else "unexpected response"

    def on_retry(record) -> None:
        retry_events.append((record.attempt_number, record.error_message))

    result = retry_with_context(
        max_attempts=2,
        build_request=build_request,
        invoke=invoke,
        validate=validate,
        on_retry=on_retry,
    )

    assert result.value == "ok"
    assert result.retry_count == 1
    assert seen_requests == [(1, None), (2, "RuntimeError('boom')")]
    assert retry_events == [(1, "RuntimeError('boom')")]


def test_retry_with_context_raises_when_budget_is_exhausted() -> None:
    def build_request(attempt_number: int, previous_error: str | None) -> tuple[int, str | None]:
        return attempt_number, previous_error

    def invoke(request: tuple[int, str | None]) -> str:
        raise ValueError(f"failed on attempt {request[0]}")

    def validate(response: str) -> str | None:
        return None

    with pytest.raises(RetryExhaustedError) as exc_info:
        retry_with_context(
            max_attempts=2,
            build_request=build_request,
            invoke=invoke,
            validate=validate,
        )

    assert exc_info.value.retry_budget == 2
    assert exc_info.value.last_error == "ValueError('failed on attempt 2')"
    assert len(exc_info.value.attempts) == 2


@pytest.mark.parametrize("mode", ["shadow", "rust"])
def test_retry_native_policy_does_not_double_invoke_callbacks(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("KOGWISTAR_IMPL_RUNTIME", mode)
    invocations: list[int] = []
    retries: list[int] = []

    def invoke(request: int) -> str:
        invocations.append(request)
        if request == 1:
            raise RuntimeError("once")
        return "ok"

    result = retry_with_context(
        max_attempts=2,
        build_request=lambda attempt, _previous: attempt,
        invoke=invoke,
        validate=lambda _response: None,
        on_retry=lambda record: retries.append(record.attempt_number),
    )

    assert result.value == "ok"
    assert invocations == [1, 2]
    assert retries == [1]
