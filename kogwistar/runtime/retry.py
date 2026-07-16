from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")


@dataclass(frozen=True, slots=True)
class RetryAttemptRecord(Generic[TRequest]):
    """Record of a single failed retry attempt."""

    attempt_number: int
    request: TRequest
    error_message: str


@dataclass(frozen=True, slots=True)
class RetryResult(Generic[TResponse]):
    """Successful retry outcome together with prior failed attempts."""

    value: TResponse
    attempts: tuple[RetryAttemptRecord[Any], ...] = ()

    @property
    def retry_count(self) -> int:
        return len(self.attempts)


class RetryExhaustedError(RuntimeError):
    """Raised when a retry loop exhausts its configured attempts."""

    def __init__(
        self,
        *,
        attempts: tuple[RetryAttemptRecord[Any], ...],
        last_error: str,
        retry_budget: int,
    ) -> None:
        super().__init__(last_error)
        self.attempts = attempts
        self.last_error = last_error
        self.retry_budget = retry_budget


def retry_with_context(
    *,
    max_attempts: int,
    build_request: Callable[[int, str | None], TRequest],
    invoke: Callable[[TRequest], TResponse],
    validate: Callable[[TResponse], str | None],
    on_retry: Callable[[RetryAttemptRecord[TRequest]], None] | None = None,
    error_formatter: Callable[[BaseException], str] = repr,
) -> RetryResult[TResponse]:
    """Retry a structured call while carrying the previous error into the next request.

    The helper stays provider and adapter agnostic: the caller owns request
    construction, invocation, and semantic validation.
    """

    budget = max(1, int(max_attempts or 0))
    attempts: list[RetryAttemptRecord[TRequest]] = []
    previous_error: str | None = None

    for attempt_number in range(1, budget + 1):
        request = build_request(attempt_number, previous_error)
        try:
            response = invoke(request)
            validation_reason = validate(response)
            if validation_reason:
                raise ValueError(validation_reason)
            return RetryResult(value=response, attempts=tuple(attempts))
        except Exception as exc:
            error_message = error_formatter(exc)
            record = RetryAttemptRecord(
                attempt_number=attempt_number,
                request=request,
                error_message=error_message,
            )
            attempts.append(record)
            previous_error = error_message
            if callable(on_retry):
                on_retry(record)
            python_decision = {
                "retry_budget": budget,
                "attempt_number": attempt_number,
                "should_retry": attempt_number < budget,
                "exhausted": attempt_number >= budget,
                "next_attempt_number": (
                    attempt_number + 1 if attempt_number < budget else None
                ),
            }
            from kogwistar._rust_bridge import runtime_decide_retry

            decision = runtime_decide_retry(
                payload={
                    "retry_budget": budget,
                    "attempt_number": attempt_number,
                },
                python_value=python_decision,
            )
            if decision["exhausted"]:
                raise RetryExhaustedError(
                    attempts=tuple(attempts),
                    last_error=error_message,
                    retry_budget=budget,
                ) from exc

    raise RetryExhaustedError(
        attempts=tuple(attempts),
        last_error=previous_error or "retry loop exhausted without attempts",
        retry_budget=budget,
    )
