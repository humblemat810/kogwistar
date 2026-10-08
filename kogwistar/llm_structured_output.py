from __future__ import annotations

from typing import TypeVar

from .llm_tasks.providers import (
    StructuredModelLike,
    StructuredOutputRunnable,
    SupportsStructuredOutput,
)


TStructuredModel = TypeVar("TStructuredModel", bound=StructuredModelLike)


def build_structured_output_runnable(
    model: SupportsStructuredOutput,
    schema: type[TStructuredModel],
    *,
    include_raw: bool = True,
    prefer_json_schema: bool = True,
 ) -> StructuredOutputRunnable[TStructuredModel]:
    """Build a structured-output runnable with strict-schema-first fallback."""
    attempts: list[tuple[bool, str | None]] = []
    if prefer_json_schema:
        attempts.append((include_raw, "json_schema"))
    attempts.append((include_raw, "function_calling"))
    attempts.append((include_raw, None))

    last_error: Exception | None = None
    for attempt_include_raw, method in attempts:
        try:
            if method is None:
                return model.with_structured_output(
                    schema, include_raw=attempt_include_raw
                )
            return model.with_structured_output(
                schema, include_raw=attempt_include_raw, method=method
            )
        except (TypeError, ValueError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise TypeError("with_structured_output is unavailable on this model")
