from __future__ import annotations

import hashlib
import os
from typing import Any

import pytest
from joblib import Memory
from pydantic import BaseModel, Field, create_model

pytestmark = [pytest.mark.integration]


def _build_non_recursive_nested_schema(depth: int) -> type[BaseModel]:
    """
    Build a strictly non-recursive schema:
      LevelN -> child: LevelN-1 -> ... -> child: Leaf(value: str)
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")

    current: type[BaseModel] = create_model(
        f"DepthProbeLeafD{depth}",
        value=(str, Field(..., description="Leaf value marker")),
    )
    for level in range(depth, 0, -1):
        current = create_model(
            f"DepthProbeLevelD{depth}L{level}",
            child=(current, Field(..., description="Nested child object")),
        )
    return current


def _invoke_structured(llm: Any, schema: type[BaseModel], prompt: str) -> tuple[bool, str | None]:
    try:
        runnable = llm.with_structured_output(schema, method="json_schema", include_raw=True)
    except TypeError:
        # Older langchain-google-genai versions may not expose `method`.
        runnable = llm.with_structured_output(schema, include_raw=True)

    try:
        result = runnable.invoke(prompt)
    except Exception as exc:
        return False, f"invoke exception: {type(exc).__name__}: {exc}"

    if isinstance(result, dict):
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
        if parsed is not None and parsing_error is None:
            return True, None
        if parsing_error is not None:
            return False, f"parsing_error: {parsing_error}"
        return False, f"empty parsed result payload: {result}"

    # include_raw=False shape
    if result is None:
        return False, "result is None"
    return True, None


def _probe_depth(llm: Any, depth: int, attempts: int) -> tuple[bool, str | None]:
    schema = _build_non_recursive_nested_schema(depth)
    prompt = (
        "Return a valid object for the provided schema. "
        "Fill all nested `child` objects and set final leaf `value` to 'ok'."
    )
    last_reason: str | None = None
    for _ in range(max(1, attempts)):
        ok, reason = _invoke_structured(llm, schema, prompt)
        if ok:
            return True, None
        last_reason = reason
    return False, last_reason


def _probe_depth_live(
    *,
    model_name: str,
    depth: int,
    attempts: int,
    api_key_fingerprint: str,
    cache_version: str = "v1",
) -> tuple[bool, str | None]:
    # `api_key_fingerprint` and `cache_version` are included for cache key stability / invalidation.
    _ = api_key_fingerprint
    _ = cache_version
    genai = pytest.importorskip("langchain_google_genai")
    ChatGoogleGenerativeAI = genai.ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    return _probe_depth(llm, depth=depth, attempts=attempts)


@pytest.mark.manual
def test_gemini_detect_max_non_recursive_schema_nesting_depth():
    """
    Detect the maximum nesting depth Gemini can handle for non-recursive schemas.

    Optional knobs:
      TEST_GEMINI_MODEL=gemini-2.5-flash
      GEMINI_SCHEMA_DEPTH_HARD_CAP=64
      GEMINI_SCHEMA_DEPTH_PROBE_ATTEMPTS=2
      GEMINI_SCHEMA_DEPTH_CACHE_DIR=.joblib/gemini_schema_depth_limit
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    os.environ.setdefault("GOOGLE_API_KEY", api_key)

    pytest.importorskip("langchain_google_genai")

    model_name = os.getenv("TEST_GEMINI_MODEL", "gemini-3-flash-preview")
    hard_cap = int(os.getenv("GEMINI_SCHEMA_DEPTH_HARD_CAP", "64"))
    attempts = int(os.getenv("GEMINI_SCHEMA_DEPTH_PROBE_ATTEMPTS", "2"))
    cache_dir = os.getenv("GEMINI_SCHEMA_DEPTH_CACHE_DIR", ".joblib/gemini_schema_depth_limit")
    api_key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    memory = Memory(location=cache_dir, verbose=0)
    cached_probe_depth = memory.cache(_probe_depth_live)

    # Binary search under monotonic assumption:
    # if depth d fails, deeper schemas are also expected to fail.
    probe_cache: dict[int, tuple[bool, str | None]] = {}

    def supports(depth: int) -> bool:
        if depth <= 0:
            return True
        if depth not in probe_cache:
            probe_cache[depth] = cached_probe_depth(
                model_name=model_name,
                depth=depth,
                attempts=attempts,
                api_key_fingerprint=api_key_fingerprint,
            )
        return probe_cache[depth][0]

    low = 0
    high = hard_cap + 1  # sentinel "known failing/out-of-scope" boundary
    while high - low > 1:
        mid = (low + high) // 2
        if supports(mid):
            low = mid
        else:
            high = mid

    highest_supported = low
    first_failing = high if high <= hard_cap else None
    failure_reason = probe_cache.get(first_failing, (None, None))[1] if first_failing is not None else None

    print(
        "Gemini non-recursive schema nesting depth probe: "
        f"model={model_name}, highest_supported={highest_supported}, "
        f"first_failing={first_failing}, hard_cap={hard_cap}, failure_reason={failure_reason}"
    )

    assert highest_supported >= 1, (
        f"Gemini failed even depth=1 non-recursive schema; reason={failure_reason}"
    )
