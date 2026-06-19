from __future__ import annotations

from types import SimpleNamespace

from kogwistar.ingester import BaseDocumentGraphIngestor
from kogwistar.llm_structured_output import build_structured_output_runnable


class _FakeRunnable:
    def invoke(self, payload):
        return payload


class _FunctionCallingOnlyModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def with_structured_output(self, schema, include_raw: bool = True, **kwargs):
        self.calls.append({"schema": schema.__name__, "include_raw": include_raw, **kwargs})
        if "method" in kwargs and kwargs["method"] == "json_schema":
            raise TypeError("json_schema unsupported")
        return _FakeRunnable()


def test_structured_output_helper_falls_back_to_function_calling() -> None:
    model = _FunctionCallingOnlyModel()

    runnable = build_structured_output_runnable(
        model,
        schema=SimpleNamespace(__name__="Schema"),
        include_raw=True,
        prefer_json_schema=True,
    )

    assert isinstance(runnable, _FakeRunnable)
    assert [call.get("method") for call in model.calls] == ["json_schema", "function_calling"]


def test_ingester_structured_output_chains_use_safe_wrapper(tmp_path) -> None:
    model = _FunctionCallingOnlyModel()
    engine = SimpleNamespace(persist_directory=str(tmp_path / "engine"))

    ingestor = BaseDocumentGraphIngestor(engine=engine, llm=model, cache_dir=str(tmp_path / "cache"))

    assert ingestor._coerce_summarized_one_chain is not None
    assert ingestor._summarize_chain is not None
    assert ingestor._group_chain is not None
    assert [call.get("method") for call in model.calls] == [
        "json_schema",
        "function_calling",
        "json_schema",
        "function_calling",
        "json_schema",
        "function_calling",
    ]
