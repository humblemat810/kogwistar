from __future__ import annotations

import asyncio
import json

import pytest

from kogwistar._rust_bridge import (
    RustParityError,
    RustBridgeTrace,
    _TracingRustExtension,
    api_health,
    capture_rust_bridge_calls,
    graph_model_capability_profiles,
    project_graph_model_slice,
    project_graph_model_slices,
    store_postgres,
)

pytestmark = [pytest.mark.ci, pytest.mark.core]


class _ExtensionStub:
    def api_health(self, payload: str) -> str:
        return json.dumps({"ok": bool(json.loads(payload))})

    def project_graph_model_slice(self, payload: str) -> str:
        return payload

    def project_graph_model_slices(self, payload: str) -> str:
        return payload

    def store_postgres_json(self, payload: str) -> str:
        return "null"


def test_bridge_trace_counts_keyword_payload_and_transaction_metadata() -> None:
    class KeywordExtension:
        def store_postgres_json(self, *, payload: str) -> str:
            return "null"

    request = json.dumps(
        {
            "transaction_id": "private-transaction-id",
            "operation": {"kind": "commit_transaction"},
        },
        separators=(",", ":"),
    )
    trace = RustBridgeTrace()
    extension = _TracingRustExtension(KeywordExtension(), (trace,))

    assert extension.store_postgres_json(payload=request) == "null"
    metrics = trace.snapshot()
    assert metrics["ffi_calls"] == 1
    assert metrics["input_bytes"] == len(request.encode("utf-8"))
    assert metrics["transaction_active_calls"] == 1
    assert metrics["transaction_boundary_calls"] == 1
    assert metrics["calls_by_operation"] == {"commit_transaction": 1}
    assert "private-transaction-id" not in repr(metrics)


def test_bridge_trace_counts_per_item_and_batched_native_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _ExtensionStub()
    monkeypatch.setattr("kogwistar._rust_bridge.import_module", lambda _name: extension)
    payloads = [{"item": index} for index in range(4)]

    with capture_rust_bridge_calls() as individual:
        individual_results = [project_graph_model_slice(item) for item in payloads]
    with capture_rust_bridge_calls() as batched:
        batched_results = project_graph_model_slices(payloads)

    assert individual_results == batched_results == payloads
    assert individual.snapshot()["ffi_calls"] == 4
    assert batched.snapshot()["ffi_calls"] == 1
    assert individual.snapshot()["calls_by_method"] == {"project_graph_model_slice": 4}
    assert batched.snapshot()["calls_by_method"] == {"project_graph_model_slices": 1}
    assert individual.snapshot()["input_bytes"] > 0
    assert batched.snapshot()["output_bytes"] > 0


def test_nested_bridge_traces_aggregate_calls_and_restore_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _ExtensionStub()
    monkeypatch.setattr("kogwistar._rust_bridge.import_module", lambda _name: extension)
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")

    with capture_rust_bridge_calls() as outer:
        api_health(payload={"scope": "outer-before"}, python_value={})
        with pytest.raises(RuntimeError, match="inner scope ended"):
            with capture_rust_bridge_calls() as inner:
                api_health(payload={"scope": "inner"}, python_value={})
                raise RuntimeError("inner scope ended")
        api_health(payload={"scope": "outer-after"}, python_value={})

    assert outer.snapshot()["ffi_calls"] == 3
    assert outer.snapshot()["calls_by_method"] == {"api_health": 3}
    assert inner.snapshot()["ffi_calls"] == 1
    assert inner.snapshot()["calls_by_method"] == {"api_health": 1}


def test_bridge_trace_preserves_python_mode_and_transaction_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = _ExtensionStub()
    monkeypatch.setattr("kogwistar._rust_bridge.import_module", lambda _name: extension)
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "python")
    with capture_rust_bridge_calls() as python_trace:
        assert api_health(payload={"request": 1}, python_value={"ok": "python"}) == {
            "ok": "python"
        }
    assert python_trace.snapshot()["ffi_calls"] == 0

    with capture_rust_bridge_calls() as rust_trace:
        assert store_postgres(
            dsn="postgresql://local/test",
            schema="test",
            operation={"kind": "begin_transaction"},
        ) is None
        assert store_postgres(
            dsn="postgresql://local/test",
            schema="test",
            transaction_id="test-transaction",
            operation={"kind": "read_graph"},
        ) is None
    metrics = rust_trace.snapshot()
    assert metrics["ffi_calls"] == 2
    assert metrics["transaction_boundary_calls"] == 1
    assert metrics["transaction_active_calls"] == 1
    assert "postgresql://local/test" not in repr(metrics)


def test_postgres_graph_mutation_bridge_preserves_unbounded_integers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingExtension(_ExtensionStub):
        payload: str | None = None

        def store_postgres_json(self, payload: str) -> str:
            self.payload = payload
            return "null"

    extension = RecordingExtension()
    monkeypatch.setattr("kogwistar._rust_bridge.import_module", lambda _name: extension)
    value = 1 << 200

    assert store_postgres(
        dsn="postgresql://local/test",
        schema="test",
        operation={
            "kind": "graph_mutation",
            "record": {"id": "n1", "metadata": {"level_from_root": value}},
        },
    ) is None

    assert extension.payload is not None
    operation = json.loads(extension.payload)["operation"]
    assert operation["record"]["metadata"]["level_from_root"] == value

    with pytest.raises(RustParityError):
        store_postgres(
            dsn="postgresql://local/test",
            schema="test",
            operation={"kind": "read_graph", "limit": value},
        )
    # Rejected generic payloads must not cross the native boundary.
    assert json.loads(extension.payload)["operation"]["kind"] == "graph_mutation"


def test_bridge_trace_is_local_to_each_async_task(monkeypatch: pytest.MonkeyPatch) -> None:
    extension = _ExtensionStub()
    monkeypatch.setattr("kogwistar._rust_bridge.import_module", lambda _name: extension)
    monkeypatch.setenv("KOGWISTAR_IMPL_SERVER", "rust")

    async def run_calls(count: int) -> int:
        with capture_rust_bridge_calls() as trace:
            for index in range(count):
                await asyncio.sleep(0)
                result = await asyncio.to_thread(
                    api_health, payload={"index": index}, python_value={}
                )
                assert result == {"ok": True}
        return trace.ffi_calls

    async def concurrent_calls() -> list[int]:
        return list(await asyncio.gather(run_calls(2), run_calls(3)))

    assert asyncio.run(concurrent_calls()) == [2, 3]


@pytest.mark.ci_full
def test_bridge_trace_counts_real_extension_call_when_installed() -> None:
    pytest.importorskip("kogwistar._rust")
    with capture_rust_bridge_calls() as trace:
        assert graph_model_capability_profiles()
    assert trace.snapshot()["calls_by_method"] == {"graph_model_capability_profiles": 1}


@pytest.mark.ci_full
def test_graph_slice_benchmark_reports_actual_crossing_budget() -> None:
    pytest.importorskip("kogwistar._rust")
    from scripts.graph_slice_seam_benchmark import run

    report = run(model_count=1, iterations=1, field_mode="llm_in")
    crossings = report["bridge_crossings"]
    assert crossings["measurement"] == "untimed_after_latency_samples"
    assert crossings["per_envelope"]["ffi_calls"] == report["envelope_count"] == 2
    assert crossings["batch"]["ffi_calls"] == 1
    assert crossings["batch"]["input_bytes"] > 0
    assert report["bridge_projection_digest"] == report["bridge_batch_projection_digest"]
    assert "python_slice_envelope_construction" in report["stages"]
