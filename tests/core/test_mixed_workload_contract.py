from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest

from scripts.mixed_workload import (
    ROOT_NODE_ID,
    build_concurrent_trace,
    build_graph_payloads,
    build_normalized_vector,
    build_read_after_write_trace,
    build_root_graph_payload,
    build_source_text,
    normalized_vector_digest,
    read_concurrent_trace,
    read_trace,
    write_concurrent_trace,
    write_trace,
)
from kogwistar.engine_core.models import Edge, Node
from scripts.mixed_concurrent_benchmark import _run_sample
from scripts.mixed_backend_benchmark import (
    _run_sample as _run_backend_sample,
    _rss_bytes_from_statm,
    _vector_query_indexes,
    run_benchmark as _run_backend_matrix,
)
from kogwistar.engine_core.engine_sqlite import EngineSQLite


def test_linux_statm_rss_uses_resident_pages() -> None:
    assert _rss_bytes_from_statm("100 25 0 0 0 0 0", 4096) == 102400


@pytest.mark.parametrize(
    ("statm", "page_size"),
    [("", 4096), ("100", 4096), ("100 25", 0), ("100 nope", 4096)],
)
def test_linux_statm_rss_rejects_invalid_data(statm: str, page_size: int) -> None:
    with pytest.raises((ValueError, IndexError)):
        _rss_bytes_from_statm(statm, page_size)


def test_read_after_write_trace_is_deterministic_and_checks_visibility() -> None:
    first = build_read_after_write_trace(2)
    second = build_read_after_write_trace(2)

    assert first == second
    assert [step["operation"] for step in first] == [
        "write_record",
        "read_event",
        "read_projection",
    ] * 2
    assert [step["expected_entity_id"] for step in first[:3]] == [
        "mixed-node-00000",
    ] * 3
    assert first[0]["expected_edge_id"] == "mixed-edge-00000"
    assert first[0]["expected_source_id"] == ROOT_NODE_ID


def test_graph_fixture_uses_valid_node_edge_models_and_exact_source_spans() -> None:
    items = 3
    source = build_source_text(items)
    root = Node.model_validate(build_root_graph_payload())
    assert root.id == ROOT_NODE_ID
    for index in range(items):
        node_data, edge_data = build_graph_payloads(
            index, vector_sha256="0" * 64
        )
        node = Node.model_validate(node_data)
        edge = Edge.model_validate(edge_data)
        assert node.id == f"mixed-node-{index:05d}"
        assert edge.id == f"mixed-edge-{index:05d}"
        assert edge.source_ids == [ROOT_NODE_ID if index == 0 else f"mixed-node-{index - 1:05d}"]
        assert edge.target_ids == [node.id]
        for entity in (node, edge):
            span = entity.mentions[0].spans[0]
            assert source[span.start_char : span.end_char] == span.excerpt


def test_workload_trace_round_trips_and_rejects_noncanonical_order(tmp_path) -> None:
    path = tmp_path / "trace.json"
    expected = write_trace(path, 3)
    assert read_trace(path) == expected

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["operations"][0], payload["operations"][1] = (
        payload["operations"][1],
        payload["operations"][0],
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed or not canonical"):
        read_trace(path)


def test_workload_trace_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"schema": "future/v99", "operations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported mixed workload trace schema"):
        read_trace(path)


def test_duplicate_write_is_idempotent_and_missing_read_stays_missing(tmp_path) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()
    first_seq = store.append_entity_event(
        namespace="duplicate_contract",
        event_id="stable-event",
        entity_kind="node",
        entity_id="node-1",
        op="UPSERT",
        payload_json='{"id":"node-1"}',
    )
    duplicate_seq = store.append_entity_event(
        namespace="duplicate_contract",
        event_id="stable-event",
        entity_kind="node",
        entity_id="node-1",
        op="UPSERT",
        payload_json='{"id":"node-1"}',
    )

    assert duplicate_seq == first_seq
    assert len(list(store.iter_entity_events(
        namespace="duplicate_contract", from_seq=1, batch_size=10
    ))) == 1
    assert list(store.iter_entity_events(
        namespace="duplicate_contract", from_seq=2, to_seq=2, batch_size=1
    )) == []


def test_duplicate_graph_bundle_write_keeps_node_edge_pair(tmp_path) -> None:
    store = EngineSQLite(tmp_path)
    store.ensure_initialized()
    node_payload, edge_payload = build_graph_payloads(
        0, vector_sha256="0" * 64
    )

    def write_bundle() -> tuple[int, int]:
        with store.transaction():
            node_seq = store.append_entity_event(
                namespace="graph_bundle_idempotency",
                event_id="stable-node-event",
                entity_kind="node",
                entity_id=node_payload["id"],
                op="UPSERT",
                payload_json=json.dumps(node_payload, sort_keys=True, separators=(",", ":")),
            )
            edge_seq = store.append_entity_event(
                namespace="graph_bundle_idempotency",
                event_id="stable-edge-event",
                entity_kind="edge",
                entity_id=edge_payload["id"],
                op="UPSERT",
                payload_json=json.dumps(edge_payload, sort_keys=True, separators=(",", ":")),
            )
        return node_seq, edge_seq

    first = write_bundle()
    duplicate = write_bundle()
    events = list(
        store.iter_entity_events(
            namespace="graph_bundle_idempotency", from_seq=1, batch_size=10
        )
    )
    assert duplicate == first
    assert [(row[1], row[2]) for row in events] == [
        ("node", node_payload["id"]),
        ("edge", edge_payload["id"]),
    ]


@pytest.mark.parametrize(
    ("mix", "expected_write_counts"),
    [("balanced", [4, 4, 1]), ("read_heavy", [1, 1, 1, 1, 1, 1, 1, 1, 1])],
)
def test_concurrent_trace_has_deterministic_revision_windows(
    tmp_path, mix, expected_write_counts
) -> None:
    phases = build_concurrent_trace(9, mix=mix)
    assert [len(phase["steps"]) for phase in phases] == expected_write_counts
    for phase in phases:
        for step in phase["steps"]:
            assert len(step["reads"]) == (4 if mix == "balanced" else 8)
            for visibility, expected_revision in (("before_commit", step["write_index"]),
                                                   ("after_commit", step["write_index"] + 1)):
                observations = [
                    read for read in step["reads"] if read["visibility"] == visibility
                ]
                assert len(set(read["reader_id"] for read in observations)) == len(observations)
                assert all(
                    read["minimum_revision"] == read["maximum_revision"] == expected_revision
                    for read in observations
                )

    path = tmp_path / f"{mix}.json"
    expected = write_concurrent_trace(path, 9, mix=mix)
    actual_mix, actual = read_concurrent_trace(path)
    assert actual_mix == mix
    assert actual == expected


def test_concurrent_trace_rejects_changed_visibility_window(tmp_path) -> None:
    path = tmp_path / "trace.json"
    write_concurrent_trace(path, 5, mix="balanced")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phases"][0]["steps"][0]["reads"][0]["minimum_revision"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed or not canonical"):
        read_concurrent_trace(path)


@pytest.mark.parametrize("mix", ["balanced", "read_heavy"])
def test_concurrent_fixture_trace_digest_is_stable(tmp_path, mix) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "benchmarks"
        / "mixed_rewrite_fixture.v3.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    trace_path = tmp_path / f"concurrent-{mix}.json"
    write_concurrent_trace(trace_path, 8, mix=mix)
    digest_key = f"concurrent_{mix}_trace_sha256"
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == fixture["expected"][digest_key]


def test_vector_query_schedule_is_deterministic_bounded_and_includes_edges() -> None:
    assert _vector_query_indexes(8, 100) == list(range(8))
    selected = _vector_query_indexes(10_000, 100)
    assert len(selected) == 100
    assert selected[0] == 0
    assert selected[-1] == 9_999
    assert selected == sorted(set(selected))
    assert selected == _vector_query_indexes(10_000, 100)


@pytest.mark.parametrize(
    ("mode", "mix"),
    [
        ("python", "balanced"),
        ("python", "read_heavy"),
        ("rust", "balanced"),
        ("rust", "read_heavy"),
    ],
)
def test_concurrent_sqlite_sample_preserves_committed_visibility(
    tmp_path, mode, mix
) -> None:
    phases = build_concurrent_trace(8, mix=mix)
    result = _run_sample(
        tmp_path / f"{mode}-{mix}.sqlite3",
        mode=mode,
        mix=mix,
        phases=phases,
    )
    expected_writes = sum(len(phase["steps"]) for phase in phases)
    expected_reads = sum(
        len(step["reads"])
        for phase in phases
        for step in phase["steps"]
    )
    assert result["operations"] == expected_writes + expected_reads
    assert result["state_digest"]
    assert result["per_operation"]["write"]["count"] == expected_writes
    assert result["per_operation"]["read"]["count"] == expected_reads
    assert result["elapsed_ms"] > 0


def test_concurrent_sample_rejects_stale_revision_window(tmp_path) -> None:
    phases = build_concurrent_trace(1, mix="read_heavy")
    phases[0]["steps"][0]["reads"][0]["minimum_revision"] = 1
    phases[0]["steps"][0]["reads"][0]["maximum_revision"] = 0

    with pytest.raises(RuntimeError, match="concurrent workload worker failed"):
        _run_sample(
            tmp_path / "stale.sqlite3",
            mode="python",
            mix="read_heavy",
            phases=phases,
        )


def test_seeded_vector_fixture_is_reproducible_and_normalized() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "benchmarks"
        / "mixed_rewrite_fixture.v3.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    vectors = [build_normalized_vector(index) for index in range(8)]
    vector_digests = [normalized_vector_digest(vector) for vector in vectors]

    assert [row["vector_sha256"] for row in fixture["records"]] == vector_digests
    assert all(math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, abs_tol=1e-6)
               for vector in vectors)
    actual_set_digest = hashlib.sha256(
        json.dumps(vector_digests, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual_set_digest == fixture["expected"]["vector_digest_list_sha256"]


def test_small_fixture_expected_trace_and_canonical_state_digest(tmp_path) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "benchmarks"
        / "mixed_rewrite_fixture.v3.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    trace_path = tmp_path / "trace.json"
    trace = write_trace(trace_path, 8)
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == fixture["expected"][
        "read_after_write_trace_sha256"
    ]
    result = _run_backend_sample(
        tmp_path / "fixture-run",
        backend_name="sqlite_event_projection",
        items=8,
        trace=trace,
        vector_query_indexes=set(range(8)),
        postgres_dsn=None,
        sample_id="fixture",
    )
    assert result["state_digest"] == fixture["expected"][
        "canonical_event_projection_state_sha256"
    ]
    assert result["process_rss_start_bytes"] > 0
    assert result["process_rss_end_bytes"] > 0
    assert result["vector_query_count"] == 0
    assert result["backend_reopen_verified"] is True


def test_repeated_backend_runs_create_fresh_preserved_sample_directories(
    tmp_path: Path,
) -> None:
    output = tmp_path / "reused-output"
    arguments = {
        "items": 2,
        "samples": 2,
        "postgres_dsn": None,
        "backends": ("sqlite_event_projection",),
        "vector_query_indexes": [],
    }
    first = _run_backend_matrix(output, **arguments)
    first_run_root = output / first["run_id"]
    first_sample = first_run_root / "sqlite_event_projection-0" / "app-meta.sqlite3"
    assert first_sample.is_file()

    second = _run_backend_matrix(output, **arguments)
    assert second["run_id"] != first["run_id"]
    assert first_sample.is_file()
    assert (output / second["run_id"] / "sqlite_event_projection-0" / "app-meta.sqlite3").is_file()
