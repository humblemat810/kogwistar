from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rust_port_compat.py"
MANIFEST = json.loads(
    (ROOT / "contracts" / "rust-port-v1.json").read_text(encoding="utf-8")
)

pytestmark = pytest.mark.regression


def _runner():
    spec = importlib.util.spec_from_file_location("rust_port_compat", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capability_modes_preserve_legacy_shadow_without_shadow_server() -> None:
    runner = _runner()

    modes = runner._capability_modes(
        manifest=MANIFEST,
        implementation_mode="shadow",
        overrides={key: None for key in runner._capability_configurations(MANIFEST)},
    )

    assert modes == {
        "KOGWISTAR_IMPL_CONTRACTS": "shadow",
        "KOGWISTAR_IMPL_META_STORE": "shadow",
        "KOGWISTAR_IMPL_GRAPH_STORE": "shadow",
        "KOGWISTAR_IMPL_RUNTIME": "shadow",
        "KOGWISTAR_IMPL_SERVER": "python",
    }


def test_capability_overrides_select_one_writer_per_manifest_capability() -> None:
    runner = _runner()
    configurations = runner._capability_configurations(MANIFEST)
    modes = runner._capability_modes(
        manifest=MANIFEST,
        implementation_mode="python",
        overrides={key: "rust" for key in configurations},
    )

    writers = runner._active_writers(MANIFEST, modes)

    assert set(writers) == {
        item["capability"] for item in MANIFEST["capability_ownership"]
    }
    assert writers["deterministic-contracts"] == "rust"
    assert writers["sqlite-meta"] == "rust"
    assert writers["postgres-sequence-event-log"] == "python"
    assert writers["projections-snapshots-run-registry"] == "python"
    assert writers["queues-leases-lanes"] == "python"
    assert writers["graph-pgvector"] == "python"
    assert writers["chroma-adapter"] == "python"
    assert writers["workflow-runtime"] == "python"
    assert writers["server-rest-sse-mcp-cli"] == "python"


def test_active_writers_only_cut_over_when_manifest_marks_rust_ready() -> None:
    runner = _runner()
    manifest = json.loads(json.dumps(MANIFEST))
    ownership = next(
        item
        for item in manifest["capability_ownership"]
        if item["capability"] == "workflow-runtime"
    )
    modes = runner._capability_modes(
        manifest=manifest,
        implementation_mode="rust",
        overrides={key: None for key in runner._capability_configurations(manifest)},
    )

    assert runner._active_writers(manifest, modes)["workflow-runtime"] == "python"
    ownership["rust_cutover_ready"] = True
    assert runner._active_writers(manifest, modes)["workflow-runtime"] == "rust"


def test_missing_cutover_readiness_is_fail_closed() -> None:
    runner = _runner()
    manifest = json.loads(json.dumps(MANIFEST))
    ownership = next(
        item
        for item in manifest["capability_ownership"]
        if item["capability"] == "deterministic-contracts"
    )
    ownership.pop("rust_cutover_ready")
    modes = runner._capability_modes(
        manifest=manifest,
        implementation_mode="rust",
        overrides={key: None for key in runner._capability_configurations(manifest)},
    )

    assert (
        runner._active_writers(manifest, modes)["deterministic-contracts"] == "python"
    )


def test_environment_exports_global_and_coarse_capability_modes(tmp_path: Path) -> None:
    runner = _runner()
    modes = {
        "KOGWISTAR_IMPL_CONTRACTS": "rust",
        "KOGWISTAR_IMPL_META_STORE": "shadow",
        "KOGWISTAR_IMPL_GRAPH_STORE": "python",
        "KOGWISTAR_IMPL_RUNTIME": "rust",
        "KOGWISTAR_IMPL_SERVER": "python",
    }

    env = runner._environment(tmp_path, "shadow", modes)

    assert env["KOGWISTAR_IMPL_MODE"] == "shadow"
    assert {key: env[key] for key in modes} == modes


def test_server_override_rejects_shadow() -> None:
    runner = _runner()
    configurations = runner._capability_configurations(MANIFEST)
    overrides = {key: None for key in configurations}
    overrides["KOGWISTAR_IMPL_SERVER"] = "shadow"

    with pytest.raises(ValueError, match="KOGWISTAR_IMPL_SERVER"):
        runner._capability_modes(
            manifest=MANIFEST,
            implementation_mode="python",
            overrides=overrides,
        )


def test_postgres_authority_selector_is_explicit_and_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kogwistar._rust_bridge import postgres_authority_implementation_mode

    monkeypatch.delenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", raising=False)
    assert postgres_authority_implementation_mode() == "python"
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "rust")
    assert postgres_authority_implementation_mode() == "rust"
    monkeypatch.setenv("KOGWISTAR_IMPL_POSTGRES_AUTHORITY", "invalid")
    with pytest.raises(ValueError, match="KOGWISTAR_IMPL_POSTGRES_AUTHORITY"):
        postgres_authority_implementation_mode()


def test_identity_fingerprint_changes_with_capability_ownership() -> None:
    runner = _runner()
    base = {
        "implementation_mode": "python",
        "capability_modes": {"KOGWISTAR_IMPL_CONTRACTS": "python"},
        "active_writers": {"deterministic-contracts": "python"},
    }
    changed = {
        **base,
        "capability_modes": {"KOGWISTAR_IMPL_CONTRACTS": "rust"},
        "active_writers": {"deterministic-contracts": "rust"},
    }

    assert runner._identity_fingerprint(base) != runner._identity_fingerprint(changed)


def test_identity_fingerprint_changes_with_python_runtime() -> None:
    runner = _runner()
    base = {"python_version": "3.12.0", "python_abi": "cpython-312-x86_64-linux-gnu"}
    changed = {"python_version": "3.13.0", "python_abi": "cpython-313-x86_64-linux-gnu"}

    assert runner._identity_fingerprint(base) != runner._identity_fingerprint(changed)


def test_candidate_identity_ignores_job_local_paths_but_keeps_runtime_abi() -> None:
    runner = _runner()
    base = {
        "candidate_source_sha256": "a" * 64,
        "python_abi": "cpython-313-x86_64-linux-gnu",
        "python_executable": "/tmp/job-a/bin/python",
        "rust_extension_file": "/tmp/job-a/kogwistar/_rust.abi3.so",
        "layer_interpreters": {"core": "/tmp/job-a/bin/python"},
    }
    moved = {
        **base,
        "python_executable": "/opt/job-b/bin/python",
        "rust_extension_file": "/opt/job-b/kogwistar/_rust.abi3.so",
        "layer_interpreters": {"core": "/opt/job-b/bin/python"},
    }
    changed_abi = {**moved, "python_abi": "cpython-312-x86_64-linux-gnu"}

    assert runner._candidate_identity_fingerprint(
        base
    ) == runner._candidate_identity_fingerprint(moved)
    assert runner._candidate_identity_fingerprint(
        base
    ) != runner._candidate_identity_fingerprint(changed_abi)


def test_verification_harness_fingerprint_changes_with_runner_or_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    application = tmp_path / "app"
    application.mkdir()
    config = application / "pytest.ini"
    config.write_text("[pytest]\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path / "candidate")
    runner.ROOT.mkdir()
    (runner.ROOT / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    first = runner._verification_harness_fingerprint(application)
    config.write_text("[pytest]\naddopts = -q\n", encoding="utf-8")
    second = runner._verification_harness_fingerprint(application)

    assert first != second


def test_outer_harness_commit_provenance_overrides_missing_stage_git() -> None:
    runner = _runner()

    assert runner._resolved_commit_provenance(
        {"core_commit": None, "application_commit": "detected"},
        {"core_commit": "a" * 40, "application_commit": None},
    ) == {
        "core_commit": "a" * 40,
        "application_commit": "detected",
    }


def test_suite_test_files_skips_generated_test_trees(tmp_path: Path) -> None:
    runner = _runner()
    kept = tmp_path / "tests" / "unit" / "test_kept.py"
    generated = tmp_path / "tests" / "_tmp" / "test_generated.py"
    hidden = tmp_path / "tests" / ".tmp_workflow" / "test_hidden.py"
    for path in (kept, generated, hidden):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    assert runner._suite_test_files(tmp_path / "tests") == [kept]


def test_absolute_python_path_preserves_symlink_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    monkeypatch.chdir(tmp_path)

    assert (
        runner._absolute_path_without_symlink_resolution(Path("venv/bin/python"))
        == tmp_path / "venv" / "bin" / "python"
    )


def test_pytest_option_terminator_is_not_forwarded() -> None:
    runner = _runner()

    assert runner._pytest_args(["--", "-k", "focused"]) == ["-k", "focused"]
    assert runner._pytest_args(["-k", "focused"]) == ["-k", "focused"]


def test_failed_command_output_is_bounded_from_the_tail() -> None:
    runner = _runner()
    value = "prefix-marker\n" + ("x" * runner._FAILURE_OUTPUT_LIMIT) + "tail-marker"

    captured = runner._failure_output(value)

    assert len(captured) < len(value)
    assert "output truncated" in captured
    assert "prefix-marker" not in captured
    assert captured.endswith("tail-marker")


def test_milestone_treats_file_with_no_selected_items_as_covered() -> None:
    runner = _runner()

    assert runner._command_succeeded(layer="core", profile="milestone", returncode=5)
    assert not runner._command_succeeded(layer="core", profile="milestone", returncode=1)


@pytest.mark.parametrize("layer", ["core", "parser", "application"])
def test_milestone_layers_select_only_explicit_ci_gates(layer: str) -> None:
    runner = _runner()

    expression = runner._marker_expression(layer, profile="milestone")

    assert expression.startswith("(ci or ci_full)")
    for excluded in ("manual", "llm_real", "legacy", "requires_ollama"):
        assert f"not {excluded}" in expression


def test_sink_milestone_layer_runs_deterministic_unmarked_suite() -> None:
    runner = _runner()

    expression = runner._marker_expression("sink", profile="milestone")

    assert "ci or ci_full" not in expression
    for excluded in (
        "manual",
        "integration",
        "longrun",
        "external",
        "llm_real",
        "requires_ollama",
    ):
        assert f"not {excluded}" in expression


def test_fast_profile_marker_expressions() -> None:
    runner = _runner()

    feature = runner._marker_expression("core", profile="feature")
    assert feature.startswith("(ci or regression) and not slow")
    regression = runner._marker_expression("core", profile="regression")
    assert regression.startswith("regression and not slow")
    for expression in (feature, regression):
        for excluded in ("manual", "llm_real", "legacy", "requires_ollama"):
            assert f"not {excluded}" in expression
    sink = runner._marker_expression("sink", profile="feature")
    assert "not slow" in sink
    assert "not integration" in sink


def test_release_core_groups_cover_every_test_file_once_and_are_bounded(
    tmp_path: Path,
) -> None:
    runner = _runner()
    for name in (
        "test_z.py",
        "test_a.py",
        "core/test_b.py",
        "runtime/test_c.py",
        "nested/deeper/test_d.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    first = runner._target_groups(
        layer="core", tests_path=tmp_path, profile="milestone"
    )
    second = runner._target_groups(
        layer="core", tests_path=tmp_path, profile="milestone"
    )
    flattened = [path for group in first for path in group]
    expected = sorted(
        tmp_path.rglob("test*.py"),
        key=lambda path: path.relative_to(tmp_path).as_posix(),
    )

    assert first == second
    assert flattened == expected
    assert len(flattened) == len(set(flattened))
    assert all(1 <= len(group) <= runner._RELEASE_GROUP_SIZE for group in first)


def test_release_parser_groups_cover_every_test_file_once(tmp_path: Path) -> None:
    runner = _runner()
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = sorted(
        [tmp_path / "test_a.py", nested / "test_b.py"],
        key=lambda path: path.relative_to(tmp_path).as_posix(),
    )
    for path in expected:
        path.write_text("", encoding="utf-8")

    groups = runner._target_groups(
        layer="parser", tests_path=tmp_path, profile="milestone"
    )

    assert [path for group in groups for path in group] == expected
    assert all(len(group) == 1 for group in groups)
    assert runner._target_groups(
        layer="parser", tests_path=tmp_path, profile="feature"
    ) == [[expected[0]], [expected[1]]]


def test_release_core_groups_split_at_bound_and_normal_core_stays_one_group(
    tmp_path: Path,
) -> None:
    runner = _runner()
    count = 3
    for index in range(count):
        (tmp_path / f"test_{index:02d}.py").write_text("", encoding="utf-8")

    release_groups = runner._target_groups(
        layer="core", tests_path=tmp_path, profile="milestone"
    )
    normal_groups = runner._target_groups(
        layer="core", tests_path=tmp_path, profile="feature"
    )

    assert [len(group) for group in release_groups] == [1, 1, 1]
    assert normal_groups == [[tmp_path]]


@pytest.mark.parametrize(
    ("layer", "profile", "returncode", "expected"),
    [
        ("core", "milestone", 0, True),
        ("core", "milestone", 5, True),
        ("core", "milestone", 1, False),
        ("core", "feature", 5, False),
        ("parser", "milestone", 5, True),
        ("parser", "feature", 5, True),
        ("parser", "milestone", 1, False),
        ("application", "regression", 5, True),
    ],
)
def test_no_selected_tests_is_success_only_for_file_isolated_release_layers(
    layer: str, profile: str, returncode: int, expected: bool
) -> None:
    runner = _runner()

    assert (
        runner._command_succeeded(layer=layer, profile=profile, returncode=returncode)
        is expected
    )


def test_release_core_resume_reuses_only_matching_successful_commands() -> None:
    runner = _runner()
    command = ["python", "test_one.py", "-m", "ci"]
    prior = [
        {"command": command, "returncode": 0, "duration_seconds": 1.0},
        {"command": ["python", "other.py"], "returncode": 1},
    ]

    reused = runner._reusable_run(
        command=command,
        prior_runs=prior,
        layer="core",
        profile="milestone",
    )

    assert reused == {**prior[0], "reused": True}
    assert (
        runner._reusable_run(
            command=["python", "missing.py"],
            prior_runs=prior,
            layer="core",
            profile="milestone",
        )
        is None
    )


def test_release_core_resume_accepts_recorded_code_five_only_in_release() -> None:
    runner = _runner()
    command = ["python", "unmarked.py"]
    prior = [{"command": command, "returncode": 5, "duration_seconds": 0.1}]

    assert (
        runner._reusable_run(
            command=command, prior_runs=prior, layer="core", profile="milestone"
        )
        is not None
    )
    assert (
        runner._reusable_run(
            command=command, prior_runs=prior, layer="core", profile="feature"
        )
        is None
    )


def test_shard_assignments_are_stable_balanced_and_complete() -> None:
    runner = _runner()
    keys = ["slow.py", "medium.py", "fast-a.py", "fast-b.py"]
    timings = {"slow.py": 8.0, "medium.py": 4.0, "fast-a.py": 2.0, "fast-b.py": 2.0}

    first = runner._shard_assignments(keys, shard_count=2, timing_estimates=timings)
    second = runner._shard_assignments(keys, shard_count=2, timing_estimates=timings)

    assert first == second == [0, 1, 1, 1]
    assert sorted(
        index
        for shard in range(2)
        for index, value in enumerate(first)
        if value == shard
    ) == list(range(len(keys)))

    rotated = runner._shard_assignments(["one.py"], shard_count=3, shard_offset=2)
    assert rotated == [2]


def test_target_group_keys_do_not_depend_on_workspace_root(tmp_path: Path) -> None:
    runner = _runner()
    tests = tmp_path / "tests"
    targets = [tests / "unit" / "test_a.py", tests / "unit" / "test_b.py"]

    assert runner._target_group_key(tests, targets) == "unit/test_a.py|unit/test_b.py"


def test_timing_history_uses_median_group_duration(tmp_path: Path) -> None:
    runner = _runner()
    reports = []
    for index, duration in enumerate((1.0, 9.0, 3.0)):
        path = tmp_path / f"report-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "layers": [
                        {
                            "name": "core",
                            "runs": [
                                {
                                    "group_key": "unit/test_a.py",
                                    "duration_seconds": duration,
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        reports.append(path)

    assert runner._timing_history(reports) == {"core": {"unit/test_a.py": 3.0}}


def test_run_layer_shards_existing_boundaries_and_enables_xdist(
    tmp_path: Path,
) -> None:
    runner = _runner()
    for name in ("test_a.py", "test_b.py", "test_c.py"):
        (tmp_path / name).write_text("", encoding="utf-8")

    record = runner._run_layer(
        python=Path(__import__("sys").executable),
        application_root=ROOT,
        mode="python",
        capability_modes={},
        active_writers={},
        layer="core",
        tests_path=tmp_path,
        cwd=ROOT,
        profile="feature",
        extra_pytest_args=[],
        dry_run=True,
        identity_fingerprint="candidate",
        shard_index=1,
        shard_count=2,
        pytest_workers=2,
    )

    assert record["selected_group_indexes"] == []
    assert record["target_group_count"] == 1
    assert len(record["commands"]) == 0
    record = runner._run_layer(
        python=Path(__import__("sys").executable),
        application_root=ROOT,
        mode="python",
        capability_modes={},
        active_writers={},
        layer="core",
        tests_path=tmp_path,
        cwd=ROOT,
        profile="feature",
        extra_pytest_args=[],
        dry_run=True,
        identity_fingerprint="candidate",
        shard_index=0,
        shard_count=2,
        pytest_workers=2,
    )
    assert len(record["commands"]) == 1
    command = record["commands"][0]
    assert command[command.index("-n") + 1] == "2"
    assert "--dist" in command
    bootstrap = Path(command[2])
    assert bootstrap.name == "adr015_pytest_bootstrap.py"
    source = bootstrap.read_text(encoding="utf-8")
    assert "pytest.main(args)" in source
    assert "console_main" not in source


def test_parser_is_file_isolated_and_xdist_is_disabled(tmp_path: Path) -> None:
    runner = _runner()
    for name in ("test_a.py", "test_b.py"):
        (tmp_path / name).write_text("", encoding="utf-8")

    record = runner._run_layer(
        python=Path(__import__("sys").executable),
        application_root=ROOT,
        mode="python",
        capability_modes={},
        active_writers={},
        layer="parser",
        tests_path=tmp_path,
        cwd=ROOT,
        profile="feature",
        extra_pytest_args=[],
        dry_run=True,
        identity_fingerprint="candidate",
        shard_index=0,
        shard_count=1,
        pytest_workers=2,
    )

    assert record["target_group_count"] == 2
    assert record["pytest_workers"] == 0
    assert record["requested_pytest_workers"] == 2
    assert all("-n" not in command for command in record["commands"])


def test_tiny_suite_serial_and_xdist_cover_same_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner()
    tests_path = tmp_path / "tests"
    tests_path.mkdir()
    for name in ("a", "b", "c"):
        (tests_path / f"test_{name}.py").write_text(
            "import os, pathlib, pytest\n"
            "@pytest.mark.ci\n"
            f"def test_{name}():\n"
            f"    pathlib.Path(os.environ['ADR015_TINY_OUT']).joinpath('{name}').touch()\n",
            encoding="utf-8",
        )

    def run(workers: int, output: Path) -> dict[str, object]:
        output.mkdir()
        monkeypatch.setenv("ADR015_TINY_OUT", str(output))
        return runner._run_layer(
            python=Path(__import__("sys").executable),
            application_root=ROOT,
            mode="python",
            capability_modes={},
            active_writers={},
            layer="core",
            tests_path=tests_path,
            cwd=ROOT,
            profile="feature",
            extra_pytest_args=[],
            dry_run=False,
            identity_fingerprint="candidate",
            pytest_workers=workers,
        )

    serial_output = tmp_path / "serial"
    parallel_output = tmp_path / "parallel"
    serial = run(0, serial_output)
    parallel = run(2, parallel_output)

    assert serial["status"] == parallel["status"] == "passed"
    assert {path.name for path in serial_output.iterdir()} == {"a", "b", "c"}
    assert {path.name for path in parallel_output.iterdir()} == {"a", "b", "c"}
