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

    assert runner._active_writers(manifest, modes)["deterministic-contracts"] == "python"


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

    assert runner._candidate_identity_fingerprint(base) == runner._candidate_identity_fingerprint(
        moved
    )
    assert runner._candidate_identity_fingerprint(base) != runner._candidate_identity_fingerprint(
        changed_abi
    )


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

    assert runner._absolute_path_without_symlink_resolution(
        Path("venv/bin/python")
    ) == tmp_path / "venv" / "bin" / "python"


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

    first = runner._target_groups(layer="core", tests_path=tmp_path, profile="milestone")
    second = runner._target_groups(layer="core", tests_path=tmp_path, profile="milestone")
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

    groups = runner._target_groups(layer="parser", tests_path=tmp_path, profile="milestone")

    assert [path for group in groups for path in group] == expected
    assert all(len(group) == 1 for group in groups)
    assert runner._target_groups(
        layer="parser", tests_path=tmp_path, profile="feature"
    ) == [[tmp_path]]


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
        ("parser", "milestone", 1, False),
        ("application", "regression", 5, True),
    ],
)
def test_no_selected_tests_is_success_only_for_file_isolated_release_layers(
    layer: str, profile: str, returncode: int, expected: bool
) -> None:
    runner = _runner()

    assert runner._command_succeeded(
        layer=layer, profile=profile, returncode=returncode
    ) is expected


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

    assert runner._reusable_run(
        command=command, prior_runs=prior, layer="core", profile="milestone"
    ) is not None
    assert runner._reusable_run(
        command=command, prior_runs=prior, layer="core", profile="feature"
    ) is None
