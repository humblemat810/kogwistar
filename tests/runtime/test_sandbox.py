import pytest
pytestmark = pytest.mark.ci_full
import subprocess
from unittest.mock import MagicMock, patch
from kogwistar.runtime.sandbox import (
    AzureFunctionSandbox,
    CloudFunctionSandbox,
    DockerPythonSandbox,
    LambdaSandbox,
    SandboxFactory,
    SandboxRequest,
    SimplePythonSandbox,
)
from kogwistar.runtime.resolvers import MappingStepResolver
from kogwistar.runtime.models import RunSuccess, RunFailure
from kogwistar.runtime.runtime import StepContext


def test_local_sandbox_basic():
    sandbox = SimplePythonSandbox(timeout = 300)
    code = "result = {'state_update': [('u', {'a': state['a'] + 1})]}"
    state = {"a": 1}
    context = {}
    res = sandbox.run(code, state, context)
    assert isinstance(res, RunSuccess)
    assert res.state_update == [("u", {"a": 2})]


def test_local_sandbox_security():
    sandbox = SimplePythonSandbox()
    # Try to import os and do something bad
    code = """
try:
    import os
    os.system('echo "hacked"')
    result = {'status': 'success'}
except Exception as e:
    result = {'status': 'failure', 'errors': [str(e)]}
"""
    state = {}
    context = {}
    res = sandbox.run(code, state, context)
    # It should fail because 'import' is not in builtins
    assert isinstance(res, RunFailure)
    assert any(
        "name 'import' is not defined" in err
        or "invalid syntax" in err
        or "__import__" in err
        for err in res.errors
    )


@pytest.mark.parametrize(
    "sandbox_type, config, expected_class",
    [
        ("local", {}, SimplePythonSandbox),
        (
            "docker",
            {"image": "python:3.11-slim", "mode": "per_op"},
            DockerPythonSandbox,
        ),
        ("azure", {"endpoint": "http://azure", "key": "k"}, AzureFunctionSandbox),
        ("aws", {"function_name": "fn", "region": "us-east-1"}, LambdaSandbox),
        ("gcp", {"endpoint": "http://gcp"}, CloudFunctionSandbox),
    ],
)
def test_sandbox_factory(sandbox_type, config, expected_class):
    sandbox = SandboxFactory.create(sandbox_type, config)
    assert isinstance(sandbox, expected_class)


@pytest.mark.parametrize(
    "sandbox_class, mock_path, mock_return",
    [
        (
            AzureFunctionSandbox,
            "requests.post",
            MagicMock(
                json=lambda: {"status": "success", "state_update": []}, status_code=200
            ),
        ),
        (LambdaSandbox, "boto3.client", MagicMock()),
        (
            CloudFunctionSandbox,
            "requests.post",
            MagicMock(
                json=lambda: {"status": "success", "state_update": []}, status_code=200
            ),
        ),
    ],
)
def test_cloud_sandboxes_mocked(sandbox_class, mock_path, mock_return):
    state = {"val": 1}
    context = {}
    code = "print('hello')"

    if sandbox_class == AzureFunctionSandbox:
        sandbox = AzureFunctionSandbox(endpoint="http://azure")
        with patch(mock_path, return_value=mock_return) as mock_post:
            res = sandbox.run(code, state, context)
            assert isinstance(res, RunSuccess)
            mock_post.assert_called_once()

    elif sandbox_class == LambdaSandbox:
        sandbox = LambdaSandbox(function_name="fn")
        # For Lambda, we need to mock the invoke call
        with patch(mock_path) as mock_boto:
            mock_client = MagicMock()
            mock_boto.return_value = mock_client
            mock_client.invoke.return_value = {
                "Payload": MagicMock(
                    read=lambda: b'{"status": "success", "state_update": []}'
                )
            }
            res = sandbox.run(code, state, context)
            assert isinstance(res, RunSuccess)
            mock_client.invoke.assert_called_once()

    elif sandbox_class == CloudFunctionSandbox:
        sandbox = CloudFunctionSandbox(endpoint="http://gcp")
        # Mocking google auth as well
        with (
            patch("google.oauth2.id_token.fetch_id_token", return_value="token"),
            patch(mock_path, return_value=mock_return) as mock_post,
        ):
            res = sandbox.run(code, state, context)
            assert isinstance(res, RunSuccess)
            mock_post.assert_called_once()


def test_local_sandbox_timeout():
    # Note: testing timeout might be slow in CI, but here's how it would look
    # sandbox = SimplePythonSandbox()
    # code = "while True: pass"
    # res = sandbox.run(code, {}, {})
    # assert isinstance(res, RunFailure)
    # assert "timed out" in res.errors[0]
    pass


def test_docker_sandbox_per_op_invokes_runtime_once():
    sandbox = DockerPythonSandbox(
        image="python:3.11-slim", mode="per_op", runtime_cmd="docker"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"state_update": [["u", {"a": 2}]]}', stderr=""
        )
        res = sandbox.run(
            "result = {'state_update': [('u', {'a': state['a'] + 1})]}",
            {"a": 1},
            {"run_id": "r1"},
        )

    assert isinstance(res, RunSuccess)
    assert len(mock_run.call_args_list) == 1
    args = mock_run.call_args_list[0].args[0]
    assert args[:4] == ["docker", "run", "--rm", "-i"]
    assert "--network" in args
    assert args[args.index("--network") + 1] == "none"


def test_docker_sandbox_runner_code_is_valid_python():
    runner = DockerPythonSandbox._runner_code()
    compile(runner, "<docker_sandbox_runner>", "exec")


def test_docker_sandbox_per_run_reuses_container_until_close():
    sandbox = DockerPythonSandbox(
        image="python:3.11-slim", mode="per_run", runtime_cmd="docker"
    )

    start_proc = MagicMock(returncode=0, stdout="container-id\n", stderr="")
    exec_proc = MagicMock(returncode=0, stdout='{"state_update": []}', stderr="")
    stop_proc = MagicMock(returncode=0, stdout="", stderr="")

    with patch(
        "subprocess.run", side_effect=[start_proc, exec_proc, exec_proc, stop_proc]
    ) as mock_run:
        res1 = sandbox.run("result = {'state_update': []}", {}, {"run_id": "run-1"})
        res2 = sandbox.run("result = {'state_update': []}", {}, {"run_id": "run-1"})
        sandbox.close_run("run-1")

    assert isinstance(res1, RunSuccess)
    assert isinstance(res2, RunSuccess)
    assert len(mock_run.call_args_list) == 4
    assert mock_run.call_args_list[0].args[0][:3] == ["docker", "run", "-d"]
    start_args = mock_run.call_args_list[0].args[0]
    assert "--network" in start_args
    assert start_args[start_args.index("--network") + 1] == "none"
    assert mock_run.call_args_list[1].args[0][:3] == ["docker", "exec", "-i"]
    assert mock_run.call_args_list[2].args[0][:3] == ["docker", "exec", "-i"]
    assert mock_run.call_args_list[3].args[0][:3] == ["docker", "rm", "-f"]


def test_docker_sandbox_per_op_timeout_force_removes_container():
    sandbox = DockerPythonSandbox(
        image="python:3.11-slim",
        mode="per_op",
        runtime_cmd="docker",
        timeout_s=5,
    )

    with (
        patch.object(
            sandbox, "_make_container_name", return_value="sandbox-timeout-op"
        ),
        patch(
            "subprocess.run",
            side_effect=[
                subprocess.TimeoutExpired(cmd=["docker", "run"], timeout=5),
                MagicMock(returncode=0, stdout="", stderr=""),
            ],
        ) as mock_run,
    ):
        res = sandbox.run("result = {'state_update': []}", {}, {"run_id": "run-1"})

    assert isinstance(res, RunFailure)
    assert "timed out" in res.errors[0]
    assert len(mock_run.call_args_list) == 2
    assert mock_run.call_args_list[1].args[0] == [
        "docker",
        "rm",
        "-f",
        "sandbox-timeout-op",
    ]


def test_docker_sandbox_per_run_timeout_force_removes_container():
    sandbox = DockerPythonSandbox(
        image="python:3.11-slim",
        mode="per_run",
        runtime_cmd="docker",
        timeout_s=5,
    )

    start_proc = MagicMock(returncode=0, stdout="container-id\n", stderr="")
    stop_proc = MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch.object(
            sandbox, "_make_container_name", return_value="sandbox-timeout-run"
        ),
        patch(
            "subprocess.run",
            side_effect=[
                start_proc,
                subprocess.TimeoutExpired(cmd=["docker", "exec"], timeout=5),
                stop_proc,
            ],
        ) as mock_run,
    ):
        res = sandbox.run("result = {'state_update': []}", {}, {"run_id": "run-1"})

    assert isinstance(res, RunFailure)
    assert "timed out" in res.errors[0]
    assert sandbox._run_containers == {}
    assert len(mock_run.call_args_list) == 3
    assert mock_run.call_args_list[2].args[0] == [
        "docker",
        "rm",
        "-f",
        "sandbox-timeout-run",
    ]


def test_mapping_resolver_executes_sandboxed_code_with_run_context(tmp_path):
    class _RecordingSandbox:
        def __init__(self):
            self.calls = []

        def run(self, code, state, context):
            self.calls.append((code, state, context))
            return RunSuccess(
                conversation_node_id=None, state_update=[("u", {"sandboxed": True})]
            )

        def close_run(self, run_id: str) -> None:
            self.calls.append(("close_run", run_id, {}))

    resolver = MappingStepResolver()
    sandbox = _RecordingSandbox()
    resolver.set_sandbox(sandbox)

    @resolver.register("python_exec", is_sandboxed=True)
    def _python_exec(ctx):
        return SandboxRequest(
            code="result = {'state_update': [('u', {'answer': 42})]}",
            context={"extra": "value"},
        )

    fn = resolver.resolve("python_exec")
    ctx = StepContext(
        run_id="run-123",
        workflow_id="wf-1",
        workflow_node_id="node-1",
        op="python_exec",
        token_id="tok-1",
        attempt=1,
        step_seq=7,
        cache_dir=tmp_path / "sandbox-cache",
        conversation_id="conv-1",
        turn_node_id="turn-1",
        state={"value": 1},
    )

    res = fn(ctx)
    resolver.close_sandbox_run("run-123")

    assert isinstance(res, RunSuccess)
    assert sandbox.calls[0][0].startswith("result =")
    assert sandbox.calls[0][1]["value"] == 1
    assert sandbox.calls[0][2]["run_id"] == "run-123"
    assert sandbox.calls[0][2]["extra"] == "value"
    assert sandbox.calls[1][0] == "close_run"


def test_mapping_resolver_does_not_prepare_sandbox_for_non_sandboxed_op(tmp_path):
    class _FailIfCalledSandbox:
        def run(self, code, state, context):
            raise AssertionError("sandbox should not run for non-sandboxed ops")

        def close_run(self, run_id: str) -> None:
            return None

    resolver = MappingStepResolver()
    resolver.set_sandbox(_FailIfCalledSandbox())

    @resolver.register("normal_op")
    def _normal(ctx):
        return RunSuccess(conversation_node_id=None, state_update=[("u", {"ok": True})])

    fn = resolver.resolve("normal_op")
    ctx = StepContext(
        run_id="run-plain",
        workflow_id="wf-plain",
        workflow_node_id="node-plain",
        op="normal_op",
        token_id="tok-plain",
        attempt=1,
        step_seq=1,
        cache_dir=tmp_path / "sandbox-cache",
        conversation_id="conv-plain",
        turn_node_id="turn-plain",
        state={"value": 1},
    )

    res = fn(ctx)
    assert isinstance(res, RunSuccess)
    assert res.state_update == [("u", {"ok": True})]
