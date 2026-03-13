from __future__ import annotations

import abc
from dataclasses import dataclass, field
import json
import logging
import multiprocessing
import subprocess
import threading
import uuid
from typing import Any, Dict, Optional

from graph_knowledge_engine.runtime.models import (
    StepRunResult,
    RunSuccess,
    RunFailure,
    RunSuspended,
)

logger = logging.getLogger("workflow.sandbox")


@dataclass(frozen=True, slots=True)
class SandboxRequest:
    code: str
    state: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


def _coerce_step_result(payload: Any) -> StepRunResult:
    if isinstance(payload, (RunSuccess, RunFailure, RunSuspended)):
        return payload
    if isinstance(payload, dict):
        # Support legacy sandbox payloads that omit nullable result fields.
        payload = dict(payload)
        payload.setdefault("conversation_node_id", None)
        payload.setdefault("state_update", [])
        status = payload.get("status")
        if status == "failure":
            return RunFailure(**payload)
        if status == "suspended":
            return RunSuspended(**payload)
        return RunSuccess(**payload)
    return RunFailure(
        conversation_node_id=None,
        state_update=[],
        errors=[f"Unsupported sandbox result type: {type(payload).__name__}"],
    )


class Sandbox(abc.ABC):
    @abc.abstractmethod
    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        """Execute code in the sandbox."""
        pass

    def close_run(self, run_id: str) -> None:
        """Best-effort cleanup for run-scoped resources."""

    def close(self) -> None:
        """Best-effort cleanup for process-scoped resources."""


class SimplePythonSandbox(Sandbox):
    """A local process-based sandbox using restricted globals."""

    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        # We use a Queue to get the result back from the child process
        result_queue: multiprocessing.Queue = multiprocessing.Queue()

        p = multiprocessing.Process(
            target=self._worker, args=(code, state, context, result_queue)
        )
        p.start()
        p.join(timeout=30)  # Default 30s timeout

        if p.is_alive():
            p.terminate()
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["Sandbox execution timed out"],
            )

        if result_queue.empty():
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["Sandbox execution failed with no result"],
            )

        res = result_queue.get()
        if isinstance(res, Exception):
            return RunFailure(
                conversation_node_id=None, state_update=[], errors=[str(res)]
            )

        return _coerce_step_result(res)

    def _worker(
        self,
        code: str,
        state: Dict[str, Any],
        context: Dict[str, Any],
        queue: multiprocessing.Queue,
    ):
        try:
            # Restricted globals
            safe_globals = {
                "__builtins__": {
                    k: __builtins__[k]
                    for k in [
                        "abs",
                        "all",
                        "any",
                        "bin",
                        "bool",
                        "bytes",
                        "chr",
                        "dict",
                        "divmod",
                        "enumerate",
                        "filter",
                        "float",
                        "format",
                        "frozenset",
                        "hash",
                        "hex",
                        "int",
                        "isinstance",
                        "issubclass",
                        "iter",
                        "len",
                        "list",
                        "map",
                        "max",
                        "min",
                        "next",
                        "object",
                        "oct",
                        "ord",
                        "pow",
                        "print",
                        "range",
                        "repr",
                        "reversed",
                        "round",
                        "set",
                        "slice",
                        "sorted",
                        "str",
                        "sum",
                        "tuple",
                        "type",
                        "zip",
                        "Exception",
                        "ValueError",
                        "TypeError",
                        "KeyError",
                    ]
                },
                "state": state,
                "context": context,
            }

            # Execute the code
            exec(code, safe_globals)

            # Extract result from globals if a 'result' variable was set
            # or just return the modified state if that's the pattern.
            # For simplicity, we assume the code can modify 'state' or set 'result'.
            result = safe_globals.get("result")
            if result is None:
                # Fallback: assume the code modified the state and we want to return it
                result = {"state_update": [("u", safe_globals.get("state", {}))]}

            queue.put(result)
        except Exception as e:
            queue.put(e)


class DockerPythonSandbox(Sandbox):
    """Execute Python snippets inside a container runtime.

    Modes:
      - per_op: start a fresh container for every sandbox.run(...)
      - per_run: reuse one long-lived container per workflow run_id

    Guardrails:
      - containers are force-removed when execution times out
      - networking is disabled by default unless explicitly enabled
    """

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        runtime_cmd: str = "docker",
        python_bin: str = "python",
        mode: str = "per_op",
        timeout_s: float = 30.0,
        container_name_prefix: str = "gke-sandbox",
        network_disabled: bool = True,
    ) -> None:
        if mode not in {"per_op", "per_run"}:
            raise ValueError("DockerPythonSandbox mode must be 'per_op' or 'per_run'")
        self.image = image
        self.runtime_cmd = runtime_cmd
        self.python_bin = python_bin
        self.mode = mode
        self.timeout_s = float(timeout_s)
        self.container_name_prefix = container_name_prefix
        self.network_disabled = bool(network_disabled)
        self._lock = threading.RLock()
        self._run_containers: dict[str, str] = {}

    @staticmethod
    def _runner_code() -> str:
        return "\n".join(
            [
                "import json",
                "import sys",
                "import traceback",
                "",
                "payload = json.load(sys.stdin)",
                "state = payload.get('state') or {}",
                "context = payload.get('context') or {}",
                "code = payload.get('code') or ''",
                "scope = {'state': state, 'context': context}",
                "",
                "try:",
                "    exec(code, scope)",
                "    result = scope.get('result')",
                "    if result is None:",
                "        result = {'state_update': [('u', scope.get('state', {}))]}",
                "    sys.stdout.write(json.dumps(result))",
                "except Exception as e:",
                "    sys.stdout.write(json.dumps({",
                "        'status': 'failure',",
                "        'conversation_node_id': None,",
                "        'state_update': [],",
                "        'errors': [str(e), traceback.format_exc()],",
                "    }))",
                "",
            ]
        )

    def _run_subprocess(
        self, args: list[str], *, payload_json: Optional[str] = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            input=payload_json,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )

    def _make_container_name(self, run_id: str) -> str:
        safe = (
            "".join(ch if ch.isalnum() else "-" for ch in run_id).strip("-").lower()
            or "run"
        )
        suffix = uuid.uuid4().hex[:8]
        return f"{self.container_name_prefix}-{safe[:32]}-{suffix}"

    def _network_args(self) -> list[str]:
        if self.network_disabled:
            return ["--network", "none"]
        return []

    def _remove_container(
        self, name: Optional[str], *, run_id: Optional[str] = None
    ) -> None:
        if not name:
            return
        if run_id:
            with self._lock:
                if self._run_containers.get(run_id) == name:
                    self._run_containers.pop(run_id, None)
        try:
            self._run_subprocess([self.runtime_cmd, "rm", "-f", name])
        except Exception:
            logger.exception("Failed to remove sandbox container %s", name)

    def _ensure_run_container(self, run_id: str) -> str:
        with self._lock:
            existing = self._run_containers.get(run_id)
            if existing:
                return existing

            name = self._make_container_name(run_id)
            try:
                proc = self._run_subprocess(
                    [
                        self.runtime_cmd,
                        "run",
                        "-d",
                        "--rm",
                        "--name",
                        name,
                        *self._network_args(),
                        self.image,
                        "tail",
                        "-f",
                        "/dev/null",
                    ]
                )
            except subprocess.TimeoutExpired as exc:
                self._remove_container(name)
                raise RuntimeError(
                    f"Timed out starting sandbox container after {self.timeout_s:.0f}s"
                ) from exc
            if proc.returncode != 0:
                self._remove_container(name)
                stderr = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"Failed to start sandbox container: {stderr or 'unknown error'}"
                )

            self._run_containers[run_id] = name
            return name

    def _exec_in_container(
        self,
        cmd: list[str],
        *,
        payload_json: str,
        container_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> StepRunResult:
        try:
            proc = self._run_subprocess(cmd, payload_json=payload_json)
        except subprocess.TimeoutExpired:
            self._remove_container(container_name, run_id=run_id)
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"Sandbox execution timed out after {self.timeout_s:.0f}s"],
            )

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"Container sandbox failed: {detail or 'unknown error'}"],
            )

        stdout = (proc.stdout or "").strip()
        if not stdout:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["Container sandbox returned no output"],
            )

        try:
            payload = json.loads(stdout)
        except Exception as exc:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"Container sandbox returned invalid JSON: {exc}", stdout],
            )
        return _coerce_step_result(payload)

    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        payload_json = json.dumps({"code": code, "state": state, "context": context})
        runner = self._runner_code()
        container_name: Optional[str] = None
        run_id: Optional[str] = None

        try:
            if self.mode == "per_run":
                run_id = str(context.get("run_id") or "").strip()
                if not run_id:
                    return RunFailure(
                        conversation_node_id=None,
                        state_update=[],
                        errors=[
                            "Container sandbox in per_run mode requires context['run_id']"
                        ],
                    )
                container_name = self._ensure_run_container(run_id)
                cmd = [
                    self.runtime_cmd,
                    "exec",
                    "-i",
                    container_name,
                    self.python_bin,
                    "-c",
                    runner,
                ]
            else:
                container_name = self._make_container_name(
                    str(context.get("run_id") or "op")
                )
                cmd = [
                    self.runtime_cmd,
                    "run",
                    "--rm",
                    "-i",
                    "--name",
                    container_name,
                    *self._network_args(),
                    self.image,
                    self.python_bin,
                    "-c",
                    runner,
                ]
        except Exception as exc:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"Container sandbox setup failed: {exc}"],
            )

        return self._exec_in_container(
            cmd,
            payload_json=payload_json,
            container_name=container_name,
            run_id=run_id,
        )

    def close_run(self, run_id: str) -> None:
        with self._lock:
            name = self._run_containers.pop(run_id, None)
        if not name:
            return
        self._remove_container(name)

    def close(self) -> None:
        with self._lock:
            run_ids = list(self._run_containers.keys())
        for run_id in run_ids:
            self.close_run(run_id)


class AzureFunctionSandbox(Sandbox):
    def __init__(self, endpoint: str, key: Optional[str] = None):
        self.endpoint = endpoint
        self.key = key

    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        import requests

        headers = {"x-functions-key": self.key} if self.key else {}
        payload = {"code": code, "state": state, "context": context}
        try:
            resp = requests.post(
                self.endpoint, json=payload, headers=headers, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return _coerce_step_result(data)
        except Exception as e:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"Azure Sandbox error: {str(e)}"],
            )


class LambdaSandbox(Sandbox):
    def __init__(self, function_name: str, region_name: Optional[str] = None):
        self.function_name = function_name
        self.region_name = region_name

    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        import boto3

        client = boto3.client("lambda", region_name=self.region_name)
        payload = json.dumps({"code": code, "state": state, "context": context})
        try:
            resp = client.invoke(FunctionName=self.function_name, Payload=payload)
            data = json.loads(resp["Payload"].read().decode("utf-8"))
            return _coerce_step_result(data)
        except Exception as e:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"AWS Sandbox error: {str(e)}"],
            )


class CloudFunctionSandbox(Sandbox):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        import google.auth.transport.requests
        import google.oauth2.id_token
        import requests

        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, self.endpoint)

        headers = {"Authorization": f"Bearer {id_token}"}
        payload = {"code": code, "state": state, "context": context}
        try:
            resp = requests.post(
                self.endpoint, json=payload, headers=headers, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return _coerce_step_result(data)
        except Exception as e:
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[f"GCP Sandbox error: {str(e)}"],
            )


class ClientSideSandbox(Sandbox):
    """
    A sandbox that doesn't execute code directly but suspends the run,
    yielding a payload that can be sent to a client-side execution environment.
    """

    def run(
        self, code: str, state: Dict[str, Any], context: Dict[str, Any]
    ) -> StepRunResult:
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={
                "type": "client_sandbox_task",
                "code": code,
                "state": state,
                "context": context,
            },
        )


class SandboxFactory:
    @staticmethod
    def create(sandbox_type: str, config: Dict[str, Any]) -> Sandbox:
        if sandbox_type == "local":
            return SimplePythonSandbox()
        elif sandbox_type in {"docker", "container"}:
            return DockerPythonSandbox(
                image=config.get("image", "python:3.11-slim"),
                runtime_cmd=config.get("runtime_cmd", "docker"),
                python_bin=config.get("python_bin", "python"),
                mode=config.get("mode", "per_op"),
                timeout_s=float(config.get("timeout_s", 30.0)),
                container_name_prefix=config.get(
                    "container_name_prefix", "gke-sandbox"
                ),
                network_disabled=bool(config.get("network_disabled", True)),
            )
        elif sandbox_type == "client":
            return ClientSideSandbox()
        elif sandbox_type == "azure":
            return AzureFunctionSandbox(
                endpoint=config["endpoint"], key=config.get("key")
            )
        elif sandbox_type == "aws":
            return LambdaSandbox(
                function_name=config["function_name"], region_name=config.get("region")
            )
        elif sandbox_type == "gcp":
            return CloudFunctionSandbox(endpoint=config["endpoint"])
        else:
            logger.warning(
                f"Unknown sandbox type {sandbox_type}, falling back to local"
            )
            return SimplePythonSandbox()
