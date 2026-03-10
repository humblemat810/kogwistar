from __future__ import annotations

import abc
import json
import logging
import multiprocessing
import sys
from typing import Any, Dict, Optional, Protocol, Type, Union

from graph_knowledge_engine.runtime.models import StepRunResult, RunSuccess, RunFailure, RunSuspended

logger = logging.getLogger("workflow.sandbox")

class Sandbox(abc.ABC):
    @abc.abstractmethod
    def run(self, code: str, state: Dict[str, Any], context: Dict[str, Any]) -> StepRunResult:
        """Execute code in the sandbox."""
        pass

class SimplePythonSandbox(Sandbox):
    """A local process-based sandbox using restricted globals."""
    
    def run(self, code: str, state: Dict[str, Any], context: Dict[str, Any]) -> StepRunResult:
        # We use a Queue to get the result back from the child process
        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        
        p = multiprocessing.Process(
            target=self._worker, 
            args=(code, state, context, result_queue)
        )
        p.start()
        p.join(timeout=30) # Default 30s timeout
        
        if p.is_alive():
            p.terminate()
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["Sandbox execution timed out"]
            )
            
        if result_queue.empty():
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=["Sandbox execution failed with no result"]
            )
            
        res = result_queue.get()
        if isinstance(res, Exception):
            return RunFailure(
                conversation_node_id=None,
                state_update=[],
                errors=[str(res)]
            )
            
        # Expecting a dict or StepRunResult compatible structure
        if isinstance(res, dict):
            if res.get("status") == "failure":
                return RunFailure(**res)
            return RunSuccess(**res)
        
        return res

    def _worker(self, code: str, state: Dict[str, Any], context: Dict[str, Any], queue: multiprocessing.Queue):
        try:
            # Restricted globals
            safe_globals = {
                "__builtins__": {
                    k: __builtins__[k] for k in [
                        'abs', 'all', 'any', 'bin', 'bool', 'bytes', 'chr', 'dict', 
                        'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset', 
                        'hash', 'hex', 'int', 'isinstance', 'issubclass', 'iter', 'len', 
                        'list', 'map', 'max', 'min', 'next', 'object', 'oct', 'ord', 
                        'pow', 'print', 'range', 'repr', 'reversed', 'round', 'set', 
                        'slice', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
                        'Exception', 'ValueError', 'TypeError', 'KeyError'
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

class AzureFunctionSandbox(Sandbox):
    def __init__(self, endpoint: str, key: Optional[str] = None):
        self.endpoint = endpoint
        self.key = key

    def run(self, code: str, state: Dict[str, Any], context: Dict[str, Any]) -> StepRunResult:
        import requests
        headers = {"x-functions-key": self.key} if self.key else {}
        payload = {"code": code, "state": state, "context": context}
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "failure":
                return RunFailure(**data)
            return RunSuccess(**data)
        except Exception as e:
            return RunFailure(conversation_node_id=None, state_update=[], errors=[f"Azure Sandbox error: {str(e)}"])

class LambdaSandbox(Sandbox):
    def __init__(self, function_name: str, region_name: Optional[str] = None):
        self.function_name = function_name
        self.region_name = region_name

    def run(self, code: str, state: Dict[str, Any], context: Dict[str, Any]) -> StepRunResult:
        import boto3
        client = boto3.client("lambda", region_name=self.region_name)
        payload = json.dumps({"code": code, "state": state, "context": context})
        try:
            resp = client.invoke(FunctionName=self.function_name, Payload=payload)
            data = json.loads(resp["Payload"].read().decode("utf-8"))
            if data.get("status") == "failure":
                return RunFailure(**data)
            return RunSuccess(**data)
        except Exception as e:
            return RunFailure(conversation_node_id=None, state_update=[], errors=[f"AWS Sandbox error: {str(e)}"])

class CloudFunctionSandbox(Sandbox):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def run(self, code: str, state: Dict[str, Any], context: Dict[str, Any]) -> StepRunResult:
        import google.auth.transport.requests
        import google.oauth2.id_token
        import requests

        auth_req = google.auth.transport.requests.Request()
        id_token = google.oauth2.id_token.fetch_id_token(auth_req, self.endpoint)
        
        headers = {"Authorization": f"Bearer {id_token}"}
        payload = {"code": code, "state": state, "context": context}
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "failure":
                return RunFailure(**data)
            return RunSuccess(**data)
        except Exception as e:
            return RunFailure(conversation_node_id=None, state_update=[], errors=[f"GCP Sandbox error: {str(e)}"])

class ClientSideSandbox(Sandbox):
    """
    A sandbox that doesn't execute code directly but suspends the run,
    yielding a payload that can be sent to a client-side execution environment.
    """
    def run(self, code: str, state: Dict[str, Any], context: Dict[str, Any]) -> StepRunResult:
        return RunSuspended(
            conversation_node_id=None,
            state_update=[],
            resume_payload={
                "type": "client_sandbox_task",
                "code": code,
                "state": state,
                "context": context
            }
        )

class SandboxFactory:
    @staticmethod
    def create(sandbox_type: str, config: Dict[str, Any]) -> Sandbox:
        if sandbox_type == "local":
            return SimplePythonSandbox()
        elif sandbox_type == "client":
            return ClientSideSandbox()
        elif sandbox_type == "azure":
            return AzureFunctionSandbox(endpoint=config["endpoint"], key=config.get("key"))
        elif sandbox_type == "aws":
            return LambdaSandbox(function_name=config["function_name"], region_name=config.get("region"))
        elif sandbox_type == "gcp":
            return CloudFunctionSandbox(endpoint=config["endpoint"])
        else:
            logger.warning(f"Unknown sandbox type {sandbox_type}, falling back to local")
            return SimplePythonSandbox()
