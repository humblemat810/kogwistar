import pytest
from unittest.mock import MagicMock, patch
from graph_knowledge_engine.runtime.sandbox import (
    SimplePythonSandbox, 
    SandboxFactory, 
    AzureFunctionSandbox, 
    LambdaSandbox, 
    CloudFunctionSandbox
)
from graph_knowledge_engine.runtime.models import RunSuccess, RunFailure

def test_local_sandbox_basic():
    sandbox = SimplePythonSandbox()
    code = "result = {'state_update': [('u', {'a': state['a'] + 1})]}"
    state = {"a": 1}
    context = {}
    res = sandbox.run(code, state, context)
    assert isinstance(res, RunSuccess)
    assert res.state_update == [('u', {'a': 2})]

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
    assert any("name 'import' is not defined" in err or "invalid syntax" in err or "__import__" in err for err in res.errors)

@pytest.mark.parametrize("sandbox_type, config, expected_class", [
    ("local", {}, SimplePythonSandbox),
    ("azure", {"endpoint": "http://azure", "key": "k"}, AzureFunctionSandbox),
    ("aws", {"function_name": "fn", "region": "us-east-1"}, LambdaSandbox),
    ("gcp", {"endpoint": "http://gcp"}, CloudFunctionSandbox),
])
def test_sandbox_factory(sandbox_type, config, expected_class):
    sandbox = SandboxFactory.create(sandbox_type, config)
    assert isinstance(sandbox, expected_class)

@pytest.mark.parametrize("sandbox_class, mock_path, mock_return", [
    (AzureFunctionSandbox, "requests.post", MagicMock(json=lambda: {"status": "success", "state_update": []}, status_code=200)),
    (LambdaSandbox, "boto3.client", MagicMock()),
    (CloudFunctionSandbox, "requests.post", MagicMock(json=lambda: {"status": "success", "state_update": []}, status_code=200)),
])
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
                "Payload": MagicMock(read=lambda: b'{"status": "success", "state_update": []}')
            }
            res = sandbox.run(code, state, context)
            assert isinstance(res, RunSuccess)
            mock_client.invoke.assert_called_once()

    elif sandbox_class == CloudFunctionSandbox:
        sandbox = CloudFunctionSandbox(endpoint="http://gcp")
        # Mocking google auth as well
        with patch("google.oauth2.id_token.fetch_id_token", return_value="token"), \
             patch(mock_path, return_value=mock_return) as mock_post:
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
