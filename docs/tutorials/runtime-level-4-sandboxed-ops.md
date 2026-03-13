# Runtime Level 4: Sandboxed Ops With Docker

Goal: show how to run LLM-generated code as an untrusted workflow step in a Docker sandbox instead of executing it directly on the host.

## What You Will Build

You will run a tiny workflow with one sandboxed op, inspect the `SandboxRequest` pattern, and confirm the generated code executes inside Docker rather than inside the resolver process.

## Why This Matters

LLM-generated code is not trustworthy by default. It can be:

- hallucinated and simply wrong
- prompt-injected by hostile input
- steered into unsafe host actions such as file access, subprocesses, or network calls

If a resolver executes that code directly on the host, the workflow runtime becomes an unsafe code-execution path. The safer pattern is:

`LLM proposes code -> resolver returns SandboxRequest -> runtime executes in sandbox`

This tutorial uses Docker because it is concrete, inspectable, and matches the current runtime sandbox implementation in this repo.

## Run or Inspect

## Quick Run

```powershell
python scripts/runtime_tutorial_ladder.py level4 `
  --data-dir .gke-data/runtime-tutorial-ladder
```

Expected output fields when Docker is available:

- `"sandbox_available": true`
- `"sandbox_executed": true`
- `"sandbox_type": "docker"`
- `"sandbox_mode": "per_op"`
- `"sandbox_op": "python_exec"`
- `"sandbox_result": "HELLO FROM LLM SANDBOX"`
- `"checkpoint_pass": true`

Expected output fields when Docker is not available:

- `"sandbox_available": false`
- `"sandbox_executed": false`
- `"status": "sandbox_unavailable"`

## Inspect The Result

- Confirm the tutorial does not run the generated code directly in the resolver.
- Confirm the runtime reports a sandboxed op named `python_exec`.
- Confirm the state update came back from the sandbox and not from host-side resolver logic.
- Compare the safety boundary here with [Runtime Level 1 - Custom Resolvers](./runtime-level-1-resolvers.md).

## What This Level Teaches

- Resolver code should treat LLM-generated code as untrusted input.
- `MappingStepResolver.register(..., is_sandboxed=True)` marks an op as sandboxed.
- The resolver returns `SandboxRequest` and the runtime executes that request through the configured sandbox.
- Docker is an example sandbox boundary that is safer than host execution for untrusted generated code.
- This is a risk-reduction pattern, not a claim of perfect isolation.

## Relevant Surfaces

- `graph_knowledge_engine.runtime.sandbox.SandboxRequest`
- `graph_knowledge_engine.runtime.sandbox.SandboxFactory.create("docker", ...)`
- `graph_knowledge_engine.runtime.resolvers.MappingStepResolver.register(..., is_sandboxed=True)`
- `graph_knowledge_engine.runtime.runtime.WorkflowRuntime`

Short example:

```python
@resolver.register("python_exec", is_sandboxed=True)
def _python_exec(ctx):
    return SandboxRequest(
        code="result = {'state_update': [('u', {'answer': 42})], 'status': 'success'}",
        context={"sandbox_mode": "per_op"},
    )
```

## Checkpoint

Pass when:

- Docker is available and the run succeeds with a sandbox-produced state update
- the output proves the sandboxed op name and mode
- the tutorial reports `sandbox_executed=true`

If Docker is unavailable, pass operational review when:

- the tutorial reports `sandbox_unavailable`
- no host-side fallback executed the generated code directly

## Invariant Demonstrated

LLM-generated code stays outside the host resolver path. The resolver proposes code, but the runtime only executes it through the configured sandbox boundary.

## Troubleshooting

- If `"sandbox_available": false`, verify Docker Desktop or the Docker daemon is running and the `docker` CLI is on your `PATH`.
- If the run fails before the sandbox step, inspect the JSON output and the runtime trace DB path.
- If you want longer-lived containers, adapt the example to `mode="per_run"` after this baseline works.

## Next Tutorial

Return to [Runtime Ladder Overview](./runtime-ladder-overview.md), revisit [Runtime Level 3 - CDC Viewer and LangGraph Interop](./runtime-level-3-observability-interop.md), or continue with [13 How to Test This Repo](./13_how_to_test_this_repo.md).
