# Workflows Are Just Graphs

We've been treating workflows and knowledge as two different things.

They're not.

A workflow is just relationships:
which step depends on which, what input feeds into what decision, what output triggers the next action.

That's already a graph.

People will say "it's a DAG" and sure, for simple scripts that's true.

But the moment things get real, with multiple inputs, shared context, retries, and human feedback, it stops being a clean DAG.

Now you have something else:
a structure where one relationship can connect many things at once.

That's where hypergraph semantics start to matter.

That modeling shift matters. A workflow design is a graph schema: typed step nodes, typed transition edges, and higher-order relations for joins, shared preconditions, approvals, context, and evidence. Execution is not outside that model. It is an instance moving through it.

---

The weird part is not that workflows are graphs.

The weird part is that we built completely separate systems for them.

- workflow engines to run steps
- databases to store state
- vector stores to retrieve context
- logs to figure out what just happened

And then we spend most of our time stitching them back together.

---

What if we just... didn't?

If workflows and knowledge share the same substrate, execution does not need to live outside the system.

A step is a node.

A dependency is an edge.

A decision that depends on multiple inputs is a higher-order relation.

An execution trace is not a log. It is more graph.

And that distinction matters. The design graph describes what may happen. The trace graph records what did happen. State is the current projection materialized from append-only events over that history. Once those are separate in the model, they stop fighting each other at runtime.

---

Once you look at it this way, durability stops being special machinery bolted onto orchestration.

Durable execution comes from append-only change records plus deterministic projection.

Checkpoints, traces, and current state are all graph artifacts derived from the same history.

Replay is possible because the execution semantics were modeled explicitly, not hidden in process memory.

At runtime, execution advances by moving tokens through the workflow design graph. Fanout creates parallel paths. Joins are explicit barriers, not accidental convergence. Suspension parks a token without losing the run, and resume continues from persisted frontier state instead of ad hoc callback memory.

State is not a mutable blob being shoved through a pipeline. It is a set of explicit artifacts: checkpoints, context snapshots, evidence selections, pins, pointers, trace records. Each one is replayable state with provenance, not incidental metadata that disappears once the function returns.

---

Most AI systems today are pipelines glued together.

They work in demos, then slowly fall apart as state drifts and context leaks.

Not because the models are bad, but because the structure is wrong.

---

If everything is already a graph, then execution, memory, and observability should share one substrate.

---

## Repo Notes

Here is how that idea shows up in this codebase.

Workflow design nodes are typed graph objects with explicit runtime semantics:

```python
from kogwistar.runtime.models import WorkflowNode, WorkflowEdge

start = WorkflowNode(
    id="wf|add_turn|start",
    type="workflow_node",
    metadata={
        "workflow_id": "add_turn",
        "wf_op": "start",
        "wf_start": True,
        "wf_fanout": False,
    },
)

route = WorkflowEdge(
    id="wf|add_turn|start->answer",
    type="workflow_edge",
    metadata={
        "workflow_id": "add_turn",
        "wf_priority": 10,
        "wf_multiplicity": "one",
        "wf_is_default": True,
    },
)
```

Runtime steps return structured results instead of mutating hidden control flow:

```python
from kogwistar.runtime.models import RunSuccess, RunSuspended

return RunSuccess(
    conversation_node_id=None,
    state_update=[("u", {"answer_text": "ready"})],
    _route_next=["join"],
)

return RunSuspended(
    conversation_node_id=None,
    state_update=[("u", {"resume_needed": True})],
    resume_payload={
        "type": "recoverable_error",
        "op": "sandbox",
        "category": "missing_input",
        "message": "Need a client supplied value before continuing.",
    },
)
```

The runtime keeps execution identity and replay state explicit:

```python
from kogwistar.runtime.runtime import StepContext

ctx = StepContext(
    run_id="run_123",
    workflow_id="add_turn",
    workflow_node_id="answer",
    op="answer",
    token_id="token_abc",
    attempt=1,
    step_seq=42,
    conversation_id="conv_123",
    turn_node_id="turn_456",
    cache_dir=None,
    state={"_deps": {}, "_rt_join": {}},
)
```

And the conversation layer persists replayable graph artifacts rather than loose logs:

```python
# context snapshot persistence creates a node plus depends_on edges
snapshot = {
    "entity_type": "context_snapshot",
    "conversation_id": "conv_123",
    "stage": "answer",
    "prompt_hash": "sha256:...",
}

# workflow trace and checkpoints are stored as graph nodes too
workflow_step_exec = {"entity_type": "workflow_step_exec", "run_id": "run_123"}
workflow_checkpoint = {"entity_type": "workflow_checkpoint", "run_id": "run_123"}
```

That is the practical version of the thesis: workflow design, execution trace, context, and provenance are not separate systems here. They are different projections of the same graph-native substrate.
