# ADR-019: Goal Mode - Cyclic Workflow Pattern

**Status:** Proposed
**Date:** 2026-08-30
**Owner:** Maintainers

## Context

Kogwistar persists workflow designs as graph nodes and edges, then executes
them through `WorkflowRuntime`. The runtime already owns resolver execution,
predicates, tokens, fanout, joins, nested invoke-and-await workflows,
workflow state, checkpoints, suspension/resume, and workflow runs.

Some workflows need a durable objective loop: observe current evidence,
decide next work, invoke a workflow, inspect its result, and check whether an
objective is satisfied. Different business meaning does not imply a new
runtime primitive.

## Decision

Goal Mode is a documented, reusable ordinary cyclic workflow pattern. It is
executed entirely by the existing `WorkflowRuntime`; it is not a workflow
type, scheduler, controller service, token model, or separate state machine.

Typical topology:

```text
Observe -> Decide -> Act -> Check
                         |      |
                         |      +-> satisfied -> Done
                         |
                         +-> not satisfied -> Observe
```

Cycles are already valid when a terminal remains reachable. The same graph can
run with no Goal-specific metadata whatsoever.

Applications and tools MAY attach descriptive metadata such as
`wf_mode = "goal"` to the start node, because `WorkflowNodeMetadata` permits
extra metadata. Such metadata is advisory only: UI discovery, linting, and
documentation may use it; `WorkflowRuntime` correctness, scheduling, routing,
and recovery MUST NOT depend on it. `wf_goal_contract_version` is not adopted.

## Architecture

Nodes retain policy meanings, but all have ordinary runtime semantics:

- **Observe** gathers evidence into workflow state.
- **Decide** selects next action when objective remains unsatisfied.
- **Act** invokes one ordinary Kogwistar workflow.
- **Check** answers only whether objective is satisfied.
- Ordinary predicates route `satisfied -> Done` or
  `not_satisfied -> Observe`.

Check and Decide are distinct business policies. Neither is a new resolver
kind or runtime primitive.

Act reuses nested invoke-and-await:

```text
Act resolver
  -> RunSuccess.workflow_invocations
  -> WorkflowInvocationRequest
  -> WorkflowRuntime._run_workflow_invocation(...)
  -> WorkflowRuntime.run(...)
```

Act may invoke either an existing workflow or an existing
`WorkflowDesignArtifact`. Invoked workflows retain all ordinary capabilities:
nodes, edges, predicates, fanout, joins, and further nested invocation.

## State and Lifecycle

Goal business data is ordinary root workflow state and checkpointed under
ordinary runtime rules. An application may use conventions such as:

```text
goal_id                 optional business identifier
goal_objective
goal_status             active | satisfied | blocked | cancelled | failed
goal_iteration          application counter
goal_evidence
```

These are neither mandatory framework fields nor a mandatory `Goal` node.
`run_id` identifies one control-workflow execution; `goal_id`, when useful,
identifies an application business objective. They are not trace identity.

Workflow terminal status and business status remain separate. For example, a
control workflow may succeed after recording `goal_status = blocked` or
`cancelled_by_policy`.

One root control run may invoke many child workflow runs across loop cycles.
Existing step executions, state transitions, checkpoints, and child run results
are normal history of those iterations; no `GoalIteration` or `GoalAction`
artifact is introduced.

## Provenance and Recovery

This ADR introduces no Goal-specific recovery algorithm. A cyclic workflow
resumes by ordinary `resume_from_latest_checkpoint(...)` behavior; an Act uses
ordinary nested invocation behavior.

Current runtime already provides limited generic nested-invocation recovery
evidence: `invocation_key` participates in deterministic child-run identity,
the child run records its parent metadata, a `planned` child run and
`wf_invoked` lineage edge are persisted before child execution begins, and an
already terminal child run is reused rather than re-executed. These are
ordinary workflow facilities, not Goal facilities.

Generic durability gaps remain for every nested workflow, including this
pattern:

- resolver-selected action identity is not atomically checkpointed with the
  pre-child invocation record. A caller needing retry-safe selection must
  persist its action/key in ordinary parent state before Act;
- parent cannot generically resume a nonterminal child from its checkpoint
  rather than decide whether to invoke it again after crash/retry;
- caller cannot require immediate durable intent/checkpoint barrier before
  starting child when `checkpoint_every_n_steps` exceeds one;
- dynamic invocation materializes `WorkflowDesignArtifact` before generic
  engine-backed preflight validation and immutable design identity are defined.

These are generic `WorkflowRuntime` requirements. They MUST be designed and
implemented as reusable nested-invocation facilities before this pattern can
claim crash-safe, duplicate-avoiding child invocation. They do not create Goal
classes, tables, edges, or transaction protocols.

Likely generic facility:

```text
parent step intent + stable invocation identity
  -> child WorkflowRun identity
  -> persist planned child run and lineage before child start
  -> inspect terminal/checkpoint child before invoke/resume
```

Its naming, persistence shape, checkpoint boundary, and dynamic-artifact
fingerprint belong to a runtime/base ADR or implementation design, not here.
Applications still own external-effect idempotency; neither normal workflows
nor this pattern promises exactly-once effects.

## Interaction with Existing Runtime

- `WorkflowRuntime.run(...)` executes root control graph.
- `RunSuccess.workflow_invocations` and `WorkflowInvocationRequest` execute
  Act as normal nested invoke-and-await work.
- `WorkflowDesignArtifact` is existing dynamic workflow representation.
- WorkflowRun, step execution, terminal nodes, workflow state, and checkpoints
  provide ordinary iteration evidence.
- `resume_from_latest_checkpoint(...)` resumes root or child under common
  runtime rules.
- Validation already permits cycles when terminal is reachable. No
  Goal-specific runtime validation is required; optional linter may verify
  documented profile for discoverability.

`Plan -> Approve -> Act -> Observe -> End` in framework demo is related policy
graph. Goal Mode is reusable cyclic form, not agent framework.

## Cross-ADR Interaction

ADR-017 gives root and child workflow runs one W3C trace once generic trace
propagation exists. Each retains its own `run_id` and execution span. OTel
failure cannot alter workflow state.

ADR-018 governs derived semantic readiness only. An Act that creates a node is
correct after canonical Stage 1 persistence; goal completion need not wait for
embedding/HNSW/FTS unless ordinary business node chooses to do so.

## Consequences

- Goal control is visible and editable as ordinary workflow topology.
- Existing resolver, routing, token, fanout, join, checkpoint, and resume
  semantics are reused without reinterpretation.
- No Goal runtime class, controller, scheduler, queue, node type, edge type,
  mandatory metadata, or persistence artifact is added.
- Optional `wf_mode = "goal"` is descriptive only and may be omitted.
- Generic nested-invocation durability and dynamic-artifact correctness need
  separate runtime design before crash-safe production use is claimed.

## Alternatives

### Separate Goal node plus GoalController

Rejected. Root workflow state and checkpoints already model one control-loop
execution. Business Goal artifact may be justified later only when objective
must span/reuse distinct root control runs.

### Goal-specific runtime, scheduler, or recovery engine

Rejected. These duplicate `WorkflowRuntime`; generic nested-invocation gaps
must remain generic.

### Special GoalCheck primitive

Rejected. Ordinary resolver plus predicate routing is sufficient.

### Mandatory Goal metadata or contract version

Rejected. Topology and ordinary state execute independently. Metadata is only
optional discoverability/profile material.

## Non-Goals

- second workflow engine, scheduler, token model, join model, or controller;
- Goal-specific recovery, lineage, queue, or transaction protocol;
- durable background sub-agent dispatch or delayed join handles;
- exactly-once external effects;
- mandatory LLM planner/ReAct prompt or agent framework;
- full workflow revision governance;
- multi-goal scheduling/fairness;
- OTel/control-plane authority or knowledge-plane replacement.

## Future Extensions

- generic durable nested-invocation lineage and reconciliation;
- generic immediate checkpoint/intent barrier where needed;
- generic dynamic workflow preflight, immutable identity, and fingerprinting;
- optional goal-pattern linting and UI projection;
- standalone business Goal artifacts spanning separate root runs;
- durable background invocation and completion reconciliation;
- budget, ownership, cancellation propagation, and multi-goal policies;
- wisdom-layer distillation from completed workflow histories.
