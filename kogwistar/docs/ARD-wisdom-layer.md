ARD: Wisdom Layer and Dynamic Workflow Orchestration
Status: Draft
Date: 2026-03-23
Owner: Maintainers

1. Summary
Define wisdom as the meta-learning layer above conversation/workflow/core, and define workflow invocation semantics so a workflow step can either:

spawn or reuse another persisted workflow design by workflow_id, or
emit multiple _route_next targets that the runtime executes as fanout branches.
The key goal is to make workflow orchestration flexible enough for LLM-driven planning while keeping execution traceable, reusable, and graph-native.

2. Architectural Decision
2.1 Wisdom Layer
wisdom is a top-layer graph namespace for distilled, reusable knowledge about:

successful workflow patterns
run outcomes and evaluation signals
user- or domain-level preferences inferred from history
curated projections from conversation and workflow traces
It is not a live execution layer.

It may read from lower layers, but it should not own conversation turns or mutable workflow execution state.

2.2 Workflow Invocation Semantics
A workflow step may invoke another workflow by workflow_id.

Reuse means reuse the persisted workflow design, not the prior execution instance.
Each invocation creates a fresh child run with its own run/checkpoint/step trace.
The parent step consumes the child result and continues execution.
This is an explicit runtime capability, not an implicit side effect.
2.3 LLM-Driven Dynamic Workflow Design
An LLM may synthesize a new workflow on the fly.

The synthesized workflow must be persisted as a first-class workflow artifact.
The runtime should execute it through the normal persisted workflow path.
The generated design must be inspectable and reusable after the run.
2.4 _route_next Fanout
A step may return one or many next-step targets.

One target behaves as normal routing.
Many targets create runtime fanout branches.
Fanout must preserve lineage and join semantics.
This should apply to both static workflow edges and LLM-decided routing.
3. Implementation Changes
3.1 Layer Mapping
Keep the existing layer map aligned with the repo’s current architecture:

core: graph/storage/provenance substrate
workflow: runtime execution, routing, fanout, nested runs
conversation: domain-specific orchestration and step assembly
wisdom: pattern extraction, evaluation, and workflow reuse guidance
3.2 Workflow Design Artifacts
Extend the workflow design model so a synthesized workflow can be represented and persisted like any other design.

The design contract should support:

a stable workflow_id
versioned graph persistence
explicit start/terminal semantics
traceability back to the originating run or prompt context
3.3 Execution Contract
Clarify the step result contract so a step can express:

a single next step
multiple next steps
a nested workflow invocation
a request to persist or reuse a synthesized workflow design
The runtime remains responsible for:

child run creation
fanout token/branch management
checkpointing and replay
merging or joining when the workflow design declares it
3.4 Wisdom Materialization
Wisdom should be populated from stable artifacts, not raw transient state.

Typical inputs:

completed workflow traces
failure and recovery outcomes
user corrections
pinned conversation/workflow projections
Typical outputs:

reusable workflow patterns
candidate route policies
learned heuristics for future orchestration
distilled “what worked” summaries
4. Test Plan
Add coverage for these scenarios:

wisdom layer stores meta-patterns and does not become a live execution state holder
a parent workflow invokes a child workflow by workflow_id
child workflow invocation creates a fresh run and returns a usable result
LLM-synthesized workflows are persisted before reuse
_route_next with one route behaves as today
_route_next with many routes fans out correctly
fanout preserves join behavior and does not collapse branches unintentionally
invalid synthesized workflows are rejected before execution
child workflow failure propagates in a controlled result shape
5. Assumptions
The plan assumes spec-first work, not code changes yet.
The plan assumes persisted workflow designs are the source of reuse.
The plan assumes runtime fanout is the primary mechanism for multi-route execution.
The plan assumes wisdom starts as a graph-native layer over existing traces and projections, not as a new backend.
6. Notes
This ARD intentionally keeps the semantics strict:

workflow design is reusable
workflow execution is ephemeral per run
wisdom is distilled from executions, not merged into them
fanout is explicit and traceable