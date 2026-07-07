# ARD: Wisdom Layer and Dynamic Workflow Orchestration

**Status:** Draft  
**Date:** 2026-03-23  
**Owner:** Maintainers

## 1. Summary

Define `wisdom` as the meta-learning layer above conversation/workflow/core, and define workflow execution so a step can:

- spawn or reuse another persisted workflow by `workflow_id`
- persist a synthesized workflow design on the fly
- fan out to multiple `_route_next` targets

The goal is to keep orchestration flexible for LLM-driven planning while preserving graph-native traceability and reuse.

## 2. Architectural Decision

### 2.1 Wisdom Layer

`wisdom` is a top-layer graph namespace for distilled, reusable knowledge about:

- successful workflow patterns
- run outcomes and evaluation signals
- user- or domain-level preferences inferred from history
- curated projections from conversation and workflow traces

It is not a live execution layer.

### 2.2 Dreaming Agent Semantics

A dreaming agent is a maintenance-oriented agent that may run in foreground or background mode with an explicit token budget and policy gates.

Its internal self-to-self reasoning may be recorded as compact conversation-layer planning traces, but not as live execution truth.
The resulting candidate proposal belongs in wisdom as a suggestion, scored hypothesis, or reusable pattern.
Only after approval does the system materialize that suggestion into workflow design.

This keeps the layers clean:

- `conversation`: internal planning dialogue, compact reasoning trace, scratchpad-like state
- `wisdom`: proposal, confidence, evidence, failure lesson, reuse hint
- `workflow`: approved design, versioned routing, executable control flow

The self-to-self trace is reasoning, not the proposal itself.
The proposal is a compact summary distilled from that reasoning.

### 2.3 Workflow Invocation Semantics

A workflow step may invoke another workflow by `workflow_id`.

- Reuse means reuse the persisted workflow design, not the prior execution instance.
- Each invocation creates a fresh child run with its own trace/checkpoint history.
- The parent step consumes the child result and continues execution.
- This is an explicit runtime capability.

### 2.4 Dynamic Workflow Design

An LLM may synthesize a new workflow on the fly.

- The synthesized workflow is persisted as a first-class workflow artifact.
- The runtime executes it through the normal persisted workflow path.
- The generated design remains inspectable and reusable after the run.

### 2.5 Layer Ownership

Use this placement rule:

- `conversation`: self-to-self reasoning traces, scratchpad planning, user feedback snippets, and run-local conversational evidence
- `wisdom`: proposal summaries, proposal evaluations, lessons, evidence rollups, confidence, and reuse hints
- `workflow`: executable workflow designs, workflow nodes, workflow edges, version history, and active/inactive design lineage

The split is:

- thinking stays in `conversation`
- the compact proposal and its evaluation stay in `wisdom`
- the executable design and its versioned lineage stay in `workflow`

If a candidate design is materialized before final decision, it still lives as a workflow artifact, but the accept/deprecate decision record remains in `wisdom`.

### 2.6 `_route_next` Fanout

A step may return one or many `_route_next` targets.

- one target behaves like normal routing
- many targets create runtime fanout branches
- fanout preserves lineage and join semantics
- this applies to both static graph edges and LLM-decided routing

## 3. Implementation Changes

### 3.1 Layer Mapping

Keep the existing mapping aligned with the repo architecture:

- `core`: graph/storage/provenance substrate
- `workflow`: runtime execution, routing, fanout, nested runs
- `conversation`: domain-specific orchestration and step assembly
- `wisdom`: pattern extraction, evaluation, and workflow reuse guidance
- `maintenance agent`: background or foreground orchestration for distillation, proposal generation, and repair under budget/policy control

### 3.2 Workflow Design Artifacts

Extend the workflow design model so a synthesized workflow can be represented and persisted like any other design.

The design contract supports:

- a stable `workflow_id`
- versioned graph persistence
- explicit start/terminal semantics
- traceability back to the originating run or prompt context

### 3.3 Execution Contract

Clarify the step result contract so a step can express:

- a single next step
- multiple next steps
- a nested workflow invocation
- a request to persist or reuse a synthesized workflow design

The runtime remains responsible for:

- child run creation
- fanout token/branch management
- checkpointing and replay
- merging or joining when the workflow design declares it

### 3.4 Wisdom Materialization

Wisdom should be populated from stable artifacts, not raw transient state.

Typical inputs:

- completed workflow traces
- failure and recovery outcomes
- user corrections
- pinned conversation/workflow projections

Typical outputs:

- reusable workflow patterns
- candidate route policies
- learned heuristics for future orchestration
- distilled summaries of what worked

Candidate workflow revisions belong here until approved. A failed proposal is still a wisdom artifact: it becomes a negative example, not a control-plane mutation.

### 3.5 Dreaming Loop

A dreaming agent should follow a small closed loop:

- observe stable traces and user feedback
- distill compact reasoning and candidate improvements
- write proposal artifacts into wisdom
- gate promotion through policy or human approval
- materialize pending or approved candidate revisions into workflow when execution evidence is needed
- keep accept / reject / deprecate decision records in wisdom via `proposal_evaluation`

The edge kind for proposal outcome is `proposal_evaluation`.
Use it for both `proposal -> approved workflow` and `proposal -> rejected lesson`.
It records the decision, rationale, and resulting target artifact or lesson artifact.

The loop may run in background maintenance mode or foreground assist mode.
In both modes, budget exhaustion must stop further dreaming work before it mutates workflow truth.

### 3.6 Dream Agent Own Workflow

The dreaming agent itself should be modeled as a workflow in `workflow`, not only as a helper function.

Its workflow can include steps such as:

- observe recent workflow outcomes and conversation feedback
- select signals under budget and policy gates
- generate self-to-self reasoning in `conversation`
- distill a proposal in `wisdom`
- evaluate accumulated evidence across multiple rounds
- materialize, keep pending, accept, deprecate, or revert a candidate workflow revision

This makes the dream loop inspectable, replayable, and revisable. It also lets us later dream about the dream agent itself.

### 3.7 Delayed Evaluation

Accept or deprecate does not need to happen in the same dream run as proposal creation.

If evidence is insufficient, the candidate stays pending or evaluating. Later dream cycles can keep collecting evidence from runtime bugs, performance signals, trusted positive/negative user feedback, or sabotage-like guidance that should count against naive approval. Only when policy thresholds are met should the system promote, reject, or deprecate the candidate.

Implementation note: the dream agent itself is now modeled as a first-class workflow with explicit runtime step execution. Its self-to-self reasoning stays in conversation, its compact proposal and evaluation stay in wisdom, and its executable revision lineage stays in workflow.

## 4. Test Plan

Add coverage for these scenarios:

- wisdom layer stores meta-patterns and does not become a live execution state holder
- a dreaming agent can emit compact planning traces in conversation without exposing raw private reasoning as system truth
- proposal generation respects token budget and policy gates
- failed proposal feeds back into wisdom as lesson / negative example
- candidate workflow stays pending until enough evaluation rounds have accumulated
- candidate workflow can be accepted, deprecated, or reverted after later evidence
- sabotage-like user feedback is evaluated as evidence, not blindly promoted into workflow truth
- a parent workflow invokes a child workflow by `workflow_id`
- child workflow invocation creates a fresh run and returns a usable result
- LLM-synthesized workflows are persisted before reuse
- `_route_next` with one route behaves as today
- `_route_next` with many routes fans out correctly
- fanout preserves join behavior and does not collapse branches unintentionally
- invalid synthesized workflows are rejected before execution
- child workflow failure propagates in a controlled result shape

## 5. Assumptions

- persisted workflow designs are the source of reuse
- runtime fanout is the primary mechanism for multi-route execution
- wisdom starts as a graph-native layer over existing traces and projections
- spec and implementation should stay compatible with the current conversation/workflow split

## 6. Notes

This ARD keeps the semantics strict:

- workflow design is reusable
- workflow execution is ephemeral per run
- wisdom is distilled from executions, not merged into them
- fanout is explicit and traceable
