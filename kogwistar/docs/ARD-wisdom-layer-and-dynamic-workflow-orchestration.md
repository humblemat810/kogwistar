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

### 2.2 Workflow Invocation Semantics

A workflow step may invoke another workflow by `workflow_id`.

- Reuse means reuse the persisted workflow design, not the prior execution instance.
- Each invocation creates a fresh child run with its own trace/checkpoint history.
- The parent step consumes the child result and continues execution.
- This is an explicit runtime capability.

### 2.3 Dynamic Workflow Design

An LLM may synthesize a new workflow on the fly.

- The synthesized workflow is persisted as a first-class workflow artifact.
- The runtime executes it through the normal persisted workflow path.
- The generated design remains inspectable and reusable after the run.

### 2.4 `_route_next` Fanout

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

## 4. Test Plan

Add coverage for these scenarios:

- wisdom layer stores meta-patterns and does not become a live execution state holder
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
