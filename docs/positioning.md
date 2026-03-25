# Positioning

Kogwistar is a graph-native system for storing and replaying knowledge, conversations, workflows, and provenance in one model.

It is easiest to understand as two layers:

- `substrate`: the shared node/edge/event foundation
- `harness`: the runtime and orchestration layer that executes on top of that foundation

## Core Terms

- `node` and `edge`: the basic graph entities used to represent facts, relations, steps, and links
- `event`: a durable record of something that happened
- `provenance`: the trace back to source data, source spans, or source execution
- `replay`: rebuilding state from stored history
- `projection`: turning history into queryable views or materialized state
- `authoritative evented path`: the path where event history is treated as the source of truth

## How The Layers Work

The substrate layer is the common representation. Workflow structure, conversation structure, knowledge, provenance, and replay-oriented history all share the same graph-oriented model. That makes lifecycle state and structural relationships first-class instead of incidental metadata.

The harness layer is the execution surface. It runs workflows, coordinates multi-step behavior, and exposes developer-facing control paths. This is where the graph model becomes runnable.

The important distinction is that not every path has the same guarantees. The authoritative evented path is the one that supports the strongest replay, provenance, and projection semantics. Lower-level primitives remain available for advanced composition, but they should not be described as if they automatically provide the same invariants.

## What This Means In Practice

When input arrives, it is captured as durable history, normalized into graph artifacts, linked to related artifacts, and then projected into views that can be queried later.

That is why Kogwistar is not best described as only a GraphRAG repository or only a workflow runner. It is an attempt to unify execution, structure, and provenance on one graph-native system model.

## Relationship To Other Tools

Compared with runtime-first frameworks such as LangGraph, Kogwistar overlaps at the harness layer but is broader in scope. The runtime is important, but it is not the whole product surface. The system model also treats structure, provenance, replay, and projection as core concerns.
