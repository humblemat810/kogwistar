# Positioning

Kogwistar is a graph-native substrate for storing and replaying knowledge, conversations, workflows, governance semantics, and provenance in one model.

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

The substrate layer is the common representation. Workflow structure, conversation structure, knowledge, governance artifacts, provenance, and replay-oriented history all share the same graph-oriented model. That makes lifecycle state and structural relationships first-class instead of incidental metadata.

The harness layer is the execution surface. It runs workflows, coordinates multi-step behavior, exposes developer-facing control paths, and provides seams where policy, approval, and other governance logic can become runnable.

The important distinction is that not every path has the same guarantees. The authoritative evented path is the one that supports the strongest replay, provenance, and projection semantics. Lower-level primitives remain available for advanced composition, but they should not be described as if they automatically provide the same invariants.

## What This Means In Practice

When input arrives, it is captured as durable history, normalized into graph artifacts, linked to related artifacts, and then projected into views that can be queried later. The same model can also capture governance decisions, approval state, and security-relevant boundaries as inspectable artifacts instead of opaque side effects.

That is why Kogwistar is not best described as only a GraphRAG repository, only a memory system, or only a workflow runner. It is an attempt to unify execution, structure, provenance, and governance on one graph-native system model.

## Security And Governance Framing

The repo includes memory, conversation, and workflow capabilities, but those are not the ceiling of the design. A useful way to read the substrate is as the lower layer beneath security and governance semantics for agent systems:

- tool calls can be represented as durable events
- approval or policy outcomes can be persisted as explicit graph artifacts
- replay and projection make governance receipts inspectable later
- privacy and slicing boundaries can be encoded in data/model paths rather than left as conventions

A concrete example is [`humblemat810/cloistar`](https://github.com/humblemat810/cloistar), which uses Kogwistar as the substrate for an OpenClaw governance semantics layer. That example is closer to "security/governance backplane for agent execution" than to "chat memory demo", and it is the best external reference point for this broader framing.

At a lighter and more application-facing layer, [`humblemat810/kogwistar-chat`](https://github.com/humblemat810/kogwistar-chat) is a useful complementary example. Together, these examples show that the substrate can support multiple levels of implementation:

- application layer on top of the substrate
- workflow/runtime integration in the middle
- governance and security semantics at a deeper layer

## Relationship To Other Tools

Compared with runtime-first frameworks such as LangGraph, Kogwistar overlaps at the harness layer but is broader in scope. The runtime is important, but it is not the whole product surface. The system model also treats structure, provenance, replay, projection, and governance-carrying artifacts as core concerns.
