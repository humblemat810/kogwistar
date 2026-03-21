# ARD: Custom Runtime Rationale

## Status

Accepted

## Context

The original intent was not to build a large custom orchestration framework. The goal was to have a simple runtime that could execute workflow designs and prove that the workflow graph could be converted into something runnable on a real platform.

At the time, a natural path was to rely more heavily on external orchestration libraries such as LangChain, LangGraph, and MCP integrations.

## Problem

In practice, mixing FastMCP, LangChain, LangGraph, and the surrounding testing/debugging loop led to poor developer experience:

- early implementation async integration only, debugger often locked up in vscode.
- debugger lockups
- unclear execution behavior during development
- harder-to-control runtime semantics
- more friction when trying to test workflow design as a real runnable surface

Even if those integrations later improve, the local development experience during implementation was bad enough that it became a systems constraint.

## Decision

Build a small synchronous native runtime for this repo.

The runtime should:

- stay simple enough to debug locally
- preserve explicit control over execution, replay, and state behavior
- keep workflow design visibly runnable rather than only declarative
- remain compatible with conversion or interop paths to other workflow platforms where useful

## Consequences

### Positive

- Better local debuggability and development control.
- Easier reasoning about replay, checkpoints, and provenance.
- Less dependency on external runtime integration behavior for core workflow semantics.

### Negative

- The repo carries its own runtime surface instead of delegating completely to a more widely adopted framework.
- Interop with external workflow ecosystems is an additional maintenance concern rather than the default execution path.
- Some readers may expect LangGraph-style reuse and wonder why a native runtime exists.

## Rationale

The custom runtime exists because runtime semantics, replay, provenance, and workflow design validation are core to the architecture, and the external stack available during implementation did not provide acceptable developer experience for those goals.

This should be understood as a pragmatic architecture decision, not as a blanket rejection of LangGraph, LangChain, FastMCP, or future interop.
