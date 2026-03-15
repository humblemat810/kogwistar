# Author Notes

This document captures practical motivation behind the repo that does not belong in the main product-facing README flow.

## Build Cost

The rough build cost was about 3-4 months of ChatGPT subscription spend, plus the author time required to repeatedly refine architecture, experiments, and implementation direction. So there is no token bill shock.

This is not meant as a pricing claim. It is a rough statement of the iteration cost behind the repo.

## Why Build This

- Raise the engineering bar for graph-native agent systems.
- Avoid every team rebuilding the same retrieval, memory, workflow, and provenance stack from scratch.
- Build a reusable foundation where those concerns are already structurally integrated.

## Why Open Source It

- Make the architecture legible.
- Preserve the design work as a reusable base instead of a one-off internal prototype.
- Reduce repeated effort across teams trying to assemble similar systems from disconnected tools.

## Design History

- Early phase:
  - The repo started from handwritten node and edge abstractions, with ChatGPT used mainly for design discussion and architecture thinking.
- Middle phase:
  - Some code was copied into ChatGPT discussions and adapted back into the repo after review and iteration.
- Recent phase:
  - Codex-style coding agents became part of the implementation loop for faster iteration, refactoring support, and documentation work.

The important point is that the repo evolved through multiple stages of human-led design and AI-assisted iteration, rather than appearing all at once from one generation pass.

## Development Context

This system was developed as a solo-not-even-a-preneur engineering project.

It is not part of a startup or venture-backed initiative.  
The goal is exploration of graph-native AI infrastructure and knowledge systems.

At the time of writing, the author prefers a quiet, research-oriented approach and focuses on building reliable systems rather than pursuing hype-driven development, but the author does inspect and review existing work to avoid reinventing the wheel.