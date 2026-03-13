# 01 Hello Graph Engine

Audience: Beginner / evaluator
Time: 10-15 minutes
Companion: [scripts/tutorial_sections/01_hello_graph_engine.py](../../scripts/tutorial_sections/01_hello_graph_engine.py)

## What You Will Build

You will create a tiny graph with a few nodes and edges, query it back, persist it to disk, and reopen it to confirm the data survives restart.

## Why This Matters

This is the fastest proof that the repo is an actual graph engine, not just a prompt wrapper. If this step feels solid, the later retrieval and workflow tutorials have a concrete base.

## Run or Inspect

- Open the companion file in VS Code and run the cells one section at a time.
- Or run the whole file with `python scripts/tutorial_sections/01_hello_graph_engine.py`.
- Inspect the engine primitives in `graph_knowledge_engine/engine_core/engine.py` and `graph_knowledge_engine/engine_core/models.py`.

## Inspect The Result

- Confirm the three seeded node ids can be read back.
- Confirm one relation edge survives the reopen.
- Inspect the `.gke-data/tutorial-sections/01_hello_graph_engine` directory after the final cell runs.

## Invariant Demonstrated

Persistence survives restart. A graph written by the engine can be reopened without losing node identity or relation structure.

## Next Tutorial

Continue to [02 Core Data Model](./02_core_data_model.md).
