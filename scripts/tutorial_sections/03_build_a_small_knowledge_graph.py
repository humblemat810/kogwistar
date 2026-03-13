# %% [markdown]
# # 03 Build a Small Knowledge Graph
# This companion reuses the existing tutorial ladder seed and level helpers.

# %%
from tutorial_ladder import _ensure_seed, reset_data, run_level0, seed_data

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("03_build_a_small_knowledge_graph")
show("reset", reset_data(data_dir))
show("seed", seed_data(data_dir))
kg_engine, _conv_engine = _ensure_seed(data_dir)
banner("The small tutorial graph is now persisted and queryable.")

# %% [markdown]
# ## Inspect the seeded entities
# The tutorial ladder uses stable ids so readers can inspect the graph by name.

# %%
nodes = kg_engine.get_nodes(["K:architecture", "K:provenance", "K:ttl_guardrail"], resolve_mode="redirect")
edges = kg_engine.get_edges(["E:arch->retrieval", "E:retrieval->prov"], resolve_mode="redirect")
show(
    "seeded graph primitives",
    {
        "node_ids": [node.id for node in nodes],
        "edge_ids": [edge.id for edge in edges],
        "node_summaries": {node.id: node.summary for node in nodes},
    },
)

# %% [markdown]
# ## Ask the baseline retrieval question
# This is the same Level 0 proof used by the CLI ladder.

# %%
level0 = run_level0(data_dir, question="How does this repo implement simple RAG?")
show("level0 baseline", level0)

# %% [markdown]
# ## Invariant
# Retrieved evidence is named, persisted graph structure, not anonymous text chunks.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": bool(level0.get("evidence")),
        "invariant": "retrieval returns graph-native evidence",
    },
)
