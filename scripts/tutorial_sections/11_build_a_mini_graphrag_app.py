# %% [markdown]
# # 11 Build a Mini GraphRAG App
# This companion stitches together the existing RAG ladder into one short walkthrough.

# %%
from tutorial_ladder import reset_data, run_level0, run_level1, run_level2, seed_data

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("11_build_a_mini_graphrag_app")
show("reset", reset_data(data_dir))
show("seed", seed_data(data_dir))
banner(
    "The next cells walk from baseline retrieval to seeded retrieval to provenance pinning."
)

# %% [markdown]
# ## Baseline retrieval and answer

# %%
level0 = run_level0(data_dir, question="How does this repo implement simple RAG?")
show("rag level0", level0)

# %% [markdown]
# ## Seeded retrieval expansion

# %%
level1 = run_level1(
    data_dir,
    question="How does architecture reinforce retrieval?",
    max_retrieval_level=2,
)
show("rag level1", level1)

# %% [markdown]
# ## Provenance pinning into the conversation graph

# %%
level2 = run_level2(
    data_dir,
    question="Show evidence and provenance for retrieval decisions.",
    max_retrieval_level=2,
)
show("rag level2", level2)

# %% [markdown]
# ## Invariant
# The answer path leaves graph-native evidence, expansion, and provenance artifacts behind.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": bool(level0.get("checkpoint_pass"))
        and bool(level1.get("checkpoint_pass"))
        and bool(level2.get("checkpoint_pass")),
        "invariant": "retrieval, seeded expansion, and provenance projection are all inspectable",
    },
)
