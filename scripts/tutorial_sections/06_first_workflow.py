# %% [markdown]
# # 06 First Workflow
# This companion stays close to the runtime tutorial ladder and reuses its public helpers.

# %%
from runtime_tutorial_ladder import level0_runtime_basics, reset_data

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("06_first_workflow")
show("reset", reset_data(data_dir))
banner("Running WorkflowRuntime Level 0 on the canonical branch/join workflow.")

# %% [markdown]
# ## Execute the workflow to its first suspension point

# %%
level0 = level0_runtime_basics(data_dir)
show("runtime level0", level0)

# %% [markdown]
# ## Invariant
# The run persists enough state to inspect where it paused.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": level0.get("checkpoint_pass"),
        "status": level0.get("status"),
        "invariant": "workflow execution persists checkpoints and step records",
    },
)
