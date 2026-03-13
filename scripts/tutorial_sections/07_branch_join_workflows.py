# %% [markdown]
# # 07 Branch Join Workflows
# This walkthrough focuses on resolver telemetry and suspend/resume.

# %%
from runtime_tutorial_ladder import level1_resolvers_and_deps, level2_pause_and_resume, reset_data

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("07_branch_join_workflows")
show("reset", reset_data(data_dir))
banner("The same canonical workflow is used across both cells below.")

# %% [markdown]
# ## Resolver registration and injected dependencies

# %%
level1 = level1_resolvers_and_deps(data_dir)
show("runtime level1", level1)

# %% [markdown]
# ## Pause and resume
# Level 2 recovers the suspended token from checkpoint state and resumes it.

# %%
level2 = level2_pause_and_resume(data_dir)
show("runtime level2", level2)

# %% [markdown]
# ## Invariant
# Fanout, join, and resume happen through persisted runtime state rather than hidden control flow.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": bool(level1.get("checkpoint_pass")) and bool(level2.get("checkpoint_pass")),
        "resumed_status": level2.get("resumed_status"),
        "invariant": "branch and join behavior remains inspectable through checkpoints and traces",
    },
)
