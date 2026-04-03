# %% [markdown]
# # 19 Build Artifact Governance Workflow
# This companion proves that source maps and raw sources are filtered at a real public boundary before publish.
# The public payload is derived from the model slice helper, not manual field stripping.

# %%
from kogwistar.demo import run_build_artifact_governance_demo

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("19_build_artifact_governance_workflow")
banner("Running the artifact-governance workflow with a fixed resettable data directory.")
demo = run_build_artifact_governance_demo(data_dir=data_dir, reset_data=False)
show("summary", demo["summary"])

# %% [markdown]
# ## Safe path
# The safe workflow applies the `public` slice before validation and publish.

# %%
show(
    "safe path",
    {
        "step_ops": demo["details"]["safe"]["step_ops"],
        "event_types": demo["details"]["safe"]["event_types"],
        "published_artifact": demo["details"]["safe"]["final_state"]["published_artifact"],
        "replay_state": demo["details"]["safe"]["replay_state"],
    },
)

# %% [markdown]
# ## Blocked path
# The unsafe workflow skips filtering, hits the invariant, emits `artifact_rejected`, and never publishes.

# %%
show(
    "blocked path",
    {
        "step_ops": demo["details"]["unsafe"]["step_ops"],
        "event_types": demo["details"]["unsafe"]["event_types"],
        "validation_errors": demo["details"]["unsafe"]["final_state"]["validation_errors"],
        "replay_state": demo["details"]["unsafe"]["replay_state"],
    },
)

# %% [markdown]
# ## Invariant
# Public artifacts never contain source maps, raw sources, or sensitive metadata, and the blocked path is replayable from event-sourced execution history.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": demo["summary"]["invariant_pass"],
        "safe_public_artifact": demo["details"]["safe"]["final_state"]["published_artifact"],
        "unsafe_status": demo["summary"]["unsafe_status"],
        "data_dir": demo["summary"]["data_dir"],
    },
)
