# %% [markdown]
# # 20 Generic Named Projection Meta Layer
# This companion shows a domain-specific bridge-governance projector using the generic named-projection substrate.
# The authoritative history stays append-only in the entity event log; the latest-state projection lives in meta/sql with freshness watermarks.

# %%
from kogwistar.demo.named_projection_governance_demo import (
    run_named_projection_governance_demo,
)

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("20_generic_named_projection_meta_layer")
banner("Running the generic named-projection demo with a fixed resettable data directory.")
demo = run_named_projection_governance_demo(data_dir=data_dir, reset_data=False)
show("summary", demo["summary"])

# %% [markdown]
# ## Bridge governance projection
# The service folds authoritative history into one named projection row per interaction id.

# %%
show(
    "interaction alpha",
    {
        "projection": demo["details"]["interaction_alpha"]["projection"],
        "status_transitions": demo["details"]["interaction_alpha"]["status_transitions"],
        "rebuilt_projection": demo["details"]["interaction_alpha"]["rebuilt_projection"],
    },
)

# %% [markdown]
# ## Projection namespace operations
# The generic substrate supports list, clear one key, and clear the whole namespace without knowing anything about bridge-governance semantics.

# %%
show(
    "namespace operations",
    {
        "projection_keys_before_clear": demo["summary"]["projection_keys_before_clear"],
        "namespace_clear_removed_all": demo["summary"]["namespace_clear_removed_all"],
        "projections_after_namespace_clear": demo["details"]["projections_after_namespace_clear"],
    },
)

# %% [markdown]
# ## Invariant
# The projection is rebuildable latest-state convenience only. Authoritative truth remains the append-only event log, and the projection row tracks `last_authoritative_seq`, `last_materialized_seq`, schema version, and status.

# %%
show(
    "checkpoint",
    {
        "rebuilt_matches_before_clear": demo["summary"]["rebuilt_matches_before_clear"],
        "alpha_projection_payload": demo["details"]["interaction_alpha"]["projection"]["payload"],
        "beta_projection_payload": demo["details"]["interaction_beta"]["projection"]["payload"],
        "data_dir": demo["summary"]["data_dir"],
    },
)
