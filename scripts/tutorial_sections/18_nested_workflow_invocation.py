# %% [markdown]
# # 18 Nested Workflow Invocation
# This companion is a golden example for both predesigned and dynamically persisted child workflows.

# %%
from kogwistar.demo import run_nested_workflow_invocation_demo

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("18_nested_workflow_invocation")
banner("Running a deterministic nested-workflow example with a fixed resettable data directory.")
demo = run_nested_workflow_invocation_demo(data_dir=data_dir, reset_data=False)
show("summary", demo["summary"])

# %% [markdown]
# ## Predesigned child workflow
# The parent invokes an already-persisted child by `workflow_id` only.

# %%
show(
    "predesigned child",
    {
        "workflow_shape": demo["details"]["workflow_shapes"]["predesigned_child"],
        "step_ops": demo["details"]["conversation_trace"]["predesigned_child_step_ops"],
        "child_result": demo["details"]["final_state"]["predesigned_child"],
    },
)

# %% [markdown]
# ## Dynamically generated child workflow
# The "planner" payload is fake and predetermined, but the child workflow graph is inserted during the run and then executed.

# %%
show(
    "dynamic child",
    {
        "planner_payload": demo["details"]["planner_payload"],
        "workflow_shape": demo["details"]["workflow_shapes"]["dynamic_child"],
        "step_ops": demo["details"]["conversation_trace"]["dynamic_child_step_ops"],
        "child_result": demo["details"]["final_state"]["dynamic_child"],
    },
)

# %% [markdown]
# ## Invariant
# Both child workflows are visible in the workflow graph, while their execution traces are visible in the conversation graph.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": (
            demo["summary"]["status"] == "succeeded"
            and len(demo["details"]["workflow_shapes"]["predesigned_child"]["node_ids"]) > 0
            and len(demo["details"]["workflow_shapes"]["dynamic_child"]["node_ids"]) > 0
            and len(demo["details"]["conversation_trace"]["predesigned_child_step_ops"]) > 0
            and len(demo["details"]["conversation_trace"]["dynamic_child_step_ops"]) > 0
        ),
        "data_dir": demo["summary"]["data_dir"],
        "invariant": "predesigned and dynamically persisted child workflows are both inspectable as workflow graphs plus conversation-side execution traces",
    },
)
