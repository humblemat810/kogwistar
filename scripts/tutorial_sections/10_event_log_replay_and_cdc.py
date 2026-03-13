# %% [markdown]
# # 10 Event Log Replay and CDC
# This companion combines the runtime level 3 telemetry surface with the claw loop command path.

# %%
from runtime_tutorial_ladder import level3_observability_and_langgraph, reset_data
from tutorial_ladder import level3_command_hints

from _helpers import banner, reset_data_dir, show

runtime_data_dir = reset_data_dir("10_event_log_replay_and_cdc_runtime")
claw_hint_dir = reset_data_dir("10_event_log_replay_and_cdc_claw")
show("runtime reset", reset_data(runtime_data_dir))
banner("Inspecting runtime telemetry first, then the event-loop command path.")

# %% [markdown]
# ## Runtime trace inventory and viewer surface

# %%
runtime_level3 = level3_observability_and_langgraph(runtime_data_dir)
show("runtime level3", runtime_level3)

# %% [markdown]
# ## Claw loop commands
# The claw runtime is the script-backed example of durable inbox/outbox behavior.

# %%
claw_hints = level3_command_hints(claw_hint_dir)
show("claw loop hints", claw_hints)

# %% [markdown]
# ## Invariant
# Event-sourced flows expose inspectable traces and commandable replay/debug paths.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": bool(runtime_level3.get("checkpoint_pass")) and bool(claw_hints.get("commands")),
        "trace_event_types": runtime_level3.get("trace_event_types"),
        "invariant": "execution history remains observable through trace sink and event-loop surfaces",
    },
)
