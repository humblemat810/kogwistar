# %% [markdown]
# # 05 Context Snapshot and Replay
# This companion persists a prompt view as a `context_snapshot`.

# %%
from kogwistar.conversation.service import ConversationService
from tutorial_ladder import (
    _ensure_seed,
    deterministic_filter_callback,
    reset_data,
    seed_data,
)

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("05_context_snapshot_and_replay")
show("reset", reset_data(data_dir))
show("seed", seed_data(data_dir))
kg_engine, conv_engine = _ensure_seed(data_dir)
svc = ConversationService.from_engine(conv_engine, knowledge_engine=kg_engine)
conversation_id, _start_node_id = svc.create_conversation(
    "demo-user", "conv-snapshot", "conv-snapshot-start"
)
turn = svc.add_conversation_turn(
    user_id="demo-user",
    conversation_id=conversation_id,
    turn_id="conv-snapshot-turn-1",
    mem_id="conv-snapshot-mem",
    role="user",
    content="Explain why context snapshots matter for replay.",
    ref_knowledge_engine=kg_engine,
    filtering_callback=deterministic_filter_callback,
    max_retrieval_level=2,
    add_turn_only=True,
)
banner(
    "Conversation seeded. Next cell assembles a prompt view under a small token budget."
)

# %% [markdown]
# ## Build the prompt view
# The service packs summaries, memory context, pinned refs, and tail turns into one prompt-facing artifact.

# %%
view = svc.get_conversation_view(
    conversation_id=conversation_id,
    user_id="demo-user",
    purpose="answer",
    budget_tokens=256,
    tail_turns=4,
)
show(
    "prompt view",
    {
        "message_count": len(view.messages),
        "tokens_used": view.tokens_used,
        "included_node_ids": list(view.included_node_ids),
        "dropped_count": len(view.dropped),
    },
)

# %% [markdown]
# ## Persist a snapshot
# This records what the model-facing context actually looked like.

# %%
snapshot_id = svc.persist_context_snapshot(
    conversation_id=conversation_id,
    run_id="tutorial-sections-run",
    run_step_seq=1,
    stage="tutorial_snapshot",
    view=view,
    budget_tokens=256,
    tail_turn_index=int(turn.turn_index),
    llm_input_payload={"question": "Explain why context snapshots matter for replay."},
)
snapshot_node = svc.latest_context_snapshot_node(
    conversation_id=conversation_id,
    run_id="tutorial-sections-run",
    stage="tutorial_snapshot",
)
payload = svc.get_context_snapshot_payload(snapshot_node_id=snapshot_id)
depends_on_edges = conv_engine.get_edges(
    where={"run_id": "tutorial-sections-run"}, limit=50
)
show(
    "snapshot artifacts",
    {
        "snapshot_id": snapshot_id,
        "latest_snapshot_node_id": None if snapshot_node is None else snapshot_node.id,
        "depends_on_edge_count": len(depends_on_edges),
        "snapshot_metadata": payload.get("metadata", {}),
    },
)

# %% [markdown]
# ## Invariant
# Prompt construction can be persisted and replayed as a graph artifact.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": snapshot_node is not None and len(depends_on_edges) > 0,
        "invariant": "context assembly is replayable through context_snapshot nodes",
    },
)
