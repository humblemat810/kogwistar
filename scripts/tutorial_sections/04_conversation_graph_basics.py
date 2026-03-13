# %% [markdown]
# # 04 Conversation Graph Basics
# This file shows that conversation artifacts live in the graph as first-class nodes.

# %%
from graph_knowledge_engine.conversation.service import ConversationService
from tutorial_ladder import (
    _ensure_seed,
    deterministic_filter_callback,
    reset_data,
    seed_data,
)

from _helpers import banner, reset_data_dir, show

data_dir = reset_data_dir("04_conversation_graph_basics")
show("reset", reset_data(data_dir))
show("seed", seed_data(data_dir))
kg_engine, conv_engine = _ensure_seed(data_dir)
svc = ConversationService.from_engine(conv_engine, knowledge_engine=kg_engine)
conversation_id, start_node_id = svc.create_conversation(
    "demo-user", "conv-sections", "conv-sections-start"
)
banner("Conversation created. Next cell appends a user turn with add_turn_only=True.")

# %% [markdown]
# ## Append one turn
# The tutorial keeps this example small and avoids a full answer flow.

# %%
turn = svc.add_conversation_turn(
    user_id="demo-user",
    conversation_id=conversation_id,
    turn_id="conv-sections-turn-1",
    mem_id="conv-sections-mem",
    role="user",
    content="Why model conversation as a graph instead of a flat transcript?",
    ref_knowledge_engine=kg_engine,
    filtering_callback=deterministic_filter_callback,
    max_retrieval_level=2,
    add_turn_only=True,
)
all_nodes = conv_engine.get_nodes(limit=200)
conversation_nodes = [
    node
    for node in all_nodes
    if getattr(node, "conversation_id", None) == conversation_id
]
seeded_refs = [
    node
    for node in all_nodes
    if (getattr(node, "metadata", {}) or {}).get("entity_type") == "knowledge_reference"
]
show(
    "conversation graph snapshot",
    {
        "conversation_id": conversation_id,
        "start_node_id": start_node_id,
        "user_turn_node_id": turn.user_turn_node_id,
        "conversation_node_ids": [node.id for node in conversation_nodes],
        "seeded_reference_pointer_ids": [node.id for node in seeded_refs[:5]],
    },
)

# %% [markdown]
# ## Inspect the ordering fields
# Conversation nodes preserve graph identity and turn order while allowing non-turn artifacts to coexist.

# %%
turn_like = [
    {
        "id": node.id,
        "entity_type": (getattr(node, "metadata", {}) or {}).get("entity_type"),
        "turn_index": getattr(node, "turn_index", None),
        "conversation_id": getattr(node, "conversation_id", None),
    }
    for node in conversation_nodes
]
show("ordered artifacts", {"nodes": turn_like})

# %% [markdown]
# ## Invariant
# Turn order and sidecar artifacts can coexist in one conversation graph.

# %%
show(
    "checkpoint",
    {
        "checkpoint_pass": bool(turn.user_turn_node_id) and bool(conversation_nodes),
        "invariant": "conversation graph stores ordered turns plus sidecar artifacts",
    },
)
