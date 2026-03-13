# %% [markdown]
# # 01 Hello Graph Engine
# Run this file in VS Code with `Run Cell` for a notebook-like walkthrough.
# It stays plain Python so it can also be executed directly with:
# `python scripts/tutorial_sections/01_hello_graph_engine.py`

# %%
from graph_knowledge_engine.engine_core.engine import GraphKnowledgeEngine
from graph_knowledge_engine.engine_core.models import Edge, Node

from _helpers import (
    LexicalHashEmbeddingFunction,
    banner,
    reset_data_dir,
    show,
    tutorial_grounding,
)

data_dir = reset_data_dir("01_hello_graph_engine")
engine = GraphKnowledgeEngine(
    persist_directory=str(data_dir / "knowledge"),
    kg_graph_type="knowledge",
    embedding_function=LexicalHashEmbeddingFunction(),
)
banner("Fresh engine created under .gke-data/tutorial-sections/01_hello_graph_engine")

# %% [markdown]
# ## Seed a tiny graph
# The example stays deliberately small: three nodes and two edges.

# %%
nodes = [
    Node(
        id="hello:engine",
        label="Graph Engine",
        type="entity",
        summary="The engine persists graph nodes and edges.",
        doc_id="hello:engine",
        mentions=[
            tutorial_grounding(
                "hello:engine", "The engine persists graph nodes and edges."
            )
        ],
        properties={},
        metadata={"level_from_root": 0},
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    ),
    Node(
        id="hello:persistence",
        label="Persistence",
        type="entity",
        summary="State survives restart when written to the engine persist directory.",
        doc_id="hello:persistence",
        mentions=[
            tutorial_grounding(
                "hello:persistence",
                "State survives restart when written to the engine persist directory.",
            )
        ],
        properties={},
        metadata={"level_from_root": 1},
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=1,
        embedding=None,
    ),
    Node(
        id="hello:provenance",
        label="Provenance",
        type="entity",
        summary="Every graph write can carry provenance-bearing mentions.",
        doc_id="hello:provenance",
        mentions=[
            tutorial_grounding(
                "hello:provenance",
                "Every graph write can carry provenance-bearing mentions.",
            )
        ],
        properties={},
        metadata={"level_from_root": 1},
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=1,
        embedding=None,
    ),
]

edges = [
    Edge(
        id="hello:engine->persistence",
        source_ids=["hello:engine"],
        target_ids=["hello:persistence"],
        relation="supports",
        label="supports",
        type="relationship",
        summary="The graph engine supports persistence across restarts.",
        doc_id="hello:edge:1",
        mentions=[
            tutorial_grounding(
                "hello:edge:1", "The graph engine supports persistence across restarts."
            )
        ],
        properties={},
        metadata={"level_from_root": 1},
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    ),
    Edge(
        id="hello:persistence->provenance",
        source_ids=["hello:persistence"],
        target_ids=["hello:provenance"],
        relation="preserves",
        label="preserves",
        type="relationship",
        summary="Persistence preserves provenance-bearing writes for later inspection.",
        doc_id="hello:edge:2",
        mentions=[
            tutorial_grounding(
                "hello:edge:2",
                "Persistence preserves provenance-bearing writes for later inspection.",
            )
        ],
        properties={},
        metadata={"level_from_root": 1},
        source_edge_ids=[],
        target_edge_ids=[],
        domain_id=None,
        canonical_entity_id=None,
        embedding=None,
    ),
]

for node in nodes:
    engine.add_node(node)
for edge in edges:
    engine.add_edge(edge)

show(
    "seeded ids", {"node_ids": [n.id for n in nodes], "edge_ids": [e.id for e in edges]}
)

# %% [markdown]
# ## Query the graph back
# First read by id, then run a simple similarity query.

# %%
read_back = engine.get_nodes(
    ["hello:engine", "hello:persistence", "hello:provenance"], resolve_mode="redirect"
)
query_hits = engine.query_nodes(
    query_embeddings=[
        engine._iterative_defensive_emb("restart persistence provenance")
    ],
    n_results=3,
    where={"level_from_root": {"$lte": 3}},
    include=["metadatas", "documents", "embeddings"],
)[0]

show(
    "query results",
    {
        "read_back_ids": [node.id for node in read_back],
        "query_hit_ids": [node.id for node in query_hits],
    },
)

# %% [markdown]
# ## Reopen the engine
# Reconstruct a fresh engine object on the same persist directory and confirm the ids remain available.

# %%
reopened = GraphKnowledgeEngine(
    persist_directory=str(data_dir / "knowledge"),
    kg_graph_type="knowledge",
    embedding_function=LexicalHashEmbeddingFunction(),
)
reopened_nodes = reopened.get_nodes(
    ["hello:engine", "hello:persistence", "hello:provenance"], resolve_mode="redirect"
)
show(
    "reopened state",
    {
        "persist_directory": str(data_dir / "knowledge"),
        "reopened_ids": [node.id for node in reopened_nodes],
        "checkpoint_pass": len(reopened_nodes) == 3,
        "invariant": "persistence survives restart",
    },
)
