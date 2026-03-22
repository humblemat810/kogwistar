# %% [markdown]
# # 15 Historical Search With Tombstone and Redirect
# Audit semantics demo only. Not medical guidance.

# %%
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kogwistar.conversation.service import ConversationService
from kogwistar.engine_core.engine import GraphKnowledgeEngine
from kogwistar.engine_core.models import Node

from _helpers import (
    LexicalHashEmbeddingFunction,
    banner,
    reset_data_dir,
    show,
    tutorial_grounding,
)


def _claim_node(
    *,
    node_id: str,
    label: str,
    summary: str,
    doc_id: str,
    effective_from: str,
) -> Node:
    return Node(
        id=node_id,
        label=label,
        type="entity",
        summary=summary,
        doc_id=doc_id,
        mentions=[
            tutorial_grounding(doc_id, label, insertion_method="tutorial_historical")
        ],
        properties={},
        metadata={"effective_from": effective_from, "topic": "historical_audit"},
        domain_id=None,
        canonical_entity_id=None,
        level_from_root=0,
        embedding=None,
    )


data_dir = reset_data_dir("15_historical_search_tombstone_redirect")
kg_engine = GraphKnowledgeEngine(
    persist_directory=str(data_dir / "knowledge"),
    kg_graph_type="knowledge",
    embedding_function=LexicalHashEmbeddingFunction(),
)
conv_engine = GraphKnowledgeEngine(
    persist_directory=str(data_dir / "conversation"),
    kg_graph_type="conversation",
    embedding_function=LexicalHashEmbeddingFunction(),
)
svc = ConversationService.from_engine(conv_engine, knowledge_engine=kg_engine)
conversation_id, _ = svc.create_conversation(
    "demo-user", "conv-historical", "conv-historical-start"
)
banner("Knowledge and conversation engines initialized.")

# %% [markdown]
# ## Seed two historical revisions
# Sugar/fat and egg/cholesterol examples use lifecycle redirects with cutover timestamps.

# %%
sugar_old = _claim_node(
    node_id="N_SUGAR_OLD",
    label="Sugar-Fat Claim (Old)",
    summary="Historic framing overemphasized fat and underemphasized added sugar.",
    doc_id="doc:historical:sugar",
    effective_from="1967-01-01T00:00:00+00:00",
)
sugar_new = _claim_node(
    node_id="N_SUGAR_NEW",
    label="Sugar-Fat Claim (Revised)",
    summary="Revised framing emphasizes added sugar risk and balanced dietary context.",
    doc_id="doc:historical:sugar",
    effective_from="2016-01-01T00:00:00+00:00",
)
egg_old = _claim_node(
    node_id="N_EGG_OLD",
    label="Egg Claim (Old)",
    summary="Historic framing strongly discouraged eggs for most diets.",
    doc_id="doc:historical:egg",
    effective_from="1970-01-01T00:00:00+00:00",
)
egg_new = _claim_node(
    node_id="N_EGG_NEW",
    label="Egg Claim (Revised)",
    summary="Revised framing allows eggs in broader dietary context.",
    doc_id="doc:historical:egg",
    effective_from="2015-01-01T00:00:00+00:00",
)

for node in (sugar_old, sugar_new, egg_old, egg_new):
    kg_engine.write.add_node(node)

kg_engine.redirect_node(
    "N_SUGAR_OLD",
    "N_SUGAR_NEW",
    deleted_at="2016-01-01T00:00:00+00:00",
    reason="historical_revision",
)
kg_engine.redirect_node(
    "N_EGG_OLD",
    "N_EGG_NEW",
    deleted_at="2015-01-01T00:00:00+00:00",
    reason="historical_revision",
)
show(
    "seeded lifecycle",
    {
        "old_nodes": ["N_SUGAR_OLD", "N_EGG_OLD"],
        "new_nodes": ["N_SUGAR_NEW", "N_EGG_NEW"],
        "cutovers": {
            "N_SUGAR_OLD": "2016-01-01T00:00:00+00:00",
            "N_EGG_OLD": "2015-01-01T00:00:00+00:00",
        },
    },
)

# %% [markdown]
# ## Search historical slices
# `search_nodes_as_of` exposes what is visible at the selected timestamp.

# %%
query_text = "sugar fat cholesterol eggs dietary claim"
then_ts = "2010-01-01T00:00:00+00:00"
now_ts = "2022-01-01T00:00:00+00:00"

hits_then = kg_engine.search_nodes_as_of(
    query=query_text, as_of_ts=then_ts, n_results=100
)
hits_now = kg_engine.search_nodes_as_of(
    query=query_text, as_of_ts=now_ts, n_results=100
)
then_ids = [n.id for n in hits_then]
now_ids = [n.id for n in hits_now]

show(
    "as_of search",
    {
        "as_of_then": then_ts,
        "then_ids": then_ids,
        "as_of_now": now_ts,
        "now_ids": now_ids,
    },
)

# %% [markdown]
# ## Explicit resolve_mode inspection
# This mirrors the legacy lifecycle read patterns for one old id.

# %%
mode_demo = {
    "active_only": [
        n.id
        for n in kg_engine.get_nodes(ids=["N_SUGAR_OLD"], resolve_mode="active_only")
    ],
    "redirect": [
        n.id for n in kg_engine.get_nodes(ids=["N_SUGAR_OLD"], resolve_mode="redirect")
    ],
    "include_tombstones": [
        n.id
        for n in kg_engine.get_nodes(
            ids=["N_SUGAR_OLD"], resolve_mode="include_tombstones"
        )
    ],
}
show("resolve mode comparison", mode_demo)

# %% [markdown]
# ## Persist context snapshots for auditing
# We persist two snapshots to show what a model context could have seen then vs now.

# %%
view = svc.get_conversation_view(
    conversation_id=conversation_id,
    user_id="demo-user",
    purpose="answer",
    budget_tokens=300,
    tail_turns=4,
)
snapshot_then = svc.persist_context_snapshot(
    conversation_id=conversation_id,
    run_id="historical-audit-run",
    run_step_seq=1,
    stage="historical_then",
    view=view,
    budget_tokens=300,
    tail_turn_index=1,
    llm_input_payload={
        "as_of_ts": then_ts,
        "evidence_node_ids": then_ids,
        "context_text": " ".join([n.summary for n in hits_then[:4]]),
    },
)
snapshot_now = svc.persist_context_snapshot(
    conversation_id=conversation_id,
    run_id="historical-audit-run",
    run_step_seq=2,
    stage="historical_now",
    view=view,
    budget_tokens=300,
    tail_turn_index=1,
    llm_input_payload={
        "as_of_ts": now_ts,
        "evidence_node_ids": now_ids,
        "context_text": " ".join([n.summary for n in hits_now[:4]]),
    },
)
payload_then = svc.get_context_snapshot_payload(snapshot_node_id=snapshot_then)
payload_now = svc.get_context_snapshot_payload(snapshot_node_id=snapshot_now)
show(
    "snapshot audit view",
    {
        "snapshot_then": snapshot_then,
        "snapshot_now": snapshot_now,
        "then_evidence_ids": (
            payload_then.get("properties", {}).get("llm_input_payload") or ""
        )[:400],
        "now_evidence_ids": (
            payload_now.get("properties", {}).get("llm_input_payload") or ""
        )[:400],
    },
)

# %% [markdown]
# ## Invariant
# Distinct time slices produce distinct visible evidence and audit snapshot artifacts.

# %%
checkpoint = {
    "checkpoint_pass": (
        "N_SUGAR_OLD" in then_ids
        and "N_SUGAR_NEW" in now_ids
        and snapshot_then != snapshot_now
    ),
    "then_ids": then_ids,
    "now_ids": now_ids,
    "snapshot_then": snapshot_then,
    "snapshot_now": snapshot_now,
    "invariant": "historical view and current canonical view can both be reconstructed",
}
show("checkpoint", checkpoint)
