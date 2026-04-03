# Scoped Sequence Policy

## Governance Example

Use the generic scoped-sequence hook when a graph needs append-only ordering within a logical stream such as `agent_interaction_id`.

Recommended pattern:

```python
from kogwistar.engine_core.scoped_seq import (
    ScopedSeqHookConfig,
    install_scoped_seq_hooks,
)


install_scoped_seq_hooks(
    engine,
    ScopedSeqHookConfig(
        metadata_field="seq",
        should_stamp_node=lambda _e, node: (
            (getattr(node, "metadata", {}) or {}).get("entity_type")
            in {"governance_turn", "governance_event"}
        ),
        scope_id_for_node=lambda _e, node: (
            f"governance:{(getattr(node, 'metadata', {}) or {}).get('agent_interaction_id')}"
            if (getattr(node, "metadata", {}) or {}).get("agent_interaction_id")
            else None
        ),
    ),
    ready_attr="_governance_scoped_seq_hooks_ready",
)
```

Rules to follow:

- Always return a namespaced scope id such as `governance:{agent_interaction_id}`. Do not return the raw id if another subsystem might use the same string.
- Only stamp the node or edge types that belong to the append-only governance history.
- Treat `metadata["seq"]` as a write-time ordering stamp, and read the latest watermark through `engine.meta_sqlite.current_scoped_seq(scope_id)`.
- Keep the scope getter stable over time. If the scope id format changes, the sequence stream changes too.
- Prefer the generic scoped API for new code. The `*_user_seq(...)` methods remain as compatibility aliases for older conversation code.

Notes:

- The current storage backend still persists scoped counters in the legacy `user_seq` table/keyspace. The generic API is semantic sugar over that storage for now.
- Because of that legacy storage, namespacing the scope id is the safest way to avoid cross-domain counter collisions.

## Field-Presence Filter Example

If a node should only be stamped when it carries a field such as `governance_section_id`, express that in `should_stamp_node` and `scope_id_for_node`.

```python
from kogwistar.engine_core.scoped_seq import (
    ScopedSeqHookConfig,
    install_scoped_seq_hooks,
)


install_scoped_seq_hooks(
    engine,
    ScopedSeqHookConfig(
        should_stamp_node=lambda _e, node: bool(
            (getattr(node, "metadata", {}) or {}).get("governance_section_id")
        ),
        scope_id_for_node=lambda _e, node: (
            f"governance_section:{(getattr(node, 'metadata', {}) or {}).get('governance_section_id')}"
            if (getattr(node, "metadata", {}) or {}).get("governance_section_id")
            else None
        ),
    ),
    ready_attr="_governance_section_seq_hooks_ready",
)
```

This means:

- nodes without `governance_section_id` are ignored
- nodes with the field get a `metadata["seq"]`
- the counter is scoped per `governance_section:{id}`
