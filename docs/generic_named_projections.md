# Generic Named Projections

The generic named-projection meta layer is the reusable substrate for latest-state views that are derived from authoritative append-only history.

## What Is Generic

The meta store owns only the container mechanics:

- one row per `namespace + key`
- arbitrary JSON `payload`
- `last_authoritative_seq`
- `last_materialized_seq`
- `projection_schema_version`
- `materialization_status`
- `updated_at_ms`

Backends expose the same CRUD surface:

- `get_named_projection(namespace, key)`
- `replace_named_projection(namespace, key, payload, *, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status)`
- `list_named_projections(namespace)`
- `clear_named_projection(namespace, key)`
- `clear_projection_namespace(namespace)`

Feature code should never write backend-specific SQL for these rows directly.

## What Is Not Generic

The fold algorithm is always domain-specific.

The generic substrate does not decide:

- how history is interpreted
- what the projection payload shape means
- how conflicts are resolved
- when a stale projection should be rebuilt

That logic belongs in a service-layer projector such as:

- `get_bridge_governance_projection(...)`
- `refresh_bridge_governance_projection(...)`
- `sync_bridge_governance_projection(...)`

The same shape applies to workflow, governance, wisdom, and any future projection consumer.

## Why `seq` Is Metadata, Not The Projector

`seq` only tells you freshness and replay position.

It is useful for:

- ordering authoritative events
- telling whether a projection is caught up
- deciding whether rebuild is needed

It is not the projection algorithm itself. The projection still needs a domain-specific fold over history plus a payload schema that makes sense for that domain.

## Choosing Namespace And Key

Use a namespace for the projection family and a key for one concrete latest-state row.

Examples:

- namespace `workflow_design`, key `<workflow_id>`
- namespace `bridge_governance`, key `<interaction_id>`
- namespace `wisdom_summary`, key `<topic_id>`

Pick names that are domain-stable and do not overload unrelated concepts.

## Compatibility With Workflow Design

Workflow design remains a first compatibility consumer of this substrate.

The public workflow-facing APIs stay the same:

- `get_workflow_design_projection(...)`
- `replace_workflow_design_projection(...)`
- `clear_workflow_design_projection(...)`

Internally, those wrappers now serialize their latest-state payload into the generic named-projection row, while richer workflow reconstruction logic still lives at the service layer.
