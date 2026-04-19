# Lane Messaging Contract

This repo keeps lane messaging contract small and stable:

- `send_message(...)`
- `claim_pending(...)`
- `ack(...)`
- `requeue(...)`
- `dead_letter(...)`
- `list_projected(...)`

Truth split:

- graph/entity event = authoritative message truth
- metastore projection = serving state
- concrete store = storage primitive only

Hard concepts:

- `prev / next / tail` are cache-like pointers, not truth
- if pointer rows vanish or drift, rebuild from graph truth
- worker reads projection, not raw graph, for hot path

```mermaid
flowchart TB
  T[Authoritative truth\nGraph / entity events] --> P[Projection builder]
  P --> S[Metastore rows\nseq / claim state / prev-next / tail]
  S --> W[Worker hot path\nclaim / ack / requeue / list]
  S -. cache only .-> L[Optional linked-list pointers]
  L -. rebuild .-> P
  W -. never truth .-> T
  P -. rebuild from truth .-> T
```

```mermaid
flowchart TB
  subgraph Core["Core truth"]
    GE[Graph / entity events]
    MT[Message node]
    GE --> MT
  end

  subgraph Meta["Metastore projection"]
    PR[Projected rows]
    ST[Claim / ack / requeue / dead-letter]
    LL[Optional prev / next / tail]
    PR --> ST
    PR --> LL
  end

  MT --> PR
  LL -. rebuild .-> MT
  ST -. derived .-> MT
```

```mermaid
flowchart LR
  A[Graph / entity events] --> B[Lane message truth]
  A --> C[Projection builder]
  C --> D[Metastore projected rows]
  D --> E[Claim / ack / requeue / list]
  D --> F[Optional prev / next / tail pointers]
  F -. rebuild .-> C
  B -. rebuild .-> C
```

Recovery rule:

- projections must rebuild from authoritative truth
- optional linked-list materialization must also rebuild from authoritative truth

```mermaid
sequenceDiagram
  participant Writer
  participant Graph
  participant Projector
  participant Store
  participant Worker

  Writer->>Graph: append message entity event
  Graph->>Projector: authoritative truth available
  Projector->>Store: materialize seq / claim state / pointers
  Worker->>Store: claim pending message
  Worker->>Store: ack / requeue / dead-letter
  Note over Projector,Store: if prev/next/tail disappear, rebuild again
```

Do not turn this into a chimera of extra wrapper layers.
