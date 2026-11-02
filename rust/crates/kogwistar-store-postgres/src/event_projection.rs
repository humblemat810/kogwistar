use super::*;

pub(super) async fn allocate_event_seq<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(&allocation_sql(tables), &[&namespace])
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

pub(super) async fn next_global_seq<C>(client: &C, tables: &Tables) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            &format!(
                "UPDATE {} SET value=value+1 RETURNING value",
                tables.global_seq
            ),
            &[],
        )
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

pub(super) async fn current_global_seq<C>(client: &C, tables: &Tables) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!("SELECT value FROM {} LIMIT 1", tables.global_seq),
            &[],
        )
        .await
        .map_err(backend)?
        .map(|row| row.try_get(0).map_err(backend))
        .unwrap_or(Ok(0))
}

pub(super) async fn next_user_seq<C>(
    client: &C,
    tables: &Tables,
    user_id: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    let row = client
        .query_one(
            &format!(
                "INSERT INTO {}(user_id,value) VALUES($1,1) ON CONFLICT(user_id) DO UPDATE SET value={}.value+1 RETURNING value",
                tables.user_seq, tables.user_seq
            ),
            &[&user_id],
        )
        .await
        .map_err(backend)?;
    row.try_get(0).map_err(backend)
}

pub(super) async fn current_user_seq<C>(
    client: &C,
    tables: &Tables,
    user_id: &str,
) -> PostgresStoreResult<i64>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!("SELECT value FROM {} WHERE user_id=$1", tables.user_seq),
            &[&user_id],
        )
        .await
        .map_err(backend)?
        .map(|row| row.try_get(0).map_err(backend))
        .unwrap_or(Ok(0))
}

pub(super) async fn set_user_seq<C>(
    client: &C,
    tables: &Tables,
    user_id: &str,
    value: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    if value < 0 {
        return Err(PostgresStoreError::Backend(
            "sequence value must be non-negative".to_owned(),
        ));
    }
    client
        .execute(
            &format!(
                "INSERT INTO {}(user_id,value) VALUES($1,$2) ON CONFLICT(user_id) DO UPDATE SET value=EXCLUDED.value",
                tables.user_seq
            ),
            &[&user_id, &value],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(super) async fn get_index_applied_fingerprint<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    coalesce_key: &str,
) -> PostgresStoreResult<Option<String>>
where
    C: GenericClient + Sync,
{
    client
        .query_opt(
            &format!(
                "SELECT applied_fingerprint FROM {} WHERE namespace=$1 AND coalesce_key=$2",
                tables.index_applied_state
            ),
            &[&namespace, &coalesce_key],
        )
        .await
        .map_err(backend)?
        .map(|row| row.try_get(0).map_err(backend))
        .transpose()
}

pub(super) async fn set_index_applied_fingerprint<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    coalesce_key: &str,
    applied_fingerprint: Option<String>,
    last_job_id: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "INSERT INTO {}(namespace,coalesce_key,applied_fingerprint,applied_at,last_job_id) VALUES($1,$2,$3,NOW(),$4) ON CONFLICT(namespace,coalesce_key) DO UPDATE SET applied_fingerprint=EXCLUDED.applied_fingerprint,applied_at=NOW(),last_job_id=EXCLUDED.last_job_id",
                tables.index_applied_state
            ),
            &[&namespace, &coalesce_key, &applied_fingerprint, &last_job_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(super) fn unix_epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock precedes Unix epoch")
        .as_millis() as i64
}

pub(super) fn allocation_sql(tables: &Tables) -> String {
    format!(
        "INSERT INTO {} AS target(namespace, next_seq) VALUES ($1, 2) \
         ON CONFLICT(namespace) DO UPDATE SET next_seq = target.next_seq + 1 \
         RETURNING next_seq - 1",
        tables.namespace_seq
    )
}

pub(super) async fn append_raw_entity_event<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    event: NewRawEntityEvent,
) -> PostgresStoreResult<AppendedRawEvent>
where
    C: GenericClient + Sync,
{
    // A per-event advisory xact lock closes the absent-row unique-key race:
    // same-id retries observe the committed row before allocating a sequence.
    advisory_lock(client, &event.event_id, 0).await?;
    if let Some(existing) = event_by_id(client, tables, &event.event_id).await? {
        if existing.namespace != namespace {
            return Err(PostgresStoreError::EventIdNamespaceCollision {
                event_id: event.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: namespace.to_owned(),
            });
        }
        return Ok(AppendedRawEvent {
            event: existing,
            inserted: false,
        });
    }
    let seq = allocate_event_seq(client, tables, namespace).await?;
    let row = client
        .query_one(
            &format!(
                "INSERT INTO {}(namespace, seq, event_id, entity_kind, entity_id, op, payload_json) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7) \
                 RETURNING namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT",
                tables.entity_events
            ),
            &[
                &namespace,
                &seq,
                &event.event_id,
                &event.entity_kind,
                &event.entity_id,
                &event.op,
                &event.payload_json,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(AppendedRawEvent {
        event: raw_event_from_row(&row)?,
        inserted: true,
    })
}

pub(super) async fn append_restored_raw_entity_event<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    event: RestoredRawEntityEvent,
) -> PostgresStoreResult<AppendedRawEvent>
where
    C: GenericClient + Sync,
{
    require_namespace(namespace)?;
    serde_json::from_str::<Value>(&event.payload_json)?;
    event.seq.checked_add(1).ok_or_else(|| {
        PostgresStoreError::Backend("restored event sequence exceeds PostgreSQL BIGINT".to_owned())
    })?;

    advisory_lock(client, &event.event_id, 0).await?;
    if let Some(row) = client
        .query_opt(
            &format!(
                "SELECT namespace, seq, event_id, entity_kind, entity_id, op, payload_json, \
                 created_at::TEXT, EXTRACT(EPOCH FROM created_at)::BIGINT \
                 FROM {} WHERE event_id=$1",
                tables.entity_events
            ),
            &[&event.event_id],
        )
        .await
        .map_err(backend)?
    {
        let existing = raw_event_from_row(&row)?;
        if existing.namespace != namespace {
            return Err(PostgresStoreError::EventIdNamespaceCollision {
                event_id: event.event_id,
                existing_namespace: existing.namespace,
                requested_namespace: namespace.to_owned(),
            });
        }
        let existing_created_at: i64 = row.try_get(8).map_err(backend)?;
        if existing.seq != event.seq
            || existing.entity_kind != event.entity_kind
            || existing.entity_id != event.entity_id
            || existing.op != event.op
            || existing.payload_json != event.payload_json
            || existing_created_at != event.created_at
        {
            return Err(PostgresStoreError::Backend(format!(
                "restored event_id {:?} conflicts with stored event",
                event.event_id
            )));
        }
        return Ok(AppendedRawEvent {
            event: existing,
            inserted: false,
        });
    }

    let latest = latest_retained_event_seq(client, tables, namespace).await?;
    let expected = latest.checked_add(1).ok_or_else(|| {
        PostgresStoreError::Backend("latest event sequence exceeds PostgreSQL BIGINT".to_owned())
    })?;
    if event.seq != expected {
        return Err(PostgresStoreError::Backend(format!(
            "restored event sequence for {namespace:?} must be {expected}, got {}",
            event.seq
        )));
    }
    let created_at = event.created_at as f64;
    let row = client
        .query_one(
            &format!(
                "INSERT INTO {}(namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at) \
                 VALUES($1, $2, $3, $4, $5, $6, $7, to_timestamp($8)) \
                 RETURNING namespace, seq, event_id, entity_kind, entity_id, op, payload_json, created_at::TEXT",
                tables.entity_events
            ),
            &[
                &namespace,
                &event.seq,
                &event.event_id,
                &event.entity_kind,
                &event.entity_id,
                &event.op,
                &event.payload_json,
                &created_at,
            ],
        )
        .await
        .map_err(backend)?;
    let stored = raw_event_from_row(&row)?;
    let next_seq = event.seq + 1;
    client
        .execute(
            &format!(
                "INSERT INTO {}(namespace, next_seq) VALUES($1, $2) \
                 ON CONFLICT(namespace) DO UPDATE SET next_seq=GREATEST({}.next_seq, EXCLUDED.next_seq)",
                tables.namespace_seq, tables.namespace_seq
            ),
            &[&namespace, &next_seq],
        )
        .await
        .map_err(backend)?;
    Ok(AppendedRawEvent {
        event: stored,
        inserted: true,
    })
}

pub(super) fn graph_table(tables: &Tables, table: &str) -> PostgresStoreResult<String> {
    quote_identifier(table)?;
    qualified(&tables.quoted_schema, table)
}

pub(super) async fn ensure_graph_scope_identity_indexes<C>(
    client: &C,
    tables: &Tables,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    // Python owns graph-table creation. Add only Rust's conflict arbiters when
    // those Python-compatible tables already exist; never alter their id PK.
    let graph_tables = [
        "gke_nodes",
        "gke_edges",
        "gke_documents",
        "gke_domains",
        EDGE_ENDPOINTS_TABLE,
        NODE_DOCS_TABLE,
        NODE_REFS_TABLE,
        EDGE_REFS_TABLE,
    ];
    for (index, name) in graph_tables.into_iter().enumerate() {
        let relation = qualified(&tables.quoted_schema, name)?;
        let exists: bool = client
            .query_one("SELECT to_regclass($1) IS NOT NULL", &[&relation])
            .await
            .map_err(backend)?
            .try_get(0)
            .map_err(backend)?;
        if !exists {
            continue;
        }
        let index_name = quote_identifier(&format!("uq_graph_scope_identity_{index}"))?;
        client
            .batch_execute(&format!(
                "CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {relation} \
                 (id, (COALESCE(metadata->>'namespace', 'default')), \
                 (COALESCE(metadata->>'workspace_id', '')), \
                 (COALESCE(metadata->>'graph_space', '')))"
            ))
            .await
            .map_err(|error| {
                PostgresStoreError::Backend(format!(
                    "graph scope conflict index creation failed for {relation}: {error:?}"
                ))
            })?;
    }
    Ok(())
}

pub(super) async fn create_graph_schema<C>(
    client: &C,
    tables: &Tables,
    embedding_dim: usize,
    names: GraphTableNames,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    if embedding_dim == 0 {
        return Err(PostgresStoreError::Backend(
            "graph embedding dimension must be positive".to_owned(),
        ));
    }
    let dim = i32::try_from(embedding_dim).map_err(|_| {
        PostgresStoreError::Backend("graph embedding dimension exceeds INTEGER".to_owned())
    })?;
    let relations = [names.nodes, names.edges, names.documents, names.domains]
        .into_iter()
        .map(|name| qualified(&tables.quoted_schema, &name))
        .collect::<PostgresStoreResult<Vec<_>>>()?;
    let edge_endpoints = qualified(&tables.quoted_schema, EDGE_ENDPOINTS_TABLE)?;
    let node_docs = qualified(&tables.quoted_schema, NODE_DOCS_TABLE)?;
    let node_refs = qualified(&tables.quoted_schema, NODE_REFS_TABLE)?;
    let edge_refs = qualified(&tables.quoted_schema, EDGE_REFS_TABLE)?;
    // Relation/type/dimension are validated constants. Data remains bind-bound
    // everywhere possible; pgvector requires explicit `TEXT::vector` casts.
    let sql = format!(
        "{} \
         CREATE EXTENSION IF NOT EXISTS vector; \
         CREATE SCHEMA IF NOT EXISTS {}; \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, embedding vector({}) NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_edge_endpoints_endpoint_id\" ON {} ((metadata->>'endpoint_id')); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_edge_endpoints_edge\" ON {} ((metadata->>'edge_id')); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_node_docs_node\" ON {} ((metadata->>'node_id')); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_node_docs_doc\" ON {} ((metadata->>'doc_id')); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_node_refs_node\" ON {} ((metadata->>'node_id')); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_node_refs_doc\" ON {} ((metadata->>'doc_id')); \
         CREATE TABLE IF NOT EXISTS {} (id VARCHAR PRIMARY KEY, document TEXT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_edge_refs_edge\" ON {} ((metadata->>'edge_id')); \
         CREATE INDEX IF NOT EXISTS \"ix_gke_edge_refs_doc\" ON {} ((metadata->>'doc_id')); \
         CREATE TABLE IF NOT EXISTS {} (namespace TEXT NOT NULL,collection TEXT NOT NULL,id TEXT NOT NULL,document TEXT NULL,metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,embedding vector NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),PRIMARY KEY(namespace,collection,id)); \
         CREATE INDEX IF NOT EXISTS \"idx_gke_vector_collection_lookup\" ON {} (namespace,collection); \
         CREATE TABLE IF NOT EXISTS {} (namespace VARCHAR NOT NULL DEFAULT 'default', entity_kind VARCHAR NOT NULL, entity_id VARCHAR NOT NULL, document TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb, source_fingerprint VARCHAR NOT NULL, revision BIGINT NOT NULL DEFAULT 0, materialization_status VARCHAR NOT NULL DEFAULT 'pending', updated_at_ms BIGINT NOT NULL, PRIMARY KEY(namespace, entity_kind, entity_id)); \
         CREATE INDEX IF NOT EXISTS \"idx_gke_stage1_projections_namespace_status\" ON {} (namespace, materialization_status, updated_at_ms);",
        schema_sql(tables),
        tables.quoted_schema,
        relations[0],
        dim,
        relations[1],
        dim,
        relations[2],
        dim,
        relations[3],
        dim,
        edge_endpoints,
        edge_endpoints,
        edge_endpoints,
        node_docs,
        node_docs,
        node_docs,
        node_refs,
        node_refs,
        node_refs,
        edge_refs,
        edge_refs,
        edge_refs,
        tables.vector_collection_records,
        tables.vector_collection_records,
        tables.stage1_projections,
        tables.stage1_projections,
    );
    client.batch_execute(&sql).await.map_err(|error| {
        PostgresStoreError::Backend(format!("graph schema base DDL failed: {error:?}"))
    })?;

    // Keep graph projection identity scoped across namespace, workspace, and
    // graph_space. This is not a substitute for Python's GraphType schema routing.
    let scoped_relations = [
        relations[0].as_str(),
        relations[1].as_str(),
        relations[2].as_str(),
        relations[3].as_str(),
        edge_endpoints.as_str(),
        node_docs.as_str(),
        node_refs.as_str(),
        edge_refs.as_str(),
    ];
    for (index, relation) in scoped_relations.into_iter().enumerate() {
        // Preserve Python's global id primary key. This additional arbiter
        // supports the scoped Rust upsert target without changing its identity contract.
        let index = quote_identifier(&format!("uq_graph_scope_identity_{index}"))?;
        client
            .batch_execute(&format!(
                "CREATE UNIQUE INDEX IF NOT EXISTS {index} ON {relation} \
                 (id, (COALESCE(metadata->>'namespace', 'default')), \
                 (COALESCE(metadata->>'workspace_id', '')), \
                 (COALESCE(metadata->>'graph_space', '')))"
            ))
            .await
            .map_err(|error| {
                PostgresStoreError::Backend(format!(
                    "scoped graph identity index creation failed for {relation}: {error:?}"
                ))
            })?;
    }
    Ok(())
}
