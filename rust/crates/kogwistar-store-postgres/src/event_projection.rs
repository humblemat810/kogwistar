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

pub(super) fn projection_payload_json(
    payload: &serde_json::Map<String, serde_json::Value>,
) -> PostgresStoreResult<String> {
    let compact = serde_json::to_string(&BTreeMap::<_, _>::from_iter(payload.iter()))
        .map_err(|error| PostgresStoreError::TransactionAborted(error.to_string()))?;
    Ok(python_ascii_json(&compact))
}

pub(super) fn python_ascii_json(compact_json: &str) -> String {
    compact_json
        .chars()
        .flat_map(|character| match character as u32 {
            0..=0x7f => character.to_string().chars().collect::<Vec<_>>(),
            value @ 0x80..=0xffff => format!("\\u{value:04x}").chars().collect(),
            value => {
                let offset = value - 0x1_0000;
                let high = 0xd800 + (offset >> 10);
                let low = 0xdc00 + (offset & 0x3ff);
                format!("\\u{high:04x}\\u{low:04x}").chars().collect()
            }
        })
        .collect()
}

pub(super) fn projection_from_row(row: &Row) -> PostgresStoreResult<NamedProjection> {
    let payload_json: String = row.try_get(2).map_err(backend)?;
    let payload = serde_json::from_str::<serde_json::Value>(&payload_json)
        .map_err(PostgresStoreError::InvalidPayload)?
        .as_object()
        .cloned()
        .ok_or_else(|| {
            PostgresStoreError::Backend(
                "named projection payload must deserialize to an object".to_owned(),
            )
        })?;
    Ok(NamedProjection {
        namespace: row.try_get(0).map_err(backend)?,
        key: row.try_get(1).map_err(backend)?,
        payload,
        last_authoritative_seq: row.try_get(3).map_err(backend)?,
        last_materialized_seq: row.try_get(4).map_err(backend)?,
        projection_schema_version: i64::from(row.try_get::<_, i32>(5).map_err(backend)?),
        materialization_status: row.try_get(6).map_err(backend)?,
        updated_at_ms: row.try_get(7).map_err(backend)?,
    })
}

pub(super) async fn named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
) -> PostgresStoreResult<Option<NamedProjection>>
where
    C: GenericClient + Sync,
{
    client.query_opt(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {} WHERE namespace = $1 AND key = $2", tables.named_projections),
        &[&namespace, &key],
    ).await.map_err(backend)?.map(|row| projection_from_row(&row)).transpose()
}

pub(super) async fn named_projection_for_update<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
) -> PostgresStoreResult<Option<NamedProjection>>
where
    C: GenericClient + Sync,
{
    client.query_opt(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {} WHERE namespace = $1 AND key = $2 FOR UPDATE", tables.named_projections),
        &[&namespace, &key],
    ).await.map_err(backend)?.map(|row| projection_from_row(&row)).transpose()
}

pub(super) async fn named_projections<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<Vec<NamedProjection>>
where
    C: GenericClient + Sync,
{
    client.query(
        &format!("SELECT namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms FROM {} WHERE namespace = $1 ORDER BY key ASC", tables.named_projections),
        &[&namespace],
    ).await.map_err(backend)?.iter().map(projection_from_row).collect()
}

pub(super) async fn replace_named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
    projection: NamedProjectionWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let payload_json = projection_payload_json(&projection.payload)?;
    let updated_at_ms = unix_epoch_millis();
    let projection_schema_version =
        i32::try_from(projection.projection_schema_version).map_err(|_| {
            PostgresStoreError::Backend(
                "projection schema version exceeds PostgreSQL INTEGER".to_owned(),
            )
        })?;
    client.execute(
        &format!("INSERT INTO {}(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(namespace,key) DO UPDATE SET payload_json=EXCLUDED.payload_json,last_authoritative_seq=EXCLUDED.last_authoritative_seq,last_materialized_seq=EXCLUDED.last_materialized_seq,projection_schema_version=EXCLUDED.projection_schema_version,materialization_status=EXCLUDED.materialization_status,updated_at_ms=EXCLUDED.updated_at_ms", tables.named_projections),
        &[&namespace, &key, &payload_json, &projection.last_authoritative_seq, &projection.last_materialized_seq, &projection_schema_version, &projection.materialization_status, &updated_at_ms],
    ).await.map_err(backend)?;
    Ok(())
}

pub(super) async fn compare_and_swap_named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
    expected_last_authoritative_seq: Option<i64>,
    expected_last_materialized_seq: Option<i64>,
    projection: NamedProjectionWrite,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    client
        .query_one(
            "SELECT pg_advisory_xact_lock(hashtextextended($1::text, hashtextextended($2::text, 0)))",
            &[&namespace, &key],
        )
        .await
        .map_err(backend)?;
    let payload_json = projection_payload_json(&projection.payload)?;
    let updated_at_ms = unix_epoch_millis();
    let projection_schema_version =
        i32::try_from(projection.projection_schema_version).map_err(|_| {
            PostgresStoreError::Backend(
                "projection schema version exceeds PostgreSQL INTEGER".to_owned(),
            )
        })?;
    let changes = match (expected_last_authoritative_seq, expected_last_materialized_seq) {
        (None, None) => client.execute(
            &format!("INSERT INTO {}(namespace, key, payload_json, last_authoritative_seq, last_materialized_seq, projection_schema_version, materialization_status, updated_at_ms) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT(namespace,key) DO NOTHING", tables.named_projections),
            &[&namespace, &key, &payload_json, &projection.last_authoritative_seq, &projection.last_materialized_seq, &projection_schema_version, &projection.materialization_status, &updated_at_ms],
        ).await.map_err(backend)?,
        (Some(expected_authoritative), Some(expected_materialized)) => client.execute(
            &format!("UPDATE {} SET payload_json=$1,last_authoritative_seq=$2,last_materialized_seq=$3,projection_schema_version=$4,materialization_status=$5,updated_at_ms=$6 WHERE namespace=$7 AND key=$8 AND last_authoritative_seq=$9 AND last_materialized_seq=$10", tables.named_projections),
            &[&payload_json, &projection.last_authoritative_seq, &projection.last_materialized_seq, &projection_schema_version, &projection.materialization_status, &updated_at_ms, &namespace, &key, &expected_authoritative, &expected_materialized],
        ).await.map_err(backend)?,
        _ => 0,
    };
    Ok(changes == 1)
}

pub(super) async fn compare_and_swap_named_projections<C>(
    client: &C,
    tables: &Tables,
    mut updates: Vec<NamedProjectionCas>,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    updates
        .sort_by(|left, right| (&left.namespace, &left.key).cmp(&(&right.namespace, &right.key)));
    let mut keys = BTreeSet::new();
    for update in &updates {
        require_namespace(&update.namespace)?;
        if !keys.insert((update.namespace.clone(), update.key.clone())) {
            return Err(PostgresStoreError::Backend(
                "duplicate named-projection CAS key".to_owned(),
            ));
        }
        if update.expected_last_authoritative_seq.is_some()
            != update.expected_last_materialized_seq.is_some()
        {
            return Ok(false);
        }
    }

    // Lock keys in deterministic order. Advisory locks also serialize absent-row
    // creates, for which a normal FOR UPDATE has no row to lock.
    for update in &updates {
        client
            .query_one(
                "SELECT pg_advisory_xact_lock(hashtextextended($1::text, hashtextextended($2::text, 0)))",
                &[&update.namespace, &update.key],
            )
            .await
            .map_err(backend)?;
        let current =
            named_projection_for_update(client, tables, &update.namespace, &update.key).await?;
        let matches = match (
            &current,
            update.expected_last_authoritative_seq,
            update.expected_last_materialized_seq,
        ) {
            (None, None, None) => true,
            (Some(row), Some(expected_authoritative), Some(expected_materialized)) => {
                row.last_authoritative_seq == expected_authoritative
                    && row.last_materialized_seq == expected_materialized
            }
            _ => false,
        };
        if !matches {
            return Ok(false);
        }
    }

    for update in updates {
        replace_named_projection(
            client,
            tables,
            &update.namespace,
            &update.key,
            update.projection,
        )
        .await?;
    }
    Ok(true)
}

pub(super) async fn clear_named_projection<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    key: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE namespace = $1 AND key = $2",
                tables.named_projections
            ),
            &[&namespace, &key],
        )
        .await
        .map_err(|error| PostgresStoreError::Backend(format!("{error:?}")))?;
    Ok(())
}

pub(super) async fn clear_projection_namespace<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE namespace = $1",
                tables.named_projections
            ),
            &[&namespace],
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

pub(super) fn graph_table(tables: &Tables, table: &str) -> PostgresStoreResult<String> {
    quote_identifier(table)?;
    qualified(&tables.quoted_schema, table)
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
         CREATE INDEX IF NOT EXISTS \"ix_gke_edge_refs_doc\" ON {} ((metadata->>'doc_id'));",
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
    );
    client.batch_execute(&sql).await.map_err(backend)
}
