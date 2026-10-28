use super::*;

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
