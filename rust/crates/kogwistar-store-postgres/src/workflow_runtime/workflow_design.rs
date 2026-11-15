use super::super::*;

pub(crate) async fn put_workflow_design_snapshot<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    snapshot: WorkflowDesignSnapshotWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(snapshot.schema_version)?;
    let created_at_ms = unix_epoch_millis();
    client
        .execute(
            &format!(
                "INSERT INTO {}(workflow_id, version, seq, payload_json, schema_version, created_at_ms) \
                 VALUES ($1,$2,$3,$4,$5,$6) \
                 ON CONFLICT(workflow_id,version) DO UPDATE SET \
                 seq=EXCLUDED.seq,payload_json=EXCLUDED.payload_json, \
                 schema_version=EXCLUDED.schema_version,created_at_ms=EXCLUDED.created_at_ms",
                tables.workflow_design_snapshots
            ),
            &[
                &workflow_id,
                &snapshot.version,
                &snapshot.seq,
                &snapshot.payload_json,
                &schema_version,
                &created_at_ms,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(crate) async fn workflow_design_snapshot<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    max_version: i64,
    schema_version: i64,
) -> PostgresStoreResult<Option<WorkflowDesignSnapshot>>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(schema_version)?;
    client
        .query_opt(
            &format!(
                "SELECT workflow_id,version,seq,payload_json,schema_version,created_at_ms \
                 FROM {} WHERE workflow_id=$1 AND version <= $2 AND schema_version=$3 \
                 ORDER BY version DESC LIMIT 1",
                tables.workflow_design_snapshots
            ),
            &[&workflow_id, &max_version, &schema_version],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(workflow_design_snapshot_from_row)
        .transpose()
}

pub(crate) async fn clear_workflow_design_snapshots<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE workflow_id = $1",
                tables.workflow_design_snapshots
            ),
            &[&workflow_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(crate) async fn put_workflow_design_delta<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    delta: WorkflowDesignDeltaWrite,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(delta.schema_version)?;
    let created_at_ms = unix_epoch_millis();
    client
        .execute(
            &format!(
                "INSERT INTO {}(workflow_id,version,prev_version,target_seq,forward_json,inverse_json,schema_version,created_at_ms) \
                 VALUES ($1,$2,$3,$4,$5,$6,$7,$8) \
                 ON CONFLICT(workflow_id,version) DO UPDATE SET \
                 prev_version=EXCLUDED.prev_version,target_seq=EXCLUDED.target_seq, \
                 forward_json=EXCLUDED.forward_json,inverse_json=EXCLUDED.inverse_json, \
                 schema_version=EXCLUDED.schema_version,created_at_ms=EXCLUDED.created_at_ms",
                tables.workflow_design_version_deltas
            ),
            &[
                &workflow_id,
                &delta.version,
                &delta.prev_version,
                &delta.target_seq,
                &delta.forward_json,
                &delta.inverse_json,
                &schema_version,
                &created_at_ms,
            ],
        )
        .await
        .map_err(backend)?;
    Ok(())
}

pub(crate) async fn workflow_design_delta<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
    version: i64,
    schema_version: i64,
) -> PostgresStoreResult<Option<WorkflowDesignDelta>>
where
    C: GenericClient + Sync,
{
    let schema_version = pg_schema_version(schema_version)?;
    client
        .query_opt(
            &format!(
                "SELECT workflow_id,version,prev_version,target_seq,forward_json,inverse_json,schema_version,created_at_ms \
                 FROM {} WHERE workflow_id=$1 AND version=$2 AND schema_version=$3",
                tables.workflow_design_version_deltas
            ),
            &[&workflow_id, &version, &schema_version],
        )
        .await
        .map_err(backend)?
        .as_ref()
        .map(workflow_design_delta_from_row)
        .transpose()
}

pub(crate) async fn clear_workflow_design_deltas<C>(
    client: &C,
    tables: &Tables,
    workflow_id: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client
        .execute(
            &format!(
                "DELETE FROM {} WHERE workflow_id = $1",
                tables.workflow_design_version_deltas
            ),
            &[&workflow_id],
        )
        .await
        .map_err(backend)?;
    Ok(())
}
