use super::*;

pub(super) async fn mark_index_job_done<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    claim_token: Option<&str>,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    Ok(client.execute(&format!("UPDATE {} SET status='DONE',lease_until=NULL,claim_token=NULL,updated_at=NOW() WHERE job_id=$1 AND status='DOING' AND ($2::TEXT IS NULL OR (claim_token=$2 AND lease_until IS NOT NULL AND lease_until>=NOW()))",tables.index_jobs), &[&job_id,&claim_token]).await.map_err(backend)?!=0)
}

pub(super) async fn index_job_result<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
) -> PostgresStoreResult<Option<AcceptedIndexJobResult>>
where
    C: GenericClient + Sync,
{
    let row = client.query_opt(&format!("SELECT accepted_result_json, accepted_result_sha256, accepted_at::TEXT FROM {} WHERE job_id=$1", tables.index_jobs), &[&job_id]).await.map_err(backend)?;
    Ok(row.map(|row| AcceptedIndexJobResult {
        status: "existing".to_owned(),
        result_json: row.get(0),
        result_sha256: row.get(1),
        accepted_at: row
            .get::<_, Option<String>>(2)
            .map(serde_json::Value::String),
    }))
}

pub(super) async fn accept_index_job_result<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    claim_token: &str,
    result_json: &str,
    result_sha256: &str,
) -> PostgresStoreResult<AcceptedIndexJobResult>
where
    C: GenericClient + Sync,
{
    if let Some(existing) = index_job_result(client, tables, job_id).await?
        && existing.result_json.is_some()
    {
        return Ok(existing);
    }
    let row = client.query_opt(&format!("UPDATE {} SET accepted_result_json=$1,accepted_result_sha256=$2,accepted_at=NOW() WHERE job_id=$3 AND status='DOING' AND claim_token=$4 AND (lease_until IS NULL OR lease_until>=NOW()) AND accepted_result_json IS NULL RETURNING accepted_at::TEXT", tables.index_jobs), &[&result_json, &result_sha256, &job_id, &claim_token]).await.map_err(backend)?;
    if let Some(row) = row {
        return Ok(AcceptedIndexJobResult {
            status: "accepted".to_owned(),
            result_json: Some(result_json.to_owned()),
            result_sha256: Some(result_sha256.to_owned()),
            accepted_at: row
                .get::<_, Option<String>>(0)
                .map(serde_json::Value::String),
        });
    }
    Ok(AcceptedIndexJobResult {
        status: "rejected".to_owned(),
        result_json: None,
        result_sha256: None,
        accepted_at: None,
    })
}

pub(super) async fn mark_index_job_failed<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    error: &str,
    final_: bool,
    claim_token: Option<&str>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    client.execute(&format!("UPDATE {} SET status=CASE WHEN $1 THEN 'FAILED' ELSE status END,lease_until=NULL,claim_token=NULL,last_error=$2,updated_at=NOW() WHERE job_id=$3 AND status='DOING' AND ($4::TEXT IS NULL OR (claim_token=$4 AND lease_until IS NOT NULL AND lease_until>=NOW()))",tables.index_jobs), &[&final_,&truncate_error(error),&job_id,&claim_token]).await.map_err(backend)?;
    Ok(())
}

pub(super) async fn bump_retry_and_requeue<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    error: &str,
    delay: i64,
    claim_token: Option<&str>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    client.execute(&format!("UPDATE {} SET retry_count=retry_count+1,last_error=$1,status=CASE WHEN retry_count+1>=max_retries THEN 'FAILED' ELSE 'PENDING' END,lease_until=NULL,claim_token=NULL,next_run_at=CASE WHEN retry_count+1>=max_retries THEN NULL ELSE NOW()+($2::TEXT||' seconds')::interval END,updated_at=NOW() WHERE job_id=$3 AND status='DOING' AND ($4::TEXT IS NULL OR (claim_token=$4 AND lease_until IS NOT NULL AND lease_until>=NOW()))",tables.index_jobs), &[&truncate_error(error),&delay,&job_id,&claim_token]).await.map_err(backend)?;
    Ok(())
}

pub(super) async fn renew_index_job_lease<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    claim_token: &str,
    lease_seconds: i64,
) -> PostgresStoreResult<bool>
where
    C: GenericClient + Sync,
{
    let lease_seconds = lease_seconds.to_string();
    Ok(client.execute(&format!("UPDATE {} SET lease_until=NOW()+($1::TEXT||' seconds')::interval,updated_at=NOW() WHERE job_id=$2 AND status='DOING' AND claim_token=$3 AND (lease_until IS NULL OR lease_until>=NOW())",tables.index_jobs), &[&lease_seconds,&job_id,&claim_token]).await.map_err(backend)?!=0)
}

pub(super) async fn requeue_index_job_at_tail<C>(
    client: &C,
    tables: &Tables,
    job_id: &str,
    payload_json: String,
    delay: i64,
    claim_token: Option<&str>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    client.execute(&format!("UPDATE {} SET status='PENDING',lease_until=NULL,claim_token=NULL,next_run_at=NOW()+($1::TEXT||' seconds')::interval,payload_json=$2,created_at=(SELECT COALESCE(MAX(created_at),NOW())+INTERVAL '1 microsecond' FROM {}),updated_at=NOW() WHERE job_id=$3 AND status='DOING' AND ($4::TEXT IS NULL OR (claim_token=$4 AND lease_until IS NOT NULL AND lease_until>=NOW()))",tables.index_jobs,tables.index_jobs), &[&delay,&payload_json,&job_id,&claim_token]).await.map_err(backend)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn list_index_jobs<C>(
    client: &C,
    tables: &Tables,
    namespace: Option<&str>,
    status: Option<&str>,
    entity_kind: Option<&str>,
    entity_id: Option<&str>,
    index_kind: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<IndexJob>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let rows=client.query(&format!("SELECT job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until::TEXT,next_run_at::TEXT,max_retries,retry_count,last_error,payload_json,created_at::TEXT,updated_at::TEXT,claim_token,claim_attempts,accepted_result_json,accepted_result_sha256,accepted_at::TEXT FROM {} WHERE ($1::TEXT IS NULL OR namespace=$1) AND ($2::TEXT IS NULL OR status=$2) AND ($3::TEXT IS NULL OR entity_kind=$3) AND ($4::TEXT IS NULL OR entity_id=$4) AND ($5::TEXT IS NULL OR index_kind=$5) ORDER BY created_at ASC,job_id ASC LIMIT $6",tables.index_jobs), &[&namespace,&status,&entity_kind,&entity_id,&index_kind,&limit]).await.map_err(backend)?;
    rows.iter().map(index_job_from_row).collect()
}

pub(super) fn index_job_from_row(row: &Row) -> PostgresStoreResult<IndexJob> {
    Ok(IndexJob {
        job_id: row.try_get(0).map_err(backend)?,
        namespace: row.try_get(1).map_err(backend)?,
        entity_kind: row.try_get(2).map_err(backend)?,
        entity_id: row.try_get(3).map_err(backend)?,
        index_kind: row.try_get(4).map_err(backend)?,
        coalesce_key: row.try_get(5).map_err(backend)?,
        op: row.try_get(6).map_err(backend)?,
        status: row.try_get(7).map_err(backend)?,
        lease_until: row
            .try_get::<_, Option<String>>(8)
            .map_err(backend)?
            .map(serde_json::Value::String),
        next_run_at: row
            .try_get::<_, Option<String>>(9)
            .map_err(backend)?
            .map(serde_json::Value::String),
        max_retries: i64::from(row.try_get::<_, i32>(10).map_err(backend)?),
        retry_count: i64::from(row.try_get::<_, i32>(11).map_err(backend)?),
        last_error: row.try_get(12).map_err(backend)?,
        payload_json: row.try_get(13).map_err(backend)?,
        created_at: serde_json::Value::String(row.try_get(14).map_err(backend)?),
        updated_at: serde_json::Value::String(row.try_get(15).map_err(backend)?),
        claim_token: row.try_get(16).map_err(backend)?,
        claim_attempts: i64::from(row.try_get::<_, i32>(17).map_err(backend)?),
        accepted_result_json: row.try_get(18).map_err(backend)?,
        accepted_result_sha256: row.try_get(19).map_err(backend)?,
        accepted_at: row
            .try_get::<_, Option<String>>(20)
            .map_err(backend)?
            .map(serde_json::Value::String),
    })
}

pub(super) fn truncate_error(error: &str) -> String {
    error.chars().take(2000).collect()
}
