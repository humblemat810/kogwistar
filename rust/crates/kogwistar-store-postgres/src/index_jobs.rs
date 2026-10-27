use super::*;

pub(super) async fn enqueue_index_job<C>(
    client: &C,
    tables: &Tables,
    job: NewIndexJob,
) -> PostgresStoreResult<String>
where
    C: GenericClient + Sync,
{
    require_namespace(&job.namespace)?;
    let key = job.coalesce_key();
    let max_retries = i32::try_from(job.max_retries).map_err(|_| {
        PostgresStoreError::Backend("max_retries exceeds PostgreSQL INTEGER".to_owned())
    })?;
    let row=client.query_one(&format!("INSERT INTO {}(job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token,claim_attempts) VALUES ($1,$2,$3,$4,$5,$6,$7,'PENDING',NULL,NULL,$8,0,NULL,$9,NOW(),NOW(),NULL,0) ON CONFLICT(namespace,coalesce_key) WHERE status='PENDING' DO UPDATE SET op=CASE WHEN EXCLUDED.op='DELETE' OR {}.op='DELETE' THEN 'DELETE' ELSE EXCLUDED.op END,payload_json=EXCLUDED.payload_json,updated_at=NOW() RETURNING job_id",tables.index_jobs,tables.index_jobs), &[&job.job_id,&job.namespace,&job.entity_kind,&job.entity_id,&job.index_kind,&key,&job.op,&max_retries,&job.payload_json]).await.map_err(backend)?;
    row.try_get(0).map_err(backend)
}
pub(super) async fn claim_index_jobs<C>(
    client: &C,
    tables: &Tables,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    index_kinds: Option<&[String]>,
) -> PostgresStoreResult<Vec<IndexJob>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(Vec::new());
    };
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let lease_seconds = lease_seconds.to_string();
    let index_kinds = index_kinds.map(<[String]>::to_vec);
    client.execute(
        &format!("UPDATE {} SET status='FAILED',last_error='lease ownership exceeded',lease_until=NULL,claim_token=NULL,updated_at=NOW() WHERE status='DOING' AND lease_until<NOW() AND claim_attempts>=$1 AND ($2::TEXT[] IS NULL OR index_kind=ANY($2))", tables.index_jobs),
        &[&3_i32, &index_kinds],
    ).await.map_err(backend)?;
    let rows=client.query(&format!("WITH candidates AS (SELECT job_id FROM {} WHERE ((status='PENDING' AND (next_run_at IS NULL OR next_run_at<=NOW())) OR (status='DOING' AND lease_until IS NOT NULL AND lease_until<NOW())) AND ($1::TEXT IS NULL OR namespace=$1) AND ($4::TEXT[] IS NULL OR index_kind=ANY($4)) ORDER BY created_at ASC,job_id ASC LIMIT $2 FOR UPDATE SKIP LOCKED) UPDATE {} j SET status='DOING',lease_until=NOW()+($3::TEXT||' seconds')::interval,claim_token=md5(random()::TEXT||clock_timestamp()::TEXT||j.job_id),claim_attempts=j.claim_attempts+CASE WHEN j.status='DOING' THEN 1 ELSE 0 END,updated_at=NOW() FROM candidates c WHERE j.job_id=c.job_id RETURNING j.job_id,j.namespace,j.entity_kind,j.entity_id,j.index_kind,j.coalesce_key,j.op,j.status,j.lease_until::TEXT,j.next_run_at::TEXT,j.max_retries,j.retry_count,j.last_error,j.payload_json,j.created_at::TEXT,j.updated_at::TEXT,j.claim_token,j.claim_attempts,j.accepted_result_json,j.accepted_result_sha256,j.accepted_at::TEXT",tables.index_jobs,tables.index_jobs), &[&namespace,&limit,&lease_seconds,&index_kinds]).await.map_err(backend)?;
    rows.iter().map(index_job_from_row).collect()
}

pub(super) async fn reconcile_join_index_jobs<C>(
    client: &C,
    base_tables: &Tables,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    names: GraphIndexTableNames,
) -> PostgresStoreResult<usize>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(0);
    }
    if lease_seconds <= 0 {
        return Err(PostgresStoreError::Backend(
            "index job lease must be positive".to_owned(),
        ));
    }
    for name in [
        &names.nodes,
        &names.edges,
        &names.node_docs,
        &names.node_refs,
        &names.edge_refs,
        &names.edge_endpoints,
    ] {
        quote_identifier(name)?;
    }
    let mut tables = base_tables.clone();
    tables.node_docs = qualified(&tables.quoted_schema, &names.node_docs)?;
    tables.node_refs = qualified(&tables.quoted_schema, &names.node_refs)?;
    tables.edge_refs = qualified(&tables.quoted_schema, &names.edge_refs)?;
    tables.edge_endpoints = qualified(&tables.quoted_schema, &names.edge_endpoints)?;

    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let lease_seconds = lease_seconds.to_string();
    client.execute(
        &format!(
            "UPDATE {} SET status='FAILED',last_error='lease ownership exceeded',lease_until=NULL,claim_token=NULL,updated_at=NOW() \
             WHERE status='DOING' AND lease_until<NOW() AND claim_attempts>=3 AND accepted_result_json IS NULL \
             AND ((entity_kind='node' AND index_kind IN ('node_docs','node_refs')) OR \
                  (entity_kind='edge' AND index_kind IN ('edge_refs','edge_endpoints'))) \
             AND ($1::TEXT IS NULL OR namespace=$1)",
            tables.index_jobs
        ),
        &[&namespace],
    ).await.map_err(backend)?;
    let jobs = client.query(
        &format!(
            "WITH candidates AS (SELECT job_id FROM {} \
             WHERE ((status='PENDING' AND (next_run_at IS NULL OR next_run_at<=NOW())) OR \
                    (status='DOING' AND lease_until IS NOT NULL AND lease_until<NOW())) \
               AND accepted_result_json IS NULL \
               AND ((entity_kind='node' AND index_kind IN ('node_docs','node_refs')) OR \
                    (entity_kind='edge' AND index_kind IN ('edge_refs','edge_endpoints'))) \
               AND ($1::TEXT IS NULL OR namespace=$1) \
             ORDER BY created_at ASC,job_id ASC LIMIT $2 FOR UPDATE SKIP LOCKED) \
             UPDATE {} j SET status='DOING',lease_until=NOW()+($3::TEXT||' seconds')::interval, \
               claim_token=md5(random()::TEXT||clock_timestamp()::TEXT||j.job_id), \
               claim_attempts=j.claim_attempts+CASE WHEN j.status='DOING' THEN 1 ELSE 0 END,updated_at=NOW() \
             FROM candidates c WHERE j.job_id=c.job_id \
             RETURNING j.job_id,j.namespace,j.entity_kind,j.entity_id,j.index_kind,j.coalesce_key,j.op,j.status, \
               j.lease_until::TEXT,j.next_run_at::TEXT,j.max_retries,j.retry_count,j.last_error,j.payload_json, \
               j.created_at::TEXT,j.updated_at::TEXT,j.claim_token,j.claim_attempts,j.accepted_result_json, \
               j.accepted_result_sha256,j.accepted_at::TEXT",
            tables.index_jobs, tables.index_jobs
        ),
        &[&namespace, &limit, &lease_seconds],
    ).await.map_err(backend)?;

    let mut applied = 0;
    for row in jobs {
        let job = index_job_from_row(&row)?;
        let Some(claim_token) = job.claim_token.as_deref() else {
            continue;
        };
        client
            .batch_execute("SAVEPOINT reconcile_join_index_job")
            .await
            .map_err(backend)?;
        let result = async {
            let (table, entity_kind, index_table) =
                match (job.entity_kind.as_str(), job.index_kind.as_str()) {
                    ("node", "node_docs") => (&names.nodes, "node", "node_docs"),
                    ("node", "node_refs") => (&names.nodes, "node", "node_refs"),
                    ("edge", "edge_refs") => (&names.edges, "edge", "edge_refs"),
                    ("edge", "edge_endpoints") => (&names.edges, "edge", "edge_endpoints"),
                    _ => {
                        return Err(PostgresStoreError::Backend(
                            "unsupported join index job was claimed".to_owned(),
                        ));
                    }
                };
            let scope = kogwistar_store::GraphScope {
                namespace: job.namespace.clone(),
                workspace_id: None,
                graph_space: None,
            };
            let record = if job.op == "DELETE" {
                GraphRecord {
                    id: job.entity_id.clone(),
                    document: None,
                    metadata: Map::new(),
                    embedding: None,
                }
            } else {
                graph_projection_records_with_lock(
                    client,
                    &tables,
                    GraphProjectionRead {
                        scope: scope.clone(),
                        table: table.clone(),
                        ids: Some(vec![job.entity_id.clone()]),
                        metadata: Default::default(),
                        limit: 1,
                    },
                    None,
                    false,
                    true,
                )
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    PostgresStoreError::Backend(format!(
                        "{} {} not found in its namespace",
                        job.entity_kind, job.entity_id
                    ))
                })?
            };
            let tombstoned = job.op == "DELETE"
                || record
                    .metadata
                    .get("lifecycle_status")
                    .and_then(Value::as_str)
                    == Some("tombstoned");
            let mutation = GraphMutation {
                scope,
                table: table.clone(),
                entity_kind: entity_kind.to_owned(),
                event_id: format!("index-reconcile:{}", job.job_id),
                op: if job.op == "DELETE" || tombstoned {
                    "TOMBSTONE"
                } else {
                    "REPLACE"
                }
                .to_owned(),
                payload: Value::Null,
                record,
                embedding_dim: 1,
            };
            match index_table {
                "node_docs" => sync_graph_node_docs(client, &tables, &mutation).await?,
                "node_refs" | "edge_refs" => sync_graph_refs(client, &tables, &mutation).await?,
                "edge_endpoints" => sync_graph_edge_endpoints(client, &tables, &mutation).await?,
                _ => unreachable!(),
            }
            if !mark_index_job_done(client, &tables, &job.job_id, Some(claim_token)).await? {
                return Err(PostgresStoreError::Backend(
                    "lost index job claim before acknowledgement".to_owned(),
                ));
            }
            Ok::<(), PostgresStoreError>(())
        }
        .await;
        match result {
            Ok(()) => {
                client
                    .batch_execute("RELEASE SAVEPOINT reconcile_join_index_job")
                    .await
                    .map_err(backend)?;
                applied += 1;
            }
            Err(error) => {
                client
                    .batch_execute("ROLLBACK TO SAVEPOINT reconcile_join_index_job")
                    .await
                    .map_err(backend)?;
                let delay = 2_i64
                    .saturating_pow(job.retry_count.clamp(0, 8) as u32)
                    .min(300);
                bump_retry_and_requeue(
                    client,
                    &tables,
                    &job.job_id,
                    &error.to_string(),
                    delay,
                    Some(claim_token),
                )
                .await?;
                client
                    .batch_execute("RELEASE SAVEPOINT reconcile_join_index_job")
                    .await
                    .map_err(backend)?;
            }
        }
    }
    Ok(applied)
}

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

pub(super) const LANE_SELECT: &str = "message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until::TEXT,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id";
pub(super) const LANE_RETURNING: &str = "x.message_id,x.namespace,x.purpose,x.inbox_id,x.conversation_id,x.recipient_id,x.sender_id,x.msg_type,x.status,x.seq,x.conversation_seq,x.claimed_by,x.lease_until::TEXT,x.retry_count,x.created_at,x.available_at,x.run_id,x.step_id,x.correlation_id,x.payload_json,x.error_json,x.prev_message_id,x.next_message_id,x.inbox_tail_message_id,x.conversation_tail_message_id";
