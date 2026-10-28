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
