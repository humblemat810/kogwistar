use super::*;

pub(super) fn enqueue_index_job(conn: &Connection, job: NewIndexJob) -> SqliteStoreResult<String> {
    require_queue_namespace(&job.namespace)?;
    let now = unix_epoch_seconds();
    let key = job.coalesce_key();
    if let Some((job_id, existing_op)) = conn.query_row(
        "SELECT job_id, op FROM index_jobs WHERE namespace=?1 AND coalesce_key=?2 AND status='PENDING' ORDER BY created_at ASC LIMIT 1",
        params![job.namespace, key], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
    ).optional()? {
        let op = if job.op == "DELETE" || existing_op == "DELETE" { "DELETE" } else { &job.op };
        conn.execute("UPDATE index_jobs SET op=?1,payload_json=?2,updated_at=?3 WHERE job_id=?4", params![op, job.payload_json, now, job_id])?;
        return Ok(job_id);
    }
    conn.execute(
        "INSERT OR IGNORE INTO index_jobs(job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token) VALUES (?1,?2,?3,?4,?5,?6,?7,'PENDING',NULL,NULL,?8,0,NULL,?9,?10,?10,NULL)",
        params![job.job_id, job.namespace, job.entity_kind, job.entity_id, job.index_kind, key, job.op, job.max_retries, job.payload_json, now],
    )?;
    Ok(job.job_id)
}

pub(super) fn claim_index_jobs(
    conn: &Connection,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
) -> SqliteStoreResult<Vec<IndexJob>> {
    claim_index_jobs_inner(conn, limit, lease_seconds, namespace, None)
}

pub(super) fn claim_index_jobs_filtered(
    conn: &Connection,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    index_kinds: &[String],
) -> SqliteStoreResult<Vec<IndexJob>> {
    if index_kinds.is_empty() || index_kinds.iter().any(|kind| kind.trim().is_empty()) {
        return Err(SqliteStoreError::Store(StoreError::Backend {
            backend: "index-job-queue".to_owned(),
            message: "filtered index-job claim requires non-empty index kinds".to_owned(),
        }));
    }
    claim_index_jobs_inner(conn, limit, lease_seconds, namespace, Some(index_kinds))
}

fn claim_index_jobs_inner(
    conn: &Connection,
    limit: usize,
    lease_seconds: i64,
    namespace: Option<&str>,
    index_kinds: Option<&[String]>,
) -> SqliteStoreResult<Vec<IndexJob>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let now = unix_epoch_seconds();
    let lease_until = now + lease_seconds;
    let token = format!(
        "sqlite-{}-{}-{}",
        now,
        std::process::id(),
        NEXT_CLAIM_TOKEN.fetch_add(1, Ordering::Relaxed)
    );
    let mut parameter_index = 4;
    let namespace_sql = namespace.map(|_| {
        parameter_index += 1;
        format!("AND namespace=?{parameter_index}")
    });
    let kinds_sql = index_kinds.map(|kinds| {
        let placeholders = kinds
            .iter()
            .map(|_| {
                parameter_index += 1;
                format!("?{parameter_index}")
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("AND index_kind IN ({placeholders})")
    });
    let sql = format!(
        "WITH candidates AS (SELECT job_id FROM index_jobs WHERE ((status='PENDING' AND (next_run_at IS NULL OR next_run_at <= ?1)) OR (status='DOING' AND lease_until IS NOT NULL AND lease_until < ?1)) {} {} ORDER BY created_at ASC,job_id ASC LIMIT ?2) UPDATE index_jobs SET status='DOING',lease_until=?3,claim_token=?4,updated_at=?1 WHERE job_id IN (SELECT job_id FROM candidates) RETURNING job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token,accepted_result_json,accepted_result_sha256,accepted_at",
        namespace_sql.as_deref().unwrap_or_default(),
        kinds_sql.as_deref().unwrap_or_default(),
    );
    let mut values = vec![
        rusqlite::types::Value::Integer(now),
        rusqlite::types::Value::Integer(i64::try_from(limit).unwrap_or(i64::MAX)),
        rusqlite::types::Value::Integer(lease_until),
        rusqlite::types::Value::Text(token),
    ];
    if let Some(namespace) = namespace {
        values.push(rusqlite::types::Value::Text(namespace.to_owned()));
    }
    if let Some(index_kinds) = index_kinds {
        values.extend(
            index_kinds
                .iter()
                .cloned()
                .map(rusqlite::types::Value::Text),
        );
    }
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values), index_job_from_row)?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub(super) fn mark_index_job_done(
    conn: &Connection,
    job_id: &str,
    claim_token: Option<&str>,
) -> SqliteStoreResult<bool> {
    let now = unix_epoch_seconds();
    Ok(conn.execute("UPDATE index_jobs SET status='DONE',lease_until=NULL,claim_token=NULL,updated_at=?1 WHERE job_id=?2 AND status='DOING' AND (?3 IS NULL OR (claim_token=?3 AND lease_until IS NOT NULL AND lease_until>=?1))", params![now, job_id, claim_token])? != 0)
}
pub(super) fn index_job_result(
    conn: &Connection,
    job_id: &str,
) -> SqliteStoreResult<Option<AcceptedIndexJobResult>> {
    conn.query_row(
        "SELECT accepted_result_json, accepted_result_sha256, accepted_at FROM index_jobs WHERE job_id=?1",
        [job_id],
        |row| {
            Ok(AcceptedIndexJobResult {
                status: "existing".to_owned(),
                result_json: row.get(0)?,
                result_sha256: row.get(1)?,
                accepted_at: row.get::<_, Option<i64>>(2)?.map(serde_json::Value::from),
            })
        },
    )
    .optional()
    .map_err(Into::into)
}
pub(super) fn accept_index_job_result(
    conn: &Connection,
    job_id: &str,
    claim_token: &str,
    result_json: &str,
    result_sha256: &str,
) -> SqliteStoreResult<AcceptedIndexJobResult> {
    if let Some(existing) = index_job_result(conn, job_id)?
        && existing.result_json.is_some()
    {
        return Ok(existing);
    }
    let now = unix_epoch_seconds();
    let changed = conn.execute(
        "UPDATE index_jobs SET accepted_result_json=?1, accepted_result_sha256=?2, accepted_at=?3 WHERE job_id=?4 AND status='DOING' AND claim_token=?5 AND (lease_until IS NULL OR lease_until>=?3) AND accepted_result_json IS NULL",
        params![result_json, result_sha256, now, job_id, claim_token],
    )?;
    if changed == 0 {
        return Ok(AcceptedIndexJobResult {
            status: "rejected".to_owned(),
            result_json: None,
            result_sha256: None,
            accepted_at: None,
        });
    }
    Ok(AcceptedIndexJobResult {
        status: "accepted".to_owned(),
        result_json: Some(result_json.to_owned()),
        result_sha256: Some(result_sha256.to_owned()),
        accepted_at: Some(serde_json::Value::from(now)),
    })
}
pub(super) fn mark_index_job_failed(
    conn: &Connection,
    job_id: &str,
    error: &str,
    final_: bool,
    claim_token: Option<&str>,
) -> SqliteStoreResult<()> {
    let now = unix_epoch_seconds();
    conn.execute("UPDATE index_jobs SET status=CASE WHEN ?1 THEN 'FAILED' ELSE status END,lease_until=NULL,claim_token=NULL,last_error=?2,updated_at=?3 WHERE job_id=?4 AND status='DOING' AND (?5 IS NULL OR (claim_token=?5 AND lease_until IS NOT NULL AND lease_until>=?3))", params![final_, truncate_error(error), now, job_id, claim_token])?;
    Ok(())
}
pub(super) fn bump_retry_and_requeue(
    conn: &Connection,
    job_id: &str,
    error: &str,
    delay_seconds: i64,
    claim_token: Option<&str>,
) -> SqliteStoreResult<()> {
    let now = unix_epoch_seconds();
    conn.execute("UPDATE index_jobs SET retry_count=retry_count+1,last_error=?1,status=CASE WHEN retry_count+1>=max_retries THEN 'FAILED' ELSE 'PENDING' END,lease_until=NULL,claim_token=NULL,next_run_at=CASE WHEN retry_count+1>=max_retries THEN NULL ELSE ?2 END,updated_at=?3 WHERE job_id=?4 AND status='DOING' AND (?5 IS NULL OR (claim_token=?5 AND lease_until IS NOT NULL AND lease_until>=?3))", params![truncate_error(error), now+delay_seconds.max(0), now, job_id, claim_token])?;
    Ok(())
}
pub(super) fn renew_index_job_lease(
    conn: &Connection,
    job_id: &str,
    claim_token: &str,
    lease_seconds: i64,
) -> SqliteStoreResult<bool> {
    let now = unix_epoch_seconds();
    Ok(conn.execute("UPDATE index_jobs SET lease_until=?1,updated_at=?2 WHERE job_id=?3 AND status='DOING' AND claim_token=?4 AND (lease_until IS NULL OR lease_until>=?2)", params![now+lease_seconds,now,job_id,claim_token])? != 0)
}
pub(super) fn requeue_index_job_at_tail(
    conn: &Connection,
    job_id: &str,
    payload_json: String,
    delay_seconds: i64,
    claim_token: Option<&str>,
) -> SqliteStoreResult<()> {
    let now = unix_epoch_seconds();
    let tail: i64 = conn.query_row(
        "SELECT COALESCE(MAX(created_at),?1)+1 FROM index_jobs",
        [now],
        |row| row.get(0),
    )?;
    conn.execute("UPDATE index_jobs SET status='PENDING',lease_until=NULL,claim_token=NULL,next_run_at=?1,payload_json=?2,created_at=?3,updated_at=?4 WHERE job_id=?5 AND status='DOING' AND (?6 IS NULL OR (claim_token=?6 AND lease_until IS NOT NULL AND lease_until>=?4))", params![now+delay_seconds.max(0),payload_json,tail,now,job_id,claim_token])?;
    Ok(())
}
pub(super) fn list_index_jobs(
    conn: &Connection,
    namespace: Option<&str>,
    status: Option<&str>,
    entity_kind: Option<&str>,
    entity_id: Option<&str>,
    index_kind: Option<&str>,
    limit: usize,
) -> SqliteStoreResult<Vec<IndexJob>> {
    let query = "SELECT job_id,namespace,entity_kind,entity_id,index_kind,coalesce_key,op,status,lease_until,next_run_at,max_retries,retry_count,last_error,payload_json,created_at,updated_at,claim_token,accepted_result_json,accepted_result_sha256,accepted_at FROM index_jobs WHERE (?1 IS NULL OR namespace=?1) AND (?2 IS NULL OR status=?2) AND (?3 IS NULL OR entity_kind=?3) AND (?4 IS NULL OR entity_id=?4) AND (?5 IS NULL OR index_kind=?5) ORDER BY created_at ASC,job_id ASC LIMIT ?6";
    let mut statement = conn.prepare(query)?;
    let rows = statement.query_map(
        params![
            namespace,
            status,
            entity_kind,
            entity_id,
            index_kind,
            i64::try_from(limit).unwrap_or(i64::MAX)
        ],
        index_job_from_row,
    )?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}
pub(super) fn index_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<IndexJob> {
    Ok(IndexJob {
        job_id: row.get(0)?,
        namespace: row.get(1)?,
        entity_kind: row.get(2)?,
        entity_id: row.get(3)?,
        index_kind: row.get(4)?,
        coalesce_key: row.get(5)?,
        op: row.get(6)?,
        status: row.get(7)?,
        lease_until: row.get::<_, Option<i64>>(8)?.map(serde_json::Value::from),
        next_run_at: row.get::<_, Option<i64>>(9)?.map(serde_json::Value::from),
        max_retries: row.get(10)?,
        retry_count: row.get(11)?,
        last_error: row.get(12)?,
        payload_json: row.get(13)?,
        created_at: serde_json::Value::from(row.get::<_, i64>(14)?),
        updated_at: serde_json::Value::from(row.get::<_, i64>(15)?),
        claim_token: row.get(16)?,
        claim_attempts: 0,
        accepted_result_json: row.get(17)?,
        accepted_result_sha256: row.get(18)?,
        accepted_at: row.get::<_, Option<i64>>(19)?.map(serde_json::Value::from),
    })
}
pub(super) fn require_queue_namespace(namespace: &str) -> SqliteStoreResult<()> {
    if namespace.is_empty() {
        Err(SqliteStoreError::EmptyNamespace)
    } else {
        Ok(())
    }
}

pub(super) const LANE_SELECT: &str = "message_id,namespace,purpose,inbox_id,conversation_id,recipient_id,sender_id,msg_type,status,seq,conversation_seq,claimed_by,lease_until,retry_count,created_at,available_at,run_id,step_id,correlation_id,payload_json,error_json,prev_message_id,next_message_id,inbox_tail_message_id,conversation_tail_message_id";
