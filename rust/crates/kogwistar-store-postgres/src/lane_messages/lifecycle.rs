use super::projection::lane_from_row;
use super::*;

pub(crate) async fn ack_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status='completed',claimed_by=NULL,lease_until=NULL WHERE message_id=$1 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=$2 AND lease_until>=NOW()))",t.projected_lane_messages),&[&id,&owner]).await.map_err(backend)?;
    Ok(())
}
pub(crate) async fn requeue_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
    error: Option<String>,
    delay: i64,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    let delay = delay.max(0).to_string();
    c.execute(&format!("UPDATE {} SET status='pending',claimed_by=NULL,lease_until=NULL,retry_count=retry_count+1,available_at=EXTRACT(EPOCH FROM(NOW()+($1::TEXT||' seconds')::interval))::BIGINT,error_json=COALESCE($2,error_json) WHERE message_id=$3 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=$4 AND lease_until>=NOW()))",t.projected_lane_messages),&[&delay,&error,&id,&owner]).await.map_err(backend)?;
    Ok(())
}
pub(crate) async fn dead_letter_projected_lane_message<C>(
    c: &C,
    t: &Tables,
    id: &str,
    owner: &str,
    error: Option<String>,
) -> PostgresStoreResult<()>
where
    C: GenericClient + Sync,
{
    c.execute(&format!("UPDATE {} SET status='dead-letter',claimed_by=NULL,lease_until=NULL,error_json=COALESCE($1,error_json) WHERE message_id=$2 AND status NOT IN ('completed','failed','cancelled','dead-letter') AND (claimed_by IS NULL OR (status='claimed' AND claimed_by=$3 AND lease_until>=NOW()))",t.projected_lane_messages),&[&error,&id,&owner]).await.map_err(backend)?;
    Ok(())
}
pub(crate) async fn repair_orphaned_claimed_lane_messages<C>(
    client: &C,
    tables: &Tables,
    namespace: &str,
    inbox_id: Option<&str>,
    limit: usize,
) -> PostgresStoreResult<Vec<String>>
where
    C: GenericClient + Sync,
{
    if limit == 0 {
        return Ok(Vec::new());
    }
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let rows = client
        .query(
            &format!(
                "SELECT message_id FROM {} WHERE namespace=$1 \
                 AND ($2::TEXT IS NULL OR inbox_id=$2) AND status='claimed' \
                 AND (lease_until IS NULL OR lease_until<=NOW()) \
                 ORDER BY seq ASC, created_at ASC LIMIT $3 FOR UPDATE SKIP LOCKED",
                tables.projected_lane_messages
            ),
            &[&namespace, &inbox_id, &limit],
        )
        .await
        .map_err(backend)?;
    let ids = rows
        .iter()
        .map(|row| row.try_get::<_, String>(0).map_err(backend))
        .collect::<PostgresStoreResult<Vec<_>>>()?;
    for id in &ids {
        client
            .execute(
                &format!(
                    "UPDATE {} SET status='pending', claimed_by=NULL, lease_until=NULL \
                     WHERE message_id=$1 AND status='claimed' \
                     AND (lease_until IS NULL OR lease_until<=NOW())",
                    tables.projected_lane_messages
                ),
                &[id],
            )
            .await
            .map_err(backend)?;
    }
    Ok(ids)
}
pub(crate) async fn list_projected_lane_messages<C>(
    c: &C,
    t: &Tables,
    f: &LaneMessageFilter,
) -> PostgresStoreResult<Vec<ProjectedLaneMessage>>
where
    C: GenericClient + Sync,
{
    let limit = i64::try_from(f.limit).unwrap_or(i64::MAX);
    let order = if f.newest_first { "DESC" } else { "ASC" };
    let rows=c.query(&format!("SELECT {LANE_SELECT} FROM {} WHERE ($1::TEXT IS NULL OR namespace=$1) AND ($2::TEXT IS NULL OR purpose=$2) AND ($3::TEXT IS NULL OR inbox_id=$3) AND ($4::TEXT IS NULL OR conversation_id=$4) AND ($5::TEXT IS NULL OR status=$5) AND ($6::TEXT IS NULL OR msg_type=$6) AND ($7::TEXT IS NULL OR sender_id=$7) AND ($8::TEXT IS NULL OR recipient_id=$8) AND ($9::TEXT IS NULL OR correlation_id=$9) AND ($10::BIGINT IS NULL OR created_at>=$10) AND ($11::BIGINT IS NULL OR created_at<=$11) AND ($12::BIGINT IS NULL OR available_at>=$12) AND ($13::BIGINT IS NULL OR available_at<=$13) ORDER BY created_at {order},seq {order},message_id {order} LIMIT $14",t.projected_lane_messages),&[&f.namespace,&f.purpose,&f.inbox_id,&f.conversation_id,&f.status,&f.msg_type,&f.sender_id,&f.recipient_id,&f.correlation_id,&f.created_at_gte,&f.created_at_lte,&f.available_at_gte,&f.available_at_lte,&limit]).await.map_err(backend)?;
    rows.iter()
        .map(lane_from_row)
        .collect::<PostgresStoreResult<Vec<_>>>()
}
